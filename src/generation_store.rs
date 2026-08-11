//! Crash-safe generation persistence for [`HybridMemory`](crate::HybridMemory).
//!
//! A hybrid store is two coupled physical indexes. Publishing them as two files
//! in-place can expose a mixed generation after a crash. This module instead
//! writes an immutable generation directory, hashes both components, then
//! publishes a small `CURRENT` pointer. Readers always open one complete
//! generation and validate its manifest before deserialising either index.

use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use crate::{FractalMemory3D, HybridMemory, SketchIndex};

const MANIFEST_MAGIC_V1: &str = "OCTASOMA-HYBRID-GENERATION-V1";
const MANIFEST_MAGIC_V2: &str = "OCTASOMA-HYBRID-GENERATION-V2";
const CURRENT_MAGIC: &str = "OCTASOMA-HYBRID-CURRENT-V1";
const GENERATION_PREFIX: &str = "generation-";
const TREE_FILE: &str = "tree.frac";
const SKETCH_FILE: &str = "index.skch";
const MANIFEST_FILE: &str = "MANIFEST";
const CURRENT_FILE: &str = "CURRENT";
const MAX_MANIFEST_BYTES: u64 = 4096;
const MAX_CURRENT_BYTES: u64 = 512;
const MAX_FINGERPRINT_BYTES: usize = 256;

/// Reviewed SciRust revision defining the numerical/retrieval foundation
/// of the current OctaSoma v0.5 line.
pub const SCIRUST_REVISION: &str = "9b3d9492bb20e097231598a731df689ad4bd4bcc";

/// Exact interpretation contract bound to a persisted generation.
///
/// Fields are opaque deterministic identifiers supplied by the embedding /
/// indexing pipeline. Bound generations require an exact match at reopen,
/// preventing silent model/projection/quantization/calibration drift.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationFingerprint {
    pub embedding: String,
    pub projection: String,
    pub quantization: String,
    pub index: String,
    pub scirust_revision: String,
    pub calibration: Option<String>,
}

impl GenerationFingerprint {
    /// Creates a fingerprint pinned to the reviewed SciRust revision used by
    /// this OctaSoma release.
    pub fn canonical(
        embedding: impl Into<String>,
        projection: impl Into<String>,
        quantization: impl Into<String>,
        index: impl Into<String>,
    ) -> Self {
        Self {
            embedding: embedding.into(),
            projection: projection.into(),
            quantization: quantization.into(),
            index: index.into(),
            scirust_revision: SCIRUST_REVISION.to_string(),
            calibration: None,
        }
    }

    fn validate(&self) -> io::Result<()> {
        for (name, value) in [
            ("embedding", self.embedding.as_str()),
            ("projection", self.projection.as_str()),
            ("quantization", self.quantization.as_str()),
            ("index", self.index.as_str()),
            ("scirust_revision", self.scirust_revision.as_str()),
        ] {
            validate_fingerprint_field(name, value)?;
        }
        if let Some(value) = self.calibration.as_deref() {
            validate_fingerprint_field("calibration", value)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct Manifest {
    generation: u64,
    dim: usize,
    items: usize,
    default_shortlist: usize,
    octasoma_version: String,
    fingerprint: Option<GenerationFingerprint>,
    tree_sha256: String,
    sketch_sha256: String,
}

pub(crate) fn save(memory: &HybridMemory, dir: &str) -> io::Result<()> {
    save_impl(memory, dir, None)
}

pub(crate) fn save_bound(
    memory: &HybridMemory,
    dir: &str,
    fingerprint: &GenerationFingerprint,
) -> io::Result<()> {
    fingerprint.validate()?;
    save_impl(memory, dir, Some(fingerprint))
}

fn save_impl(
    memory: &HybridMemory,
    dir: &str,
    fingerprint: Option<&GenerationFingerprint>,
) -> io::Result<()> {
    let root = Path::new(dir);
    fs::create_dir_all(root)?;
    reject_symlink_if_present("hybrid store root", root)?;

    let generation = highest_generation(root)?
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(|| invalid("hybrid generation counter overflow"))?;
    let generation_name = generation_name(generation);
    let final_dir = root.join(&generation_name);
    if fs::symlink_metadata(&final_dir).is_ok() {
        return Err(invalid(&format!(
            "refusing to overwrite existing hybrid generation {}",
            final_dir.display()
        )));
    }

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| invalid("system clock is before UNIX_EPOCH"))?
        .as_nanos();
    let staging = root.join(format!(
        ".{generation_name}-{}-{nonce}.tmp",
        std::process::id()
    ));
    fs::create_dir(&staging)?;

    let result = (|| {
        let tree_path = staging.join(TREE_FILE);
        let sketch_path = staging.join(SKETCH_FILE);
        memory
            .tree
            .save_to_disk(tree_path.to_string_lossy().as_ref())?;
        memory
            .sketch
            .save_to_disk(sketch_path.to_string_lossy().as_ref())?;
        sync_file(&tree_path)?;
        sync_file(&sketch_path)?;

        let tree_sha256 = hash_file(&tree_path)?;
        let sketch_sha256 = hash_file(&sketch_path)?;
        let manifest = Manifest {
            generation,
            dim: memory.dim,
            items: memory.len(),
            default_shortlist: memory.default_shortlist,
            octasoma_version: env!("CARGO_PKG_VERSION").to_string(),
            fingerprint: fingerprint.cloned(),
            tree_sha256,
            sketch_sha256,
        };
        let manifest_bytes = encode_manifest(&manifest);
        let manifest_path = staging.join(MANIFEST_FILE);
        write_synced(&manifest_path, manifest_bytes.as_bytes())?;
        let manifest_sha256 = hash_bytes(manifest_bytes.as_bytes());
        sync_dir(&staging)?;

        // Publishing the immutable directory is the transaction boundary for the
        // coupled tree+sketch payload. A failed CURRENT update can leave an
        // unreferenced complete generation, but never a mixed generation.
        fs::rename(&staging, &final_dir)?;
        sync_dir(root)?;
        publish_current(root, &generation_name, &manifest_sha256)?;
        Ok(())
    })();

    if result.is_err() && staging.exists() {
        let _ = fs::remove_dir_all(&staging);
    }
    result
}

pub(crate) fn open(dir: &str, dim: usize) -> io::Result<HybridMemory> {
    open_impl(dir, dim, None)
}

pub(crate) fn open_bound(
    dir: &str,
    dim: usize,
    fingerprint: &GenerationFingerprint,
) -> io::Result<HybridMemory> {
    fingerprint.validate()?;
    open_impl(dir, dim, Some(fingerprint))
}

fn open_impl(
    dir: &str,
    dim: usize,
    expected_fingerprint: Option<&GenerationFingerprint>,
) -> io::Result<HybridMemory> {
    let root = Path::new(dir);
    reject_symlink_if_present("hybrid store root", root)?;

    let current = root.join(CURRENT_FILE);
    if fs::symlink_metadata(&current).is_ok() {
        crate::fileguard::guard_not_symlink("hybrid CURRENT", &current)?;
        let raw = read_small_text(&current, MAX_CURRENT_BYTES, "hybrid CURRENT")?;
        let (name, expected_manifest_hash) = parse_current(&raw)?;
        return open_generation(
            root,
            &name,
            dim,
            Some(&expected_manifest_hash),
            expected_fingerprint,
        );
    }

    // Crash recovery: if the pointer has not yet been published (or a platform
    // had to remove it before replacement), the highest immutable generation is
    // still self-contained and can be validated. Hidden staging directories are
    // deliberately ignored by `highest_generation`.
    if let Some(generation) = highest_generation(root)? {
        return open_generation(
            root,
            &generation_name(generation),
            dim,
            None,
            expected_fingerprint,
        );
    }

    // Backward compatibility for v0.4 stores. A bound open deliberately
    // rejects legacy/unbound state because no interpretation fingerprint can be
    // proven for it.
    if expected_fingerprint.is_some() {
        return Err(invalid(
            "bound generation open cannot accept an unbound legacy store",
        ));
    }
    HybridMemory::open_legacy_dir(dir, dim)
}

fn open_generation(
    root: &Path,
    name: &str,
    dim: usize,
    expected_manifest_hash: Option<&str>,
    expected_fingerprint: Option<&GenerationFingerprint>,
) -> io::Result<HybridMemory> {
    let generation = parse_generation_name(name)
        .ok_or_else(|| invalid(&format!("invalid hybrid generation name {name:?}")))?;
    let generation_dir = root.join(name);
    crate::fileguard::guard_not_symlink("hybrid generation", &generation_dir)?;
    if !generation_dir.is_dir() {
        return Err(invalid(&format!(
            "hybrid generation is not a directory: {}",
            generation_dir.display()
        )));
    }

    let manifest_path = generation_dir.join(MANIFEST_FILE);
    crate::fileguard::guard_not_symlink("hybrid generation manifest", &manifest_path)?;
    let manifest_raw = read_small_text(
        &manifest_path,
        MAX_MANIFEST_BYTES,
        "hybrid generation manifest",
    )?;
    if let Some(expected) = expected_manifest_hash {
        let actual = hash_bytes(manifest_raw.as_bytes());
        if actual != expected {
            return Err(invalid("CURRENT manifest hash does not match MANIFEST"));
        }
    }
    let manifest = parse_manifest(&manifest_raw)?;
    if manifest.generation != generation {
        return Err(invalid(
            "generation directory and MANIFEST generation disagree",
        ));
    }
    if manifest.dim != dim {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "hybrid store dimension mismatch: manifest {}, requested {dim}",
                manifest.dim
            ),
        ));
    }
    match (expected_fingerprint, manifest.fingerprint.as_ref()) {
        (Some(expected), Some(actual)) if expected == actual => {}
        (Some(_), Some(_)) => {
            return Err(invalid("hybrid generation fingerprint mismatch"));
        }
        (Some(_), None) => {
            return Err(invalid(
                "bound generation open cannot accept an unbound generation",
            ));
        }
        (None, Some(_)) => {
            return Err(invalid(
                "bound hybrid generation requires HybridMemory::open_dir_bound",
            ));
        }
        (None, None) => {}
    }

    let tree_path = generation_dir.join(TREE_FILE);
    let sketch_path = generation_dir.join(SKETCH_FILE);
    crate::fileguard::guard_not_symlink("hybrid tree", &tree_path)?;
    crate::fileguard::guard_not_symlink("hybrid sketch", &sketch_path)?;
    verify_hash(&tree_path, &manifest.tree_sha256, "hybrid tree")?;
    verify_hash(&sketch_path, &manifest.sketch_sha256, "hybrid sketch")?;

    let tree = FractalMemory3D::load_from_disk(tree_path.to_string_lossy().as_ref(), dim)?;
    let sketch = SketchIndex::load_from_disk(sketch_path.to_string_lossy().as_ref(), dim)?;
    if tree.item_count() != manifest.items || sketch.len() != manifest.items {
        return Err(invalid(&format!(
            "hybrid generation item-count mismatch: manifest {}, tree {}, sketch {}",
            manifest.items,
            tree.item_count(),
            sketch.len()
        )));
    }

    Ok(HybridMemory {
        tree,
        sketch,
        dim,
        default_shortlist: manifest.default_shortlist.max(1),
    })
}

fn publish_current(root: &Path, generation: &str, manifest_sha256: &str) -> io::Result<()> {
    let current = root.join(CURRENT_FILE);
    reject_symlink_if_present("hybrid CURRENT", &current)?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| invalid("system clock is before UNIX_EPOCH"))?
        .as_nanos();
    let tmp = root.join(format!(".CURRENT-{}-{nonce}.tmp", std::process::id()));
    let body =
        format!("{CURRENT_MAGIC}\ngeneration={generation}\nmanifest_sha256={manifest_sha256}\n");
    write_synced(&tmp, body.as_bytes())?;

    #[cfg(unix)]
    {
        // POSIX rename replaces the old file atomically.
        fs::rename(&tmp, &current)?;
    }

    #[cfg(not(unix))]
    {
        // Windows does not guarantee replacement of an existing destination.
        // The generation directory is already durable, so a crash in this tiny
        // pointer window is recovered by scanning immutable generations on open.
        let backup = root.join(".CURRENT.previous");
        if current.exists() {
            let _ = fs::remove_file(&backup);
            fs::rename(&current, &backup)?;
        }
        if let Err(err) = fs::rename(&tmp, &current) {
            if backup.exists() {
                let _ = fs::rename(&backup, &current);
            }
            return Err(err);
        }
        let _ = fs::remove_file(&backup);
    }

    sync_dir(root)
}

fn highest_generation(root: &Path) -> io::Result<Option<u64>> {
    let mut highest = None;
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if let Some(generation) = parse_generation_name(&name) {
            highest = Some(highest.map_or(generation, |old: u64| old.max(generation)));
        }
    }
    Ok(highest)
}

fn generation_name(generation: u64) -> String {
    format!("{GENERATION_PREFIX}{generation:020}")
}

fn parse_generation_name(name: &str) -> Option<u64> {
    let digits = name.strip_prefix(GENERATION_PREFIX)?;
    if digits.len() != 20 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

fn encode_manifest(manifest: &Manifest) -> String {
    let mut lines = vec![
        MANIFEST_MAGIC_V2.to_string(),
        format!("generation={}", manifest.generation),
        format!("dim={}", manifest.dim),
        format!("items={}", manifest.items),
        format!("default_shortlist={}", manifest.default_shortlist),
        format!("octasoma_version={}", manifest.octasoma_version),
    ];
    if let Some(fingerprint) = &manifest.fingerprint {
        lines.push("binding=bound".to_string());
        lines.push(format!(
            "embedding_hex={}",
            hex_string(&fingerprint.embedding)
        ));
        lines.push(format!(
            "projection_hex={}",
            hex_string(&fingerprint.projection)
        ));
        lines.push(format!(
            "quantization_hex={}",
            hex_string(&fingerprint.quantization)
        ));
        lines.push(format!("index_hex={}", hex_string(&fingerprint.index)));
        lines.push(format!(
            "scirust_revision_hex={}",
            hex_string(&fingerprint.scirust_revision)
        ));
        lines.push(format!(
            "calibration_hex={}",
            fingerprint
                .calibration
                .as_deref()
                .map(hex_string)
                .unwrap_or_else(|| "-".to_string())
        ));
    } else {
        lines.push("binding=unbound".to_string());
    }
    lines.push(format!("tree_sha256={}", manifest.tree_sha256));
    lines.push(format!("sketch_sha256={}", manifest.sketch_sha256));
    lines.join("\n") + "\n"
}

fn parse_manifest(raw: &str) -> io::Result<Manifest> {
    let lines: Vec<&str> = raw.lines().collect();
    match lines.first().copied() {
        Some(MANIFEST_MAGIC_V1) => parse_manifest_v1(&lines),
        Some(MANIFEST_MAGIC_V2) => parse_manifest_v2(&lines),
        _ => Err(invalid("invalid hybrid generation MANIFEST header")),
    }
}

fn parse_manifest_v1(lines: &[&str]) -> io::Result<Manifest> {
    if lines.len() != 8 {
        return Err(invalid("invalid v1 hybrid generation MANIFEST field count"));
    }
    let generation = parse_number(lines[1], "generation=")?;
    let dim = parse_number(lines[2], "dim=")?;
    let items = parse_number(lines[3], "items=")?;
    let default_shortlist = parse_number(lines[4], "default_shortlist=")?;
    if default_shortlist == 0 {
        return Err(invalid("MANIFEST default_shortlist must be non-zero"));
    }
    let octasoma_version = required_value(lines[5], "octasoma_version=")?.to_string();
    let tree_sha256 = parse_hash(lines[6], "tree_sha256=")?;
    let sketch_sha256 = parse_hash(lines[7], "sketch_sha256=")?;
    Ok(Manifest {
        generation,
        dim,
        items,
        default_shortlist,
        octasoma_version,
        fingerprint: None,
        tree_sha256,
        sketch_sha256,
    })
}

fn parse_manifest_v2(lines: &[&str]) -> io::Result<Manifest> {
    if lines.len() != 9 && lines.len() != 15 {
        return Err(invalid("invalid v2 hybrid generation MANIFEST field count"));
    }
    let generation = parse_number(lines[1], "generation=")?;
    let dim = parse_number(lines[2], "dim=")?;
    let items = parse_number(lines[3], "items=")?;
    let default_shortlist = parse_number(lines[4], "default_shortlist=")?;
    if default_shortlist == 0 {
        return Err(invalid("MANIFEST default_shortlist must be non-zero"));
    }
    let octasoma_version = required_value(lines[5], "octasoma_version=")?.to_string();
    let binding = required_value(lines[6], "binding=")?;
    let (fingerprint, tree_line, sketch_line) = match binding {
        "unbound" => {
            if lines.len() != 9 {
                return Err(invalid("unbound v2 MANIFEST has fingerprint fields"));
            }
            (None, 7, 8)
        }
        "bound" => {
            if lines.len() != 15 {
                return Err(invalid("bound v2 MANIFEST is missing fingerprint fields"));
            }
            let calibration_raw = required_value(lines[12], "calibration_hex=")?;
            let fingerprint = GenerationFingerprint {
                embedding: decode_hex_string(lines[7], "embedding_hex=")?,
                projection: decode_hex_string(lines[8], "projection_hex=")?,
                quantization: decode_hex_string(lines[9], "quantization_hex=")?,
                index: decode_hex_string(lines[10], "index_hex=")?,
                scirust_revision: decode_hex_string(lines[11], "scirust_revision_hex=")?,
                calibration: if calibration_raw == "-" {
                    None
                } else {
                    Some(decode_hex_value(calibration_raw, "calibration")?)
                },
            };
            fingerprint.validate()?;
            (Some(fingerprint), 13, 14)
        }
        _ => return Err(invalid("invalid v2 MANIFEST binding mode")),
    };
    let tree_sha256 = parse_hash(lines[tree_line], "tree_sha256=")?;
    let sketch_sha256 = parse_hash(lines[sketch_line], "sketch_sha256=")?;
    Ok(Manifest {
        generation,
        dim,
        items,
        default_shortlist,
        octasoma_version,
        fingerprint,
        tree_sha256,
        sketch_sha256,
    })
}

fn validate_fingerprint_field(name: &str, value: &str) -> io::Result<()> {
    if value.is_empty() {
        return Err(invalid(&format!("generation fingerprint {name} is empty")));
    }
    if value.len() > MAX_FINGERPRINT_BYTES {
        return Err(invalid(&format!(
            "generation fingerprint {name} exceeds {MAX_FINGERPRINT_BYTES} bytes"
        )));
    }
    Ok(())
}

fn hex_string(value: &str) -> String {
    to_hex(value.as_bytes())
}

fn decode_hex_string(line: &str, prefix: &str) -> io::Result<String> {
    let value = required_value(line, prefix)?;
    decode_hex_value(value, prefix)
}

fn decode_hex_value(value: &str, field: &str) -> io::Result<String> {
    if value.len() % 2 != 0 || value.len() > MAX_FINGERPRINT_BYTES * 2 {
        return Err(invalid(&format!("invalid hex length for {field}")));
    }
    let mut bytes = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().chunks_exact(2) {
        let hi = hex_nibble(pair[0]).ok_or_else(|| invalid(&format!("invalid hex in {field}")))?;
        let lo = hex_nibble(pair[1]).ok_or_else(|| invalid(&format!("invalid hex in {field}")))?;
        bytes.push((hi << 4) | lo);
    }
    String::from_utf8(bytes)
        .map_err(|_| invalid(&format!("decoded fingerprint {field} is not UTF-8")))
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn parse_current(raw: &str) -> io::Result<(String, String)> {
    let lines: Vec<&str> = raw.lines().collect();
    if lines.len() != 3 || lines[0] != CURRENT_MAGIC {
        return Err(invalid("invalid hybrid CURRENT header/field count"));
    }
    let generation = required_value(lines[1], "generation=")?;
    if parse_generation_name(generation).is_none() {
        return Err(invalid("CURRENT contains an invalid generation name"));
    }
    let manifest_hash = parse_hash(lines[2], "manifest_sha256=")?;
    Ok((generation.to_string(), manifest_hash))
}

fn parse_number<T>(line: &str, prefix: &str) -> io::Result<T>
where
    T: std::str::FromStr,
{
    required_value(line, prefix)?
        .parse()
        .map_err(|_| invalid(&format!("invalid numeric MANIFEST field {prefix}")))
}

fn required_value<'a>(line: &'a str, prefix: &str) -> io::Result<&'a str> {
    let value = line
        .strip_prefix(prefix)
        .ok_or_else(|| invalid(&format!("missing MANIFEST/CURRENT field {prefix}")))?;
    if value.is_empty() {
        return Err(invalid(&format!("empty MANIFEST/CURRENT field {prefix}")));
    }
    Ok(value)
}

fn parse_hash(line: &str, prefix: &str) -> io::Result<String> {
    let value = required_value(line, prefix)?;
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(invalid(&format!("invalid SHA-256 field {prefix}")));
    }
    Ok(value.to_string())
}

fn verify_hash(path: &Path, expected: &str, what: &str) -> io::Result<()> {
    let actual = hash_file(path)?;
    if actual != expected {
        return Err(invalid(&format!(
            "{what} SHA-256 mismatch: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn hash_file(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(to_hex(&hasher.finalize()))
}

fn hash_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    to_hex(&hasher.finalize())
}

fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn read_small_text(path: &Path, max: u64, what: &str) -> io::Result<String> {
    let metadata = fs::metadata(path)?;
    if metadata.len() > max {
        return Err(invalid(&format!(
            "{what} is {} bytes, above the {max}-byte limit",
            metadata.len()
        )));
    }
    fs::read_to_string(path).map_err(|err| {
        if err.kind() == io::ErrorKind::InvalidData {
            invalid(&format!("{what} is not valid UTF-8"))
        } else {
            err
        }
    })
}

fn write_synced(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

fn sync_file(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

#[cfg(unix)]
fn sync_dir(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

#[cfg(not(unix))]
fn sync_dir(_path: &Path) -> io::Result<()> {
    Ok(())
}

fn reject_symlink_if_present(what: &str, path: &Path) -> io::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(invalid(&format!(
            "{what}: symbolic links are not allowed: {}",
            path.display()
        ))),
        Ok(_) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err),
    }
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    fn temp_store(label: &str) -> PathBuf {
        let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "octasoma-generation-store-{label}-{}-{id}",
            std::process::id()
        ))
    }

    fn populated(shortlist: usize) -> HybridMemory {
        let mut memory = HybridMemory::new(4, 42, 64).with_shortlist(shortlist);
        assert!(memory.insert(&[1.0, 0.0, 0.0, 0.0], b"first"));
        memory
    }

    fn bound_fingerprint(label: &str) -> GenerationFingerprint {
        GenerationFingerprint::canonical(
            format!("embedder:{label}"),
            "projection:pca-v1",
            "quantization:f32",
            "index:simhash-64",
        )
    }

    #[test]
    fn bound_generation_requires_exact_bound_open() {
        let root = temp_store("bound");
        let memory = populated(17);
        let fingerprint = bound_fingerprint("a");
        save_bound(&memory, root.to_string_lossy().as_ref(), &fingerprint).unwrap();

        let reopened = open_bound(root.to_string_lossy().as_ref(), 4, &fingerprint).unwrap();
        assert_eq!(reopened.len(), 1);
        assert!(open(root.to_string_lossy().as_ref(), 4).is_err());

        let wrong = bound_fingerprint("b");
        let err = match open_bound(root.to_string_lossy().as_ref(), 4, &wrong) {
            Err(err) => err,
            Ok(_) => panic!("mismatched fingerprint unexpectedly opened"),
        };
        assert!(err.to_string().contains("fingerprint mismatch"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn unbound_generation_cannot_satisfy_bound_open() {
        let root = temp_store("unbound");
        let memory = populated(17);
        save(&memory, root.to_string_lossy().as_ref()).unwrap();
        let err = match open_bound(root.to_string_lossy().as_ref(), 4, &bound_fingerprint("a")) {
            Err(err) => err,
            Ok(_) => panic!("unbound generation unexpectedly satisfied bound open"),
        };
        assert!(err.to_string().contains("unbound generation"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn fingerprint_validation_rejects_empty_and_oversized_fields() {
        let root = temp_store("invalid-fingerprint");
        let memory = populated(17);
        let mut fingerprint = bound_fingerprint("a");
        fingerprint.calibration = Some(String::new());
        assert!(save_bound(&memory, root.to_string_lossy().as_ref(), &fingerprint,).is_err());
        fingerprint.calibration = None;
        fingerprint.embedding = "x".repeat(MAX_FINGERPRINT_BYTES + 1);
        assert!(save_bound(&memory, root.to_string_lossy().as_ref(), &fingerprint,).is_err());
        assert!(!root.exists());
    }

    #[test]
    fn v1_manifest_parser_remains_backward_compatible() {
        let hash = "0".repeat(64);
        let raw = format!(
            "{MANIFEST_MAGIC_V1}\ngeneration=1\ndim=4\nitems=1\ndefault_shortlist=17\noctasoma_version=0.5.0\ntree_sha256={hash}\nsketch_sha256={hash}\n"
        );
        let manifest = parse_manifest(&raw).unwrap();
        assert_eq!(manifest.generation, 1);
        assert_eq!(manifest.dim, 4);
        assert!(manifest.fingerprint.is_none());
    }

    #[test]
    fn successive_saves_publish_one_complete_latest_generation() {
        let root = temp_store("successive");
        let mut memory = populated(17);
        save(&memory, root.to_string_lossy().as_ref()).unwrap();
        assert!(memory.insert(&[0.0, 1.0, 0.0, 0.0], b"second"));
        save(&memory, root.to_string_lossy().as_ref()).unwrap();

        let reopened = open(root.to_string_lossy().as_ref(), 4).unwrap();
        assert_eq!(reopened.len(), 2);
        assert_eq!(reopened.default_shortlist, 17);
        let current = fs::read_to_string(root.join(CURRENT_FILE)).unwrap();
        assert!(current.contains("generation-00000000000000000002"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn missing_current_recovers_highest_immutable_generation_and_ignores_staging() {
        let root = temp_store("recovery");
        let memory = populated(23);
        save(&memory, root.to_string_lossy().as_ref()).unwrap();
        fs::remove_file(root.join(CURRENT_FILE)).unwrap();
        fs::create_dir(root.join(".generation-99999999999999999999-dead.tmp")).unwrap();

        let reopened = open(root.to_string_lossy().as_ref(), 4).unwrap();
        assert_eq!(reopened.len(), 1);
        assert_eq!(reopened.default_shortlist, 23);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn component_corruption_is_rejected_before_deserialisation() {
        let root = temp_store("corrupt");
        let memory = populated(31);
        save(&memory, root.to_string_lossy().as_ref()).unwrap();
        let generation = root.join(generation_name(1));
        let mut tree = fs::OpenOptions::new()
            .append(true)
            .open(generation.join(TREE_FILE))
            .unwrap();
        tree.write_all(b"corruption").unwrap();
        tree.sync_all().unwrap();

        let err = match open(root.to_string_lossy().as_ref(), 4) {
            Err(err) => err,
            Ok(_) => panic!("corrupt generation unexpectedly opened"),
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("SHA-256 mismatch"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn legacy_two_file_store_remains_readable() {
        let root = temp_store("legacy");
        fs::create_dir_all(&root).unwrap();
        let memory = populated(99);
        memory
            .tree
            .save_to_disk(root.join(TREE_FILE).to_string_lossy().as_ref())
            .unwrap();
        memory
            .sketch
            .save_to_disk(root.join(SKETCH_FILE).to_string_lossy().as_ref())
            .unwrap();

        let reopened = open(root.to_string_lossy().as_ref(), 4).unwrap();
        assert_eq!(reopened.len(), 1);
        // Legacy stores did not persist this setting, so they retain the legacy default.
        assert_eq!(reopened.default_shortlist, 256);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn current_rejects_path_traversal_and_manifest_tampering() {
        let root = temp_store("current");
        let memory = populated(13);
        save(&memory, root.to_string_lossy().as_ref()).unwrap();

        let original_current = fs::read_to_string(root.join(CURRENT_FILE)).unwrap();
        fs::write(
            root.join(CURRENT_FILE),
            format!(
                "{CURRENT_MAGIC}\ngeneration=../escape\nmanifest_sha256={}\n",
                "0".repeat(64)
            ),
        )
        .unwrap();
        assert!(open(root.to_string_lossy().as_ref(), 4).is_err());

        fs::write(root.join(CURRENT_FILE), original_current).unwrap();
        let manifest = root.join(generation_name(1)).join(MANIFEST_FILE);
        let mut raw = fs::read_to_string(&manifest).unwrap();
        raw = raw.replace("default_shortlist=13", "default_shortlist=14");
        fs::write(&manifest, raw).unwrap();
        let err = match open(root.to_string_lossy().as_ref(), 4) {
            Err(err) => err,
            Ok(_) => panic!("tampered manifest unexpectedly opened"),
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("manifest hash"));
        let _ = fs::remove_dir_all(root);
    }
}
