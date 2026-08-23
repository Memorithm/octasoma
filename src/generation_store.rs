//! Crash-safe generation persistence for [`HybridMemory`](crate::HybridMemory).
//!
//! A hybrid store is two coupled physical indexes. Publishing them as two files
//! in-place can expose a mixed generation after a crash. This module instead
//! writes an immutable generation directory, hashes both components, then
//! publishes a small `CURRENT` pointer. Readers always open one complete
//! generation and validate its manifest before deserialising either index.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use sha2::{Digest, Sha256};

use crate::{FractalMemory3D, GenerationFingerprint, HybridMemory, SketchIndex};

const MANIFEST_MAGIC_V1: &str = "OCTASOMA-HYBRID-GENERATION-V1";
const MANIFEST_MAGIC_V2: &str = "OCTASOMA-HYBRID-GENERATION-V2";
const CURRENT_MAGIC: &str = "OCTASOMA-HYBRID-CURRENT-V1";
const GENERATION_PREFIX: &str = "generation-";
const TREE_FILE: &str = "tree.frac";
const SKETCH_FILE: &str = "index.skch";
const MANIFEST_FILE: &str = "MANIFEST";
const CURRENT_FILE: &str = "CURRENT";
#[cfg(not(unix))]
const PREVIOUS_CURRENT_FILE: &str = ".CURRENT.previous";
const MAX_TEMP_ATTEMPTS: usize = 1024;
static NEXT_TEMP_NONCE: AtomicU64 = AtomicU64::new(0);
const MAX_MANIFEST_BYTES: u64 = 16 * 1024;
const MAX_CURRENT_BYTES: u64 = 512;

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

pub(crate) fn save_with_fingerprint(
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

    let staging = create_unique_staging_dir(root, &generation_name)?;

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

pub(crate) fn open_with_fingerprint(
    dir: &str,
    dim: usize,
    expected: &GenerationFingerprint,
) -> io::Result<HybridMemory> {
    expected.validate()?;
    open_impl(dir, dim, Some(expected))
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
        return open_pointer(root, &current, "hybrid CURRENT", dim, expected_fingerprint);
    }

    #[cfg(not(unix))]
    {
        // Windows publication temporarily moves the previously published pointer
        // aside because std::fs::rename cannot portably replace an existing file.
        // If a crash happens in that narrow window, only that *previous pointer*
        // is authoritative. An immutable generation without a pointer is never
        // promoted merely because it has the largest number.
        let previous = root.join(PREVIOUS_CURRENT_FILE);
        if fs::symlink_metadata(&previous).is_ok() {
            return open_pointer(
                root,
                &previous,
                "hybrid previous CURRENT",
                dim,
                expected_fingerprint,
            );
        }
    }

    if highest_generation(root)?.is_some() {
        return Err(invalid(
            "hybrid generation exists but no published CURRENT pointer is present",
        ));
    }

    // Compatibility mode can read legacy v0.4. Strict mode never
    // downgrades to bytes that have no interpretation identity.
    if expected_fingerprint.is_some() {
        return Err(invalid(
            "strict fingerprint open refuses a legacy store with no interpretation binding",
        ));
    }
    HybridMemory::open_legacy_dir(dir, dim)
}

fn open_pointer(
    root: &Path,
    pointer: &Path,
    what: &str,
    dim: usize,
    expected_fingerprint: Option<&GenerationFingerprint>,
) -> io::Result<HybridMemory> {
    crate::fileguard::guard_not_symlink(what, pointer)?;
    let raw = read_small_text(pointer, MAX_CURRENT_BYTES, what)?;
    let (name, expected_manifest_hash) = parse_current(&raw)?;
    open_generation(
        root,
        &name,
        dim,
        Some(&expected_manifest_hash),
        expected_fingerprint,
    )
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

    if let Some(expected) = expected_fingerprint {
        let actual = manifest
            .fingerprint
            .as_ref()
            .ok_or_else(|| invalid("strict fingerprint open refuses an unbound generation"))?;
        if actual != expected {
            return Err(invalid(
                "hybrid generation interpretation fingerprint mismatch",
            ));
        }
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
    let body =
        format!("{CURRENT_MAGIC}\ngeneration={generation}\nmanifest_sha256={manifest_sha256}\n");
    let tmp = write_unique_temp_file(root, "CURRENT", body.as_bytes())?;

    #[cfg(unix)]
    {
        // POSIX rename replaces the old file atomically. The generation becomes
        // authoritative only at this pointer publication step.
        if let Err(err) = fs::rename(&tmp, &current) {
            let _ = fs::remove_file(&tmp);
            return Err(err);
        }
    }

    #[cfg(not(unix))]
    {
        // Windows does not guarantee replacement of an existing destination.
        // Preserve the *previously published pointer* during the replacement
        // window. Recovery may use this pointer, never an unreferenced generation.
        let backup = root.join(PREVIOUS_CURRENT_FILE);
        reject_symlink_if_present("hybrid previous CURRENT", &backup)?;
        if current.exists() {
            if backup.exists() {
                fs::remove_file(&backup)?;
            }
            fs::rename(&current, &backup)?;
        }
        if let Err(err) = fs::rename(&tmp, &current) {
            if backup.exists() {
                let _ = fs::rename(&backup, &current);
            }
            let _ = fs::remove_file(&tmp);
            return Err(err);
        }
        if backup.exists() {
            fs::remove_file(&backup)?;
        }
    }

    sync_dir(root)
}

fn create_unique_staging_dir(root: &Path, generation_name: &str) -> io::Result<PathBuf> {
    for _ in 0..MAX_TEMP_ATTEMPTS {
        let nonce = NEXT_TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let path = root.join(format!(
            ".{generation_name}-{}-{nonce}.tmp",
            std::process::id()
        ));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(err) if err.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(err) => return Err(err),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not allocate a unique hybrid staging directory",
    ))
}

fn write_unique_temp_file(root: &Path, stem: &str, bytes: &[u8]) -> io::Result<PathBuf> {
    for _ in 0..MAX_TEMP_ATTEMPTS {
        let nonce = NEXT_TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let path = root.join(format!(".{stem}-{}-{nonce}.tmp", std::process::id()));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut file) => {
                if let Err(err) = file.write_all(bytes).and_then(|_| file.sync_all()) {
                    let _ = fs::remove_file(&path);
                    return Err(err);
                }
                return Ok(path);
            }
            Err(err) if err.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(err) => return Err(err),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not allocate a unique hybrid pointer temporary file",
    ))
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

/// Deletes all but the newest `keep` published generations under `root`. The
/// generation `CURRENT` points at is always preserved, even when it falls
/// outside the newest window; an absent pointer refuses to prune anything
/// (nothing is authoritative). Staging directories of interrupted saves are
/// left alone — a concurrent writer may still own one.
///
/// Returns how many generation directories were removed. Call after a save,
/// when no reader is mid-open: readers that already opened a generation keep
/// operating on their in-memory copy.
pub(crate) fn prune_generations(root: &Path, keep: usize) -> io::Result<usize> {
    if keep == 0 {
        return Err(invalid("generation pruning must keep at least one"));
    }

    let current_path = root.join(CURRENT_FILE);
    match fs::symlink_metadata(&current_path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            return Err(invalid("hybrid CURRENT: symbolic links are not allowed"));
        }
        Ok(_) => {}
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            return Err(invalid(
                "refusing to prune: no published CURRENT pointer is present",
            ));
        }
        Err(err) => return Err(err),
    }
    let raw = read_small_text(&current_path, MAX_CURRENT_BYTES, "hybrid CURRENT")?;
    let (current_name, _) = parse_current(&raw)?;

    let mut generations: Vec<u64> = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if let Some(generation) = parse_generation_name(&name) {
            generations.push(generation);
        }
    }
    // Newest first: the survivors are the newest `keep` generations *including*
    // the one CURRENT names, which is preserved unconditionally.
    generations.sort_unstable_by(|a, b| b.cmp(a));
    let current_number = parse_generation_name(&current_name)
        .ok_or_else(|| invalid("CURRENT contains an invalid generation name"))?;

    let mut removed = 0;
    for generation in generations.into_iter().skip(keep) {
        if generation == current_number {
            continue;
        }
        let name = generation_name(generation);
        let path = root.join(&name);
        crate::fileguard::guard_not_symlink("pruned hybrid generation", &path)?;
        fs::remove_dir_all(&path)?;
        removed += 1;
    }
    if removed > 0 {
        sync_dir(root)?;
    }
    Ok(removed)
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
    if let Some(f) = &manifest.fingerprint {
        let (calibration_present, calibration) = match &f.calibration {
            Some(value) => (1, value.as_str()),
            None => (0, ""),
        };
        return format!(
            "{MANIFEST_MAGIC_V2}\ngeneration={}\ndim={}\nitems={}\ndefault_shortlist={}\noctasoma_version={}\nembedding={}\nprojection={}\nquantization={}\nindex={}\nscirust_revision={}\ncalibration_present={}\ncalibration={}\ntree_sha256={}\nsketch_sha256={}\n",
            manifest.generation,
            manifest.dim,
            manifest.items,
            manifest.default_shortlist,
            manifest.octasoma_version,
            f.embedding,
            f.projection,
            f.quantization,
            f.index,
            f.scirust_revision,
            calibration_present,
            calibration,
            manifest.tree_sha256,
            manifest.sketch_sha256,
        );
    }
    format!(
        "{MANIFEST_MAGIC_V1}\ngeneration={}\ndim={}\nitems={}\ndefault_shortlist={}\noctasoma_version={}\ntree_sha256={}\nsketch_sha256={}\n",
        manifest.generation,
        manifest.dim,
        manifest.items,
        manifest.default_shortlist,
        manifest.octasoma_version,
        manifest.tree_sha256,
        manifest.sketch_sha256,
    )
}

fn parse_manifest(raw: &str) -> io::Result<Manifest> {
    let lines: Vec<&str> = raw.lines().collect();
    match lines.first().copied() {
        Some(MANIFEST_MAGIC_V1) => parse_manifest_v1(&lines),
        Some(MANIFEST_MAGIC_V2) => parse_manifest_v2(&lines),
        _ => Err(invalid("invalid hybrid generation MANIFEST magic")),
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
    Ok(Manifest {
        generation,
        dim,
        items,
        default_shortlist,
        octasoma_version: required_value(lines[5], "octasoma_version=")?.to_string(),
        fingerprint: None,
        tree_sha256: parse_hash(lines[6], "tree_sha256=")?,
        sketch_sha256: parse_hash(lines[7], "sketch_sha256=")?,
    })
}

fn parse_manifest_v2(lines: &[&str]) -> io::Result<Manifest> {
    if lines.len() != 15 {
        return Err(invalid("invalid v2 hybrid generation MANIFEST field count"));
    }
    let generation = parse_number(lines[1], "generation=")?;
    let dim = parse_number(lines[2], "dim=")?;
    let items = parse_number(lines[3], "items=")?;
    let default_shortlist = parse_number(lines[4], "default_shortlist=")?;
    if default_shortlist == 0 {
        return Err(invalid("MANIFEST default_shortlist must be non-zero"));
    }
    let calibration_present: u8 = parse_number(lines[11], "calibration_present=")?;
    let calibration_raw = lines[12]
        .strip_prefix("calibration=")
        .ok_or_else(|| invalid("missing MANIFEST field calibration="))?;
    let calibration = match calibration_present {
        0 if calibration_raw.is_empty() => None,
        1 if !calibration_raw.is_empty() => Some(calibration_raw.to_string()),
        0 => {
            return Err(invalid(
                "calibration value present while calibration_present=0",
            ));
        }
        1 => return Err(invalid("calibration_present=1 but calibration is empty")),
        _ => return Err(invalid("calibration_present must be 0 or 1")),
    };
    let fingerprint = GenerationFingerprint {
        embedding: required_value(lines[6], "embedding=")?.to_string(),
        projection: required_value(lines[7], "projection=")?.to_string(),
        quantization: required_value(lines[8], "quantization=")?.to_string(),
        index: required_value(lines[9], "index=")?.to_string(),
        scirust_revision: required_value(lines[10], "scirust_revision=")?.to_string(),
        calibration,
    };
    fingerprint
        .validate()
        .map_err(|e| invalid(&format!("invalid persisted generation fingerprint: {e}")))?;
    Ok(Manifest {
        generation,
        dim,
        items,
        default_shortlist,
        octasoma_version: required_value(lines[5], "octasoma_version=")?.to_string(),
        fingerprint: Some(fingerprint),
        tree_sha256: parse_hash(lines[13], "tree_sha256=")?,
        sketch_sha256: parse_hash(lines[14], "sketch_sha256=")?,
    })
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
    // Windows note: FlushFileBuffers requires a handle opened with write
    // access — a read-only open fails with ACCESS_DENIED there. Write-mode
    // fsync is equally valid on Unix, so one form serves both.
    OpenOptions::new().write(true).open(path)?.sync_all()
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
    fn missing_current_never_promotes_an_unpublished_generation() {
        let root = temp_store("unpublished");
        let memory = populated(23);
        save(&memory, root.to_string_lossy().as_ref()).unwrap();
        fs::remove_file(root.join(CURRENT_FILE)).unwrap();
        fs::create_dir(root.join(".generation-99999999999999999999-dead.tmp")).unwrap();

        let err = open(root.to_string_lossy().as_ref(), 4)
            .err()
            .expect("generation without CURRENT was silently promoted");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("no published CURRENT"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn orphan_generation_does_not_override_the_published_pointer() {
        let root = temp_store("orphan");
        let memory = populated(29);
        save(&memory, root.to_string_lossy().as_ref()).unwrap();

        let orphan = root.join(generation_name(2));
        fs::create_dir(&orphan).unwrap();
        for name in [TREE_FILE, SKETCH_FILE, MANIFEST_FILE] {
            fs::write(orphan.join(name), b"not published").unwrap();
        }

        let reopened = open(root.to_string_lossy().as_ref(), 4).unwrap();
        assert_eq!(reopened.len(), 1);
        assert_eq!(reopened.default_shortlist, 29);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn temp_allocation_does_not_depend_on_wall_clock() {
        let root = temp_store("temp-nonce");
        fs::create_dir_all(&root).unwrap();
        let first = create_unique_staging_dir(&root, "generation-00000000000000000001").unwrap();
        let second = create_unique_staging_dir(&root, "generation-00000000000000000001").unwrap();
        assert_ne!(first, second);
        assert!(first.is_dir());
        assert!(second.is_dir());
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

    #[test]
    fn strict_fingerprint_roundtrip_and_mismatch_rejection() {
        let root = temp_store("fingerprint");
        let memory = populated(41);
        let fingerprint =
            GenerationFingerprint::canonical("embed:test:v1", "jl:seed=42", "f32", "simhash:64")
                .with_calibration("rcps:sha256:abc123");
        save_with_fingerprint(&memory, root.to_string_lossy().as_ref(), &fingerprint).unwrap();
        let reopened =
            open_with_fingerprint(root.to_string_lossy().as_ref(), 4, &fingerprint).unwrap();
        assert_eq!(reopened.len(), 1);
        assert_eq!(reopened.default_shortlist, 41);

        let mut wrong = fingerprint.clone();
        wrong.embedding = "embed:test:v2".into();
        let err = open_with_fingerprint(root.to_string_lossy().as_ref(), 4, &wrong)
            .err()
            .expect("mismatched interpretation unexpectedly opened");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(
            err.to_string()
                .contains("interpretation fingerprint mismatch")
        );
        assert_eq!(open(root.to_string_lossy().as_ref(), 4).unwrap().len(), 1);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn strict_fingerprint_open_refuses_unbound_generation() {
        let root = temp_store("unbound");
        let memory = populated(43);
        save(&memory, root.to_string_lossy().as_ref()).unwrap();
        let fingerprint =
            GenerationFingerprint::canonical("embed:test:v1", "jl:seed=42", "f32", "simhash:64");
        let err = open_with_fingerprint(root.to_string_lossy().as_ref(), 4, &fingerprint)
            .err()
            .expect("unbound generation unexpectedly opened strictly");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("unbound generation"));
        let _ = fs::remove_dir_all(root);
    }

    // -- generation pruning ----------------------------------------------------

    #[test]
    fn prune_keeps_the_newest_window_and_reopens_cleanly() {
        let root = temp_store("prune-window");
        let mut memory = populated(7);
        for round in 0..3u32 {
            memory.insert(
                &[0.0, 1.0, round as f32, 0.0],
                format!("r{round}").as_bytes(),
            );
            save(&memory, root.to_string_lossy().as_ref()).unwrap();
        }
        assert_eq!(
            fs::read_dir(&root)
                .unwrap()
                .filter(|e| {
                    e.as_ref()
                        .unwrap()
                        .file_name()
                        .to_str()
                        .unwrap()
                        .starts_with(GENERATION_PREFIX)
                })
                .count(),
            3
        );

        assert_eq!(prune_generations(&root, 2).unwrap(), 1);
        let remaining: Vec<String> = fs::read_dir(&root)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_str().unwrap().to_string())
            .filter(|n| n.starts_with(GENERATION_PREFIX))
            .collect();
        assert_eq!(remaining.len(), 2);
        assert!(remaining.contains(&generation_name(3)));
        assert!(remaining.contains(&generation_name(2)));

        let reopened = open(root.to_string_lossy().as_ref(), 4).unwrap();
        assert_eq!(reopened.len(), 4); // populated's first item + r0..r2
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn prune_refuses_without_a_published_pointer_or_zero_keep() {
        let root = temp_store("prune-refuse");
        let memory = populated(11);
        save(&memory, root.to_string_lossy().as_ref()).unwrap();

        assert_eq!(
            prune_generations(&root, 0).err().unwrap().kind(),
            io::ErrorKind::InvalidData
        );

        fs::remove_file(root.join(CURRENT_FILE)).unwrap();
        let err = prune_generations(&root, 2).expect_err("pruned without CURRENT");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        // Nothing was touched.
        assert!(root.join(generation_name(1)).is_dir());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn prune_always_preserves_the_generation_current_points_at() {
        let root = temp_store("prune-current");
        let mut memory = populated(13);
        save(&memory, root.to_string_lossy().as_ref()).unwrap();
        memory.insert(&[0.0, 0.0, 0.0, 9.0], b"second");
        save(&memory, root.to_string_lossy().as_ref()).unwrap();

        // An unpublished newer generation (crashed publish leftovers aside,
        // this is the state right after rename but before CURRENT landed).
        let orphan = root.join(generation_name(3));
        fs::create_dir(&orphan).unwrap();
        fs::write(orphan.join(MANIFEST_FILE), "junk").unwrap();

        // keep=1: the newest window holds only generation-3, but CURRENT names
        // generation-2 — both must survive; generation-1 goes.
        assert_eq!(prune_generations(&root, 1).unwrap(), 1);
        assert!(!root.join(generation_name(1)).exists());
        assert!(root.join(generation_name(2)).is_dir());
        assert!(root.join(generation_name(3)).is_dir());
        let reopened = open(root.to_string_lossy().as_ref(), 4).unwrap();
        assert_eq!(reopened.len(), 2);
        let _ = fs::remove_dir_all(root);
    }
}
