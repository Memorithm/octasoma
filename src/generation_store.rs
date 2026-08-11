//! Crash-safe generation persistence for [`HybridMemory`](crate::HybridMemory).
//!
//! A hybrid store is two coupled physical indexes. Publishing them as two files
//! in-place can expose a mixed generation after a crash. This module instead
//! writes an immutable generation directory, hashes both components, then
//! publishes a small `CURRENT` pointer. Readers always open one complete
//! generation and validate its manifest before deserialising either index.

use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use crate::{FractalMemory3D, HybridMemory, SketchIndex};

const MANIFEST_MAGIC: &str = "OCTASOMA-HYBRID-GENERATION-V1";
const CURRENT_MAGIC: &str = "OCTASOMA-HYBRID-CURRENT-V1";
const GENERATION_PREFIX: &str = "generation-";
const TREE_FILE: &str = "tree.frac";
const SKETCH_FILE: &str = "index.skch";
const MANIFEST_FILE: &str = "MANIFEST";
const CURRENT_FILE: &str = "CURRENT";
const MAX_MANIFEST_BYTES: u64 = 4096;
const MAX_CURRENT_BYTES: u64 = 512;

#[derive(Debug)]
struct Manifest {
    generation: u64,
    dim: usize,
    items: usize,
    default_shortlist: usize,
    octasoma_version: String,
    tree_sha256: String,
    sketch_sha256: String,
}

pub(crate) fn save(memory: &HybridMemory, dir: &str) -> io::Result<()> {
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
    let root = Path::new(dir);
    reject_symlink_if_present("hybrid store root", root)?;

    let current = root.join(CURRENT_FILE);
    if fs::symlink_metadata(&current).is_ok() {
        crate::fileguard::guard_not_symlink("hybrid CURRENT", &current)?;
        let raw = read_small_text(&current, MAX_CURRENT_BYTES, "hybrid CURRENT")?;
        let (name, expected_manifest_hash) = parse_current(&raw)?;
        return open_generation(root, &name, dim, Some(&expected_manifest_hash));
    }

    // Crash recovery: if the pointer has not yet been published (or a platform
    // had to remove it before replacement), the highest immutable generation is
    // still self-contained and can be validated. Hidden staging directories are
    // deliberately ignored by `highest_generation`.
    if let Some(generation) = highest_generation(root)? {
        return open_generation(root, &generation_name(generation), dim, None);
    }

    // Backward compatibility for v0.4 stores. New saves never write this layout.
    HybridMemory::open_legacy_dir(dir, dim)
}

fn open_generation(
    root: &Path,
    name: &str,
    dim: usize,
    expected_manifest_hash: Option<&str>,
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
        return Err(invalid("generation directory and MANIFEST generation disagree"));
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
    let body = format!(
        "{CURRENT_MAGIC}\ngeneration={generation}\nmanifest_sha256={manifest_sha256}\n"
    );
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
    format!(
        "{MANIFEST_MAGIC}\ngeneration={}\ndim={}\nitems={}\ndefault_shortlist={}\noctasoma_version={}\ntree_sha256={}\nsketch_sha256={}\n",
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
    if lines.len() != 8 || lines[0] != MANIFEST_MAGIC {
        return Err(invalid("invalid hybrid generation MANIFEST header/field count"));
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
        tree_sha256,
        sketch_sha256,
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

        let err = open(root.to_string_lossy().as_ref(), 4).unwrap_err();
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

        fs::write(
            root.join(CURRENT_FILE),
            format!(
                "{CURRENT_MAGIC}\ngeneration=../escape\nmanifest_sha256={}\n",
                "0".repeat(64)
            ),
        )
        .unwrap();
        assert!(open(root.to_string_lossy().as_ref(), 4).is_err());

        let manifest = root.join(generation_name(1)).join(MANIFEST_FILE);
        let mut raw = fs::read_to_string(&manifest).unwrap();
        raw = raw.replace("default_shortlist=13", "default_shortlist=14");
        fs::write(&manifest, raw).unwrap();
        let manifest_hash = hash_file(&manifest).unwrap();
        fs::write(
            root.join(CURRENT_FILE),
            format!(
                "{CURRENT_MAGIC}\ngeneration={}\nmanifest_sha256={manifest_hash}\n",
                generation_name(1)
            ),
        )
        .unwrap();
        let reopened = open(root.to_string_lossy().as_ref(), 4).unwrap();
        assert_eq!(reopened.default_shortlist, 14);
        let _ = fs::remove_dir_all(root);
    }
}
