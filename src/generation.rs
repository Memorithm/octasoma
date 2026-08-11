//! Transactional, integrity-checked persistence generations for OctaSoma v0.5.
//!
//! A generation is immutable once published. The tree and precision index are
//! written under a staging directory, hashed, bound by a manifest, atomically
//! renamed into `generations/`, then made current by publishing a unique
//! append-only pointer under `current/`. A crash can therefore leave an orphan
//! generation, but can never make a mixed/partial generation current.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::HybridMemory;

const MANIFEST_MAGIC: [u8; 4] = *b"OSGM";
const POINTER_MAGIC: [u8; 4] = *b"OSGP";
const FORMAT_VERSION: u32 = 1;
const MAX_FINGERPRINT_BYTES: usize = 4 * 1024;
const TREE_FILE: &str = "tree.frac";
const SKETCH_FILE: &str = "index.skch";
const MANIFEST_FILE: &str = "manifest.osg";
const GENERATIONS_DIR: &str = "generations";
const CURRENT_DIR: &str = "current";

/// Reviewed SciRust revision currently defining OctaSoma's v0.5 numerical /
/// retrieval foundation. Generation manifests bind to this value by default.
pub const SCIRUST_REVISION: &str = "9b3d9492bb20e097231598a731df689ad4bd4bcc";

/// Interpretation fingerprint bound into every persisted generation.
///
/// These strings are opaque deterministic identifiers chosen by the embedding /
/// index pipeline. Loading requires an exact match, so a store cannot silently be
/// reopened with another model, projection, quantization or calibration contract.
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
    /// Creates a fingerprint bound to the canonical SciRust revision used by this
    /// OctaSoma release.
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
            if value.is_empty() {
                return Err(invalid(format!("generation fingerprint {name} is empty")));
            }
            if value.len() > MAX_FINGERPRINT_BYTES {
                return Err(invalid(format!(
                    "generation fingerprint {name} exceeds {MAX_FINGERPRINT_BYTES} bytes"
                )));
            }
        }
        if self
            .calibration
            .as_ref()
            .is_some_and(|value| value.len() > MAX_FINGERPRINT_BYTES)
        {
            return Err(invalid(format!(
                "generation calibration fingerprint exceeds {MAX_FINGERPRINT_BYTES} bytes"
            )));
        }
        Ok(())
    }
}

/// Integrity and interpretation metadata for one immutable store generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationManifest {
    pub generation: u64,
    pub dim: usize,
    pub fingerprint: GenerationFingerprint,
    pub tree_sha256: [u8; 32],
    pub sketch_sha256: [u8; 32],
}

/// An opened current generation and its verified manifest.
pub struct OpenGeneration {
    pub manifest: GenerationManifest,
    pub memory: HybridMemory,
}

/// Filesystem layout and commit protocol for immutable HybridMemory generations.
pub struct GenerationStore;

impl GenerationStore {
    /// Persists `memory` as a new immutable generation and publishes it current.
    ///
    /// The generation number must be strictly greater than every already-published
    /// pointer. Publishing happens last, so any earlier failure leaves at most an
    /// unreferenced generation directory.
    pub fn save(
        root: impl AsRef<Path>,
        generation: u64,
        memory: &HybridMemory,
        fingerprint: &GenerationFingerprint,
    ) -> io::Result<GenerationManifest> {
        fingerprint.validate()?;
        let root = root.as_ref();
        fs::create_dir_all(root)?;
        crate::fileguard::guard_not_symlink("generation store root", root)?;

        let generations = root.join(GENERATIONS_DIR);
        let current = root.join(CURRENT_DIR);
        fs::create_dir_all(&generations)?;
        fs::create_dir_all(&current)?;
        crate::fileguard::guard_not_symlink("generation directory", &generations)?;
        crate::fileguard::guard_not_symlink("current-pointer directory", &current)?;

        if let Some(latest) = latest_published_generation(&current)?
            && generation <= latest
        {
            return Err(invalid(format!(
                "generation must increase monotonically: latest={latest}, proposed={generation}"
            )));
        }

        let final_dir = generations.join(generation_dir_name(generation));
        if final_dir.exists() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("generation {} already exists", final_dir.display()),
            ));
        }

        let staging = generations.join(format!(
            ".staging-{:020}-{}",
            generation,
            std::process::id()
        ));
        fs::create_dir(&staging)?;

        let staged = (|| -> io::Result<GenerationManifest> {
            let staging_text = path_text(&staging)?;
            memory.save_dir(staging_text)?;

            let tree_path = staging.join(TREE_FILE);
            let sketch_path = staging.join(SKETCH_FILE);
            sync_regular_file(&tree_path)?;
            sync_regular_file(&sketch_path)?;

            let manifest = GenerationManifest {
                generation,
                dim: memory.dim(),
                fingerprint: fingerprint.clone(),
                tree_sha256: sha256_file(&tree_path)?,
                sketch_sha256: sha256_file(&sketch_path)?,
            };
            let manifest_bytes = encode_manifest(&manifest)?;
            let manifest_path = staging.join(MANIFEST_FILE);
            write_new_synced(&manifest_path, &manifest_bytes)?;
            sync_directory(&staging)?;
            Ok(manifest)
        })();

        let manifest = match staged {
            Ok(manifest) => manifest,
            Err(error) => {
                let _ = fs::remove_dir_all(&staging);
                return Err(error);
            }
        };

        // Same-parent rename: the generation becomes immutable/complete in one
        // filesystem operation. It is still not current until the pointer below.
        fs::rename(&staging, &final_dir)?;
        sync_directory(&generations)?;

        let manifest_bytes = fs::read(final_dir.join(MANIFEST_FILE))?;
        let pointer = encode_pointer(generation, sha256_bytes(&manifest_bytes));
        let pointer_name = pointer_file_name(generation);
        let final_pointer = current.join(&pointer_name);
        if final_pointer.exists() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("generation pointer already exists: {pointer_name}"),
            ));
        }
        let temp_pointer = current.join(format!(".{pointer_name}.{}.tmp", std::process::id()));
        write_new_synced(&temp_pointer, &pointer)?;
        fs::rename(&temp_pointer, &final_pointer)?;
        sync_directory(&current)?;

        Ok(manifest)
    }

    /// Opens the highest published generation and verifies every integrity and
    /// interpretation binding before constructing a [`HybridMemory`].
    ///
    /// A corrupt newest pointer/generation is an error. This function never falls
    /// back silently to an older generation.
    pub fn open_current(
        root: impl AsRef<Path>,
        expected_dim: usize,
        expected: &GenerationFingerprint,
    ) -> io::Result<OpenGeneration> {
        expected.validate()?;
        let root = root.as_ref();
        crate::fileguard::guard_not_symlink("generation store root", root)?;
        let generations = root.join(GENERATIONS_DIR);
        let current = root.join(CURRENT_DIR);
        crate::fileguard::guard_not_symlink("generation directory", &generations)?;
        crate::fileguard::guard_not_symlink("current-pointer directory", &current)?;

        let generation = latest_published_generation(&current)?.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "generation store has no published pointer",
            )
        })?;
        let pointer_path = current.join(pointer_file_name(generation));
        crate::fileguard::guard_not_symlink("generation pointer", &pointer_path)?;
        let pointer = decode_pointer(&fs::read(&pointer_path)?)?;
        if pointer.generation != generation {
            return Err(invalid(
                "generation pointer body does not match its filename",
            ));
        }

        let generation_dir = generations.join(generation_dir_name(generation));
        crate::fileguard::guard_not_symlink("published generation", &generation_dir)?;
        let manifest_path = generation_dir.join(MANIFEST_FILE);
        crate::fileguard::guard_not_symlink("generation manifest", &manifest_path)?;
        let manifest_bytes = fs::read(&manifest_path)?;
        if sha256_bytes(&manifest_bytes) != pointer.manifest_sha256 {
            return Err(invalid(
                "generation manifest SHA-256 does not match current pointer",
            ));
        }
        let manifest = decode_manifest(&manifest_bytes)?;
        if manifest.generation != generation {
            return Err(invalid(
                "generation manifest number does not match current pointer",
            ));
        }
        if manifest.dim != expected_dim {
            return Err(invalid(format!(
                "generation dimension mismatch: manifest={}, expected={expected_dim}",
                manifest.dim
            )));
        }
        if &manifest.fingerprint != expected {
            return Err(invalid("generation interpretation fingerprint mismatch"));
        }

        let tree_path = generation_dir.join(TREE_FILE);
        let sketch_path = generation_dir.join(SKETCH_FILE);
        crate::fileguard::guard_not_symlink("generation tree", &tree_path)?;
        crate::fileguard::guard_not_symlink("generation sketch", &sketch_path)?;
        if sha256_file(&tree_path)? != manifest.tree_sha256 {
            return Err(invalid("tree.frac SHA-256 mismatch"));
        }
        if sha256_file(&sketch_path)? != manifest.sketch_sha256 {
            return Err(invalid("index.skch SHA-256 mismatch"));
        }

        let memory = HybridMemory::open_dir(path_text(&generation_dir)?, expected_dim)?;
        Ok(OpenGeneration { manifest, memory })
    }
}

#[derive(Clone, Copy)]
struct GenerationPointer {
    generation: u64,
    manifest_sha256: [u8; 32],
}

fn generation_dir_name(generation: u64) -> String {
    format!("gen-{generation:020}")
}

fn pointer_file_name(generation: u64) -> String {
    format!("{generation:020}.ptr")
}

fn latest_published_generation(current: &Path) -> io::Result<Option<u64>> {
    let mut latest: Option<u64> = None;
    for entry in fs::read_dir(current)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| invalid("current-pointer filename is not valid UTF-8"))?;
        // A crash before pointer publication can leave only a hidden temporary.
        if name.starts_with('.') {
            continue;
        }
        if file_type.is_symlink() {
            return Err(invalid(format!(
                "symbolic link is not allowed in current pointer directory: {name}"
            )));
        }
        if !file_type.is_file() {
            return Err(invalid(format!(
                "unexpected non-file in current pointer directory: {name}"
            )));
        }
        let digits = name
            .strip_suffix(".ptr")
            .ok_or_else(|| invalid(format!("unexpected current pointer entry: {name}")))?;
        if digits.len() != 20 || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return Err(invalid(format!("invalid current pointer filename: {name}")));
        }
        let generation = digits
            .parse::<u64>()
            .map_err(|_| invalid(format!("invalid generation number in pointer: {name}")))?;
        latest = Some(latest.map_or(generation, |old| old.max(generation)));
    }
    Ok(latest)
}

fn encode_manifest(manifest: &GenerationManifest) -> io::Result<Vec<u8>> {
    manifest.fingerprint.validate()?;
    let dim = u32::try_from(manifest.dim)
        .map_err(|_| invalid("embedding dimension does not fit generation manifest u32"))?;
    let mut out = Vec::new();
    out.extend_from_slice(&MANIFEST_MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&manifest.generation.to_le_bytes());
    out.extend_from_slice(&dim.to_le_bytes());
    write_string(&mut out, &manifest.fingerprint.embedding)?;
    write_string(&mut out, &manifest.fingerprint.projection)?;
    write_string(&mut out, &manifest.fingerprint.quantization)?;
    write_string(&mut out, &manifest.fingerprint.index)?;
    write_string(&mut out, &manifest.fingerprint.scirust_revision)?;
    match &manifest.fingerprint.calibration {
        Some(value) => {
            out.push(1);
            write_string(&mut out, value)?;
        }
        None => out.push(0),
    }
    out.extend_from_slice(&manifest.tree_sha256);
    out.extend_from_slice(&manifest.sketch_sha256);
    Ok(out)
}

fn decode_manifest(bytes: &[u8]) -> io::Result<GenerationManifest> {
    let mut r = bytes;
    read_magic(&mut r, MANIFEST_MAGIC, "generation manifest")?;
    let version = read_u32(&mut r)?;
    if version != FORMAT_VERSION {
        return Err(invalid(format!(
            "unsupported generation manifest version {version}"
        )));
    }
    let generation = read_u64(&mut r)?;
    let dim = read_u32(&mut r)? as usize;
    let fingerprint = GenerationFingerprint {
        embedding: read_string(&mut r)?,
        projection: read_string(&mut r)?,
        quantization: read_string(&mut r)?,
        index: read_string(&mut r)?,
        scirust_revision: read_string(&mut r)?,
        calibration: match read_u8(&mut r)? {
            0 => None,
            1 => Some(read_string(&mut r)?),
            _ => return Err(invalid("invalid calibration-presence flag")),
        },
    };
    fingerprint.validate()?;
    let tree_sha256 = read_hash(&mut r)?;
    let sketch_sha256 = read_hash(&mut r)?;
    crate::fileguard::guard_no_trailing_bytes("generation manifest", r.len())?;
    Ok(GenerationManifest {
        generation,
        dim,
        fingerprint,
        tree_sha256,
        sketch_sha256,
    })
}

fn encode_pointer(generation: u64, manifest_sha256: [u8; 32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(48);
    out.extend_from_slice(&POINTER_MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&generation.to_le_bytes());
    out.extend_from_slice(&manifest_sha256);
    out
}

fn decode_pointer(bytes: &[u8]) -> io::Result<GenerationPointer> {
    let mut r = bytes;
    read_magic(&mut r, POINTER_MAGIC, "generation pointer")?;
    let version = read_u32(&mut r)?;
    if version != FORMAT_VERSION {
        return Err(invalid(format!(
            "unsupported generation pointer version {version}"
        )));
    }
    let generation = read_u64(&mut r)?;
    let manifest_sha256 = read_hash(&mut r)?;
    crate::fileguard::guard_no_trailing_bytes("generation pointer", r.len())?;
    Ok(GenerationPointer {
        generation,
        manifest_sha256,
    })
}

fn write_string(out: &mut Vec<u8>, value: &str) -> io::Result<()> {
    if value.len() > MAX_FINGERPRINT_BYTES {
        return Err(invalid("generation fingerprint field is too long"));
    }
    let len = u32::try_from(value.len())
        .map_err(|_| invalid("generation fingerprint field length does not fit u32"))?;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

fn read_string(r: &mut &[u8]) -> io::Result<String> {
    let len = read_u32(r)? as usize;
    if len > MAX_FINGERPRINT_BYTES {
        return Err(invalid(format!(
            "generation fingerprint field exceeds {MAX_FINGERPRINT_BYTES} bytes"
        )));
    }
    crate::fileguard::guard_count("generation fingerprint", len, 1, r.len() as u64)?;
    let mut bytes = vec![0u8; len];
    r.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|e| invalid(format!("fingerprint is not UTF-8: {e}")))
}

fn read_magic(r: &mut &[u8], expected: [u8; 4], what: &str) -> io::Result<()> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if magic != expected {
        return Err(invalid(format!("invalid {what} magic")));
    }
    Ok(())
}

fn read_u8<R: Read>(r: &mut R) -> io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_hash<R: Read>(r: &mut R) -> io::Result<[u8; 32]> {
    let mut hash = [0u8; 32];
    r.read_exact(&mut hash)?;
    Ok(hash)
}

fn sha256_bytes(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn sha256_file(path: &Path) -> io::Result<[u8; 32]> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(hasher.finalize().into())
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

fn sync_regular_file(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> io::Result<()> {
    // Windows does not expose portable directory fsync through std::fs::File.
    // File contents are sync_all'd and final names are still published by rename.
    Ok(())
}

fn path_text(path: &Path) -> io::Result<&str> {
    path.to_str().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "OctaSoma persistence path is not valid UTF-8: {}",
                path.display()
            ),
        )
    })
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn memory() -> HybridMemory {
        let mut memory = HybridMemory::new(4, 7, 64);
        assert!(memory.insert(&[1.0, 0.0, 0.0, 0.0], b"alpha"));
        assert!(memory.insert(&[0.0, 1.0, 0.0, 0.0], b"beta"));
        memory
    }

    fn fingerprint() -> GenerationFingerprint {
        GenerationFingerprint::canonical("test-embedder:v1", "jl:seed=7", "f32", "simhash:64")
    }

    fn temp_root(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "octasoma-generation-{label}-{}",
            std::process::id()
        ))
    }

    #[test]
    fn round_trip_and_newest_pointer_wins() {
        let root = temp_root("roundtrip");
        let _ = fs::remove_dir_all(&root);
        let memory = memory();
        GenerationStore::save(&root, 1, &memory, &fingerprint()).unwrap();
        GenerationStore::save(&root, 2, &memory, &fingerprint()).unwrap();
        let opened = GenerationStore::open_current(&root, 4, &fingerprint()).unwrap();
        assert_eq!(opened.manifest.generation, 2);
        assert_eq!(
            opened.memory.recall(&[1.0, 0.0, 0.0, 0.0], 1, 2)[0].0,
            b"alpha"
        );
        assert!(GenerationStore::save(&root, 2, &memory, &fingerprint()).is_err());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn orphan_generation_without_pointer_is_ignored() {
        let root = temp_root("orphan");
        let _ = fs::remove_dir_all(&root);
        let memory = memory();
        GenerationStore::save(&root, 4, &memory, &fingerprint()).unwrap();
        let orphan = root.join(GENERATIONS_DIR).join(generation_dir_name(5));
        fs::create_dir(&orphan).unwrap();
        let opened = GenerationStore::open_current(&root, 4, &fingerprint()).unwrap();
        assert_eq!(opened.manifest.generation, 4);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn component_corruption_is_detected_before_open() {
        let root = temp_root("corrupt-component");
        let _ = fs::remove_dir_all(&root);
        let memory = memory();
        GenerationStore::save(&root, 8, &memory, &fingerprint()).unwrap();
        let tree = root
            .join(GENERATIONS_DIR)
            .join(generation_dir_name(8))
            .join(TREE_FILE);
        let mut file = OpenOptions::new().append(true).open(tree).unwrap();
        file.write_all(b"corruption").unwrap();
        file.sync_all().unwrap();
        let error = GenerationStore::open_current(&root, 4, &fingerprint())
            .err()
            .unwrap();
        assert!(error.to_string().contains("tree.frac SHA-256 mismatch"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn published_manifest_corruption_never_falls_back() {
        let root = temp_root("corrupt-manifest");
        let _ = fs::remove_dir_all(&root);
        let memory = memory();
        GenerationStore::save(&root, 10, &memory, &fingerprint()).unwrap();
        GenerationStore::save(&root, 11, &memory, &fingerprint()).unwrap();
        let manifest = root
            .join(GENERATIONS_DIR)
            .join(generation_dir_name(11))
            .join(MANIFEST_FILE);
        let mut file = OpenOptions::new().append(true).open(manifest).unwrap();
        file.write_all(b"x").unwrap();
        file.sync_all().unwrap();
        assert!(GenerationStore::open_current(&root, 4, &fingerprint()).is_err());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn interpretation_fingerprint_must_match_exactly() {
        let root = temp_root("fingerprint");
        let _ = fs::remove_dir_all(&root);
        let memory = memory();
        GenerationStore::save(&root, 3, &memory, &fingerprint()).unwrap();
        let mut wrong = fingerprint();
        wrong.embedding = "other-model".into();
        assert!(GenerationStore::open_current(&root, 4, &wrong).is_err());
        let _ = fs::remove_dir_all(root);
    }
}
