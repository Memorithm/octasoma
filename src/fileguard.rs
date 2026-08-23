//! Validate-before-allocate guards for the versioned loaders (FRAC / SKCH / the
//! shard manifests) — proposal C1 of `docs/scirust-improvements.md`, following the
//! header-validation pattern of SciRust's safetensors loader.
//!
//! The rule: **never allocate from a file-declared count before checking the file
//! can actually supply that many bytes.** Without it, a hostile (or corrupt)
//! 24-byte header declaring `count = u64::MAX` makes the loader request a
//! multi-gigabyte allocation and abort the process. With it, an attacker can make
//! us allocate at most on the order of what they actually uploaded, and every
//! rejection is a clean [`std::io::ErrorKind::InvalidData`] error naming the field.
//!
//! No format change: these guards accept every well-formed file the previous
//! loaders accepted, and reject only files that could never parse to completion.

use std::io::{self, Read};
use std::path::Path;

/// An LZ4 block cannot expand by more than ~255× on decompression; a declared
/// decompressed length beyond `comp_len × 256` is corrupt or hostile — reject it
/// before handing the allocation to the decompressor.
pub(crate) const MAX_LZ4_RATIO: u64 = 256;

/// A [`Read`] adapter that counts consumed bytes, so a loader that knows the total
/// file size can bound every declared count against what the file can still supply.
pub(crate) struct CountingReader<R> {
    inner: R,
    consumed: u64,
}

impl<R: Read> CountingReader<R> {
    pub(crate) fn new(inner: R) -> Self {
        Self { inner, consumed: 0 }
    }

    /// Bytes read through this adapter so far.
    pub(crate) fn consumed(&self) -> u64 {
        self.consumed
    }
}

impl<R: Read> Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.consumed += n as u64;
        Ok(n)
    }
}

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// Rejects a file-declared `count` of records needing at least `min_record_bytes`
/// each when they cannot fit in the `remaining` bytes of the file. The `u128`
/// product cannot overflow for any `u64`-declared count.
pub(crate) fn guard_count(
    what: &str,
    count: usize,
    min_record_bytes: usize,
    remaining: u64,
) -> io::Result<()> {
    let need = count as u128 * min_record_bytes as u128;
    if need > remaining as u128 {
        return Err(invalid(format!(
            "{what}: {count} declared records need at least {need} bytes, \
             but only {remaining} remain in the file"
        )));
    }
    Ok(())
}

/// Rejects a declared decompressed length no LZ4 block of `comp_len` bytes could
/// ever produce (see [`MAX_LZ4_RATIO`]).
pub(crate) fn guard_decompressed(what: &str, decomp_len: u64, comp_len: u64) -> io::Result<()> {
    if decomp_len > comp_len.saturating_mul(MAX_LZ4_RATIO) {
        return Err(invalid(format!(
            "{what}: declared decompressed length {decomp_len} exceeds \
             {MAX_LZ4_RATIO}x the {comp_len} compressed bytes — corrupt or hostile"
        )));
    }
    Ok(())
}

/// Requires a manifest-selected component to match the deterministic writer name exactly.
pub(crate) fn guard_generated_component(
    what: &str,
    actual: &str,
    expected: &str,
) -> io::Result<()> {
    if actual != expected {
        return Err(invalid(format!(
            "{what}: expected generated component {expected:?}, got {actual:?}"
        )));
    }
    Ok(())
}

/// Refuses symbolic links at persistence trust boundaries.
pub(crate) fn guard_not_symlink(what: &str, path: &Path) -> io::Result<()> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() {
        return Err(invalid(format!(
            "{what}: symbolic links are not allowed in an OctaSoma store: {}",
            path.display()
        )));
    }
    Ok(())
}

// -- shared little-endian / length-prefixed IO ------------------------------
//
// One copy of the small readers/writers every persisted format in the crate
// uses (FRAC/SKCH stay in lib.rs; these serve the manifest-style formats).
// All length-prefixed reads validate-before-allocate.

/// Uniform `InvalidData` constructor for the crate's format errors.
pub(crate) fn invalid_data(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

pub(crate) fn read_u32_le(r: &mut &[u8]) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

pub(crate) fn read_u64_le(r: &mut &[u8]) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

/// A `u32`-length-prefixed byte string, validated against the unread input.
pub(crate) fn read_lp_bytes(what: &str, r: &mut &[u8]) -> io::Result<Vec<u8>> {
    let len = read_u32_le(r)? as usize;
    guard_count(what, len, 1, r.len() as u64)?;
    let mut b = vec![0u8; len];
    r.read_exact(&mut b)?;
    Ok(b)
}

/// A `u32`-length-prefixed UTF-8 string, validated against the unread input.
pub(crate) fn read_lp_string(what: &str, r: &mut &[u8]) -> io::Result<String> {
    String::from_utf8(read_lp_bytes(what, r)?).map_err(|e| invalid_data(&e.to_string()))
}

/// Appends a `u32`-length-prefixed byte string to a growing buffer.
pub(crate) fn write_lp_bytes(buf: &mut Vec<u8>, bytes: &[u8]) {
    buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(bytes);
}

/// An `f32` little-endian scalar.
pub(crate) fn read_f32_le(r: &mut &[u8]) -> io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

/// A `u64`-length-prefixed UTF-8 string (the OSHH/OSMS manifest convention,
/// kept byte-compatible); validated against the unread input.
pub(crate) fn read_u64_lp_string(what: &str, r: &mut &[u8]) -> io::Result<String> {
    let len = read_u64_le(r)? as usize;
    guard_count(what, len, 1, r.len() as u64)?;
    let mut b = vec![0u8; len];
    r.read_exact(&mut b)?;
    String::from_utf8(b).map_err(|e| invalid_data(&e.to_string()))
}

/// Appends a `u64`-length-prefixed byte string (OSHH/OSMS convention).
pub(crate) fn write_u64_lp(buf: &mut Vec<u8>, bytes: &[u8]) {
    buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(bytes);
}

/// Requires a manifest parser to consume the complete input.
pub(crate) fn guard_no_trailing_bytes(what: &str, remaining: usize) -> io::Result<()> {
    if remaining != 0 {
        return Err(invalid(format!(
            "{what}: {remaining} trailing bytes remain after the declared records"
        )));
    }
    Ok(())
}

/// Rejects a payload `(offset, len)` record that falls outside the decompressed
/// arena — catching it at load turns a would-be panic (or silently missing
/// payload) at query time into a clean load error.
pub(crate) fn guard_payload_bounds(
    what: &str,
    offset: usize,
    len: usize,
    arena_len: usize,
) -> io::Result<()> {
    if offset.checked_add(len).is_none_or(|end| end > arena_len) {
        return Err(invalid(format!(
            "{what}: payload record [{offset}, {offset}+{len}) exceeds the \
             {arena_len}-byte payload arena"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_count_rejects_impossible_declarations_without_allocating() {
        // u64::MAX records can never fit in a 24-byte file — and the check itself
        // must not overflow.
        assert!(guard_count("nodes", u64::MAX as usize, 52, 24).is_err());
        assert!(guard_count("nodes", 1, 52, 24).is_err());
        assert!(guard_count("nodes", 1, 52, 52).is_ok());
        assert!(guard_count("nodes", 0, 52, 0).is_ok());
    }

    #[test]
    fn guard_decompressed_bounds_the_lz4_expansion() {
        assert!(guard_decompressed("arena", 10, 10).is_ok());
        assert!(guard_decompressed("arena", 2560, 10).is_ok()); // exactly 256×
        assert!(guard_decompressed("arena", 2561, 10).is_err());
        assert!(guard_decompressed("arena", u64::MAX, u64::MAX).is_ok()); // no overflow
    }

    #[test]
    fn generated_component_guard_is_exact() {
        assert!(guard_generated_component("shard", "shard_00000000", "shard_00000000").is_ok());
        for hostile in [
            "../escape",
            &{
                let p = std::env::temp_dir().join("escape");
                p.to_string_lossy().into_owned()
            },
            "shard_00000000/child",
            "shard_00000001",
        ] {
            assert!(guard_generated_component("shard", hostile, "shard_00000000").is_err());
        }
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        assert!(guard_no_trailing_bytes("manifest", 0).is_ok());
        assert!(guard_no_trailing_bytes("manifest", 1).is_err());
    }

    #[test]
    fn guard_payload_bounds_catches_out_of_arena_records() {
        assert!(guard_payload_bounds("item", 0, 10, 10).is_ok());
        assert!(guard_payload_bounds("item", 5, 6, 10).is_err());
        assert!(guard_payload_bounds("item", usize::MAX, 1, 10).is_err()); // overflow
    }

    #[test]
    fn counting_reader_counts() {
        let data = [0u8; 10];
        let mut r = CountingReader::new(&data[..]);
        let mut buf = [0u8; 6];
        r.read_exact(&mut buf).unwrap();
        assert_eq!(r.consumed(), 6);
    }
}
