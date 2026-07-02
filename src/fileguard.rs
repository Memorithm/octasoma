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
