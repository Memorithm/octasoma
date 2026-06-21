# OctaSoma — `FRAC` v3 file format

`save_to_disk` and `load_from_disk` use a single, sequential, **little-endian**
binary stream. The payload arena is **LZ4-compressed**; everything else is
stored verbatim. The writer/reader live in [`src/lib.rs`](../src/lib.rs) and this
document matches them byte-for-byte.

## Conventions

- All integers little-endian; `f32` is IEEE-754 binary32 little-endian.
- `u32` node/item references use the sentinel `NONE = 0xFFFF_FFFF` for "absent".
- Sections appear in exactly the order below, with no padding or alignment.

## Layout

```
┌──────────────────────────────────────────────────────────────┐
│ HEADER                                                        │
│   magic            : 4 bytes  = "FRAC"                        │
│   version          : u32      = 3                             │
│   high_dim         : u32                                      │
├──────────────────────────────────────────────────────────────┤
│ PARAMETERS                                                    │
│   world_half_size  : f32                                      │
│   bucket_capacity  : u64                                      │
│   min_half_size    : f32                                      │
├──────────────────────────────────────────────────────────────┤
│ NODES                                                         │
│   node_count       : u64                                      │
│   node[node_count] : 52 bytes each →                         │
│       center       : 3 × f32   (12 B)                         │
│       half_size    : f32       ( 4 B)                         │
│       children     : 8 × u32   (32 B)   NONE = absent         │
│       bucket_id    : u32       ( 4 B)   NONE = internal node  │
├──────────────────────────────────────────────────────────────┤
│ LEAF BUCKETS                                                  │
│   bucket_count     : u64                                      │
│   for each bucket:                                            │
│       len          : u64                                      │
│       ids          : len × u32  (ItemId)                     │
├──────────────────────────────────────────────────────────────┤
│ ITEMS                                                         │
│   item_count       : u64                                      │
│   item[item_count] : 28 bytes each →                         │
│       point        : 3 × f32   (12 B)                         │
│       payload_offset: u64      ( 8 B)                         │
│       payload_len  : u64       ( 8 B)                         │
├──────────────────────────────────────────────────────────────┤
│ PROJECTION MATRIX                                            │
│   proj_len         : u64       (= 3 × high_dim)              │
│   values           : proj_len × f32                          │
├──────────────────────────────────────────────────────────────┤
│ PAYLOAD ARENA (LZ4)                                          │
│   arena_decomp_len : u64       (raw size)                    │
│   arena_comp_len   : u64       (compressed size)            │
│   arena_compressed : arena_comp_len bytes (lz4_flex block)  │
└──────────────────────────────────────────────────────────────┘
```

## Notes

- **`bucket_id` indexing.** A leaf's `bucket_id` indexes the LEAF BUCKETS section
  positionally (bucket `0`, `1`, …). Subdividing a leaf leaves an empty,
  orphaned bucket entry behind; these are written out and reloaded as-is, so
  indices stay stable. Internal nodes store `bucket_id = NONE`.
- **Node `0` is the root** by construction.
- **Decompression is size-checked.** `load_from_disk` passes `arena_decomp_len`
  to `lz4_flex::decompress`; a corrupt block yields an
  `io::ErrorKind::InvalidData` error rather than a panic.
- **Validation gates** (in order): the 4-byte magic must equal `FRAC`; `version`
  must equal `3`; `high_dim` must equal the caller's `expected_high_dim`. Any
  failure returns a descriptive `io::Error`.

## Compatibility

The format is versioned. This crate writes and reads **v3 only**; earlier
experimental layouts are not supported. Bump `FILE_VERSION` and add a migration
path if you change any section.

## Size intuition

For `N` items with `bucket_capacity = 16`, expect roughly `N/12`–`N/8` nodes (52
B each) and `N` items (28 B each). The projection matrix is `12 × high_dim`
bytes. The payload arena dominates for text-like payloads and typically
compresses 4–6× with LZ4 (see [evaluation.md](evaluation.md)).
