# OctaSoma — API Reference

Everything is exported from the crate root. Generate browsable docs with
`cargo doc --open`. Types are 100 % safe Rust; the only dependency is `lz4_flex`.

## Types & constants

```rust
pub type NodeId = u32;            // index into FractalMemory3D::nodes
pub type ItemId = u32;            // index into FractalMemory3D::items
pub const NONE: u32 = u32::MAX;   // "absent" sentinel (no child / internal node)

pub const DEFAULT_BUCKET_CAPACITY: usize = 16;
pub const DEFAULT_MIN_HALF_SIZE: f32 = 1e-6;
```

## `FractalMemory3D`

The engine. Public fields are exposed for inspection and tooling; prefer the
methods below for normal use.

### Construction

| Method | Signature | Notes |
|---|---|---|
| `new` | `(high_dim: usize, seed: u64) -> Self` | Deterministic Johnson–Lindenstrauss projection. Panics if `high_dim == 0`. |
| `new_with_pca` | `(high_dim: usize, calibration_data: &[f32], num_samples: usize) -> Self` | Learns the projection by PCA on a flat row-major `num_samples × high_dim` matrix. |
| `new_from_calibration` | `(high_dim: usize, projection_matrix: Vec<f32>) -> Self` | Bring your own `3 × high_dim` matrix (rows are L2-normalised for you). Panics on wrong length. |

```rust
use octasoma::FractalMemory3D;

let mut jl  = FractalMemory3D::new(768, 42);
let mut pca = FractalMemory3D::new_with_pca(768, &calibration, num_samples);
```

### Insertion

```rust
pub fn insert(&mut self, embedding: &[f32], payload: Option<&[u8]>) -> Option<ItemId>
```

Projects, grows the world if needed, stores the payload, and routes the item.
Returns the new `ItemId`, or `None` iff `embedding.len() != high_dim`.

```rust
let id = mem.insert(&vec![0.1; 768], Some(b"a memory")).unwrap();
mem.insert(&vec![0.2; 768], None); // payload is optional
```

### Retrieval

| Method | Signature | Returns |
|---|---|---|
| `query` | `(&self, embedding: &[f32]) -> Option<&[u8]>` | Payload of the single nearest item |
| `query_k` | `(&self, embedding: &[f32], k: usize) -> Vec<&[u8]>` | Payloads of the `k` nearest, nearest first |
| `nearest_embedding` | `(&self, embedding: &[f32], k: usize) -> Vec<(ItemId, f32)>` | `(id, squared distance)`, ascending |
| `nearest` | `(&self, point: [f32; 3], k: usize) -> Vec<(ItemId, f32)>` | Same, for a pre-projected point |
| `nearest_bruteforce` | `(&self, point: [f32; 3], k: usize) -> Vec<(ItemId, f32)>` | Reference linear scan (testing/benchmarking) |
| `get_payload` | `(&self, id: ItemId) -> Option<&[u8]>` | Bounds-checked zero-copy payload slice |
| `project` | `(&self, embedding: &[f32]) -> Option<[f32; 3]>` | The 3-D projection of an embedding |

`nearest` is **exact**: its results are identical to `nearest_bruteforce`, only
faster (branch-and-bound pruning).

```rust
if let Some(p) = mem.query(&q) {
    println!("{}", String::from_utf8_lossy(p));
}
for (id, dist2) in mem.nearest_embedding(&q, 5) {
    println!("item {id} at squared distance {dist2}");
}
```

### Persistence

| Method | Signature |
|---|---|
| `save_to_disk` | `(&self, path: &str) -> std::io::Result<()>` |
| `load_from_disk` | `(path: &str, expected_high_dim: usize) -> std::io::Result<Self>` |

`save_to_disk` writes a `FRAC` v3 file with an LZ4-compressed payload arena.
`load_from_disk` validates magic, version, and `high_dim`, returning a
descriptive `io::Error` (never a panic) on mismatch. See
[file-format.md](file-format.md).

```rust
mem.save_to_disk("memory.frac")?;
let mem = FractalMemory3D::load_from_disk("memory.frac", 768)?;
```

### Introspection

| Method | Returns |
|---|---|
| `node_count(&self) -> usize` | Number of octree nodes |
| `item_count(&self) -> usize` | Number of stored items (successful inserts) |
| `arena_size(&self) -> usize` | Payload-arena size in bytes |

### Static geometry helpers

```rust
FractalMemory3D::octant_index(center: [f32;3], point: [f32;3]) -> usize // 0..=7
FractalMemory3D::child_center(parent_center: [f32;3], parent_half: f32, octant: usize) -> [f32;3]
```

## Free functions

```rust
pub fn project_to_3d(embedding: &[f32], projection_matrix: &[f32], high_dim: usize) -> Option<[f32; 3]>
pub fn compute_pca_projection(data: &[f32], num_samples: usize, high_dim: usize, max_iters: usize) -> Vec<f32>
```

- `project_to_3d` — the platform-deterministic `f64`-accumulated projection.
- `compute_pca_projection` — top-3 PCs via power iteration + Hotelling deflation;
  returns a flat `3 × high_dim` matrix with unit-norm rows. Panics on a
  zero/inconsistent shape.

## `DeterministicRng`

A dependency-free, seedable Xorshift64 generator (also useful for reproducible
test/benchmark data).

```rust
let mut rng = octasoma::DeterministicRng::new(42);
let _u: u64 = rng.next_u64();
let _f: f32 = rng.next_f32();   // [-1.0, 1.0)
let _d: f64 = rng.next_f64();   // [-1.0, 1.0)
```

## Error & edge-case behaviour

- Dimension mismatch (`embedding.len() != high_dim`) → `insert`/`query` return
  `None`/empty rather than panicking.
- `get_payload` is fully bounds-checked (`checked_add`), so a corrupt/foreign id
  yields `None`.
- Constructors panic only on programmer error (`high_dim == 0`, wrong matrix
  length) — never on user data at runtime.
