# OctaSoma — Architecture

This document describes how the engine is built, from the in-memory data
structures up to the exact `k`-nearest-neighbour search. Everything here lives
in [`src/lib.rs`](../src/lib.rs) and is 100 % safe, stable Rust
(`#![forbid(unsafe_code)]`), with a single runtime dependency (`lz4_flex`).

## 1. The pipeline at a glance

```
embedding x ∈ ℝᴰ ──project_to_3d──▶ point p ∈ ℝ³ ──route──▶ leaf bucket
                                                                  │
query x' ─project─▶ p' ─nearest(p', k)─▶ exact k-NN ◀────────────┘
                                              │
                                       payload bytes (arena)
```

1. A `3 × D` **projection matrix** maps an embedding to a 3-D point.
2. The point is **routed** into a bucket point-region (PR) octree.
3. Queries run an **exact branch-and-bound k-NN** over the stored 3-D points.
4. Each item carries a slice into a flat **payload arena** (LZ4-compressed on disk).

## 2. Core data structures

### `FractalMemory3D`

The owning container. All state is in contiguous, cache-friendly vectors:

| Field | Type | Role |
|---|---|---|
| `nodes` | `Vec<OctreeNode>` | The octree; node `0` is always the root |
| `leaf_buckets` | `Vec<Vec<ItemId>>` | Per-leaf item-id lists, indexed by `bucket_id` |
| `items` | `Vec<Item>` | One entry per insertion: 3-D point + payload locator |
| `projection_matrix` | `Vec<f32>` | Flat row-major `3 × high_dim`, **unit-norm rows** |
| `payload_arena` | `Vec<u8>` | All payloads back-to-back |
| `world_half_size` | `f32` | Half-edge of the origin-centred root cube |
| `bucket_capacity` | `usize` | Items a leaf holds before subdividing (default 16) |
| `min_half_size` | `f32` | Subdivision stops at this cell size (default 1e-6) |

### `OctreeNode` — exactly one cache line

```rust
#[repr(C)]
pub struct OctreeNode {
    pub center: [f32; 3],   // 12 B — cube centre
    pub half_size: f32,     //  4 B — half the edge length
    pub children: [u32; 8], // 32 B — octant children, NONE = absent
    pub bucket_id: u32,     //  4 B — leaf bucket index, NONE = internal node
    _padding: [u8; 12],     // 12 B — pad to 64
}
const _: () = assert!(std::mem::size_of::<OctreeNode>() == 64);
```

A node is **internal** when `bucket_id == NONE` (routing through `children`) or a
**leaf** otherwise (owning a list in `leaf_buckets`). Keeping item lists *out* of
the node keeps the node array pure POD: sequential traversal touches one 64-byte
cache line per node and never chases a heap pointer until it reaches a leaf.

The `u32` index design (rather than `Option<Box<…>>`) means the whole tree is a
flat array that clones, serialises, and prefetches trivially. `NONE = u32::MAX`
is the universal "absent" sentinel, capping capacity at ~4.29 B nodes/items.

### `Item`

```rust
pub struct Item {
    pub point: [f32; 3],     // projection of the embedding — what distances use
    pub payload_offset: usize,
    pub payload_len: usize,
}
```

Storing the projected point per item is what turns retrieval into a genuine
nearest-neighbour search (instead of an arbitrary tree walk). The high-D
embedding itself is **not** retained — only its 3-D image.

## 3. Projection (high-D → 3-D)

`project_to_3d(embedding, matrix, D)` computes three dot products, each in an
`f64` accumulator over a 4-wide chunked loop. The `f64` accumulation plus a fixed
reduction order make the result **bit-identical across platforms** (x86-64 FMA
vs. ARM64 split multiply-add), which matters because the spatial layout — and
therefore every query answer — must be reproducible.

Two ways to obtain the matrix:

- **Johnson–Lindenstrauss (`FractalMemory3D::new`)** — fill the `3 × D` matrix
  from a seeded `Xorshift64` RNG. Deterministic, instantaneous, data-independent.
- **PCA (`new_with_pca`)** — `compute_pca_projection` centres the calibration
  data and extracts the top-3 principal components by **power iteration with
  Hotelling deflation**, all in `f64`. This learns the directions of maximal
  variance and is dramatically better at preserving topical structure (see
  [evaluation.md](evaluation.md)).

In both cases the three rows are **L2-normalised** (`normalise_rows`). For a
unit-norm embedding this bounds every coordinate to `[-1, 1]` (Cauchy–Schwarz),
so the default `world_half_size = 1.0` already contains normalised data.

## 4. Insertion

`insert(embedding, payload)`:

1. **Project** to `p`; return `None` on a dimension mismatch.
2. **Grow the world** if `p` falls outside it (Section 6).
3. **Stage the payload** into the arena, recording `(offset, len)`.
4. Push the `Item`, then **route** its id into a leaf.

### Routing & subdivision

`route` walks from the root following the octant of `p` at each node:

- **Internal node** → descend into `children[octant]`, lazily creating an empty
  leaf there if absent.
- **Leaf with room** (`len < capacity`) **or at minimum size** → append the id;
  done.
- **Full, divisible leaf** → `subdivide` it (convert to internal, redistribute
  its bucket's items into freshly created octant children), then continue the
  descent into the now-correct child.

The octant index is bitwise: `bit0 = x ≥ cx`, `bit1 = y ≥ cy`, `bit2 = z ≥ cz`.
Because each subdivision splits a cube into eight octants that exactly tile it,
and the point is guaranteed to be inside the root cube, containment is preserved
all the way down — the invariant the k-NN pruning relies on.

**Duplicates are safe.** Bit-identical points all funnel to the same cell; once
that cell reaches `min_half_size` the bucket simply grows (it is a `Vec`), so
nothing is ever dropped — verified by `duplicate_points_all_retained`.

## 5. Exact k-NN search

`nearest(point, k)` is a textbook **branch-and-bound** descent:

```
visit(node):
  if leaf:        offer every bucket item's squared distance to the result set
  else:           order existing children by cube-lower-bound distance, nearest
                  first; recurse, skipping any child whose lower bound already
                  exceeds the current k-th best (prune)
```

The lower bound is `cube_min_dist2(p, center, half)` — the squared distance from
`p` to the nearest point of a child's cube (0 if `p` is inside). Because this is
an **admissible** bound, pruning never discards a true neighbour: the result is
*identical to brute force*, only faster. This is asserted directly by
`octree_knn_matches_bruteforce`, which compares the two over thousands of random
points.

The bounded result set `KnnSet` keeps the best `k` pairs with linear scans —
cheaper than a binary heap for the small `k` typical of retrieval. `nearest`
returns `(ItemId, squared_distance)` ascending; `query`/`query_k` wrap it to
return payload bytes.

### Complexity

| Operation | Cost |
|---|---|
| Projection | `O(D)` |
| Insert (amortised) | `O(D + depth)`, `depth ≈ log₈ N` |
| World growth | `O(N)` rebuild, geometric ⇒ amortised `O(1)` |
| k-NN query | `O(D)` projection + pruned descent (≈ `O(log N + k)` typical; `O(N)` worst case) |
| Memory | `≈ N/capacity` leaves × 64 B + `N` × (`Item` 28 B + payload) |

## 6. The dynamic world

The root cube is centred at the origin with half-edge `world_half_size`. If an
incoming point has a coordinate outside `[-world, world]`, `grow_to` **doubles**
the half-size until it fits and **rebuilds** the tree from the stored item points
(`reset_root` + re-`route`). Doubling makes growth geometric, so the amortised
cost stays `O(1)` per insert. For L2-normalised embeddings with unit-norm
projection rows, coordinates already lie in `[-1, 1]` and growth never triggers.

## 7. Persistence

`save_to_disk` / `load_from_disk` write a versioned, little-endian `FRAC` v3 file
with the payload arena **LZ4-compressed**; the loader validates magic, version,
and `high_dim` before allocating, returning a descriptive `io::Error` (never a
panic) on any mismatch. The exact byte layout is in
[file-format.md](file-format.md).

## 8. What is deliberately *not* here

- **No `unsafe`, no `Box`/`Rc`/`RefCell`** — the tree is index-based.
- **No interior mutability / locks** — mutation is `&mut self`; concurrency is the
  caller's policy.
- **No non-linear projection** — a single `3 × D` linear map keeps the index
  exact and cheap; this is also its main limitation (see
  [evaluation.md](evaluation.md)).
