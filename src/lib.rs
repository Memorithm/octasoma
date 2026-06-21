//! # OctaSoma — 3D Fractal Semantic Memory Engine
//!
//! A 100% safe, stable Rust implementation of a 3-D semantic-memory engine for
//! embedding-based retrieval.  High-dimensional embeddings are projected to a
//! 3-D point with a *learned* (PCA) or *deterministic* (Johnson–Lindenstrauss)
//! linear map, then indexed by a **bucket point-region octree** that supports
//! **exact `k`-nearest-neighbour** queries in the projected space, with
//! LZ4-compressed zero-dependency persistence.
//!
//! ## Design
//!
//! - **Contiguous octree.** Every [`OctreeNode`] lives in a single [`Vec`] and
//!   is referenced by a `u32` index (sentinel [`NONE`] = no child).  There is no
//!   `Rc`, `RefCell`, `Box`, or `unsafe`.  Node traversal is cache-friendly:
//!   each node is exactly 64 bytes (one cache line).
//! - **Bucket leaves.** A leaf collects up to `bucket_capacity` items before it
//!   subdivides into eight octants and redistributes them.  Item id lists live
//!   in [`FractalMemory3D::leaf_buckets`], keeping the node array pure POD.
//! - **Stored items.** Each inserted embedding contributes one [`Item`] holding
//!   its *projected 3-D point* plus an offset/length into the payload arena.
//!   This is what makes retrieval a genuine nearest-neighbour search rather than
//!   an arbitrary tree walk.
//! - **Exact 3-D k-NN.** [`FractalMemory3D::nearest`] performs a branch-and-bound
//!   octree descent, pruning any sub-cube whose minimum distance to the query
//!   already exceeds the current k-th best.  The result is *identical* to brute
//!   force over the projected points — only faster.
//! - **Learned projection.** The `3 × D` matrix is filled either deterministically
//!   (Xorshift64, JL) or from a calibration dataset via power-iteration PCA with
//!   Hotelling deflation.  Rows are L2-normalised, so for a unit-norm embedding
//!   every projected coordinate lies in `[-1, 1]`.
//! - **Dynamic world.** The root cube grows (doubling) and the tree is rebuilt
//!   whenever a point falls outside the current bounds, so the index is correct
//!   for embeddings of any scale.
//! - **Compressed persistence.** [`FractalMemory3D::save_to_disk`] writes a
//!   versioned `FRAC` v3 file with an LZ4-compressed payload arena.

#![forbid(unsafe_code)]

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};

pub mod agent;
pub mod embed;
pub mod explain;
pub mod fractal;
pub mod kernel;
pub mod sharded;

pub use agent::OctaSomaAgent;
pub use embed::{EmbedError, Embedder, HashEmbedder, OllamaEmbedder};
pub use explain::{Explanation, Neighbor};
pub use fractal::RegionView;
pub use kernel::{KernelConfig, MEMORY_TOOL_SCHEMA_JSON, MemoryKernel, MemoryStep};
pub use sharded::ShardedMemory;

// ---------------------------------------------------------------------------
// Type aliases & sentinels
// ---------------------------------------------------------------------------

/// A direct index into the [`FractalMemory3D::nodes`] vector.
pub type NodeId = u32;
/// A direct index into the [`FractalMemory3D::items`] vector.
pub type ItemId = u32;

/// Sentinel meaning "no node" (used for absent children and for internal nodes
/// that own no bucket).
pub const NONE: u32 = u32::MAX;

/// Default number of items a leaf holds before it subdivides.
pub const DEFAULT_BUCKET_CAPACITY: usize = 16;
/// Default minimum half-size: subdivision stops once a cell is this small,
/// which bounds tree depth and lets bit-identical points share a leaf bucket.
pub const DEFAULT_MIN_HALF_SIZE: f32 = 1e-6;

// ---------------------------------------------------------------------------
// Deterministic pseudo-random number generator (Xorshift64)
// ---------------------------------------------------------------------------

/// A minimal, dependency-free deterministic RNG backed by the Xorshift64
/// algorithm.  From a fixed `u64` seed it always produces the same sequence,
/// making it suitable for reproducible projection-matrix generation.
#[derive(Clone, Debug)]
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    /// Seeds the generator.  A seed of `0` is promoted to a non-zero constant
    /// to avoid a permanently dead state (Xorshift64 requires `state != 0`).
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0xDEAD_BEEF_CAFE_BABE
            } else {
                seed
            },
        }
    }

    /// Advances the internal state and returns the raw 64-bit value.
    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns a deterministic `f32` in the range `[-1.0, 1.0)`.
    #[inline(always)]
    pub fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 32) as u32;
        // 0x3F80_0000 is 1.0_f32 in IEEE-754 binary32; OR in 23 mantissa bits to
        // build a uniform float in [1.0, 2.0), then rescale to [-1.0, 1.0).
        let normalised = f32::from_bits(0x3F80_0000 | ((bits >> 9) & 0x007F_FFFF));
        (normalised - 1.0) * 2.0 - 1.0
    }

    /// Returns a deterministic `f64` in the range `[-1.0, 1.0)` using the high
    /// 52 bits of the generator state for a full double-precision mantissa.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64();
        let normalised =
            f64::from_bits(0x3FF0_0000_0000_0000 | ((bits >> 12) & 0x000F_FFFF_FFFF_FFFF));
        (normalised - 1.0) * 2.0 - 1.0
    }
}

// ---------------------------------------------------------------------------
// Linear-algebra helpers (projection & PCA)
// ---------------------------------------------------------------------------

/// L2-normalises a slice in-place.  A (near-)zero vector is left untouched.
fn l2_normalise(v: &mut [f64]) {
    let norm_sq: f64 = v.iter().map(|x| x * x).sum();
    if norm_sq < f64::EPSILON {
        return;
    }
    let inv = 1.0 / norm_sq.sqrt();
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// L2-normalises each of the three rows of a flat `3 × d` row-major matrix.
fn normalise_rows(matrix: &mut [f32], d: usize) {
    for row in matrix.chunks_exact_mut(d) {
        let norm_sq: f64 = row.iter().map(|&x| (x as f64) * (x as f64)).sum();
        if norm_sq < f64::EPSILON {
            continue;
        }
        let inv = (1.0 / norm_sq.sqrt()) as f32;
        for x in row.iter_mut() {
            *x *= inv;
        }
    }
}

/// Matrix-vector multiplication on a row-major `n × d` matrix.
///
/// - `transpose == false`: returns `X · v` (length `n`).
/// - `transpose == true` : returns `Xᵀ · v` (length `d`).
fn mat_vec_mul(data: &[f64], n: usize, d: usize, vec: &[f64], transpose: bool) -> Vec<f64> {
    if transpose {
        let mut out = vec![0.0f64; d];
        for i in 0..n {
            let row = &data[i * d..(i + 1) * d];
            let scalar = vec[i];
            for j in 0..d {
                out[j] += row[j] * scalar;
            }
        }
        out
    } else {
        let mut out = vec![0.0f64; n];
        for i in 0..n {
            let row = &data[i * d..(i + 1) * d];
            let mut sum = 0.0f64;
            for j in 0..d {
                sum += row[j] * vec[j];
            }
            out[i] = sum;
        }
        out
    }
}

/// Computes the top-three principal components of a `num_samples × high_dim`
/// row-major data matrix via power iteration with Hotelling deflation.
///
/// Returns a flat `3 × high_dim` row-major matrix with unit-norm rows, suitable
/// for direct use as [`FractalMemory3D::projection_matrix`].  All arithmetic
/// uses `f64` accumulators for cross-platform reproducibility.
///
/// # Panics
/// Panics if `data.len() != num_samples * high_dim`, or if any dimension is 0.
pub fn compute_pca_projection(
    data: &[f32],
    num_samples: usize,
    high_dim: usize,
    max_iters: usize,
) -> Vec<f32> {
    assert_eq!(data.len(), num_samples * high_dim);
    assert!(num_samples > 0 && high_dim > 0 && max_iters > 0);

    let n = num_samples;
    let d = high_dim;

    // 1. Centre the data (f64 working copy).
    let mut mean = vec![0.0f64; d];
    for i in 0..n {
        let row = &data[i * d..(i + 1) * d];
        for (j, &val) in row.iter().enumerate() {
            mean[j] += val as f64;
        }
    }
    let inv_n = 1.0 / n as f64;
    for v in mean.iter_mut() {
        *v *= inv_n;
    }

    let mut centered: Vec<f64> = Vec::with_capacity(n * d);
    for i in 0..n {
        let row = &data[i * d..(i + 1) * d];
        for (j, &val) in row.iter().enumerate() {
            centered.push(val as f64 - mean[j]);
        }
    }

    // 2. Extract the top-3 eigenvectors.
    let mut projection = vec![0.0f32; 3 * d];
    let mut v = vec![0.0f64; d];
    let mut rng = DeterministicRng::new(0x50_4443_415F_4341); // "PDCA_CA"

    for comp in 0..3 {
        for elem in v.iter_mut() {
            *elem = rng.next_f64();
        }
        l2_normalise(&mut v);

        for _ in 0..max_iters {
            let xv = mat_vec_mul(&centered, n, d, &v, false); // X · v   → N-vec
            let xtxv = mat_vec_mul(&centered, n, d, &xv, true); // Xᵀ·(X·v) → D-vec
            v.copy_from_slice(&xtxv);
            l2_normalise(&mut v);
        }

        for j in 0..d {
            projection[comp * d + j] = v[j] as f32;
        }

        // Deflate: X ← X − (X·v)·vᵀ.
        if comp < 2 {
            let xv = mat_vec_mul(&centered, n, d, &v, false);
            for (row, &scalar) in centered.chunks_exact_mut(d).zip(xv.iter()) {
                for (elem, &vj) in row.iter_mut().zip(v.iter()) {
                    *elem -= scalar * vj;
                }
            }
        }
    }

    projection
}

/// Projects a high-dimensional `embedding` to a 3-D point using a flat
/// row-major projection matrix of shape `3 × high_dim`.
///
/// Each output coordinate is accumulated in `f64` over a chunked (4-element)
/// inner loop: this encourages SIMD auto-vectorisation while preserving a
/// strict sequential reduction order, so the result is bit-identical across
/// platforms (x86-64 FMA vs ARM64 split multiply-add).
///
/// Returns `None` if `embedding.len() != high_dim` or the matrix is too short.
pub fn project_to_3d(
    embedding: &[f32],
    projection_matrix: &[f32],
    high_dim: usize,
) -> Option<[f32; 3]> {
    if embedding.len() != high_dim || projection_matrix.len() < 3 * high_dim {
        return None;
    }

    let chunks = high_dim / 4;
    let remainder = high_dim % 4;
    let mut result = [0.0f64; 3];

    for (dim, acc) in result.iter_mut().enumerate() {
        let row = &projection_matrix[dim * high_dim..dim * high_dim + high_dim];
        let mut sum = 0.0f64;
        let mut j = 0;
        for _ in 0..chunks {
            sum += embedding[j] as f64 * row[j] as f64
                + embedding[j + 1] as f64 * row[j + 1] as f64
                + embedding[j + 2] as f64 * row[j + 2] as f64
                + embedding[j + 3] as f64 * row[j + 3] as f64;
            j += 4;
        }
        for k in 0..remainder {
            sum += embedding[j + k] as f64 * row[j + k] as f64;
        }
        *acc = sum;
    }

    Some([result[0] as f32, result[1] as f32, result[2] as f32])
}

/// Squared Euclidean distance between two 3-D points.
#[inline(always)]
fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// Squared distance from `p` to the nearest point of the axis-aligned cube
/// centred at `center` with the given `half` edge half-length.  Zero when `p`
/// is inside the cube.  This is the admissible lower bound that drives k-NN
/// branch-and-bound pruning.
#[inline]
fn cube_min_dist2(p: [f32; 3], center: [f32; 3], half: f32) -> f32 {
    let mut d2 = 0.0f32;
    for a in 0..3 {
        let excess = (p[a] - center[a]).abs() - half;
        if excess > 0.0 {
            d2 += excess * excess;
        }
    }
    d2
}

// ---------------------------------------------------------------------------
// Core node type (cache-line sized: exactly 64 bytes)
// ---------------------------------------------------------------------------

/// A single octree node, `#[repr(C)]` and padded to exactly 64 bytes (one cache
/// line) so sequential traversal never straddles an L1 boundary.
///
/// A node is either **internal** (`bucket_id == NONE`, routing through
/// `children`) or a **leaf** (`bucket_id` indexes [`FractalMemory3D::leaf_buckets`]).
#[repr(C)]
#[derive(Clone, Debug)]
pub struct OctreeNode {
    /// Spatial centre of this node's bounding cube.
    pub center: [f32; 3],
    /// Half the side-length of the cube (full side = `half_size * 2`).
    pub half_size: f32,
    /// Octant children; [`NONE`] marks an absent child.  Bit layout of the
    /// octant index `i`: bit0 = `x ≥ cx`, bit1 = `y ≥ cy`, bit2 = `z ≥ cz`.
    pub children: [NodeId; 8],
    /// Index into [`FractalMemory3D::leaf_buckets`] for a leaf, or [`NONE`] if
    /// this node is internal.
    pub bucket_id: u32,
    /// Padding to reach a 64-byte (single cache-line) footprint.
    _padding: [u8; 12],
}

// Compile-time assertion: a node must be exactly one cache line.
const _: () = {
    assert!(std::mem::size_of::<OctreeNode>() == 64);
};

impl OctreeNode {
    #[inline]
    fn leaf(center: [f32; 3], half_size: f32, bucket_id: u32) -> Self {
        Self {
            center,
            half_size,
            children: [NONE; 8],
            bucket_id,
            _padding: [0u8; 12],
        }
    }

    /// Whether this node is a leaf (owns a bucket) rather than an internal node.
    #[inline(always)]
    pub fn is_leaf(&self) -> bool {
        self.bucket_id != NONE
    }
}

// ---------------------------------------------------------------------------
// Stored item: a projected point plus a payload slice
// ---------------------------------------------------------------------------

/// One inserted memory: the 3-D projection of its embedding and the location of
/// its raw byte payload inside [`FractalMemory3D::payload_arena`].
#[derive(Clone, Debug)]
pub struct Item {
    /// 3-D projection of the embedding (the value distances are computed on).
    pub point: [f32; 3],
    /// Start offset of the payload within the arena.
    pub payload_offset: usize,
    /// Length of the payload within the arena.
    pub payload_len: usize,
}

// ---------------------------------------------------------------------------
// File header for on-disk persistence
// ---------------------------------------------------------------------------

/// Magic bytes identifying an OctaSoma binary file.
const FILE_MAGIC: [u8; 4] = *b"FRAC";
/// Current on-disk format version (v3: bucket octree + stored items).
const FILE_VERSION: u32 = 3;

struct FileHeader {
    magic: [u8; 4],
    version: u32,
    high_dim: u32,
}

impl FileHeader {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_all(&self.magic)?;
        w.write_all(&self.version.to_le_bytes())?;
        w.write_all(&self.high_dim.to_le_bytes())
    }

    fn read_from<R: Read>(r: &mut R) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        let version = read_u32(r)?;
        let high_dim = read_u32(r)?;
        Ok(Self {
            magic,
            version,
            high_dim,
        })
    }
}

// ---------------------------------------------------------------------------
// Primary container: FractalMemory3D
// ---------------------------------------------------------------------------

/// The central octree-backed 3-D semantic-memory engine.
///
/// All data lives in contiguous, cache-friendly vectors: [`nodes`] is the octree
/// itself, [`items`] holds projected points + payload locators, [`leaf_buckets`]
/// maps each leaf to its item ids, and [`payload_arena`] is one flat byte buffer.
#[derive(Clone)]
pub struct FractalMemory3D {
    /// Contiguous octree node store (node 0 is always the root).
    pub nodes: Vec<OctreeNode>,
    /// Per-leaf item-id lists, indexed by [`OctreeNode::bucket_id`].
    pub leaf_buckets: Vec<Vec<ItemId>>,
    /// Every inserted item (projected point + payload locator), in insertion order.
    pub items: Vec<Item>,
    /// Flat row-major projection matrix of shape `3 × high_dim`, rows unit-norm.
    pub projection_matrix: Vec<f32>,
    /// Dimensionality of input embeddings.
    pub high_dim: usize,
    /// Monolithic byte arena holding all payloads back to back.
    pub payload_arena: Vec<u8>,
    /// Half-edge of the (origin-centred) root cube; grows by doubling as needed.
    pub world_half_size: f32,
    /// Items a leaf holds before subdividing.
    pub bucket_capacity: usize,
    /// Subdivision stops once a cell's half-size reaches this value.
    pub min_half_size: f32,
}

impl FractalMemory3D {
    // -- construction --------------------------------------------------------

    fn new_empty(high_dim: usize, projection_matrix: Vec<f32>) -> Self {
        let mut s = Self {
            nodes: Vec::new(),
            leaf_buckets: Vec::new(),
            items: Vec::new(),
            projection_matrix,
            high_dim,
            payload_arena: Vec::new(),
            world_half_size: 1.0,
            bucket_capacity: DEFAULT_BUCKET_CAPACITY,
            min_half_size: DEFAULT_MIN_HALF_SIZE,
        };
        s.reset_root();
        s
    }

    /// Creates an empty engine with a deterministic Johnson–Lindenstrauss
    /// projection matrix derived from `seed`.  Identical `(high_dim, seed)`
    /// pairs produce bit-identical spatial layouts on any platform.
    pub fn new(high_dim: usize, seed: u64) -> Self {
        assert!(high_dim > 0, "high_dim must be non-zero");
        let mut rng = DeterministicRng::new(seed);
        let mut matrix = Vec::with_capacity(3 * high_dim);
        for _ in 0..3 * high_dim {
            matrix.push(rng.next_f32());
        }
        normalise_rows(&mut matrix, high_dim);
        Self::new_empty(high_dim, matrix)
    }

    /// Creates an empty engine from a pre-computed `3 × high_dim` projection.
    ///
    /// # Panics
    /// Panics if `projection_matrix.len() != 3 * high_dim`.
    pub fn new_from_calibration(high_dim: usize, mut projection_matrix: Vec<f32>) -> Self {
        assert_eq!(
            projection_matrix.len(),
            3 * high_dim,
            "projection_matrix must have length 3 * high_dim"
        );
        normalise_rows(&mut projection_matrix, high_dim);
        Self::new_empty(high_dim, projection_matrix)
    }

    /// Creates an empty engine whose projection is learned via PCA on a flat,
    /// row-major `num_samples × high_dim` calibration matrix.
    pub fn new_with_pca(high_dim: usize, calibration_data: &[f32], num_samples: usize) -> Self {
        let projection = compute_pca_projection(calibration_data, num_samples, high_dim, 20);
        Self::new_from_calibration(high_dim, projection)
    }

    /// Resets the tree to a single empty root leaf sized to `world_half_size`,
    /// keeping `items` and `payload_arena` intact (used during world growth).
    fn reset_root(&mut self) {
        self.nodes.clear();
        self.leaf_buckets.clear();
        self.nodes
            .push(OctreeNode::leaf([0.0; 3], self.world_half_size, 0));
        self.leaf_buckets.push(Vec::new());
    }

    // -- small accessors -----------------------------------------------------

    /// Number of octree nodes (internal + leaf).
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of stored items (i.e. successful insertions).
    #[inline]
    pub fn item_count(&self) -> usize {
        self.items.len()
    }

    /// Total payload-arena size in bytes.
    #[inline]
    pub fn arena_size(&self) -> usize {
        self.payload_arena.len()
    }

    /// Projects an embedding to its 3-D point (or `None` on dimension mismatch).
    #[inline]
    pub fn project(&self, embedding: &[f32]) -> Option<[f32; 3]> {
        project_to_3d(embedding, &self.projection_matrix, self.high_dim)
    }

    // -- octant geometry -----------------------------------------------------

    /// Octant index (0..=7) of `point` relative to `center`.
    #[inline(always)]
    pub fn octant_index(center: [f32; 3], point: [f32; 3]) -> usize {
        let mut i = 0;
        if point[0] >= center[0] {
            i |= 1;
        }
        if point[1] >= center[1] {
            i |= 2;
        }
        if point[2] >= center[2] {
            i |= 4;
        }
        i
    }

    /// Centre of child octant `octant` given the parent's centre and half-size.
    #[inline(always)]
    pub fn child_center(parent_center: [f32; 3], parent_half: f32, octant: usize) -> [f32; 3] {
        let q = parent_half * 0.5;
        [
            parent_center[0] + if octant & 1 != 0 { q } else { -q },
            parent_center[1] + if octant & 2 != 0 { q } else { -q },
            parent_center[2] + if octant & 4 != 0 { q } else { -q },
        ]
    }

    // -- insertion -----------------------------------------------------------

    /// Inserts an embedding with an optional byte payload and returns the new
    /// [`ItemId`].  Returns `None` only when `embedding.len() != high_dim`.
    ///
    /// The projected point is stored verbatim; the world grows (and the tree is
    /// rebuilt) if the point lies outside the current bounds; finally the item
    /// is routed to its containing leaf, which subdivides if it overflows.
    pub fn insert(&mut self, embedding: &[f32], payload: Option<&[u8]>) -> Option<ItemId> {
        let point = self.project(embedding)?;

        // Reject non-finite projections (NaN/inf embeddings) instead of storing a
        // poisoned point that would corrupt distance ordering and tree geometry.
        if !point.iter().all(|c| c.is_finite()) {
            return None;
        }

        // Grow the world to contain the point if necessary.
        let max_coord = point[0].abs().max(point[1].abs()).max(point[2].abs());
        if max_coord > self.world_half_size {
            self.grow_to(max_coord);
        }

        // Stage the payload into the arena.
        let (offset, len) = match payload {
            Some(data) => {
                let off = self.payload_arena.len();
                self.payload_arena.extend_from_slice(data);
                (off, data.len())
            }
            None => (self.payload_arena.len(), 0),
        };

        let item_id = self.items.len() as ItemId;
        self.items.push(Item {
            point,
            payload_offset: offset,
            payload_len: len,
        });
        self.route(item_id);
        Some(item_id)
    }

    /// Doubles `world_half_size` until it covers `max_coord`, then rebuilds the
    /// octree from the stored item points.  Amortised O(1) per insert.
    fn grow_to(&mut self, max_coord: f32) {
        while self.world_half_size < max_coord {
            self.world_half_size *= 2.0;
        }
        self.reset_root();
        for id in 0..self.items.len() as ItemId {
            self.route(id);
        }
    }

    /// Routes an already-stored item to its containing leaf, subdividing on
    /// overflow.  The point is guaranteed to be inside the root cube.
    fn route(&mut self, item_id: ItemId) {
        let point = self.items[item_id as usize].point;
        let mut cur: NodeId = 0;

        loop {
            let bucket_id = self.nodes[cur as usize].bucket_id;

            if bucket_id == NONE {
                // Internal node: descend into the octant child, creating it lazily.
                let center = self.nodes[cur as usize].center;
                let octant = Self::octant_index(center, point);
                let mut child = self.nodes[cur as usize].children[octant];
                if child == NONE {
                    let half = self.nodes[cur as usize].half_size;
                    child = self.new_leaf(Self::child_center(center, half, octant), half * 0.5);
                    self.nodes[cur as usize].children[octant] = child;
                }
                cur = child;
                continue;
            }

            // Leaf node: append unless it is full and still divisible.
            let half = self.nodes[cur as usize].half_size;
            let full = self.leaf_buckets[bucket_id as usize].len() >= self.bucket_capacity;
            if !full || half <= self.min_half_size {
                self.leaf_buckets[bucket_id as usize].push(item_id);
                return;
            }
            self.subdivide(cur);
            // `cur` is now internal — loop again to descend.
        }
    }

    /// Allocates a fresh empty leaf node and returns its id.
    fn new_leaf(&mut self, center: [f32; 3], half: f32) -> NodeId {
        let bucket_id = self.leaf_buckets.len() as u32;
        self.leaf_buckets.push(Vec::new());
        let id = self.nodes.len() as NodeId;
        self.nodes.push(OctreeNode::leaf(center, half, bucket_id));
        id
    }

    /// Converts a full leaf into an internal node, redistributing its items into
    /// freshly created octant children.
    fn subdivide(&mut self, node_id: NodeId) {
        let center = self.nodes[node_id as usize].center;
        let half = self.nodes[node_id as usize].half_size;
        let bucket_id = self.nodes[node_id as usize].bucket_id;

        let moved = std::mem::take(&mut self.leaf_buckets[bucket_id as usize]);
        self.nodes[node_id as usize].bucket_id = NONE; // now internal

        for item in moved {
            let p = self.items[item as usize].point;
            let octant = Self::octant_index(center, p);
            let mut child = self.nodes[node_id as usize].children[octant];
            if child == NONE {
                child = self.new_leaf(Self::child_center(center, half, octant), half * 0.5);
                self.nodes[node_id as usize].children[octant] = child;
            }
            let child_bucket = self.nodes[child as usize].bucket_id;
            self.leaf_buckets[child_bucket as usize].push(item);
        }
    }

    // -- retrieval -----------------------------------------------------------

    /// Exact `k`-nearest items to a query embedding, ascending by distance.
    /// Returns `(item_id, squared_distance)` pairs.  Empty if the engine holds
    /// no items or the embedding has the wrong dimensionality.
    pub fn nearest_embedding(&self, embedding: &[f32], k: usize) -> Vec<(ItemId, f32)> {
        match self.project(embedding) {
            Some(p) => self.nearest(p, k),
            None => Vec::new(),
        }
    }

    /// Exact `k`-nearest items to a 3-D `point`, ascending by distance.
    ///
    /// Branch-and-bound octree descent: children are visited nearest-cube-first
    /// and any sub-cube whose lower-bound distance already exceeds the current
    /// k-th best is pruned.  The result is identical to brute force, only faster.
    pub fn nearest(&self, point: [f32; 3], k: usize) -> Vec<(ItemId, f32)> {
        if k == 0 || self.items.is_empty() || !point.iter().all(|c| c.is_finite()) {
            return Vec::new();
        }
        let mut heap = KnnSet::new(k);
        self.nn_descend(0, point, &mut heap);
        heap.into_sorted()
    }

    fn nn_descend(&self, node_id: NodeId, point: [f32; 3], heap: &mut KnnSet) {
        let node = &self.nodes[node_id as usize];

        if node.is_leaf() {
            for &item in &self.leaf_buckets[node.bucket_id as usize] {
                let d2 = dist2(point, self.items[item as usize].point);
                heap.offer(item, d2);
            }
            return;
        }

        // Order existing children by their cube lower-bound distance.
        let mut order: [(f32, NodeId); 8] = [(f32::INFINITY, NONE); 8];
        let mut count = 0;
        for &child in node.children.iter() {
            if child != NONE {
                let c = &self.nodes[child as usize];
                order[count] = (cube_min_dist2(point, c.center, c.half_size), child);
                count += 1;
            }
        }
        order[..count].sort_by(|a, b| a.0.total_cmp(&b.0));

        for &(lower_bound, child) in &order[..count] {
            if heap.is_full() && lower_bound >= heap.worst() {
                continue; // prune: nothing in this sub-cube can beat the k-th best
            }
            self.nn_descend(child, point, heap);
        }
    }

    /// Brute-force exact `k`-NN over every stored item (reference implementation
    /// used for testing and benchmarking the octree against ground truth).
    pub fn nearest_bruteforce(&self, point: [f32; 3], k: usize) -> Vec<(ItemId, f32)> {
        if k == 0 {
            return Vec::new();
        }
        let mut heap = KnnSet::new(k);
        for (id, item) in self.items.iter().enumerate() {
            heap.offer(id as ItemId, dist2(point, item.point));
        }
        heap.into_sorted()
    }

    /// Convenience top-1 lookup: the payload of the single nearest item.
    pub fn query(&self, embedding: &[f32]) -> Option<&[u8]> {
        let best = self.nearest_embedding(embedding, 1);
        self.get_payload(best.first()?.0)
    }

    /// Top-`k` payloads for a query embedding, nearest first.
    pub fn query_k(&self, embedding: &[f32], k: usize) -> Vec<&[u8]> {
        self.nearest_embedding(embedding, k)
            .into_iter()
            .filter_map(|(id, _)| self.get_payload(id))
            .collect()
    }

    /// Zero-copy reference to an item's raw payload, with full bounds checking.
    pub fn get_payload(&self, item_id: ItemId) -> Option<&[u8]> {
        let item = self.items.get(item_id as usize)?;
        let end = item.payload_offset.checked_add(item.payload_len)?;
        if end > self.payload_arena.len() {
            return None;
        }
        Some(&self.payload_arena[item.payload_offset..end])
    }

    // -- persistence ---------------------------------------------------------

    /// Serialises the engine to a `FRAC` v3 file (little-endian, LZ4-compressed
    /// payload arena).
    pub fn save_to_disk(&self, path: &str) -> io::Result<()> {
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);

        FileHeader {
            magic: FILE_MAGIC,
            version: FILE_VERSION,
            high_dim: self.high_dim as u32,
        }
        .write_to(&mut w)?;

        // Engine parameters.
        w.write_all(&self.world_half_size.to_le_bytes())?;
        w.write_all(&(self.bucket_capacity as u64).to_le_bytes())?;
        w.write_all(&self.min_half_size.to_le_bytes())?;

        // Nodes.
        w.write_all(&(self.nodes.len() as u64).to_le_bytes())?;
        for node in &self.nodes {
            for c in node.center {
                w.write_all(&c.to_le_bytes())?;
            }
            w.write_all(&node.half_size.to_le_bytes())?;
            for &child in &node.children {
                w.write_all(&child.to_le_bytes())?;
            }
            w.write_all(&node.bucket_id.to_le_bytes())?;
        }

        // Leaf buckets.
        w.write_all(&(self.leaf_buckets.len() as u64).to_le_bytes())?;
        for bucket in &self.leaf_buckets {
            w.write_all(&(bucket.len() as u64).to_le_bytes())?;
            for &id in bucket {
                w.write_all(&id.to_le_bytes())?;
            }
        }

        // Items.
        w.write_all(&(self.items.len() as u64).to_le_bytes())?;
        for item in &self.items {
            for c in item.point {
                w.write_all(&c.to_le_bytes())?;
            }
            w.write_all(&(item.payload_offset as u64).to_le_bytes())?;
            w.write_all(&(item.payload_len as u64).to_le_bytes())?;
        }

        // Projection matrix.
        w.write_all(&(self.projection_matrix.len() as u64).to_le_bytes())?;
        for &v in &self.projection_matrix {
            w.write_all(&v.to_le_bytes())?;
        }

        // Payload arena (LZ4-compressed).
        w.write_all(&(self.payload_arena.len() as u64).to_le_bytes())?;
        let compressed = lz4_flex::compress(&self.payload_arena);
        w.write_all(&(compressed.len() as u64).to_le_bytes())?;
        w.write_all(&compressed)?;

        w.flush()
    }

    /// Loads an engine previously written by [`save_to_disk`].
    ///
    /// Validates the magic bytes, format version, and that the file's `high_dim`
    /// equals `expected_high_dim`; any mismatch yields a descriptive
    /// [`io::Error`] rather than a panic.
    pub fn load_from_disk(path: &str, expected_high_dim: usize) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut r = BufReader::new(file);

        let header = FileHeader::read_from(&mut r)?;
        if header.magic != FILE_MAGIC {
            return Err(invalid(format!(
                "invalid magic: expected {:?}, got {:?}",
                FILE_MAGIC, header.magic
            )));
        }
        if header.version != FILE_VERSION {
            return Err(invalid(format!(
                "unsupported file version {} (this build writes/reads v{})",
                header.version, FILE_VERSION
            )));
        }
        if header.high_dim as usize != expected_high_dim {
            return Err(invalid(format!(
                "high_dim mismatch: file has {}, caller expected {}",
                header.high_dim, expected_high_dim
            )));
        }

        let world_half_size = read_f32(&mut r)?;
        let bucket_capacity = read_u64(&mut r)? as usize;
        let min_half_size = read_f32(&mut r)?;

        // Nodes.
        let node_count = read_u64(&mut r)? as usize;
        let mut nodes = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            let center = [read_f32(&mut r)?, read_f32(&mut r)?, read_f32(&mut r)?];
            let half_size = read_f32(&mut r)?;
            let mut children = [NONE; 8];
            for child in children.iter_mut() {
                *child = read_u32(&mut r)?;
            }
            let bucket_id = read_u32(&mut r)?;
            nodes.push(OctreeNode {
                center,
                half_size,
                children,
                bucket_id,
                _padding: [0u8; 12],
            });
        }

        // Leaf buckets.
        let bucket_count = read_u64(&mut r)? as usize;
        let mut leaf_buckets = Vec::with_capacity(bucket_count);
        for _ in 0..bucket_count {
            let len = read_u64(&mut r)? as usize;
            let mut bucket = Vec::with_capacity(len);
            for _ in 0..len {
                bucket.push(read_u32(&mut r)?);
            }
            leaf_buckets.push(bucket);
        }

        // Items.
        let item_count = read_u64(&mut r)? as usize;
        let mut items = Vec::with_capacity(item_count);
        for _ in 0..item_count {
            let point = [read_f32(&mut r)?, read_f32(&mut r)?, read_f32(&mut r)?];
            let payload_offset = read_u64(&mut r)? as usize;
            let payload_len = read_u64(&mut r)? as usize;
            items.push(Item {
                point,
                payload_offset,
                payload_len,
            });
        }

        // Projection matrix.
        let proj_len = read_u64(&mut r)? as usize;
        let mut projection_matrix = Vec::with_capacity(proj_len);
        for _ in 0..proj_len {
            projection_matrix.push(read_f32(&mut r)?);
        }

        // Payload arena (LZ4).
        let decomp_len = read_u64(&mut r)? as usize;
        let comp_len = read_u64(&mut r)? as usize;
        let mut compressed = vec![0u8; comp_len];
        r.read_exact(&mut compressed)?;
        let payload_arena = lz4_flex::decompress(&compressed, decomp_len)
            .map_err(|e| invalid(format!("lz4 decompression failed: {e}")))?;

        Ok(Self {
            nodes,
            leaf_buckets,
            items,
            projection_matrix,
            high_dim: expected_high_dim,
            payload_arena,
            world_half_size,
            bucket_capacity,
            min_half_size,
        })
    }
}

// ---------------------------------------------------------------------------
// Bounded k-NN result set (small-k friendly)
// ---------------------------------------------------------------------------

/// A bounded best-of-`k` set keyed by squared distance.  For the small `k`
/// typical of retrieval, the linear scans here beat a binary heap's overhead.
struct KnnSet {
    k: usize,
    best: Vec<(ItemId, f32)>,
}

impl KnnSet {
    fn new(k: usize) -> Self {
        let k = k.max(1);
        Self {
            k,
            best: Vec::with_capacity(k),
        }
    }

    #[inline]
    fn is_full(&self) -> bool {
        self.best.len() >= self.k
    }

    /// Current worst (largest) squared distance held, or +∞ when not yet full.
    #[inline]
    fn worst(&self) -> f32 {
        if self.best.len() < self.k {
            f32::INFINITY
        } else {
            self.best
                .iter()
                .fold(f32::NEG_INFINITY, |m, &(_, d)| m.max(d))
        }
    }

    fn offer(&mut self, item: ItemId, d2: f32) {
        if self.best.len() < self.k {
            self.best.push((item, d2));
            return;
        }
        // Replace the current maximum if this candidate is strictly closer.
        let mut max_i = 0;
        let mut max_v = f32::NEG_INFINITY;
        for (i, &(_, d)) in self.best.iter().enumerate() {
            if d > max_v {
                max_v = d;
                max_i = i;
            }
        }
        if d2 < max_v {
            self.best[max_i] = (item, d2);
        }
    }

    fn into_sorted(mut self) -> Vec<(ItemId, f32)> {
        self.best.sort_by(|a, b| a.1.total_cmp(&b.1));
        self.best
    }
}

// ---------------------------------------------------------------------------
// Little-endian read helpers
// ---------------------------------------------------------------------------

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
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

fn read_f32<R: Read>(r: &mut R) -> io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- DeterministicRng ---------------------------------------------------

    #[test]
    fn rng_determinism() {
        let mut a = DeterministicRng::new(42);
        let mut b = DeterministicRng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_f32(), b.next_f32());
        }
    }

    #[test]
    fn rng_range() {
        let mut rng = DeterministicRng::new(12345);
        for _ in 0..10_000 {
            let v = rng.next_f32();
            assert!((-1.0..1.0).contains(&v), "out of range: {v}");
        }
    }

    // -- projection ---------------------------------------------------------

    #[test]
    fn projection_shape_mismatch_rejected() {
        assert!(project_to_3d(&[1.0, 2.0], &[], 4).is_none());
        assert!(project_to_3d(&[1.0], &[0.0; 2], 1).is_none());
        assert!(project_to_3d(&[1.0], &[0.0, 0.0, 0.0], 1).is_some());
    }

    #[test]
    fn projection_rows_are_unit_norm() {
        let mem = FractalMemory3D::new(32, 7);
        for row in mem.projection_matrix.chunks_exact(32) {
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "row norm {norm} not unit");
        }
    }

    #[test]
    fn projection_with_remainder() {
        // 7-dim embedding (not a multiple of 4) exercises the tail loop.
        let mat: Vec<f32> = (0..21).map(|i| i as f32 * 0.1).collect();
        let emb: Vec<f32> = (0..7).map(|i| (i + 1) as f32 * 0.2).collect();
        let pt = project_to_3d(&emb, &mat, 7).unwrap();
        assert!(pt.iter().all(|x| x.is_finite()));
    }

    // -- octant geometry ----------------------------------------------------

    #[test]
    fn octant_bits() {
        assert_eq!(FractalMemory3D::octant_index([0.0; 3], [1.0, 1.0, 1.0]), 7);
        assert_eq!(
            FractalMemory3D::octant_index([0.0; 3], [-1.0, -1.0, -1.0]),
            0
        );
        assert_eq!(
            FractalMemory3D::octant_index([0.0; 3], [1.0, -1.0, -1.0]),
            1
        );
        assert_eq!(
            FractalMemory3D::octant_index([0.0; 3], [-1.0, 1.0, -1.0]),
            2
        );
        assert_eq!(
            FractalMemory3D::octant_index([0.0; 3], [-1.0, -1.0, 1.0]),
            4
        );
    }

    #[test]
    fn cube_min_dist2_zero_inside() {
        assert_eq!(cube_min_dist2([0.1, -0.2, 0.3], [0.0; 3], 1.0), 0.0);
        // Outside on a single axis: distance is the axis excess.
        let d = cube_min_dist2([3.0, 0.0, 0.0], [0.0; 3], 1.0);
        assert!((d - 4.0).abs() < 1e-6);
    }

    // -- node layout --------------------------------------------------------

    #[test]
    fn octree_node_is_one_cache_line() {
        assert_eq!(std::mem::size_of::<OctreeNode>(), 64);
    }

    // -- insertion & exact k-NN --------------------------------------------

    #[test]
    fn empty_engine_has_root_only() {
        let mem = FractalMemory3D::new(8, 0);
        assert_eq!(mem.node_count(), 1);
        assert_eq!(mem.item_count(), 0);
        assert!(mem.query(&[0.0; 8]).is_none());
    }

    #[test]
    fn octree_knn_matches_bruteforce() {
        let mut mem = FractalMemory3D::new(12, 2024);
        let mut rng = DeterministicRng::new(1);
        for i in 0..2000 {
            let emb: Vec<f32> = (0..12).map(|_| rng.next_f32()).collect();
            mem.insert(&emb, Some(format!("item-{i}").as_bytes()))
                .unwrap();
        }
        // For many random queries, exact octree k-NN must equal brute force.
        for _ in 0..200 {
            let q: Vec<f32> = (0..12).map(|_| rng.next_f32()).collect();
            let p = mem.project(&q).unwrap();
            let a = mem.nearest(p, 5);
            let b = mem.nearest_bruteforce(p, 5);
            let da: Vec<f32> = a.iter().map(|x| x.1).collect();
            let db: Vec<f32> = b.iter().map(|x| x.1).collect();
            assert_eq!(da, db, "octree k-NN distances diverged from brute force");
        }
    }

    #[test]
    fn self_query_returns_own_payload() {
        let mut mem = FractalMemory3D::new(16, 99);
        let mut rng = DeterministicRng::new(7);
        let mut embeddings = Vec::new();
        for i in 0..500 {
            let emb: Vec<f32> = (0..16).map(|_| rng.next_f32()).collect();
            mem.insert(&emb, Some(format!("p{i}").as_bytes())).unwrap();
            embeddings.push((i, emb));
        }
        // Querying an inserted embedding returns its own payload (nearest = self).
        for (i, emb) in embeddings.iter().take(100) {
            let got = mem.query(emb).unwrap();
            assert_eq!(got, format!("p{i}").as_bytes());
        }
    }

    #[test]
    fn duplicate_points_all_retained() {
        let mut mem = FractalMemory3D::new(4, 5);
        let emb = [0.3f32, -0.1, 0.7, 0.2];
        for i in 0..100 {
            mem.insert(&emb, Some(format!("dup{i}").as_bytes()))
                .unwrap();
        }
        assert_eq!(mem.item_count(), 100);
        // All 100 identical points must be returned for a k=100 query.
        let res = mem.nearest_embedding(&emb, 100);
        assert_eq!(res.len(), 100);
        assert!(res.iter().all(|&(_, d)| d == 0.0));
    }

    #[test]
    fn world_grows_for_large_embeddings() {
        let mut mem = FractalMemory3D::new(3, 0);
        // Identity-like projection rows are unit-norm, so a big embedding yields
        // a large coordinate that forces the world to grow.
        let big: Vec<f32> = vec![1000.0, -500.0, 250.0];
        mem.insert(&big, Some(b"far")).unwrap();
        assert!(mem.world_half_size >= 1000.0);
        assert_eq!(mem.query(&big).unwrap(), b"far");
    }

    // -- payloads -----------------------------------------------------------

    #[test]
    fn payload_roundtrip_and_guards() {
        let mut mem = FractalMemory3D::new(8, 99);
        let id = mem.insert(&[0.0; 8], Some(b"hello fractal")).unwrap();
        assert_eq!(mem.get_payload(id).unwrap(), b"hello fractal");
        assert!(mem.get_payload(NONE).is_none());
        assert!(mem.get_payload(12345).is_none());
    }

    // -- persistence --------------------------------------------------------

    #[test]
    fn save_and_load_roundtrip() {
        let mut mem = FractalMemory3D::new(10, 123);
        let mut rng = DeterministicRng::new(3);
        for i in 0..1000 {
            let emb: Vec<f32> = (0..10).map(|_| rng.next_f32()).collect();
            mem.insert(&emb, Some(format!("m{i}").as_bytes())).unwrap();
        }
        let path = "/tmp/octasoma_v3_roundtrip.frac";
        mem.save_to_disk(path).unwrap();
        let loaded = FractalMemory3D::load_from_disk(path, 10).unwrap();

        assert_eq!(loaded.node_count(), mem.node_count());
        assert_eq!(loaded.item_count(), mem.item_count());
        assert_eq!(loaded.payload_arena, mem.payload_arena);
        assert_eq!(loaded.projection_matrix, mem.projection_matrix);

        // Retrieval is identical after a round-trip.
        let mut rng2 = DeterministicRng::new(55);
        for _ in 0..100 {
            let q: Vec<f32> = (0..10).map(|_| rng2.next_f32()).collect();
            assert_eq!(mem.query(&q), loaded.query(&q));
        }
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn load_rejects_wrong_dimension_and_bad_magic() {
        let mut mem = FractalMemory3D::new(8, 0);
        mem.insert(&[0.0; 8], None).unwrap();
        let path = "/tmp/octasoma_v3_dim.frac";
        mem.save_to_disk(path).unwrap();
        assert!(FractalMemory3D::load_from_disk(path, 16).is_err());
        std::fs::remove_file(path).ok();

        let bad = "/tmp/octasoma_v3_badmagic.frac";
        std::fs::write(bad, b"NOPE....").unwrap();
        assert!(FractalMemory3D::load_from_disk(bad, 8).is_err());
        std::fs::remove_file(bad).ok();
    }

    #[test]
    fn arena_compresses_redundant_payloads() {
        let mut mem = FractalMemory3D::new(4, 1);
        mem.insert(&[0.0; 4], Some(&vec![0xABu8; 8192])).unwrap();
        let path = "/tmp/octasoma_v3_compress.frac";
        mem.save_to_disk(path).unwrap();
        let size = std::fs::metadata(path).unwrap().len();
        assert!(
            size < 2048,
            "8 KiB of redundant data should compress well; got {size}"
        );
        let loaded = FractalMemory3D::load_from_disk(path, 4).unwrap();
        assert_eq!(loaded.payload_arena.len(), 8192);
        std::fs::remove_file(path).ok();
    }

    // -- PCA ----------------------------------------------------------------

    #[test]
    fn pca_first_component_aligns_with_dominant_axis() {
        let n = 80;
        let d = 8;
        let mut data = vec![0.0f32; n * d];
        for i in 0..n {
            data[i * d] = (i as f32 - 40.0) * 0.1; // dominant variance on dim 0
            data[i * d + 1] = (i as f32 - 40.0) * 0.05;
        }
        let mut rng = DeterministicRng::new(999);
        for v in data.iter_mut() {
            *v += rng.next_f32() * 0.001;
        }
        let proj = compute_pca_projection(&data, n, d, 20);
        assert_eq!(proj.len(), 3 * d);
        let row0 = &proj[0..d];
        let max_idx = row0
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .unwrap()
            .0;
        assert!(
            max_idx <= 1,
            "first PC should load on dim 0/1, got {max_idx}"
        );
    }

    #[test]
    fn pca_engine_is_deterministic() {
        let n = 40;
        let d = 6;
        let mut data = vec![0.0f32; n * d];
        for i in 0..n {
            data[i * d] = i as f32 * 0.1;
        }
        let a = FractalMemory3D::new_with_pca(d, &data, n);
        let b = FractalMemory3D::new_with_pca(d, &data, n);
        assert_eq!(a.projection_matrix, b.projection_matrix);
    }

    #[test]
    fn new_from_calibration_panics_on_bad_shape() {
        let r = std::panic::catch_unwind(|| {
            FractalMemory3D::new_from_calibration(8, vec![0.0f32; 20]);
        });
        assert!(r.is_err());
    }
}
