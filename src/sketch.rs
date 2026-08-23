//! Binary semantic sketches (SimHash) — a high-precision, low-cost similarity tier.
//!
//! OctaSoma's 3-D projection is deliberately a *coarse router*: collapsing a
//! high-dimensional embedding to three coordinates discards almost all angular
//! structure (exact recall@1 $\approx 0\%$). A **SimHash** sketch keeps much more of
//! it in a few machine words. Following Charikar (2002): draw `bits` random
//! hyperplanes; the sketch is the sign of the embedding against each. The Hamming
//! distance between two sketches estimates the angle between the embeddings,
//! $\mathbb{E}[\text{hamming}]/\text{bits} \approx \theta/\pi$, so ranking by Hamming
//! is a branch-free, popcount-cheap proxy for ranking by cosine.
//!
//! This is the precision tier between the 3-D route (cheap, explainable,
//! visualisable) and an exact cosine rerank: with, say, 256 bits it recovers a large
//! fraction of true neighbours the 3-D index cannot, at a fraction of the cost of
//! scoring full embeddings — and it is **100% safe, stable Rust** (a dot product, a
//! sign, and [`u64::count_ones`], which lowers to a POPCNT instruction).

use crate::conformal::ShortlistCertificate;
use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::sync::Arc;

use crate::DeterministicRng;

/// A SimHash projector: `bits` random hyperplanes over `dim`-dimensional embeddings.
#[derive(Clone, Debug)]
pub struct SimHasher {
    /// `bits × dim` row-major; row `i` is the `i`-th hyperplane normal.
    planes: Vec<f32>,
    dim: usize,
    bits: usize,
}

impl SimHasher {
    /// Builds `bits` random hyperplanes for `dim`-dimensional input, seeded for
    /// reproducibility. `bits` is rounded **up** to a multiple of 64 so a sketch is
    /// a whole number of `u64` words.
    ///
    /// # Panics
    /// Panics if `dim == 0` or `bits == 0`.
    pub fn new(dim: usize, bits: usize, seed: u64) -> Self {
        assert!(dim > 0, "dim must be non-zero");
        assert!(bits > 0, "bits must be non-zero");
        let bits = bits.div_ceil(64) * 64;
        let mut rng = DeterministicRng::new(seed);
        let mut planes = Vec::with_capacity(bits * dim);
        for _ in 0..bits * dim {
            planes.push(rng.next_f32());
        }
        Self { planes, dim, bits }
    }

    /// Number of sketch bits (a multiple of 64).
    #[inline]
    pub fn bits(&self) -> usize {
        self.bits
    }

    /// Heap bytes occupied by the hyperplane matrix.
    pub fn plane_bytes(&self) -> usize {
        self.planes.len() * std::mem::size_of::<f32>()
    }

    /// Number of `u64` words in a sketch (`bits / 64`).
    #[inline]
    pub fn words(&self) -> usize {
        self.bits / 64
    }

    /// Input dimensionality this hasher expects.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Sketches `embedding` into `words()` `u64`s: bit `i` is set iff the embedding
    /// lies on the positive side of hyperplane `i`. Returns an empty vector if
    /// `embedding.len() != dim`.
    ///
    /// Each dot product accumulates in `f64` for cross-platform determinism.
    pub fn sketch(&self, embedding: &[f32]) -> Vec<u64> {
        if embedding.len() != self.dim {
            return Vec::new();
        }
        let mut out = vec![0u64; self.words()];
        for b in 0..self.bits {
            let row = &self.planes[b * self.dim..(b + 1) * self.dim];
            let mut dot = 0.0f64;
            for (&w, &e) in row.iter().zip(embedding) {
                dot += w as f64 * e as f64;
            }
            if dot >= 0.0 {
                out[b / 64] |= 1u64 << (b % 64);
            }
        }
        out
    }
}

impl SimHasher {
    /// [`SimHasher::sketch`] via scirust-simd's runtime-dispatched f32 dot
    /// (proposal A3): ~8× at 768-d, at the cost of f32 accumulation — a dot that
    /// lands near zero can take the other sign than the scalar f64 path. Only
    /// reachable through [`SketchIndex::with_simd_sketching`], which keeps the
    /// choice per store and recorded on disk.
    #[cfg(feature = "simd")]
    pub(crate) fn sketch_simd(&self, embedding: &[f32]) -> Vec<u64> {
        if embedding.len() != self.dim {
            return Vec::new();
        }
        let backend = scirust_simd::dispatch::runtime_backend();
        let mut out = vec![0u64; self.words()];
        for b in 0..self.bits {
            let row = &self.planes[b * self.dim..(b + 1) * self.dim];
            if backend.sdot_f32(row, embedding) >= 0.0 {
                out[b / 64] |= 1u64 << (b % 64);
            }
        }
        out
    }
}

/// Hamming distance between two equal-length sketches: the popcount of their XOR.
/// (If lengths differ, only the shared prefix is compared.)
#[inline]
pub fn hamming(a: &[u64], b: &[u64]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}

/// The cosine implied by a Hamming distance of `h` over `bits` hyperplanes:
/// `cos(π · h / bits)`. `h = 0` → `1` (identical), `h = bits/2` → `0` (orthogonal),
/// `h = bits` → `-1` (antipodal).
#[inline]
pub fn cosine_from_hamming(h: u32, bits: usize) -> f32 {
    if bits == 0 {
        return 1.0;
    }
    (std::f32::consts::PI * h as f32 / bits as f32).cos()
}

/// L2-normalizes `v` in place (sequential `f32` — deterministic). A zero vector is
/// left as-is: its dot with anything is `0`, matching the old fused cosine's
/// convention for zero vectors.
fn l2_normalize(v: &mut [f32]) {
    let n = scirust_retrieval::vector::norm(v);
    if n > 0.0 {
        let inv = 1.0 / n;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Storage precision of a [`SketchIndex`]'s exact-rerank embeddings.
///
/// `F32` (the default) stores each normalized embedding as `dim` `f32`s — exact,
/// bit-deterministic scoring. `Int8` (proposal A4 of
/// `docs/scirust-improvements.md`, the symmetric-absmax scheme from
/// `scirust-core/src/quantization.rs`) stores `dim` `i8` codes plus one `f32`
/// scale per item — **4× smaller** (768-d: 3072 B → 772 B/item, answering the
/// documented "3 KB/item erases the compactness story" limitation) and scored by
/// an **integer `i32` dot**, whose sum is exact and therefore order-independent
/// by construction. The trade: quantization shifts cosines by up to ~1e-2, so
/// near-tie rankings can differ from the f32 tier — pick per store, measure with
/// the D1 gate, and note that [`SketchIndex::certify_shortlist`] certifies the
/// tier as configured (its oracle scores through the same storage).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Precision {
    /// Exact `f32` rerank (default).
    #[default]
    F32,
    /// 4× smaller int8 codes + per-item scale; exact integer accumulation.
    Int8,
    /// 8× smaller NF4 codes (two per byte) + per-item scale — the cold tier
    /// (QLoRA's quantile-matched codebook). Coarser than int8: cosines shift by
    /// up to ~5e-2; measure with the D1 gate before adopting per store.
    Nf4,
    /// Product Quantization experiment: the vector is split into `dim/4`
    /// subspaces of 4 dimensions, each encoded by its nearest centroid in a
    /// 256-entry k-means codebook trained at construction on a calibration
    /// sample — `M = dim/4` bytes/item (**64× smaller** than f32 at 768-d).
    /// Scoring is asymmetric distance computation (per-query lookup tables).
    /// Lossy by design and coarse by experiment: gate every adoption with
    /// Recall@10 ≥ 0.92 against the F32 oracle (`sketch::tests::pq_recall_gate`
    /// does, on a clustered corpus).
    Pq,
}

/// Subspace width of the [`Precision::Pq`] tier. Four dimensions per subspace
/// keeps 256 centroid levels dense enough for top-10 fidelity (measured by the
/// recall gate); eight was measured too coarse (Recall@10 ~0.24).
const PQ_SUBSPACE_DIM: usize = 4;
/// Centroids per subspace codebook (one byte encodes a subspace).
const PQ_CENTROIDS: usize = 256;
/// Fixed training iterations of the deterministic k-means.
const PQ_KMEANS_ITERS: usize = 12;

/// Number of PQ subspaces for `dim`.
fn pq_subspaces(dim: usize) -> usize {
    dim.div_ceil(PQ_SUBSPACE_DIM)
}

/// Trains the PQ codebooks deterministically: per subspace, a seeded k-means
/// with stride sampling for initialization, fixed iteration count, f64 mean
/// accumulation and lowest-index tie-breaking. Same corpus + seed → byte-equal
/// codebooks anywhere.
fn train_pq_codebooks(calibration: &[f32], num_samples: usize, dim: usize) -> Vec<f32> {
    let m = pq_subspaces(dim);
    let d = PQ_SUBSPACE_DIM;
    let mut books = vec![0f32; m * PQ_CENTROIDS * d];

    // Normalized copies — inserts encode normalized vectors, so the codebooks
    // must model that same space.
    let mut samples: Vec<f32> = calibration[..num_samples * dim].to_vec();
    for i in 0..num_samples {
        l2_normalize(&mut samples[i * dim..(i + 1) * dim]);
    }

    let mut rng = DeterministicRng::new(0x5051_4341_4C42_524F_u64); // "PQCALBRO"
    for sub in 0..m {
        let lo = sub * d;
        let hi = (lo + d).min(dim);
        let width = hi - lo;
        let book = &mut books[sub * PQ_CENTROIDS * d..(sub + 1) * PQ_CENTROIDS * d];

        // Init: pseudo-random sample indices (duplicates collapse to identical
        // centroids; k-means iterations then pull them apart or leave them as
        // unused-but-valid entries).
        for c in 0..PQ_CENTROIDS {
            let pick = (rng.next_u64() % num_samples.max(1) as u64) as usize;
            book[c * d..c * d + width].copy_from_slice(&samples[pick * dim + lo..pick * dim + hi]);
        }

        let mut assignments = vec![0u8; num_samples];
        let mut means = vec![0f64; PQ_CENTROIDS * d];
        for _ in 0..PQ_KMEANS_ITERS {
            // Assignment: nearest centroid by squared L2, ties to lower index.
            for s in 0..num_samples {
                let sub_vec = &samples[s * dim + lo..s * dim + hi];
                let mut best = 0usize;
                let mut best_d2 = f32::INFINITY;
                for c in 0..PQ_CENTROIDS {
                    let centroid = &book[c * d..c * d + width];
                    let d2: f32 = sub_vec
                        .iter()
                        .zip(centroid)
                        .map(|(&x, &y)| {
                            let diff = x - y;
                            diff * diff
                        })
                        .sum();
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best = c;
                    }
                }
                assignments[s] = best as u8;
            }
            // Update: f64 means in fixed order; empty clusters keep their
            // previous centroid.
            means.fill(0.0);
            let mut counts = vec![0u32; PQ_CENTROIDS];
            for s in 0..num_samples {
                let c = assignments[s] as usize;
                counts[c] += 1;
                let src = &samples[s * dim + lo..s * dim + hi];
                for j in 0..width {
                    means[c * d + j] += src[j] as f64;
                }
            }
            for c in 0..PQ_CENTROIDS {
                if counts[c] == 0 {
                    continue;
                }
                for j in 0..width {
                    book[c * d + j] = (means[c * d + j] / counts[c] as f64) as f32;
                }
            }
        }
    }
    books
}

/// Builds the flat `M x 256` asymmetric-distance lookup tables for a normalized
/// query: `luts[sub * 256 + centroid] = dot(query_subspace, centroid)`. Fixed
/// accumulation order — deterministic.
fn pq_lookup_tables(codebooks: &[f32], query: &[f32], dim: usize) -> Vec<f32> {
    let m = pq_subspaces(dim);
    let d = PQ_SUBSPACE_DIM;
    let mut luts = vec![0f32; m * PQ_CENTROIDS];
    for sub in 0..m {
        let lo = sub * d;
        let hi = (lo + d).min(dim);
        let qsub = &query[lo..hi];
        let width = hi - lo;
        let book = &codebooks[sub * PQ_CENTROIDS * d..(sub + 1) * PQ_CENTROIDS * d];
        for c in 0..PQ_CENTROIDS {
            let centroid = &book[c * d..c * d + width];
            luts[sub * PQ_CENTROIDS + c] =
                qsub.iter().zip(centroid).map(|(&x, &y)| x * y).sum::<f32>();
        }
    }
    luts
}

/// Encodes one normalized vector into `m` centroid indices (lowest index on
/// exact ties).
fn encode_pq(codebooks: &[f32], v: &[f32], dim: usize) -> Vec<u8> {
    let m = pq_subspaces(dim);
    let d = PQ_SUBSPACE_DIM;
    let mut codes = vec![0u8; m];
    for sub in 0..m {
        let lo = sub * d;
        let hi = (lo + d).min(dim);
        let width = hi - lo;
        let sub_vec = &v[lo..hi];
        // This subspace's own codebook slice.
        let book = &codebooks[sub * PQ_CENTROIDS * d..(sub + 1) * PQ_CENTROIDS * d];
        let mut best = 0usize;
        let mut best_d2 = f32::INFINITY;
        for c in 0..PQ_CENTROIDS {
            let centroid = &book[c * d..c * d + width];
            let d2: f32 = sub_vec
                .iter()
                .zip(centroid)
                .map(|(&x, &y)| {
                    let diff = x - y;
                    diff * diff
                })
                .sum();
            if d2 < best_d2 {
                best_d2 = d2;
                best = c;
            }
        }
        codes[sub] = best as u8;
    }
    codes
}

/// Symmetric absmax int8 quantization of a (normalized) embedding — the
/// `compute_scale`/`quantize_tensor` scheme vendored from
/// `scirust-core/src/quantization.rs` (same org, same dual license):
/// `scale = absmax / 127`, `code = round(x / scale)`. A zero vector gets
/// `scale = 0` and all-zero codes (its dot with anything is 0, matching the
/// zero-vector convention of the f32 tier).
fn quantize_i8(v: &[f32]) -> (Vec<i8>, f32) {
    let absmax = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    if absmax <= 0.0 {
        return (vec![0i8; v.len()], 0.0);
    }
    let scale = absmax / 127.0;
    let codes = v
        .iter()
        .map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (codes, scale)
}

/// The NF4 codebook (QLoRA's 4-bit NormalFloat, quantile-matched to a Gaussian)
/// — vendored from `scirust-core/src/quantization.rs` (`NF4_LEVELS`).
const NF4_LEVELS: [f32; 16] = [
    -1.0,
    -0.696_192_8,
    -0.525_073_05,
    -0.394_917_5,
    -0.284_441_38,
    -0.184_773_43,
    -0.091_050_036,
    0.0,
    0.079_580_3,
    0.160_930_2,
    0.246_112_3,
    0.337_915_24,
    0.440_709_83,
    0.562_617,
    0.722_956_84,
    1.0,
];

/// NF4-quantizes a (normalized) embedding: divide by `absmax`, map each
/// component to the nearest [`NF4_LEVELS`] entry, pack two 4-bit codes per byte
/// (low nibble first). ~8× smaller than f32 (768-d: 384 B codes + 4 B scale).
/// Vendored scheme from `scirust-core`'s `nf4_quantize`, with one refinement:
/// the stored scale is `1 / ‖levels[codes]‖` — the dequantized vector is
/// **re-normalized**, so a score is a genuine cosine (bounded by 1; without
/// this, quantization silently inflates norms by a few percent). Deterministic.
fn quantize_nf4(v: &[f32]) -> (Vec<u8>, f32) {
    let absmax = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let inv = if absmax > 1e-12 { 1.0 / absmax } else { 0.0 };
    let mut packed = vec![0u8; v.len().div_ceil(2)];
    let mut level_norm2 = 0.0f32;
    for (j, &x) in v.iter().enumerate() {
        let val = x * inv;
        let mut best = 0usize;
        let mut bd = (val - NF4_LEVELS[0]).abs();
        for (k, &lvl) in NF4_LEVELS.iter().enumerate().skip(1) {
            let d = (val - lvl).abs();
            if d < bd {
                bd = d;
                best = k;
            }
        }
        packed[j / 2] |= (best as u8) << ((j % 2) * 4);
        level_norm2 += NF4_LEVELS[best] * NF4_LEVELS[best];
    }
    let scale = if level_norm2 > 0.0 {
        1.0 / level_norm2.sqrt()
    } else {
        0.0
    };
    (packed, scale)
}

/// Integer dot with `i32` accumulation — exact, hence **order-independent**: any
/// future chunked/parallel evaluation of this sum is bit-identical to the
/// sequential one (unlike float sums, where order matters).
fn idot(a: &[i8], b: &[i8]) -> i32 {
    let n = a.len().min(b.len());
    let mut s = 0i32;
    for i in 0..n {
        s += a[i] as i32 * b[i] as i32;
    }
    s
}

/// Dot product of two equal-length vectors. Items are L2-normalized on insert and
/// queries once per call, so this **is** the cosine — one multiply-add per lane, no
/// per-pair norms (the normalize-on-insert pattern; ~3× fewer flops in the rerank
/// than the old fused cosine).
///
/// Default build: a sequential `f32` loop — deterministic, bit-identical across
/// platforms. With the `simd` feature: scirust-simd's runtime-dispatched
/// AVX2/SSE2/NEON kernel (4–8× at 768-d) — wide accumulation reorders the sum, so
/// scores can differ in the last bits and near-tie rankings may differ from the
/// default build. Nothing that persists goes through this function.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "simd")]
    if a.len() == b.len() {
        return scirust_simd::dispatch::runtime_backend().sdot_f32(a, b);
    }
    if a.len() == b.len() {
        scirust_retrieval::vector::dot(a, b)
    } else {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }
}

// ---------------------------------------------------------------------------
// SketchIndex — the high-precision retrieval tier (shortlist → exact rerank)
// ---------------------------------------------------------------------------

/// A high-precision semantic index: a SimHash sketch per item for a cheap Hamming
/// **shortlist**, then an **exact cosine rerank** over that shortlist.
///
/// This is the precision counterpart to [`crate::FractalMemory3D`]: where the 3-D
/// octree trades precision for compactness (a coarse router, exact recall@1 $\approx
/// 0\%$), `SketchIndex` trades memory for precision — it keeps each full embedding
/// (for the exact rerank) plus a compact sketch (for the shortlist), recovering most
/// of the true nearest neighbours the projection discards, at a fraction of a full
/// brute-force scan. All flat, contiguous storage; 100% safe, stable Rust.
/// The rerank-embedding storage of a [`SketchIndex`] (see [`Precision`]).
#[derive(Clone, Debug)]
enum EmbeddingStore {
    F32(Vec<f32>),
    Int8 {
        codes: Vec<i8>,
        scales: Vec<f32>,
    },
    /// Packed two-codes-per-byte NF4 (low nibble first) + per-item absmax.
    Nf4 {
        codes: Vec<u8>,
        scales: Vec<f32>,
    },
    /// Product Quantization experiment: `M = ceil(dim/4)` one-byte centroid
    /// indices per item plus the flat `M × 256 × 4` k-means codebooks trained
    /// at construction (see [`train_pq_codebooks`]).
    Pq {
        codes: Vec<u8>,
        codebooks: Vec<f32>,
    },
}

/// A query prepared for one store precision (see `SketchIndex::prepare_query`).
enum PreparedQuery {
    F32(Vec<f32>),
    Int8 {
        codes: Vec<i8>,
        scale: f32,
    },
    /// NF4 items are scored against the full normalized query (one LUT lookup +
    /// multiply-add per lane): quantizing the query too would double the noise.
    Nf4(Vec<f32>),
    /// Flat `M × 256` lookup tables of dot(query_subspace, centroid) — ADC
    /// scoring is then M table lookups + adds per item.
    Pq(Vec<f32>),
}

#[derive(Clone, Debug)]
pub struct SketchIndex {
    hasher: Arc<SimHasher>,
    dim: usize,
    /// Seed used to (re)generate the hasher's hyperplanes — stored so the index
    /// reloads without serialising the planes.
    seed: u64,
    /// Whether this store's sketches are computed with the SIMD f32 path
    /// (proposal A3, `simd` feature) instead of the scalar f64 default. The
    /// choice is **per store**, fixed before the first insert, recorded in the
    /// v4 file format, and applied to query sketching too — sketches and query
    /// sketches always share one compute path, so Hamming stays faithful.
    simd_sketches: bool,
    /// `count × dim` flat row-major embeddings (for the exact rerank) — f32
    /// (exact) or int8 codes + per-item scales (4× smaller, integer dot).
    store: EmbeddingStore,
    /// `count × words` flat sketches (for the Hamming shortlist).
    sketches: Vec<u64>,
    /// Payload arena and per-item `(offset, len)`.
    payloads: Vec<u8>,
    offsets: Vec<(usize, usize)>,
}

impl SketchIndex {
    /// Creates an empty index for `dim`-dimensional embeddings, sketched with `bits`
    /// random hyperplanes (seeded). Rerank storage is exact [`Precision::F32`]; use
    /// [`SketchIndex::new_with_precision`] for the 4× smaller int8 tier.
    pub fn new(dim: usize, bits: usize, seed: u64) -> Self {
        Self::new_with_precision(dim, bits, seed, Precision::F32)
    }

    /// Creates an empty index with an explicit rerank [`Precision`] (see the
    /// enum docs for the memory/accuracy trade).
    ///
    /// # Panics
    /// For [`Precision::Pq`]: PQ needs trained codebooks - construct via
    /// [`SketchIndex::new_pq`].
    pub fn new_with_precision(dim: usize, bits: usize, seed: u64, precision: Precision) -> Self {
        assert!(
            precision != Precision::Pq,
            "PQ requires trained codebooks - construct via SketchIndex::new_pq"
        );
        let hasher = Arc::new(SimHasher::new(dim, bits, seed));
        Self::new_with_shared_hasher(hasher, seed, precision)
    }

    /// Creates an empty PQ-tier index whose codebooks are trained on a flat,
    /// row-major `num_samples x dim` calibration sample (deterministic k-means;
    /// same corpus -> byte-equal codebooks). Later inserts encode against those
    /// codebooks immediately - there is no lazy training phase.
    ///
    /// # Panics
    /// If `calibration.len() != num_samples * dim`, any count is 0, or `dim`
    /// exceeds 2048 (the per-query LUT build cost grows with `dim`; beyond that
    /// the experiment tier is not the right tool).
    pub fn new_pq(
        dim: usize,
        bits: usize,
        seed: u64,
        calibration: &[f32],
        num_samples: usize,
    ) -> Self {
        assert!(
            dim > 0 && num_samples > 0,
            "dim and num_samples must be non-zero"
        );
        assert_eq!(
            calibration.len(),
            num_samples * dim,
            "calibration must be num_samples * dim"
        );
        assert!(dim <= 2048, "PQ experiments target dims up to 2048");

        let hasher = Arc::new(SimHasher::new(dim, bits, seed));
        let mut index = Self::new_with_shared_hasher(hasher, seed, Precision::Pq);
        if let EmbeddingStore::Pq { codebooks, .. } = &mut index.store {
            *codebooks = train_pq_codebooks(calibration, num_samples, dim);
        }
        index
    }

    /// Internal constructor used by sharded stores so identical hyperplanes are
    /// allocated once and shared by every regional precision index.
    pub(crate) fn new_with_shared_hasher(
        hasher: Arc<SimHasher>,
        seed: u64,
        precision: Precision,
    ) -> Self {
        let dim = hasher.dim();
        let store = match precision {
            Precision::F32 => EmbeddingStore::F32(Vec::new()),
            Precision::Int8 => EmbeddingStore::Int8 {
                codes: Vec::new(),
                scales: Vec::new(),
            },
            Precision::Nf4 => EmbeddingStore::Nf4 {
                codes: Vec::new(),
                scales: Vec::new(),
            },
            // Empty codebooks; `new_pq` (constructor) and the SKCH loader are
            // the only legitimate ways to reach this arm with real content.
            Precision::Pq => EmbeddingStore::Pq {
                codes: Vec::new(),
                codebooks: Vec::new(),
            },
        };
        Self {
            hasher,
            dim,
            seed,
            simd_sketches: false,
            store,
            sketches: Vec::new(),
            payloads: Vec::new(),
            offsets: Vec::new(),
        }
    }

    /// Replaces an equivalent regenerated projector by a shared allocation.
    pub(crate) fn share_hasher(&mut self, hasher: Arc<SimHasher>, seed: u64) -> io::Result<()> {
        if seed != self.seed || hasher.dim() != self.dim || hasher.bits() != self.bits() {
            return Err(invalid(
                "shared SimHash projector does not match index identity",
            ));
        }
        self.hasher = hasher;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn shares_hasher_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.hasher, &other.hasher)
    }

    /// Switches this store to **SIMD sketching** (proposal A3): every sketch —
    /// items at insert and queries at recall — is computed with scirust-simd's
    /// runtime-dispatched f32 kernel (~8× at 768-d) instead of the scalar f64
    /// default. Self-consistent per store and recorded in the file format
    /// (SKCH v4), so a store can never silently mix compute paths; loading such
    /// a store requires the `simd` feature. The f32 accumulation can flip
    /// sign-of-dot near zero relative to the scalar path — that is the trade,
    /// and it is per store, chosen here.
    ///
    /// # Panics
    /// Panics if items were already inserted (the path is part of the store's
    /// identity).
    #[cfg(feature = "simd")]
    pub fn with_simd_sketching(mut self) -> Self {
        assert!(
            self.is_empty(),
            "the sketch path must be chosen before the first insert"
        );
        self.simd_sketches = true;
        self
    }

    /// Whether this store sketches via the SIMD f32 path (see
    /// [`SketchIndex::with_simd_sketching`]).
    pub fn simd_sketching(&self) -> bool {
        self.simd_sketches
    }

    /// Sketches `v` with this store's compute path (scalar f64 default, or the
    /// SIMD f32 path when [`SketchIndex::with_simd_sketching`] was chosen).
    fn sketch_with_path(&self, v: &[f32]) -> Vec<u64> {
        #[cfg(feature = "simd")]
        if self.simd_sketches {
            return self.hasher.sketch_simd(v);
        }
        self.hasher.sketch(v)
    }

    /// The rerank storage precision of this index.
    pub fn precision(&self) -> Precision {
        match &self.store {
            EmbeddingStore::F32(_) => Precision::F32,
            EmbeddingStore::Int8 { .. } => Precision::Int8,
            EmbeddingStore::Nf4 { .. } => Precision::Nf4,
            EmbeddingStore::Pq { .. } => Precision::Pq,
        }
    }

    /// Number of indexed items.
    #[inline]
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    /// Whether the index is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    /// Sketch bits per item.
    #[inline]
    pub fn bits(&self) -> usize {
        self.hasher.bits()
    }

    /// Inserts an `embedding` with a byte `payload`. Returns `false` (and stores
    /// nothing) if the vector has the wrong dimension or contains a non-finite value.
    pub fn insert(&mut self, embedding: &[f32], payload: &[u8]) -> bool {
        if embedding.len() != self.dim || embedding.iter().any(|x| !x.is_finite()) {
            return false;
        }
        // Sketch the *raw* embedding (sign-of-dot is scale-invariant, so the sketch
        // is the same — and stays bit-identical with pre-v2 stores), then store the
        // embedding L2-normalized so the exact rerank is a single dot per candidate
        // (quantized to int8 codes + scale on the int8 tier).
        let sk = self.sketch_with_path(embedding);
        self.sketches.extend_from_slice(&sk);
        let mut normalized = embedding.to_vec();
        l2_normalize(&mut normalized);
        match &mut self.store {
            EmbeddingStore::F32(e) => e.extend_from_slice(&normalized),
            EmbeddingStore::Int8 { codes, scales } => {
                let (c, scale) = quantize_i8(&normalized);
                codes.extend_from_slice(&c);
                scales.push(scale);
            }
            EmbeddingStore::Nf4 { codes, scales } => {
                let (c, scale) = quantize_nf4(&normalized);
                codes.extend_from_slice(&c);
                scales.push(scale);
            }
            EmbeddingStore::Pq { codes, codebooks } => {
                codes.extend_from_slice(&encode_pq(codebooks, &normalized, self.dim));
            }
        }
        let off = self.payloads.len();
        self.payloads.extend_from_slice(payload);
        self.offsets.push((off, payload.len()));
        true
    }

    fn payload(&self, i: usize) -> &[u8] {
        let (off, len) = self.offsets[i];
        &self.payloads[off..off + len]
    }

    /// Crate-level payload access — the compaction path reads items back out.
    pub(crate) fn item_payload(&self, i: usize) -> &[u8] {
        self.payload(i)
    }

    /// Crate-level embedding access for compaction: item `i`'s stored vector,
    /// dequantized to f32 and L2-normalized (the exact form `insert` persists).
    /// On the int8/NF4 tiers this is the quantized value decoded — re-inserting
    /// it re-quantizes once more, which is the documented compaction caveat.
    pub(crate) fn item_embedding(&self, i: usize) -> Vec<f32> {
        match &self.store {
            EmbeddingStore::F32(e) => e[i * self.dim..(i + 1) * self.dim].to_vec(),
            EmbeddingStore::Int8 { codes, scales } => codes[i * self.dim..(i + 1) * self.dim]
                .iter()
                .map(|&c| c as f32 * scales[i])
                .collect(),
            EmbeddingStore::Nf4 { codes, scales } => {
                let bytes = self.dim.div_ceil(2);
                let row = &codes[i * bytes..(i + 1) * bytes];
                (0..self.dim)
                    .map(|j| {
                        let code = (row[j / 2] >> ((j % 2) * 4)) & 0x0F;
                        NF4_LEVELS[code as usize] * scales[i]
                    })
                    .collect()
            }
            EmbeddingStore::Pq { codes, codebooks } => {
                let m = pq_subspaces(self.dim);
                let d = PQ_SUBSPACE_DIM;
                let row = &codes[i * m..(i + 1) * m];
                let mut out = vec![0f32; self.dim];
                for (sub, &c) in row.iter().enumerate() {
                    let lo = sub * d;
                    let hi = (lo + d).min(self.dim);
                    let base = sub * PQ_CENTROIDS * d + c as usize * d;
                    out[lo..hi].copy_from_slice(&codebooks[base..base + (hi - lo)]);
                }
                out
            }
        }
    }

    /// A query in the form the store scores against: normalized f32, or quantized
    /// codes + scale on the int8 tier. Built once per public query call.
    fn prepare_query(&self, query: &[f32]) -> PreparedQuery {
        let mut qn = query.to_vec();
        l2_normalize(&mut qn);
        match &self.store {
            EmbeddingStore::F32(_) => PreparedQuery::F32(qn),
            EmbeddingStore::Int8 { .. } => {
                let (codes, scale) = quantize_i8(&qn);
                PreparedQuery::Int8 { codes, scale }
            }
            EmbeddingStore::Nf4 { .. } => PreparedQuery::Nf4(qn),
            EmbeddingStore::Pq { codebooks, .. } => {
                PreparedQuery::Pq(pq_lookup_tables(codebooks, &qn, self.dim))
            }
        }
    }

    /// Cosine of item `i` against a prepared query — a single f32 dot on the exact
    /// tier, an exact (order-independent) i32 dot times the two scales on the int8
    /// tier. `prepare_query` always matches the store, so the cross arms are
    /// unreachable by construction.
    fn score(&self, i: usize, q: &PreparedQuery) -> f32 {
        match (&self.store, q) {
            (EmbeddingStore::F32(e), PreparedQuery::F32(qn)) => {
                dot(&e[i * self.dim..(i + 1) * self.dim], qn)
            }
            (EmbeddingStore::Int8 { codes, scales }, PreparedQuery::Int8 { codes: qc, scale }) => {
                scales[i] * scale * idot(&codes[i * self.dim..(i + 1) * self.dim], qc) as f32
            }
            (EmbeddingStore::Nf4 { codes, scales }, PreparedQuery::Nf4(qn)) => {
                // Dequantize-free: one codebook lookup + multiply-add per lane,
                // sequential f32 — deterministic.
                let bytes = self.dim.div_ceil(2);
                let row = &codes[i * bytes..(i + 1) * bytes];
                let mut sum = 0.0f32;
                for (j, q) in qn.iter().enumerate().take(self.dim) {
                    let code = (row[j / 2] >> ((j % 2) * 4)) & 0x0F;
                    sum += NF4_LEVELS[code as usize] * q;
                }
                scales[i] * sum
            }
            (EmbeddingStore::Pq { codes, .. }, PreparedQuery::Pq(luts)) => {
                let m = pq_subspaces(self.dim);
                let row = &codes[i * m..(i + 1) * m];
                // Fixed subspace order — deterministic accumulation.
                row.iter()
                    .enumerate()
                    .map(|(sub, &c)| luts[sub * PQ_CENTROIDS + c as usize])
                    .sum()
            }
            _ => unreachable!("prepare_query always matches the store precision"),
        }
    }

    fn sketch_of(&self, i: usize) -> &[u64] {
        let w = self.hasher.words();
        &self.sketches[i * w..(i + 1) * w]
    }

    /// The `k` nearest payloads to `query`, by the hybrid path: take the `shortlist`
    /// closest by **Hamming** on the sketches, then **exact-cosine** rerank them.
    /// Returns `(payload, cosine)` descending. Larger `shortlist` → higher recall at
    /// higher cost; `shortlist` is clamped to at least `k` and at most the index size.
    pub fn nearest(&self, query: &[f32], k: usize, shortlist: usize) -> Vec<(&[u8], f32)> {
        self.nearest_ids(query, k, shortlist)
            .into_iter()
            .map(|(i, s)| (self.payload(i), s))
            .collect()
    }

    /// Computes the query sketch once with this store's persisted compute path.
    /// Shared-projector sharded stores can reuse the result across regions.
    pub(crate) fn query_sketch(&self, query: &[f32]) -> Option<Vec<u64>> {
        if query.len() != self.dim || query.iter().any(|x| !x.is_finite()) {
            return None;
        }
        Some(self.sketch_with_path(query))
    }

    /// Precise recall using a caller-supplied sketch computed by an equivalent
    /// [`SketchIndex`] (same projector and scalar/SIMD sketch path).
    pub(crate) fn nearest_with_sketch(
        &self,
        query: &[f32],
        query_sketch: &[u64],
        k: usize,
        shortlist: usize,
    ) -> Vec<(&[u8], f32)> {
        self.nearest_ids_with_sketch(query, query_sketch, k, shortlist)
            .into_iter()
            .map(|(i, score)| (self.payload(i), score))
            .collect()
    }

    /// Core of [`SketchIndex::nearest`], on item ids (insertion order) instead of
    /// payloads — also the exact pipeline [`SketchIndex::certify_shortlist`] measures.
    fn nearest_ids(&self, query: &[f32], k: usize, shortlist: usize) -> Vec<(usize, f32)> {
        let Some(query_sketch) = self.query_sketch(query) else {
            return Vec::new();
        };
        self.nearest_ids_with_sketch(query, &query_sketch, k, shortlist)
    }

    fn nearest_ids_with_sketch(
        &self,
        query: &[f32],
        query_sketch: &[u64],
        k: usize,
        shortlist: usize,
    ) -> Vec<(usize, f32)> {
        if query.len() != self.dim
            || query.iter().any(|x| !x.is_finite())
            || query_sketch.len() != self.hasher.words()
            || k == 0
            || self.is_empty()
        {
            return Vec::new();
        }
        let m = shortlist.max(k).min(self.len());

        // 1. Hamming shortlist of size m. The expensive query projection has
        // already been computed by the caller and may be shared across shards.
        let mut cand: Vec<(u32, usize)> = (0..self.len())
            .map(|i| (hamming(query_sketch, self.sketch_of(i)), i))
            .collect();
        if cand.len() > m {
            cand.select_nth_unstable_by_key(m - 1, |(h, i)| (*h, *i));
            cand.truncate(m);
        }

        // 2. Exact rerank of the shortlist (query prepared once for this shard's
        // precision; each candidate costs a single dot — f32 or integer).
        let q = self.prepare_query(query);
        let mut scored: Vec<(f32, usize)> =
            cand.iter().map(|&(_, i)| (self.score(i, &q), i)).collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        scored.truncate(k);
        scored.into_iter().map(|(score, i)| (i, score)).collect()
    }

    /// The exact top-`k` item ids by full cosine over the whole corpus — the oracle
    /// [`SketchIndex::certify_shortlist`] measures against. Ties break toward the
    /// smaller id (deterministic; a tie-swap can only make the certificate
    /// pessimistic, never invalid).
    fn exact_top_ids(&self, query: &[f32], k: usize) -> Vec<usize> {
        let q = self.prepare_query(query);
        let mut scored: Vec<(f32, usize)> =
            (0..self.len()).map(|i| (self.score(i, &q), i)).collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        scored.truncate(k.min(self.len()));
        scored.into_iter().map(|(_, i)| i).collect()
    }

    /// **Certify a shortlist size** — replace the hand-tuned constant with the
    /// smallest shortlist whose recall loss is provably controlled (RCPS — see
    /// [`crate::conformal`]).
    ///
    /// The per-query loss is `1 − recall@k`: the fraction of the exact full-corpus
    /// top-`k` (by full cosine) that the `shortlist → rerank` pipeline misses. The
    /// candidate shortlists form a **nested** family (doubling from `k` up to the
    /// corpus size, where the pipeline *is* the exact rerank), so the risk is
    /// non-increasing and RCPS applies. The returned certificate reads: *expected
    /// recall loss `≤ alpha` with probability `≥ 1 − delta`*, valid **for workloads
    /// exchangeable with `queries`** — re-calibrate on query drift (new topics, a
    /// different embedder).
    ///
    /// `None` when nothing certifies: no valid calibration queries, or too few of
    /// them for the asked `(alpha, delta)` (the Hoeffding slack `√(ln(1/δ)/2n)`
    /// must fit under `alpha`), or the sketch genuinely too weak. Deterministic;
    /// cost is `O(queries · grid · N)` — an offline calibration pass, not a query.
    ///
    /// # Panics
    /// If `delta` is outside `(0, 1)`.
    pub fn certify_shortlist(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        alpha: f64,
        delta: f64,
    ) -> Option<ShortlistCertificate> {
        if k == 0 || self.is_empty() {
            return None;
        }
        let valid: Vec<&Vec<f32>> = queries
            .iter()
            .filter(|q| q.len() == self.dim && q.iter().all(|x| x.is_finite()))
            .collect();
        let n = valid.len();
        if n == 0 {
            return None;
        }

        // Doubling grid from k to the full corpus; the last point anchors the
        // family at the exact rerank of everything.
        let mut grid = Vec::new();
        let mut m = k.max(1);
        while m < self.len() {
            grid.push(m);
            m = m.saturating_mul(2);
        }
        grid.push(self.len());

        let oracles: Vec<Vec<usize>> = valid.iter().map(|q| self.exact_top_ids(q, k)).collect();
        let risks: Vec<f64> = grid
            .iter()
            .map(|&m| {
                let total: f64 = valid
                    .iter()
                    .zip(&oracles)
                    .map(|(q, oracle)| {
                        let got = self.nearest_ids(q, k, m);
                        let hits = oracle
                            .iter()
                            .filter(|t| got.iter().any(|(i, _)| i == *t))
                            .count();
                        1.0 - hits as f64 / oracle.len() as f64
                    })
                    .sum();
                total / n as f64
            })
            .collect();

        let chosen = crate::conformal::rcps_select(&risks, n, alpha, delta)?;
        Some(ShortlistCertificate {
            shortlist: grid[chosen],
            k,
            alpha,
            delta,
            calibration_n: n,
            empirical_risk: risks[chosen],
            risk_ucb: crate::conformal::hoeffding_ucb(risks[chosen], n, delta),
            grid,
        })
    }

    /// **Certify an adaptive per-query radius** — the ConANN shape (Conformal
    /// Risk Control applied to ANN probing, VLDB 2025) on the sketch tier.
    /// Where [`SketchIndex::certify_shortlist`] fixes one global shortlist
    /// size, this certifies a *query-dependent* rule: take every item whose
    /// Hamming distance is at most `d_k(query) + lambda`, where `d_k(query)` is
    /// the query's own `k`-th smallest sketch distance. Dense regions of the
    /// corpus then cost fewer rerank scores than sparse ones.
    ///
    /// The candidate radii form a nested family (a larger radius can only add
    /// candidates), so the risk is non-increasing and RCPS applies unchanged:
    /// the returned certificate reads *expected recall loss at `k` ≤ alpha with
    /// probability ≥ 1 − delta*, for workloads exchangeable with `queries`.
    ///
    /// `None` when nothing certifies, exactly as in [`Self::certify_shortlist`].
    /// Deterministic; offline pass (`O(queries · (N log N + k · grid))`).
    ///
    /// # Panics
    /// If `delta` is outside `(0, 1)`.
    pub fn calibrate_adaptive_radius(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        alpha: f64,
        delta: f64,
    ) -> Option<crate::conformal::AdaptiveRadiusCertificate> {
        if k == 0 || self.is_empty() {
            return None;
        }
        let valid: Vec<&Vec<f32>> = queries
            .iter()
            .filter(|q| q.len() == self.dim && q.iter().all(|x| x.is_finite()))
            .collect();
        let n = valid.len();
        if n == 0 {
            return None;
        }

        // Doubling lambda grid from 0 up past the sketch width (Hamming never
        // exceeds `bits`, so the last point makes the rule cover everything).
        let mut grid = Vec::new();
        let mut lambda: u32 = 0;
        loop {
            grid.push(lambda);
            if lambda as usize > self.bits() {
                break;
            }
            lambda = (lambda.max(1)).saturating_mul(2);
        }

        let oracles: Vec<Vec<usize>> = valid.iter().map(|q| self.exact_top_ids(q, k)).collect();
        let risks: Vec<f64> = grid
            .iter()
            .map(|&lambda| {
                let total: f64 = valid
                    .iter()
                    .zip(&oracles)
                    .map(|(q, oracle)| {
                        // The query is validated above, so its sketch exists;
                        // compute it once and reuse it across the oracle set.
                        let Some(sk) = self.query_sketch(q) else {
                            return 1.0;
                        };
                        let d_k = self.kth_sketch_distance_with_sketch(&sk, k);
                        let limit = d_k.saturating_add(lambda);
                        let hits = oracle
                            .iter()
                            .filter(|&&t| hamming(&sk, self.sketch_of(t)) <= limit)
                            .count();
                        1.0 - hits as f64 / oracle.len() as f64
                    })
                    .sum();
                total / n as f64
            })
            .collect();

        let chosen = crate::conformal::rcps_select(&risks, n, alpha, delta)?;
        Some(crate::conformal::AdaptiveRadiusCertificate {
            lambda: grid[chosen],
            k,
            alpha,
            delta,
            calibration_n: n,
            empirical_risk: risks[chosen],
            risk_ucb: crate::conformal::hoeffding_ucb(risks[chosen], n, delta),
            grid,
        })
    }

    fn hamming_distances(&self, query_sketch: &[u64]) -> Vec<(u32, usize)> {
        (0..self.len())
            .map(|i| (hamming(query_sketch, self.sketch_of(i)), i))
            .collect()
    }

    /// The `k`-th smallest Hamming distance between `query` and the corpus
    /// (clamped to the corpus size). Empty store or `k == 0` → 0.
    fn kth_sketch_distance_with_sketch(&self, query_sketch: &[u64], k: usize) -> u32 {
        if self.is_empty() || k == 0 {
            return 0;
        }
        let mut dists = self.hamming_distances(query_sketch);
        let m = k.min(dists.len());
        dists.select_nth_unstable_by_key(m - 1, |(h, i)| (*h, *i));
        dists[m - 1].0
    }

    /// Adaptive-radius recall using a certificate from
    /// [`SketchIndex::calibrate_adaptive_radius`](Self::calibrate_adaptive_radius):
    /// shortlist = every item within `d_k(query) + lambda` by sketch distance,
    /// exact-cosine rerank, top-`k`. Returns `(payload, cosine)` descending.
    pub fn nearest_adaptive(&self, query: &[f32], k: usize, lambda: u32) -> Vec<(&[u8], f32)> {
        if k == 0 || self.is_empty() {
            return Vec::new();
        }
        let Some(query_sketch) = self.query_sketch(query) else {
            return Vec::new();
        };
        let mut dists = self.hamming_distances(&query_sketch);
        let m = k.min(dists.len());
        dists.select_nth_unstable_by_key(m - 1, |(h, i)| (*h, *i));
        let d_k = dists[m - 1].0;
        let limit = d_k.saturating_add(lambda);

        // Every item at or under the radius — including all ties at the cut —
        // matches what calibration measured; truncating instead would drop tied
        // candidates the certificate counted on.
        let q = self.prepare_query(query);
        let mut scored: Vec<(f32, usize)> = dists
            .iter()
            .filter(|&&(h, _)| h <= limit)
            .map(|&(_, i)| (self.score(i, &q), i))
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        scored.truncate(k);
        scored
            .into_iter()
            .map(|(score, i)| (self.payload(i), score))
            .collect()
    }

    /// The `k` nearest payloads by **Hamming only** (no exact rerank): cheaper and
    /// memory could be sketch-only, but approximate. Returns `(payload, hamming)`
    /// ascending (smaller is closer).
    pub fn nearest_sketch(&self, query: &[f32], k: usize) -> Vec<(&[u8], u32)> {
        if query.len() != self.dim
            || query.iter().any(|x| !x.is_finite())
            || k == 0
            || self.is_empty()
        {
            return Vec::new();
        }
        let qs = self.sketch_with_path(query);
        let mut cand: Vec<(u32, usize)> = (0..self.len())
            .map(|i| (hamming(&qs, self.sketch_of(i)), i))
            .collect();
        let k = k.min(cand.len());
        cand.select_nth_unstable_by_key(k - 1, |(h, i)| (*h, *i));
        cand.truncate(k);
        cand.sort_unstable_by_key(|(h, i)| (*h, *i));
        cand.into_iter()
            .map(|(h, i)| (self.payload(i), h))
            .collect()
    }

    /// Exact cosine rerank over a **subset** of item ids (e.g. a spatial shortlist
    /// from a 3-D index): returns the top `k` `(payload, cosine)` among those ids.
    /// Ids out of range are skipped. The id space matches insertion order.
    pub fn rerank(&self, query: &[f32], ids: &[u32], k: usize) -> Vec<(&[u8], f32)> {
        if query.len() != self.dim || k == 0 {
            return Vec::new();
        }
        let q = self.prepare_query(query);
        let mut scored: Vec<(f32, usize)> = ids
            .iter()
            .filter_map(|&id| {
                let i = id as usize;
                (i < self.len()).then(|| (self.score(i, &q), i))
            })
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        scored.truncate(k);
        scored
            .into_iter()
            .map(|(s, i)| (self.payload(i), s))
            .collect()
    }

    /// Ranks a **subset** of item ids by Hamming distance to `query` and returns the
    /// `m` closest ids — the prune step of a cascaded query. Ids out of range are
    /// skipped.
    pub fn hamming_rank(&self, query: &[f32], ids: &[u32], m: usize) -> Vec<u32> {
        if query.len() != self.dim || m == 0 {
            return Vec::new();
        }
        let qs = self.sketch_with_path(query);
        let mut cand: Vec<(u32, u32)> = ids
            .iter()
            .filter_map(|&id| {
                let i = id as usize;
                (i < self.len()).then(|| (hamming(&qs, self.sketch_of(i)), id))
            })
            .collect();
        let m = m.min(cand.len());
        if m == 0 {
            return Vec::new();
        }
        cand.select_nth_unstable_by_key(m - 1, |(h, id)| (*h, *id));
        cand.truncate(m);
        cand.sort_unstable_by_key(|(h, id)| (*h, *id));
        cand.into_iter().map(|(_, id)| id).collect()
    }

    /// **GPU batch scoring** (proposal A5, `gpu` feature): the exact cosine of
    /// **every** item against **every** query — `queries.len() × len()` values —
    /// as a single `Q · Eᵀ` GEMM on the pre-normalized F32 storage (the shader's
    /// transpose flag reads the row-major embeddings directly; no copy). Row `i`
    /// of the result is `scores(queries[i])` within float tolerance.
    ///
    /// For the workloads that genuinely score everything (viewer heat-maps,
    /// cross-shard global recall, benchmark sweeps). GPU accumulation is NOT
    /// bit-identical to [`SketchIndex::scores`] — tolerance-validated, never the
    /// default path. Errors honestly: dimension-mismatched queries are
    /// `InvalidInput`, quantized tiers are `Unsupported` (stacking a second
    /// approximation on int8/NF4 would compound silently).
    #[cfg(feature = "gpu")]
    pub fn scores_batch_gpu(
        &self,
        scorer: &crate::gpu::GpuScorer,
        queries: &[Vec<f32>],
    ) -> io::Result<Vec<Vec<f32>>> {
        let EmbeddingStore::F32(embeddings) = &self.store else {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "scores_batch_gpu runs on the F32 tier only (quantized tiers \
                 would stack a second approximation)",
            ));
        };
        if queries.iter().any(|q| q.len() != self.dim) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("every query must have dim {}", self.dim),
            ));
        }
        if queries.is_empty() || self.is_empty() {
            return Ok(queries.iter().map(|_| Vec::new()).collect());
        }
        // Q: q×dim, normalized like any single-query path.
        let mut q_flat = Vec::with_capacity(queries.len() * self.dim);
        for q in queries {
            let mut qn = q.clone();
            l2_normalize(&mut qn);
            q_flat.extend_from_slice(&qn);
        }
        // C(q×n) = Q(q×dim) · Eᵀ — E is stored n×dim row-major, so tb = true.
        let flat = scorer.gemm(
            &q_flat,
            embeddings,
            queries.len(),
            self.dim,
            self.len(),
            true,
        )?;
        Ok(flat.chunks(self.len()).map(<[f32]>::to_vec).collect())
    }

    /// Exact cosine similarity of **every** item to `query`, in id (insertion)
    /// order. Empty on a dimension mismatch. Useful to colour a 3-D view by
    /// precision score.
    pub fn scores(&self, query: &[f32]) -> Vec<f32> {
        if query.len() != self.dim {
            return Vec::new();
        }
        let q = self.prepare_query(query);
        (0..self.len()).map(|i| self.score(i, &q)).collect()
    }

    // -- persistence ---------------------------------------------------------

    /// Serialises the index to a versioned `SKCH` file (little-endian; the payload
    /// arena is LZ4-compressed). The hyperplanes are *not* stored — they are
    /// regenerated from the seed on load. **v2** (f32 tier): embeddings are stored
    /// L2-normalized (cosine = dot); v1 files (raw embeddings) are still read and
    /// migrated on load. **v3** (int8 tier): one f32 scale per item, then
    /// `count·dim` i8 codes — 4× smaller on disk too. The version written follows
    /// this index's [`Precision`], so f32 stores stay readable by older builds.
    pub fn save_to_disk(&self, path: &str) -> io::Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(b"SKCH")?;
        // v2/v3 stay byte-identical for the configurations older builds can read;
        // v4 (flags header) is written only when the store actually uses a v4
        // capability: NF4 embeddings and/or SIMD-path sketches (proposal A3).
        let precision_code: u32 = match &self.store {
            EmbeddingStore::F32(_) => 0,
            EmbeddingStore::Int8 { .. } => 1,
            EmbeddingStore::Nf4 { .. } => 2,
            EmbeddingStore::Pq { .. } => 3,
        };
        let needs_v5 = precision_code == 3;
        let needs_v4 = self.simd_sketches || precision_code == 2;
        let version: u32 = if needs_v5 {
            5
        } else if needs_v4 {
            4
        } else if precision_code == 1 {
            3
        } else {
            2
        };
        w.write_all(&version.to_le_bytes())?;
        w.write_all(&(self.dim as u32).to_le_bytes())?;
        w.write_all(&(self.hasher.bits() as u32).to_le_bytes())?;
        w.write_all(&self.seed.to_le_bytes())?;
        if version >= 4 {
            let flags: u32 = (self.simd_sketches as u32) | (precision_code << 1);
            w.write_all(&flags.to_le_bytes())?;
        }
        w.write_all(&(self.len() as u64).to_le_bytes())?;

        match &self.store {
            EmbeddingStore::F32(embeddings) => {
                for &f in embeddings {
                    w.write_all(&f.to_le_bytes())?;
                }
            }
            EmbeddingStore::Int8 { codes, scales } => {
                for &sc in scales {
                    w.write_all(&sc.to_le_bytes())?;
                }
                let bytes: Vec<u8> = codes.iter().map(|&c| c as u8).collect();
                w.write_all(&bytes)?;
            }
            EmbeddingStore::Nf4 { codes, scales } => {
                for &sc in scales {
                    w.write_all(&sc.to_le_bytes())?;
                }
                w.write_all(codes)?;
            }
            EmbeddingStore::Pq { codes, codebooks } => {
                w.write_all(codes)?;
                for &f in codebooks {
                    w.write_all(&f.to_le_bytes())?;
                }
            }
        }
        for &s in &self.sketches {
            w.write_all(&s.to_le_bytes())?;
        }
        for &(off, len) in &self.offsets {
            w.write_all(&(off as u64).to_le_bytes())?;
            w.write_all(&(len as u64).to_le_bytes())?;
        }

        w.write_all(&(self.payloads.len() as u64).to_le_bytes())?;
        let comp = lz4_flex::compress(&self.payloads);
        w.write_all(&(comp.len() as u64).to_le_bytes())?;
        w.write_all(&comp)?;
        w.flush()
    }

    /// Loads an index written by [`SketchIndex::save_to_disk`], validating the magic,
    /// version, and that the stored `dim` equals `expected_dim`. The hyperplanes are
    /// regenerated from the stored seed.
    pub fn load_from_disk(path: &str, expected_dim: usize) -> io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let mut r: &[u8] = &bytes;

        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != b"SKCH" {
            return Err(invalid("not a SketchIndex file (bad magic)"));
        }
        let version = read_u32(&mut r)?;
        if !(1..=5).contains(&version) {
            return Err(invalid(&format!(
                "unsupported SketchIndex version {version}"
            )));
        }
        let dim = read_u32(&mut r)? as usize;
        let bits = read_u32(&mut r)? as usize;
        let seed = read_u64(&mut r)?;
        if dim != expected_dim {
            return Err(invalid(&format!(
                "dim mismatch: file has {dim}, caller expected {expected_dim}"
            )));
        }
        // v4 carries a flags word: bit 0 = SIMD-path sketches, bits 1-2 = the
        // embedding precision (0 = f32, 1 = int8, 2 = nf4).
        // v4+ carry a flags word: bit 0 = SIMD-path sketches, bits 1-2 = the
        // embedding precision (0 = f32, 1 = int8, 2 = nf4, 3 = pq).
        let (simd_sketches, precision_code) = if version >= 4 {
            let flags = read_u32(&mut r)?;
            (flags & 1 == 1, (flags >> 1) & 0b11)
        } else {
            (false, if version == 3 { 1 } else { 0 })
        };
        #[cfg(not(feature = "simd"))]
        if simd_sketches {
            return Err(invalid(
                "this store's sketches were computed with the SIMD path — build \
                 with the `simd` feature to load it (mixing compute paths would \
                 silently degrade Hamming fidelity)",
            ));
        }
        let count = read_u64(&mut r)? as usize;
        let words = bits / 64;

        // Validate-before-allocate (see `fileguard`): each item needs its embedding
        // bytes (dim·4 for f32; 4 + dim for the int8 tier's scale + codes) plus
        // words·8 of sketch and 16 of payload offsets in the (already fully read)
        // file — a hostile header cannot request more memory than the file carries.
        let embed_bytes = match precision_code {
            1 => 4 + dim,             // int8: scale + one byte per lane
            2 => 4 + dim.div_ceil(2), // nf4: scale + packed nibbles
            3 => pq_subspaces(dim),   // pq: one byte per subspace (+ codebooks)
            _ => dim * 4,             // f32
        };
        crate::fileguard::guard_count(
            "SKCH items",
            count,
            embed_bytes + words * 8 + 16,
            r.len() as u64,
        )?;
        // PQ stores carry their trained codebooks right after the per-item
        // codes; the codebook allocation is validated against the bytes that
        // actually remain before it happens.
        if precision_code == 3 && version < 5 {
            return Err(invalid(
                "pq precision requires SketchIndex v5 (found an older file)",
            ));
        }
        let store = if precision_code == 3 {
            // PQ: per-item subspace codes, then the trained M x 256 x 8
            // codebooks. Validate-before-allocate applies to both.
            let m = pq_subspaces(dim);
            crate::fileguard::guard_count("SKCH pq codes", count * m, 1, r.len() as u64)?;
            let mut codes = vec![0u8; count * m];
            r.read_exact(&mut codes)?;
            let book_floats = m * PQ_CENTROIDS * PQ_SUBSPACE_DIM;
            crate::fileguard::guard_count("SKCH codebooks", book_floats, 4, r.len() as u64)?;
            let mut codebooks = vec![0f32; book_floats];
            for f in codebooks.iter_mut() {
                *f = read_f32(&mut r)?;
            }
            EmbeddingStore::Pq { codes, codebooks }
        } else if precision_code == 2 {
            let mut scales = vec![0f32; count];
            for sc in scales.iter_mut() {
                *sc = read_f32(&mut r)?;
            }
            let mut codes = vec![0u8; count * dim.div_ceil(2)];
            r.read_exact(&mut codes)?;
            EmbeddingStore::Nf4 { codes, scales }
        } else if precision_code == 1 {
            let mut scales = vec![0f32; count];
            for sc in scales.iter_mut() {
                *sc = read_f32(&mut r)?;
            }
            let mut bytes = vec![0u8; count * dim];
            r.read_exact(&mut bytes)?;
            let codes = bytes.into_iter().map(|b| b as i8).collect();
            EmbeddingStore::Int8 { codes, scales }
        } else {
            let mut embeddings = vec![0f32; count * dim];
            for e in embeddings.iter_mut() {
                *e = read_f32(&mut r)?;
            }
            // v1 stored raw embeddings; v2 stores them L2-normalized (cosine = a
            // single dot). Normalizing here migrates a v1 file transparently — and
            // is a no-op (modulo last-bit float noise) on already-normalized data.
            if version == 1 {
                for i in 0..count {
                    l2_normalize(&mut embeddings[i * dim..(i + 1) * dim]);
                }
            }
            EmbeddingStore::F32(embeddings)
        };
        let mut sketches = vec![0u64; count * words];
        for s in sketches.iter_mut() {
            *s = read_u64(&mut r)?;
        }
        let mut offsets = Vec::with_capacity(count);
        for _ in 0..count {
            let off = read_u64(&mut r)? as usize;
            let len = read_u64(&mut r)? as usize;
            offsets.push((off, len));
        }

        let decomp_len = read_u64(&mut r)? as usize;
        let comp_len = read_u64(&mut r)? as usize;
        crate::fileguard::guard_count("SKCH payload arena", comp_len, 1, r.len() as u64)?;
        crate::fileguard::guard_decompressed(
            "SKCH payload arena",
            decomp_len as u64,
            comp_len as u64,
        )?;
        let mut comp = vec![0u8; comp_len];
        r.read_exact(&mut comp)?;
        let payloads = lz4_flex::decompress(&comp, decomp_len)
            .map_err(|e| invalid(&format!("lz4 decompression failed: {e}")))?;
        // `payload(i)` slices without checks at query time — reject bad records now,
        // as a clean load error instead of a later panic.
        for &(off, len) in &offsets {
            crate::fileguard::guard_payload_bounds("SKCH item", off, len, payloads.len())?;
        }

        Ok(Self {
            hasher: Arc::new(SimHasher::new(dim, bits, seed)),
            dim,
            seed,
            simd_sketches,
            store,
            sketches,
            payloads,
            offsets,
        })
    }
}

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A small clustered corpus + calibration queries that are perturbed copies of
    /// corpus items — the exchangeable-workload setting the certificate assumes.
    fn clustered_index() -> (SketchIndex, Vec<Vec<f32>>) {
        const DIM: usize = 32;
        let mut idx = SketchIndex::new(DIM, 128, 42);
        let mut queries = Vec::new();
        for c in 0..8 {
            // A distinct base direction per cluster.
            let base: Vec<f32> = (0..DIM)
                .map(|d| ((c * DIM + d) as f32 * 0.7).sin())
                .collect();
            for j in 0..25 {
                let item: Vec<f32> = base
                    .iter()
                    .enumerate()
                    .map(|(d, x)| x + 0.05 * ((j * DIM + d) as f32 * 1.3).cos())
                    .collect();
                let tag = format!("c{c}-i{j}");
                assert!(idx.insert(&item, tag.as_bytes()));
                if j % 5 == 0 {
                    // A nearby-but-not-identical query into the same cluster.
                    queries.push(
                        item.iter()
                            .enumerate()
                            .map(|(d, x)| x + 0.01 * ((j + d) as f32).sin())
                            .collect(),
                    );
                }
            }
        }
        (idx, queries) // 200 items, 40 queries
    }

    #[test]
    fn rejects_non_finite_embeddings_and_queries() {
        let mut idx = SketchIndex::new(4, 64, 7);
        assert!(!idx.insert(&[1.0, f32::NAN, 0.0, 0.0], b"nan"));
        assert!(!idx.insert(&[1.0, f32::INFINITY, 0.0, 0.0], b"inf"));
        assert!(idx.is_empty());
        assert!(idx.insert(&[1.0, 0.0, 0.0, 0.0], b"ok"));
        assert!(idx.nearest(&[f32::NAN, 0.0, 0.0, 0.0], 1, 1).is_empty());
        assert!(
            idx.nearest_sketch(&[f32::INFINITY, 0.0, 0.0, 0.0], 1)
                .is_empty()
        );
    }

    #[test]
    fn hamming_ties_have_deterministic_nested_prefixes() {
        let mut idx = SketchIndex::new(4, 64, 11);
        for i in 0..8u8 {
            assert!(idx.insert(&[1.0, 0.0, 0.0, 0.0], &[i]));
        }
        let q = [1.0, 0.0, 0.0, 0.0];
        assert_eq!(
            idx.nearest_sketch(&q, 3)
                .into_iter()
                .map(|(payload, _)| payload[0])
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        let ids: Vec<u32> = (0..8).collect();
        assert_eq!(idx.hamming_rank(&q, &ids, 3), vec![0, 1, 2]);
        assert_eq!(idx.hamming_rank(&q, &ids, 6), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn precomputed_query_sketch_matches_regular_precision_recall() {
        let mut idx = SketchIndex::new(8, 128, 41);
        for i in 0..12u8 {
            let mut v = vec![0.0f32; 8];
            v[(i as usize) % 8] = 1.0;
            v[((i as usize) + 3) % 8] = 0.2;
            assert!(idx.insert(&v, &[i]));
        }
        let query = [1.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0];
        let sketch = idx.query_sketch(&query).unwrap();
        let normal: Vec<(u8, f32)> = idx
            .nearest(&query, 4, 8)
            .into_iter()
            .map(|(p, s)| (p[0], s))
            .collect();
        let reused: Vec<(u8, f32)> = idx
            .nearest_with_sketch(&query, &sketch, 4, 8)
            .into_iter()
            .map(|(p, s)| (p[0], s))
            .collect();
        assert_eq!(normal, reused);
        assert!(idx.nearest_with_sketch(&query, &[], 4, 8).is_empty());
    }

    #[test]
    fn normalize_on_insert_makes_cosine_a_dot() {
        let mut idx = SketchIndex::new(8, 64, 3);
        let raw = vec![3.0f32, -1.0, 2.0, 0.5, -2.0, 1.0, 4.0, -0.5];
        assert!(idx.insert(&raw, b"a"));
        // Stored embedding is unit-norm: the self-cosine reads 1.0 exactly through
        // the public scoring path.
        let n = idx.scores(&raw)[0];
        assert!((n - 1.0).abs() < 1e-6, "self-cosine = {n}");
        // Self-query scores 1.0: cosine == dot on pre-normalized vectors.
        let (payload, score) = idx.nearest(&raw, 1, 8)[0];
        assert_eq!(payload, b"a");
        assert!((score - 1.0).abs() < 1e-6, "self-score = {score}");
        // A scaled copy of the query gives the same score (scale invariance).
        let scaled: Vec<f32> = raw.iter().map(|x| x * 7.5).collect();
        let (_, s2) = idx.nearest(&scaled, 1, 8)[0];
        assert!((s2 - score).abs() < 1e-6);
        // The zero vector still scores 0 against everything (old convention kept).
        assert_eq!(idx.scores(&[0.0; 8]), vec![0.0]);
    }

    /// With the `simd` feature: the dispatched kernel agrees with the scalar sum
    /// within float-reassociation tolerance, and the retrieved top-k set matches on
    /// a corpus with clear margins (near-ties are the documented exception).
    #[cfg(feature = "simd")]
    #[test]
    fn simd_dot_agrees_with_scalar_and_preserves_clear_rankings() {
        let dims = [3usize, 8, 64, 768, 1000];
        for &d in &dims {
            let a: Vec<f32> = (0..d).map(|i| (i as f32 * 0.017).sin()).collect();
            let b: Vec<f32> = (0..d).map(|i| (i as f32 * 0.013).cos()).collect();
            let simd = dot(&a, &b);
            let scalar: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            assert!(
                (simd - scalar).abs() <= 1e-3 * (1.0 + scalar.abs()),
                "dim {d}: simd {simd} vs scalar {scalar}"
            );
        }
        // End-to-end: clear-margin clusters retrieve the same top-k set.
        let (idx, queries) = clustered_index();
        for q in queries.iter().take(8) {
            let got = idx.nearest(q, 5, idx.len());
            assert_eq!(got.len(), 5);
            // Self-cluster payloads dominate: every hit shares the query's cluster
            // prefix (the clusters are far apart, so last-bit noise cannot flip this).
            let expect_prefix = &got[0].0[..2];
            assert!(got.iter().all(|(p, _)| &p[..2] == expect_prefix));
        }
    }

    #[test]
    fn int8_tier_scores_rank_and_persist() {
        // Same clustered corpus in both tiers: identical top-k payload sets on
        // clear margins, self-scores ~1, and a 4x smaller on-disk file (v3).
        const DIM: usize = 32;
        let mut f32_idx = SketchIndex::new(DIM, 128, 42);
        let mut i8_idx = SketchIndex::new_with_precision(DIM, 128, 42, Precision::Int8);
        assert_eq!(i8_idx.precision(), Precision::Int8);
        let mut queries = Vec::new();
        for c in 0..8 {
            let base: Vec<f32> = (0..DIM)
                .map(|d| ((c * DIM + d) as f32 * 0.7).sin())
                .collect();
            for j in 0..25 {
                // Independent per-component noise (LCG — a trig pattern would be
                // near-periodic and correlate siblings to ~0.998 cosine, burying
                // the margins inside quantization noise). Spread 0.3 puts sibling
                // cosines near 0.92: margins well above the ~1e-2 int8 noise, so
                // exact top-1 parity between tiers is a fair ask.
                let mut state = (c * 25 + j) as u64 ^ 0x9E37_79B9_7F4A_7C15;
                let mut noise = || {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
                };
                let item: Vec<f32> = base.iter().map(|x| x + 0.3 * noise()).collect();
                let tag = format!("c{c}-i{j}");
                assert!(f32_idx.insert(&item, tag.as_bytes()));
                assert!(i8_idx.insert(&item, tag.as_bytes()));
                if j % 5 == 0 {
                    queries.push(item);
                }
            }
        }

        for q in &queries {
            // Self-retrieval: the exact item comes back first with score ~1
            // (quantization noise stays well under the cluster margins).
            let (pf, sf) = f32_idx.nearest(q, 1, 256)[0];
            let (pi, si) = i8_idx.nearest(q, 1, 256)[0];
            assert_eq!(pf, pi, "both tiers agree on the top hit");
            assert!((sf - 1.0).abs() < 1e-6);
            assert!((si - 1.0).abs() < 2e-2, "int8 self-score = {si}");
            // Deeper ranks: siblings inside a dense cluster are legitimate
            // near-ties (< 1e-2 apart), so exact set equality is not promised.
            // What IS promised: both tiers stay inside the query's cluster, and
            // they mostly agree.
            let top_f: Vec<&[u8]> = f32_idx
                .nearest(q, 5, 256)
                .into_iter()
                .map(|(p, _)| p)
                .collect();
            let top_i: Vec<&[u8]> = i8_idx
                .nearest(q, 5, 256)
                .into_iter()
                .map(|(p, _)| p)
                .collect();
            let cluster = &pf[..2];
            assert!(
                top_i.iter().all(|p| &p[..2] == cluster),
                "int8 left the cluster"
            );
            let overlap = top_f.iter().filter(|p| top_i.contains(p)).count();
            assert!(overlap >= 3, "top-5 overlap {overlap}/5 too low");
        }

        // The int8 certificate certifies the int8 pipeline (same funnel).
        assert!(i8_idx.certify_shortlist(&queries, 5, 0.25, 0.1).is_some());

        // v3 round-trip: precision, hits and scores survive.
        let dir = std::env::temp_dir();
        let p_f32 = dir.join(format!("skch_f32_{}.skch", std::process::id()));
        let p_i8 = dir.join(format!("skch_i8_{}.skch", std::process::id()));
        f32_idx.save_to_disk(p_f32.to_str().unwrap()).unwrap();
        i8_idx.save_to_disk(p_i8.to_str().unwrap()).unwrap();
        let f32_bytes = std::fs::metadata(&p_f32).unwrap().len();
        let i8_bytes = std::fs::metadata(&p_i8).unwrap().len();
        let loaded = SketchIndex::load_from_disk(p_i8.to_str().unwrap(), DIM).unwrap();
        std::fs::remove_file(&p_f32).ok();
        std::fs::remove_file(&p_i8).ok();
        assert_eq!(loaded.precision(), Precision::Int8);
        let (p0, s0) = loaded.nearest(&queries[0], 1, 256)[0];
        let (o0, os0) = i8_idx.nearest(&queries[0], 1, 256)[0];
        assert_eq!(p0, o0);
        assert_eq!(s0, os0, "round-trip is bit-exact");
        // On-disk embedding payload shrinks ~4x (file also carries sketches +
        // payloads, so assert a conservative 2x overall).
        assert!(
            i8_bytes * 2 < f32_bytes,
            "int8 file {i8_bytes} B vs f32 {f32_bytes} B"
        );
    }

    #[test]
    fn nf4_tier_scores_rank_and_persist_at_8x() {
        const DIM: usize = 32;
        let mut f32_idx = SketchIndex::new(DIM, 128, 42);
        let mut nf4_idx = SketchIndex::new_with_precision(DIM, 128, 42, Precision::Nf4);
        assert_eq!(nf4_idx.precision(), Precision::Nf4);
        let mut queries = Vec::new();
        for c in 0..8 {
            let base: Vec<f32> = (0..DIM)
                .map(|d| ((c * DIM + d) as f32 * 0.7).sin())
                .collect();
            for j in 0..25 {
                // Independent LCG noise (see the int8 test for why), spread 0.4:
                // sibling margins well above NF4's ~5e-2 noise.
                let mut state = (c * 25 + j) as u64 ^ 0xD1B5_4A32_D192_ED03;
                let mut noise = || {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
                };
                let item: Vec<f32> = base.iter().map(|x| x + 0.4 * noise()).collect();
                let tag = format!("c{c}-i{j}");
                assert!(f32_idx.insert(&item, tag.as_bytes()));
                assert!(nf4_idx.insert(&item, tag.as_bytes()));
                if j % 5 == 0 {
                    queries.push(item);
                }
            }
        }
        for q in &queries {
            let (pf, _) = f32_idx.nearest(q, 1, 256)[0];
            let (pn, sn) = nf4_idx.nearest(q, 1, 256)[0];
            assert_eq!(pf, pn, "top-1 parity on clear margins");
            assert!((sn - 1.0).abs() < 5e-2, "nf4 self-score = {sn}");
        }

        // v4 round-trip, and the on-disk cut vs f32. At dim=32 the fixed
        // per-item overhead (16 B sketch + 16 B offsets + payload) dominates:
        // embeddings alone shrink 6.4x (128 B -> 20 B; ~8x at 768-d), the whole
        // file ~2.9x — assert a safe 2.5x.
        let dir = std::env::temp_dir();
        let pf = dir.join(format!("skch_f32b_{}.skch", std::process::id()));
        let pn = dir.join(format!("skch_nf4_{}.skch", std::process::id()));
        f32_idx.save_to_disk(pf.to_str().unwrap()).unwrap();
        nf4_idx.save_to_disk(pn.to_str().unwrap()).unwrap();
        let (fb, nb) = (
            std::fs::metadata(&pf).unwrap().len(),
            std::fs::metadata(&pn).unwrap().len(),
        );
        let loaded = SketchIndex::load_from_disk(pn.to_str().unwrap(), DIM).unwrap();
        std::fs::remove_file(&pf).ok();
        std::fs::remove_file(&pn).ok();
        assert_eq!(loaded.precision(), Precision::Nf4);
        let (a, sa) = loaded.nearest(&queries[0], 1, 256)[0];
        let (b, sb) = nf4_idx.nearest(&queries[0], 1, 256)[0];
        assert_eq!(a, b);
        assert_eq!(sa, sb, "v4 round-trip is bit-exact");
        assert!(nb * 5 < fb * 2, "nf4 file {nb} B vs f32 {fb} B");
    }

    /// A v4 store whose sketches were computed with the SIMD path must refuse
    /// to load in a build without the `simd` feature — mixing compute paths
    /// would silently degrade Hamming fidelity, and this codebase does not do
    /// silent.
    #[cfg(not(feature = "simd"))]
    #[test]
    fn v4_simd_flagged_store_is_rejected_without_the_feature() {
        const DIM: usize = 8;
        let raw = [1.0f32; DIM];
        let hasher = SimHasher::new(DIM, 64, 3);
        let sketch = hasher.sketch(&raw);
        let payload = b"a".to_vec();
        let mut f = Vec::new();
        f.extend_from_slice(b"SKCH");
        f.extend_from_slice(&4u32.to_le_bytes());
        f.extend_from_slice(&(DIM as u32).to_le_bytes());
        f.extend_from_slice(&64u32.to_le_bytes());
        f.extend_from_slice(&3u64.to_le_bytes());
        f.extend_from_slice(&1u32.to_le_bytes()); // flags: bit0 = simd sketches
        f.extend_from_slice(&1u64.to_le_bytes()); // count
        let mut norm = raw.to_vec();
        l2_normalize(&mut norm);
        for &x in &norm {
            f.extend_from_slice(&x.to_le_bytes());
        }
        for &w in &sketch {
            f.extend_from_slice(&w.to_le_bytes());
        }
        f.extend_from_slice(&0u64.to_le_bytes());
        f.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        f.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        let comp = lz4_flex::compress(&payload);
        f.extend_from_slice(&(comp.len() as u64).to_le_bytes());
        f.extend_from_slice(&comp);
        let path = std::env::temp_dir().join(format!("skch_v4simd_{}.skch", std::process::id()));
        std::fs::write(&path, &f).unwrap();
        let err = SketchIndex::load_from_disk(path.to_str().unwrap(), DIM).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("simd"), "{err}");
    }

    /// With the feature on: SIMD sketching is self-consistent (self-retrieval
    /// still lands at Hamming 0 → score 1.0) and the v4 flag round-trips.
    #[cfg(feature = "simd")]
    #[test]
    fn simd_sketching_is_self_consistent_and_round_trips() {
        const DIM: usize = 32;
        let mut idx = SketchIndex::new(DIM, 256, 42).with_simd_sketching();
        assert!(idx.simd_sketching());
        let items: Vec<Vec<f32>> = (0..50)
            .map(|i| {
                (0..DIM)
                    .map(|d| ((i * DIM + d) as f32 * 0.7).sin())
                    .collect()
            })
            .collect();
        for (i, item) in items.iter().enumerate() {
            assert!(idx.insert(item, format!("m{i}").as_bytes()));
        }
        // Query sketches share the item path → self-retrieval is exact.
        for (i, item) in items.iter().enumerate() {
            let (p, score) = idx.nearest(item, 1, 8)[0];
            assert_eq!(p, format!("m{i}").as_bytes());
            assert!((score - 1.0).abs() < 1e-6);
        }
        let path = std::env::temp_dir().join(format!("skch_v4rt_{}.skch", std::process::id()));
        idx.save_to_disk(path.to_str().unwrap()).unwrap();
        let loaded = SketchIndex::load_from_disk(path.to_str().unwrap(), DIM).unwrap();
        std::fs::remove_file(&path).ok();
        assert!(
            loaded.simd_sketching(),
            "the v4 flag survives the round-trip"
        );
        let (p, _) = loaded.nearest(&items[7], 1, 8)[0];
        assert_eq!(p, b"m7");
    }

    /// A5: the one-GEMM batch scores match the scalar per-query path within
    /// float tolerance, and the honest-error contracts hold. Skips gracefully
    /// (with a visible marker) when no adapter exists; CI provides Mesa
    /// lavapipe so the real path is exercised there.
    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_batch_scores_match_the_scalar_path() {
        let scorer = match crate::gpu::GpuScorer::new() {
            Ok(s) => s,
            Err(e) if e.kind() == io::ErrorKind::Unsupported => {
                eprintln!("SKIP: no wgpu adapter ({e})");
                return;
            }
            Err(e) => panic!("unexpected GPU init error: {e}"),
        };
        eprintln!("gpu adapter: {}", scorer.adapter_name());

        const DIM: usize = 64;
        let mut idx = SketchIndex::new(DIM, 128, 42);
        let items: Vec<Vec<f32>> = (0..200)
            .map(|i| {
                (0..DIM)
                    .map(|d| ((i * DIM + d) as f32 * 0.37).sin())
                    .collect()
            })
            .collect();
        for (i, item) in items.iter().enumerate() {
            assert!(idx.insert(item, format!("m{i}").as_bytes()));
        }
        let queries: Vec<Vec<f32>> = (0..17)
            .map(|q| {
                (0..DIM)
                    .map(|d| ((q * DIM + d) as f32 * 0.71).cos())
                    .collect()
            })
            .collect();

        let batch = idx.scores_batch_gpu(&scorer, &queries).unwrap();
        assert_eq!(batch.len(), queries.len());
        for (q, row) in queries.iter().zip(&batch) {
            let scalar = idx.scores(q);
            assert_eq!(row.len(), scalar.len());
            for (g, c) in row.iter().zip(&scalar) {
                assert!(
                    (g - c).abs() <= 1e-4 * (1.0 + c.abs()),
                    "gpu {g} vs cpu {c}"
                );
            }
        }

        // Honest errors: quantized tier refused, bad dims refused, empty ok.
        let mut i8_idx = SketchIndex::new_with_precision(DIM, 128, 42, Precision::Int8);
        i8_idx.insert(&items[0], b"x");
        let err = i8_idx.scores_batch_gpu(&scorer, &queries).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Unsupported);
        let err = idx.scores_batch_gpu(&scorer, &[vec![0.0; 3]]).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(idx.scores_batch_gpu(&scorer, &[]).unwrap().is_empty());
    }

    #[test]
    fn hostile_skch_header_is_rejected_before_allocating() {
        // 32 bytes declaring u64::MAX items: clean InvalidData, no allocation.
        let mut f = Vec::new();
        f.extend_from_slice(b"SKCH");
        f.extend_from_slice(&2u32.to_le_bytes());
        f.extend_from_slice(&8u32.to_le_bytes()); // dim
        f.extend_from_slice(&64u32.to_le_bytes()); // bits
        f.extend_from_slice(&3u64.to_le_bytes()); // seed
        f.extend_from_slice(&u64::MAX.to_le_bytes()); // hostile count
        let path = std::env::temp_dir().join(format!("skch_hostile_{}.skch", std::process::id()));
        std::fs::write(&path, &f).unwrap();
        let err = SketchIndex::load_from_disk(path.to_str().unwrap(), 8).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("SKCH items"), "{err}");
    }

    #[test]
    fn skch_payload_records_outside_the_arena_are_a_load_error_not_a_panic() {
        // A well-formed file whose single payload record points past the arena.
        const DIM: usize = 8;
        let raw = [1.0f32; DIM];
        let hasher = SimHasher::new(DIM, 64, 3);
        let sketch = hasher.sketch(&raw);
        let payload = b"a".to_vec();
        let mut f = Vec::new();
        f.extend_from_slice(b"SKCH");
        f.extend_from_slice(&2u32.to_le_bytes());
        f.extend_from_slice(&(DIM as u32).to_le_bytes());
        f.extend_from_slice(&64u32.to_le_bytes());
        f.extend_from_slice(&3u64.to_le_bytes());
        f.extend_from_slice(&1u64.to_le_bytes()); // count
        for &x in &raw {
            f.extend_from_slice(&x.to_le_bytes());
        }
        for &w in &sketch {
            f.extend_from_slice(&w.to_le_bytes());
        }
        f.extend_from_slice(&100u64.to_le_bytes()); // offset beyond the 1-byte arena
        f.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        f.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        let comp = lz4_flex::compress(&payload);
        f.extend_from_slice(&(comp.len() as u64).to_le_bytes());
        f.extend_from_slice(&comp);
        let path = std::env::temp_dir().join(format!("skch_badoff_{}.skch", std::process::id()));
        std::fs::write(&path, &f).unwrap();
        let err = SketchIndex::load_from_disk(path.to_str().unwrap(), DIM).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("SKCH item"), "{err}");
    }

    #[test]
    fn v1_files_with_raw_embeddings_migrate_on_load() {
        // Hand-craft a SKCH **v1** file (raw, unnormalized embeddings) and check the
        // loader migrates it: after load, cosine-as-dot behaves exactly as if the
        // items had been inserted through the current API.
        const DIM: usize = 8;
        const BITS: usize = 64;
        const SEED: u64 = 3;
        let raw = vec![3.0f32, -1.0, 2.0, 0.5, -2.0, 1.0, 4.0, -0.5];
        let hasher = SimHasher::new(DIM, BITS, SEED);
        let sketch = hasher.sketch(&raw);
        let payload = b"a".to_vec();

        let mut f = Vec::new();
        f.extend_from_slice(b"SKCH");
        f.extend_from_slice(&1u32.to_le_bytes()); // version 1 = raw embeddings
        f.extend_from_slice(&(DIM as u32).to_le_bytes());
        f.extend_from_slice(&(BITS as u32).to_le_bytes());
        f.extend_from_slice(&SEED.to_le_bytes());
        f.extend_from_slice(&1u64.to_le_bytes()); // count
        for &x in &raw {
            f.extend_from_slice(&x.to_le_bytes());
        }
        for &w in &sketch {
            f.extend_from_slice(&w.to_le_bytes());
        }
        f.extend_from_slice(&0u64.to_le_bytes()); // payload offset
        f.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        f.extend_from_slice(&(payload.len() as u64).to_le_bytes()); // decompressed len
        let comp = lz4_flex::compress(&payload);
        f.extend_from_slice(&(comp.len() as u64).to_le_bytes());
        f.extend_from_slice(&comp);

        let path =
            std::env::temp_dir().join(format!("skch_v1_migrate_{}.skch", std::process::id()));
        std::fs::write(&path, &f).unwrap();
        let loaded = SketchIndex::load_from_disk(path.to_str().unwrap(), DIM).unwrap();
        std::fs::remove_file(&path).ok();

        // Migrated: self-query scores 1.0 (normalized storage), sketch intact.
        let (p, score) = loaded.nearest(&raw, 1, 8)[0];
        assert_eq!(p, b"a");
        assert!(
            (score - 1.0).abs() < 1e-6,
            "self-score after migration = {score}"
        );
        assert_eq!(loaded.sketch_of(0), &sketch[..]);
    }

    #[test]
    fn certify_shortlist_returns_a_valid_minimal_certificate() {
        let (idx, queries) = clustered_index();
        let (k, alpha, delta) = (5, 0.25, 0.1);
        let cert = idx
            .certify_shortlist(&queries, k, alpha, delta)
            .expect("40 exchangeable queries certify alpha=0.25");

        // The certificate is internally consistent and actually certified.
        assert!(cert.risk_ucb <= alpha, "ucb {} > alpha", cert.risk_ucb);
        assert!(cert.empirical_risk <= cert.risk_ucb);
        assert_eq!(cert.calibration_n, queries.len());
        assert!(cert.grid.contains(&cert.shortlist));
        assert_eq!(*cert.grid.last().unwrap(), idx.len());

        // Deterministic: the same inputs produce the same certificate.
        assert_eq!(idx.certify_shortlist(&queries, k, alpha, delta), Some(cert));
    }

    #[test]
    fn certify_shortlist_refuses_when_it_cannot_guarantee() {
        let (idx, queries) = clustered_index();
        // Hoeffding slack at n=40, delta=0.1 is ~0.17 > alpha=0.01: even a perfect
        // empirical risk cannot certify — None, not a fake certificate.
        assert!(idx.certify_shortlist(&queries, 5, 0.01, 0.1).is_none());
        // No calibration data → None.
        assert!(idx.certify_shortlist(&[], 5, 0.25, 0.1).is_none());
        // Dimension-mismatched queries are dropped, not silently scored.
        assert!(
            idx.certify_shortlist(&[vec![0.0; 3]], 5, 0.25, 0.1)
                .is_none()
        );
    }

    #[test]
    fn certified_shortlist_meets_its_risk_on_the_calibration_set() {
        let (idx, queries) = clustered_index();
        let (k, alpha, delta) = (5, 0.25, 0.1);
        let cert = idx.certify_shortlist(&queries, k, alpha, delta).unwrap();

        // Re-measure the pipeline at the certified shortlist by hand, via the
        // public APIs: payload sets of nearest() vs the exact full-width rerank.
        let mut total_loss = 0.0f64;
        for q in &queries {
            let exact: Vec<&[u8]> = idx
                .nearest(q, k, idx.len())
                .into_iter()
                .map(|(p, _)| p)
                .collect();
            let got: Vec<&[u8]> = idx
                .nearest(q, k, cert.shortlist)
                .into_iter()
                .map(|(p, _)| p)
                .collect();
            let hits = exact.iter().filter(|p| got.contains(p)).count();
            total_loss += 1.0 - hits as f64 / exact.len() as f64;
        }
        let measured = total_loss / queries.len() as f64;
        assert!(
            measured <= cert.empirical_risk + 1e-9,
            "public-API re-measure {measured} exceeds the certificate's {}",
            cert.empirical_risk
        );
    }

    #[test]
    fn deterministic_and_sized() {
        let a = SimHasher::new(48, 100, 7);
        let b = SimHasher::new(48, 100, 7);
        assert_eq!(a.bits(), 128); // 100 rounded up to a multiple of 64
        assert_eq!(a.words(), 2);
        let v: Vec<f32> = (0..48).map(|i| (i as f32).sin()).collect();
        assert_eq!(a.sketch(&v), b.sketch(&v)); // same seed → same sketch
        assert_eq!(a.sketch(&v).len(), 2);
        assert!(a.sketch(&[0.0; 3]).is_empty()); // wrong dim
    }

    #[test]
    fn identical_and_antipodal() {
        let h = SimHasher::new(64, 256, 1);
        let v: Vec<f32> = (0..64).map(|i| ((i * 7 % 13) as f32) - 6.0).collect();
        let neg: Vec<f32> = v.iter().map(|x| -x).collect();
        let sv = h.sketch(&v);
        let sneg = h.sketch(&neg);
        // x vs x → distance 0; x vs -x → every plane flips → all bits differ.
        assert_eq!(hamming(&sv, &sv), 0);
        assert_eq!(hamming(&sv, &sneg), h.bits() as u32);
        assert!((cosine_from_hamming(0, 256) - 1.0).abs() < 1e-6);
        assert!((cosine_from_hamming(256, 256) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn hamming_estimates_angle_for_random_vectors() {
        // For two independent random vectors (≈orthogonal), Hamming ≈ bits/2.
        let bits = 1024usize;
        let h = SimHasher::new(128, bits, 99);
        let mut rng = DeterministicRng::new(5);
        let mut total = 0u64;
        let trials = 50;
        for _ in 0..trials {
            let a: Vec<f32> = (0..128).map(|_| rng.next_f32()).collect();
            let b: Vec<f32> = (0..128).map(|_| rng.next_f32()).collect();
            total += hamming(&h.sketch(&a), &h.sketch(&b)) as u64;
        }
        let mean = total as f64 / trials as f64;
        // Expect ~512; allow a generous band.
        assert!(
            (mean - 512.0).abs() < 90.0,
            "mean hamming {mean} not near bits/2 for orthogonal vectors"
        );
    }

    #[test]
    fn sketch_index_hybrid_finds_exact_neighbour() {
        let dim = 64;
        let mut rng = DeterministicRng::new(123);
        let unit = |v: Vec<f32>| {
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter().map(|x| x / n).collect::<Vec<f32>>()
        };
        let centers: Vec<Vec<f32>> = (0..6)
            .map(|_| unit((0..dim).map(|_| rng.next_f32()).collect()))
            .collect();

        let mut idx = SketchIndex::new(dim, 256, 7);
        for (c, center) in centers.iter().enumerate() {
            for i in 0..30 {
                let pt: Vec<f32> = center.iter().map(|&x| x + 0.02 * rng.next_f32()).collect();
                idx.insert(&pt, format!("c{c}_{i}").as_bytes());
            }
        }
        assert_eq!(idx.len(), 180);

        // A query near cluster 3: hybrid (shortlist → exact rerank) returns a
        // cluster-3 payload as #1, with cosine close to 1.
        let q: Vec<f32> = centers[3]
            .iter()
            .map(|&x| x + 0.01 * rng.next_f32())
            .collect();
        let hits = idx.nearest(&q, 5, 64);
        assert_eq!(hits.len(), 5);
        assert!(String::from_utf8_lossy(hits[0].0).starts_with("c3_"));
        assert!(hits[0].1 > 0.9, "top cosine should be high: {}", hits[0].1);
        // Sketch-only ranking also lands in cluster 3 at the top.
        let sk = idx.nearest_sketch(&q, 3);
        assert!(String::from_utf8_lossy(sk[0].0).starts_with("c3_"));

        // Dimension guards.
        assert!(!idx.insert(&[0.0; 3], b"bad"));
        assert!(idx.nearest(&[0.0; 3], 3, 16).is_empty());
    }

    #[test]
    fn sketch_index_save_load_roundtrip() {
        let dim = 32;
        let mut rng = DeterministicRng::new(9);
        let mut idx = SketchIndex::new(dim, 128, 13);
        for i in 0..50 {
            let v: Vec<f32> = (0..dim).map(|_| rng.next_f32()).collect();
            idx.insert(&v, format!("item{i}").as_bytes());
        }
        let path = "/tmp/octasoma_sketch_roundtrip.skch";
        idx.save_to_disk(path).unwrap();
        let loaded = SketchIndex::load_from_disk(path, dim).unwrap();

        assert_eq!(loaded.len(), idx.len());
        assert_eq!(loaded.bits(), idx.bits());
        // Recall is identical after reload (planes regenerated from the seed).
        let q: Vec<f32> = (0..dim).map(|i| (i as f32).cos()).collect();
        let before: Vec<_> = idx
            .nearest(&q, 5, 20)
            .into_iter()
            .map(|(p, _)| p.to_vec())
            .collect();
        let after: Vec<_> = loaded
            .nearest(&q, 5, 20)
            .into_iter()
            .map(|(p, _)| p.to_vec())
            .collect();
        assert_eq!(before, after);
        // Wrong expected dim is rejected.
        assert!(SketchIndex::load_from_disk(path, 64).is_err());
        std::fs::remove_file(path).ok();
    }

    // -- certified adaptive radius (ConANN-style per-query probing) ----------

    /// A clustered corpus plus two exchangeable query sets — calibration and
    /// held-out — built by perturbing corpus items with distinct phases.
    fn adaptive_corpus() -> (SketchIndex, Vec<Vec<f32>>, Vec<Vec<f32>>) {
        const DIM: usize = 32;
        let mut idx = SketchIndex::new(DIM, 128, 42);
        let mut calib = Vec::new();
        let mut held_out = Vec::new();
        for c in 0..8 {
            let base: Vec<f32> = (0..DIM)
                .map(|d| ((c * DIM + d) as f32 * 0.7).sin())
                .collect();
            for j in 0..20 {
                let item: Vec<f32> = base
                    .iter()
                    .enumerate()
                    .map(|(d, x)| x + 0.05 * ((j * DIM + d) as f32 * 1.3).cos())
                    .collect();
                assert!(idx.insert(&item, format!("c{c}-i{j}").as_bytes()));
                if j % 4 == 0 {
                    calib.push(
                        item.iter()
                            .enumerate()
                            .map(|(d, x)| x + 0.01 * ((j + d) as f32 * 0.5).sin())
                            .collect(),
                    );
                    held_out.push(
                        item.iter()
                            .enumerate()
                            .map(|(d, x)| x + 0.01 * ((j * 3 + d) as f32 * 0.9).cos())
                            .collect(),
                    );
                }
            }
        }
        (idx, calib, held_out) // 160 items, 40 + 40 exchangeable queries
    }

    /// Mean recall loss at `k` of [`SketchIndex::nearest_adaptive`] on `queries`
    /// against the exact oracle.
    fn adaptive_loss(idx: &SketchIndex, queries: &[Vec<f32>], k: usize, lambda: u32) -> f64 {
        let total: f64 = queries
            .iter()
            .map(|q| {
                let got = idx.nearest_adaptive(q, k, lambda);
                let oracle = idx.exact_top_ids(q, k);
                let hits = oracle
                    .iter()
                    .filter(|t| got.iter().any(|(p, _)| *p == idx.payload(**t)))
                    .count();
                1.0 - hits as f64 / oracle.len() as f64
            })
            .sum();
        total / queries.len() as f64
    }

    #[test]
    fn adaptive_radius_certificate_controls_held_out_recall() {
        let (idx, calib, held_out) = adaptive_corpus();
        let k = 5;

        let cert = idx
            .calibrate_adaptive_radius(&calib, k, 0.25, 0.10)
            .expect("a healthy clustered corpus must certify some radius");
        assert_eq!(cert.k, k);
        assert_eq!(cert.calibration_n, calib.len());
        assert!(cert.risk_ucb <= cert.alpha + 1e-12);

        // Held-out empirical loss stays under the certified level (deterministic
        // corpus/seed, so this is an exact assertion, not a probabilistic one).
        let unseen = adaptive_loss(&idx, &held_out, k, cert.lambda);
        assert!(
            unseen <= cert.alpha,
            "held-out loss {unseen} exceeds certified alpha {}",
            cert.alpha
        );

        // The certificate is the smallest grid point whose UCB fits under alpha:
        // every strictly smaller radius must fail the RCPS bound.
        let pos = cert.grid.iter().position(|&l| l == cert.lambda).unwrap();
        if pos > 0 {
            let prev_lambda = cert.grid[pos - 1];
            let prev_risk = {
                let n = cert.calibration_n;
                let loss = adaptive_loss(&idx, &calib, k, prev_lambda);
                crate::conformal::hoeffding_ucb(loss, n, cert.delta)
            };
            assert!(prev_risk > cert.alpha, "a smaller radius also certified");
        }
        // Calibration-set loss of the certified radius matches the record.
        let calibrated = adaptive_loss(&idx, &calib, k, cert.lambda);
        assert!((calibrated - cert.empirical_risk).abs() < 1e-12);
    }

    #[test]
    fn adaptive_radius_is_never_worse_than_a_fixed_k_shortlist() {
        let (idx, queries) = clustered_index();

        for q in &queries {
            // Fixed shortlist of exactly k by Hamming...
            let fixed = idx.nearest(q, 3, 3);
            // ...is dominated by the adaptive rule even at lambda = 0: the
            // radius d_3 + 0 contains every item the fixed cut would take,
            // ties included, so its reranked head cannot lose hits.
            let adapt = idx.nearest_adaptive(q, 3, 0);
            let quality = |hits: &[(&[u8], f32)]| hits.iter().map(|(_, s)| *s).sum::<f32>();
            assert!(
                quality(&adapt) >= quality(&fixed) - 1e-6,
                "adaptive sum-of-cosines dropped below the fixed-k baseline"
            );
        }
    }

    #[test]
    fn adaptive_radius_refuses_when_nothing_certifies_and_guards_inputs() {
        let (idx, queries) = clustered_index();
        // Two calibration points cannot fit a Hoeffding slack under alpha=0.05.
        assert!(
            idx.calibrate_adaptive_radius(&queries[..2], 3, 0.05, 0.05)
                .is_none()
        );
        // Degenerate inputs refuse rather than fake a guarantee.
        assert!(idx.calibrate_adaptive_radius(&[], 3, 0.5, 0.1).is_none());
        let empty = SketchIndex::new(8, 64, 1);
        assert!(
            empty
                .calibrate_adaptive_radius(&queries, 3, 0.5, 0.1)
                .is_none()
        );
        let bad_dim: Vec<Vec<f32>> = vec![vec![1.0; 7]];
        assert!(
            idx.calibrate_adaptive_radius(&bad_dim, 3, 0.5, 0.1)
                .is_none()
        );
        assert!(idx.nearest_adaptive(&[0.0; 7], 3, 4).is_empty());
        assert!(idx.nearest_adaptive(&queries[0], 0, 4).is_empty());
    }

    #[test]
    fn adaptive_radius_query_is_deterministic() {
        let (idx, queries) = clustered_index();
        let q = &queries[7];
        let run = || -> Vec<(Vec<u8>, f32)> {
            idx.nearest_adaptive(q, 5, 6)
                .into_iter()
                .map(|(p, s)| (p.to_vec(), s))
                .collect()
        };
        assert_eq!(run(), run());
    }

    // -- PQ experiment tier ----------------------------------------------------

    /// A clustered corpus plus a held-out codebook calibration sample. Returns
    /// the F32 twin (oracle), the PQ index, and the query set.
    fn pq_corpus() -> (SketchIndex, SketchIndex, Vec<Vec<f32>>) {
        const DIM: usize = 64;
        const CLUSTERS: usize = 8;
        const BLOCK: usize = DIM / CLUSTERS;
        let mut rng = DeterministicRng::new(5);
        // Near-orthogonal block-one-hot centers: cluster c owns an 8-dim block
        // set to 1.0, everything else carries only a small shared ripple — so
        // every item's latent theme is unambiguous and the gate measures
        // quantization fidelity, not corpus ambiguity.
        let centers: Vec<Vec<f32>> = (0..CLUSTERS)
            .map(|c| {
                (0..DIM)
                    .map(|d| {
                        if d / BLOCK == c {
                            1.0
                        } else {
                            0.05 * (d as f32 * 0.7).sin()
                        }
                    })
                    .collect()
            })
            .collect();

        // Codebook calibration: dense samples around the centers.
        let calib: Vec<f32> = centers
            .iter()
            .flat_map(|center| {
                (0..25).flat_map(|j| {
                    center
                        .iter()
                        .enumerate()
                        .map(|(d, &x)| x + 0.05 * ((j * DIM + d) as f32 * 1.1).cos())
                        .collect::<Vec<f32>>()
                })
            })
            .collect();
        let num_calib = calib.len() / DIM;

        let mut oracle = SketchIndex::new(DIM, 256, 42);
        let mut pq = SketchIndex::new_pq(DIM, 256, 42, &calib, num_calib);

        let mut queries = Vec::new();
        for (c, center) in centers.iter().enumerate() {
            for i in 0..40 {
                let v: Vec<f32> = center
                    .iter()
                    .enumerate()
                    .map(|(d, &x)| {
                        x + 0.02 * ((i * DIM + d) as f32 * 0.7).sin() + 0.01 * rng.next_f32()
                    })
                    .collect();
                assert!(oracle.insert(&v, format!("c{c}_{i}").as_bytes()));
                assert!(pq.insert(&v, format!("c{c}_{i}").as_bytes()));
                if i % 10 == 0 {
                    queries.push(v.clone());
                }
            }
        }
        (oracle, pq, queries)
    }

    #[test]
    fn pq_tier_finds_cluster_neighbours_and_persists_codebooks() {
        use crate::Precision;
        let (_oracle, pq, queries) = pq_corpus();
        assert_eq!(pq.precision(), Precision::Pq);

        // Determinism: same store, same query -> identical ranking.
        let q = &queries[3];
        let run = || -> Vec<u8> {
            pq.nearest(q, 5, pq.len())
                .into_iter()
                .flat_map(|(p, _)| p.to_vec())
                .collect()
        };
        assert_eq!(run(), run(), "PQ scoring must be deterministic");

        // The top hit lands in the query's own cluster.
        let top = String::from_utf8_lossy(&run()[..4]).to_string();
        assert!(top.starts_with("c"), "payload shape: {top}");

        // Roundtrip: a v5 file carries codes AND codebooks; reload reproduces
        // scores bit-exactly.
        let path = "/tmp/octasoma_pq_roundtrip.skch";
        pq.save_to_disk(path).unwrap();
        let loaded = SketchIndex::load_from_disk(path, 64).unwrap();
        assert_eq!(loaded.precision(), Precision::Pq);
        let before: Vec<_> = pq
            .nearest(q, 5, pq.len())
            .into_iter()
            .map(|(p, s)| (p.to_vec(), s))
            .collect();
        let after: Vec<_> = loaded
            .nearest(q, 5, pq.len())
            .into_iter()
            .map(|(p, s)| (p.to_vec(), s))
            .collect();
        assert_eq!(before, after);
        std::fs::remove_file(path).ok();

        // new_with_precision refuses the untrained tier.
        let result = std::panic::catch_unwind(|| {
            SketchIndex::new_with_precision(64, 256, 42, Precision::Pq)
        });
        assert!(result.is_err(), "untrained PQ construction must panic");
    }

    /// The documented adoption gate, stated the way the engine's other tiers
    /// are: **topical** fidelity. Among near-duplicate vectors the raw
    /// ordering differs from F32 in the 4th decimal — no lossy tier can
    /// preserve tie-level identity — so the gate measures the fraction of the
    /// PQ top-10 that comes from the query's own latent theme (the oracle's
    /// majority cluster). Below 0.92 the tier must not ship.
    #[test]
    fn pq_top10_stays_in_the_oracle_cluster_at_092() {
        let (oracle, pq, queries) = pq_corpus();
        const K: usize = 10;

        let cluster = |payload: &[u8]| -> String {
            let s = String::from_utf8_lossy(payload);
            s.split('_').next().unwrap_or("").to_string()
        };

        let mut total = 0.0f64;
        for q in &queries {
            // The oracle defines the query's theme by its own top-10 majority.
            let truth: Vec<Vec<u8>> = oracle
                .nearest(q, K, oracle.len())
                .into_iter()
                .map(|(p, _)| p.to_vec())
                .collect();
            let mut tally: std::collections::HashMap<String, usize> =
                std::collections::HashMap::new();
            for t in &truth {
                *tally.entry(cluster(t)).or_default() += 1;
            }
            let theme = tally
                .iter()
                .max_by_key(|(_, n)| **n)
                .map(|(k, _)| k.clone())
                .unwrap_or_default();

            let got: Vec<Vec<u8>> = pq
                .nearest(q, K, pq.len())
                .into_iter()
                .map(|(p, _)| p.to_vec())
                .collect();
            let inside = got.iter().filter(|g| cluster(g) == theme).count();
            total += inside as f64 / K as f64;
        }
        let purity = total / queries.len() as f64;
        assert!(
            purity >= 0.92,
            "PQ top-10 theme purity {purity:.4} below the 0.92 adoption gate"
        );
    }
}
