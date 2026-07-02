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

/// Full cosine similarity between two equal-length vectors (`0` if either is zero).
fn cosine_full(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
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
#[derive(Clone, Debug)]
pub struct SketchIndex {
    hasher: SimHasher,
    dim: usize,
    /// Seed used to (re)generate the hasher's hyperplanes — stored so the index
    /// reloads without serialising the planes.
    seed: u64,
    /// `count × dim` flat row-major embeddings (for the exact rerank).
    embeddings: Vec<f32>,
    /// `count × words` flat sketches (for the Hamming shortlist).
    sketches: Vec<u64>,
    /// Payload arena and per-item `(offset, len)`.
    payloads: Vec<u8>,
    offsets: Vec<(usize, usize)>,
}

impl SketchIndex {
    /// Creates an empty index for `dim`-dimensional embeddings, sketched with `bits`
    /// random hyperplanes (seeded).
    pub fn new(dim: usize, bits: usize, seed: u64) -> Self {
        let hasher = SimHasher::new(dim, bits, seed);
        Self {
            hasher,
            dim,
            seed,
            embeddings: Vec::new(),
            sketches: Vec::new(),
            payloads: Vec::new(),
            offsets: Vec::new(),
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
    /// nothing) if `embedding.len() != dim`.
    pub fn insert(&mut self, embedding: &[f32], payload: &[u8]) -> bool {
        if embedding.len() != self.dim {
            return false;
        }
        self.embeddings.extend_from_slice(embedding);
        self.sketches
            .extend_from_slice(&self.hasher.sketch(embedding));
        let off = self.payloads.len();
        self.payloads.extend_from_slice(payload);
        self.offsets.push((off, payload.len()));
        true
    }

    fn payload(&self, i: usize) -> &[u8] {
        let (off, len) = self.offsets[i];
        &self.payloads[off..off + len]
    }

    fn embedding(&self, i: usize) -> &[f32] {
        &self.embeddings[i * self.dim..(i + 1) * self.dim]
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

    /// Core of [`SketchIndex::nearest`], on item ids (insertion order) instead of
    /// payloads — also the exact pipeline [`SketchIndex::certify_shortlist`] measures.
    fn nearest_ids(&self, query: &[f32], k: usize, shortlist: usize) -> Vec<(usize, f32)> {
        if query.len() != self.dim || k == 0 || self.is_empty() {
            return Vec::new();
        }
        let qs = self.hasher.sketch(query);
        let m = shortlist.max(k).min(self.len());

        // 1. Hamming shortlist of size m.
        let mut cand: Vec<(u32, usize)> = (0..self.len())
            .map(|i| (hamming(&qs, self.sketch_of(i)), i))
            .collect();
        if cand.len() > m {
            cand.select_nth_unstable_by_key(m - 1, |(h, _)| *h);
            cand.truncate(m);
        }

        // 2. Exact cosine rerank of the shortlist.
        let mut scored: Vec<(f32, usize)> = cand
            .iter()
            .map(|&(_, i)| (cosine_full(self.embedding(i), query), i))
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored.truncate(k);
        scored.into_iter().map(|(s, i)| (i, s)).collect()
    }

    /// The exact top-`k` item ids by full cosine over the whole corpus — the oracle
    /// [`SketchIndex::certify_shortlist`] measures against. Ties break toward the
    /// smaller id (deterministic; a tie-swap can only make the certificate
    /// pessimistic, never invalid).
    fn exact_top_ids(&self, query: &[f32], k: usize) -> Vec<usize> {
        let mut scored: Vec<(f32, usize)> = (0..self.len())
            .map(|i| (cosine_full(self.embedding(i), query), i))
            .collect();
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
        let valid: Vec<&Vec<f32>> = queries.iter().filter(|q| q.len() == self.dim).collect();
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

    /// The `k` nearest payloads by **Hamming only** (no exact rerank): cheaper and
    /// memory could be sketch-only, but approximate. Returns `(payload, hamming)`
    /// ascending (smaller is closer).
    pub fn nearest_sketch(&self, query: &[f32], k: usize) -> Vec<(&[u8], u32)> {
        if query.len() != self.dim || k == 0 || self.is_empty() {
            return Vec::new();
        }
        let qs = self.hasher.sketch(query);
        let mut cand: Vec<(u32, usize)> = (0..self.len())
            .map(|i| (hamming(&qs, self.sketch_of(i)), i))
            .collect();
        let k = k.min(cand.len());
        cand.select_nth_unstable_by_key(k - 1, |(h, _)| *h);
        cand.truncate(k);
        cand.sort_by_key(|(h, _)| *h);
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
        let mut scored: Vec<(f32, usize)> = ids
            .iter()
            .filter_map(|&id| {
                let i = id as usize;
                (i < self.len()).then(|| (cosine_full(self.embedding(i), query), i))
            })
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
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
        let qs = self.hasher.sketch(query);
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
        cand.select_nth_unstable_by_key(m - 1, |(h, _)| *h);
        cand.truncate(m);
        cand.into_iter().map(|(_, id)| id).collect()
    }

    /// Exact cosine similarity of **every** item to `query`, in id (insertion)
    /// order. Empty on a dimension mismatch. Useful to colour a 3-D view by
    /// precision score.
    pub fn scores(&self, query: &[f32]) -> Vec<f32> {
        if query.len() != self.dim {
            return Vec::new();
        }
        (0..self.len())
            .map(|i| cosine_full(self.embedding(i), query))
            .collect()
    }

    // -- persistence ---------------------------------------------------------

    /// Serialises the index to a versioned `SKCH` file (little-endian; the payload
    /// arena is LZ4-compressed). The hyperplanes are *not* stored — they are
    /// regenerated from the seed on load — so the file is `count·(dim·4 + bits/8)`
    /// bytes plus payloads.
    pub fn save_to_disk(&self, path: &str) -> io::Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(b"SKCH")?;
        w.write_all(&1u32.to_le_bytes())?; // version
        w.write_all(&(self.dim as u32).to_le_bytes())?;
        w.write_all(&(self.hasher.bits() as u32).to_le_bytes())?;
        w.write_all(&self.seed.to_le_bytes())?;
        w.write_all(&(self.len() as u64).to_le_bytes())?;

        for &f in &self.embeddings {
            w.write_all(&f.to_le_bytes())?;
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
        if version != 1 {
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
        let count = read_u64(&mut r)? as usize;
        let words = bits / 64;

        let mut embeddings = vec![0f32; count * dim];
        for e in embeddings.iter_mut() {
            *e = read_f32(&mut r)?;
        }
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
        let mut comp = vec![0u8; comp_len];
        r.read_exact(&mut comp)?;
        let payloads = lz4_flex::decompress(&comp, decomp_len)
            .map_err(|e| invalid(&format!("lz4 decompression failed: {e}")))?;

        Ok(Self {
            hasher: SimHasher::new(dim, bits, seed),
            dim,
            seed,
            embeddings,
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
}
