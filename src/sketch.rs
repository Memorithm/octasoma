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

#[cfg(test)]
mod tests {
    use super::*;

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
}
