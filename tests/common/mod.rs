//! Shared helpers for the integration test suite.
#![allow(dead_code)]

use octasoma::{DeterministicRng, ItemId};

/// A random vector in `[-1, 1]^d`.
pub fn rand_vec(rng: &mut DeterministicRng, d: usize) -> Vec<f32> {
    (0..d).map(|_| rng.next_f32()).collect()
}

/// A random L2-normalised vector (typical of real embeddings).
pub fn rand_unit(rng: &mut DeterministicRng, d: usize) -> Vec<f32> {
    let mut v = rand_vec(rng, d);
    l2(&mut v);
    v
}

/// A sample near `center` (centre + noise), renormalised.
pub fn near(center: &[f32], spread: f32, rng: &mut DeterministicRng) -> Vec<f32> {
    let mut v: Vec<f32> = center
        .iter()
        .map(|&c| c + spread * rng.next_f32())
        .collect();
    l2(&mut v);
    v
}

pub fn l2(v: &mut [f32]) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-12 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// The distance column of a k-NN result — the tie-robust thing to compare,
/// since two exact searches must agree on distances even if they break ties
/// between equidistant items differently.
pub fn dists(r: &[(ItemId, f32)]) -> Vec<f32> {
    r.iter().map(|x| x.1).collect()
}
