//! Oracle test: interleave inserts and queries, and after every query assert the
//! incrementally-maintained octree (with subdivision and world growth) still
//! matches the brute-force ground truth exactly.
mod common;
use common::*;
use octasoma::{DeterministicRng, FractalMemory3D};

#[test]
fn interleaved_inserts_and_queries_stay_exact() {
    for seed in 0..25u64 {
        let d = 3 + (seed as usize % 80);
        let mut mem = FractalMemory3D::new(d, seed.wrapping_mul(0x0100_0000_01b3).wrapping_add(1));
        let mut rng = DeterministicRng::new(seed + 1);

        for step in 0..1500usize {
            if step.is_multiple_of(3) && mem.item_count() > 0 {
                let p = mem.project(&rand_vec(&mut rng, d)).unwrap();
                let k = 1 + (rng.next_u64() % 8) as usize;
                assert_eq!(
                    dists(&mem.nearest(p, k)),
                    dists(&mem.nearest_bruteforce(p, k)),
                    "seed={seed} step={step}"
                );
            } else {
                // Mix unit-norm vectors with occasional large ones to exercise
                // world growth in the middle of a live session.
                let v = if rng.next_u64().is_multiple_of(11) {
                    rand_vec(&mut rng, d).iter().map(|x| x * 500.0).collect()
                } else {
                    rand_unit(&mut rng, d)
                };
                mem.insert(&v, None).unwrap();
            }
        }
    }
}
