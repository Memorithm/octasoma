//! Property test: the octree's exact k-NN must equal brute force, always.
mod common;
use common::*;
use octasoma::{DeterministicRng, FractalMemory3D};

#[test]
fn knn_equals_bruteforce_across_random_configs() {
    let mut meta = DeterministicRng::new(0xBEEF_CAFE);
    for case in 0..30 {
        let d = 2 + (meta.next_u64() % 250) as usize;
        let n = 64 + (meta.next_u64() % 2000) as usize;
        let k = 1 + (meta.next_u64() % 20) as usize;
        let seed = meta.next_u64();

        let mut mem = FractalMemory3D::new(d, seed);
        let mut rng = DeterministicRng::new(seed ^ 0x9E37_79B9);
        for i in 0..n {
            mem.insert(&rand_vec(&mut rng, d), Some(format!("{i}").as_bytes()))
                .unwrap();
        }
        for _ in 0..25 {
            let q = rand_vec(&mut rng, d);
            let p = mem.project(&q).unwrap();
            assert_eq!(
                dists(&mem.nearest(p, k)),
                dists(&mem.nearest_bruteforce(p, k)),
                "case {case}: d={d} n={n} k={k} seed={seed}"
            );
        }
    }
}

#[test]
fn knn_exact_on_clustered_and_normalised_data() {
    let mut meta = DeterministicRng::new(0x1234_5678);
    for _ in 0..15 {
        let d = 16 + (meta.next_u64() % 200) as usize;
        let clusters = 1 + (meta.next_u64() % 40) as usize;
        let seed = meta.next_u64();
        let mut rng = DeterministicRng::new(seed);
        let centers: Vec<Vec<f32>> = (0..clusters).map(|_| rand_unit(&mut rng, d)).collect();

        let mut mem = FractalMemory3D::new(d, seed ^ 1);
        for i in 0..1500 {
            let v = near(&centers[i % clusters], 0.3, &mut rng);
            mem.insert(&v, None).unwrap();
        }
        for _ in 0..20 {
            let q = near(
                &centers[(rng.next_u64() as usize) % clusters],
                0.3,
                &mut rng,
            );
            let p = mem.project(&q).unwrap();
            assert_eq!(
                dists(&mem.nearest(p, 10)),
                dists(&mem.nearest_bruteforce(p, 10))
            );
        }
    }
}

#[test]
fn knn_exact_under_varied_bucket_and_min_size() {
    for &cap in &[1usize, 2, 4, 16, 64, 256] {
        let d = 24;
        let mut mem = FractalMemory3D::new(d, 99);
        mem.bucket_capacity = cap;
        mem.min_half_size = 1e-4;
        let mut rng = DeterministicRng::new(cap as u64 + 7);
        for i in 0..3000 {
            mem.insert(&rand_vec(&mut rng, d), Some(format!("{i}").as_bytes()))
                .unwrap();
        }
        for _ in 0..40 {
            let p = mem.project(&rand_vec(&mut rng, d)).unwrap();
            assert_eq!(
                dists(&mem.nearest(p, 7)),
                dists(&mem.nearest_bruteforce(p, 7)),
                "bucket_capacity={cap}"
            );
        }
    }
}
