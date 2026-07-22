//! Determinism & reproducibility: identical inputs ⇒ identical state and answers.
mod common;
use common::*;
use octasoma::{DeterministicRng, FractalMemory3D};

#[test]
fn same_seed_same_layout_and_answers() {
    for &seed in &[0u64, 1, 42, 12_345, u64::MAX] {
        let d = 32;
        let mut a = FractalMemory3D::new(d, seed);
        let mut b = FractalMemory3D::new(d, seed);
        assert_eq!(a.projection_matrix, b.projection_matrix);

        let mut rng = DeterministicRng::new(seed ^ 7);
        let data: Vec<Vec<f32>> = (0..600).map(|_| rand_vec(&mut rng, d)).collect();
        for (i, v) in data.iter().enumerate() {
            a.insert(v, Some(format!("{i}").as_bytes())).unwrap();
            b.insert(v, Some(format!("{i}").as_bytes())).unwrap();
        }
        assert_eq!(a.node_count(), b.node_count());

        let mut q = DeterministicRng::new(seed.wrapping_add(99));
        for _ in 0..100 {
            let query = rand_vec(&mut q, d);
            assert_eq!(
                dists(&a.nearest_embedding(&query, 5)),
                dists(&b.nearest_embedding(&query, 5))
            );
            assert_eq!(a.query(&query), b.query(&query));
        }
    }
}

#[test]
fn pca_projection_is_reproducible() {
    let (n, d) = (80, 12);
    let mut rng = DeterministicRng::new(5);
    let data: Vec<f32> = (0..n * d).map(|_| rng.next_f32()).collect();
    let a = FractalMemory3D::new_with_pca(d, &data, n);
    let b = FractalMemory3D::new_with_pca(d, &data, n);
    assert_eq!(a.projection_matrix, b.projection_matrix);
    assert!(a.projection_matrix.iter().all(|x| x.is_finite()));
}

#[test]
fn rebuild_after_growth_is_order_consistent() {
    // Inserting the same vectors yields the same tree regardless of the world
    // growth that happens along the way.
    let d = 8;
    let mut rng = DeterministicRng::new(123);
    let data: Vec<Vec<f32>> = (0..400usize)
        .map(|i| {
            let scale = if i % 5 == 0 { 1000.0 } else { 1.0 };
            rand_vec(&mut rng, d).iter().map(|x| x * scale).collect()
        })
        .collect();

    let build = || {
        let mut m = FractalMemory3D::new(d, 77);
        for v in &data {
            m.insert(v, None).unwrap();
        }
        m
    };
    let a = build();
    let b = build();
    assert_eq!(a.node_count(), b.node_count());
    assert_eq!(a.world_half_size, b.world_half_size);
    let mut q = DeterministicRng::new(9);
    for _ in 0..50 {
        let query = rand_vec(&mut q, d);
        assert_eq!(
            dists(&a.nearest_embedding(&query, 8)),
            dists(&b.nearest_embedding(&query, 8))
        );
    }
}
