//! Degenerate and adversarial inputs must be handled without panicking.
mod common;
use common::*;
use octasoma::{DeterministicRng, FractalMemory3D};

#[test]
fn empty_engine_queries() {
    let m = FractalMemory3D::new(8, 0);
    assert_eq!(m.item_count(), 0);
    assert!(m.query(&[0.0; 8]).is_none());
    assert!(m.nearest_embedding(&[0.0; 8], 5).is_empty());
    assert!(m.nearest([0.0, 0.0, 0.0], 5).is_empty());
    assert!(m.query_k(&[0.0; 8], 3).is_empty());
}

#[test]
fn k_zero_and_k_over_n() {
    let mut m = FractalMemory3D::new(4, 1);
    let mut rng = DeterministicRng::new(3);
    for i in 0..10 {
        m.insert(&rand_vec(&mut rng, 4), Some(format!("{i}").as_bytes()))
            .unwrap();
    }
    // k = 0 → empty.
    assert!(m.nearest([0.0, 0.0, 0.0], 0).is_empty());
    assert!(m.nearest_embedding(&rand_vec(&mut rng, 4), 0).is_empty());
    assert!(m.nearest_bruteforce([0.1, 0.2, 0.3], 0).is_empty());
    // k > N → all N, and equal to brute force.
    let p = m.project(&rand_vec(&mut rng, 4)).unwrap();
    let r = m.nearest(p, 100);
    assert_eq!(r.len(), 10);
    assert_eq!(dists(&r), dists(&m.nearest_bruteforce(p, 100)));
}

#[test]
fn dimension_mismatch_is_rejected() {
    let mut m = FractalMemory3D::new(8, 0);
    assert!(m.insert(&[0.0; 4], None).is_none());
    assert!(m.insert(&[0.0; 16], Some(b"x")).is_none());
    assert_eq!(m.item_count(), 0);
    assert!(m.query(&[0.0; 4]).is_none());
    assert!(m.nearest_embedding(&[0.0; 7], 3).is_empty());
}

#[test]
fn nan_and_inf_embeddings_do_not_panic() {
    let mut m = FractalMemory3D::new(4, 0);
    // Non-finite embeddings are rejected on insert (not stored).
    assert!(m.insert(&[f32::NAN, 0.0, 0.0, 0.0], Some(b"nan")).is_none());
    assert!(m.insert(&[f32::INFINITY, 0.0, 0.0, 0.0], None).is_none());
    assert!(
        m.insert(&[0.0, f32::NEG_INFINITY, 0.0, 0.0], None)
            .is_none()
    );
    assert_eq!(m.item_count(), 0);

    // A valid item, then non-finite *queries* must return empty, never panic.
    m.insert(&[0.1, 0.2, 0.3, 0.4], Some(b"ok")).unwrap();
    assert!(m.query(&[f32::NAN, 0.0, 0.0, 0.0]).is_none());
    assert!(m.nearest_embedding(&[f32::NAN; 4], 3).is_empty());
    assert!(m.nearest([f32::NAN, 0.0, 0.0], 1).is_empty());
    assert!(m.nearest([f32::INFINITY, 0.0, 0.0], 5).is_empty());
}

#[test]
fn many_identical_and_zero_vectors() {
    let mut m = FractalMemory3D::new(4, 2);
    for i in 0..2000 {
        m.insert(&[0.0; 4], Some(format!("{i}").as_bytes()))
            .unwrap();
    }
    assert_eq!(m.item_count(), 2000);
    let r = m.nearest_embedding(&[0.0; 4], 2000);
    assert_eq!(r.len(), 2000);
    assert!(r.iter().all(|&(_, dist)| dist == 0.0));
}

#[test]
fn extreme_world_growth_preserves_exactness() {
    let mut m = FractalMemory3D::new(3, 0);
    m.insert(&[1e6, -1e6, 1e6], Some(b"huge")).unwrap();
    m.insert(&[1e-6, 1e-6, 1e-6], Some(b"tiny")).unwrap();
    m.insert(&[-3.2e5, 7.7e5, -1.0e4], Some(b"mid")).unwrap();
    assert!(m.world_half_size >= 1e6);
    assert_eq!(m.query(&[1e6, -1e6, 1e6]).unwrap(), b"huge");

    let p = m.project(&[1e6, -1e6, 1e6]).unwrap();
    assert_eq!(dists(&m.nearest(p, 3)), dists(&m.nearest_bruteforce(p, 3)));
}

#[test]
fn empty_and_large_payloads() {
    let mut m = FractalMemory3D::new(4, 0);
    let id0 = m.insert(&[0.1, 0.2, 0.3, 0.4], Some(b"")).unwrap();
    assert_eq!(m.get_payload(id0).unwrap(), b"");

    let big = vec![0x5Au8; 1 << 20]; // 1 MiB
    let id1 = m.insert(&[0.5, 0.6, 0.7, 0.8], Some(&big)).unwrap();
    assert_eq!(m.get_payload(id1).unwrap().len(), 1 << 20);
    assert_eq!(m.get_payload(id1).unwrap(), &big[..]);

    // Out-of-range ids are rejected.
    assert!(m.get_payload(octasoma::NONE).is_none());
    assert!(m.get_payload(9_999).is_none());
}

#[test]
fn single_item_engine() {
    let mut m = FractalMemory3D::new(6, 1);
    m.insert(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], Some(b"only"))
        .unwrap();
    assert_eq!(m.query(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap(), b"only");
    assert_eq!(m.nearest_embedding(&[0.0; 6], 10).len(), 1);
}
