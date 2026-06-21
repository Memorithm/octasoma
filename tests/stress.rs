//! Heavy soak tests. These are `#[ignore]`d so the normal suite stays fast;
//! run them explicitly with:
//!
//! ```bash
//! cargo test --release --test stress -- --ignored --nocapture
//! ```
mod common;
use common::*;
use octasoma::{DeterministicRng, FractalMemory3D};

#[test]
#[ignore]
fn soak_one_million_inserts_stay_exact_and_persist() {
    let d = 128;
    let n = 1_000_000;
    let mut mem = FractalMemory3D::new(d, 7);
    let mut rng = DeterministicRng::new(1);

    let t = std::time::Instant::now();
    for i in 0..n {
        mem.insert(&rand_unit(&mut rng, d), Some(&(i as u32).to_le_bytes()))
            .unwrap();
    }
    eprintln!(
        "inserted {n} items in {:.2?} ({:.0}/s), {} nodes",
        t.elapsed(),
        n as f64 / t.elapsed().as_secs_f64(),
        mem.node_count()
    );
    assert_eq!(mem.item_count(), n);

    // Completeness: buckets are a permutation of all item ids.
    let total: usize = mem.leaf_buckets.iter().map(|b| b.len()).sum();
    assert_eq!(total, n);

    // Exactness on a sample of queries.
    for _ in 0..50 {
        let p = mem.project(&rand_unit(&mut rng, d)).unwrap();
        assert_eq!(
            dists(&mem.nearest(p, 10)),
            dists(&mem.nearest_bruteforce(p, 10))
        );
    }

    // Large-store persistence round-trip.
    let path = format!("/tmp/octasoma_soak_{}.frac", std::process::id());
    mem.save_to_disk(&path).unwrap();
    let loaded = FractalMemory3D::load_from_disk(&path, d).unwrap();
    assert_eq!(loaded.item_count(), n);
    let p = mem.project(&rand_unit(&mut rng, d)).unwrap();
    assert_eq!(dists(&loaded.nearest(p, 10)), dists(&mem.nearest(p, 10)));
    std::fs::remove_file(&path).ok();
}

#[test]
#[ignore]
fn soak_high_churn_growth() {
    // Many inserts spanning a huge dynamic range to hammer world growth/rebuild.
    let d = 64;
    let mut mem = FractalMemory3D::new(d, 3);
    let mut rng = DeterministicRng::new(2);
    for i in 0..200_000 {
        let scale = 10f32.powi((i % 12) - 6); // 1e-6 .. 1e5
        let v: Vec<f32> = rand_vec(&mut rng, d).iter().map(|x| x * scale).collect();
        mem.insert(&v, None).unwrap();
    }
    assert_eq!(mem.item_count(), 200_000);
    for _ in 0..40 {
        let p = mem.project(&rand_vec(&mut rng, d)).unwrap();
        assert_eq!(
            dists(&mem.nearest(p, 5)),
            dists(&mem.nearest_bruteforce(p, 5))
        );
    }
}
