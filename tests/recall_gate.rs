//! CI recall-regression gate — proposal D1 of `docs/scirust-improvements.md`.
//!
//! A deterministic clustered corpus (raw vectors, seeded, no embedder) is
//! indexed in a `HybridMemory`, held-out queries are answered by every
//! [`QueryStrategy`], and mean Recall@10 against the exact full-corpus rerank
//! is pinned with a floor. Everything is seeded and single-threaded, so a
//! failure here is a real recall regression (or a deliberate change that must
//! update the floors consciously), never flakiness.

use octasoma::{HybridMemory, QueryStrategy, metrics};
use std::collections::HashSet;

const DIM: usize = 64;
const CLUSTERS: usize = 200;
const PER_CLUSTER: usize = 30; // 6000 items — well beyond the 256 default shortlist,
// so the coarse 3-D route can no longer hide behind an exhaustive-ish candidate set
const K: usize = 10;

/// Deterministic clustered corpus + held-out queries (one per cluster ×4).
fn build() -> (HybridMemory, Vec<Vec<f32>>) {
    let mut mem = HybridMemory::new(DIM, 256, 42);
    let mut queries = Vec::new();
    for c in 0..CLUSTERS {
        let base: Vec<f32> = (0..DIM)
            .map(|d| ((c * DIM + d) as f32 * 0.61).sin())
            .collect();
        for j in 0..PER_CLUSTER {
            let item: Vec<f32> = base
                .iter()
                .enumerate()
                .map(|(d, x)| x + 0.35 * ((j * DIM + d) as f32 * 1.37).cos())
                .collect();
            let id = (c * PER_CLUSTER + j) as u64;
            assert!(mem.insert(&item, &id.to_le_bytes()));
            if j % 15 == 0 {
                queries.push(
                    item.iter()
                        .enumerate()
                        .map(|(d, x)| x + 0.05 * ((j + d) as f32 * 0.7).sin())
                        .collect(),
                );
            }
        }
    }
    (mem, queries)
}

fn ids(results: Vec<(&[u8], f32)>) -> Vec<u64> {
    results
        .into_iter()
        .map(|(p, _)| u64::from_le_bytes(p.try_into().expect("8-byte id payloads")))
        .collect()
}

#[test]
fn recall_floors_per_strategy() {
    let (mem, queries) = build();
    let n = CLUSTERS * PER_CLUSTER;

    let mean_recall = |strategy: QueryStrategy| -> f64 {
        let mut total = 0.0;
        for q in &queries {
            // Oracle: the exact full-corpus rerank (shortlist = N is exhaustive).
            let oracle: HashSet<u64> = ids(mem.recall(q, K, n)).into_iter().collect();
            let got = ids(mem.query(q, strategy, K));
            total += metrics::recall_at_k(&got, &oracle, K);
        }
        total / queries.len() as f64
    };

    let precision = mean_recall(QueryStrategy::PrecisionSketch);
    let cascade = mean_recall(QueryStrategy::HybridCascade);
    let spatial = mean_recall(QueryStrategy::FastSpatial);
    println!(
        "mean recall@{K}: PrecisionSketch={precision:.4} HybridCascade={cascade:.4} \
         FastSpatial={spatial:.4}"
    );

    // On clustered data every strategy saturates (a query's true top-10 is its
    // own cluster, and any 256-candidate route captures the home cluster), which
    // makes these floors a sharp regression tripwire: a broken shortlist,
    // sketch, or rerank shows up as a drop from ~1.0 immediately. Measured
    // 1.0000 / 1.0000 / 1.0000 on the current implementation.
    assert!(
        precision >= 0.98,
        "PrecisionSketch recall@{K} = {precision}"
    );
    assert!(cascade >= 0.98, "HybridCascade recall@{K} = {cascade}");
    assert!(spatial >= 0.98, "FastSpatial recall@{K} = {spatial}");
}

/// On **unstructured** vectors — where the 3-D projection's documented weakness
/// lives — the SimHash precision tier must strictly beat the coarse 3-D route at
/// exact recall@1. This is the engine's core precision claim, now CI-enforced
/// with standard metrics.
#[test]
fn sketch_tier_beats_the_router_on_unstructured_vectors() {
    const N: usize = 4000;
    const Q: usize = 60;

    // Deterministic pseudo-random unit-ish vectors (seeded LCG, no structure).
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };
    let items: Vec<Vec<f32>> = (0..N).map(|_| (0..DIM).map(|_| next()).collect()).collect();

    let mut mem = HybridMemory::new(DIM, 256, 42);
    for (i, item) in items.iter().enumerate() {
        assert!(mem.insert(item, &(i as u64).to_le_bytes()));
    }

    // Fresh random queries — *not* perturbed copies. A near-duplicate query is
    // trivially easy for any locality-preserving route (a linear projection
    // preserves ε-closeness); the documented 3-D weakness is fine-margin
    // ranking, where the true nearest neighbour of an unrelated query wins by a
    // sliver that three coordinates cannot represent.
    let queries: Vec<Vec<f32>> = (0..Q).map(|_| (0..DIM).map(|_| next()).collect()).collect();
    let (mut sketch_hits, mut spatial_hits) = (0usize, 0usize);
    for query in &queries {
        let oracle = ids(mem.recall(query, 1, N))[0];
        if ids(mem.query(query, QueryStrategy::PrecisionSketch, 1))[0] == oracle {
            sketch_hits += 1;
        }
        if ids(mem.query(query, QueryStrategy::FastSpatial, 1))[0] == oracle {
            spatial_hits += 1;
        }
    }
    let sketch = sketch_hits as f64 / Q as f64;
    let spatial = spatial_hits as f64 / Q as f64;
    println!("unstructured recall@1: PrecisionSketch={sketch:.3} FastSpatial={spatial:.3}");

    // Floors pinned below the measured values — 0.450 vs 0.100 on the current
    // implementation (deterministic — no flakiness).
    assert!(sketch >= 0.40, "PrecisionSketch recall@1 = {sketch}");
    assert!(
        sketch >= spatial + 0.20,
        "the precision tier must clearly beat the 3-D router: \
         sketch={sketch} spatial={spatial}"
    );
}
