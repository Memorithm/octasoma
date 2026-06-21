//! Structural & completeness invariants of the octree.
mod common;
use common::*;
use octasoma::{DeterministicRng, FractalMemory3D, NONE};

#[test]
fn every_item_is_retrievable_at_zero_distance() {
    for seed in 0..20u64 {
        let d = 4 + (seed as usize % 60);
        let mut mem = FractalMemory3D::new(d, seed);
        let mut rng = DeterministicRng::new(seed + 7);
        let n = 200 + (seed as usize * 17);
        for i in 0..n {
            mem.insert(&rand_vec(&mut rng, d), Some(format!("{i}").as_bytes()))
                .unwrap();
        }
        for i in 0..mem.item_count() {
            let p = mem.items[i].point;
            let nn = mem.nearest(p, 1);
            assert!(!nn.is_empty(), "item {i} unreachable");
            assert_eq!(nn[0].1, 0.0, "seed={seed} item {i} not at distance 0");
        }
    }
}

#[test]
fn buckets_are_a_permutation_of_all_items() {
    let mut mem = FractalMemory3D::new(16, 3);
    let mut rng = DeterministicRng::new(9);
    for i in 0..6000 {
        mem.insert(&rand_vec(&mut rng, 16), Some(format!("{i}").as_bytes()))
            .unwrap();
    }

    // Every id appears exactly once across all leaf buckets.
    let mut all: Vec<u32> = mem.leaf_buckets.iter().flatten().copied().collect();
    all.sort_unstable();
    let expected: Vec<u32> = (0..mem.item_count() as u32).collect();
    assert_eq!(all, expected, "items lost, duplicated, or stranded");
}

#[test]
fn node_links_and_kinds_are_consistent() {
    let mut mem = FractalMemory3D::new(12, 5);
    let mut rng = DeterministicRng::new(11);
    for i in 0..5000 {
        mem.insert(&rand_vec(&mut rng, 12), Some(format!("{i}").as_bytes()))
            .unwrap();
    }

    for (idx, node) in mem.nodes.iter().enumerate() {
        // Child indices are valid or NONE.
        for &c in &node.children {
            if c != NONE {
                assert!((c as usize) < mem.nodes.len(), "node {idx} dangling child");
            }
        }
        if node.is_leaf() {
            assert!(
                (node.bucket_id as usize) < mem.leaf_buckets.len(),
                "leaf {idx} bad bucket_id"
            );
        } else {
            let kids = node.children.iter().filter(|&&c| c != NONE).count();
            assert!(kids >= 1, "internal node {idx} has no children");
        }
    }
}
