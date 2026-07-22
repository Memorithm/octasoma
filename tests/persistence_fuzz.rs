//! Persistence: round-trip fidelity across many random states, plus rejection
//! of corrupt / incompatible files.
mod common;
use common::*;
use octasoma::{DeterministicRng, FractalMemory3D};

fn tmp(label: &str, seed: u64) -> String {
    format!("/tmp/octasoma_{}_{}_{seed}.frac", label, std::process::id())
}

#[test]
fn roundtrip_fidelity_many_states() {
    for seed in 0..25u64 {
        let d = 3 + (seed as usize % 100);
        let mut mem = FractalMemory3D::new(d, seed);
        let mut rng = DeterministicRng::new(seed + 1);
        let n = (seed as usize % 1500) + 1;
        for i in 0..n {
            let payload = if i % 5 == 0 {
                Some(format!("record-{i}-{}", "z".repeat(i % 40)).into_bytes())
            } else {
                None
            };
            mem.insert(&rand_vec(&mut rng, d), payload.as_deref())
                .unwrap();
        }

        let path = tmp("fuzz", seed);
        mem.save_to_disk(&path).unwrap();
        let loaded = FractalMemory3D::load_from_disk(&path, d).unwrap();

        assert_eq!(loaded.item_count(), mem.item_count());
        assert_eq!(loaded.node_count(), mem.node_count());
        assert_eq!(loaded.payload_arena, mem.payload_arena);
        assert_eq!(loaded.projection_matrix, mem.projection_matrix);
        assert_eq!(loaded.world_half_size, mem.world_half_size);

        for _ in 0..30 {
            let q = rand_vec(&mut rng, d);
            assert_eq!(
                dists(&mem.nearest_embedding(&q, 5)),
                dists(&loaded.nearest_embedding(&q, 5))
            );
            assert_eq!(mem.query(&q), loaded.query(&q));
        }
        std::fs::remove_file(&path).ok();
    }
}

#[test]
fn rejects_corrupt_and_incompatible_files() {
    let mut mem = FractalMemory3D::new(8, 0);
    let mut rng = DeterministicRng::new(2);
    for i in 0..50 {
        mem.insert(&rand_vec(&mut rng, 8), Some(format!("{i}").as_bytes()))
            .unwrap();
    }
    let path = tmp("corrupt", 0);
    mem.save_to_disk(&path).unwrap();

    // Wrong expected dimension.
    assert!(FractalMemory3D::load_from_disk(&path, 9).is_err());

    let good = std::fs::read(&path).unwrap();

    // Truncated mid-stream.
    std::fs::write(&path, &good[..good.len() / 2]).unwrap();
    assert!(FractalMemory3D::load_from_disk(&path, 8).is_err());

    // Bad magic.
    std::fs::write(&path, b"NOPEnope").unwrap();
    assert!(FractalMemory3D::load_from_disk(&path, 8).is_err());

    // Empty file.
    std::fs::write(&path, b"").unwrap();
    assert!(FractalMemory3D::load_from_disk(&path, 8).is_err());

    // Bad version (flip the version u32 right after the 4-byte magic).
    let mut bad_version = good.clone();
    bad_version[4] = 0xFF;
    std::fs::write(&path, &bad_version).unwrap();
    assert!(FractalMemory3D::load_from_disk(&path, 8).is_err());

    std::fs::remove_file(&path).ok();
}

#[test]
fn empty_engine_roundtrips() {
    let mem = FractalMemory3D::new(5, 1);
    let path = tmp("empty", 0);
    mem.save_to_disk(&path).unwrap();
    let loaded = FractalMemory3D::load_from_disk(&path, 5).unwrap();
    assert_eq!(loaded.item_count(), 0);
    assert!(loaded.query(&[0.0; 5]).is_none());
    std::fs::remove_file(&path).ok();
}
