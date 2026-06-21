//! Contract tests for the CCOS adapter (`integration/ccos/octa_index.rs`).
//!
//! The adapter is the file CCOS vendors; it is otherwise only compile-checked by
//! `examples/ccos_bridge.rs`. These tests pin its behaviour in CI: region derivation,
//! global anchors, and region-scoped recall — all offline with `HashEmbedder`.

// Some adapter methods are part of the CCOS-facing API but unused here.
#[allow(dead_code)]
#[path = "../integration/ccos/octa_index.rs"]
mod octa_index;

use octa_index::{OctaIndex, ShardedOctaIndex, region_of};
use octasoma::HashEmbedder;

#[test]
fn region_of_derives_the_causal_file() {
    assert_eq!(region_of("sym:src/db.rs:query"), "src/db.rs");
    assert_eq!(region_of("mod:src/cache.rs"), "src/cache.rs");
    assert_eq!(region_of("file:src/main.rs"), "src/main.rs");
    // No recognisable scheme → the whole string is the region.
    assert_eq!(region_of("noscheme"), "noscheme");
}

#[test]
fn global_octaindex_returns_scored_anchors() {
    let mut idx = OctaIndex::new(HashEmbedder::new(128));
    idx.index_node("sym:src/db.rs:query", "build and run SQL queries");
    idx.index_node("sym:src/auth.rs:login", "authenticate a user");

    let anchors = idx.semantic_anchors("build and run SQL queries", 1);
    assert_eq!(anchors.len(), 1);
    assert_eq!(anchors[0].0, "sym:src/db.rs:query");
    // Exact-text hit → distance ~0 → score ~1.
    assert!(
        anchors[0].1 > 0.99,
        "exact hit should score ~1: {}",
        anchors[0].1
    );
}

#[test]
fn sharded_adapter_scopes_recall_to_a_region() {
    let mut idx = ShardedOctaIndex::new(HashEmbedder::new(128));
    idx.index_node("sym:src/db.rs:query", "build and run SQL queries");
    idx.index_node("sym:src/db.rs:pool", "a pool of db connections");
    idx.index_node("sym:src/auth.rs:login", "authenticate a user");

    assert_eq!(idx.regions(), 2);
    assert_eq!(idx.len(), 3);

    // Scoped anchors recall the right uri and never leave the region.
    let hits = idx.semantic_anchors_in("src/db.rs", "a pool of db connections", 2);
    assert_eq!(hits[0].0, "sym:src/db.rs:pool");
    assert!(hits.iter().all(|(u, _)| u.starts_with("sym:src/db.rs:")));

    // The auth region cannot see db nodes (causal scoping).
    let auth = idx.semantic_anchors_in("src/auth.rs", "a pool of db connections", 5);
    assert!(auth.iter().all(|(u, _)| !u.starts_with("sym:src/db.rs:")));

    // An unknown region yields nothing.
    assert!(
        idx.semantic_anchors_in("src/missing.rs", "anything", 3)
            .is_empty()
    );
}
