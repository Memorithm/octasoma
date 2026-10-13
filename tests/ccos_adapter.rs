//! Contract tests for the CCOS adapter (`integration/ccos/octa_index.rs`).
//!
//! The adapter is a file CCOS may vendor; it is otherwise only compile-checked by
//! `examples/ccos_bridge.rs`. These tests pin its behaviour in CI: region derivation,
//! global anchors, and region-scoped recall — all offline with `HashEmbedder`.

// Some adapter methods are part of the CCOS-facing API but unused here.
#[allow(dead_code)]
#[path = "../integration/ccos/octa_index.rs"]
mod octa_index;

use octa_index::{IndexBudget, IndexErrorKind, OctaIndex, ShardedOctaIndex, region_of};
use octasoma::{EmbedError, Embedder, HashEmbedder, ShardedMemory};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[test]
fn region_of_derives_the_causal_file() {
    assert_eq!(region_of("sym:src/db.rs:query"), "src/db.rs");
    assert_eq!(region_of("mod:src/cache.rs"), "src/cache.rs");
    assert_eq!(region_of("file:src/main.rs"), "src/main.rs");
    // No recognisable scheme → the whole string is the region.
    assert_eq!(region_of("noscheme"), "noscheme");
}

struct ControlledEmbedder {
    calls: Arc<AtomicUsize>,
}

impl Embedder for ControlledEmbedder {
    fn dim(&self) -> usize {
        4
    }
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match text {
            "offline" => Err(EmbedError::Io(std::io::Error::new(
                std::io::ErrorKind::ConnectionRefused,
                "offline fixture",
            ))),
            "bad-dimension" => Ok(vec![1.0]),
            "nan" => Ok(vec![f32::NAN; 4]),
            _ => Ok(vec![0.5; 4]),
        }
    }
}

#[test]
fn transport_failure_is_not_an_empty_recall_or_successful_insert() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut idx = ShardedOctaIndex::new(ControlledEmbedder { calls });
    let error = idx
        .index_node_in("src/db.rs", "sym:src/db.rs:query", "offline")
        .unwrap_err();
    assert!(matches!(error.kind, IndexErrorKind::EmbedderUnavailable(_)));
    assert_eq!(error.operation, "insert");
    assert_eq!(error.region.as_deref(), Some("src/db.rs"));
    assert_eq!(error.uri.as_deref(), Some("sym:src/db.rs:query"));
    assert!(idx.is_empty());
    assert_eq!(idx.regions(), 0);
    idx.index_node_in("src/db.rs", "sym:src/db.rs:query", "good")
        .unwrap();
    assert!(matches!(
        idx.semantic_anchors_in("src/db.rs", "offline", 1)
            .unwrap_err()
            .kind,
        IndexErrorKind::EmbedderUnavailable(_)
    ));
    assert!(matches!(
        idx.semantic_anchors("offline", 1).unwrap_err().kind,
        IndexErrorKind::EmbedderUnavailable(_)
    ));
    // Unknown region is a genuine miss and does not require an embedder call.
    assert!(
        idx.semantic_anchors_in("missing", "offline", 1)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn global_adapter_preserves_transport_and_vector_format_errors() {
    let mut idx = OctaIndex::new(ControlledEmbedder {
        calls: Arc::new(AtomicUsize::new(0)),
    });
    assert!(matches!(
        idx.index_node("file:a", "offline").unwrap_err().kind,
        IndexErrorKind::EmbedderUnavailable(_)
    ));
    for text in ["bad-dimension", "nan"] {
        assert!(matches!(
            idx.index_node("file:a", text).unwrap_err().kind,
            IndexErrorKind::InvalidFormat(_)
        ));
    }
    assert!(idx.is_empty());
    idx.index_node("file:a", "good").unwrap();
    assert!(matches!(
        idx.semantic_anchors("offline", 1).unwrap_err().kind,
        IndexErrorKind::EmbedderUnavailable(_)
    ));
    assert!(matches!(
        idx.semantic_anchors("nan", 1).unwrap_err().kind,
        IndexErrorKind::InvalidFormat(_)
    ));
}

#[test]
fn sharded_bad_embeddings_do_not_create_phantom_regions() {
    let mut idx = ShardedOctaIndex::new(ControlledEmbedder {
        calls: Arc::new(AtomicUsize::new(0)),
    });
    for text in ["bad-dimension", "nan"] {
        assert!(matches!(
            idx.index_node_in("a", "file:a", text).unwrap_err().kind,
            IndexErrorKind::InvalidFormat(_)
        ));
        assert!(idx.is_empty());
        assert_eq!(idx.regions(), 0);
    }
}

#[test]
fn budget_refusals_happen_before_embedding_and_do_not_mutate_memory() {
    let calls = Arc::new(AtomicUsize::new(0));
    let budget = IndexBudget {
        max_text_bytes: 4,
        max_key_bytes: 32,
        max_anchors: 2,
        max_items: 1,
    };
    let mut idx = ShardedOctaIndex::with_budget(
        ControlledEmbedder {
            calls: calls.clone(),
        },
        budget,
    );
    assert!(matches!(
        idx.index_node("file:a", "oversized").unwrap_err().kind,
        IndexErrorKind::BudgetExceeded("text_bytes")
    ));
    assert!(matches!(
        idx.semantic_anchors("good", usize::MAX).unwrap_err().kind,
        IndexErrorKind::BudgetExceeded("anchors")
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    idx.index_node("file:a", "good").unwrap();
    assert!(matches!(
        idx.index_node("file:b", "good").unwrap_err().kind,
        IndexErrorKind::BudgetExceeded("items")
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(idx.len(), 1);
    assert!(idx.semantic_anchors("good", 0).unwrap().is_empty());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn keys_fail_closed_before_uri_region_derivation() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut idx = ShardedOctaIndex::with_budget(
        ControlledEmbedder {
            calls: calls.clone(),
        },
        IndexBudget {
            max_key_bytes: 8,
            ..IndexBudget::default()
        },
    );
    for key in ["", "file:a\n"] {
        assert!(matches!(
            idx.index_node(key, "good").unwrap_err().kind,
            IndexErrorKind::InvalidFormat(_)
        ));
    }
    let oversized = "x".repeat(9);
    let error = idx.index_node(&oversized, "good").unwrap_err();
    assert!(matches!(
        error.kind,
        IndexErrorKind::BudgetExceeded("key_bytes")
    ));
    assert!(error.uri.is_none());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn provenance_identifies_adapter_and_offline_fixture_without_semantic_claim() {
    let idx = ShardedOctaIndex::new(HashEmbedder::new(128));
    let provenance = idx.provenance();
    assert_eq!(provenance.contract, "octasoma.ccos-anchor@2");
    assert_eq!(provenance.dimension, 128);
    assert!(provenance.embedder_type.ends_with("HashEmbedder"));
}

#[test]
fn destination_contract_rejects_corrupt_utf8_anchor_after_store_reopen() {
    let path = std::env::temp_dir().join(format!(
        "octa-ccos-invalid-anchor-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut mem = ShardedMemory::new(HashEmbedder::new(4));
    let query = HashEmbedder::new(4).embed("query").unwrap();
    mem.insert_vec("a", &[0xff], &query).unwrap();
    mem.save_dir(path.to_str().unwrap()).unwrap();
    let idx = ShardedOctaIndex::open(HashEmbedder::new(4), path.to_str().unwrap()).unwrap();
    assert!(matches!(
        idx.semantic_anchors_in("a", "query", 1).unwrap_err().kind,
        IndexErrorKind::InvalidFormat(_)
    ));
    assert!(matches!(
        idx.semantic_anchors("query", 1).unwrap_err().kind,
        IndexErrorKind::InvalidFormat(_)
    ));
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn global_octaindex_returns_scored_anchors() {
    let mut idx = OctaIndex::new(HashEmbedder::new(128));
    idx.index_node("sym:src/db.rs:query", "build and run SQL queries")
        .unwrap();
    idx.index_node("sym:src/auth.rs:login", "authenticate a user")
        .unwrap();

    let anchors = idx
        .semantic_anchors("build and run SQL queries", 1)
        .unwrap();
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
    idx.index_node("sym:src/db.rs:query", "build and run SQL queries")
        .unwrap();
    idx.index_node("sym:src/db.rs:pool", "a pool of db connections")
        .unwrap();
    idx.index_node("sym:src/auth.rs:login", "authenticate a user")
        .unwrap();

    assert_eq!(idx.regions(), 2);
    assert_eq!(idx.len(), 3);

    // Scoped anchors recall the right uri and never leave the region.
    let hits = idx
        .semantic_anchors_in("src/db.rs", "a pool of db connections", 2)
        .unwrap();
    assert_eq!(hits[0].0, "sym:src/db.rs:pool");
    assert!(hits.iter().all(|(u, _)| u.starts_with("sym:src/db.rs:")));

    // The auth region cannot see db nodes (causal scoping).
    let auth = idx
        .semantic_anchors_in("src/auth.rs", "a pool of db connections", 5)
        .unwrap();
    assert!(auth.iter().all(|(u, _)| !u.starts_with("sym:src/db.rs:")));

    // An unknown region yields nothing.
    assert!(
        idx.semantic_anchors_in("src/missing.rs", "anything", 3)
            .unwrap()
            .is_empty()
    );
}
