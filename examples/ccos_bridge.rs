//! Demo + compile-check of the CCOS semantic-recall bridge.
//!
//! Loads `integration/ccos/octa_index.rs` (the helper CCOS would vendor) and shows
//! the anchor node URIs CCOS's `Recall::Semantic` arm would receive from OctaSoma.
//! Offline with `HashEmbedder` (exact-text); with `OllamaEmbedder` it is semantic.
//!
//! Run: `cargo run --release --example ccos_bridge`

// `open`/`save`/`is_empty` are part of the CCOS-facing API but unused by this demo.
#[allow(dead_code)]
#[path = "../integration/ccos/octa_index.rs"]
mod octa_index;

use octa_index::OctaIndex;
use octasoma::HashEmbedder;

fn main() {
    let mut idx = OctaIndex::new(HashEmbedder::new(256));

    // CCOS-style nodes (uri + content), as produced by `ingest_source`.
    let nodes = [
        (
            "sym:src/auth.rs:login",
            "user login and authentication flow",
        ),
        (
            "sym:src/db.rs:query",
            "SQL query builder and connection pool",
        ),
        ("mod:src/cache.rs", "in-memory LRU cache for hot keys"),
        ("file:src/main.rs", "program entry point and CLI wiring"),
    ];
    for (uri, content) in nodes {
        idx.index_node(uri, content);
    }
    println!("indexed {} CCOS nodes\n", idx.len());

    // `assemble_window` in CCOS would take these anchors and expand them causally.
    let query = "SQL query builder and connection pool";
    println!("query: {query:?}\nsemantic anchors → CCOS:");
    for (uri, score) in idx.semantic_anchors(query, 3) {
        println!("  {score:.3}  {uri}");
    }
    println!(
        "\n(offline HashEmbedder = exact-text; swap in OllamaEmbedder for real \
         semantic anchors — e.g. \"how do users sign in?\" → sym:src/auth.rs:login.)"
    );
}
