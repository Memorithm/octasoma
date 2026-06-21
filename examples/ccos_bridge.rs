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

use octa_index::{OctaIndex, ShardedOctaIndex, region_of};
use octasoma::HashEmbedder;

// CCOS-style nodes (uri + content), as produced by `ingest_source`.
const NODES: [(&str, &str); 5] = [
    (
        "sym:src/auth.rs:login",
        "user login and authentication flow",
    ),
    ("sym:src/auth.rs:token", "verify a JWT session token"),
    (
        "sym:src/db.rs:query",
        "SQL query builder and connection pool",
    ),
    ("mod:src/cache.rs", "in-memory LRU cache for hot keys"),
    ("file:src/main.rs", "program entry point and CLI wiring"),
];

fn main() {
    // --- 1. Global index: coarse semantic anchors (no causal scope) ---------
    let mut idx = OctaIndex::new(HashEmbedder::new(256));
    for (uri, content) in NODES {
        idx.index_node(uri, content);
    }
    println!("global OctaIndex: indexed {} CCOS nodes\n", idx.len());

    // `assemble_window` in CCOS would take these anchors and expand them causally.
    let query = "SQL query builder and connection pool";
    println!("query: {query:?}\nglobal semantic anchors → CCOS:");
    for (uri, score) in idx.semantic_anchors(query, 3) {
        println!("  {score:.3}  {uri}");
    }

    // --- 2. Sharded index: the validated per-region deployment --------------
    // CCOS narrows to a causal region first; OctaSoma reranks *within* it.
    let mut sharded = ShardedOctaIndex::new(HashEmbedder::new(256));
    for (uri, content) in NODES {
        sharded.index_node(uri, content); // region derived via region_of(uri)
    }
    println!(
        "\nsharded OctaIndex: {} nodes across {} causal regions",
        sharded.len(),
        sharded.regions()
    );
    println!(
        "region_of(\"sym:src/auth.rs:login\") = {:?}",
        region_of("sym:src/auth.rs:login")
    );

    // When CCOS knows the region, recall is scoped to it — the 99 %-hit path.
    let region = "src/auth.rs";
    println!("\nscoped anchors in {region:?} for \"verify a JWT session token\":");
    for (uri, score) in sharded.semantic_anchors_in(region, "verify a JWT session token", 2) {
        println!("  {score:.3}  {uri}");
    }

    println!(
        "\n(offline HashEmbedder = exact-text; swap in OllamaEmbedder for real \
         semantic anchors — e.g. \"how do users sign in?\" → sym:src/auth.rs:login.)"
    );
}
