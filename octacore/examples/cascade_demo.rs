//! OctaCore cascade demo — causal scope (toy) + OctaSoma semantic rerank.
//!
//! Offline and deterministic (`HashEmbedder`). With a real embedder
//! (`OllamaEmbedder`) and CCOS as the causal scope (`--features ccos`), this is the
//! validated cascade. Run: `cargo run --release --example cascade_demo`

use octacore::{Cascade, InMemoryScope};
use octasoma::HashEmbedder;

fn main() {
    // The causal layer (CCOS's role) — here a toy keyword→region scope.
    let scope = InMemoryScope::new()
        .region(
            &["sql", "database", "connection", "pool", "postgres"],
            &[
                (
                    "sym:src/db.rs:query",
                    "build and run SQL queries against Postgres",
                ),
                (
                    "sym:src/db.rs:pool",
                    "manage a pool of reusable database connections",
                ),
            ],
        )
        .region(
            &["login", "auth", "token", "sign in", "session", "password"],
            &[
                (
                    "sym:src/auth.rs:login",
                    "authenticate a user with username and password",
                ),
                (
                    "sym:src/auth.rs:token",
                    "issue and verify JSON web tokens for sessions",
                ),
            ],
        )
        .region(
            &["cache", "evict", "lru"],
            &[(
                "sym:src/cache.rs:evict",
                "evict least-recently-used entries when full",
            )],
        );

    // OctaSoma is the semantic layer (its Embedder + an exact cosine rerank).
    let core = Cascade::new(scope, HashEmbedder::new(256));

    println!("OctaCore cascade (toy CCOS scope + real OctaSoma rerank)\n");
    for q in [
        "open a pooled connection to the database",
        "how do users sign in?",
        "evict the least recently used cache entries",
    ] {
        let w = core.recall(q, 3, 64).unwrap();
        println!(
            "query: {q:?}\nstrategy: {} · {} tokens",
            w.strategy, w.tokens
        );
        for it in &w.items {
            println!("  ({:+.3}) {} — {}", it.score, it.uri, it.content);
        }
        println!();
    }
    println!(
        "Swap InMemoryScope for CcosScope (--features ccos) and HashEmbedder for\n\
         OllamaEmbedder to run the real cascade. SLHAv2 (--features slha) visualises\n\
         the KV-cache via octacore::slha::kv_cache_view. See README.md."
    );
}
