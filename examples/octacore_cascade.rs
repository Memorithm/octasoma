//! OctaCore (prototype) — the intelligent assembly of the triad as **one recall**.
//!
//! OctaCore wires the three CHECKUPAUTO memories into a single cascade — the
//! deployment the benchmark validated (99 % hit at ~26 tokens/turn on real data,
//! where no single brick suffices):
//!
//! ```text
//!   query
//!     │  1. CAUSAL  (CCOS)      narrow to a small causal region
//!     ▼
//!   region ──► 2. SEMANTIC (OctaSoma)  rank memories *within* the region
//!     │                                (ShardedMemory — the validated layer)
//!     ▼
//!   shortlist ─► 3. ATTENTION (SLHAv2)  rerank for the final window
//!     ▼
//!   token-budgeted context window  (CCOS RecallWindow shape)
//! ```
//!
//! OctaSoma is the real layer here ([`ShardedMemory`]); CCOS and SLHAv2 are toy
//! in-file stubs (offline, deterministic) so the *shape* runs without the other
//! crates. This is an early in-repo sketch — the real, honest crate is **`octacore/`**
//! (CCOS as the causal scope, OctaSoma an exact cosine rerank, and SLHAv2 the
//! KV-cache *visualisation lens*, not a text reranker). Its `ccos`/`slha` features
//! are verified to compile against the real upstream crates. See `docs/octacore.md`.
//!
//! Run: `cargo run --release --example octacore_cascade`

use std::collections::HashSet;
use std::fmt::Write as _;

use octasoma::{Embedder, HashEmbedder, ShardedMemory};

/// Unit separator packing `"uri␟content"` into one OctaSoma payload.
const SEP: char = '\u{1f}';

/// One assembled context item (CCOS `RecallItem` shape).
#[derive(Clone)]
struct RecallItem {
    uri: String,
    content: String,
    score: f32,
}

/// The **causal** function (CCOS's role): map a query to its causal region.
trait CausalMemory {
    fn region_for(&self, query: &str) -> Option<String>;
}

/// The **attention** function (SLHAv2's role, or an exact reranker): order the
/// shortlist for the final window.
trait AttentionKernel {
    fn rerank(&self, query: &str, items: Vec<RecallItem>) -> Vec<RecallItem>;
}

/// OctaCore: the orchestrator that assembles causal → semantic → attention.
struct OctaCore<E: Embedder, C: CausalMemory, A: AttentionKernel> {
    causal: C,
    semantic: ShardedMemory<E>,
    attention: A,
}

impl<E: Embedder, C: CausalMemory, A: AttentionKernel> OctaCore<E, C, A> {
    /// The cascade: narrow (CCOS) → rank within region (OctaSoma) → rerank
    /// (SLHAv2) → assemble a token-budgeted window. Returns `(strategy, window,
    /// tokens)`.
    fn recall(&self, query: &str, k: usize, budget_tokens: usize) -> (&'static str, String, usize) {
        // 1. CCOS narrows to a causal region.
        let Some(region) = self.causal.region_for(query) else {
            return ("none", "<no causal region for query>\n".into(), 0);
        };

        // 2. OctaSoma ranks within the region (payload = "uri␟content").
        let hits = self
            .semantic
            .recall_scored(&region, query, k)
            .unwrap_or_default();
        let items: Vec<RecallItem> = hits
            .into_iter()
            .map(|(packed, d2)| {
                let (uri, content) = split(&packed);
                RecallItem {
                    uri,
                    content,
                    score: 1.0 / (1.0 + d2),
                }
            })
            .collect();

        // 3. SLHAv2 reranks the shortlist.
        let items = self.attention.rerank(query, items);

        // 4. Assemble a token-budgeted window (region header + items).
        let mut out = String::new();
        let _ = writeln!(out, "# region: {region}");
        let mut tokens = 0usize;
        for it in items {
            let t = it.content.split_whitespace().count();
            if tokens + t > budget_tokens {
                break;
            }
            tokens += t;
            let _ = writeln!(out, "  ({:.2}) {} — {}", it.score, it.uri, it.content);
        }
        ("causal+semantic+attention", out, tokens)
    }
}

/// Toy CCOS: keyword → causal region (a source file). The real CCOS resolves this
/// through its event-sourced code graph.
struct ToyCausal;
impl CausalMemory for ToyCausal {
    fn region_for(&self, query: &str) -> Option<String> {
        let q = query.to_lowercase();
        let hit = |ws: &[&str]| ws.iter().any(|w| q.contains(w));
        if hit(&["sql", "database", "connection", "pool", "postgres"]) {
            Some("src/db.rs".into())
        } else if hit(&["login", "auth", "token", "sign in", "password", "session"]) {
            Some("src/auth.rs".into())
        } else if hit(&["cache", "evict", "lru"]) {
            Some("src/cache.rs".into())
        } else {
            None
        }
    }
}

/// Toy SLHAv2: rerank by whole-word overlap with the query (a stand-in for the
/// kernel's `compute_score`). The real SLHAv2 scores compressed KV-cache tiles.
struct LexicalAttention;
impl AttentionKernel for LexicalAttention {
    fn rerank(&self, query: &str, mut items: Vec<RecallItem>) -> Vec<RecallItem> {
        let qw = words(query);
        items.sort_by(|a, b| {
            let (oa, ob) = (overlap(&qw, &a.content), overlap(&qw, &b.content));
            ob.cmp(&oa).then(b.score.total_cmp(&a.score))
        });
        items
    }
}

fn words(s: &str) -> HashSet<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect()
}

fn overlap(qw: &HashSet<String>, content: &str) -> usize {
    words(content).intersection(qw).count()
}

fn split(packed: &str) -> (String, String) {
    match packed.split_once(SEP) {
        Some((u, c)) => (u.to_string(), c.to_string()),
        None => (String::new(), packed.to_string()),
    }
}

fn main() {
    // OctaSoma holds the real semantic layer (here a deterministic offline
    // embedder; in production an OllamaEmbedder).
    let mut semantic = ShardedMemory::new(HashEmbedder::new(256));
    let corpus = [
        (
            "src/db.rs",
            "sym:src/db.rs:query",
            "build and run SQL queries against Postgres",
        ),
        (
            "src/db.rs",
            "sym:src/db.rs:pool",
            "manage a pool of reusable database connections",
        ),
        (
            "src/auth.rs",
            "sym:src/auth.rs:login",
            "authenticate a user with username and password",
        ),
        (
            "src/auth.rs",
            "sym:src/auth.rs:token",
            "issue and verify JSON web tokens for sessions",
        ),
        (
            "src/cache.rs",
            "sym:src/cache.rs:evict",
            "evict least-recently-used entries when full",
        ),
    ];
    for (region, uri, content) in corpus {
        let packed = format!("{uri}{SEP}{content}");
        semantic.insert(region, &packed, content).unwrap();
    }

    let core = OctaCore {
        causal: ToyCausal,
        semantic,
        attention: LexicalAttention,
    };

    println!("OctaCore cascade (toy CCOS + real OctaSoma + toy SLHAv2)\n");
    for q in [
        "open a pooled connection to the database",
        "how do users sign in?",
        "evict the least recently used cache entries",
    ] {
        let (strategy, window, tokens) = core.recall(q, 3, 64);
        println!("query: {q:?}\nstrategy: {strategy} · {tokens} tokens\n{window}");
    }
    println!(
        "Each brick is replaceable: CCOS implements CausalMemory, SLHAv2 implements\n\
         AttentionKernel, OctaSoma is the ShardedMemory. See docs/octacore.md."
    );
}
