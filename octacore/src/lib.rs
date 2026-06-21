//! # OctaCore — the intelligent assembly of the triad
//!
//! OctaCore wires the three CHECKUPAUTO memories into a single recall — the
//! cascade the OctaSoma benchmark validated (99 % hit at ~26 tokens/turn on real
//! data, ~137× fewer than naive injection, where no single brick suffices):
//!
//! ```text
//!   query
//!     │  1. CAUSAL    (CCOS)      narrow to a small causal region
//!     ▼
//!   region ──► 2. SEMANTIC (OctaSoma)  exact cosine rerank within the region
//!     ▼                                (the embedding finisher that lands the hit)
//!   token-budgeted context window
//! ```
//!
//! It is **not a fourth memory**; it is the thin layer that makes the other three
//! behave as one. Each brick is honest about its limits, so the cascade is where
//! they compose:
//!
//! - **Causal (CCOS)** — narrows a query to its causal region; recalls lexically,
//!   not semantically. Adapter behind the `ccos` feature ([`ccos_adapter`]).
//! - **Semantic (OctaSoma)** — the embedding layer; here an exact cosine rerank
//!   *within* the small region (where 3-D would only be a coarse router).
//! - **Attention (SLHAv2)** — the inference-time KV-cache kernel that *consumes*
//!   the produced window. OctaSoma serves it as a **visualisation lens** (project
//!   tile latents to 3-D), not a text reranker. Behind the `slha` feature
//!   ([`slha`]).
//!
//! ```
//! use octacore::{Cascade, InMemoryScope};
//! use octasoma::HashEmbedder;
//!
//! let scope = InMemoryScope::new().region(
//!     &["sql", "database", "pool"],
//!     &[("sym:src/db.rs:pool", "manage a pool of reusable database connections")],
//! );
//! let core = Cascade::new(scope, HashEmbedder::new(64));
//! let window = core.recall("open a pooled database connection", 3, 64).unwrap();
//! assert_eq!(window.items[0].uri, "sym:src/db.rs:pool");
//! ```

#![forbid(unsafe_code)]

use octasoma::{EmbedError, Embedder, SketchIndex};

/// Unit separator packing `"uri␟content"` into one global-index payload.
const SEP: char = '\u{1f}';

/// A candidate surfaced by the causal layer (CCOS): a node uri and its content.
#[derive(Clone, Debug)]
pub struct ScopeItem {
    /// The node id (e.g. `sym:src/db.rs:pool`).
    pub uri: String,
    /// The node's content (what a window would carry).
    pub content: String,
}

/// The **causal** function (CCOS's role): narrow a query to its causal region.
///
/// Implementations return the region's candidate items (already causally
/// relevant); OctaCore then reranks them semantically.
pub trait CausalScope {
    /// Candidate items for `query`, within roughly `budget_tokens` of context.
    fn scope(&self, query: &str, budget_tokens: usize) -> Vec<ScopeItem>;
}

/// One item in the final window.
#[derive(Clone, Debug)]
pub struct RecallItem {
    /// The node id.
    pub uri: String,
    /// The node content.
    pub content: String,
    /// Cosine similarity to the query in embedding space, in `[-1, 1]`.
    pub score: f32,
}

/// The assembled context window (CCOS `RecallWindow` shape).
#[derive(Clone, Debug)]
pub struct RecallWindow {
    /// Which path produced this window.
    pub strategy: String,
    /// Items, most relevant first.
    pub items: Vec<RecallItem>,
    /// Estimated tokens of the assembled window.
    pub tokens: usize,
}

/// OctaCore: assemble **causal scope** (CCOS) + **semantic rerank** (OctaSoma).
///
/// Holds a global [`SketchIndex`] (a flat, contiguous SimHash sketch per indexed
/// item, parallel to its embedding) so scope-free queries get a precise
/// shortlist→rerank recall instead of the coarse 3-D router.
pub struct Cascade<E: Embedder, C: CausalScope> {
    causal: C,
    embedder: E,
    global: SketchIndex,
}

impl<E: Embedder, C: CausalScope> Cascade<E, C> {
    /// Builds a cascade from a causal scope and an [`Embedder`] (OctaSoma's trait —
    /// `OllamaEmbedder` in production, `HashEmbedder` for offline/tests). The global
    /// sketch index uses 256-bit SimHash by default; see [`Cascade::with_sketch_bits`].
    pub fn new(causal: C, embedder: E) -> Self {
        Self::with_sketch_bits(causal, embedder, 256)
    }

    /// Like [`Cascade::new`], choosing the global index's SimHash width (e.g. 1024
    /// for higher recall at more storage/scan cost).
    pub fn with_sketch_bits(causal: C, embedder: E, bits: usize) -> Self {
        let global = SketchIndex::new(embedder.dim(), bits, 0x0C7A_0C7A);
        Self {
            causal,
            embedder,
            global,
        }
    }

    /// The cascade: CCOS narrows to a region → OctaSoma reranks the region by
    /// **exact cosine** to the query → keep the top `k` and compact to
    /// `budget_tokens`. This is the validated finisher (the precise hit comes from
    /// causal narrowing + the exact rerank, not a global 3-D index).
    pub fn recall(
        &self,
        query: &str,
        k: usize,
        budget_tokens: usize,
    ) -> Result<RecallWindow, EmbedError> {
        // 1. CCOS narrows (ask for a generous region; we compact below).
        let region = self
            .causal
            .scope(query, budget_tokens.saturating_mul(4).max(budget_tokens));
        if region.is_empty() {
            return Ok(RecallWindow {
                strategy: "causal+semantic".into(),
                items: Vec::new(),
                tokens: 0,
            });
        }

        // 2. OctaSoma: exact cosine rerank within the region.
        let q = self.embedder.embed(query)?;
        let mut scored: Vec<RecallItem> = Vec::with_capacity(region.len());
        for it in region {
            let v = self.embedder.embed(&it.content)?;
            scored.push(RecallItem {
                score: cosine(&q, &v),
                uri: it.uri,
                content: it.content,
            });
        }
        scored.sort_by(|a, b| b.score.total_cmp(&a.score));
        scored.truncate(k);

        // 3. Compact to the token budget (always keep at least the top item).
        let mut items = Vec::new();
        let mut tokens = 0usize;
        for it in scored {
            let t = it.content.split_whitespace().count();
            if !items.is_empty() && tokens + t > budget_tokens {
                break;
            }
            tokens += t;
            items.push(it);
        }
        Ok(RecallWindow {
            strategy: "causal+semantic".into(),
            items,
            tokens,
        })
    }

    /// Indexes a node into OctaCore's **global** semantic index: embed `content`
    /// once, store its SimHash sketch (in a flat contiguous buffer) and embedding,
    /// keyed by `uri`. Use this so [`Cascade::recall_global`] works without a region.
    pub fn index_node(&mut self, uri: &str, content: &str) -> Result<(), EmbedError> {
        let v = self.embedder.embed(content)?;
        let packed = format!("{uri}{SEP}{content}");
        self.global.insert(&v, packed.as_bytes());
        Ok(())
    }

    /// Precise **global** recall for the scope-free case: a SimHash Hamming shortlist
    /// over everything added via [`Cascade::index_node`], then an exact cosine rerank
    /// — the high-precision tier the 3-D router cannot provide. Returns the top `k`.
    pub fn recall_global(&self, query: &str, k: usize) -> Result<RecallWindow, EmbedError> {
        // A generous shortlist: recall climbs steeply with it (256-bit: recall@1 of
        // the rerank is ~12% @32, ~70% @512). The rerank cost is linear in the
        // shortlist (one stored-embedding dot product each), so 256+ is cheap.
        self.recall_global_shortlisted(query, k, (k * 32).max(256))
    }

    /// Like [`Cascade::recall_global`], but with an explicit SimHash `shortlist` (how
    /// many Hamming-nearest candidates are exact-reranked). Smaller shortlists let the
    /// sketch *width* matter even on a small corpus (where the default shortlist would
    /// cover most of it and make the rerank near-exact); larger ones maximise recall.
    pub fn recall_global_shortlisted(
        &self,
        query: &str,
        k: usize,
        shortlist: usize,
    ) -> Result<RecallWindow, EmbedError> {
        let q = self.embedder.embed(query)?;
        let mut items = Vec::new();
        let mut tokens = 0usize;
        for (payload, score) in self.global.nearest(&q, k, shortlist) {
            let (uri, content) = split(&String::from_utf8_lossy(payload));
            tokens += content.split_whitespace().count();
            items.push(RecallItem {
                uri,
                content,
                score,
            });
        }
        Ok(RecallWindow {
            strategy: "semantic-global".into(),
            items,
            tokens,
        })
    }
}

/// Splits a packed `"uri␟content"` payload back into its parts.
fn split(packed: &str) -> (String, String) {
    match packed.split_once(SEP) {
        Some((u, c)) => (u.to_string(), c.to_string()),
        None => (String::new(), packed.to_string()),
    }
}

/// Cosine similarity over the shared prefix of two vectors (0 if either is zero).
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

// ---------------------------------------------------------------------------
// Built-in causal scope (standalone use / tests, no CCOS dependency)
// ---------------------------------------------------------------------------

/// A built-in [`CausalScope`] for standalone use and tests: a tiny in-memory set
/// of regions, each gated by keywords. A query is routed to the first region whose
/// keywords it mentions; that region's items are returned. (The real causal layer
/// is CCOS — see the `ccos` feature.)
#[derive(Default)]
pub struct InMemoryScope {
    regions: Vec<(Vec<String>, Vec<ScopeItem>)>,
}

impl InMemoryScope {
    /// An empty scope.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a region: queries mentioning any of `keywords` recall `items`
    /// (`(uri, content)` pairs).
    pub fn region(mut self, keywords: &[&str], items: &[(&str, &str)]) -> Self {
        self.regions.push((
            keywords.iter().map(|s| s.to_lowercase()).collect(),
            items
                .iter()
                .map(|(u, c)| ScopeItem {
                    uri: (*u).into(),
                    content: (*c).into(),
                })
                .collect(),
        ));
        self
    }
}

impl CausalScope for InMemoryScope {
    fn scope(&self, query: &str, _budget_tokens: usize) -> Vec<ScopeItem> {
        let q = query.to_lowercase();
        self.regions
            .iter()
            .find(|(kw, _)| kw.iter().any(|k| q.contains(k.as_str())))
            .map(|(_, items)| items.clone())
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// CCOS adapter (feature = "ccos")
// ---------------------------------------------------------------------------

/// Adapts CCOS's `ExternalMemory` into OctaCore's [`CausalScope`].
#[cfg(feature = "ccos")]
pub mod ccos_adapter {
    use super::{CausalScope, ScopeItem};
    use ccos::external_memory::{ExternalMemory, Recall};

    /// Wraps any CCOS [`ExternalMemory`] (e.g. `CcosMemory`) as a causal scope: a
    /// query becomes a `Recall::Task`, and the recalled region's items become
    /// [`ScopeItem`]s for OctaSoma to rerank.
    pub struct CcosScope<M: ExternalMemory> {
        mem: M,
    }

    impl<M: ExternalMemory> CcosScope<M> {
        /// Wraps a CCOS memory.
        pub fn new(mem: M) -> Self {
            Self { mem }
        }

        /// Borrow the underlying memory (e.g. to `ingest_source` or `checkpoint`).
        pub fn memory(&self) -> &M {
            &self.mem
        }

        /// Mutably borrow the underlying memory.
        pub fn memory_mut(&mut self) -> &mut M {
            &mut self.mem
        }
    }

    impl<M: ExternalMemory> CausalScope for CcosScope<M> {
        fn scope(&self, query: &str, budget_tokens: usize) -> Vec<ScopeItem> {
            self.mem
                .recall(&Recall::task(query), budget_tokens)
                .items
                .into_iter()
                .map(|it| ScopeItem {
                    uri: it.uri,
                    content: it.content,
                })
                .collect()
        }
    }
}

// ---------------------------------------------------------------------------
// SLHAv2 lens (feature = "slha")
// ---------------------------------------------------------------------------

/// Visualise SLHAv2 KV-cache tiles through OctaSoma — the honest SLHAv2 ↔ OctaSoma
/// link (a lens, not attention routing).
#[cfg(feature = "slha")]
pub mod slha {
    use octasoma::FractalMemory3D;
    use scirust::{D_C, SciRustSlhaTile};

    /// Projects each tile's `dequant_latent()` (128-d) to 3-D with a PCA-calibrated
    /// OctaSoma index and returns viewer JSON (`{count, half_size, points:[…]}`),
    /// labelled `"head {head_id} tok {token_id}"` so `viewer/index.html` colours by
    /// head. Inspection/debug only — SLHAv2's `compute_score` owns attention.
    pub fn kv_cache_view(tiles: &[SciRustSlhaTile], max_points: usize) -> String {
        let latents: Vec<[f32; D_C]> = tiles.iter().map(|t| t.dequant_latent()).collect();
        let flat: Vec<f32> = latents.iter().flat_map(|v| v.iter().copied()).collect();
        let mut mem = FractalMemory3D::new_with_pca(D_C, &flat, latents.len().max(1));
        for (t, v) in tiles.iter().zip(&latents) {
            let label = format!("head {} tok {}", t.head_id, t.token_id);
            mem.insert(v, Some(label.as_bytes()));
        }
        mem.export_points_json(max_points)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use octasoma::HashEmbedder;

    fn scope() -> InMemoryScope {
        InMemoryScope::new()
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
                &["login", "auth", "token", "sign in", "session"],
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
    }

    #[test]
    fn cascade_scopes_to_the_causal_region() {
        let core = Cascade::new(scope(), HashEmbedder::new(128));
        let w = core
            .recall("open a pooled connection to the database", 5, 64)
            .unwrap();
        assert_eq!(w.strategy, "causal+semantic");
        assert!(!w.items.is_empty());
        // Every item stays inside the db region (causal scoping).
        assert!(
            w.items
                .iter()
                .all(|it| it.uri.starts_with("sym:src/db.rs:"))
        );
        // No auth node can appear for a db query.
        assert!(!w.items.iter().any(|it| it.uri.contains("auth")));
    }

    #[test]
    fn respects_k_and_token_budget() {
        let core = Cascade::new(scope(), HashEmbedder::new(128));
        let w = core
            .recall("verify a session token for login", 1, 1000)
            .unwrap();
        assert_eq!(w.items.len(), 1); // k = 1
        assert!(w.items[0].uri.starts_with("sym:src/auth.rs:"));

        // A tiny budget still returns at least the top item, and tokens are bounded
        // to that first item's length.
        let w2 = core
            .recall("verify a session token for login", 5, 1)
            .unwrap();
        assert_eq!(w2.items.len(), 1);
    }

    #[test]
    fn global_sketch_recall_finds_indexed_node() {
        let mut core = Cascade::new(InMemoryScope::new(), HashEmbedder::new(128));
        core.index_node(
            "sym:src/db.rs:pool",
            "manage a pool of reusable database connections",
        )
        .unwrap();
        core.index_node(
            "sym:src/auth.rs:login",
            "authenticate a user with username and password",
        )
        .unwrap();
        // Scope-free recall via the sketch shortlist → exact rerank.
        let w = core
            .recall_global("manage a pool of reusable database connections", 1)
            .unwrap();
        assert_eq!(w.strategy, "semantic-global");
        assert_eq!(w.items.len(), 1);
        assert_eq!(w.items[0].uri, "sym:src/db.rs:pool");
        assert!(w.items[0].score > 0.99); // exact text → cosine ~1
    }

    #[test]
    fn unknown_region_yields_empty_window() {
        let core = Cascade::new(scope(), HashEmbedder::new(128));
        let w = core.recall("render a 3-D triangle mesh", 5, 64).unwrap();
        assert!(w.items.is_empty());
        assert_eq!(w.tokens, 0);
    }

    #[test]
    fn cosine_basics() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }
}
