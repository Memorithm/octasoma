# OctaCore — the intelligent assembly of the triad

**OctaCore** is the orchestrator that wires the three CHECKUPAUTO memories into a
single recall: the [validated cascade](integration-ecosystem.md#measured-the-cascade-validated-at-real-scale)
(99 % hit at ~26 tokens/turn on real data, ~137× fewer than naive injection — where
no single brick suffices). It is **not a fourth memory**; it is the thin layer that
makes the other three behave as one.

```
  query
    │  1. CAUSAL   (CCOS)      narrow to a small causal region  →  ScopeItems
    ▼
  region ──► 2. SEMANTIC (OctaSoma)  exact cosine rerank within the region
    ▼                                (the embedding finisher that lands the hit)
  token-budgeted context window   (CCOS RecallWindow shape)

  SLHAv2 = the inference-time attention kernel that CONSUMES the window;
           OctaSoma visualises its KV-cache (a lens, not a text reranker).
```

## The three functions

| Function | Memory kind | Owner | Role in the cascade |
|---|---|---|---|
| **Causal / structural** | long-term, "what depends on what" | **CCOS** | narrow a query to its causal region (small *N*) |
| **Semantic / spatial** | long-term, embedding recall | **OctaSoma** | rank memories *within* the region; cheap, explainable, visualizable |
| **Working memory / attention** | short-term, compressed KV-cache | **SLHAv2** | consume the window at inference; OctaSoma visualises its KV-cache |

Each is honest about its limits: CCOS recalls lexically ("not a semantic
retriever"), OctaSoma's global 3-D is a coarse router (0 % at scale, decisive *per
region*), and SLHAv2 owns attention scoring (OctaSoma is only a visualization lens
there). The cascade is where they compose into something none achieves alone.

## The crate (`octacore/`)

The real crate lives at [`octacore/`](../octacore/) (default build depends only on
OctaSoma). Its surface:

```rust
/// CCOS's role: narrow a query to its causal region's candidate items.
pub trait CausalScope { fn scope(&self, query: &str, budget_tokens: usize) -> Vec<ScopeItem>; }
pub struct ScopeItem { pub uri: String, pub content: String }

/// The orchestrator: causal scope (CCOS) + exact cosine rerank (OctaSoma).
pub struct Cascade<E: Embedder, C: CausalScope> { /* causal, embedder */ }
impl Cascade {
    // recall(query, k, budget) = scope(CCOS) → embed + cosine rerank → compact
    pub fn recall(&self, query: &str, k: usize, budget: usize) -> Result<RecallWindow, EmbedError>;
}
```

`InMemoryScope` is a built-in keyword `CausalScope` for offline use/tests; the real
causal layer is CCOS, behind the `ccos` feature.

## How the real systems plug in (verified)

Both adapters are **verified to compile and lint against the real upstream crates**
(`cargo build/clippy --features ccos,slha` here, against CCOS `v0.3.0` and
SLHAv2/`scirust` `v0.2.0`):

- **CCOS** → `ccos_adapter::CcosScope<M: ExternalMemory>` (`--features ccos`). A query
  becomes `Recall::task(query)`; the recalled region's `RecallItem`s become
  `ScopeItem`s for OctaSoma to rerank. CCOS can also *call* OctaCore as its missing
  `Recall::Semantic` strategy (see `integration/ccos/PATCH.md`) — the two compose
  either direction.
- **OctaSoma** → the `Embedder` + an exact cosine rerank inside `Cascade::recall`
  (within the small region, where full-D cosine beats a 3-D index). Its
  `ShardedMemory` remains available as the explainable/visualisable spatial layer.
- **SLHAv2** → `slha::kv_cache_view(tiles, max_points)` (`--features slha`): projects
  each tile's `dequant_latent()` (128-d) to 3-D via OctaSoma and emits viewer JSON
  (colour by head). A visualisation lens — SLHAv2's `compute_score` owns attention.

## Where it lives

OctaCore is the **top crate** of the stack — it depends on `ccos`, `octasoma`, and
`scirust` (SLHAv2):

```
octacore  ──depends on──►  ccos, octasoma, scirust
```

It cannot live *inside* octasoma (octasoma is the leaf dependency; reversing that
would create a cycle). Because the standalone repo `checkupauto/octacore` does not
exist yet, the crate is **staged inside this repo** under `octacore/` as its own
isolated workspace — it builds and tests here against the local OctaSoma and does
not affect octasoma's own build. To extract it into its own repository, see
[`octacore/README.md`](../octacore/README.md) (one `git subtree split`, then switch
the OctaSoma path dependency to a git/version dependency).

## Honest framing

OctaCore's value is the **assembly**, not a new algorithm. The measured win
(99 % @ ~26 tokens) comes from causal narrowing + an exact rerank within a small
region; OctaSoma is the cheap, explainable, visualizable coarse layer that proposes
and organises. The product claim is exactly the paper's
[inference-pyramid](../paper/en/main.tex) result, packaged as one API.
