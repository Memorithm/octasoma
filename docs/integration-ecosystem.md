# OctaSoma — ecosystem integration plan (CCOS · SLHAv2)

A grounded plan for wiring OctaSoma into the CHECKUPAUTO stack. Interfaces below
are quoted from the actual source of each project.

## The stack

| Project | Role | Memory kind |
|---|---|---|
| `scirust` | math core (SVD/PCA/SIMD), pure Rust | — |
| `SLHAv2` | compressed **working memory** (KV-cache → 128-byte INT4 tiles), MCP | short-term |
| `CCOS` | **causal/structural** memory (code graph, event-sourced, MCP) | long-term, structural |
| `OctaSoma` | **semantic/spatial** memory (embeddings → fractal 3-D, explainable) | long-term, semantic |

OctaSoma is the **semantic + visualizable** layer the others lack. CCOS states its
own `Task` recall is *"a deliberately simple lexical entry point (no embeddings) …
not a semantic retriever."* That is exactly OctaSoma's job.

---

## 1. The connector — an OctaSoma MCP server

Both CCOS (`ccos mcp`) and SLHAv2 (`slha-mcp`) integrate over **MCP** (stdio
JSON-RPC). The highest-leverage, decoupled move is to give OctaSoma the same
surface: a `octasoma mcp <store.frac>` server. It lets CCOS, SLHAv2, and any agent
use OctaSoma's semantic recall + explanation without coupling crates.

**Tools (speaking CCOS's own vocabulary so results are drop-in compatible):**

| Tool | Args | Returns |
|---|---|---|
| `ingest` | `uri`, `text` | `{uri, nodes_added}` (embed `text`, `perceive` with payload=`uri`) |
| `recall` | `text`, `budget`/`k` | `{strategy:"semantic", items:[{uri,score,kind,content}], tokens}` |
| `explain` | `text`, `k` | `{query_point, zoom_path:[…], neighbors:[{uri,distance,point}]}` |
| `stats` | — | `{memories, nodes, arena}` |

The `recall` result mirrors CCOS's `RecallWindow { strategy, items: Vec<RecallItem>,
tokens }` with `RecallItem { uri, score, kind, content }` — so a `Recall::Semantic`
in CCOS can consume it verbatim, and an agent can mix CCOS (causal) and OctaSoma
(semantic) windows.

**Framing:** line-delimited JSON-RPC 2.0 over stdio (`initialize`, `tools/list`,
`tools/call`). **Open decision:** robust JSON needs either `serde_json` behind a new
`mcp` cargo feature (keeps the default build at one dependency) or a hand-rolled
minimal parser. Recommendation: **`serde_json` behind `--features mcp`**.

Built entirely on existing OctaSoma API: `perceive` / `recall` / `explain` /
`export_points_json`.

---

## 2. Semantic recall for CCOS (in-process, tightest)

CCOS's trait (verbatim from `src/external_memory.rs`):

```rust
pub trait ExternalMemory {
    fn ingest_source(&mut self, uri: &str, source: &str) -> IngestReport;
    fn signal_failure(&mut self, node: &str, depth: u32) -> Result<usize, MemoryError>;
    fn recall(&self, recall: &Recall, budget_tokens: usize) -> RecallWindow;
    fn verify(&self) -> Integrity;
    fn stats(&self) -> MemoryStats;
    fn checkpoint(&self) -> Result<(), MemoryError>;
}
pub enum Recall { WorkingSet, Around(String), Task(String) }
pub struct RecallItem { pub uri: String, pub score: f64, pub kind: String, pub content: String }
```

**Plan (changes live in CCOS, OctaSoma stays a dependency):**

1. Add `octasoma = { path = "../octasoma" }` to CCOS.
2. On `ingest_source`, for each created node (`sym:…`, `mod:…`, `file:…`) embed its
   label+content and `perceive(embedding, payload = node_uri)` into an OctaSoma
   `FractalMemory3D` held beside the `MemoryGraph`.
3. Add a `Recall::Semantic(String)` variant. Its arm: embed the query →
   `octa.nearest_embedding(q, k)` → map item ids back to node URIs → reuse CCOS's
   existing `assemble_window("semantic", ids, budget, …)`. **Semantic anchor
   (OctaSoma) → causal expansion (CCOS).**
4. Expose it through the existing `ccos memory` / `ccos mcp` transports as a new
   `strategy: "semantic"`.

Bonus: CCOS's v0.3 `ContextRegionEngine` ("clusters graph into spatial regions")
overlaps OctaSoma's 3-D regions — OctaSoma can *supply* those regions and a
**visualizable** view of CCOS memory via `explain` + the 3-D viewer.

**Embedder note:** both can share one `Embedder` (OctaSoma's trait) — `OllamaEmbedder`
in production, `HashEmbedder` for deterministic tests.

---

## 3. "See your KV-cache" — SLHAv2 tile visualizer

SLHAv2 tiles (verbatim from `SLHAv2.md`): `SciRustSlhaTile` holds a 128-dim latent
(`D_C = 128`) recoverable with **`dequant_latent() -> [f32; 128]`**, plus
`token_id`, `position`, `head_id`, `scale`, `flags`. The `slha-audit` binary already
emits JSON.

**Plan (OctaSoma as the lens):**

1. For each tile, `dequant_latent()` → a 128-dim vector; build an OctaSoma
   `FractalMemory3D::new_with_pca(128, …)` and `perceive(latent, payload =
   "tok {token_id} pos {position} head {head_id}")`.
2. `export_points_json()` → open `viewer/index.html`: the KV-cache becomes a
   **navigable 3-D map** (coloured by head via the payload prefix), revealing tile
   clusters and compression structure — an inspection/debug tool SLHAv2 lacks.
3. Two wirings: (a) consume `slha-audit` JSON in a small adapter; (b) a tiny Rust
   example linking both crates. Start with (a) — zero coupling.

Secondary (exploratory): a 3-D **spatial pre-filter** over tile latents to shortlist
tiles (OctaSoma's measured ~93% recall at ~90× less work). SLHAv2 already scores via
`compute_score`, so the clear win here is **visualization/diversity**, not core
selection.

---

## Build order

1. **OctaSoma MCP server** (`octasoma mcp`) — the connector; unblocks CCOS *and*
   agent use. (decision: `serde_json` behind `--features mcp`.)
2. **SLHAv2 visualizer** — adapter from `slha-audit` JSON → OctaSoma export → viewer.
   Mostly reuses what exists; fastest visible payoff.
3. **CCOS `Recall::Semantic`** — in-process adapter (edits live in CCOS).

Caveat: this plan is grounded in each repo's README/specs and the quoted
`external_memory.rs` / `SLHAv2.md`; exact module wiring (e.g. CCOS `assemble_window`
signature, `slha-audit` JSON shape) will be confirmed against source when building.
