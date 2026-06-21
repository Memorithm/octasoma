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

**Status: implemented** (`src/bin/octasoma-mcp.rs`, `--features mcp`). The server is
**region-sharded** (`ShardedMemory`) — the validated deployment. `ingest`/`recall`
take an optional `region`; when omitted it is derived from the CCOS-style uri
(`sym:src/db.rs:query` → `src/db.rs`). The store is a **directory** of per-region
shards.

**Tools (speaking CCOS's own vocabulary so results are drop-in compatible):**

| Tool | Args | Returns |
|---|---|---|
| `ingest` | `uri`, `text`, `region?` | `{uri, region, nodes_added}` |
| `recall` | `text`, `region?`, `k`/`budget` | `{strategy, region, items:[{uri,score,kind,content}], tokens}` |
| `explain` | `text`, `region?`, `k` | `{region, query_point, zoom_path:[…], neighbors:[{uri,distance,point}]}` |
| `stats` | — | `{memories, regions, region_keys:[…]}` |

With a `region`, `recall` is scoped to that causal region (`strategy:"semantic"` —
the 99 %-hit path); without one it is a coarse cross-region merge
(`strategy:"semantic-global"`).

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

**Status: implemented** (`examples/kv_cache_viz.rs`). For each tile,
`dequant_latent()` → a 128-dim vector; the example builds an OctaSoma
`FractalMemory3D::new_with_pca(128, …)`, inserts each latent with payload
`"head {head_id} tok {token_id}"`, and writes `export_points_json()`. Open
`viewer/index.html` and drop the JSON: the KV-cache becomes a **navigable 3-D map,
coloured by head** (the viewer derives the category `head N` from the payload and
shows a legend), revealing tile clusters and compression structure — an
inspection/debug tool SLHAv2 lacks.

Input is a TSV (`label⇥f0 f1 … f127`, the documented `dequant_latent()` output) so
there is **zero coupling**; a Rust example linking both crates is the alternative.

Secondary (measured, honest): `examples/slha_prefilter.rs` tests OctaSoma as a 3-D
**spatial pre-filter** over tile latents — but the coarse 3-D recovers only a
fraction of the *exact* attention top-k, so the clear win is **visualization /
diversity**, not core selection (SLHAv2's own `compute_score` owns that).

---

## Build order (status)

1. ✅ **OctaSoma MCP server** (`octasoma-mcp`, `--features mcp`) — the connector;
   region-sharded, `serde_json` behind `--features mcp`.
2. ✅ **SLHAv2 visualizer** — `examples/kv_cache_viz.rs` → OctaSoma export → viewer
   (coloured by head, with a legend).
3. ✅ **CCOS semantic recall** — `integration/ccos/octa_index.rs` ships `OctaIndex`
   (global) and `ShardedOctaIndex` (per-region, the validated path); final wiring
   edits live in CCOS (`PATCH.md`).

Caveat: this plan is grounded in each repo's README/specs and the quoted
`external_memory.rs` / `SLHAv2.md`; exact module wiring (e.g. CCOS `assemble_window`
signature, `slha-audit` JSON shape) will be confirmed against source when building.

---

## Measured: the cascade, validated at real scale

Run on **795 real CCOS nodes** (`scripts/rs_to_nodes.sh ~/CCOS/src`), 763 queries,
embedded by **Ollama `nomic-embed-text` (768-d)**, via
`examples/pipeline_bench_text.rs`:

| strategy | tokens/turn | target hit | causal-relevant |
|---|---|---|---|
| naive (inject all) | 3622 | 100 % | 3 % |
| **semantic-only (OctaSoma global 3-D)** | 30 | **0 %** | 4 % |
| causal-only (CCOS region) | 158 | 100 % | 100 % |
| **causal + semantic (triad, exact rerank)** | **26** | **99 %** | **100 %** |

**Honest reading:**

- **The cascade is validated.** The triad hits the target **99 % at ~26 tokens/turn
  — ~137× fewer than naive (3622)** — with **100 % causally-relevant** context.
  Neither brick alone works: naive is wasteful, semantic-only fails.
- **OctaSoma's *global* 3-D recall is 0 % at this scale** — the 768→3 projection is
  a *coarse router*, exactly as characterised in [evaluation.md](evaluation.md). It
  is **not** the component that lands the hit.
- **The hit comes from CCOS's causal narrowing + an exact rerank** within the small
  region. So OctaSoma's defensible role here is the **cheap, compact, explainable,
  visualisable coarse layer** — not the precise retriever.
- **Deployment lesson:** OctaSoma's 3-D works *per causal region* (small N), not as
  one global index. `pipeline_bench_text` includes two rerank rows —
  `OctaSoma rerank` (global 3-D) vs `OctaSoma/module` (a 3-D PCA per region) — whose
  gap measures exactly this. Index OctaSoma **per CCOS region**, not globally.

This lesson is now a first-class type: [`ShardedMemory`](../src/sharded.rs) keeps one
OctaSoma index per region key and recalls *within* a region (`recall`/`recall_scored`),
with a coarse cross-region fallback (`recall_global`) and directory persistence
(`save_dir`/`open_dir`). The CCOS adapter exposes it as `ShardedOctaIndex`
(`integration/ccos/octa_index.rs`): `index_node_in(region, uri, content)` then
`semantic_anchors_in(region, text, k)` — the 99 %-hit path. See
`examples/ccos_bridge.rs` for a runnable demo.

Reproduce:

```bash
bash scripts/rs_to_nodes.sh <SRC_DIR> > nodes.tsv
grep '^sym:' nodes.tsv | awk -F'\t' '{n=$1; sub(/.*:/,"",n); print "what does " n " do?\t" $1}' > queries.tsv
cargo run --release --example pipeline_bench_text -- \
  --corpus nodes.tsv --queries queries.tsv \
  --url http://localhost:11434 --model nomic-embed-text --dim 768
```
