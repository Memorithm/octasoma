# Adding semantic recall to CCOS with OctaSoma

This wires `OctaIndex` (see `octa_index.rs`) into CCOS so its `ExternalMemory` gains
a `Recall::Semantic` strategy — the embedding-based recall CCOS lacks today
(its `Recall::Task` is "a deliberately simple lexical entry point … not a semantic
retriever"). **Semantic anchor (OctaSoma) → causal expansion (CCOS).**

The verbatim CCOS interfaces this patch targets (from `src/external_memory.rs`):

```rust
pub trait ExternalMemory {
    fn ingest_source(&mut self, uri: &str, source: &str) -> IngestReport;
    fn recall(&self, recall: &Recall, budget_tokens: usize) -> RecallWindow;
    // …
}
pub enum Recall { WorkingSet, Around(String), Task(String) }
pub struct RecallItem { pub uri: String, pub score: f64, pub kind: String, pub content: String }
```

## Steps

**1. Dependency** — `Cargo.toml`:

```toml
octasoma = { git = "https://github.com/checkupauto/octasoma" }
```

**2. Vendor the index** — copy `octa_index.rs` to `src/octa_index.rs` and
`mod octa_index;` (or `pub use`). Pick an embedder: `OllamaEmbedder` for real
semantics, `HashEmbedder` for deterministic tests.

**3. Hold an index beside the graph** — in `CcosMemory`:

```rust
use crate::octa_index::OctaIndex;
use octasoma::OllamaEmbedder;

pub struct CcosMemory {
    // … existing graph / event_log fields …
    octa: OctaIndex<OllamaEmbedder>,
}
// in open()/new():
octa: OctaIndex::new(OllamaEmbedder::new("http://localhost:11434", "nomic-embed-text", 768)),
```

**4. Index on ingest** — inside `ingest_source`, after the graph creates nodes, for
each new node feed its uri + content:

```rust
for node in newly_created_nodes {            // adapt to your node iterator
    self.octa.index_node(&node.uri, &node.content);
}
```

**5. Add the strategy** — extend the enum and its constructors:

```rust
pub enum Recall { WorkingSet, Around(String), Task(String), Semantic(String) }
impl Recall {
    pub fn semantic(text: impl Into<String>) -> Self { Recall::Semantic(text.into()) }
}
```

**6. Dispatch it** — add a match arm in `recall`, mirroring the `Around`/`Task`
arms (semantic anchors → existing causal expansion):

```rust
Recall::Semantic(text) => {
    let anchors = self.octa.semantic_anchors(text, proximity_k()); // e.g. 8
    let ids: Vec<NodeId> = anchors
        .into_iter()
        .map(|(uri, _score)| NodeId(normalize(&uri)))
        .collect();
    // reuse the same proximity expansion as Around/Task:
    let prox = /* build (&dist, decay, hops) as in Around, anchored on ids[0] */ None;
    self.assemble_window("semantic", ids, budget_tokens, prox)
}
```

**7. Expose it** over `ccos memory` / `ccos mcp` by accepting `strategy: "semantic"`
alongside `around`/`task`/`working_set`.

## Caveats (confirm against your source)

- The exact names `assemble_window`, `NodeId`, `normalize`, `proximity_k`,
  `region_member_ids`, and how nodes expose `content` at ingest are CCOS-internal —
  adapt the calls to their real signatures.
- For full proximity expansion, build the `prox` tuple as the `Around` arm does,
  anchored on the first semantic hit; or start with `None` (anchors only) and add
  expansion once it works.
- `OctaIndex::save` ↔ CCOS `checkpoint`; persist alongside `workspace.ccos`.

## Decoupled alternative (no CCOS edits)

Run the OctaSoma **MCP server** instead and have CCOS/the agent call it:
`octasoma-mcp memory.frac` exposes `recall` returning the same
`{strategy, items:[{uri,score,kind,content}], tokens}` shape. See
`docs/integration-ecosystem.md`.
