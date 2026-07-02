# OctaCore

**The intelligent assembly of the CHECKUPAUTO memory triad** — CCOS (causal),
OctaSoma (semantic), SLHAv2 (attention) — into a single recall. OctaCore is not a
fourth memory; it is the thin layer that makes the other three behave as one, the
cascade the OctaSoma benchmark validated: **99 % hit at ~26 tokens/turn on real
data (~137× fewer than naive injection), where no single brick suffices.**

```text
  query
    │  1. CAUSAL    (CCOS)      narrow to a small causal region
    ▼
  region ──► 2. SEMANTIC (OctaSoma)  exact cosine rerank within the region
    ▼                                (the embedding finisher that lands the hit)
  token-budgeted context window
```

SLHAv2 is the inference-time KV-cache attention kernel that *consumes* the produced
window; OctaSoma serves it as a **visualisation lens** (project tile latents to 3-D),
not a text reranker — the honest role our measurements support.

## Quickstart

```rust
use octacore::{Cascade, InMemoryScope};
use octasoma::HashEmbedder;

let scope = InMemoryScope::new().region(
    &["sql", "database", "pool"],
    &[("sym:src/db.rs:pool", "manage a pool of reusable database connections")],
);
let core = Cascade::new(scope, HashEmbedder::new(64));
let window = core.recall("open a pooled database connection", 3, 64).unwrap();
assert_eq!(window.items[0].uri, "sym:src/db.rs:pool");
```

```bash
cargo run --release --example cascade_demo     # offline, deterministic
cargo test --release                           # default build (octasoma only)
```

## The three functions

| Function | Owner | OctaCore surface |
|---|---|---|
| **Causal / structural** | CCOS | `trait CausalScope` — `ccos_adapter::CcosScope` (`--features ccos`) wraps `ccos::ExternalMemory` |
| **Semantic / spatial** | OctaSoma | the `Embedder` + exact cosine rerank inside `Cascade::recall` |
| **Working memory / attention** | SLHAv2 | `slha::kv_cache_view` (`--features slha`) — the visualisation lens |

```bash
cargo build --features ccos      # real CCOS causal scope
cargo build --features slha      # SLHAv2 KV-cache lens via OctaSoma
```

The `ccos` and `slha` features pull the upstream crates by git and require them to
build; the default build needs only OctaSoma and is fully offline.

## Use it from an AI agent (MCP)

OctaCore ships an [MCP](https://modelcontextprotocol.io) server so any
MCP-compatible agent (Claude Desktop, IDE assistants, custom clients) can drive
the cascade as a memory tool — built only with the `mcp` feature:

```bash
cargo build --release --features mcp     # builds the `octacore-mcp` binary
```

It speaks JSON-RPC 2.0 over stdio (the MCP stdio transport) and exposes the tools
`remember`, `recall`, `stats` and `clear`. Point your client at the binary — e.g.
Claude Desktop's `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "octacore": {
      "command": "/abs/path/to/target/release/octacore-mcp",
      "env": { "OCTACORE_MCP_STORE": "/abs/path/to/octacore-memory.json" }
    }
  }
}
```

Tool schemas, an example session, and other clients are documented in
[`docs/MCP.md`](docs/MCP.md). The server runs the offline, deterministic cascade
(`HashEmbedder` + the built-in causal scope), so it needs no model or network.

## Status & staging

This directory — `octacore/` inside the OctaSoma repository (its own isolated
workspace) — is the crate's **canonical home**. The standalone
`CHECKUPAUTO/octacore` repository is **archived** (read-only): it was published from
here, evolved independently (it grew the MCP server), and that evolution has been
merged back into this staging, which has been the newer of the two ever since.

To materialise a standalone tree (for an archive refresh, or a future re-split into
its own repository), run from the octasoma checkout:

```bash
scripts/extract_octacore.sh /path/to/standalone-dir
# then follow the printed git commands (init · add · commit · remote add · push)
```

The script rewrites the OctaSoma dependency from the local path to a git dependency
pinned to the **release tag** this crate is verified against (see the `tag=` line in
the script).

Everything else (the `ccos`/`scirust` git deps, the features, the example) is already
in its final form.

## License

Dual-licensed: free for noncommercial & personal use under the
[PolyForm Noncommercial License 1.0.0](LICENSE.md); commercial use requires a
separate license (contact@checkupauto.fr). See [`LICENSING.md`](LICENSING.md).
