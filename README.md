# OctaSoma — 3-D Fractal Semantic Memory Engine

[![CI](https://github.com/checkupauto/octasoma/actions/workflows/ci.yml/badge.svg)](https://github.com/checkupauto/octasoma/actions/workflows/ci.yml)
[![rust](https://img.shields.io/badge/rust-stable%2C%20edition%202024-orange)](#)
[![unsafe](https://img.shields.io/badge/unsafe-forbidden-success)](#)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**OctaSoma** is a small, dependency-light, **100 % safe stable Rust** engine for
embedding-based memory. It projects high-dimensional embeddings to a 3-D point
with a *learned* (PCA) or *deterministic* (Johnson–Lindenstrauss) linear map,
indexes them in a **cache-efficient bucket point-region octree**, and answers
**exact `k`-nearest-neighbour** queries in the projected space — with
LZ4-compressed, versioned persistence.

It is deliberately honest about what 3 dimensions can and cannot do. See
[**Evaluation**](#evaluation) and [`docs/evaluation.md`](docs/evaluation.md):
the octree k-NN is *exact and fast*, but the 3-D projection is a **coarse
semantic router** — with a PCA projection it retrieves topically-correct
memories well when a corpus has few dominant themes, and degrades as topical
diversity grows. Use it as a compact, explainable, embeddable index — not as a
drop-in replacement for a high-dimensional ANN library.

```
            ┌──────────────────────────────────────────────┐
   embedding│  x ∈ ℝᴰ   (e.g. D = 256, 768)                │
   ─────────▶                                               │
            │   project_to_3d  (3×D matrix, PCA or JL)      │
            │            x  ↦  p ∈ ℝ³                       │
            └───────────────────────┬──────────────────────┘
                                     ▼
            ┌──────────────────────────────────────────────┐
            │  Bucket PR-octree  (contiguous Vec<OctreeNode>)│
            │   • 64-byte nodes (1 cache line)              │
            │   • leaf buckets of ≤ capacity items          │
            │   • dynamic, origin-centred world cube        │
            └───────────────────────┬──────────────────────┘
                                     ▼
            ┌──────────────────────────────────────────────┐
            │  Exact 3-D k-NN  (branch-and-bound + pruning) │
            │   nearest(p, k)  ≡  brute force, only faster  │
            └───────────────────────┬──────────────────────┘
                                     ▼
                       payload arena  (LZ4 on disk, .frac v3)
```

## Why it exists

A typical vector store keeps the full `D`-dimensional embedding and runs ANN
(HNSW, IVF-PQ) over it. OctaSoma asks a narrower question: *how far can you get
if you collapse the embedding to **three** dimensions and lean on a classic
spatial index?* The answer — measured, not asserted — is the point of the
project (and of the accompanying [paper](paper/)).

## Features

| Capability | How |
|---|---|
| Exact 3-D k-NN | Branch-and-bound octree descent with admissible cube-distance pruning |
| Learned projection | Power-iteration PCA (top-3 PCs, Hotelling deflation), unit-norm rows |
| Deterministic projection | Xorshift64 Johnson–Lindenstrauss, reproducible from a `u64` seed |
| Cache-friendly layout | `#[repr(C)]` nodes padded to exactly 64 bytes; contiguous `Vec` |
| Unbounded scale | Origin-centred world cube grows by doubling and rebuilds |
| Duplicate-safe | Co-located points share a leaf bucket; nothing is dropped |
| Persistence | Versioned `FRAC` v3 file, LZ4-compressed payload arena |
| Safety | `#![forbid(unsafe_code)]`-clean; one dependency (`lz4_flex`) |

## Install

### Command-line tool (simplest — no code)

```bash
git clone https://github.com/checkupauto/octasoma && cd octasoma
./install.sh                 # builds, tests, installs the `octasoma` CLI
```

Then store and recall memories straight from your shell:

```bash
octasoma remember "I prefer dark mode and the metric system"
octasoma recall   "what are my preferences?"
octasoma reflect  "preferences" -k 3      # a prompt-ready context block
octasoma stats
```

By default the CLI embeds text with a local [Ollama](https://ollama.com) model
(`nomic-embed-text`). Add `--hash` to run **fully offline** (exact-text recall,
no model needed). Run `octasoma help` for all options.

### As a Rust library

```bash
cargo add octasoma
# or in Cargo.toml:
#   octasoma = { git = "https://github.com/checkupauto/octasoma" }
```

### From source

```bash
git clone https://github.com/checkupauto/octasoma && cd octasoma
make build      # cargo build --release
make test       # 60+ tests (make stress for the 1M-insert soak)
make demo       # offline agent demo
```

Requires a stable Rust toolchain (edition 2024, Rust ≥ 1.85). The library has a
single dependency (`lz4_flex`); the CLI, agent, and HTTP embedder use only `std`.

## Quickstart

```toml
# Cargo.toml
[dependencies]
octasoma = { path = "." }   # or your published version
```

```rust
use octasoma::FractalMemory3D;

fn main() {
    // 1. Engine for 768-dim embeddings, deterministic JL projection (seed = 42).
    let mut mem = FractalMemory3D::new(768, 42);

    // 2. Insert observations (embedding + arbitrary byte payload).
    mem.insert(&embed("Rust's async runtime is fast."), Some(b"note A"));
    mem.insert(&embed("Python is great for prototyping."), Some(b"note B"));

    // 3. Exact nearest-neighbour query in the projected space.
    if let Some(payload) = mem.query(&embed("Tell me about Rust speed.")) {
        println!("recalled: {}", String::from_utf8_lossy(payload));
    }

    // 4. Top-k and raw distances are available too.
    let hits = mem.query_k(&embed("Rust speed"), 5);
    let ranked = mem.nearest_embedding(&embed("Rust speed"), 5); // (ItemId, dist²)

    // 5. Persist (LZ4) and reload.
    mem.save_to_disk("memory.frac").unwrap();
    let reloaded = FractalMemory3D::load_from_disk("memory.frac", 768).unwrap();
    let _ = (hits, ranked, reloaded);
}

// Bring your own embedder (Candle, ort, an HTTP call, …) — OctaSoma is agnostic.
fn embed(_text: &str) -> Vec<f32> { vec![0.0; 768] }
```

### Calibrating the projection with PCA

```rust
use octasoma::FractalMemory3D;

// Flat, row-major `num_samples × high_dim` calibration matrix.
let calibration: Vec<f32> = load_corpus_embeddings();
let num_samples = calibration.len() / 768;
let mut mem = FractalMemory3D::new_with_pca(768, &calibration, num_samples);
```

A PCA projection learned from a representative corpus is **strongly recommended**
— it is far better than a random projection at preserving topical structure
(see [Evaluation](#evaluation)).

## Agent layer

A small, 100 % Rust agent memory sits on top of the engine — `perceive` to store
text observations, `recall`/`reflect` to retrieve them. It is generic over an
`Embedder` trait, so it runs fully offline with the built-in `HashEmbedder` and
against a real model with `OllamaEmbedder` (a std-only HTTP client for a local
Ollama / OpenAI-compatible endpoint) — no extra dependencies either way.

```rust
use octasoma::{HashEmbedder, OctaSomaAgent};

let corpus = ["the user likes Rust", "the project is about octrees"];
let mut agent = OctaSomaAgent::calibrate(HashEmbedder::new(256), &corpus)?;

agent.perceive("the user just asked about fractal compression")?;
let context: String = agent.reflect("what does the user remember?", 3)?;

// Swap in a real model with one line — same agent code:
// let mut agent = OctaSomaAgent::new(
//     OllamaEmbedder::new("http://localhost:11434", "nomic-embed-text", 768), 42);
```

Run the offline demo with `cargo run --release --example agent_demo`. Details in
[`docs/agent.md`](docs/agent.md).

For a full agent integration there is a **memory kernel** — an opinionated routine
(`observe` / `step` / `recall_context`) bundled with a ready-made system prompt and
tool schema for wiring memory into an LLM. See
[`docs/integration-kernel.md`](docs/integration-kernel.md) and
`cargo run --release --example kernel_loop`.

## Evaluation

All numbers are reproducible with the bundled harness and are *machine-dependent*:

```bash
cargo run --release --example benchmark            # defaults: 50k × 256-D
cargo run --release --example benchmark -- 20000 128 16 500 10   # N D clusters queries k
```

On the evaluation machine, with `N = 50 000` clustered, unit-norm embeddings in
`D = 256` and 16 latent themes:

| Projection | cluster recall@1 | exact recall@1 | octree k-NN | speed-up vs brute-3D | inserts/s | LZ4 |
|---|---|---|---|---|---|---|
| **PCA (learned)** | **70.8 %** | ~0 % | 5.7 µs | 67× | 1.9 M | 5.5× |
| JL (random) | 13.0 % | ~0 % | 5.1 µs | 77× | 1.5 M | 5.5× |

How **cluster recall@1** (retrieving a memory from the *correct theme*) depends on
the number of latent themes (`N = 20 000`, `D = 128`):

| latent themes | 4 | 16 | 64 | 256 |
|---|---|---|---|---|
| **PCA** | **100 %** | **73 %** | 21 % | 3 % |
| JL | 33 % | 13 % | 4 % | 1 % |

**Reading of the results.**

- The octree k-NN is **exact** — identical to brute force, just 38–77× faster,
  and the speed-up grows with `N`.
- **Exact-item recall@1 ≈ 0 %**: a 3-D projection essentially never preserves the
  single true nearest neighbour of a rich high-D embedding.
- **Cluster (topical) recall** is what a memory actually needs, and a **PCA**
  projection delivers it well **when the corpus has few dominant themes**. Past a
  handful of themes the 3-D bottleneck dominates and recall falls off.
- Use OctaSoma where its strengths apply: few-topic agent memory, an explainable
  / visualisable spatial index, a coarse pre-filter, or teaching. For
  general-purpose high-recall retrieval, pair it with (or defer to) a full ANN.

## Documentation

| Document | Contents |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Data structures, octree, k-NN, projection, world growth |
| [`docs/api.md`](docs/api.md) | Full public API reference with examples |
| [`docs/agent.md`](docs/agent.md) | Agent layer: embedders, perceive/recall/reflect |
| [`docs/integration-kernel.md`](docs/integration-kernel.md) | Wiring memory into an AI agent: kernel API, system prompt, tool schemas |
| [`docs/file-format.md`](docs/file-format.md) | The `FRAC` v3 on-disk format, byte-by-byte |
| [`docs/evaluation.md`](docs/evaluation.md) | Methodology, full results, comparison vs other memory regimes |
| [`docs/positioning.md`](docs/positioning.md) | Prior art, the closest precedent, and what we can/can't claim |
| [`paper/`](paper/) | arXiv-style paper (English & French sources) |

Build the API docs locally with `cargo doc --open`.

## Tests

```bash
cargo test --release                               # 60+ unit & integration tests
cargo test --release --test stress -- --ignored    # heavy soak (1M inserts)
cargo clippy --all-targets -- -D warnings
```

A deliberately large suite covers: a **property check that the octree k-NN is
bit-identical to brute force** across hundreds of randomised configurations; an
**oracle** that interleaves inserts and queries and re-verifies exactness at every
step; **structural invariants** (items form a permutation across leaf buckets, all
node links valid); **persistence fuzzing** (round-trip fidelity + corruption
rejection); **edge cases** (NaN/inf, `k = 0`, dimension mismatch, 1 MiB payloads,
extreme world growth); **determinism**; the **agent + memory kernel**; and an
`#[ignore]`d **soak of 1,000,000 inserts** that also re-checks exactness at scale.

## Design notes & limitations

- **Three dimensions are the whole point and the whole limit.** OctaSoma is a
  study of, and a tool for, aggressive dimensionality reduction. It is not an
  HNSW competitor.
- **Linear projection only.** No non-linear manifold learning (UMAP/t-SNE); the
  map must be a single `3 × D` matrix so the index stays exact and cheap.
- **No deletion / TTL yet.** Insertion-only; eviction is on the roadmap.
- **In-memory writer.** Mutations take `&mut self`; concurrency is the caller's
  choice (e.g. wrap in `RwLock`, or rebuild-and-swap an `Arc`).

See [`docs/evaluation.md`](docs/evaluation.md) for the honest, detailed version.

## License

[MIT](LICENSE).
