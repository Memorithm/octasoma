# OctaSoma — Agent layer

The agent layer turns the raw [`FractalMemory3D`](architecture.md) engine into a
text-in / text-out memory for an agent loop. It is 100 % Rust and, with the
built-in embedder, fully offline — no model server and no extra dependencies.

Two pieces:

- [`Embedder`](api.md) — a trait abstracting "text → vector".
- `OctaSomaAgent<E: Embedder>` — `perceive` (store) and `recall`/`reflect`
  (retrieve), generic over the embedder.

## Embedders

| Type | Dependencies | Semantic? | Use |
|---|---|---|---|
| `HashEmbedder` | none (std) | **No** — a hash | Tests, demos, reproducible offline pipelines |
| `OllamaEmbedder` | none (std `TcpStream`) | Yes (via the model) | A local Ollama / OpenAI-compatible endpoint |

`HashEmbedder` maps the same text to the same unit vector (FNV-1a seed →
`DeterministicRng`). It is not semantic, but it makes round-trips exact and
deterministic, which is ideal for testing and demos.

`OllamaEmbedder` POSTs `{"model","prompt"}` to `http://host:port/api/embeddings`
using only the standard library (plain `http://`, `Connection: close`) and reads
the `"embedding"` array back. It targets a localhost model server; for TLS or
remote hosts, implement `Embedder` with your preferred HTTP client.

```rust
use octasoma::{Embedder, OllamaEmbedder};

let embedder = OllamaEmbedder::new("http://localhost:11434", "nomic-embed-text", 768);
let v = embedder.embed("hello")?;   // Result<Vec<f32>, EmbedError>
```

## The agent

```rust
use octasoma::{HashEmbedder, OctaSomaAgent};

// Learn a PCA projection from a calibration corpus, then build the agent.
let corpus = ["fact A", "fact B", "fact C"];
let mut agent = OctaSomaAgent::calibrate(HashEmbedder::new(256), &corpus)?;

// Perception loop: store observations (the text is the payload).
agent.perceive("the user prefers Rust")?;
agent.perceive("the task is about octrees")?;

// Reflection loop: retrieve context for the prompt.
let context: String = agent.reflect("what does the user prefer?", 3)?;

// Or get the raw memories.
let hits: Vec<String> = agent.recall("octrees", 5)?;

// Persistence.
agent.save("memory.frac")?;
let agent = OctaSomaAgent::from_file(HashEmbedder::new(256), "memory.frac")?;
```

### Methods

| Method | Description |
|---|---|
| `new(embedder, seed)` | Agent with a deterministic JL projection |
| `calibrate(embedder, corpus)` | Agent with a PCA projection learned from `corpus` (not stored) |
| `from_file(embedder, path)` | Load a saved `.frac` and attach an embedder |
| `perceive(text)` | Embed and store an observation |
| `recall(query, k)` | The `k` topically-nearest memories, nearest first |
| `reflect(query, k)` | The `k` memories joined into one context block |
| `save(path)` / `len()` / `is_empty()` / `core()` | Persistence & introspection |

All embedding-bearing methods return `Result<_, EmbedError>`; calibration that
hits a network/model error therefore surfaces it rather than silently degrading.

## A note on quality

Retrieval quality is governed by the engine's 3-D projection (see
[evaluation.md](evaluation.md)): the agent retrieves *topically* relevant
memories well with a PCA projection over a few-theme corpus, and is a coarse
router rather than an exact recaller. Choose the embedding dimensionality and
calibration corpus accordingly.

## Runnable demo

```bash
cargo run --release --example agent_demo
```

The demo runs the full perceive → reflect → save → reload loop offline with
`HashEmbedder`, and shows the one-line switch to `OllamaEmbedder`.
