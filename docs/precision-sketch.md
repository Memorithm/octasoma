# High-precision recall — SimHash sketches

OctaSoma's 3-D projection is a *coarse router*: it discards the angular structure
that exact retrieval needs (exact recall@1 ≈ 0%). The **SimHash** tier
([`src/sketch.rs`](../src/sketch.rs)) recovers most of it cheaply, in 100% safe,
stable Rust. It is the precision answer — not a Hilbert/SIMD rewrite of the 3-D
index, and not a non-linear projection (UMAP/t-SNE would break the cheap, exact,
incremental, explainable index and still wouldn't preserve fine angles in 3-D).

## How it works

`bits` random hyperplanes turn an embedding into `bits` sign bits (Charikar 2002);
the Hamming distance between two sketches estimates the angle between the embeddings
(`Hamming/bits ≈ θ/π`), via `u64::count_ones()` (POPCNT). [`SketchIndex`] stores a
compact sketch **and** the full embedding per item (flat, contiguous), and recalls in
two tiers:

1. **Shortlist** — scan the sketches with `hamming()`, partial-select the top `M`
   (`select_nth_unstable`, O(N), no full sort).
2. **Rerank** — exact cosine over those `M` stored embeddings → top `k`.

The hybrid's recall@1 equals the sketch's recall@`M` (the exact rerank finds the true
neighbour iff it is in the shortlist), so two knobs trade precision for cost: **bits**
(sketch fidelity) and **M** (shortlist size; rerank is one dot product per candidate).

## Measured

`examples/precision_sketch.rs`, N=20000 unit embeddings, D=768, 16 themes, 300
queries. The table is **recall@M = the hybrid's recall@1 at shortlist M** (vs the
exact full-D nearest neighbour). 3-D is bits-independent.

| method | recall@1 | recall@32 | recall@128 | recall@512 |
|---|---|---|---|---|
| 3-D PCA (octree) | 0.0% | 3.0% | 10.7% | 46.7% |
| SimHash-256 | 1.7% | 12.0% | 30.7% | 70.3% |
| SimHash-512 | 1.7% | 17.0% | 40.0% | 82.3% |
| SimHash-1024 | 1.7% | 21.7% | **52.7%** | **88.7%** |

Reproduce / sweep: `cargo run --release --example precision_sketch -- N D C Q BITS [SHORTLIST]`.

## Measured on real embeddings (Ollama)

A sanity check that the same shortlist→rerank path works end to end with a real model,
not only synthetic vectors. Corpus: OctaSoma's own source as CCOS-style nodes
(`scripts/rs_to_nodes.sh src`, **N=310** — `file:` module docs + `sym:` signatures).
Queries: **13 paraphrased** natural-language questions, each targeting a `file:` uri
(`octacore/examples/semantic_queries.tsv`; `nomic-embed-text`, D=768; recall via
OctaCore `recall_global`).

| embedder | recall@1 | recall@5 |
|---|---|---|
| HashEmbedder (exact text) — *baseline* | 0.0% | 7.7% |
| Ollama `nomic-embed-text`, 256-bit | 30.8% | 46.2% |
| Ollama `nomic-embed-text`, 1024-bit | 30.8% | 46.2% |

The near-zero exact-text baseline confirms the queries are genuine paraphrases (no
literal overlap to exploit), so the lift to **46% recall@5 is real semantic retrieval**.
Three honest caveats specific to this run:

- **256-bit and 1024-bit are identical here** because the default shortlist (256)
  covers most of the 310-node corpus — the sketch reranks nearly everything, so it is
  near-exact and its *width* cannot matter. The bit-width effect needs `N ≫ shortlist`,
  which is exactly the synthetic table above (recall@512: 70%→89% as bits go 256→1024).
  Force it on a small corpus with `--shortlist 32` (the bench warns when the shortlist
  covers most of the corpus).
- **Strict ground truth.** A query counts as a hit only if the exact `file:` uri is
  returned; a sibling `sym:` node from the *same file* is a miss. This understates
  "found the right file", so recall@5 is the fairer figure.
- **Latency** (~27 ms/query) is dominated by the query's embedding HTTP round-trip to
  Ollama, not the index — `recall_global` embeds the query text internally.

Reproduce:
```bash
cd octacore
../scripts/rs_to_nodes.sh ../src > nodes.tsv
cargo run --release --example recall_global_bench -- \
  --corpus nodes.tsv --queries examples/semantic_queries.tsv \
  --url http://localhost:11434 --model nomic-embed-text --dim 768 --bits 1024
```

## Reading & recommended defaults

- **SimHash ≫ 3-D at every shortlist** — e.g. recall@512: 47% (3-D) → 89% (1024-bit).
  A SimHash shortlist + exact rerank recovers most of the precision the projection
  threw away, at ~10–23× less than a brute-force scan.
- **More bits → more recall**, ~linear in scan cost and storage (32 / 64 / 128
  bytes/item at 256 / 512 / 1024 bits).
- **Bigger shortlist → more recall**, linear in rerank cost (one stored-embedding dot
  product per candidate).
- **Defaults:** 256-bit with a shortlist of ≥256 is a good cheap baseline; for higher
  precision use 1024-bit and shortlist 512+. OctaCore's `recall_global` defaults to
  256-bit / `shortlist = max(k·32, 256)`; raise the width with
  `Cascade::with_sketch_bits(1024)`.

## Where it fits

| case | mechanism | precision |
|---|---|---|
| **scoped** (region known) | CCOS narrows → exact full-D cosine rerank within 50–200 items | near-exact, ~free |
| **global** (no scope) | SimHash shortlist → exact full-D cosine rerank ([`SketchIndex`], OctaCore `recall_global`) | this table |
| coarse / explainable / viewer | 3-D PCA (per region via `ShardedMemory::build_pca`) | the router; ~0% exact, by design |

So the 3-D layer stays the cheap, explainable, visualisable coarse router; SimHash is
the high-precision tier for the global case; and the region rerank covers the scoped
case. This is the honest precision story end to end.
