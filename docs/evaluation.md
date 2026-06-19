# OctaSoma — Evaluation

This is the honest account of what OctaSoma does well and where it breaks down.
Every number below is produced by [`examples/benchmark.rs`](../examples/benchmark.rs)
and is **machine-dependent** (latency/throughput especially). Reproduce with:

```bash
cargo run --release --example benchmark                     # 50k × 256-D, 200 themes
cargo run --release --example benchmark -- 20000 128 16 500 10   # N D themes queries k
```

## Methodology

- **Data.** `N` unit-norm embeddings in `ℝᴰ`, drawn from `C` latent clusters
  ("themes"): each sample is a random cluster centre plus Gaussian-ish noise,
  renormalised. This gives a controllable *intrinsic dimensionality* via `C`.
- **Queries.** A separate, **held-out** set drawn from the same clusters with a
  different noise stream — never inserted. (Querying inserted points would make
  recall@1 a trivial 100 %.)
- **Ground truth.** Exact nearest neighbours in the **full `D`-dimensional**
  space, by brute force.
- **Metrics.**
  - *exact recall@k* — fraction of the true high-D `k`-NN recovered by the 3-D
    index (strict: same data points).
  - *cluster recall@k* — fraction of the returned items in the query's own
    cluster (the metric that matters for *topical* memory).
  - *latency* — mean time per query for the octree k-NN vs. a brute-force scan of
    the same 3-D points (both exact; isolates the index speed-up).
  - *throughput* — inserts/second. *compression* — LZ4 ratio on the arena.

## Results 1 — scale (`N = 50 000`, `D = 256`, `C = 16`, `k = 10`)

| Projection | cluster@1 | cluster@10 | recall@1 | recall@10 | octree k-NN | brute-force 3-D | speed-up | inserts/s | nodes |
|---|---|---|---|---|---|---|---|---|---|
| **PCA (learned)** | **70.8 %** | 69.8 % | 0.0 % | 0.3 % | 5.69 µs | 383.6 µs | **67×** | 1.86 M | 12 878 |
| JL (random) | 13.0 % | 12.3 % | 0.0 % | 0.0 % | 5.12 µs | 393.6 µs | 77× | 1.53 M | 12 517 |

Persistence, 5 000 structured text payloads:

| Payload arena (raw) | Arena (LZ4) | Ratio | Full `.frac` file |
|---|---|---|---|
| 241 872 B | 44 257 B | **5.47×** | 281 445 B |

## Results 2 — intrinsic dimensionality (`N = 20 000`, `D = 128`, `k = 10`)

**Cluster recall@1** vs. number of latent themes:

| themes `C` | 4 | 16 | 64 | 256 |
|---|---|---|---|---|
| **PCA** | **100.0 %** | **73.2 %** | 20.6 % | 2.6 % |
| JL | 33.2 % | 12.8 % | 4.2 % | 1.2 % |

Exact recall@1 stays `≈ 0 %` across the board. The octree k-NN speed-up is `≈ 38×`
at this scale (and grows with `N` — see Results 1).

## Interpretation

1. **The octree is exact and fast.** `nearest` returns precisely the brute-force
   answer (asserted in the test suite), with branch-and-bound pruning delivering
   38–77× over a linear 3-D scan; the margin widens as `N` grows.

2. **Three dimensions cannot preserve the *exact* nearest neighbour.** Collapsing
   a 128- or 256-D embedding to 3-D scrambles fine structure, so exact recall@1
   is essentially zero. This is intrinsic to the projection, not a bug in the
   index — and it is the central honest finding.

3. **Coarse topical structure *can* survive — under PCA, for few themes.** A
   learned PCA projection keeps the dominant axes of variance, so it routes a
   query to the right *theme* with high accuracy when there are only a handful of
   them (100 % at 4 themes, 73 % at 16). Three principal axes can separate only
   a few well-spread modes, so accuracy falls off as themes multiply (21 % at
   64, 3 % at 256).

4. **PCA ≫ JL for this task.** A random projection ignores where the variance is
   and trails PCA badly (e.g. 13 % vs 73 % cluster@1 at 16 themes). If you can
   afford a calibration pass, always use PCA.

5. **Storage is cheap.** 64-byte nodes, 28-byte items, and a 4–6× LZ4 ratio on
   text-like payloads make the on-disk footprint modest.

### A rule of thumb

> OctaSoma is a *coarse semantic router*. With a PCA projection it reliably
> retrieves topically-relevant memories when the corpus has roughly a dozen or
> fewer dominant themes. Beyond that, the 3-D bottleneck dominates and you want a
> higher-dimensional index.

## Where OctaSoma fits

**Good fits**

- Few-topic agent / assistant memory (a handful of personas, tasks, or domains).
- An **explainable, visualisable** spatial index — points live in real 3-D space.
- A compact, embeddable, `unsafe`-free, near-zero-dependency store.
- A cheap **pre-filter** in front of a heavier retriever, or a teaching artefact.

**Poor fits**

- General-purpose high-recall semantic search over diverse corpora — use HNSW /
  IVF-PQ / ScaNN on the full embedding.
- Tasks needing the *exact* nearest neighbour of rich embeddings.

## Threats to validity

- Synthetic, isotropic-noise clusters are friendlier than real embedding
  manifolds; treat cluster-recall numbers as an **upper** bound for real data.
- Latency/throughput are single-machine, single-thread, and release-build
  specific; re-run the harness on your target hardware.
- PCA calibration here uses up to 4 000 samples and 20 power-iterations; more of
  either helps marginally.

## Reproducing

```bash
cargo test --release        # correctness, incl. exact-k-NN-vs-brute-force
for C in 4 16 64 256; do
  cargo run --release --example benchmark -- 20000 128 $C 500 10
done
```
