# SciRust-based improvements & the CCOS premium-extension case

This document proposes concrete improvements to OctaSoma built from components of
[SciRust](https://github.com/CHECKUPAUTO/scirust) (the org's pure-Rust deterministic
deep-learning / scientific-computing framework), and assesses packaging the result as a
**premium extension for CCOS**. Every proposal below was checked against both codebases
(cited paths are real types/functions, not aspirations); each one names the exact SciRust
component and the exact OctaSoma integration site.

Guiding constraint: OctaSoma's default build stays lean (today: one dependency,
`lz4_flex`). Every SciRust dependency is opt-in behind a cargo feature or confined to
dev-dependencies; the scalar, bit-deterministic paths remain the defaults.

---

## Status (2026-07-02)

| Proposal | Status | Where |
|---|---|---|
| **D1** metrics + recall gate | ✅ landed | `src/metrics.rs`, `tests/recall_gate.rs` — recall@1 0.450 (sketch) vs 0.100 (3-D) CI-enforced |
| **B1** RCPS-certified shortlist | ✅ landed | `src/conformal.rs`, `SketchIndex::certify_shortlist`, `HybridMemory::calibrate_shortlist` |
| **A1** normalize-on-insert | ✅ landed | SKCH v2, sketches bit-identical, v1 migrates on load |
| **C1** validate-before-allocate | ✅ landed | `src/fileguard.rs` + all four loaders, hostile-file regressions |
| **A2** SIMD rerank | ✅ landed | `simd` feature, scirust-simd runtime dispatch, stable-verified |
| **A4** int8 tier | ✅ landed | `Precision::Int8`, SKCH v3, order-independent i32 dots |
| **C2** parallel PCA | ✅ landed | `compute_pca_projection_parallel`, bit-identical ∀ thread counts |
| **A3** SIMD sketching | ✅ landed | `with_simd_sketching()` (simd feature) — per-store path, recorded in SKCH v4 flags, mixed loads refused |
| **B2** conformal recall sets | ✅ landed | `MemoryKernel::recall_set` — dynamic-size recall with a coverage guarantee, calibrated on the explicit feedback log |
| **B3** temperature scaling | ✅ landed | `src/calibration.rs` + `RelevanceFeedback::fit_temperature` — binary temperature on score logits, ECE-verified |
| Feedback channel (B2/B3 prereq) | ✅ landed | `src/feedback.rs`, `MemoryKernel::feedback`, `memory_feedback` tool, MCP `feedback` tool — the explicit-channel decision on record |
| **C3** NSGA-II tuning | ✅ landed | `examples/pareto_tuning.rs` — seeded Pareto front over (bits, shortlist); scirust-evo as dev-dependency (accepted policy) |
| **C4** symreg recall law | ✅ landed | `examples/recall_law.rs` — grid sweep + Pareto front of formulas, validity domain printed |
| **B4** learned projection | ⏳ open | heaviest dep (scirust-core); research-grade |
| **A5** wgpu batch scoring | ⏳ plan ready | vendor `scirust-gpu/src/wgpu_backend.rs` (827 lines, self-contained — its `wgpu` feature would drag scirust-core) behind a `gpu` feature (wgpu 0.20 + pollster + bytemuck); `SketchIndex::scores_batch` = Q×Eᵀ GEMM, F32 tier only, tolerance-validated vs the CPU oracle; CI: scirust's recipe (`apt-get install mesa-vulkan-drivers vulkan-tools`, graceful-skip tests without an adapter). Needs its own CI iteration — no Vulkan in the dev container to pre-validate |
| NF4 cold tier | ✅ landed | `Precision::Nf4` — 8× codes+scale (norm-corrected), dequantize-free LUT scoring, SKCH v4 |

## A. Performance

### A1. Normalize-on-insert: cosine rerank becomes a single dot — **S**
- **SciRust pattern**: `DenseIndex::add`/`search` (`scirust-retrieval/src/index.rs`) —
  vectors L2-normalised once on add; queries scored with a plain dot.
- **OctaSoma site**: `SketchIndex` — `embeddings` storage, `cosine_full`, and its callers
  `nearest`, `rerank`, `scores` (`src/sketch.rs`).
- **Benefit**: ~3× fewer flops and less memory traffic in the exact-cosine rerank, in
  `scores()` (the O(N·d) viewer heat-colouring scan) and in cross-shard `recall_global` —
  the documented PrecisionSketch bottleneck. Zero new dependencies (pattern only).
- **Risk**: scores change in the last bits (normalise-then-dot vs fused cosine); score
  regression fixtures need regeneration, SKCH readers must know vectors are pre-normalised.

### A2. Vectorised SimHash sketching (bits×dim GEMV) — **S**
- **SciRust**: `SimdBackend::sgemv_f32` (`scirust-simd/src/dispatch.rs`, AVX2 FMA rows,
  SSE2/scalar fallbacks, runtime feature detection).
- **OctaSoma site**: `SimHasher::sketch` and its row-major `planes` layout (`src/sketch.rs`).
- **Benefit**: ~8× faster sketching — insert throughput on the precision tier currently
  pays 6–12× the octree-insert cost for wide sketches.
- **Risk**: sign-of-dot near zero can flip between the scalar and SIMD accumulation
  orders. Sketches are self-consistent per store, so gate behind the same opt-in `simd`
  feature as A3 and document that stores are not byte-portable across the flag.

### A3. Optional `simd` feature: SIMD dot kernel for the rerank path — **M**
- **SciRust**: `SimdBackend::sdot_f32` → `sdot_f32_avx2` (8-wide FMA,
  `is_x86_feature_detected!` dispatch) in `scirust-simd`.
- **OctaSoma site**: the exact-cosine rerank inside `SketchIndex::nearest`, `rerank`,
  `scores` (`src/sketch.rs`); `octacore`'s per-item cosine in `Cascade::recall`.
- **Benefit**: 4–8× on the rerank stage at 768-d (shortlist 256 ≈ 200k scalar MACs per
  query today).
- **Risk**: SIMD accumulation is not bit-identical to scalar; near-tie rankings can
  differ across platforms. Keep scalar as default; `simd` is opt-in and documented as
  trading bit-portability for speed. Note OctaSoma has no SciRust dependency today —
  this feature introduces the first one, off by default.

### A4. Int8/NF4 quantized embedding tier with integer scoring (SKCH v2) — **M**
- **SciRust**: `compute_scale`, `quantize_tensor`, `dequantize_tensor`, i32-accumulating
  `matmul_int8` (`scirust-core/src/quantization.rs`); `nf4_quantize`/`nf4_dequantize`
  (same file) for a colder 4-bit tier.
- **OctaSoma site**: `SketchIndex` embedding storage, insert/rerank/scores, and the SKCH
  persistence block (`src/sketch.rs`).
- **Benefit**: 4× (int8) to 8× (NF4) smaller precise tier — directly fixes the documented
  "3 KB/item at 768-d erases the compactness story" limitation — plus 2–4× faster
  brute-force scans from quartered memory bandwidth. Integer i32 dots are
  order-independent by construction, so a future parallel scan stays bit-deterministic.
- **Risk**: quantization shifts cosine by up to ~1e-2 near ties; keep a
  `precision = f32 | int8 | nf4` constructor knob and measure recall deltas with the D1
  harness before making anything but f32 a default.

### A5. Optional wgpu GEMM backend for batched scoring — **L**
- **SciRust**: `RawComputeBackend::gemm_f32` (`scirust-gpu/src/lib.rs`), WGSL, validated
  against the CPU oracle on software Vulkan in CI.
- **OctaSoma site**: `SketchIndex::scores` (batched sibling), `HybridMemory` scored
  exports, bulk benchmark sweeps.
- **Benefit**: turns the acknowledged O(N)-per-query brute-force ceiling into one batched
  matmul for the workloads that genuinely score everything (viewer heat-maps, global
  cross-shard recall, benchmark sweeps) at N ≥ 10⁶.
- **Risk**: GPU output is tolerance-validated (1e-4), not bit-exact — never the default
  path; determinism-sensitive callers stay on CPU.

## B. Recall quality & statistical guarantees

### B1. RCPS-certified shortlist sizing: retire the magic 256 — **S**
- **SciRust**: `rcps_select`, `hoeffding_ucb`, `learn_then_test`
  (`scirust-core/src/nn/conformal.rs`).
- **OctaSoma site**: the `shortlist` parameter of `SketchIndex::nearest`,
  `HybridMemory::default_shortlist`/`with_shortlist`, and `HybridCascade`'s hardcoded ×4
  widening (`src/sketch.rs`, `src/hybrid.rs`).
- **Benefit**: upgrades `docs/precision-sketch.md`'s empirical "recall@512 = 88.7%" into a
  PAC-style certificate — *miss rate ≤ α with probability ≥ 1−δ on this corpus* — and
  auto-picks the cheapest shortlist per store instead of one global constant.
- **Risk**: assumes calibration queries are exchangeable with production queries; query
  drift voids the certificate (pairs with B2's online tracking; document the caveat).

### B2. Conformal recall sets in the memory kernel — **M**
- **SciRust**: `conformal_quantile` + `AdaptiveConformal` (online, drift-tracking) in
  `scirust-core/src/nn/conformal.rs`.
- **OctaSoma site**: `MemoryKernel::recall_context` / `KernelConfig.top_k`
  (`src/kernel.rs`); the MCP `recall` tool's fixed `k = 5`.
- **Benefit**: recall returns *as many memories as needed* with a distribution-free
  coverage statement — the set shrinks when the query is confidently answered, grows when
  uncertain. Token-frugal, matching the cascade's ~26-tokens/turn story.
- **Risk**: needs a genuine relevance-feedback channel; calibrating on self-retrieval
  pairs would overstate coverage.

### B3. Per-shard temperature scaling for cross-region scores — **M**
- **SciRust**: `temperature_scale`, `nll`, `expected_calibration_error`
  (`scirust-core/src/nn/calibration.rs`).
- **OctaSoma site**: `ShardedMemory::recall_global_scored` — self-documented as "a coarse
  heuristic" (`src/sharded.rs`) — and `ShardedHybrid::recall_global`.
- **Benefit**: fixes a documented limitation (per-region scores are not comparable across
  shards) and upgrades the MCP contract from "score = raw cosine" to a calibrated
  relevance probability, measurable via ECE.
- **Risk**: needs labeled pairs per region; small shards give noisy temperatures (fall
  back to T = 1 under a pair-count threshold).

### B4. Contrastive learned 3-D projection (supervised PCA replacement) — **L**
- **SciRust**: `contrastive::{ProjectionHead, train, ContrastiveConfig,
  ProjectedEncoder}` (`scirust-retrieval`, feature `learned`).
- **OctaSoma site**: `FractalMemory3D::new_from_calibration` / `compute_pca_projection`
  (`src/lib.rs`), behind the `Embedder` trait.
- **Benefit**: attacks the engine's acknowledged central finding — 3-D cluster recall
  collapsing from 100% at 4 themes to ~3% at 256 — with a neighbourhood-preserving
  learned projection; `docs/positioning.md` §5 already names a learned projection as the
  known improvement path.
- **Risk**: heaviest dependency of the list (`learned` pulls `scirust-core`); the SciRust
  repo pins a toolchain via `rust-toolchain.toml` while OctaSoma tracks stable — pin a
  SciRust rev and CI-test the pairing. Strictly optional feature.

## C. Determinism & robustness

### C1. Hardened, fingerprinted persistence (FRAC v4 / SKCH v2) — **S**
- **SciRust pattern**: validate-before-allocate header handling in
  `load_safetensors` (`scirust-core/src/io/safetensors.rs`).
- **OctaSoma site**: `FractalMemory3D::load_from_disk` (unchecked
  `Vec::with_capacity(node_count)` from file-supplied counts) and the SketchIndex /
  ShardedHybrid loaders (`src/lib.rs`, `src/sketch.rs`, `src/sharded.rs`).
- **Benefit**: closes a real hole — a hostile 24-byte file can currently request a
  multi-GB allocation — and adds an integrity footer no OctaSoma format has today.
- **Risk**: format bump; loaders must keep accepting footer-less FRAC v3 / SKCH v1
  (warn, don't fail) or existing stores (including MCP store dirs) break.

### C2. Order-independent reductions for thread-parallel PCA calibration — **M**
- **SciRust**: `reproducible_sum` / `reproducible_dot` / `fsum_canonical`
  (`scirust-core/src/reproducible.rs`, permutation-bit-identity proven by tests).
- **OctaSoma site**: `compute_pca_projection` and its `mat_vec_mul`/mean loops
  (`src/lib.rs`), consumed by `new_with_pca` and `ShardedMemory::build_pca`.
- **Benefit**: PCA calibration is OctaSoma's one genuinely heavy offline step and is
  single-threaded today precisely because naive parallel sums would break bit-exactness.
  Order-independent reductions unlock N-thread calibration with bit-identical output —
  the same guarantee SciRust's `DataParallelTrainer` ships.
- **Risk**: `fsum_canonical` sorts per reduction (O(n log n)); restructure to reduce once
  per output row, not per partial.

### C3. NSGA-II Pareto tuning of (bits, shortlist, cascade width) — **M**
- **SciRust**: `Nsga2::seeded` / `evolve` (`scirust-evo/src/lib.rs`, seeded, reproducible).
- **OctaSoma site**: `SimHasher::new` bits, `HybridMemory` shortlist, `HybridCascade`'s
  ×4 constant; calibrated on the synthetic clustered corpus from
  `examples/precision_sketch.rs`.
- **Benefit**: the recall-vs-cost tradeoff currently expressed as two hand-picked
  constants becomes a reproducible, seeded Pareto front per corpus — answering "how many
  bits do I need for my data?".
- **Risk**: `scirust-evo` pulls rand/rayon/tracing — keep it a **dev-dependency** (a
  tuning example/binary), never in the library build.

### C4. Symbolic-regression recall law — **M**
- **SciRust**: `discover` (`scirust-symreg/src/lib.rs`) with constant fitting via
  SciRust's own symbolic differentiation.
- **OctaSoma site**: sweep data from the D1 harness over `SketchIndex` parameters; a
  closed-form `recall(N, bits, k, shortlist)` used to auto-size stores.
- **Benefit**: an interpretable, paper-ready formula for the SimHash tier — a strong fit
  for the project's explainability brand.
- **Risk**: fitted laws extrapolate unsafely outside the swept grid; synthetic clusters
  are an upper bound for real embeddings. Publish the law with its validity domain.

## D. Measurement (do this first)

### D1. Standard IR metrics harness + CI recall-regression gate — **S**
- **SciRust**: `scirust-retrieval::metrics` — `recall_at_k`, `precision_at_k`,
  `reciprocal_rank`, `mean_reciprocal_rank`, `average_precision`, `ndcg_at_k`.
- **OctaSoma site**: `examples/benchmark.rs`, `examples/precision_sketch.rs` (ad-hoc
  recall counters today), per-`QueryStrategy` evaluation in `HybridMemory`.
- **Benefit**: the bespoke "cluster recall" numbers become standard,
  ANN-literature-comparable metrics; recall regressions get caught in CI; and it is the
  measurement plumbing every proposal above needs to prove its gain honestly.
- **Risk**: minimal. Vendor the ~150 lines (or depend with `default-features = false`) to
  keep the dependency budget at zero.

### Suggested sequencing

1. **Quick wins (S)**: D1 (measure first) → A1 → C1 → A2 → B1.
2. **Core upgrades (M)**: A4 (SKCH v2 int8) → A3 (`simd` feature) → B3 → C2 → C3.
3. **Research-grade (M/L)**: B2 → C4 → B4 (learned projection) → A5 (wgpu).

---

## E. Can this be a premium extension for CCOS?

**Yes — and unusually little needs to be invented: both sides already ship the
mechanics.** Assessment grounded in the actual code:

### Technical fit (adapter surface already exists)
- `octacore/` (staged in this repo) already composes the triad: its optional `ccos`
  feature adapts CCOS's `ExternalMemory` into `CausalScope` (causal region → exact-cosine
  rerank → token-budget compaction; validated 99% hit at ~26 tokens/turn, ~137× fewer
  tokens than naive). The reverse lever exists on the CCOS side too:
  `retrieval::Encoder` (CCOS `src/retrieval.rs`) can take OctaSoma as its semantic
  backend, and `ccos-memory-runtime` defines a neutral `MemoryProvider` trait.
- Both projects ship dependency-free MCP stdio servers, so agents can compose them today
  with zero linking.
- **The premium gate pattern already exists in CCOS**: `RetrievalAccess::unlock` gates
  the adaptive-retrieval tier behind `Feature::AdaptiveRetrieval`, verified by an
  offline ed25519 license (`src/license.rs`, ROADMAP §P4 — fail-closed, air-gap-safe,
  core never degraded). Adding a `Feature::SemanticMemory` for an OctaSoma-backed tier
  is a small, precedent-following change, not new infrastructure.

### Licensing fit (the commercial rails are laid)
- All four repos (CCOS, octasoma, scirust, SLHAv2) are dual-licensed identically —
  PolyForm Noncommercial 1.0.0 + paid commercial license — with a single copyright
  holder and a CLA preserving both-license rights. Nothing needs relicensing to sell a
  combined extension.
- SciRust already *operates* per-module premium sales: `scirust-license` verifies signed
  entitlement files (Merkle-rooted module list, optional node-locking), and
  `scirust-retrieval` is itself a gated premium module. "OctaSoma Semantic Pro" fits the
  same mold; CCOS's ed25519 gate even documents `scirust-license` as the linkable
  alternative for node-locked schemes.

### Packaging recommendation
- **Free (noncommercial) tier**: CCOS core untouched (causal graph, recall, replay —
  CCOS's own rule is that the core is never gated), OctaSoma standalone under PolyForm NC.
- **Premium "Semantic Memory" extension for CCOS**: the octacore cascade
  (causal-scope → semantic rerank → budget compaction), the SimHash precision tier with
  the B1/B2 conformal *guarantees* (a genuine differentiator: "recall with a
  certificate"), calibrated cross-shard scoring (B3), the 3-D viewer/explainability lens
  (matches the existing `TensionVisualization` Pro feature), and the B4 learned-projection
  improvement loop (the direct analogue of CCOS's already-premium `AdaptiveRetrieval`
  improvement loop).

### Honest risks
- **Dependency chain**: octacore ↔ octasoma ↔ ccos/scirust are git-pinned, unpublished
  crates (`publish = false`, `license-file`); version drift is manual. A crates.io-less
  distribution needs a rev-pin discipline and a CI job that builds the full triad.
- **Toolchain spread**: octasoma/octacore are edition 2024, CCOS is edition 2021, SciRust
  pins its own toolchain — a combined premium artifact needs one tested MSRV matrix.
- **Semantics need a real embedder**: the offline `HashEmbedder` is non-semantic; the
  product story depends on a local Ollama (or equivalent) being present and healthy.
- **Proof before pricing**: run the D1 metrics harness on realistic corpora first — the
  99%-hit cascade number comes from small validated runs; a paying tier needs the same
  number on customer-shaped data.
- **Single-maintainer bus factor** and PolyForm-NC adoption friction (not OSI-approved)
  are the standard org-wide caveats.

**Verdict**: technically ready (adapters and gates exist on both sides), legally trivial
(same owner, same dual license, CLA), commercially precedented (SciRust already sells
modules this way). The work is productization — one pinned dependency matrix, one
`Feature::SemanticMemory` gate, the D1 evidence — not research.
