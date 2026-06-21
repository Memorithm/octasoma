# OctaSoma — Positioning & prior art (research note)

A cited assessment of where OctaSoma sits relative to existing work, and what we
can and cannot claim. Compiled from a multi-source literature pass; every load-
bearing statement carries a reference.

## Bottom line

OctaSoma's **contribution is not a new algorithm**, and — importantly — **not even
a new pipeline**: a 2025 ECML-PKDD paper (Ellendula & Bajaj, §1) already clusters
embeddings, projects them to 3-D, and indexes them in octrees for RAG, beating
FAISS-IVF 4.2×. Octrees, branch-and-bound $k$-NN, PCA, Johnson–Lindenstrauss and
LZ4 are all classical too. What is defensible and ours is **(i) an honest empirical
characterization of the 3-D extreme** (when it works, when it fails, PCA≫JL),
**(ii) a clean, exact, reproducible, safe-Rust reference artifact** framed as
**LLM-agent memory** (kernel, CLI, persistence), and **(iii) a head-to-head
comparison against other memory types**. Framed that way it is original and useful;
framed as "a new method", it is neither — and a writeup **must cite and
differentiate from Ellendula & Bajaj**. The literature below justifies the design
and bounds the claims.

## 1. Novelty — is "embeddings → 3-D → octree" already done?

**Yes — there is one close, recent, peer-reviewed precedent**, independently
verified across arXiv, Springer and dblp:

> **A. S. Ellendula & C. Bajaj**, *Self-Balancing, Memory-Efficient, Dynamic
> Metric-Space Data Maintenance for Rapid Multi-Kernel Estimation*, **ECML-PKDD
> 2025** (arXiv:2504.18003). It *"clusters embeddings (k-means), projects each
> cluster to 3-D, and maintains per-cluster dynamic octrees"*, reporting a hybrid
> Octree–FAISS retriever with **O(log n) updates and ≈4.2× faster retrieval than
> FAISS-IVF** (reported ≈96% accuracy — treat the exact figure cautiously).

That is essentially OctaSoma's core pipeline (embeddings → 3-D projection →
octree → retrieval). **So the technique is not original to us.** This is a caution
*and* a validation: a strong venue published the idea and it beats FAISS-IVF, so the
approach is sound — but we cannot claim to have invented it, and any writeup **must
cite and differentiate** from it.

**How OctaSoma honestly differs:**
- **Scope / artifact** — LLM-agent long-term memory (perceive/reflect kernel,
  system prompt, tool schema, CLI, versioned persistence) vs. their importance-
  sampling / kernel-density framing with RAG as one case study.
- **Projection** — they use **per-cluster** JL operators (k-means *first*);
  OctaSoma uses a **single global** PCA *or* JL map and **quantifies PCA≫JL**.
  Their per-cluster design is very likely *why* their accuracy holds where our
  single-global projection degrades past a few topics — a concrete, citable
  insight (and a clear improvement path for us, §5).
- **Emphasis** — they headline the positive (4.2× faster, high accuracy); we add
  the **honest negative characterization** (recall@1≈0%, topical-recall falloff,
  "coarse router") plus an **exact, reproducible, safe-Rust** implementation.

Otherwise: mainstream vector stores (FAISS, HNSW, Qdrant; Rust qdrant/tinyvector/
agentmemory) keep full dimensionality; "semantic octrees" elsewhere are 3-D
vision/robotics on *natively* 3-D data (OctoMap, octree diffusion); and the name
"OctaSoma" is unused (only "octasome" in biology). **Net: the algorithm is
novel-adjacent (one 2025 precedent); our defensible originality is the agent-memory
framing, the honest characterization, the clean artifact, and the multi-memory
comparison of §3–4 — not the core mechanism.**

## 2. Why 3-D is extreme but principled (the trade-off)

Space-partitioning trees only help in **low** dimensions, and aggressive
projection **destroys exact** nearest-neighbour recall while **preserving coarse
cluster structure** — exactly OctaSoma's measured behaviour.

- **Spatial trees collapse above ~10–20 D.** k-d trees "remain practical … up to
  k=10 to 20, but beyond this threshold their efficiency plummets"; exact tree
  indexes "can rarely outperform the brute-force linear scan … when dimensionality
  is high (e.g. more than 20)" (Li et al., arXiv:1610.02455). An octree is
  *hardwired* to 3-D (eight octants), so to use one at all you must project very
  low. This is the honest justification for 3-D, not a gimmick.
- **JL needs far more than 3 dimensions.** Faithful distance preservation for $n$
  points needs $k \approx \log(n)/\varepsilon^2$ dimensions, independent of $D$
  (Dasgupta–Gupta), and $\Omega(\log n/\varepsilon^2)$ is provably necessary
  (Larsen–Nelson, JL optimality). Three is orders of magnitude below this.
- **Even mild reduction hurts recall; PCA beats random projection.** Reducing to
  $0.1\times$ dimensions drops random-projection recall "no more than 40%", while
  PCA "has significant improvement compared with random projection" (QPAD,
  arXiv:2504.16335); randomized-PCA trees reach 0.95 recall with the fewest trees
  (Randomized PCA forest, 2024); a DR-for-ANN survey confirms the trade-off
  (arXiv:2403.13491). → recall@1 ≈ 0 % at 3-D is *expected*, and PCA ≫ JL is the
  *predicted* ordering — both of which our benchmark reproduces.
- **The exact NN is fragile in high-D, but cluster membership is meaningful.**
  Classic "meaningfulness of nearest neighbour" results show the single true
  neighbour becomes unstable as $D$ grows, while *retrieving the right cluster*
  stays meaningful (Beyer et al.; He et al.). PCA's top-3 axes capture the
  largest-variance directions that separate clusters — so the **topical/cluster
  recall** OctaSoma reports is precisely the signal theory says survives.

**Takeaway:** OctaSoma operates at the principled extreme of a well-studied
trade-off. Its honesty (recall@1≈0 %, PCA≫JL, "coarse router") matches theory.

## 3. Where OctaSoma sits among agent-memory types

LLM-agent long-term memory in 2024–2026 falls into four families; a compact ANN
index is the load-bearing primitive under most of them.

| Family | Representative | How it stores/retrieves |
|---|---|---|
| Dense vector / ANN | FAISS, HNSW, Qdrant, Milvus | embed → top-$k$ by approximate similarity |
| Lexical / hybrid | BM25 + dense via RRF | exact term match, fused with embeddings |
| Summarization / paging | MemGPT / Letta (arXiv:2310.08560) | OS-style tiered context, self-edit/page |
| Knowledge / temporal graph | Zep/Graphiti (arXiv:2501.13956), GraphRAG | entity-relation graph + vector + full-text |

**Comparison metrics** are standardized: recall@$k$ / task benchmarks (LoCoMo,
LongMemEval); recall-vs-QPS curves; latency p50/p95; **memory footprint
(bytes/vector)** — e.g. 768-d float32 HNSW ≈ 3.2 KB/vector ≈ 4.8 GB per 1 M
vectors; and cost/tokens per call. Even graph systems "query using a fusion of
time, full-text, semantic, and graph" — i.e. a vector index underneath. So the
right way to argue OctaSoma's value is **along these exact axes**, especially
footprint and latency (its coordinates are 3 floats = 12 B vs ~3 KB).

## 4. The useful niche

A compact, fast, explainable, lower-recall 3-D router has a recognized place:

- **Cheap coarse pre-filter → precise reranker** is the dominant production
  pattern; the coarse stage is "much faster" and bounded by *coverage/recall*, not
  precision (two-stage retrieval). IVF's coarse quantizer is itself "non-
  exhaustive" pruning (FAISS) — a low-resolution router. OctaSoma fits as *Stage 0*.
- **On-device / edge memory under tight budgets.** Full indices can be "several
  times larger than the original data (150–700 GB for 100 GB)"; compact designs
  like LEANN cut the footprint "below 5% … on personal devices" (arXiv:2506.08276);
  PQ is "ideal for edge devices" (ObjectBox). A 12-byte-coordinate index is in
  this regime.
- **Natively visualizable / explainable.** 2-/3-D embedding projections are the
  standard tool for "identifying clusters, debugging model failures … and
  explainability for stakeholders" (Nomic Atlas; Fiddler). OctaSoma is *already*
  3-D — no post-hoc UMAP — so its index is directly inspectable.

**Niche, ranked:** (1) Stage-0 pre-filter ahead of a high-recall index;
(2) footprint-constrained on-device/few-topic agent memory; (3) explainable/
auditable index. Weakest as a standalone high-recall store — which we state plainly.

## 5. What to claim — and not

- **Claim:** an honest characterization of the 3-D projection extreme for agent
  memory; PCA≫JL quantified; exact, reproducible, safe-Rust artifact; a compact,
  explainable coarse router with a measured footprint/latency/recall trade-off.
- **Do not claim:** a novel ANN algorithm or a novel pipeline (Ellendula & Bajaj
  precede us), nor competitiveness with HNSW/IVF on recall.
- **Must do:** cite and differentiate from Ellendula & Bajaj.
- **Improvement path:** their **per-cluster** projection (k-means → project each
  cluster) would likely lift our multi-topic recall — but adopting it makes us
  *closer* to their method, so present it as "we confirm, package, and characterize
  for agent memory", not "we invented".

## References

- **Closest precedent** — Ellendula & Bajaj, *Self-Balancing, Memory-Efficient, Dynamic Metric-Space Data Maintenance for Rapid Multi-Kernel Estimation*, ECML-PKDD 2025 — https://arxiv.org/abs/2504.18003 · https://link.springer.com/chapter/10.1007/978-3-032-06109-6_16
- DR-for-ANN survey — https://arxiv.org/abs/2403.13491
- QPAD (PCA vs random projection, recall at low dims) — https://arxiv.org/abs/2504.16335
- Randomized PCA forest for approximate kNN — https://www.sciencedirect.com/science/article/pii/S095741742403121X
- JL lemma (Dasgupta–Gupta proof) — https://cseweb.ucsd.edu/~dasgupta/papers/jl.pdf
- Optimality of the JL lemma (Larsen–Nelson) — https://www.researchgate.net/publication/307902434_Optimality_of_the_Johnson-Lindenstrauss_Lemma
- Meaningfulness of NN / cluster vs exact — https://research.google.com/pubs/archive/38140.pdf
- ANN on high-dimensional data; k-d tree threshold / curse of dim (Li et al.) — https://arxiv.org/pdf/1610.02455
- FLANN (Muja & Lowe) — https://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_pami2014.pdf
- Annoy (random projection trees) — https://github.com/spotify/annoy
- MemGPT / Letta — https://arxiv.org/abs/2310.08560
- Zep / Graphiti — https://arxiv.org/abs/2501.13956 · GraphRAG — https://github.com/microsoft/graphrag
- LEANN (on-device, <5% footprint) — https://arxiv.org/abs/2506.08276
- Two-stage retrieval — https://www.emergentmind.com/topics/two-stage-retrieval-system
- FAISS IVF coarse quantizer — https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexIVF.html
- Embedding visualization (Nomic Atlas) — https://docs.nomic.ai/atlas/embeddings-and-retrieval/guides/how-to-visualize-embeddings
- Hybrid BM25+dense (RRF) — https://www.meilisearch.com/blog/hybrid-search-rag
- Agent-memory benchmarks (LoCoMo) — https://github.com/mem0ai/memory-benchmarks
