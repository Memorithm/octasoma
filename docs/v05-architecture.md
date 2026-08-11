# OctaSoma v0.5 architecture

Status: architectural baseline for the v0.5 migration, rebased on the hardened `master` containing embedding validation, deterministic RCPS shortlist ordering, and the MSRV-safe MCP guard.

## Canonical role

OctaSoma is the canonical semantic and episodic memory engine of the Memorithm
ecosystem.

It is **not** a causal authority, an attention engine, or a product-governance
layer. Consumers may use recalled memories as evidence/observations according to
their own hard policies.

## SciRust boundary

The v0.5 target makes SciRust a required implementation foundation. OctaSoma
must consume targeted SciRust crates/features rather than duplicate retrieval,
linear-algebra, learning, SIMD/GPU or statistical primitives.

The migration order is deliberate:

1. split `scirust-retrieval` into fine-grained capability features;
2. pin OctaSoma to an immutable reviewed Memorithm/scirust revision;
3. replace local LSH/SimHash/cosine primitives with the SciRust canonical
   implementation;
4. use SciRust ProjectionHead/contrastive learning for learned semantic
   projection while retaining deterministic PCA as the baseline;
5. remove legacy standalone/duplicated implementations only after parity tests.

Rust 1.89 is the v0.5 MSRV baseline.

## Memory geometry

The 3-D projection is a **Spatial / Fractal Lens**. It exists for coarse
navigation, visualisation, locality inspection and explanation.

It is not the fundamental geometry of the precision store and must not define
the recall-quality ceiling of the memory engine.

The precision tier owns semantic candidate generation/reranking in the native
embedding space or an explicitly trained/quantized derivative of that space.

## Record model target

The v0.5 logical record is stable independently of physical index location. The
target record carries at least:

- stable logical id;
- tenant/workspace/agent scope as supplied by the product adapter;
- content/payload reference;
- provenance;
- timestamp and store generation;
- causal-region hint (not causal truth);
- sensitivity level;
- lifecycle state;
- embedding/model/projection fingerprint;
- relations such as `supersedes`, `contradicts`, and `confirms`.

Those relations are memory evidence. A consumer such as CCOS decides whether and
how they alter its own causal state.

## Lifecycle target

The production memory layer must support upsert, delete, TTL, retention,
tombstones and irreversible purge/right-to-forget workflows.

Product policy, RBAC, tenant key ownership and legal retention decisions remain
outside OctaSoma. OctaSoma provides the primitive lifecycle/storage mechanisms
required by those policies.

## Persistence target

Persistence moves to immutable generations with a manifest that binds together
all components required to interpret a store, including:

- store format/generation id;
- embedding model fingerprint;
- projection fingerprint;
- quantization/index parameters;
- SciRust revision/capability fingerprint;
- calibration certificate and calibration-data fingerprint where applicable;
- integrity hashes for generation components.

A generation becomes current only through an atomic manifest/current-pointer
transition. Mixed generations must never be opened as one valid store.

## Certified recall

Certified recall remains a first-class capability, but every certificate is
conditional on its declared calibration protocol and fingerprints.

Calibration ground truth must be externally observable and must include relevant
items missed by the retriever; self-retrieval-only labels are insufficient. A
certificate is invalidated when the store, embedder, projection, quantization,
index parameters or workload assumptions covered by its contract drift.

## Performance target

Enterprise-oriented benchmark targets are evaluated as an explicit hardware and
workload profile, not as context-free guarantees:

- 10^6 to 10^7 memories;
- dynamic multi-tenant deployment;
- dimensions 768 default, 1024 high-definition, 256 lightweight;
- Recall@10 target >= 0.92 against an F32/exact oracle;
- latency target P95 < 5 ms on the declared reference profile.

The production precision sequence is F32 oracle -> validated INT8 baseline -> PQ
experiments -> Matryoshka only when the embedding/projection training contract
actually supports prefix truncation.

## Ecosystem adapters

Adapters belong on the consumer side:

- CCOS: semantic/episodic observations only; causal hard state remains CCOS-owned;
- RSI: optional experience memory for context, candidates, mutations, failures,
  benchmarks and trajectories;
- COGNO: long-term memory whose outputs remain soft/untrusted before hard gates;
- SLHAv2: experimental cache-policy experience only, never the exact attention
  score path;
- FLAT/ElasticAutoTuner: warm-start experience only; measured tuning evidence
  remains authoritative.

`octasoma-mcp` remains a development/research surface. Enterprise products expose
memory through their own authenticated gateways.
