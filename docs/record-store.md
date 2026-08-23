# The record layer — logical memory records, lifecycle and filtered recall

Status: shipped in the v0.5 line (record model + `RecordStore` + `ShardedHybrid`
integration). Design goals live in [`v05-architecture.md`](v05-architecture.md);
this page is the practical guide.

## What it is

The physical engines are append-only indexes: `FractalMemory3D`, `SketchIndex`
and their hybrids know nothing about time, ownership or deletion. The **record
layer** adds the missing logical half:

| Piece | Role |
|---|---|
| `MemoryRecord` (`src/record.rs`) | stable id, tenant/workspace/agent scope, provenance, sensitivity, TTL/retention, relations (`confirms` / `contradicts` / `supersedes`), monotonic generation |
| `RecordStore` (`src/record_store.rs`) | validated sidecar map id → record with tombstones, TTL visibility, purge accounting and versioned persistence (`RECS` v1) |
| `ShardedHybrid` integration | `remember` writes payload + record atomically; `recall_visible` filters hits by lifecycle; the store directory carries a `records.recs` component bound by an OSHH v2 manifest |

Design rules carried over from the record model:

- **No wall clock.** Every lifecycle decision takes an explicit
  `now_unix_ms`. OctaSoma never decides *when* on its own.
- **Logical ≠ physical delete.** `tombstone` hides a memory immediately;
  `purge_purgeable_at` removes the *record* after its retention floor.
  Reclaiming index space is a rebuild/compaction concern (publish a new
  generation of visible items) — the engines stay append-only.
- **Unknown ids pass through.** A payload without a record is treated as
  visible, so stores that predate the record layer keep flowing through
  filtered recalls unchanged.

## Writing memories with records

```rust
use octasoma::{
    EmbeddingFingerprint, HashEmbedder, MemoryId, MemoryRecord, MemoryScope,
    Provenance, ShardedHybrid,
};

let mut mem = ShardedHybrid::new(HashEmbedder::new(768), 256);

// The record's id IS the shard payload: one join key everywhere.
let record = MemoryRecord::new(
    MemoryId::new("sym:src/db.rs:pool")?,
    b"ignored".to_vec(),                    // payload lives in the shard
    MemoryScope::new("tenant-a", "ws", "agent-b")?,
    Provenance::new("ccos:event-log")?.with_source_record("event:42")?,
    EmbeddingFingerprint::new("ollama", "nomic-embed-text", 768)?,
    1,                                      // store generation
);

mem.remember("src/db.rs", record, "build and run SQL pools")?;
```

Generations are strictly monotonic per id: rewriting a memory requires a newer
generation (`upsert_payload` inside the record, or a fresh `MemoryRecord`),
which is what makes tombstones survive rebuilds.

## Reading with lifecycle filters

```rust
let now_ms = 1_735_689_600_000; // supplied by the product, never read here

// Tombstoned, superseded or expired records are dropped; plain payloads pass.
let hits = mem.recall_visible("src/db.rs", "database pool", 5, now_ms)?;

// Logical delete + purge accounting:
mem.tombstone("sym:src/db.rs:pool", 2)?;
let purged = mem.purge_purgeable_at(now_ms);   // removes records, not index slots
```

## Persistence

`save_dir` writes regions exactly as before, then `records.recs` (RECS v1),
then the manifest — **the manifest is the commit point**, so a crash before it
lands leaves the previous state authoritative. The manifest moves to OSHH v2
with a records flag; v1 directories remain readable (empty record layer).
Decoding validates every declared count/length against the bytes actually
present before it can drive an allocation (same discipline as `fileguard`).

## Reclaiming space: compaction + generation pruning

The engines are append-only; reclamation is explicit and rebuild-shaped:

```rust
// 1. Drop hidden index entries from a region (tombstoned/expired/superseded):
let reclaimed = mem.compact_region("src/db.rs", now_ms)?;

// 2. Persist: save_dir publishes the compacted state as a fresh generation.
mem.save_dir(dir)?;

// 3. Reclaim superseded generations across every region's chain:
let removed = octasoma::prune_sharded_hybrid_generations(dir, 1)?;
```

Compaction keeps exactly what `recall_visible` could return at `now` —
resurrecting a compacted record is an explicit `remember` with a newer
generation. On int8/NF4 tiers it re-quantizes once (F32 round-trips
bit-exactly). Pruning always preserves the generation `CURRENT` names and
refuses without one; call it when no reader is mid-open.

## Relations: multi-hop recall

Records carry evidence edges (`confirms`, `contradicts`, `supersedes`,
`superseded_by`). `relate` wires an edge between stored records at a strictly
newer generation; `recall`'s `hops` parameter traverses them BFS-style from the
direct hits:

```text
recall(text, region, k=1, now_ms=t, hops=1) →
  [ { uri: sym:g:anchor,  score: 0.91, hop: 0 },
    { uri: sym:g:evidence, score: 0.91, hop: 1,
      via: { from: sym:g:anchor, relation: confirms } } ]
```

Contract, enforced in one place (`RecordStore::related_ids`):

- **Traversal is filter-bound.** A hidden record has no traversable edges and
  unreachable targets are skipped — relations can never become a side channel
  around scoping or clearance.
- **Expanded rows are labelled, never passed off as similar.** They inherit the
  parent's cosine and carry `via { from, relation, hop }` plus
  `inherited_score: true`; hop 0 rows are the only true similarity hits.
- **Region-local.** A related record without an index entry in the queried
  region (compacted away, remembered elsewhere) is not returned.
- **Bounded.** Hops cap at 2; `max_expanded` caps total appended rows.

The supersession flow composes: remember the corrected fact, then
`supersede` on the old record records a `superseded_by` edge pointing at the
replacement (the audit-fixed direction) and hides it from every lifecycle-aware
recall.

## Over MCP

`octasoma-mcp` exposes the same lifecycle for agents (OpenClaw, CCOS):

| Tool | Effect |
|---|---|
| `remember` | ingest with a full record: tenant/workspace/agent scope, sensitivity, `expires_at_ms`, retention floor, provenance; monotonic `generation` |
| `recall` + `now_ms` | lifecycle-aware recall in a region — hidden records never surface. Optional `tenant`/`workspace`/`agent` scoping and `clearance` (records classified strictly above it are hidden). Optional `hops` (0–2) + `max_expanded` traverse relation edges within the same filter |
| `relate` | add an evidence edge `uri --relation--> target` (`confirms`, `contradicts`, `supersedes`, `superseded_by`) at a strictly newer generation |
| `tombstone` | logical delete (auto-generation by default) |
| `purge` | compacts every region first (so hidden entries die while their records can still vouch for them), then removes the purgeable records — unscoped by design |
| `compact` | per-region or store-wide rebuild under the same filter vocabulary (`now_ms` + scope + clearance); **the filter must mirror your recalls** — dropping an entry a legal query could still return is data loss |

Payloads keep the MCP convention `id<US>text`; the server extracts the join
key via `ShardedHybrid::recall_visible_by` / `compact_region_by`.

## Honest limits

- Compaction is per-region and synchronous; a bulk "compact everything"
  helper can be composed by looping over `region_keys()`.
- The record layer is process-local single-writer (`&mut self`), like the rest
  of OctaSoma; concurrency stays the caller's choice.
- Sensitivity/scope are carried and filter-ready but **not** enforced as
  authorization — consumers gate on them with their own policies.

