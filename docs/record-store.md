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

## Honest limits

- Compaction is per-region and synchronous; a bulk "compact everything"
  helper can be composed by looping over `region_keys()`.
- The record layer is process-local single-writer (`&mut self`), like the rest
  of OctaSoma; concurrency stays the caller's choice.
- Sensitivity/scope are carried and filter-ready but **not** enforced as
  authorization — consumers gate on them with their own policies.

