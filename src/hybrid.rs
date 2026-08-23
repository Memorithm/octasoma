//! Hybrid memory: the explainable 3-D layer and the high-precision sketch tier in
//! **one** store, over the same items.
//!
//! [`FractalMemory3D`] is a cheap, explainable, visualisable *coarse router* (exact
//! recall@1 ≈ 0%); [`SketchIndex`] is the precise tier (a SimHash shortlist → exact
//! cosine rerank). [`HybridMemory`] keeps both over the same inserted items, so you
//! recall **precisely** and still **explain / zoom / visualise the same memory** —
//! the two strengths the 3-D index and the sketch tier each have alone, combined.
//!
//! It trades memory for that union (the sketch tier stores the full embeddings for
//! its exact rerank); for the compact, 3-D-only deployment use [`crate::FractalMemory3D`]
//! or [`crate::ShardedMemory`].
//!
//! ```
//! use octasoma::HybridMemory;
//! let mut mem = HybridMemory::new(8, 42, 256);
//! mem.insert(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], b"a");
//! mem.insert(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], b"b");
//! let hits = mem.recall(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1, 8);
//! assert_eq!(hits[0].0, b"a");
//! assert!(mem.explain(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1).is_some());
//! ```

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{self, Read};
use std::sync::Arc;

use crate::embed::{EmbedError, Embedder};
use crate::record::RelationKind;
use crate::{Explanation, FractalMemory3D, Precision, RegionView, SimHasher, SketchIndex};

/// BFS bounds for [`ShardedHybrid::recall_related`].
#[derive(Clone, Copy, Debug)]
pub struct Traversal {
    /// Relation-following levels (capped at 2).
    pub hops: usize,
    /// Total expanded rows appended across all levels.
    pub max_expanded: usize,
}

impl Default for Traversal {
    fn default() -> Self {
        Self {
            hops: 1,
            max_expanded: 8,
        }
    }
}

/// One row of [`ShardedHybrid::recall_related`]: a direct cosine hit, or an
/// item reached by following relation edges from one.
///
/// Expanded rows inherit their parent's `score` — it is the *parent's* cosine
/// to the query, not the expanded item's own similarity. The `hop`, `via_kind`
/// and `via_from` fields make that unmistakable.
#[derive(Clone, Debug, PartialEq)]
pub struct RelatedHit {
    /// Raw index payload (the caller's key/content convention).
    pub payload: String,
    /// Direct hits: own cosine. Expanded rows: the parent's cosine.
    pub score: f32,
    /// 0 for direct recall results; 1.. for relation expansions.
    pub hop: usize,
    /// The relation followed to reach this row (`None` on direct hits).
    pub via_kind: Option<RelationKind>,
    /// The record the edge was followed from (`None` on direct hits).
    pub via_from: Option<String>,
}

const SKETCH_SEED_XOR: u64 = 0x9E37_79B9_7F4A_7C15;

/// How [`HybridMemory::query`] finds candidates before the exact cosine rerank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryStrategy {
    /// Cheapest: 3-D spatial candidates → exact rerank. Bounded by the coarse
    /// router's locality (low exact recall), but fast and explainable.
    FastSpatial,
    /// Most precise: a SimHash Hamming shortlist over **all** items → exact rerank.
    /// Scans every sketch (popcount-cheap); the high-recall default.
    PrecisionSketch,
    /// Cascade: a wide 3-D neighbourhood → Hamming prune **within it** → exact
    /// rerank. Avoids scanning all sketches; recall is capped by 3-D locality.
    HybridCascade,
}

/// A memory that is both **explainable** (3-D octree) and **precise** (SimHash
/// sketch + exact rerank) over the same items.
#[derive(Clone)]
pub struct HybridMemory {
    pub(crate) tree: FractalMemory3D,
    pub(crate) sketch: SketchIndex,
    pub(crate) dim: usize,
    pub(crate) default_shortlist: usize,
}

impl HybridMemory {
    /// Creates a hybrid memory: a deterministic JL 3-D projection (from `seed`) and
    /// `bits`-wide SimHash sketches.
    pub fn new(dim: usize, seed: u64, bits: usize) -> Self {
        let sketch_seed = seed ^ SKETCH_SEED_XOR;
        let projector = Arc::new(SimHasher::new(dim, bits, sketch_seed));
        Self::new_with_shared_projector(dim, seed, sketch_seed, projector)
    }

    fn new_with_shared_projector(
        dim: usize,
        seed: u64,
        sketch_seed: u64,
        projector: Arc<SimHasher>,
    ) -> Self {
        Self {
            tree: FractalMemory3D::new(dim, seed),
            sketch: SketchIndex::new_with_shared_hasher(projector, sketch_seed, Precision::F32),
            dim,
            default_shortlist: 256,
        }
    }

    fn share_projector(&mut self, projector: Arc<SimHasher>, sketch_seed: u64) -> io::Result<()> {
        self.sketch.share_hasher(projector, sketch_seed)
    }

    #[cfg(test)]
    fn shares_projector_with(&self, other: &Self) -> bool {
        self.sketch.shares_hasher_with(&other.sketch)
    }

    /// Sets the default shortlist size used by [`HybridMemory::query`] (a builder).
    pub fn with_shortlist(mut self, shortlist: usize) -> Self {
        self.default_shortlist = shortlist.max(1);
        self
    }

    /// **Calibrate the default shortlist with a certificate** instead of a hand-tuned
    /// constant: runs [`SketchIndex::certify_shortlist`](crate::SketchIndex::certify_shortlist)
    /// on the precision tier and, on success, makes the certified size the default used
    /// by every [`QueryStrategy`] (see [`HybridMemory::query`]). Returns the certificate
    /// so the caller can log/persist the guarantee; `None` leaves the default untouched
    /// (nothing certifies — see the certify docs for why).
    pub fn calibrate_shortlist(
        &mut self,
        queries: &[Vec<f32>],
        k: usize,
        alpha: f64,
        delta: f64,
    ) -> Option<crate::ShortlistCertificate> {
        let cert = self.sketch.certify_shortlist(queries, k, alpha, delta)?;
        self.default_shortlist = cert.shortlist.max(1);
        Some(cert)
    }

    /// Like [`HybridMemory::new`], but the 3-D layer learns a PCA projection from a
    /// flat `num_samples × dim` calibration matrix.
    pub fn new_with_pca(
        dim: usize,
        calibration: &[f32],
        num_samples: usize,
        bits: usize,
        seed: u64,
    ) -> Self {
        Self {
            tree: FractalMemory3D::new_with_pca(dim, calibration, num_samples),
            sketch: SketchIndex::new(dim, bits, seed),
            dim,
            default_shortlist: 256,
        }
    }

    /// Inserts an embedding + byte payload into **both** layers. Returns `false`
    /// (storing nothing) on a dimension mismatch or a non-finite projection, keeping
    /// the two layers exactly in sync.
    pub fn insert(&mut self, embedding: &[f32], payload: &[u8]) -> bool {
        if embedding.len() != self.dim {
            return false;
        }
        // The tree rejects non-finite projections; only then sketch the item, so the
        // two layers always hold the same set.
        if self.tree.insert(embedding, Some(payload)).is_none() {
            return false;
        }
        self.sketch.insert(embedding, payload)
    }

    /// **Precise** recall: SimHash shortlist → exact cosine rerank → top `k`
    /// `(payload, cosine)`, most similar first. Larger `shortlist` → higher recall.
    pub fn recall(&self, query: &[f32], k: usize, shortlist: usize) -> Vec<(&[u8], f32)> {
        self.sketch.nearest(query, k, shortlist)
    }

    /// **Coarse** recall via the 3-D layer (the cheap router): top `k` payloads by
    /// projected distance. Far less precise — for the explainable/visualisable view
    /// or a quick pre-filter.
    pub fn recall_coarse(&self, query: &[f32], k: usize) -> Vec<&[u8]> {
        self.tree.query_k(query, k)
    }

    /// Unified query with an adaptive [`QueryStrategy`], returning the top `k`
    /// `(payload, cosine)`. Every strategy finishes with an exact cosine rerank;
    /// they differ only in how candidates are gathered. Uses the default shortlist
    /// (see [`HybridMemory::with_shortlist`]).
    pub fn query(&self, embedding: &[f32], strategy: QueryStrategy, k: usize) -> Vec<(&[u8], f32)> {
        let shortlist = self.default_shortlist.max(k);
        match strategy {
            QueryStrategy::FastSpatial => {
                let ids: Vec<u32> = self
                    .tree
                    .nearest_embedding(embedding, shortlist)
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect();
                self.sketch.rerank(embedding, &ids, k)
            }
            QueryStrategy::PrecisionSketch => self.sketch.nearest(embedding, k, shortlist),
            QueryStrategy::HybridCascade => {
                // A wide 3-D neighbourhood, Hamming-pruned within it, then reranked.
                let broad: Vec<u32> = self
                    .tree
                    .nearest_embedding(embedding, shortlist.saturating_mul(4))
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect();
                let pruned = self.sketch.hamming_rank(embedding, &broad, shortlist);
                self.sketch.rerank(embedding, &pruned, k)
            }
        }
    }

    /// Explains a recall via the 3-D layer: the query's position, the coarse→fine
    /// zoom path, and the nearest memories with distances and coordinates.
    pub fn explain(&self, query: &[f32], k: usize) -> Option<Explanation> {
        self.tree.explain(query, k)
    }

    /// The coarse→fine fractal zoom path along `query` (via the 3-D layer).
    pub fn zoom_path(&self, query: &[f32], max_level: u32, max_samples: usize) -> Vec<RegionView> {
        self.tree.zoom_path(query, max_level, max_samples)
    }

    /// Viewer JSON (`{count, half_size, points:[…]}`) of the 3-D layer.
    pub fn export_points_json(&self, max_points: usize) -> String {
        self.tree.export_points_json(max_points)
    }

    /// Viewer JSON of the 3-D layer **heat-coloured by precision score**: each point
    /// carries its exact cosine similarity to `query`. Drop it on `viewer/index.html`
    /// to *see* which memories are closest to a query.
    pub fn export_scored_json(&self, query: &[f32], max_points: usize) -> String {
        self.tree
            .export_points_json_scored(&self.sketch.scores(query), max_points)
    }

    /// Read-only access to the 3-D layer (advanced inspection / the viewer).
    pub fn tree(&self) -> &FractalMemory3D {
        &self.tree
    }

    /// Number of stored items.
    pub fn len(&self) -> usize {
        self.sketch.len()
    }

    /// Whether nothing has been stored yet.
    pub fn is_empty(&self) -> bool {
        self.sketch.is_empty()
    }

    /// Persists this coupled spatial+precision store as a new immutable
    /// generation and publishes it through a crash-recoverable `CURRENT`
    /// pointer. Both component hashes and the default shortlist are bound by
    /// the generation manifest; a reader never combines files from different
    /// generations.
    pub fn save_dir(&self, dir: &str) -> io::Result<()> {
        crate::generation_store::save(self, dir)
    }

    /// Opens the single complete generation selected by `CURRENT`, validating
    /// its manifest and SHA-256 component hashes before deserialisation. If a
    /// crash happened before pointer publication, the highest immutable
    /// generation is recovered. Legacy v0.4 `tree.frac` + `index.skch` stores
    /// remain readable when no generation layout exists.
    pub fn open_dir(dir: &str, dim: usize) -> io::Result<Self> {
        crate::generation_store::open(dir, dim)
    }

    pub(crate) fn open_legacy_dir(dir: &str, dim: usize) -> io::Result<Self> {
        let root = std::path::Path::new(dir);
        let tree_path = root.join("tree.frac");
        let sketch_path = root.join("index.skch");
        crate::fileguard::guard_not_symlink("hybrid tree", &tree_path)?;
        crate::fileguard::guard_not_symlink("hybrid sketch", &sketch_path)?;
        let tree = FractalMemory3D::load_from_disk(tree_path.to_string_lossy().as_ref(), dim)?;
        let sketch = SketchIndex::load_from_disk(sketch_path.to_string_lossy().as_ref(), dim)?;
        Ok(Self {
            tree,
            sketch,
            dim,
            default_shortlist: 256,
        })
    }
}

/// One [`HybridMemory`] per causal region — the precise, scale-safe sharded
/// deployment. CCOS narrows a query to a region; within it `HybridMemory` gives
/// **precise** recall (sketch shortlist → exact rerank) and stays explainable, so
/// recall does not collapse as a region grows. Shares one embedder.
///
/// This is the precise sibling of [`crate::ShardedMemory`] (which keeps a compact
/// 3-D-only index per region); it trades memory for per-region precision.
pub struct ShardedHybrid<E: Embedder> {
    shards: HashMap<String, HybridMemory>,
    embedder: E,
    seed: u64,
    bits: usize,
    projector: Arc<SimHasher>,
    /// Optional logical record layer (see [`crate::RecordStore`]). Payloads that
    /// have a record here are filtered by lifecycle state in the `*_visible`
    /// recalls; payloads without one flow through untouched.
    records: crate::RecordStore,
}

impl<E: Embedder> ShardedHybrid<E> {
    /// Creates an empty sharded-hybrid memory with `bits`-wide sketches per region.
    pub fn new(embedder: E, bits: usize) -> Self {
        let seed = 42;
        let projector = Arc::new(SimHasher::new(embedder.dim(), bits, seed ^ SKETCH_SEED_XOR));
        Self {
            shards: HashMap::new(),
            embedder,
            seed,
            bits,
            projector,
            records: crate::RecordStore::new(),
        }
    }

    /// Bytes used by the single shared SimHash hyperplane matrix.
    pub fn projector_bytes(&self) -> usize {
        self.projector.plane_bytes()
    }

    /// Embedding dimensionality of this store.
    pub fn dim(&self) -> usize {
        self.embedder.dim()
    }

    /// Embeds `text` and stores it under `region`, with `uri` as the payload.
    pub fn insert(&mut self, region: &str, uri: &str, text: &str) -> Result<(), EmbedError> {
        self.insert_raw(region, uri.as_bytes(), text)
    }

    /// [`ShardedHybrid::insert`] with an arbitrary payload — for callers whose
    /// payloads pack the logical id together with content (the MCP convention).
    fn insert_raw(&mut self, region: &str, payload: &[u8], text: &str) -> Result<(), EmbedError> {
        let v = self.embedder.embed_checked(text)?;
        let (dim, seed) = (self.embedder.dim(), self.seed);
        let sketch_seed = seed ^ SKETCH_SEED_XOR;
        let projector = Arc::clone(&self.projector);
        let inserted = {
            let shard = self.shards.entry(region.to_string()).or_insert_with(|| {
                HybridMemory::new_with_shared_projector(dim, seed, sketch_seed, projector)
            });
            shard.insert(&v, payload)
        };
        if !inserted {
            // No phantom shards: roll back a region that holds nothing.
            if self.shards.get(region).is_some_and(|s| s.is_empty()) {
                self.shards.remove(region);
            }
            return Err(EmbedError::Protocol(
                "validated embedding could not be inserted into the sharded hybrid index".into(),
            ));
        }
        Ok(())
    }

    /// **Precise** recall within `region` (sketch shortlist → exact cosine rerank):
    /// the `k` nearest uris with cosine scores, descending. Empty if `region` is
    /// unknown.
    pub fn recall(
        &self,
        region: &str,
        query: &str,
        k: usize,
    ) -> Result<Vec<(String, f32)>, EmbedError> {
        self.recall_with(region, query, k, QueryStrategy::PrecisionSketch)
    }

    /// Recall within `region` with an explicit [`QueryStrategy`]. Empty if unknown.
    pub fn recall_with(
        &self,
        region: &str,
        query: &str,
        k: usize,
        strategy: QueryStrategy,
    ) -> Result<Vec<(String, f32)>, EmbedError> {
        let Some(shard) = self.shards.get(region) else {
            return Ok(Vec::new());
        };
        let v = self.embedder.embed_checked(query)?;
        Ok(shard
            .query(&v, strategy, k)
            .into_iter()
            .map(|(p, s)| (String::from_utf8_lossy(p).into_owned(), s))
            .collect())
    }

    /// Precise **global** recall across all regions, merged by **true cosine**
    /// (comparable across regions, unlike per-region 3-D distances) — the scope-free
    /// path. Each region contributes its precise top-`k`.
    pub fn recall_global(&self, query: &str, k: usize) -> Result<Vec<(String, f32)>, EmbedError> {
        let v = self.embedder.embed_checked(query)?;
        let mut hits: Vec<(String, f32)> = Vec::new();
        let Some(first) = self.shards.values().next() else {
            return Ok(hits);
        };

        // `ShardedHybrid` shares one SimHasher across regions. When every shard
        // also uses the same persisted scalar/SIMD sketch path, project the query
        // exactly once and reuse that bit-vector for every Hamming shortlist.
        let sketch_path = first.sketch.simd_sketching();
        let can_reuse = self
            .shards
            .values()
            .all(|shard| shard.sketch.simd_sketching() == sketch_path);
        let shared_sketch = can_reuse.then(|| first.sketch.query_sketch(&v)).flatten();

        for shard in self.shards.values() {
            let shard_hits = if let Some(query_sketch) = shared_sketch.as_deref() {
                let shortlist = shard.default_shortlist.max(k);
                shard
                    .sketch
                    .nearest_with_sketch(&v, query_sketch, k, shortlist)
            } else {
                shard.query(&v, QueryStrategy::PrecisionSketch, k)
            };
            for (payload, score) in shard_hits {
                hits.push((String::from_utf8_lossy(payload).into_owned(), score));
            }
        }
        hits.sort_by(|a, b| b.1.total_cmp(&a.1));
        hits.truncate(k);
        Ok(hits)
    }

    /// Explains a recall within `region` via its 3-D layer; `Ok(None)` if unknown.
    pub fn explain(
        &self,
        region: &str,
        query: &str,
        k: usize,
    ) -> Result<Option<Explanation>, EmbedError> {
        let Some(shard) = self.shards.get(region) else {
            return Ok(None);
        };
        let v = self.embedder.embed_checked(query)?;
        Ok(shard.explain(&v, k))
    }

    // -- logical record layer -------------------------------------------------

    /// Stores `text` under the record's own id (`record.id.as_str()` is both the
    /// shard payload and the join key into the record layer). The record's
    /// generation must be strictly newer than any stored one for that id.
    ///
    /// The write is all-or-nothing at the API level: a rejected generation is
    /// checked before anything is indexed, and a failed index insert never
    /// leaves a half-applied record behind.
    pub fn remember(
        &mut self,
        region: &str,
        record: crate::record::MemoryRecord,
        text: &str,
    ) -> Result<(), EmbedError> {
        let payload = record.id.as_str().to_string();
        self.remember_with_payload(region, record, payload.as_bytes(), text)
    }

    /// [`ShardedHybrid::remember`] with an arbitrary index payload (e.g. one
    /// packing the logical id *and* the content, the MCP convention). The
    /// record's id remains the join key into the record layer.
    pub fn remember_with_payload(
        &mut self,
        region: &str,
        record: crate::record::MemoryRecord,
        payload: &[u8],
        text: &str,
    ) -> Result<(), EmbedError> {
        let uri = record.id.as_str().to_string();
        if let Some(existing) = self.records.get(&uri)
            && record.generation <= existing.generation
        {
            return Err(EmbedError::Protocol(format!(
                "record {uri} generation must increase monotonically: current={}, proposed={}",
                existing.generation, record.generation
            )));
        }
        self.insert_raw(region, payload, text)?;
        if let Err(error) = self.records.put(record) {
            return Err(EmbedError::Protocol(error.to_string()));
        }
        Ok(())
    }

    /// The stored record for `id`, if any.
    pub fn record(&self, id: &str) -> Option<&crate::record::MemoryRecord> {
        self.records.get(id)
    }

    /// Number of records in the logical layer (including invisible ones).
    pub fn records_len(&self) -> usize {
        self.records.len()
    }

    /// Read-only access to the whole record layer.
    pub fn records(&self) -> &crate::RecordStore {
        &self.records
    }

    /// Logical delete: marks `id` tombstoned at a strictly newer generation;
    /// subsequent visible recalls stop returning it. Physical purge is separate
    /// ([`ShardedHybrid::purge_purgeable_at`]).
    pub fn tombstone(
        &mut self,
        id: &str,
        generation: u64,
    ) -> Result<(), crate::record::RecordError> {
        self.records.tombstone(id, generation)
    }

    /// Precise recall within `region` filtered by the record lifecycle: items
    /// whose record is tombstoned, superseded or TTL-expired at `now_unix_ms`
    /// are dropped; payloads without a record pass through. Overfetches a small
    /// multiple of `k` so hidden entries do not shrink the result below `k`
    /// when deeper candidates exist.
    pub fn recall_visible(
        &self,
        region: &str,
        query: &str,
        k: usize,
        now_unix_ms: u64,
    ) -> Result<Vec<(String, f32)>, EmbedError> {
        self.recall_visible_by(region, query, k, now_unix_ms, |payload| payload)
    }

    /// [`ShardedHybrid::recall_visible`] with a caller-supplied extractor of the
    /// logical record id from the raw index payload — for stores whose payloads
    /// pack id *and* content (e.g. the MCP server's `id<US>text` convention),
    /// where comparing whole payloads against record ids would match nothing.
    pub fn recall_visible_by(
        &self,
        region: &str,
        query: &str,
        k: usize,
        now_unix_ms: u64,
        key_of: impl Fn(&str) -> &str,
    ) -> Result<Vec<(String, f32)>, EmbedError> {
        self.recall_filtered(
            region,
            query,
            k,
            &crate::RecordFilter::at(now_unix_ms),
            key_of,
        )
    }

    /// Precise recall within `region` under a full [`RecordFilter`] — lifecycle
    /// state plus tenant/workspace/agent scoping and sensitivity clearance.
    ///
    /// **Compaction contract:** every index entry this method can never return
    /// is exactly what [`ShardedHybrid::compact_filtered`] may reclaim — the
    /// two share one predicate so scoping can never turn into data loss.
    pub fn recall_filtered(
        &self,
        region: &str,
        query: &str,
        k: usize,
        filter: &crate::RecordFilter,
        key_of: impl Fn(&str) -> &str,
    ) -> Result<Vec<(String, f32)>, EmbedError> {
        let Some(shard) = self.shards.get(region) else {
            return Ok(Vec::new());
        };
        if k == 0 {
            return Ok(Vec::new());
        }
        let fetch = k
            .saturating_mul(3)
            .saturating_add(8)
            .min(shard.len().max(k));
        let mut out = Vec::with_capacity(k);
        for (payload, score) in
            self.recall_with(region, query, fetch, QueryStrategy::PrecisionSketch)?
        {
            if self.records.admits(key_of(&payload), filter) {
                out.push((payload, score));
                if out.len() == k {
                    break;
                }
            }
        }
        Ok(out)
    }

    /// Adds an evidence edge `id --kind--> target` to a stored record, at a
    /// strictly newer generation. Both endpoints must exist; a record cannot
    /// relate to itself.
    pub fn relate(
        &mut self,
        id: &str,
        kind: RelationKind,
        target: &str,
        generation: u64,
    ) -> Result<(), EmbedError> {
        if self.records.get(target).is_none() {
            return Err(EmbedError::Protocol(format!(
                "relation target {target:?} has no stored record"
            )));
        }
        let mut record = match self.records.get(id) {
            Some(record) => record.clone(),
            None => {
                return Err(EmbedError::Protocol(format!(
                    "no memory record with id {id:?}"
                )));
            }
        };
        record
            .advance_generation(generation)
            .map_err(|e| EmbedError::Protocol(e.to_string()))?;
        record
            .add_relation(
                kind,
                crate::record::MemoryId::new(target)
                    .map_err(|e| EmbedError::Protocol(e.to_string()))?,
            )
            .map_err(|e| EmbedError::Protocol(e.to_string()))?;
        self.records
            .put(record)
            .map_err(|e| EmbedError::Protocol(e.to_string()))?;
        Ok(())
    }

    /// **Relation-aware recall**: precise filtered recall, then follow the
    /// relation graph outward from every direct hit — up to `hops` BFS levels
    /// (capped at 2), appending at most `max_expanded` rows total.
    ///
    /// Traversal respects the [`RecordFilter`] exactly like the recall itself:
    /// hidden records have no traversable edges and unreachable targets are
    /// skipped, so relations can never become a side channel around scoping or
    /// clearance. Expansion is region-local — a related record with no index
    /// entry in this region (compacted away, remembered elsewhere) is not
    /// returned. Expanded rows carry their parent's cosine and are labelled
    /// with `hop`/`via_kind`/`via_from` (see [`RelatedHit`]).
    pub fn recall_related(
        &self,
        region: &str,
        query: &str,
        k: usize,
        filter: &crate::RecordFilter,
        key_of: impl Fn(&str) -> &str,
        traversal: Traversal,
    ) -> Result<Vec<RelatedHit>, EmbedError> {
        let hops = traversal.hops.min(2);
        let max_expanded = traversal.max_expanded;
        let direct = self.recall_filtered(region, query, k, filter, &key_of)?;
        let mut hits: Vec<RelatedHit> = direct
            .iter()
            .map(|(payload, score)| RelatedHit {
                payload: payload.clone(),
                score: *score,
                hop: 0,
                via_kind: None,
                via_from: None,
            })
            .collect();

        if hops == 0 || max_expanded == 0 {
            return Ok(hits);
        }

        // Region-local key → payload resolution for expanded rows.
        let Some(shard) = self.shards.get(region) else {
            return Ok(hits);
        };
        let mut payloads: HashMap<String, String> = HashMap::new();
        for i in 0..shard.sketch.len() {
            let raw = String::from_utf8_lossy(shard.sketch.item_payload(i)).into_owned();
            payloads.insert(key_of(&raw).to_string(), raw);
        }

        let mut seen: HashSet<String> = direct
            .iter()
            .map(|(payload, _)| key_of(payload).to_string())
            .collect();
        let mut frontier: Vec<String> = seen.iter().cloned().collect();
        let mut expanded = 0usize;

        for hop in 1..=hops {
            if frontier.is_empty() || expanded >= max_expanded {
                break;
            }
            // Deterministic order: parents in recall-score order (direct hits
            // are sorted), edges in each record's own insertion order.
            let mut next_frontier = Vec::new();
            for parent in &frontier {
                for (kind, target) in self.records.related_ids(parent, filter) {
                    if !seen.insert(target.clone()) || expanded >= max_expanded {
                        continue;
                    }
                    if let Some(raw) = payloads.get(&target) {
                        let parent_score = hits
                            .iter()
                            .find(|hit| key_of(&hit.payload) == parent)
                            .map(|hit| hit.score)
                            .unwrap_or(0.0);
                        hits.push(RelatedHit {
                            payload: raw.clone(),
                            score: parent_score,
                            hop,
                            via_kind: Some(kind),
                            via_from: Some(parent.clone()),
                        });
                        expanded += 1;
                        next_frontier.push(target);
                    }
                }
            }
            frontier = next_frontier;
        }
        Ok(hits)
    }

    /// Removes every record purgeable at `now_unix_ms` from the record layer
    /// and returns how many. Index space is reclaimed only when regions are
    /// rebuilt/compacted — the engines themselves are append-only.
    pub fn purge_purgeable_at(&mut self, now_unix_ms: u64) -> usize {
        self.records.purge_purgeable_at(now_unix_ms)
    }

    /// **Compacts a region**: rebuilds its index from the surviving items —
    /// exactly those [`ShardedHybrid::recall_visible`] could ever return at
    /// `now_unix_ms` (visible records plus record-less payloads) — and returns
    /// how many index entries were reclaimed. A region emptied entirely is
    /// removed; its record layer survives compaction untouched.
    ///
    /// Semantics after compaction:
    /// - hidden-but-not-purged records (tombstoned under a retention floor)
    ///   lose their index entry too — they were unreachable anyway. Resurrecting
    ///   one is an explicit `remember` with a newer generation, which re-indexes
    ///   it;
    /// - on the int8/NF4 rerank tiers the rebuild re-quantizes dequantized
    ///   vectors once more (one extra quantization step, cosine-equivalent to
    ///   first order). The default F32 tier round-trips bit-exactly;
    /// - persistence is unchanged: `save_dir` publishes the compacted state as
    ///   a fresh immutable generation; pair with
    ///   [`ShardedHybrid::prune_generations`] to reclaim the superseded ones.
    pub fn compact_region(&mut self, region: &str, now_unix_ms: u64) -> io::Result<usize> {
        self.compact_region_by(region, now_unix_ms, |payload| payload)
    }

    /// [`ShardedHybrid::compact_region`] with a caller-supplied extractor of the
    /// logical record id from the raw index payload (see
    /// [`ShardedHybrid::recall_visible_by`] for why that exists).
    pub fn compact_region_by(
        &mut self,
        region: &str,
        now_unix_ms: u64,
        key_of: impl Fn(&str) -> &str,
    ) -> io::Result<usize> {
        self.compact_filtered(region, &crate::RecordFilter::at(now_unix_ms), key_of)
    }

    /// [`ShardedHybrid::compact_region`] under a full [`RecordFilter`]. The
    /// filter **must** be the same predicate the caller's recalls use: an
    /// entry dropped here is one no filtered recall could ever return again.
    pub fn compact_filtered(
        &mut self,
        region: &str,
        filter: &crate::RecordFilter,
        key_of: impl Fn(&str) -> &str,
    ) -> io::Result<usize> {
        let Some(shard) = self.shards.get(region) else {
            return Ok(0);
        };
        let dim = shard.dim;
        let seed = self.seed;
        let sketch_seed = seed ^ SKETCH_SEED_XOR;

        // Collect survivors first: the old shard is borrowed while building the
        // new one, so nothing half-built can replace it on failure.
        let mut survivors: Vec<(Vec<f32>, Vec<u8>)> = Vec::new();
        for i in 0..shard.sketch.len() {
            let payload = shard.sketch.item_payload(i).to_vec();
            let key = key_of(&String::from_utf8_lossy(&payload)).to_string();
            if self.records.admits(&key, filter) {
                survivors.push((shard.sketch.item_embedding(i), payload));
            }
        }
        let removed = shard.len() - survivors.len();

        if survivors.is_empty() {
            self.shards.remove(region);
            return Ok(removed);
        }

        let mut rebuilt = HybridMemory::new_with_shared_projector(
            dim,
            seed,
            sketch_seed,
            Arc::clone(&self.projector),
        )
        .with_shortlist(shard.default_shortlist);
        for (embedding, payload) in &survivors {
            // Survivors came from a validated store; failing to re-index one is
            // a corruption signal, not a silent drop — and both layers move
            // together exactly like HybridMemory::insert.
            if !rebuilt.sketch.insert(embedding, payload)
                || rebuilt.tree.insert(embedding, Some(payload)).is_none()
            {
                return Err(invalid(
                    "compact_region: a surviving item could not be re-indexed",
                ));
            }
        }
        self.shards.insert(region.to_string(), rebuilt);
        Ok(removed)
    }

    /// Number of regions (shards).
    pub fn regions(&self) -> usize {
        self.shards.len()
    }

    /// Items in one region (0 if the region is unknown).
    pub fn region_len(&self, region: &str) -> usize {
        self.shards.get(region).map_or(0, HybridMemory::len)
    }

    /// Total items across all regions.
    pub fn len(&self) -> usize {
        self.shards.values().map(HybridMemory::len).sum()
    }

    /// Whether nothing has been stored yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The region keys, sorted.
    pub fn region_keys(&self) -> Vec<&str> {
        let mut keys: Vec<&str> = self.shards.keys().map(String::as_str).collect();
        keys.sort_unstable();
        keys
    }

    /// Persists every region's [`HybridMemory`] under `dir` (one sub-directory each)
    /// plus a binary manifest and the record layer (`records.recs`). Reopen with
    /// the same embedder via [`ShardedHybrid::open_dir`].
    ///
    /// The manifest is the commit point: regions and records are written first,
    /// so a crash before it lands leaves the previous manifest authoritative.
    pub fn save_dir(&self, dir: &str) -> io::Result<()> {
        fs::create_dir_all(dir)?;
        let mut regions: Vec<&String> = self.shards.keys().collect();
        regions.sort();

        let mut m = Vec::new();
        m.extend_from_slice(b"OSHH");
        m.extend_from_slice(&2u32.to_le_bytes());
        m.extend_from_slice(&(self.embedder.dim() as u32).to_le_bytes());
        m.extend_from_slice(&self.seed.to_le_bytes());
        m.extend_from_slice(&(self.bits as u64).to_le_bytes());
        m.extend_from_slice(&(regions.len() as u64).to_le_bytes());
        for (i, region) in regions.into_iter().enumerate() {
            let name = format!("shard_{i:08}");
            self.shards[region].save_dir(&format!("{dir}/{name}"))?;
            write_bytes(&mut m, region.as_bytes());
            write_bytes(&mut m, name.as_bytes());
        }
        // Record layer: written before the manifest so the flag can never point
        // at bytes that are not there yet. A symlink squatting on the target
        // path is refused instead of being written through.
        let has_records = !self.records.is_empty();
        if has_records {
            let recs_path = std::path::Path::new(dir).join("records.recs");
            if let Ok(metadata) = fs::symlink_metadata(&recs_path)
                && metadata.file_type().is_symlink()
            {
                return Err(invalid(
                    "sharded-hybrid records: symbolic links are not allowed in an OctaSoma store",
                ));
            }
            self.records.save_to_disk(&recs_path)?;
        }
        m.push(u8::from(has_records));
        fs::write(format!("{dir}/manifest.osh"), m)
    }

    /// Reopens a sharded-hybrid memory written by [`ShardedHybrid::save_dir`], bound
    /// to `embedder` (whose `dim()` must match) and `bits` from the manifest.
    /// v1 manifests (no record layer) remain readable; their records stay empty.
    pub fn open_dir(embedder: E, dir: &str) -> io::Result<Self> {
        let bytes = fs::read(format!("{dir}/manifest.osh"))?;
        let mut r: &[u8] = &bytes;
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != b"OSHH" {
            return Err(invalid("not a sharded-hybrid manifest (bad magic)"));
        }
        let version = read_u32(&mut r)?;
        if version > 2 || version == 0 {
            return Err(invalid(&format!(
                "unsupported sharded-hybrid version {version}"
            )));
        }
        let dim = read_u32(&mut r)? as usize;
        let seed = read_u64(&mut r)?;
        let bits = read_u64(&mut r)? as usize;
        if dim != embedder.dim() {
            return Err(invalid(&format!(
                "dim mismatch: manifest {dim}, embedder {}",
                embedder.dim()
            )));
        }
        let count = read_u64(&mut r)? as usize;
        let sketch_seed = seed ^ SKETCH_SEED_XOR;
        let projector = Arc::new(SimHasher::new(dim, bits, sketch_seed));
        // Each shard record is at least two length-prefixed strings (16 bytes).
        crate::fileguard::guard_count("manifest shards", count, 16, r.len() as u64)?;
        let mut shards = HashMap::with_capacity(count);
        for i in 0..count {
            let region = read_string(&mut r)?;
            let name = read_string(&mut r)?;
            let expected = format!("shard_{i:08}");
            crate::fileguard::guard_generated_component("hybrid manifest shard", &name, &expected)?;
            let path = std::path::Path::new(dir).join(&name);
            crate::fileguard::guard_not_symlink("hybrid manifest shard", &path)?;
            let mut hm = HybridMemory::open_dir(path.to_string_lossy().as_ref(), dim)?;
            hm.share_projector(Arc::clone(&projector), sketch_seed)?;
            shards.insert(region, hm);
        }
        // v2 carries the record layer; a flagged store without its file is a
        // corrupt or tampered directory, never an empty record layer.
        let records = if version >= 2 {
            let has_records = match read_u8("manifest records flag", &mut r)? {
                0 => false,
                1 => true,
                other => {
                    return Err(invalid(&format!(
                        "manifest records flag must be 0 or 1, got {other}"
                    )));
                }
            };
            if has_records {
                let recs_path = std::path::Path::new(dir).join("records.recs");
                crate::fileguard::guard_not_symlink("sharded-hybrid records", &recs_path)?;
                crate::RecordStore::load_from_disk(&recs_path)?
            } else {
                crate::RecordStore::new()
            }
        } else {
            crate::RecordStore::new()
        };
        crate::fileguard::guard_no_trailing_bytes("sharded-hybrid manifest", r.len())?;
        Ok(Self {
            shards,
            embedder,
            seed,
            bits,
            projector,
            records,
        })
    }
}

/// Deletes all but the newest `keep` published generations in every shard of a
/// sharded-hybrid store directory `dir` (see [`ShardedHybrid::save_dir`] — each
/// region persists as its own crash-safe generation chain). Within each chain,
/// the generation `CURRENT` points at is always preserved; a chain without a
/// published pointer refuses rather than guessing. Returns how many generation
/// directories were removed across all regions.
///
/// A free function rather than an associated one so reclaiming disk space never
/// requires naming the embedder type.
pub fn prune_sharded_hybrid_generations(dir: &str, keep: usize) -> io::Result<usize> {
    let root = std::path::Path::new(dir);
    let mut removed = 0;
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        // Only deterministic shard directories are prunable.
        let is_shard = name.starts_with("shard_")
            && name.len() == "shard_".len() + 8
            && name["shard_".len()..].bytes().all(|b| b.is_ascii_digit());
        if is_shard {
            removed += crate::generation_store::prune_generations(&root.join(name), keep)?;
        }
    }
    Ok(removed)
}

fn write_bytes(buf: &mut Vec<u8>, b: &[u8]) {
    buf.extend_from_slice(&(b.len() as u64).to_le_bytes());
    buf.extend_from_slice(b);
}

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u8(what: &str, r: &mut &[u8]) -> io::Result<u8> {
    crate::fileguard::guard_count(what, 1, 1, r.len() as u64)?;
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_string(r: &mut &[u8]) -> io::Result<String> {
    let len = read_u64(r)? as usize;
    // Validate-before-allocate: the manifest is fully in memory, so a declared
    // string length beyond the unread bytes is corrupt or hostile.
    crate::fileguard::guard_count("manifest string", len, 1, r.len() as u64)?;
    let mut b = vec![0u8; len];
    r.read_exact(&mut b)?;
    String::from_utf8(b).map_err(|e| invalid(&e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct WrongDimEmbedder;

    impl Embedder for WrongDimEmbedder {
        fn dim(&self) -> usize {
            4
        }

        fn embed(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
            Ok(vec![1.0, 2.0, 3.0])
        }
    }

    struct NonFiniteEmbedder;

    impl Embedder for NonFiniteEmbedder {
        fn dim(&self) -> usize {
            3
        }

        fn embed(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
            Ok(vec![1.0, f32::NAN, 3.0])
        }
    }

    #[test]
    fn sharded_hybrid_rejects_malformed_embeddings_without_reporting_success() {
        let mut wrong = ShardedHybrid::new(WrongDimEmbedder, 64);
        assert!(wrong.insert("r", "u", "bad").is_err());
        assert!(wrong.is_empty());
        assert_eq!(wrong.regions(), 0);
        assert!(wrong.recall_global("bad", 1).is_err());

        let mut non_finite = ShardedHybrid::new(NonFiniteEmbedder, 64);
        assert!(non_finite.insert("r", "u", "bad").is_err());
        assert!(non_finite.is_empty());
        assert_eq!(non_finite.regions(), 0);
    }

    #[test]
    fn global_recall_matches_per_shard_precision_results_with_shared_query_sketch() {
        let mut mem = ShardedHybrid::new(crate::HashEmbedder::new(32), 128);
        mem.insert("a", "a:alpha", "alpha durable memory").unwrap();
        mem.insert("b", "b:beta", "beta durable memory").unwrap();
        mem.insert("c", "c:gamma", "gamma durable memory").unwrap();

        let global = mem.recall_global("beta durable memory", 3).unwrap();
        assert_eq!(global.len(), 3);
        assert_eq!(global[0].0, "b:beta");

        let mut manual = Vec::new();
        for region in ["a", "b", "c"] {
            manual.extend(mem.recall(region, "beta durable memory", 3).unwrap());
        }
        manual.sort_by(|a, b| b.1.total_cmp(&a.1));
        manual.truncate(3);
        assert_eq!(global, manual);
    }

    #[test]
    fn sharded_hybrid_shares_one_simhash_projector_across_regions_and_reload() {
        let mut mem = ShardedHybrid::new(crate::HashEmbedder::new(16), 128);
        mem.insert("a", "a:1", "alpha").unwrap();
        mem.insert("b", "b:1", "beta").unwrap();
        assert!(mem.shards["a"].shares_projector_with(&mem.shards["b"]));
        assert_eq!(mem.projector_bytes(), 128 * 16 * std::mem::size_of::<f32>());

        let dir = std::env::temp_dir()
            .join(format!("octasoma_shared_projector_{}", std::process::id()))
            .to_string_lossy()
            .into_owned();
        let _ = std::fs::remove_dir_all(&dir);
        mem.save_dir(&dir).unwrap();
        let reopened = ShardedHybrid::open_dir(crate::HashEmbedder::new(16), &dir).unwrap();
        assert!(reopened.shards["a"].shares_projector_with(&reopened.shards["b"]));
        assert_eq!(reopened.recall("a", "alpha", 1).unwrap()[0].0, "a:1");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn calibrate_shortlist_installs_a_certified_default() {
        const DIM: usize = 24;
        let mut mem = HybridMemory::new(DIM, 128, 7);
        let mut queries = Vec::new();
        for c in 0..6 {
            let base: Vec<f32> = (0..DIM)
                .map(|d| ((c * DIM + d) as f32 * 0.9).sin())
                .collect();
            for j in 0..20 {
                let item: Vec<f32> = base
                    .iter()
                    .enumerate()
                    .map(|(d, x)| x + 0.05 * ((j * DIM + d) as f32 * 1.7).cos())
                    .collect();
                assert!(mem.insert(&item, format!("c{c}-i{j}").as_bytes()));
                if j % 4 == 0 {
                    queries.push(item.iter().map(|x| x + 0.01).collect());
                }
            }
        }
        let cert = mem
            .calibrate_shortlist(&queries, 5, 0.3, 0.1)
            .expect("30 exchangeable queries certify alpha=0.3");
        assert!(cert.risk_ucb <= 0.3);
        // The certified size is now the default every strategy uses.
        assert_eq!(mem.default_shortlist, cert.shortlist.max(1));
        // An impossible target leaves the default untouched.
        let before = mem.default_shortlist;
        assert!(mem.calibrate_shortlist(&queries, 5, 0.001, 0.1).is_none());
        assert_eq!(mem.default_shortlist, before);
    }
    use crate::DeterministicRng;

    fn unit(v: Vec<f32>) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    fn clustered(dim: usize, bits: usize) -> (HybridMemory, Vec<Vec<f32>>) {
        let mut rng = DeterministicRng::new(11);
        let centers: Vec<Vec<f32>> = (0..6)
            .map(|_| unit((0..dim).map(|_| rng.next_f32()).collect()))
            .collect();
        let mut m = HybridMemory::new(dim, 7, bits);
        for (c, center) in centers.iter().enumerate() {
            for i in 0..25 {
                let pt: Vec<f32> = center.iter().map(|&x| x + 0.02 * rng.next_f32()).collect();
                m.insert(&pt, format!("c{c}_{i}").as_bytes());
            }
        }
        (m, centers)
    }

    #[test]
    fn precise_recall_plus_explain_over_the_same_items() {
        let dim = 48;
        let (m, centers) = clustered(dim, 256);
        assert_eq!(m.len(), 150);

        let mut rng = DeterministicRng::new(99);
        let q: Vec<f32> = centers[4]
            .iter()
            .map(|&x| x + 0.01 * rng.next_f32())
            .collect();

        // Precise tier: top hit is from cluster 4, cosine high.
        let hits = m.recall(&q, 3, 64);
        assert!(String::from_utf8_lossy(hits[0].0).starts_with("c4_"));
        assert!(hits[0].1 > 0.9);

        // Same memory explains/zooms via the 3-D layer.
        let e = m.explain(&q, 5).unwrap();
        assert_eq!(e.neighbors.len(), 5);
        assert!(!m.zoom_path(&q, 12, 1).is_empty());
        assert!(m.export_points_json(10).starts_with("{\"count\":150"));
    }

    #[test]
    fn scored_export_is_heat_colourable() {
        let dim = 16;
        let mut m = HybridMemory::new(dim, 5, 128);
        for i in 0..10 {
            let mut v = vec![0.0f32; dim];
            v[i % dim] = 1.0;
            m.insert(&v, format!("p{i}").as_bytes());
        }
        let mut q = vec![0.0f32; dim];
        q[0] = 1.0; // identical to p0 → score 1.0
        let json = m.export_scored_json(&q, 100);
        assert!(json.contains("\"scored\":true"));
        assert!(json.contains("\"score\":"));
        assert!(json.contains("\"payload\":\"p0\""));
        assert!(json.contains("\"score\":1.0000"));
    }

    #[test]
    fn hybrid_persistence_roundtrip() {
        let dim = 48;
        let (m, centers) = clustered(dim, 256);
        let dir = std::env::temp_dir()
            .join("octasoma_hybrid_roundtrip")
            .to_string_lossy()
            .into_owned();
        std::fs::remove_dir_all(&dir).ok();
        m.save_dir(&dir).unwrap();

        let loaded = HybridMemory::open_dir(&dir, dim).unwrap();
        assert_eq!(loaded.len(), m.len());
        let q: Vec<f32> = centers[2].clone();
        let a: Vec<_> = m
            .recall(&q, 4, 32)
            .into_iter()
            .map(|(p, _)| p.to_vec())
            .collect();
        let b: Vec<_> = loaded
            .recall(&q, 4, 32)
            .into_iter()
            .map(|(p, _)| p.to_vec())
            .collect();
        assert_eq!(a, b);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn layers_stay_in_sync_on_bad_input() {
        let mut m = HybridMemory::new(4, 1, 64);
        assert!(m.insert(&[0.1, 0.2, 0.3, 0.4], b"ok"));
        assert!(!m.insert(&[0.0; 3], b"wrong-dim")); // rejected by both
        assert!(!m.insert(&[f32::NAN, 0.0, 0.0, 0.0], b"nan")); // tree rejects → sketch skipped
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn query_strategies_all_find_the_cluster() {
        let dim = 48;
        let (m, centers) = clustered(dim, 256);
        let mut rng = DeterministicRng::new(77);
        let q: Vec<f32> = centers[1]
            .iter()
            .map(|&x| x + 0.01 * rng.next_f32())
            .collect();
        for strat in [
            QueryStrategy::FastSpatial,
            QueryStrategy::PrecisionSketch,
            QueryStrategy::HybridCascade,
        ] {
            let hits = m.query(&q, strat, 3);
            assert!(!hits.is_empty(), "{strat:?} returned nothing");
            assert!(
                String::from_utf8_lossy(hits[0].0).starts_with("c1_"),
                "{strat:?}: {}",
                String::from_utf8_lossy(hits[0].0)
            );
        }
    }

    #[test]
    fn sharded_hybrid_precise_per_region() {
        use crate::HashEmbedder;
        let mut m = ShardedHybrid::new(HashEmbedder::new(128), 256);
        m.insert(
            "src/db.rs",
            "sym:src/db.rs:query",
            "build and run SQL queries",
        )
        .unwrap();
        m.insert(
            "src/db.rs",
            "sym:src/db.rs:pool",
            "a pool of db connections",
        )
        .unwrap();
        m.insert(
            "src/auth.rs",
            "sym:src/auth.rs:login",
            "authenticate a user",
        )
        .unwrap();
        assert_eq!(m.regions(), 2);
        assert_eq!(m.len(), 3);

        let hits = m
            .recall("src/db.rs", "a pool of db connections", 1)
            .unwrap();
        assert_eq!(hits[0].0, "sym:src/db.rs:pool");
        assert!(hits[0].1 > 0.99);
        // Scoped: the auth region never surfaces a db node.
        let auth = m
            .recall("src/auth.rs", "a pool of db connections", 5)
            .unwrap();
        assert!(auth.iter().all(|(u, _)| !u.starts_with("sym:src/db.rs:")));
        // Unknown region → empty / None.
        assert!(m.recall("nope", "x", 3).unwrap().is_empty());
        assert!(m.explain("nope", "x", 1).unwrap().is_none());
    }

    #[test]
    fn sharded_hybrid_persistence_roundtrip() {
        use crate::HashEmbedder;
        let mut m = ShardedHybrid::new(HashEmbedder::new(128), 256);
        m.insert(
            "src/db.rs",
            "sym:src/db.rs:pool",
            "a pool of db connections",
        )
        .unwrap();
        m.insert(
            "src/auth.rs",
            "sym:src/auth.rs:login",
            "authenticate a user",
        )
        .unwrap();
        let dir = std::env::temp_dir()
            .join("octasoma_sharded_hybrid_roundtrip")
            .to_string_lossy()
            .into_owned();
        std::fs::remove_dir_all(&dir).ok();
        m.save_dir(&dir).unwrap();

        let loaded = ShardedHybrid::open_dir(HashEmbedder::new(128), &dir).unwrap();
        assert_eq!(loaded.regions(), m.regions());
        assert_eq!(loaded.len(), m.len());
        assert_eq!(
            loaded
                .recall("src/db.rs", "a pool of db connections", 1)
                .unwrap()[0]
                .0,
            "sym:src/db.rs:pool"
        );
        // Wrong embedder dimensionality is rejected.
        assert!(ShardedHybrid::open_dir(HashEmbedder::new(64), &dir).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    // -- logical record layer integration -------------------------------------

    fn demo_record(id: &str, generation: u64) -> crate::record::MemoryRecord {
        use crate::record::{
            EmbeddingFingerprint, MemoryId, MemoryRecord, MemoryScope, Provenance,
        };
        MemoryRecord::new(
            MemoryId::new(id).unwrap(),
            b"ignored".to_vec(),
            MemoryScope::new("tenant", "workspace", "agent").unwrap(),
            Provenance::new("test-suite").unwrap(),
            EmbeddingFingerprint::new("scirust", "hash-embedder", 128).unwrap(),
            generation,
        )
    }

    #[test]
    fn remember_and_recall_visible_filter_tombstones_ttl_and_supersession() {
        use crate::HashEmbedder;
        let mut m = ShardedHybrid::new(HashEmbedder::new(128), 256);
        let now: u64 = 10_000;

        let mut ttl = demo_record("sym:r:ttl", 1);
        ttl.retention.expires_at_unix_ms = Some(5_000); // already past
        let mut fresh = demo_record("sym:r:fresh", 1);
        fresh.retention.expires_at_unix_ms = Some(50_000);

        m.remember(
            "r",
            demo_record("sym:r:tombstoned", 1),
            "a doomed fact about octrees",
        )
        .unwrap();
        m.remember("r", ttl, "an expiring fact about octrees")
            .unwrap();
        m.remember("r", fresh, "a durable fact about octrees")
            .unwrap();
        m.insert("r", "sym:r:plain", "a plain payload without a record")
            .unwrap();
        assert_eq!(m.records_len(), 3);

        // The tombstoned id is hidden; plain payloads and the live record show.
        m.tombstone("sym:r:tombstoned", 2).unwrap();
        let hits = m
            .recall_visible("r", "a fact about octrees", 10, now)
            .unwrap();
        let uris: Vec<&str> = hits.iter().map(|(u, _)| u.as_str()).collect();
        assert!(uris.contains(&"sym:r:fresh"));
        assert!(uris.contains(&"sym:r:plain"));
        assert!(
            !uris.contains(&"sym:r:tombstoned"),
            "tombstone leaked: {uris:?}"
        );
        assert!(
            !uris.contains(&"sym:r:ttl"),
            "expired record leaked: {uris:?}"
        );

        // Non-monotonic writes are refused before anything changes.
        assert!(
            m.remember("r", demo_record("sym:r:fresh", 1), "stale rewrite")
                .is_err()
        );

        // Purge removes only inactive records past their retention floor.
        let removed = m.purge_purgeable_at(now);
        assert_eq!(removed, 1); // the tombstoned one (no retention floor)
        assert!(m.record("sym:r:tombstoned").is_none());
        assert!(m.record("sym:r:fresh").is_some());
    }

    #[test]
    fn sharded_hybrid_records_survive_save_open_roundtrip() {
        use crate::HashEmbedder;
        let mut m = ShardedHybrid::new(HashEmbedder::new(128), 256);
        m.remember(
            "src/db.rs",
            demo_record("sym:src/db.rs:durable", 1),
            "durable database knowledge",
        )
        .unwrap();
        let dir = std::env::temp_dir()
            .join("octasoma_sharded_hybrid_records")
            .to_string_lossy()
            .into_owned();
        std::fs::remove_dir_all(&dir).ok();
        m.save_dir(&dir).unwrap();

        let loaded = ShardedHybrid::open_dir(HashEmbedder::new(128), &dir).unwrap();
        assert_eq!(loaded.records_len(), 1);
        assert!(loaded.record("sym:src/db.rs:durable").is_some());
        assert_eq!(
            loaded
                .recall_visible("src/db.rs", "durable database knowledge", 1, 0)
                .unwrap()[0]
                .0,
            "sym:src/db.rs:durable"
        );

        // A genuine v1 manifest (pre-record-layer: same header fields and shard
        // entries, no records flag) still opens, with an empty layer.
        let manifest = format!("{dir}/manifest.osh");
        let bytes = fs::read(&manifest).unwrap();
        let mut v1 = Vec::new();
        v1.extend_from_slice(b"OSHH");
        v1.extend_from_slice(&1u32.to_le_bytes());
        v1.extend_from_slice(&bytes[8..bytes.len() - 1]);
        fs::write(&manifest, &v1).unwrap();
        let legacy = ShardedHybrid::open_dir(HashEmbedder::new(128), &dir).unwrap();
        assert_eq!(legacy.len(), loaded.len());
        assert_eq!(legacy.records_len(), 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compact_region_reclaims_hidden_items_and_allows_resurrection() {
        use crate::HashEmbedder;
        let now: u64 = 10_000;
        let mut m = ShardedHybrid::new(HashEmbedder::new(128), 256);

        let mut doomed = demo_record("sym:r:doomed", 1);
        doomed.retention.retain_until_unix_ms = Some(500); // floor already passed
        let mut kept = demo_record("sym:r:kept", 1);
        kept.retention.expires_at_unix_ms = Some(50_000);

        m.remember("r", doomed, "a doomed fact about octrees")
            .unwrap();
        m.remember("r", kept, "a durable fact about octrees")
            .unwrap();
        m.insert("r", "sym:r:plain", "a plain payload without a record")
            .unwrap();

        m.tombstone("sym:r:doomed", 2).unwrap();
        assert!(
            m.recall_visible("r", "fact about octrees", 10, now)
                .unwrap()
                .iter()
                .all(|(u, _)| u != "sym:r:doomed")
        );

        // Compaction reclaims exactly the hidden index entry.
        assert_eq!(m.compact_region("r", now).unwrap(), 1);
        assert_eq!(m.region_len("r"), 2);
        let uris: Vec<String> = m
            .recall_visible("r", "fact about octrees", 10, now)
            .unwrap()
            .into_iter()
            .map(|(u, _)| u)
            .collect();
        assert!(uris.contains(&"sym:r:kept".to_string()));
        assert!(uris.contains(&"sym:r:plain".to_string()));

        // Resurrection is explicit and re-indexes the record.
        let revived = demo_record("sym:r:doomed", 3);
        m.remember("r", revived, "the same fact, reinstated")
            .unwrap();
        assert_eq!(m.region_len("r"), 3);
        assert!(
            m.recall_visible("r", "reinstated fact", 5, now)
                .unwrap()
                .iter()
                .any(|(u, _)| u == "sym:r:doomed")
        );

        // Unknown regions are a no-op; emptying a region removes it.
        assert_eq!(m.compact_region("nope", now).unwrap(), 0);
    }

    #[test]
    fn relation_expansion_traverses_without_leaking_across_scopes() {
        use crate::record::{MemoryScope, RelationKind as RK};
        use crate::{HashEmbedder, RecordFilter};
        let mut m = ShardedHybrid::new(HashEmbedder::new(128), 256);
        let now: u64 = 10_000;
        const US: u8 = 0x1f;
        fn key_of(p: &str) -> &str {
            p.split('\u{1f}').next().unwrap()
        }
        let packed =
            |id: &str, text: &str| -> Vec<u8> { [id.as_bytes(), &[US], text.as_bytes()].concat() };

        // acme: an anchor memory contradicting a stale one, plus a foreign
        // tenant record the anchor also references — traversal must skip it.
        m.remember_with_payload(
            "r",
            {
                let mut rec = demo_record("sym:g:new", 2);
                rec.scope = MemoryScope::new("acme", "w", "a").unwrap();
                rec
            },
            &packed("sym:g:new", "the new corrected fact"),
            "the new corrected fact",
        )
        .unwrap();
        m.remember_with_payload(
            "r",
            demo_record("sym:g:stale", 1),
            &packed("sym:g:stale", "the old wrong fact"),
            "the old wrong fact",
        )
        .unwrap();
        m.remember_with_payload(
            "r",
            {
                let mut rec = demo_record("sym:g:foreign", 1);
                rec.scope = MemoryScope::new("other-co", "w", "a").unwrap();
                rec
            },
            &packed("sym:g:foreign", "another company fact"),
            "another company fact",
        )
        .unwrap();

        // stale --SupersededBy--> new (the audit-fixed direction), and the
        // anchor confirms itself-adjacent evidence; foreign edge from `stale`.
        m.tombstone("sym:g:stale", 3).unwrap();
        m.relate("sym:g:new", RK::Supersedes, "sym:g:stale", 3)
            .unwrap();
        m.relate("sym:g:new", RK::Confirms, "sym:g:foreign", 4)
            .unwrap();

        // Direct recall under the acme-scoped tombstone filter sees only the
        // live record…
        let filter = RecordFilter {
            tenant: Some("acme".into()),
            ..RecordFilter::at(now)
        };
        let hits = m
            .recall_related(
                "r",
                "corrected fact",
                5,
                &filter,
                key_of,
                crate::Traversal {
                    hops: 0,
                    max_expanded: 8,
                },
            )
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].hop, 0);

        // …and expansion from it cannot follow Supersedes to the hidden stale
        // record nor Confirms into the foreign tenant: the filter blocks both.
        let hits = m
            .recall_related(
                "r",
                "corrected fact",
                5,
                &filter,
                key_of,
                crate::Traversal {
                    hops: 2,
                    max_expanded: 8,
                },
            )
            .unwrap();
        assert!(
            hits.iter().all(|hit| hit.hop == 0),
            "hidden or foreign target expanded: {hits:?}"
        );

        // From the *other* direction: relate two live same-tenant records and
        // expand across them.
        m.remember_with_payload(
            "r",
            {
                let mut rec = demo_record("sym:g:evidence", 1);
                rec.scope = MemoryScope::new("acme", "w", "a").unwrap();
                rec
            },
            &packed("sym:g:evidence", "supporting evidence for the fact"),
            "supporting evidence for the fact",
        )
        .unwrap();
        m.relate("sym:g:new", RK::Confirms, "sym:g:evidence", 5)
            .unwrap();

        // Expansion from the anchor alone (k=1): evidence must arrive as a
        // hop-1 row carrying the parent's cosine and the edge metadata.
        let hits = m
            .recall_related(
                "r",
                "the new corrected fact",
                1,
                &filter,
                key_of,
                crate::Traversal::default(),
            )
            .unwrap();
        assert_eq!(hits.len(), 2, "expected direct + one expansion: {hits:?}");
        let expanded = &hits[1];
        assert_eq!(expanded.hop, 1);
        assert_eq!(expanded.via_kind, Some(RK::Confirms));
        assert_eq!(expanded.via_from.as_deref(), Some("sym:g:new"));
        assert!(
            expanded.payload.starts_with("sym:g:evidence"),
            "wrong expansion target: {expanded:?}"
        );

        // The foreign tenant's record must never appear, even though the edge
        // Confirms --> sym:g:foreign exists on the same anchor.
        assert!(!hits.iter().any(|hit| hit.payload.contains("foreign")));

        // Budget cap: with max_expanded=0 nothing is appended.
        let hits = m
            .recall_related(
                "r",
                "the new corrected fact",
                1,
                &filter,
                key_of,
                crate::Traversal {
                    hops: 1,
                    max_expanded: 0,
                },
            )
            .unwrap();
        assert_eq!(hits.len(), 1);

        // Dangling targets are refused at relate() time.
        assert!(
            m.relate("sym:g:new", RK::Contradicts, "sym:g:missing", 9)
                .is_err()
        );
    }

    #[test]
    fn scoped_recall_hides_other_tenants_and_clearance_gates_sensitivity() {
        use crate::record::{MemoryScope, Sensitivity};
        use crate::{HashEmbedder, RecordFilter};
        let mut m = ShardedHybrid::new(HashEmbedder::new(128), 256);
        let now: u64 = 10_000;
        const US: u8 = 0x1f;
        fn key_of(p: &str) -> &str {
            p.split('\u{1f}').next().unwrap()
        }

        let mut scoped = demo_record("sym:r:acme", 1);
        scoped.scope = MemoryScope::new("acme", "platform", "coder").unwrap();
        scoped.sensitivity = Sensitivity::Internal;

        let mut other = demo_record("sym:r:other", 1);
        other.scope = MemoryScope::new("other-co", "platform", "coder").unwrap();

        let mut secret = demo_record("sym:r:secret", 1);
        secret.sensitivity = Sensitivity::Restricted;

        m.remember_with_payload(
            "r",
            scoped,
            &[
                b"sym:r:acme".to_vec(),
                vec![US],
                b"acme internal fact".to_vec(),
            ]
            .concat(),
            "acme internal fact",
        )
        .unwrap();
        m.remember_with_payload(
            "r",
            other,
            &[
                b"sym:r:other".to_vec(),
                vec![US],
                b"another company fact".to_vec(),
            ]
            .concat(),
            "another company fact",
        )
        .unwrap();
        m.remember_with_payload(
            "r",
            secret,
            &[
                b"sym:r:secret".to_vec(),
                vec![US],
                b"a restricted fact".to_vec(),
            ]
            .concat(),
            "a restricted fact",
        )
        .unwrap();

        let uris = |hits: Vec<(String, f32)>| -> Vec<String> {
            hits.into_iter()
                .map(|(p, _)| p.split('\u{1f}').next().unwrap().to_string())
                .collect()
        };

        // Tenant scoping hides the other company's record entirely.
        let acme_filter = RecordFilter {
            tenant: Some("acme".into()),
            ..RecordFilter::at(now)
        };
        let got = uris(
            m.recall_filtered("r", "company fact", 10, &acme_filter, key_of)
                .unwrap(),
        );
        assert!(got.contains(&"sym:r:acme".to_string()), "{got:?}");
        assert!(
            !got.contains(&"sym:r:other".to_string()),
            "tenant leak: {got:?}"
        );

        // Clearance gating: an Internal-cleared query must not see the
        // Restricted record.
        let internal_only = RecordFilter {
            clearance: Sensitivity::Internal,
            ..RecordFilter::at(now)
        };
        let got = uris(
            m.recall_filtered("r", "restricted fact", 10, &internal_only, key_of)
                .unwrap(),
        );
        assert!(
            !got.contains(&"sym:r:secret".to_string()),
            "clearance leak: {got:?}"
        );

        // The compaction contract: compacting under the Internal-clearance
        // filter reclaims exactly the Restricted entry and nothing else.
        let reclaimed = m.compact_filtered("r", &internal_only, key_of).unwrap();
        assert_eq!(reclaimed, 1);
        assert_eq!(m.region_len("r"), 2);
        // A full-clearance recall afterwards still sees both survivors.
        assert_eq!(
            m.recall_visible_by("r", "fact", 10, now, key_of)
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn compacted_state_persists_as_a_new_generation() {
        use crate::HashEmbedder;
        let now: u64 = 10_000;
        let mut m = ShardedHybrid::new(HashEmbedder::new(128), 256);
        let mut dead = demo_record("sym:r:dead", 1);
        dead.retention.retain_until_unix_ms = Some(100);
        let mut alive = demo_record("sym:r:alive", 1);
        alive.retention.expires_at_unix_ms = Some(100_000);
        m.remember("r", dead, "soon gone").unwrap();
        m.remember("r", alive, "here to stay").unwrap();
        let dir = std::env::temp_dir()
            .join("octasoma_compaction_roundtrip")
            .to_string_lossy()
            .into_owned();
        std::fs::remove_dir_all(&dir).ok();
        m.save_dir(&dir).unwrap(); // region chain: generation-1
        prune_sharded_hybrid_generations(&dir, 4).unwrap();

        m.tombstone("sym:r:dead", 2).unwrap();
        m.compact_region("r", now).unwrap();
        m.save_dir(&dir).unwrap(); // generation-2 carries the compacted region

        // The superseded generation-1 is prunable; CURRENT stays authoritative.
        assert_eq!(prune_sharded_hybrid_generations(&dir, 1).unwrap(), 1);
        let loaded = ShardedHybrid::open_dir(HashEmbedder::new(128), &dir).unwrap();
        assert_eq!(loaded.region_len("r"), 1);
        assert_eq!(
            loaded.recall_visible("r", "here to stay", 5, now).unwrap()[0].0,
            "sym:r:alive"
        );
        // The tombstoned record itself survives compaction (logical ≠ physical).
        assert!(loaded.record("sym:r:dead").is_some());
        std::fs::remove_dir_all(&dir).ok();
    }
}
