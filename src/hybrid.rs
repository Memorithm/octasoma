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

use std::collections::HashMap;
use std::fs;
use std::io::{self, Read};

use crate::embed::{EmbedError, Embedder};
use crate::{Explanation, FractalMemory3D, RegionView, SketchIndex};

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
    tree: FractalMemory3D,
    sketch: SketchIndex,
    dim: usize,
    default_shortlist: usize,
}

impl HybridMemory {
    /// Creates a hybrid memory: a deterministic JL 3-D projection (from `seed`) and
    /// `bits`-wide SimHash sketches.
    pub fn new(dim: usize, seed: u64, bits: usize) -> Self {
        Self {
            tree: FractalMemory3D::new(dim, seed),
            // Decorrelate the sketch hyperplanes from the JL projection.
            sketch: SketchIndex::new(dim, bits, seed ^ 0x9E37_79B9_7F4A_7C15),
            dim,
            default_shortlist: 256,
        }
    }

    /// Sets the default shortlist size used by [`HybridMemory::query`] (a builder).
    pub fn with_shortlist(mut self, shortlist: usize) -> Self {
        self.default_shortlist = shortlist.max(1);
        self
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

    /// Persists both layers under `dir` (`tree.frac` + `index.skch`).
    pub fn save_dir(&self, dir: &str) -> io::Result<()> {
        fs::create_dir_all(dir)?;
        self.tree.save_to_disk(&format!("{dir}/tree.frac"))?;
        self.sketch.save_to_disk(&format!("{dir}/index.skch"))
    }

    /// Reopens a hybrid memory written by [`HybridMemory::save_dir`], for
    /// `dim`-dimensional embeddings.
    pub fn open_dir(dir: &str, dim: usize) -> io::Result<Self> {
        let tree = FractalMemory3D::load_from_disk(&format!("{dir}/tree.frac"), dim)?;
        let sketch = SketchIndex::load_from_disk(&format!("{dir}/index.skch"), dim)?;
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
}

impl<E: Embedder> ShardedHybrid<E> {
    /// Creates an empty sharded-hybrid memory with `bits`-wide sketches per region.
    pub fn new(embedder: E, bits: usize) -> Self {
        Self {
            shards: HashMap::new(),
            embedder,
            seed: 42,
            bits,
        }
    }

    /// Embeds `text` and stores it under `region`, with `uri` as the payload.
    pub fn insert(&mut self, region: &str, uri: &str, text: &str) -> Result<(), EmbedError> {
        let v = self.embedder.embed(text)?;
        let (dim, seed, bits) = (self.embedder.dim(), self.seed, self.bits);
        let shard = self
            .shards
            .entry(region.to_string())
            .or_insert_with(|| HybridMemory::new(dim, seed, bits));
        shard.insert(&v, uri.as_bytes());
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
        let v = self.embedder.embed(query)?;
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
        let v = self.embedder.embed(query)?;
        let mut hits: Vec<(String, f32)> = Vec::new();
        for shard in self.shards.values() {
            for (p, s) in shard.query(&v, QueryStrategy::PrecisionSketch, k) {
                hits.push((String::from_utf8_lossy(p).into_owned(), s));
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
        let v = self.embedder.embed(query)?;
        Ok(shard.explain(&v, k))
    }

    /// Number of regions (shards).
    pub fn regions(&self) -> usize {
        self.shards.len()
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
    /// plus a binary manifest. Reopen with the same embedder via
    /// [`ShardedHybrid::open_dir`].
    pub fn save_dir(&self, dir: &str) -> io::Result<()> {
        fs::create_dir_all(dir)?;
        let mut regions: Vec<&String> = self.shards.keys().collect();
        regions.sort();

        let mut m = Vec::new();
        m.extend_from_slice(b"OSHH");
        m.extend_from_slice(&1u32.to_le_bytes());
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
        fs::write(format!("{dir}/manifest.osh"), m)
    }

    /// Reopens a sharded-hybrid memory written by [`ShardedHybrid::save_dir`], bound
    /// to `embedder` (whose `dim()` must match) and `bits` from the manifest.
    pub fn open_dir(embedder: E, dir: &str) -> io::Result<Self> {
        let bytes = fs::read(format!("{dir}/manifest.osh"))?;
        let mut r: &[u8] = &bytes;
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != b"OSHH" {
            return Err(invalid("not a sharded-hybrid manifest (bad magic)"));
        }
        let version = read_u32(&mut r)?;
        if version != 1 {
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
        let mut shards = HashMap::with_capacity(count);
        for _ in 0..count {
            let region = read_string(&mut r)?;
            let name = read_string(&mut r)?;
            let hm = HybridMemory::open_dir(&format!("{dir}/{name}"), dim)?;
            shards.insert(region, hm);
        }
        Ok(Self {
            shards,
            embedder,
            seed,
            bits,
        })
    }
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

fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_string<R: Read>(r: &mut R) -> io::Result<String> {
    let len = read_u64(r)? as usize;
    let mut b = vec![0u8; len];
    r.read_exact(&mut b)?;
    String::from_utf8(b).map_err(|e| invalid(&e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let dir = "/tmp/octasoma_hybrid_roundtrip";
        std::fs::remove_dir_all(dir).ok();
        m.save_dir(dir).unwrap();

        let loaded = HybridMemory::open_dir(dir, dim).unwrap();
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
        std::fs::remove_dir_all(dir).ok();
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
        let dir = "/tmp/octasoma_sharded_hybrid_roundtrip";
        std::fs::remove_dir_all(dir).ok();
        m.save_dir(dir).unwrap();

        let loaded = ShardedHybrid::open_dir(HashEmbedder::new(128), dir).unwrap();
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
        assert!(ShardedHybrid::open_dir(HashEmbedder::new(64), dir).is_err());
        std::fs::remove_dir_all(dir).ok();
    }
}
