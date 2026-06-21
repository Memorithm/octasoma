//! Region-sharded memory: one small OctaSoma index per causal region.
//!
//! The real-scale benchmark ([`docs/integration-ecosystem.md`]) showed OctaSoma's
//! 3-D projection is a *coarse router* — useless as one global index over thousands
//! of nodes (0 % exact hit), but effective *per region* (small N). [`ShardedMemory`]
//! is that deployment: keep an OctaSoma index per region key (e.g. a CCOS causal
//! region — the `file:` part of a node uri), and recall *within* a region.
//!
//! ```
//! use octasoma::{HashEmbedder, ShardedMemory};
//! let mut mem = ShardedMemory::new(HashEmbedder::new(256));
//! mem.insert("src/db.rs", "sym:src/db.rs:query", "build and run SQL queries").unwrap();
//! mem.insert("src/auth.rs", "sym:src/auth.rs:login", "authenticate a user").unwrap();
//! let hits = mem.recall("src/db.rs", "build and run SQL queries", 1).unwrap();
//! assert_eq!(hits, vec!["sym:src/db.rs:query".to_string()]);
//! ```

use std::collections::HashMap;
use std::fs;
use std::io::{self, Read};

use crate::embed::{EmbedError, Embedder};
use crate::{Explanation, FractalMemory3D};

/// Magic bytes of a sharded-memory manifest (OctaSoma Multi-Shard).
const SHARD_MAGIC: [u8; 4] = *b"OSMS";
/// Manifest format version.
const SHARD_VERSION: u32 = 1;

/// A collection of per-region OctaSoma indices sharing one embedder.
///
/// Each region gets its own [`FractalMemory3D`]. [`ShardedMemory::insert`] uses a
/// deterministic Johnson–Lindenstrauss projection, ideal for online, incremental
/// use. [`ShardedMemory::build_pca`] bulk-builds regions with a PCA projection
/// calibrated on each region's own embeddings — the higher-recall per-region
/// deployment the real-scale benchmark validated.
pub struct ShardedMemory<E: Embedder> {
    shards: HashMap<String, FractalMemory3D>,
    embedder: E,
    seed: u64,
}

impl<E: Embedder> ShardedMemory<E> {
    /// Creates an empty sharded memory backed by `embedder`.
    pub fn new(embedder: E) -> Self {
        Self {
            shards: HashMap::new(),
            embedder,
            seed: 42,
        }
    }

    /// Stores `text` (embedded) under `region`, with `uri` as the payload.
    pub fn insert(&mut self, region: &str, uri: &str, text: &str) -> Result<(), EmbedError> {
        let v = self.embedder.embed(text)?;
        let (dim, seed) = (self.embedder.dim(), self.seed);
        let shard = self
            .shards
            .entry(region.to_string())
            .or_insert_with(|| FractalMemory3D::new(dim, seed));
        shard.insert(&v, Some(uri.as_bytes()));
        Ok(())
    }

    // -- pre-embedded vectors ------------------------------------------------
    //
    // For callers that already hold vectors (cached embeddings, KV-cache tile
    // latents): these bypass the embedder, which only supplies `dim()`. The
    // vector length must equal `embedder.dim()`.

    /// Stores a pre-computed `embedding` under `region` with raw `payload` bytes,
    /// without calling the embedder (per-shard JL projection).
    pub fn insert_vec(&mut self, region: &str, payload: &[u8], embedding: &[f32]) {
        let (dim, seed) = (self.embedder.dim(), self.seed);
        let shard = self
            .shards
            .entry(region.to_string())
            .or_insert_with(|| FractalMemory3D::new(dim, seed));
        shard.insert(embedding, Some(payload));
    }

    /// Builds one region's shard from pre-computed `(payload, embedding)` pairs,
    /// calibrating its 3-D projection with PCA over those vectors. **Replaces**
    /// any existing shard of that name. No embedder calls.
    pub fn build_pca_vectors(&mut self, region: &str, items: &[(&[u8], &[f32])]) {
        let dim = self.embedder.dim();
        let flat: Vec<f32> = items.iter().flat_map(|(_, v)| v.iter().copied()).collect();
        let mut shard = FractalMemory3D::new_with_pca(dim, &flat, items.len().max(1));
        for (payload, v) in items {
            shard.insert(v, Some(payload));
        }
        self.shards.insert(region.to_string(), shard);
    }

    /// Recalls within `region` by a pre-computed query `embedding`, returning
    /// `(payload, squared distance)` ascending. Empty if the region is unknown.
    pub fn recall_vec(&self, region: &str, embedding: &[f32], k: usize) -> Vec<(Vec<u8>, f32)> {
        let Some(shard) = self.shards.get(region) else {
            return Vec::new();
        };
        shard
            .nearest_embedding(embedding, k)
            .into_iter()
            .filter_map(|(id, d2)| shard.get_payload(id).map(|p| (p.to_vec(), d2)))
            .collect()
    }

    /// Recalls the `k` nearest payloads (uris) **within** `region` — the causal
    /// scope. Empty if the region is unknown.
    pub fn recall(&self, region: &str, query: &str, k: usize) -> Result<Vec<String>, EmbedError> {
        Ok(self
            .recall_scored(region, query, k)?
            .into_iter()
            .map(|(uri, _)| uri)
            .collect())
    }

    /// Like [`ShardedMemory::recall`], but returns each hit's squared distance in
    /// the 3-D projection (ascending — smaller is closer). Useful when callers
    /// need a confidence/score, e.g. a CCOS `RecallItem.score`. Empty if the
    /// region is unknown.
    pub fn recall_scored(
        &self,
        region: &str,
        query: &str,
        k: usize,
    ) -> Result<Vec<(String, f32)>, EmbedError> {
        let Some(shard) = self.shards.get(region) else {
            return Ok(Vec::new());
        };
        let v = self.embedder.embed(query)?;
        Ok(shard
            .nearest_embedding(&v, k)
            .into_iter()
            .filter_map(|(id, d2)| {
                shard
                    .get_payload(id)
                    .map(|b| (String::from_utf8_lossy(b).into_owned(), d2))
            })
            .collect())
    }

    /// Coarse recall across **all** regions (for when no causal scope is known):
    /// merges each shard's nearest and keeps the global `k` closest.
    pub fn recall_global(&self, query: &str, k: usize) -> Result<Vec<String>, EmbedError> {
        Ok(self
            .recall_global_scored(query, k)?
            .into_iter()
            .map(|(uri, _)| uri)
            .collect())
    }

    /// Like [`ShardedMemory::recall_global`], but each hit carries its squared
    /// distance (ascending). Note: distances are only comparable *within* a
    /// region's projection, so this cross-region merge is a coarse heuristic —
    /// prefer [`ShardedMemory::recall_scored`] whenever the causal scope is known.
    pub fn recall_global_scored(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<(String, f32)>, EmbedError> {
        let v = self.embedder.embed(query)?;
        let mut hits: Vec<(f32, String)> = Vec::new();
        for shard in self.shards.values() {
            for (id, d2) in shard.nearest_embedding(&v, k) {
                if let Some(p) = shard.get_payload(id) {
                    hits.push((d2, String::from_utf8_lossy(p).into_owned()));
                }
            }
        }
        hits.sort_by(|a, b| a.0.total_cmp(&b.0));
        hits.truncate(k);
        Ok(hits.into_iter().map(|(d2, uri)| (uri, d2)).collect())
    }

    /// Explains a recall **within** `region` — the query's 3-D position, the
    /// coarse→fine fractal zoom path, and the nearest memories (payload, distance,
    /// point). The explainable, visualizable view, scoped to one causal region.
    /// `Ok(None)` for an unknown region or a wrong-dimension / non-finite query.
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

    /// Exports one region's memories as viewer JSON (`{count, half_size,
    /// points:[…]}`), or `None` if the region is unknown. Drop the result onto
    /// `viewer/index.html` to *see* a causal region's semantic map.
    pub fn export_points_json(&self, region: &str, max_points: usize) -> Option<String> {
        self.shards
            .get(region)
            .map(|s| s.export_points_json(max_points))
    }

    /// Bulk-builds shards from `(region, uri, text)` triples, calibrating **each
    /// region's** 3-D projection with PCA over that region's own embeddings — the
    /// higher-recall per-region deployment the benchmark validated (vs. the
    /// default per-shard JL projection of [`ShardedMemory::insert`]).
    ///
    /// Every region named in `items` is built fresh, **replacing** any existing
    /// shard of the same name; regions not mentioned are left untouched. Each
    /// text is embedded once via the shared embedder.
    pub fn build_pca(&mut self, items: &[(&str, &str, &str)]) -> Result<(), EmbedError> {
        // Group (uri, embedding) by region, preserving each region's input order
        // (PCA calibration is order-independent, but item insertion order is kept).
        let mut grouped: HashMap<String, Vec<(String, Vec<f32>)>> = HashMap::new();
        for (region, uri, text) in items {
            let v = self.embedder.embed(text)?;
            grouped
                .entry((*region).to_string())
                .or_default()
                .push(((*uri).to_string(), v));
        }

        let dim = self.embedder.dim();
        for (region, group) in grouped {
            let flat: Vec<f32> = group.iter().flat_map(|(_, v)| v.iter().copied()).collect();
            let mut shard = FractalMemory3D::new_with_pca(dim, &flat, group.len().max(1));
            for (uri, v) in &group {
                shard.insert(v, Some(uri.as_bytes()));
            }
            self.shards.insert(region, shard);
        }
        Ok(())
    }

    /// Number of regions (shards).
    pub fn regions(&self) -> usize {
        self.shards.len()
    }

    /// The region keys, sorted (a stable order for listing/inspection).
    pub fn region_keys(&self) -> Vec<&str> {
        let mut keys: Vec<&str> = self.shards.keys().map(String::as_str).collect();
        keys.sort_unstable();
        keys
    }

    /// Total memories across all regions.
    pub fn len(&self) -> usize {
        self.shards.values().map(FractalMemory3D::item_count).sum()
    }

    /// Whether nothing has been stored yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // -- persistence ---------------------------------------------------------

    /// Persists every shard under `dir` (one `<dir>/shard_NNNNNNNN.frac` per
    /// region) plus a binary `manifest.osm` mapping region keys to shard files.
    ///
    /// The embedder is **not** stored — reopen with the same embedder via
    /// [`ShardedMemory::open_dir`]. Region keys are sorted, so the on-disk layout
    /// is deterministic for a given set of regions.
    pub fn save_dir(&self, dir: &str) -> io::Result<()> {
        fs::create_dir_all(dir)?;
        let mut regions: Vec<&String> = self.shards.keys().collect();
        regions.sort();

        let mut manifest = Vec::new();
        manifest.extend_from_slice(&SHARD_MAGIC);
        manifest.extend_from_slice(&SHARD_VERSION.to_le_bytes());
        manifest.extend_from_slice(&(self.embedder.dim() as u32).to_le_bytes());
        manifest.extend_from_slice(&self.seed.to_le_bytes());
        manifest.extend_from_slice(&(regions.len() as u64).to_le_bytes());

        for (i, region) in regions.into_iter().enumerate() {
            let fname = format!("shard_{i:08}.frac");
            self.shards[region].save_to_disk(&format!("{dir}/{fname}"))?;
            write_bytes(&mut manifest, region.as_bytes());
            write_bytes(&mut manifest, fname.as_bytes());
        }
        fs::write(format!("{dir}/manifest.osm"), manifest)
    }

    /// Reopens a [`ShardedMemory`] previously written by [`ShardedMemory::save_dir`],
    /// binding it to `embedder` (whose [`Embedder::dim`] must match the saved index).
    pub fn open_dir(embedder: E, dir: &str) -> io::Result<Self> {
        let bytes = fs::read(format!("{dir}/manifest.osm"))?;
        let mut r: &[u8] = &bytes;

        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if magic != SHARD_MAGIC {
            return Err(invalid("not an OctaSoma shard manifest (bad magic)"));
        }
        let version = read_u32(&mut r)?;
        if version != SHARD_VERSION {
            return Err(invalid(&format!(
                "unsupported shard manifest version {version} (this build reads v{SHARD_VERSION})"
            )));
        }
        let high_dim = read_u32(&mut r)? as usize;
        if high_dim != embedder.dim() {
            return Err(invalid(&format!(
                "dim mismatch: manifest has {high_dim}, embedder has {}",
                embedder.dim()
            )));
        }
        let seed = read_u64(&mut r)?;
        let count = read_u64(&mut r)? as usize;

        let mut shards = HashMap::with_capacity(count);
        for _ in 0..count {
            let region = read_string(&mut r)?;
            let fname = read_string(&mut r)?;
            let shard = FractalMemory3D::load_from_disk(&format!("{dir}/{fname}"), high_dim)?;
            shards.insert(region, shard);
        }
        Ok(Self {
            shards,
            embedder,
            seed,
        })
    }
}

// ---------------------------------------------------------------------------
// Manifest (de)serialisation helpers — little-endian, length-prefixed.
// ---------------------------------------------------------------------------

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
    use crate::HashEmbedder;

    fn populated() -> ShardedMemory<HashEmbedder> {
        let mut m = ShardedMemory::new(HashEmbedder::new(128));
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
        m
    }

    #[test]
    fn recall_is_scoped_to_region() {
        let m = populated();
        assert_eq!(m.regions(), 2);
        assert_eq!(m.len(), 3);
        // Within a region, the exact text recalls its own uri (deterministic embedder).
        assert_eq!(
            m.recall("src/db.rs", "build and run SQL queries", 1)
                .unwrap(),
            vec!["sym:src/db.rs:query".to_string()]
        );
        // The db query is NOT reachable from the auth region.
        let auth = m
            .recall("src/auth.rs", "build and run SQL queries", 5)
            .unwrap();
        assert!(!auth.contains(&"sym:src/db.rs:query".to_string()));
    }

    #[test]
    fn explain_and_export_are_region_scoped() {
        let m = populated();
        // explain within a region: neighbors + a zoom path rooted at the region.
        let e = m
            .explain("src/db.rs", "build and run SQL queries", 2)
            .unwrap()
            .unwrap();
        assert!(e.query_point.iter().all(|c| c.is_finite()));
        assert!(!e.neighbors.is_empty());
        assert_eq!(e.zoom_path[0].count, 2); // the db region holds 2 memories
        assert!(
            e.neighbors
                .iter()
                .all(|nb| String::from_utf8_lossy(&nb.payload).starts_with("sym:src/db.rs:"))
        );
        // Unknown region → Ok(None), not an error.
        assert!(m.explain("nope", "x", 1).unwrap().is_none());
        // Viewer export is scoped to the region (db has 2 points).
        let json = m.export_points_json("src/db.rs", 100).unwrap();
        assert!(json.starts_with("{\"count\":2"));
        assert!(m.export_points_json("nope", 10).is_none());
    }

    #[test]
    fn unknown_region_is_empty() {
        let m = populated();
        assert!(
            m.recall("src/missing.rs", "anything", 3)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn recall_scored_is_ascending_and_scoped() {
        let m = populated();
        let scored = m
            .recall_scored("src/db.rs", "build and run SQL queries", 2)
            .unwrap();
        assert_eq!(scored.len(), 2);
        // Exact-text hit is the query's own uri at distance 0; scores ascend.
        assert_eq!(scored[0].0, "sym:src/db.rs:query");
        assert!(scored[0].1 <= scored[1].1, "distances must be ascending");
        // All hits stay inside the region.
        assert!(
            scored
                .iter()
                .all(|(uri, _)| uri.starts_with("sym:src/db.rs:"))
        );
        // Unknown region yields no scores.
        assert!(m.recall_scored("nope", "x", 3).unwrap().is_empty());
    }

    #[test]
    fn global_recall_spans_regions() {
        let m = populated();
        let hits = m.recall_global("authenticate a user", 1).unwrap();
        assert_eq!(hits, vec!["sym:src/auth.rs:login".to_string()]);
    }

    #[test]
    fn build_pca_indexes_per_region_and_replaces() {
        let mut m = ShardedMemory::new(HashEmbedder::new(64));
        let items = [
            (
                "src/db.rs",
                "sym:src/db.rs:query",
                "build and run SQL queries",
            ),
            (
                "src/db.rs",
                "sym:src/db.rs:pool",
                "a pool of db connections",
            ),
            (
                "src/db.rs",
                "sym:src/db.rs:tx",
                "run a database transaction",
            ),
            (
                "src/auth.rs",
                "sym:src/auth.rs:login",
                "authenticate a user",
            ),
            ("src/auth.rs", "sym:src/auth.rs:token", "verify a JWT token"),
        ];
        m.build_pca(&items).unwrap();
        assert_eq!(m.regions(), 2);
        assert_eq!(m.len(), 5);
        // Exact-text recall within a region returns its uri (PCA keeps self-dist 0).
        assert_eq!(
            m.recall("src/db.rs", "a pool of db connections", 1)
                .unwrap(),
            vec!["sym:src/db.rs:pool".to_string()]
        );
        // Rebuilding a region replaces it; other regions are untouched.
        m.build_pca(&[("src/db.rs", "sym:src/db.rs:only", "single survivor")])
            .unwrap();
        assert_eq!(
            m.recall("src/db.rs", "single survivor", 1).unwrap(),
            vec!["sym:src/db.rs:only".to_string()]
        );
        assert_eq!(m.len(), 3); // db rebuilt to 1 + auth's 2
    }

    #[test]
    fn vector_api_inserts_builds_and_recalls() {
        // Pre-embedded vectors (no embedder calls); dim must match embedder.dim().
        let mut m = ShardedMemory::new(HashEmbedder::new(4));
        m.insert_vec("a", b"a0", &[1.0, 0.0, 0.0, 0.0]);
        m.insert_vec("a", b"a1", &[0.0, 1.0, 0.0, 0.0]);
        let b0 = [0.0f32, 0.0, 1.0, 0.0];
        let b1 = [0.0f32, 0.0, 0.0, 1.0];
        m.build_pca_vectors("b", &[(b"b0", &b0), (b"b1", &b1)]);

        assert_eq!(m.regions(), 2);
        assert_eq!(m.len(), 4);

        // recall_vec returns the exact vector's payload at distance ~0.
        let hits = m.recall_vec("a", &[1.0, 0.0, 0.0, 0.0], 1);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, b"a0".to_vec());
        assert!(hits[0].1.abs() < 1e-6);

        // Scoping: region b's payload is unreachable from region a.
        let from_a = m.recall_vec("a", &b0, 5);
        assert!(from_a.iter().all(|(p, _)| *p != b"b0".to_vec()));
        // Unknown region is empty.
        assert!(m.recall_vec("nope", &[1.0, 0.0, 0.0, 0.0], 3).is_empty());
    }

    #[test]
    fn save_dir_and_open_dir_roundtrip() {
        let m = populated();
        let dir = "/tmp/octasoma_sharded_roundtrip";
        std::fs::remove_dir_all(dir).ok();
        m.save_dir(dir).unwrap();

        let loaded = ShardedMemory::open_dir(HashEmbedder::new(128), dir).unwrap();
        assert_eq!(loaded.regions(), m.regions());
        assert_eq!(loaded.len(), m.len());
        // Recall is identical after a round-trip (projection + payloads restored).
        assert_eq!(
            loaded
                .recall("src/db.rs", "build and run SQL queries", 1)
                .unwrap(),
            vec!["sym:src/db.rs:query".to_string()]
        );
        // An embedder with the wrong dimensionality is rejected.
        assert!(ShardedMemory::open_dir(HashEmbedder::new(64), dir).is_err());

        std::fs::remove_dir_all(dir).ok();
    }
}
