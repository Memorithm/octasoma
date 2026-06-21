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

use crate::FractalMemory3D;
use crate::embed::{EmbedError, Embedder};

/// A collection of per-region OctaSoma indices sharing one embedder.
///
/// Each region gets its own [`FractalMemory3D`] (deterministic JL projection). For
/// best recall, regions can be re-built with a PCA projection once enough samples
/// exist; the default suits online, incremental use.
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

    /// Recalls the `k` nearest payloads (uris) **within** `region` — the causal
    /// scope. Empty if the region is unknown.
    pub fn recall(&self, region: &str, query: &str, k: usize) -> Result<Vec<String>, EmbedError> {
        let Some(shard) = self.shards.get(region) else {
            return Ok(Vec::new());
        };
        let v = self.embedder.embed(query)?;
        Ok(shard
            .query_k(&v, k)
            .into_iter()
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .collect())
    }

    /// Coarse recall across **all** regions (for when no causal scope is known):
    /// merges each shard's nearest and keeps the global `k` closest.
    pub fn recall_global(&self, query: &str, k: usize) -> Result<Vec<String>, EmbedError> {
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
        Ok(hits.into_iter().map(|(_, uri)| uri).collect())
    }

    /// Number of regions (shards).
    pub fn regions(&self) -> usize {
        self.shards.len()
    }

    /// Total memories across all regions.
    pub fn len(&self) -> usize {
        self.shards.values().map(FractalMemory3D::item_count).sum()
    }

    /// Whether nothing has been stored yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
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
    fn unknown_region_is_empty() {
        let m = populated();
        assert!(
            m.recall("src/missing.rs", "anything", 3)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn global_recall_spans_regions() {
        let m = populated();
        let hits = m.recall_global("authenticate a user", 1).unwrap();
        assert_eq!(hits, vec!["sym:src/auth.rs:login".to_string()]);
    }
}
