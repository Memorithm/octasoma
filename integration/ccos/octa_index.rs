//! `OctaIndex` — a drop-in **semantic** index for CCOS, backed by OctaSoma.
//!
//! This file depends only on the `octasoma` crate, so CCOS can vendor or import it
//! as-is. It gives CCOS the embedding-based recall its own `Recall::Task` lacks
//! ("a deliberately simple lexical entry point … not a semantic retriever").
//!
//! Wiring into CCOS is three lines (see `PATCH.md`): index each node on
//! `ingest_source`, and add a `Recall::Semantic(text)` arm that asks `OctaIndex`
//! for anchor node URIs which CCOS then expands through its causal graph.

use octasoma::{Embedder, FractalMemory3D, ShardedMemory};

/// A semantic index over CCOS nodes: content → embedding → 3-D octree, keyed by
/// the node's URI (`sym:…`, `mod:…`, `file:…`).
pub struct OctaIndex<E: Embedder> {
    core: FractalMemory3D,
    embedder: E,
}

impl<E: Embedder> OctaIndex<E> {
    /// Creates an empty index for the given embedder
    /// (`OllamaEmbedder` in production, `HashEmbedder` for offline tests).
    pub fn new(embedder: E) -> Self {
        let core = FractalMemory3D::new(embedder.dim(), 42);
        Self { core, embedder }
    }

    /// Loads a previously saved index (`.frac`) for `embedder`.
    pub fn open(embedder: E, path: &str) -> std::io::Result<Self> {
        let core = FractalMemory3D::load_from_disk(path, embedder.dim())?;
        Ok(Self { core, embedder })
    }

    /// Indexes a CCOS node: embed its `content`, store it under its `uri`.
    /// Call this for every node created/updated in `ingest_source`.
    pub fn index_node(&mut self, uri: &str, content: &str) {
        if let Ok(v) = self.embedder.embed(content) {
            self.core.insert(&v, Some(uri.as_bytes()));
        }
    }

    /// Returns the `k` semantically-nearest node URIs to `text`, each with a score
    /// in `(0, 1]` (`1 / (1 + distance²)`). These are the **anchors** CCOS feeds to
    /// `assemble_window` for causal expansion.
    pub fn semantic_anchors(&self, text: &str, k: usize) -> Vec<(String, f64)> {
        let Ok(v) = self.embedder.embed(text) else {
            return Vec::new();
        };
        self.core
            .nearest_embedding(&v, k)
            .into_iter()
            .filter_map(|(id, d2)| {
                self.core.get_payload(id).map(|b| {
                    (
                        String::from_utf8_lossy(b).into_owned(),
                        1.0 / (1.0 + d2 as f64),
                    )
                })
            })
            .collect()
    }

    /// Persists the index to a `.frac` file (mirror CCOS's `checkpoint`).
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        self.core.save_to_disk(path)
    }

    /// Number of indexed nodes.
    pub fn len(&self) -> usize {
        self.core.item_count()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.core.item_count() == 0
    }
}

/// Derives a CCOS **causal region** key from a node URI of the form
/// `kind:path[:symbol]` (e.g. `sym:src/db.rs:query` → `src/db.rs`,
/// `mod:src/cache.rs` → `src/cache.rs`, `file:src/main.rs` → `src/main.rs`).
///
/// Falls back to the whole URI when it doesn't match that shape. CCOS usually
/// already knows each node's file/region, so prefer the explicit
/// [`ShardedOctaIndex::index_node_in`] when you do.
pub fn region_of(uri: &str) -> String {
    // Drop the `kind:` prefix.
    let rest = uri.split_once(':').map(|(_, r)| r).unwrap_or(uri);
    // A `sym:` URI carries a trailing `:symbol`; the region is the file path.
    if uri.starts_with("sym:")
        && let Some(i) = rest.rfind(':')
    {
        return rest[..i].to_string();
    }
    rest.to_string()
}

/// A **region-sharded** semantic index for CCOS: one small OctaSoma index per
/// causal region (file). This is the deployment the real-scale benchmark
/// validated — OctaSoma's 3-D projection is a coarse router that fails as a
/// single global index but works *within* a region, so CCOS narrows causally
/// first and OctaSoma reranks inside the region it gives you.
///
/// Use [`ShardedOctaIndex::semantic_anchors_in`] when CCOS knows the region
/// (the validated 99 %-hit path); fall back to [`ShardedOctaIndex::semantic_anchors`]
/// (a coarse cross-region merge) only when no causal scope is known.
pub struct ShardedOctaIndex<E: Embedder> {
    mem: ShardedMemory<E>,
}

impl<E: Embedder> ShardedOctaIndex<E> {
    /// Creates an empty sharded index for `embedder`.
    pub fn new(embedder: E) -> Self {
        Self {
            mem: ShardedMemory::new(embedder),
        }
    }

    /// Reopens a sharded index previously written by [`ShardedOctaIndex::save`].
    pub fn open(embedder: E, dir: &str) -> std::io::Result<Self> {
        Ok(Self {
            mem: ShardedMemory::open_dir(embedder, dir)?,
        })
    }

    /// Indexes a node into an **explicit** causal region (recommended: CCOS
    /// already knows each node's file/region).
    pub fn index_node_in(&mut self, region: &str, uri: &str, content: &str) {
        let _ = self.mem.insert(region, uri, content);
    }

    /// Indexes a node, deriving its region from the URI via [`region_of`].
    pub fn index_node(&mut self, uri: &str, content: &str) {
        let region = region_of(uri);
        let _ = self.mem.insert(&region, uri, content);
    }

    /// Semantic anchors **within** a known causal region — the validated path.
    /// Scores are `1 / (1 + distance²)` in `(0, 1]`, descending.
    pub fn semantic_anchors_in(&self, region: &str, text: &str, k: usize) -> Vec<(String, f64)> {
        self.mem
            .recall_scored(region, text, k)
            .unwrap_or_default()
            .into_iter()
            .map(|(uri, d2)| (uri, 1.0 / (1.0 + d2 as f64)))
            .collect()
    }

    /// Coarse anchors across **all** regions (use only when no causal scope is
    /// known; cross-region distances are merely a heuristic).
    pub fn semantic_anchors(&self, text: &str, k: usize) -> Vec<(String, f64)> {
        self.mem
            .recall_global_scored(text, k)
            .unwrap_or_default()
            .into_iter()
            .map(|(uri, d2)| (uri, 1.0 / (1.0 + d2 as f64)))
            .collect()
    }

    /// Persists every region's shard under `dir` (mirror CCOS's `checkpoint`).
    pub fn save(&self, dir: &str) -> std::io::Result<()> {
        self.mem.save_dir(dir)
    }

    /// Number of causal regions (shards).
    pub fn regions(&self) -> usize {
        self.mem.regions()
    }

    /// Total indexed nodes across all regions.
    pub fn len(&self) -> usize {
        self.mem.len()
    }

    /// Whether nothing has been indexed yet.
    pub fn is_empty(&self) -> bool {
        self.mem.is_empty()
    }
}
