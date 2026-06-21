//! `OctaIndex` — a drop-in **semantic** index for CCOS, backed by OctaSoma.
//!
//! This file depends only on the `octasoma` crate, so CCOS can vendor or import it
//! as-is. It gives CCOS the embedding-based recall its own `Recall::Task` lacks
//! ("a deliberately simple lexical entry point … not a semantic retriever").
//!
//! Wiring into CCOS is three lines (see `PATCH.md`): index each node on
//! `ingest_source`, and add a `Recall::Semantic(text)` arm that asks `OctaIndex`
//! for anchor node URIs which CCOS then expands through its causal graph.

use octasoma::{Embedder, FractalMemory3D};

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
