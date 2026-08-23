//! High-level agent memory built on the [`FractalMemory3D`] engine.
//!
//! [`OctaSomaAgent`] wraps the engine with an [`Embedder`], turning raw text
//! observations into stored memories (`perceive`) and queries into recalled
//! context (`recall` / `reflect`). It is generic over the embedder, so it runs
//! fully offline with [`HashEmbedder`](crate::HashEmbedder) and against a real
//! model with [`OllamaEmbedder`](crate::OllamaEmbedder) — no code change.

use crate::FractalMemory3D;
use crate::embed::{EmbedError, Embedder};

/// Text-in / text-out semantic memory for an agent loop.
pub struct OctaSomaAgent<E: Embedder> {
    core: FractalMemory3D,
    embedder: E,
}

impl<E: Embedder> OctaSomaAgent<E> {
    /// Creates an agent with a deterministic Johnson–Lindenstrauss projection.
    pub fn new(embedder: E, seed: u64) -> Self {
        let core = FractalMemory3D::new(embedder.dim(), seed);
        Self { core, embedder }
    }

    /// Creates an agent whose projection is learned by PCA on a text corpus.
    ///
    /// The corpus is embedded once to calibrate the 3-D projection; it is *not*
    /// stored as memories (call [`perceive`](Self::perceive) for that).
    pub fn calibrate(embedder: E, corpus: &[&str]) -> Result<Self, EmbedError> {
        if corpus.is_empty() {
            return Err(EmbedError::Protocol(
                "PCA calibration corpus must not be empty".to_string(),
            ));
        }
        let dim = embedder.dim();
        let embeddings = embedder.embed_batch(corpus)?;
        let flat: Vec<f32> = embeddings.iter().flatten().copied().collect();
        let core = FractalMemory3D::new_with_pca(dim, &flat, embeddings.len());
        Ok(Self { core, embedder })
    }

    /// Loads a previously saved agent memory (`.frac`) and attaches `embedder`.
    pub fn from_file(embedder: E, path: &str) -> std::io::Result<Self> {
        let core = FractalMemory3D::load_from_disk(path, embedder.dim())?;
        Ok(Self { core, embedder })
    }

    /// Embeds an observation and stores it (the text itself is the payload).
    pub fn perceive(&mut self, text: &str) -> Result<(), EmbedError> {
        let vec = self.embedder.embed_checked(text)?;
        if self.core.insert(&vec, Some(text.as_bytes())).is_none() {
            return Err(EmbedError::Protocol(
                "validated embedding could not be projected into the memory index".to_string(),
            ));
        }
        Ok(())
    }

    /// Returns up to `k` topically-nearest memories, nearest first.
    pub fn recall(&self, query: &str, k: usize) -> Result<Vec<String>, EmbedError> {
        let vec = self.embedder.embed_checked(query)?;
        Ok(self
            .core
            .query_k(&vec, k)
            .into_iter()
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .collect())
    }

    /// Like [`OctaSomaAgent::recall`], with each memory's similarity score in
    /// `(0, 1]` (`1 / (1 + distance²)` — the same convention as the CCOS
    /// adapter). The scores are what a relevance-feedback log
    /// ([`crate::RelevanceFeedback`]) records for the calibrated tiers.
    pub fn recall_scored(&self, query: &str, k: usize) -> Result<Vec<(String, f32)>, EmbedError> {
        let vec = self.embedder.embed_checked(query)?;
        Ok(self
            .core
            .nearest_embedding(&vec, k)
            .into_iter()
            .filter_map(|(id, d2)| {
                self.core
                    .get_payload(id)
                    .map(|b| (String::from_utf8_lossy(b).into_owned(), 1.0 / (1.0 + d2)))
            })
            .collect())
    }

    /// Convenience: the `k` recalled memories joined into one context block,
    /// ready to inject into an LLM prompt.
    pub fn reflect(&self, query: &str, k: usize) -> Result<String, EmbedError> {
        Ok(self.recall(query, k)?.join("\n"))
    }

    /// Explains a recall: the query's 3-D position, the coarse→fine regions it
    /// falls through, and the `k` nearest memories with distances and positions.
    /// `Ok(None)` only if the validated embedding still cannot be projected.
    pub fn explain(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Option<crate::explain::Explanation>, EmbedError> {
        let v = self.embedder.embed_checked(query)?;
        Ok(self.core.explain(&v, k))
    }

    /// Exports up to `max_points` memories as JSON for a 3-D viewer.
    pub fn export_points_json(&self, max_points: usize) -> String {
        self.core.export_points_json(max_points)
    }

    /// Zooms to the region a query falls in at `level` (0 = the whole memory,
    /// deeper = finer), summarised. `Ok(None)` only if projection fails.
    pub fn zoom(
        &self,
        query: &str,
        level: u32,
        samples: usize,
    ) -> Result<Option<crate::RegionView>, EmbedError> {
        let v = self.embedder.embed_checked(query)?;
        Ok(self.core.zoom(&v, level, samples))
    }

    /// The coarse→fine path of regions a query falls through — progressive recall
    /// from the broad theme near the root to the exact memory at a leaf.
    pub fn zoom_path(
        &self,
        query: &str,
        max_level: u32,
        samples: usize,
    ) -> Result<Vec<crate::RegionView>, EmbedError> {
        let v = self.embedder.embed_checked(query)?;
        Ok(self.core.zoom_path(&v, max_level, samples))
    }

    /// Persists the memory to a `.frac` file.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        self.core.save_to_disk(path)
    }

    /// Number of stored memories.
    pub fn len(&self) -> usize {
        self.core.item_count()
    }

    /// Whether no memories are stored yet.
    pub fn is_empty(&self) -> bool {
        self.core.item_count() == 0
    }

    /// Borrow the underlying engine (for stats, persistence, advanced queries).
    pub fn core(&self) -> &FractalMemory3D {
        &self.core
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HashEmbedder;

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
    fn perceive_then_recall_roundtrip() {
        let mut agent = OctaSomaAgent::new(HashEmbedder::new(128), 42);
        assert!(agent.is_empty());
        for text in ["the cat sat", "rust is fast", "octrees subdivide space"] {
            agent.perceive(text).unwrap();
        }
        assert_eq!(agent.len(), 3);
        // With a deterministic embedder, querying a stored text recalls itself
        // (its own vector is the exact nearest neighbour).
        let hits = agent.recall("rust is fast", 1).unwrap();
        assert_eq!(hits, vec!["rust is fast".to_string()]);
    }

    #[test]
    fn malformed_embedding_is_rejected_without_reporting_success() {
        let mut wrong_dim = OctaSomaAgent::new(WrongDimEmbedder, 42);
        assert!(wrong_dim.perceive("bad").is_err());
        assert!(wrong_dim.is_empty());

        let mut non_finite = OctaSomaAgent::new(NonFiniteEmbedder, 42);
        assert!(non_finite.perceive("bad").is_err());
        assert!(non_finite.is_empty());
    }

    #[test]
    fn calibration_rejects_empty_and_malformed_corpora_without_panicking() {
        assert!(OctaSomaAgent::calibrate(HashEmbedder::new(32), &[]).is_err());
        assert!(OctaSomaAgent::calibrate(WrongDimEmbedder, &["bad"]).is_err());
        assert!(OctaSomaAgent::calibrate(NonFiniteEmbedder, &["bad"]).is_err());
    }

    #[test]
    fn reflect_joins_memories() {
        let mut agent = OctaSomaAgent::new(HashEmbedder::new(64), 7);
        agent.perceive("alpha").unwrap();
        let ctx = agent.reflect("alpha", 3).unwrap();
        assert!(ctx.contains("alpha"));
    }

    #[test]
    fn calibrate_builds_pca_agent() {
        let corpus = ["a", "b", "c", "d", "e", "f"];
        let agent = OctaSomaAgent::calibrate(HashEmbedder::new(32), &corpus).unwrap();
        assert_eq!(agent.core().high_dim, 32);
        assert!(agent.is_empty()); // calibration does not store memories
    }

    #[test]
    fn save_and_reload() {
        let mut agent = OctaSomaAgent::new(HashEmbedder::new(48), 1);
        agent.perceive("persist me").unwrap();
        let path = std::env::temp_dir()
            .join("octasoma_agent_test.frac")
            .to_string_lossy()
            .into_owned();
        agent.save(&path).unwrap();
        let reloaded = OctaSomaAgent::from_file(HashEmbedder::new(48), &path).unwrap();
        assert_eq!(
            reloaded.recall("persist me", 1).unwrap(),
            vec!["persist me"]
        );
        std::fs::remove_file(path).ok();
    }
}
