//! `OctaIndex` — a fallible embedding index adapter for CCOS, backed by OctaSoma.
//!
//! This file depends only on the `octasoma` crate, so CCOS can vendor or import it
//! as-is after requalification. This is a destination-facing contract, not proof
//! of adoption or retrieval quality in current CCOS. `HashEmbedder` is offline
//! and non-semantic; semantic quality depends on the chosen model and dataset.
//!
//! Wiring into a fallible CCOS boundary (see `PATCH.md`): index each node on
//! `ingest_source`, and add a `Recall::Semantic(text)` arm that asks `OctaIndex`
//! for anchor node URIs which CCOS then expands through its causal graph.

use octasoma::{EmbedError, Embedder, FractalMemory3D, ShardedMemory};

/// Version of the fallible CCOS anchor contract (anchor tuples are unchanged).
pub const ADAPTER_CONTRACT: &str = "octasoma.ccos-anchor@2";

/// Admission bounds checked before invoking the embedder or growing an index.
#[derive(Clone, Copy, Debug)]
pub struct IndexBudget {
    pub max_text_bytes: usize,
    pub max_key_bytes: usize,
    pub max_anchors: usize,
    pub max_items: usize,
}

impl Default for IndexBudget {
    fn default() -> Self {
        Self {
            max_text_bytes: 1 << 20,
            max_key_bytes: 4096,
            max_anchors: 1024,
            max_items: 1_000_000,
        }
    }
}

/// Failure categories are distinct from a successful empty recall (a miss).
#[derive(Debug)]
pub enum IndexErrorKind {
    EmbedderUnavailable(std::io::Error),
    InvalidFormat(String),
    IndexRejected,
    BudgetExceeded(&'static str),
}

/// Bounded operation context for a failed handoff; source text is never retained.
#[derive(Debug)]
pub struct IndexError {
    pub operation: &'static str,
    pub region: Option<String>,
    pub uri: Option<String>,
    pub kind: IndexErrorKind,
}

impl std::fmt::Display for IndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {:?} {:?}: {:?}",
            self.operation, self.region, self.uri, self.kind
        )
    }
}

impl std::error::Error for IndexError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            IndexErrorKind::EmbedderUnavailable(error) => Some(error),
            _ => None,
        }
    }
}

/// Adapter implementation provenance, not proof of model semantics or deployment.
#[derive(Debug)]
pub struct AdapterProvenance {
    pub contract: &'static str,
    pub embedder_type: &'static str,
    pub dimension: usize,
}

impl IndexBudget {
    fn error(
        &self,
        operation: &'static str,
        region: Option<&str>,
        uri: Option<&str>,
        kind: IndexErrorKind,
    ) -> IndexError {
        IndexError {
            operation,
            region: region
                .filter(|key| key.len() <= self.max_key_bytes)
                .map(str::to_owned),
            uri: uri
                .filter(|key| key.len() <= self.max_key_bytes)
                .map(str::to_owned),
            kind,
        }
    }

    fn check(
        &self,
        operation: &'static str,
        region: Option<&str>,
        uri: Option<&str>,
        text: &str,
        k: usize,
        items: Option<usize>,
    ) -> Result<(), IndexError> {
        for key in region.into_iter().chain(uri) {
            if key.len() > self.max_key_bytes {
                return Err(self.error(
                    operation,
                    region,
                    uri,
                    IndexErrorKind::BudgetExceeded("key_bytes"),
                ));
            }
            if key.is_empty() || key.chars().any(char::is_control) {
                return Err(self.error(
                    operation,
                    region,
                    uri,
                    IndexErrorKind::InvalidFormat("empty or control-containing key".into()),
                ));
            }
        }
        let refused = if text.len() > self.max_text_bytes {
            Some("text_bytes")
        } else if k > self.max_anchors {
            Some("anchors")
        } else if items.is_some_and(|count| count >= self.max_items) {
            Some("items")
        } else {
            None
        };
        match refused {
            Some(bound) => Err(self.error(
                operation,
                region,
                uri,
                IndexErrorKind::BudgetExceeded(bound),
            )),
            None => Ok(()),
        }
    }

    fn embedding_error(
        &self,
        operation: &'static str,
        region: Option<&str>,
        uri: Option<&str>,
        error: EmbedError,
    ) -> IndexError {
        let kind = match error {
            EmbedError::Io(error) => IndexErrorKind::EmbedderUnavailable(error),
            EmbedError::Protocol(message) => IndexErrorKind::InvalidFormat(message),
        };
        self.error(operation, region, uri, kind)
    }
}

fn score(distance: f32) -> Result<f64, IndexErrorKind> {
    if !distance.is_finite() || distance < 0.0 {
        return Err(IndexErrorKind::InvalidFormat(
            "non-finite or negative anchor distance".into(),
        ));
    }
    Ok(1.0 / (1.0 + f64::from(distance)))
}

/// A semantic index over CCOS nodes: content → embedding → 3-D octree, keyed by
/// the node's URI (`sym:…`, `mod:…`, `file:…`).
pub struct OctaIndex<E: Embedder> {
    core: FractalMemory3D,
    embedder: E,
    budget: IndexBudget,
}

impl<E: Embedder> OctaIndex<E> {
    /// Creates an empty index for the given embedder
    /// (`OllamaEmbedder` in production, `HashEmbedder` for offline tests).
    pub fn new(embedder: E) -> Self {
        Self::with_budget(embedder, IndexBudget::default())
    }

    pub fn with_budget(embedder: E, budget: IndexBudget) -> Self {
        let core = FractalMemory3D::new(embedder.dim(), 42);
        Self {
            core,
            embedder,
            budget,
        }
    }

    /// Loads a previously saved index (`.frac`) for `embedder`.
    pub fn open(embedder: E, path: &str) -> std::io::Result<Self> {
        let core = FractalMemory3D::load_from_disk(path, embedder.dim())?;
        Ok(Self {
            core,
            embedder,
            budget: IndexBudget::default(),
        })
    }

    /// Indexes a CCOS node: embed its `content`, store it under its `uri`.
    /// Call this for every node created/updated in `ingest_source`.
    pub fn index_node(&mut self, uri: &str, content: &str) -> Result<(), IndexError> {
        self.budget
            .check("insert", None, Some(uri), content, 0, Some(self.len()))?;
        let v = self.embedder.embed_checked(content).map_err(|error| {
            self.budget
                .embedding_error("insert", None, Some(uri), error)
        })?;
        self.core.insert(&v, Some(uri.as_bytes())).ok_or_else(|| {
            self.budget
                .error("insert", None, Some(uri), IndexErrorKind::IndexRejected)
        })?;
        Ok(())
    }

    /// Returns the `k` semantically-nearest node URIs to `text`, each with a score
    /// in `(0, 1]` (`1 / (1 + distance²)`). These are the **anchors** CCOS feeds to
    /// `assemble_window` for causal expansion.
    pub fn semantic_anchors(&self, text: &str, k: usize) -> Result<Vec<(String, f64)>, IndexError> {
        self.budget.check("recall", None, None, text, k, None)?;
        if k == 0 || self.is_empty() {
            return Ok(Vec::new());
        }
        let v = self
            .embedder
            .embed_checked(text)
            .map_err(|error| self.budget.embedding_error("recall", None, None, error))?;
        if self
            .core
            .project(&v)
            .is_none_or(|point| !point.iter().all(|c| c.is_finite()))
        {
            return Err(self
                .budget
                .error("recall", None, None, IndexErrorKind::IndexRejected));
        }
        self.core
            .nearest_embedding(&v, k)
            .into_iter()
            .map(|(id, d2)| {
                let bytes = self.core.get_payload(id).ok_or_else(|| {
                    self.budget.error(
                        "recall",
                        None,
                        None,
                        IndexErrorKind::InvalidFormat("missing anchor payload".into()),
                    )
                })?;
                let uri = std::str::from_utf8(bytes).map_err(|_| {
                    self.budget.error(
                        "recall",
                        None,
                        None,
                        IndexErrorKind::InvalidFormat("anchor is not UTF-8".into()),
                    )
                })?;
                self.budget.check("recall", None, Some(uri), "", 0, None)?;
                let value =
                    score(d2).map_err(|kind| self.budget.error("recall", None, Some(uri), kind))?;
                Ok((uri.to_owned(), value))
            })
            .collect()
    }

    pub fn provenance(&self) -> AdapterProvenance {
        AdapterProvenance {
            contract: ADAPTER_CONTRACT,
            embedder_type: std::any::type_name::<E>(),
            dimension: self.embedder.dim(),
        }
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
    if !uri.starts_with("sym:") {
        return rest.to_string();
    }
    if let Some(i) = rest.rfind(':') {
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
    budget: IndexBudget,
    dimension: usize,
}

impl<E: Embedder> ShardedOctaIndex<E> {
    /// Creates an empty sharded index for `embedder`.
    pub fn new(embedder: E) -> Self {
        Self::with_budget(embedder, IndexBudget::default())
    }

    pub fn with_budget(embedder: E, budget: IndexBudget) -> Self {
        let dimension = embedder.dim();
        Self {
            mem: ShardedMemory::new(embedder),
            budget,
            dimension,
        }
    }

    /// Reopens a sharded index previously written by [`ShardedOctaIndex::save`].
    pub fn open(embedder: E, dir: &str) -> std::io::Result<Self> {
        let dimension = embedder.dim();
        Ok(Self {
            mem: ShardedMemory::open_dir(embedder, dir)?,
            budget: IndexBudget::default(),
            dimension,
        })
    }

    /// Indexes a node into an **explicit** causal region (recommended: CCOS
    /// already knows each node's file/region).
    pub fn index_node_in(
        &mut self,
        region: &str,
        uri: &str,
        content: &str,
    ) -> Result<(), IndexError> {
        self.budget.check(
            "insert",
            Some(region),
            Some(uri),
            content,
            0,
            Some(self.len()),
        )?;
        self.mem.insert(region, uri, content).map_err(|error| {
            self.budget
                .embedding_error("insert", Some(region), Some(uri), error)
        })
    }

    /// Indexes a node, deriving its region from the URI via [`region_of`].
    pub fn index_node(&mut self, uri: &str, content: &str) -> Result<(), IndexError> {
        // Validate the URI before allocating its derived region.
        self.budget
            .check("insert", None, Some(uri), content, 0, Some(self.len()))?;
        let region = region_of(uri);
        self.index_node_in(&region, uri, content)
    }

    /// Semantic anchors **within** a known causal region — the validated path.
    /// Scores are `1 / (1 + distance²)` in `(0, 1]`, descending.
    pub fn semantic_anchors_in(
        &self,
        region: &str,
        text: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>, IndexError> {
        self.budget
            .check("recall", Some(region), None, text, k, None)?;
        if k == 0 {
            return Ok(Vec::new());
        }
        let hits = self.mem.recall_scored(region, text, k).map_err(|error| {
            self.budget
                .embedding_error("recall", Some(region), None, error)
        })?;
        hits.into_iter()
            .map(|(uri, d2)| {
                self.budget
                    .check("recall", Some(region), Some(&uri), "", 0, None)?;
                let value = score(d2)
                    .map_err(|kind| self.budget.error("recall", Some(region), Some(&uri), kind))?;
                Ok((uri, value))
            })
            .collect()
    }

    /// Coarse anchors across **all** regions (use only when no causal scope is
    /// known; cross-region distances are merely a heuristic).
    pub fn semantic_anchors(&self, text: &str, k: usize) -> Result<Vec<(String, f64)>, IndexError> {
        self.budget
            .check("recall_global", None, None, text, k, None)?;
        if k == 0 || self.is_empty() {
            return Ok(Vec::new());
        }
        let hits = self.mem.recall_global_scored(text, k).map_err(|error| {
            self.budget
                .embedding_error("recall_global", None, None, error)
        })?;
        hits.into_iter()
            .map(|(uri, d2)| {
                self.budget
                    .check("recall_global", None, Some(&uri), "", 0, None)?;
                let value = score(d2)
                    .map_err(|kind| self.budget.error("recall_global", None, Some(&uri), kind))?;
                Ok((uri, value))
            })
            .collect()
    }

    pub fn provenance(&self) -> AdapterProvenance {
        AdapterProvenance {
            contract: ADAPTER_CONTRACT,
            embedder_type: std::any::type_name::<E>(),
            dimension: self.dimension,
        }
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
