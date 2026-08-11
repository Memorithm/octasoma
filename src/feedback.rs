//! Explicit relevance feedback — the channel that unblocks the calibrated tiers.
//!
//! Interactive relevance feedback and independent recall ground truth are
//! deliberately different evidence classes.
//!
//! [`MemoryKernel::feedback`](crate::MemoryKernel::feedback), the
//! `memory_feedback` function-call entry and the MCP `feedback` tool label
//! **already-returned candidates**. That evidence is useful for temperature
//! calibration and adaptive ranking, but it is selection-biased: a relevant
//! memory that retrieval missed cannot be labelled through that channel.
//!
//! Distribution-free recall coverage therefore uses only entries explicitly
//! marked [`FeedbackSource::ExternalGroundTruth`]. Those targets must be chosen
//! independently of the candidate set (for example from a CCOS event log,
//! held-out evaluator or authorised benchmark) and scored against the store.
//! Ordinary interactive feedback can never make `ConformalRecall::guaranteed`
//! true by itself.
//!
//! The log is in-memory, per session, and deterministic (entries in arrival
//! order). Persistence-with-the-store is a deliberate non-goal for now:
//! feedback describes a *workload*, not the corpus, and stale labels silently
//! void the very guarantees they exist to support.

/// Provenance class of a relevance label.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeedbackSource {
    /// Label attached to a candidate that the retriever already returned.
    RetrievedCandidate,
    /// Target selected independently of the retriever candidate set.
    ExternalGroundTruth,
}

/// One relevance observation for `query` and `memory` at `score`.
#[derive(Clone, Debug, PartialEq)]
pub struct FeedbackEntry {
    /// The recall query text.
    pub query: String,
    /// The recalled memory text (or node URI in sharded/MCP deployments).
    pub memory: String,
    /// The similarity score the store reported at recall time, in `(0, 1]`.
    pub score: f32,
    /// The agent/evaluator verdict: was this memory relevant for the query?
    pub relevant: bool,
    /// Whether the label was conditioned on retrieval or independently sourced.
    pub source: FeedbackSource,
}

/// An append-only, in-memory log of [`FeedbackEntry`] — the calibration input
/// for the conformal (B2) and temperature (B3) tiers.
#[derive(Clone, Debug, Default)]
pub struct RelevanceFeedback {
    entries: Vec<FeedbackEntry>,
}

impl RelevanceFeedback {
    /// An empty log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends one interactive label for an already-returned candidate.
    /// This evidence never calibrates a recall-coverage guarantee.
    pub fn record(&mut self, query: &str, memory: &str, score: f32, relevant: bool) {
        self.record_with_source(
            query,
            memory,
            score,
            relevant,
            FeedbackSource::RetrievedCandidate,
        );
    }

    /// Appends an independently selected ground-truth target. The caller is
    /// responsible for ensuring target selection did not depend on the
    /// retriever candidate set.
    pub fn record_ground_truth(&mut self, query: &str, memory: &str, score: f32, relevant: bool) {
        self.record_with_source(
            query,
            memory,
            score,
            relevant,
            FeedbackSource::ExternalGroundTruth,
        );
    }

    fn record_with_source(
        &mut self,
        query: &str,
        memory: &str,
        score: f32,
        relevant: bool,
        source: FeedbackSource,
    ) {
        self.entries.push(FeedbackEntry {
            query: query.to_string(),
            memory: memory.to_string(),
            score,
            relevant,
            source,
        });
    }

    /// All observations, in arrival order.
    pub fn entries(&self) -> &[FeedbackEntry] {
        &self.entries
    }

    /// Number of observations.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing has been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// How many observations are positive.
    pub fn relevant_count(&self) -> usize {
        self.entries.iter().filter(|e| e.relevant).count()
    }

    /// `(score, label)` pairs — the input a temperature fit (B3) consumes.
    pub fn score_labels(&self) -> Vec<(f32, bool)> {
        self.entries.iter().map(|e| (e.score, e.relevant)).collect()
    }

    /// Fits a confidence temperature on this log's `(score, label)` pairs (see
    /// [`crate::calibration`]) — `None` while the log cannot identify one
    /// (fewer than 2 entries or a single class). Feed recall scores through
    /// [`crate::calibrated_probability`] with the result to compare recalls
    /// across stores/shards as probabilities instead of raw cosines.
    pub fn fit_temperature(&self) -> Option<f32> {
        crate::calibration::fit_temperature(&self.score_labels())
    }

    /// Nonconformity scores (`1 − score`) of confirmed-relevant targets
    /// selected **independently** of retrieval. Candidate-conditioned feedback
    /// is excluded so completely missed relevant memories cannot disappear from
    /// the calibration protocol.
    pub fn nonconformity(&self) -> Vec<f32> {
        self.entries
            .iter()
            .filter(|e| e.relevant && e.source == FeedbackSource::ExternalGroundTruth)
            .map(|e| 1.0 - e.score)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_and_projects_calibration_views() {
        let mut log = RelevanceFeedback::new();
        assert!(log.is_empty());
        log.record("q1", "m1", 0.9, true);
        log.record("q1", "m2", 0.7, false);
        log.record_ground_truth("q2", "m3", 0.8, true);

        assert_eq!(log.len(), 3);
        assert_eq!(log.relevant_count(), 2);
        assert_eq!(
            log.score_labels(),
            vec![(0.9, true), (0.7, false), (0.8, true)]
        );
        assert_eq!(log.entries()[0].source, FeedbackSource::RetrievedCandidate);
        assert_eq!(log.entries()[2].source, FeedbackSource::ExternalGroundTruth);
        // Only independent confirmed-relevant targets calibrate recall coverage.
        let nc = log.nonconformity();
        assert_eq!(nc.len(), 1);
        assert!((nc[0] - 0.2).abs() < 1e-6);
    }
}
