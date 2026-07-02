//! Explicit relevance feedback — the channel that unblocks the calibrated tiers.
//!
//! Proposals B2 (conformal recall sets) and B3 (per-shard temperature scaling)
//! in `docs/scirust-improvements.md` both need to know, after a recall, *which
//! returned memories were actually the right ones*. Calibrating on
//! self-retrieval (query == stored text) overstates every guarantee — the
//! documented pitfall — so the labels must come from the agent loop itself.
//!
//! This module is the **explicit** channel (the design decision on record): the
//! host agent reports relevance after using a recall — via
//! [`MemoryKernel::feedback`](crate::MemoryKernel::feedback) in-process, the
//! `memory_feedback` entry of
//! [`MEMORY_TOOL_SCHEMA_JSON`](crate::MEMORY_TOOL_SCHEMA_JSON) for
//! function-calling LLMs, or the `feedback` tool of `octasoma-mcp`. It is the
//! same shape CCOS's premium `ImprovementLoop` (`Feature::AdaptiveRetrieval`)
//! consumes — one channel feeds every calibrated capability in the stack.
//!
//! The log is in-memory, per session, and deterministic (entries in arrival
//! order). Persistence-with-the-store is a deliberate non-goal for now:
//! feedback describes a *workload*, not the corpus, and stale labels silently
//! void the very guarantees they exist to support.

/// One relevance observation: after a recall for `query`, the returned `memory`
/// (with its similarity `score` at recall time) was — or was not — relevant.
#[derive(Clone, Debug, PartialEq)]
pub struct FeedbackEntry {
    /// The recall query text.
    pub query: String,
    /// The recalled memory text (or node URI in sharded/MCP deployments).
    pub memory: String,
    /// The similarity score the store reported at recall time, in `(0, 1]`.
    pub score: f32,
    /// The agent's verdict: was this memory actually useful for the query?
    pub relevant: bool,
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

    /// Appends one observation.
    pub fn record(&mut self, query: &str, memory: &str, score: f32, relevant: bool) {
        self.entries.push(FeedbackEntry {
            query: query.to_string(),
            memory: memory.to_string(),
            score,
            relevant,
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

    /// Nonconformity scores (`1 − score`) of the **confirmed-relevant** recalls —
    /// the calibration set a conformal quantile (B2) consumes: "how dissimilar
    /// can the right memory look?".
    pub fn nonconformity(&self) -> Vec<f32> {
        self.entries
            .iter()
            .filter(|e| e.relevant)
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
        log.record("q2", "m3", 0.8, true);

        assert_eq!(log.len(), 3);
        assert_eq!(log.relevant_count(), 2);
        assert_eq!(
            log.score_labels(),
            vec![(0.9, true), (0.7, false), (0.8, true)]
        );
        // Only confirmed-relevant recalls calibrate the conformal radius.
        let nc = log.nonconformity();
        assert_eq!(nc.len(), 2);
        assert!((nc[0] - 0.1).abs() < 1e-6 && (nc[1] - 0.2).abs() < 1e-6);
    }
}
