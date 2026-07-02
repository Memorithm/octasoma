//! # Memory kernel — wiring OctaSoma into an AI agent
//!
//! [`OctaSomaAgent`](crate::OctaSomaAgent) is the raw store (perceive / recall).
//! [`MemoryKernel`] is the *opinionated routine* that turns it into a drop-in
//! long-term memory for an agent loop: it decides **what to store**, **what to
//! retrieve**, and **how to hand context to the model**, plus persistence.
//!
//! ## The cognitive loop
//!
//! ```text
//!            ┌─────────────────────────── agent turn ───────────────────────────┐
//!  user msg ─▶ kernel.step(msg)                                                  │
//!            │   1. recall_context(msg)  → "## Relevant memories\n- …"          │
//!            │   2. (optional) observe(msg)  → store the user message            │
//!            │   3. return MemoryStep { context, retrieved, stored_input }       │
//!            └───────────────┬───────────────────────────────────────────────────┘
//!                            ▼
//!     prompt = system_prompt() + step.context + msg  ──▶  LLM  ──▶ reply
//!                            │
//!     kernel.observe(reply)  (store durable facts the assistant produced)
//! ```
//!
//! See [`docs/integration-kernel.md`](https://github.com/checkupauto/octasoma/blob/master/docs/integration-kernel.md)
//! for the full wiring guide, system-prompt block, and tool schemas.

use std::io;

use crate::agent::OctaSomaAgent;
use crate::conformal::conformal_quantile;
use crate::embed::{EmbedError, Embedder};
use crate::feedback::RelevanceFeedback;

/// Policy knobs for the [`MemoryKernel`].
#[derive(Clone, Debug)]
pub struct KernelConfig {
    /// How many memories to retrieve per reflection.
    pub top_k: usize,
    /// Header line prepended to a non-empty context block.
    pub context_header: String,
    /// Bullet prefix for each recalled memory.
    pub bullet: String,
    /// Observations shorter than this (trimmed char count) are not stored.
    pub min_observation_chars: usize,
    /// Hard cap on the injected context length, in bytes.
    pub max_context_chars: usize,
    /// If set, the store is saved to this path by the autosave policy.
    pub autosave_path: Option<String>,
    /// Save after this many stored observations (`0` disables autosave).
    pub autosave_every: usize,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            context_header: "## Relevant memories".to_string(),
            bullet: "- ".to_string(),
            min_observation_chars: 8,
            max_context_chars: 2000,
            autosave_path: None,
            autosave_every: 0,
        }
    }
}

/// The outcome of a single [`MemoryKernel::step`].
#[derive(Clone, Debug, Default)]
pub struct MemoryStep {
    /// Prompt-ready context block (empty when nothing relevant was found).
    pub context: String,
    /// The raw recalled memories, nearest first.
    pub retrieved: Vec<String>,
    /// Whether the input was stored as a new memory this turn.
    pub stored_input: bool,
}

/// The outcome of a **conformal recall** ([`MemoryKernel::recall_set`]) — as
/// many memories as the coverage guarantee needs, not a fixed `k`.
#[derive(Clone, Debug)]
pub struct ConformalRecall {
    /// The returned memories with their similarity scores, nearest first.
    pub memories: Vec<(String, f32)>,
    /// The calibrated nonconformity radius `q̂` (`1 − score ≤ q̂` ⇒ included).
    /// `+∞` when the feedback log cannot support the asked `alpha` yet.
    pub radius: f64,
    /// Confirmed-relevant feedback entries the radius was calibrated on.
    pub calibration_n: usize,
    /// The asked miscoverage level.
    pub alpha: f64,
    /// Whether the coverage statement holds: *the relevant memory is in this
    /// set with probability `≥ 1 − alpha`* (for workloads exchangeable with the
    /// feedback log). `false` when the log is too small (fixed `top_k`
    /// fallback) or the candidate pool truncated the radius — never silent.
    pub guaranteed: bool,
}

/// An opinionated long-term-memory routine on top of [`OctaSomaAgent`].
pub struct MemoryKernel<E: Embedder> {
    agent: OctaSomaAgent<E>,
    config: KernelConfig,
    pending_since_save: usize,
    /// The last recall this kernel served: `(query, [(memory, score)])` — what
    /// [`MemoryKernel::feedback`] indices refer to.
    last_recall: Option<(String, Vec<(String, f32)>)>,
    /// The explicit relevance-feedback log (see [`crate::feedback`]).
    feedback: RelevanceFeedback,
}

impl<E: Embedder> MemoryKernel<E> {
    /// Wraps an existing agent with a policy.
    pub fn new(agent: OctaSomaAgent<E>, config: KernelConfig) -> Self {
        Self {
            agent,
            config,
            pending_since_save: 0,
            last_recall: None,
            feedback: RelevanceFeedback::new(),
        }
    }

    /// Convenience: a kernel with a JL-projection agent and default policy.
    pub fn with_defaults(embedder: E, seed: u64) -> Self {
        Self::new(OctaSomaAgent::new(embedder, seed), KernelConfig::default())
    }

    /// Convenience: a kernel whose projection is PCA-calibrated on `corpus`.
    pub fn calibrated(
        embedder: E,
        corpus: &[&str],
        config: KernelConfig,
    ) -> Result<Self, EmbedError> {
        Ok(Self::new(
            OctaSomaAgent::calibrate(embedder, corpus)?,
            config,
        ))
    }

    /// Read-only / mutable access to the policy.
    pub fn config(&self) -> &KernelConfig {
        &self.config
    }
    pub fn config_mut(&mut self) -> &mut KernelConfig {
        &mut self.config
    }

    /// Stores an observation **iff** it passes policy (length gate). Returns
    /// whether it was actually stored.
    pub fn observe(&mut self, text: &str) -> Result<bool, EmbedError> {
        if text.trim().chars().count() < self.config.min_observation_chars {
            return Ok(false);
        }
        self.agent.perceive(text)?;
        self.pending_since_save += 1;
        self.maybe_autosave();
        Ok(true)
    }

    /// Returns a prompt-ready context block for `query` (header + bullets,
    /// truncated to `max_context_chars`). Empty if nothing is recalled.
    pub fn recall_context(&mut self, query: &str) -> Result<String, EmbedError> {
        let scored = self.agent.recall_scored(query, self.config.top_k)?;
        let memories: Vec<String> = scored.iter().map(|(m, _)| m.clone()).collect();
        self.last_recall = Some((query.to_string(), scored));
        Ok(self.format_context(&memories))
    }

    /// Explains a recall (auditable): the query's 3-D position, the coarse→fine
    /// regions it falls through, and the nearest memories with distances.
    pub fn explain(&self, query: &str, k: usize) -> Result<Option<crate::Explanation>, EmbedError> {
        self.agent.explain(query, k)
    }

    /// One cognitive step: recall context for `input`, optionally store `input`.
    pub fn step(&mut self, input: &str, remember_input: bool) -> Result<MemoryStep, EmbedError> {
        let scored = self.agent.recall_scored(input, self.config.top_k)?;
        let retrieved: Vec<String> = scored.iter().map(|(m, _)| m.clone()).collect();
        self.last_recall = Some((input.to_string(), scored));
        let context = self.format_context(&retrieved);
        let stored_input = if remember_input {
            self.observe(input)?
        } else {
            false
        };
        Ok(MemoryStep {
            context,
            retrieved,
            stored_input,
        })
    }

    /// A ready-to-paste system-prompt fragment that tells the model how to treat
    /// the memory block this kernel injects.
    pub fn system_prompt(&self) -> String {
        format!(
            "You have a long-term semantic memory. When relevant, recalled \
memories are provided to you in a section titled \"{header}\". Use them as \
background recollections: prefer information that is consistent and recent, treat \
them as fallible (they may be partial or outdated), and never invent memories \
that are not listed. If the user states a durable fact, preference, or decision, \
assume it will be remembered for future turns. Do not mention the memory \
mechanism unless asked.",
            header = self.config.context_header
        )
    }

    /// **Conformal recall set** (proposal B2): returns *as many memories as the
    /// guarantee needs* instead of a fixed `top_k` — the set shrinks when the
    /// query matches confidently and grows when it is uncertain, which is the
    /// token-frugal behaviour the cascade story wants.
    ///
    /// The radius is the split-conformal quantile of the **explicit feedback
    /// log**'s confirmed-relevant nonconformities ([`crate::RelevanceFeedback`]):
    /// *the relevant memory is in the returned set with probability
    /// `≥ 1 − alpha`*, distribution-free, for workloads exchangeable with the
    /// recorded feedback. Honesty rules, all visible in [`ConformalRecall`]:
    ///
    /// - Too little feedback for `alpha` → radius `+∞` → fixed-`top_k`
    ///   fallback with `guaranteed = false` — never a fake guarantee.
    /// - If every candidate in the pool fits the radius, the pool itself may
    ///   have truncated the set → `guaranteed = false` too.
    /// - Feedback from self-retrieval calibration would overstate coverage —
    ///   record real agent feedback (see the module docs).
    pub fn recall_set(&mut self, query: &str, alpha: f64) -> Result<ConformalRecall, EmbedError> {
        let nonconformity: Vec<f64> = self
            .feedback
            .nonconformity()
            .into_iter()
            .map(f64::from)
            .collect();
        let radius = conformal_quantile(&nonconformity, alpha);
        // A pool wider than top_k so an uncertain query can grow its set.
        let pool = (self.config.top_k * 4).max(16);
        let scored = self.agent.recall_scored(query, pool)?;
        self.last_recall = Some((query.to_string(), scored.clone()));

        let (memories, guaranteed) = if radius.is_finite() {
            let kept: Vec<(String, f32)> = scored
                .iter()
                .filter(|(_, score)| (1.0 - *score as f64) <= radius)
                .cloned()
                .collect();
            // If the radius did not bind (every candidate fits), the pool may
            // have cut the true set — say so.
            let truncated = kept.len() == scored.len() && !scored.is_empty();
            (kept, !truncated)
        } else {
            (scored.into_iter().take(self.config.top_k).collect(), false)
        };

        Ok(ConformalRecall {
            memories,
            radius,
            calibration_n: nonconformity.len(),
            alpha,
            guaranteed,
        })
    }

    /// **Relevance feedback** on the last recall this kernel served (via
    /// [`MemoryKernel::step`] or [`MemoryKernel::recall_context`]): the indices
    /// refer to the order of [`MemoryStep::retrieved`] / the context bullets.
    /// Out-of-range indices are ignored; with no prior recall this is a no-op.
    /// Returns how many observations were recorded.
    ///
    /// This is the explicit channel the calibrated tiers consume (see
    /// [`crate::feedback`]): wire the `memory_feedback` tool of
    /// [`MEMORY_TOOL_SCHEMA_JSON`] here.
    pub fn feedback(&mut self, relevant: &[usize], irrelevant: &[usize]) -> usize {
        let Some((query, scored)) = &self.last_recall else {
            return 0;
        };
        let mut recorded = 0;
        for (indices, label) in [(relevant, true), (irrelevant, false)] {
            for &i in indices {
                if let Some((memory, score)) = scored.get(i) {
                    self.feedback.record(query, memory, *score, label);
                    recorded += 1;
                }
            }
        }
        recorded
    }

    /// Read access to the relevance-feedback log — the calibration input for
    /// the conformal (B2) and temperature (B3) tiers.
    pub fn feedback_log(&self) -> &RelevanceFeedback {
        &self.feedback
    }

    /// Forces a save to `autosave_path` (if configured) and resets the counter.
    pub fn save(&mut self) -> io::Result<()> {
        if let Some(path) = self.config.autosave_path.clone() {
            self.agent.save(&path)?;
            self.pending_since_save = 0;
        }
        Ok(())
    }

    /// Saves to an explicit path regardless of policy.
    pub fn save_to(&mut self, path: &str) -> io::Result<()> {
        self.agent.save(path)?;
        self.pending_since_save = 0;
        Ok(())
    }

    /// Number of stored memories.
    pub fn len(&self) -> usize {
        self.agent.len()
    }
    /// Whether no memories are stored yet.
    pub fn is_empty(&self) -> bool {
        self.agent.is_empty()
    }
    /// Borrow the wrapped agent (advanced access).
    pub fn agent(&self) -> &OctaSomaAgent<E> {
        &self.agent
    }

    // -- internals ----------------------------------------------------------

    fn format_context(&self, memories: &[String]) -> String {
        if memories.is_empty() {
            return String::new();
        }
        let mut out = String::with_capacity(self.config.max_context_chars.min(4096));
        out.push_str(&self.config.context_header);
        out.push('\n');
        for m in memories {
            out.push_str(&self.config.bullet);
            out.push_str(&m.replace('\n', " "));
            out.push('\n');
            if out.len() >= self.config.max_context_chars {
                break;
            }
        }
        if out.len() > self.config.max_context_chars {
            out.truncate(self.config.max_context_chars);
        }
        out
    }

    fn maybe_autosave(&mut self) {
        if self.config.autosave_every == 0 || self.pending_since_save < self.config.autosave_every {
            return;
        }
        if let Some(path) = self.config.autosave_path.clone() {
            let _ = self.agent.save(&path); // best-effort; never fails a turn
            self.pending_since_save = 0;
        }
    }
}

/// JSON definitions for the three tools an LLM agent should expose to drive this
/// memory (compatible with OpenAI/Anthropic-style function calling). Wire the
/// `memory_store` tool to [`MemoryKernel::observe`], `memory_recall` to
/// [`MemoryKernel::recall_context`], and `memory_feedback` to
/// [`MemoryKernel::feedback`] (the explicit relevance channel that powers the
/// calibrated tiers).
pub const MEMORY_TOOL_SCHEMA_JSON: &str = r#"[
  {
    "name": "memory_store",
    "description": "Persist a durable fact, preference, decision, or observation to long-term memory. Call this whenever the user states something worth remembering across sessions.",
    "input_schema": {
      "type": "object",
      "properties": {
        "text": { "type": "string", "description": "The information to remember, as a self-contained sentence." }
      },
      "required": ["text"]
    }
  },
  {
    "name": "memory_recall",
    "description": "Retrieve memories relevant to a query before answering. Returns up to top_k past memories, nearest first.",
    "input_schema": {
      "type": "object",
      "properties": {
        "query": { "type": "string", "description": "What to look up in memory." },
        "top_k": { "type": "integer", "description": "Maximum memories to return.", "default": 5 }
      },
      "required": ["query"]
    }
  },
  {
    "name": "memory_explain",
    "description": "Explain why a query recalls what it does: the query's 3-D position, the nested regions it falls through (coarse→fine), and the nearest memories with distances. Use to audit or justify a recall.",
    "input_schema": {
      "type": "object",
      "properties": {
        "query": { "type": "string", "description": "The query to explain." },
        "top_k": { "type": "integer", "description": "Nearest memories to include.", "default": 5 }
      },
      "required": ["query"]
    }
  },
  {
    "name": "memory_feedback",
    "description": "After using recalled memories, report which of them were actually relevant to the task. Indices refer to the order of the last memory_recall result (0-based). This feedback calibrates the memory's confidence guarantees — call it whenever you can tell a recalled memory clearly helped or clearly did not.",
    "input_schema": {
      "type": "object",
      "properties": {
        "relevant_indices": { "type": "array", "items": { "type": "integer" }, "description": "Positions of the recalled memories that were useful." },
        "irrelevant_indices": { "type": "array", "items": { "type": "integer" }, "description": "Positions of the recalled memories that were not useful." }
      }
    }
  }
]"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HashEmbedder;

    fn kernel() -> MemoryKernel<HashEmbedder> {
        MemoryKernel::with_defaults(HashEmbedder::new(128), 42)
    }

    #[test]
    fn observe_respects_min_length() {
        let mut k = kernel();
        assert!(!k.observe("short").unwrap()); // below 8 chars → skipped
        assert!(k.observe("a sufficiently long observation").unwrap());
        assert_eq!(k.len(), 1);
    }

    #[test]
    fn step_recalls_and_optionally_stores() {
        let mut k = kernel();
        k.observe("the user prefers dark mode interfaces").unwrap();
        let step = k
            .step("the user prefers dark mode interfaces", false)
            .unwrap();
        assert!(step.context.contains("Relevant memories"));
        assert!(step.context.contains("dark mode"));
        assert!(!step.stored_input);
        assert_eq!(k.len(), 1);

        let step2 = k
            .step("a brand new durable observation here", true)
            .unwrap();
        assert!(step2.stored_input);
        assert_eq!(k.len(), 2);
    }

    #[test]
    fn empty_recall_yields_empty_context() {
        let mut k = kernel();
        assert_eq!(k.recall_context("anything").unwrap(), "");
    }

    #[test]
    fn context_is_truncated() {
        let mut k = kernel();
        k.config_mut().max_context_chars = 40;
        for i in 0..10 {
            k.observe(&format!("memory number {i} with some padding text"))
                .unwrap();
        }
        let ctx = k
            .step("memory number 3 with some padding text", false)
            .unwrap()
            .context;
        assert!(ctx.len() <= 40);
    }

    #[test]
    fn tool_schema_is_present() {
        assert!(MEMORY_TOOL_SCHEMA_JSON.contains("memory_feedback"));
        assert!(MEMORY_TOOL_SCHEMA_JSON.contains("memory_store"));
        assert!(MEMORY_TOOL_SCHEMA_JSON.contains("memory_recall"));
    }

    #[test]
    fn system_prompt_mentions_header() {
        let k = kernel();
        assert!(k.system_prompt().contains("Relevant memories"));
    }

    #[test]
    fn conformal_recall_set_grows_shrinks_and_never_fakes_the_guarantee() {
        use crate::HashEmbedder;
        let mut k = MemoryKernel::with_defaults(HashEmbedder::new(64), 1);
        for i in 0..30 {
            k.observe(&format!(
                "durable fact number {i} about subsystem {}",
                i % 5
            ))
            .unwrap();
        }

        // No feedback yet → +∞ radius → top_k fallback, explicitly unguaranteed.
        let cold = k
            .recall_set("durable fact number 3 about subsystem 3", 0.2)
            .unwrap();
        assert!(!cold.guaranteed);
        assert!(cold.radius.is_infinite());
        assert_eq!(cold.memories.len(), k.config().top_k);

        // Record real feedback: exact-text recalls are relevant at score 1.0
        // (nonconformity 0), so the calibrated radius is tight.
        for i in 0..6 {
            let q = format!("durable fact number {i} about subsystem {}", i % 5);
            k.step(&q, false).unwrap();
            assert_eq!(k.feedback(&[0], &[]), 1);
        }
        let warm = k
            .recall_set("durable fact number 4 about subsystem 4", 0.2)
            .unwrap();
        assert!(warm.guaranteed, "radius = {}", warm.radius);
        assert_eq!(warm.calibration_n, 6);
        assert!(warm.radius < 1e-6, "tight radius, got {}", warm.radius);
        // The set collapsed to exactly the confident match — token-frugal.
        assert_eq!(warm.memories.len(), 1);
        assert!(warm.memories[0].0.contains("number 4"));
        // The guarantee holds on this exchangeable query: the relevant memory
        // is in the set.
        assert!((warm.memories[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn feedback_records_against_the_last_recall() {
        use crate::HashEmbedder;
        let mut k = MemoryKernel::with_defaults(HashEmbedder::new(64), 1);
        // No prior recall → no-op.
        assert_eq!(k.feedback(&[0], &[]), 0);

        k.observe("the database timeout is thirty seconds").unwrap();
        k.observe("the cache eviction policy is LRU").unwrap();
        let step = k
            .step("the database timeout is thirty seconds", false)
            .unwrap();
        assert!(!step.retrieved.is_empty());

        // Label index 0 relevant, out-of-range ignored.
        let n = k.feedback(&[0, 99], &[]);
        assert_eq!(n, 1);
        let log = k.feedback_log();
        assert_eq!(log.len(), 1);
        let e = &log.entries()[0];
        assert!(e.relevant);
        assert_eq!(e.query, "the database timeout is thirty seconds");
        assert_eq!(e.memory, step.retrieved[0]);
        // Exact-text recall (HashEmbedder): the self-item scores 1.0.
        assert!((e.score - 1.0).abs() < 1e-6, "score = {}", e.score);
        // Nonconformity view feeds the future conformal tier.
        assert_eq!(log.nonconformity().len(), 1);
    }
}
