from pathlib import Path

p = Path("src/kernel.rs")
s = p.read_text()

anchor = """pub struct MemoryStep {
    /// Prompt-ready context block (empty when nothing relevant was found).
    pub context: String,
    /// The raw recalled memories, nearest first.
    pub retrieved: Vec<String>,
    /// Whether the input was stored as a new memory this turn.
    pub stored_input: bool,
}
"""
addition = anchor + """
/// A persistence failure captured by the best-effort autosave path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AutosaveFailure {
    /// Stable I/O error category.
    pub kind: io::ErrorKind,
    /// Human-readable error returned by the storage backend.
    pub message: String,
}

/// Whether observations currently have a durable backing store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DurabilityStatus {
    /// No autosave path is configured; durability is owned by the caller.
    Disabled,
    /// Every observation known to the kernel has been persisted successfully.
    Clean,
    /// Observations have changed since the last successful save.
    Pending {
        /// Number of stored observations since the last successful save.
        observations: usize,
    },
    /// The last autosave attempt failed. Pending observations are deliberately
    /// retained and will be retried by the normal autosave threshold.
    Error {
        /// Number of observations still awaiting a successful save.
        observations: usize,
        /// The last storage failure.
        failure: AutosaveFailure,
    },
}
"""
assert s.count(anchor) == 1
s = s.replace(anchor, addition, 1)

old = """    pending_since_save: usize,
    /// The last recall this kernel served: `(query, [(memory, score)])` — what
"""
new = """    pending_since_save: usize,
    last_autosave_failure: Option<AutosaveFailure>,
    /// The last recall this kernel served: `(query, [(memory, score)])` — what
"""
assert s.count(old) == 1
s = s.replace(old, new, 1)

old = """            pending_since_save: 0,
            last_recall: None,
"""
new = """            pending_since_save: 0,
            last_autosave_failure: None,
            last_recall: None,
"""
assert s.count(old) == 1
s = s.replace(old, new, 1)

old = """    /// Forces a save to `autosave_path` (if configured) and resets the counter.
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
"""
new = """    /// Forces a save to `autosave_path` (if configured) and resets the counter
    /// only after the storage backend confirms success.
    pub fn save(&mut self) -> io::Result<()> {
        if let Some(path) = self.config.autosave_path.clone() {
            if let Err(error) = self.agent.save(&path) {
                self.record_autosave_failure(&error);
                return Err(error);
            }
            self.pending_since_save = 0;
            self.last_autosave_failure = None;
        }
        Ok(())
    }

    /// Saves to an explicit path regardless of policy.
    pub fn save_to(&mut self, path: &str) -> io::Result<()> {
        if let Err(error) = self.agent.save(path) {
            self.record_autosave_failure(&error);
            return Err(error);
        }
        self.pending_since_save = 0;
        self.last_autosave_failure = None;
        Ok(())
    }

    /// Current persistence state. Autosave remains best-effort for agent turns,
    /// but a failed write is never silently acknowledged as durable.
    pub fn durability_status(&self) -> DurabilityStatus {
        if let Some(failure) = self.last_autosave_failure.clone() {
            return DurabilityStatus::Error {
                observations: self.pending_since_save,
                failure,
            };
        }
        if self.config.autosave_path.is_none() {
            return DurabilityStatus::Disabled;
        }
        if self.pending_since_save == 0 {
            DurabilityStatus::Clean
        } else {
            DurabilityStatus::Pending {
                observations: self.pending_since_save,
            }
        }
    }
"""
assert s.count(old) == 1
s = s.replace(old, new, 1)

old = """    fn maybe_autosave(&mut self) {
        if self.config.autosave_every == 0 || self.pending_since_save < self.config.autosave_every {
            return;
        }
        if let Some(path) = self.config.autosave_path.clone() {
            let _ = self.agent.save(&path); // best-effort; never fails a turn
            self.pending_since_save = 0;
        }
    }
"""
new = """    fn maybe_autosave(&mut self) {
        if self.config.autosave_every == 0 || self.pending_since_save < self.config.autosave_every {
            return;
        }
        if let Some(path) = self.config.autosave_path.clone() {
            match self.agent.save(&path) {
                Ok(()) => {
                    self.pending_since_save = 0;
                    self.last_autosave_failure = None;
                }
                Err(error) => self.record_autosave_failure(&error),
            }
        }
    }

    fn record_autosave_failure(&mut self, error: &io::Error) {
        self.last_autosave_failure = Some(AutosaveFailure {
            kind: error.kind(),
            message: error.to_string(),
        });
    }
"""
assert s.count(old) == 1
s = s.replace(old, new, 1)
p.write_text(s)

p = Path("src/lib.rs")
s = p.read_text()
old = """pub use kernel::{
    ConformalRecall, KernelConfig, MEMORY_TOOL_SCHEMA_JSON, MemoryKernel, MemoryStep,
};
"""
new = """pub use kernel::{
    AutosaveFailure, ConformalRecall, DurabilityStatus, KernelConfig, MEMORY_TOOL_SCHEMA_JSON,
    MemoryKernel, MemoryStep,
};
"""
assert s.count(old) == 1
p.write_text(s.replace(old, new, 1))

p = Path("tests/agent_kernel.rs")
s = p.read_text()
old = """    Embedder, HashEmbedder, KernelConfig, MEMORY_TOOL_SCHEMA_JSON, MemoryKernel, OctaSomaAgent,
    OllamaEmbedder,
"""
new = """    DurabilityStatus, Embedder, HashEmbedder, KernelConfig, MEMORY_TOOL_SCHEMA_JSON, MemoryKernel,
    OctaSomaAgent, OllamaEmbedder,
"""
assert s.count(old) == 1
s = s.replace(old, new, 1)
marker = """#[test]
fn kernel_exposes_system_prompt_and_tools() {
"""
test = """#[test]
fn kernel_autosave_failure_remains_pending_and_observable() {
    let root = std::env::temp_dir().join(format!(
        "octasoma_kernel_autosave_failure_{}",
        std::process::id()
    ));
    std::fs::remove_dir_all(&root).ok();
    let path = root.join("store.frac");
    let cfg = KernelConfig {
        autosave_path: Some(path.to_string_lossy().into_owned()),
        autosave_every: 1,
        min_observation_chars: 1,
        ..KernelConfig::default()
    };
    let mut k = MemoryKernel::new(OctaSomaAgent::new(HashEmbedder::new(64), 0), cfg);

    assert!(k.observe("this observation must become durable").unwrap());
    match k.durability_status() {
        DurabilityStatus::Error {
            observations,
            failure,
        } => {
            assert_eq!(observations, 1);
            assert!(!failure.message.is_empty());
        }
        other => panic!("expected observable autosave error, got {other:?}"),
    }

    std::fs::create_dir_all(&root).unwrap();
    k.save().unwrap();
    assert_eq!(k.durability_status(), DurabilityStatus::Clean);
    assert!(path.exists());
    std::fs::remove_dir_all(&root).ok();
}

"""
assert s.count(marker) == 1
p.write_text(s.replace(marker, test + marker, 1))
