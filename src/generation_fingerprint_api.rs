//! Strict interpretation-bound persistence methods for [`HybridMemory`](crate::HybridMemory).

use std::io;

use crate::{GenerationFingerprint, HybridMemory};

impl HybridMemory {
    /// Persists a new immutable generation with an explicit interpretation
    /// fingerprint. Governed stores should prefer this over [`HybridMemory::save_dir`]
    /// so embedder/projection/quantisation/index/calibration/SciRust drift is
    /// detectable before persisted bytes are interpreted.
    pub fn save_dir_with_fingerprint(
        &self,
        dir: &str,
        fingerprint: &GenerationFingerprint,
    ) -> io::Result<()> {
        crate::generation_store::save_with_fingerprint(self, dir, fingerprint)
    }

    /// Opens a persisted generation only when its interpretation fingerprint
    /// exactly matches `expected`. Legacy and unbound generations are rejected;
    /// this API never silently downgrades to integrity-only loading.
    pub fn open_dir_with_fingerprint(
        dir: &str,
        dim: usize,
        expected: &GenerationFingerprint,
    ) -> io::Result<Self> {
        crate::generation_store::open_with_fingerprint(dir, dim, expected)
    }

    /// Alias for [`HybridMemory::save_dir_with_fingerprint`] using the concise
    /// bound-generation terminology introduced by the v0.5 persistence contract.
    pub fn save_dir_bound(
        &self,
        dir: &str,
        fingerprint: &GenerationFingerprint,
    ) -> io::Result<()> {
        self.save_dir_with_fingerprint(dir, fingerprint)
    }

    /// Alias for [`HybridMemory::open_dir_with_fingerprint`] using the concise
    /// bound-generation terminology introduced by the v0.5 persistence contract.
    pub fn open_dir_bound(
        dir: &str,
        dim: usize,
        expected: &GenerationFingerprint,
    ) -> io::Result<Self> {
        Self::open_dir_with_fingerprint(dir, dim, expected)
    }
}
