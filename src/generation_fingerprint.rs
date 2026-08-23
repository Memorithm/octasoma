//! Interpretation identity bound to strict persisted generations.
//!
//! Component hashes prove that a generation is internally intact. They do not
//! prove that callers are reopening those bytes with the same embedding model,
//! projection, quantisation, index contract, calibration state, or reviewed
//! SciRust foundation. [`GenerationFingerprint`] supplies that second binding.

use std::io;

/// Reviewed SciRust revision that defines the numerical/retrieval foundation of
/// the current OctaSoma v0.5 line.
pub const SCIRUST_REVISION: &str = "9b3d9492bb20e097231598a731df689ad4bd4bcc";

pub(crate) const MAX_FINGERPRINT_BYTES: usize = 1024;

/// Deterministic identity of the interpretation required to reopen a persisted
/// generation safely.
///
/// The fields are intentionally opaque strings: callers can use model digests,
/// projection descriptors, quantisation identifiers, index-policy hashes, and
/// calibration artifact hashes without OctaSoma depending on those systems.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationFingerprint {
    /// Embedding provider/model/revision identity.
    pub embedding: String,
    /// Projection identity (algorithm, seed or learned projection digest).
    pub projection: String,
    /// Stored-vector precision / quantisation contract.
    pub quantization: String,
    /// Precision-index configuration identity.
    pub index: String,
    /// Reviewed SciRust revision defining the numerical foundation.
    pub scirust_revision: String,
    /// Optional calibration/certificate artifact identity.
    pub calibration: Option<String>,
}

impl GenerationFingerprint {
    /// Builds a fingerprint bound to OctaSoma's currently reviewed SciRust
    /// revision. Callers still provide identities for every interpretation layer
    /// they own.
    pub fn canonical(
        embedding: impl Into<String>,
        projection: impl Into<String>,
        quantization: impl Into<String>,
        index: impl Into<String>,
    ) -> Self {
        Self {
            embedding: embedding.into(),
            projection: projection.into(),
            quantization: quantization.into(),
            index: index.into(),
            scirust_revision: SCIRUST_REVISION.to_string(),
            calibration: None,
        }
    }

    /// Binds a calibration/certificate artifact to this interpretation.
    pub fn with_calibration(mut self, calibration: impl Into<String>) -> Self {
        self.calibration = Some(calibration.into());
        self
    }

    pub(crate) fn validate(&self) -> io::Result<()> {
        for (name, value) in [
            ("embedding", self.embedding.as_str()),
            ("projection", self.projection.as_str()),
            ("quantization", self.quantization.as_str()),
            ("index", self.index.as_str()),
            ("scirust_revision", self.scirust_revision.as_str()),
        ] {
            validate_value(name, value)?;
        }
        if let Some(calibration) = &self.calibration {
            validate_value("calibration", calibration)?;
        }
        Ok(())
    }
}

fn validate_value(name: &str, value: &str) -> io::Result<()> {
    if value.is_empty() {
        return Err(invalid(format!("generation fingerprint {name} is empty")));
    }
    if value.len() > MAX_FINGERPRINT_BYTES {
        return Err(invalid(format!(
            "generation fingerprint {name} exceeds {MAX_FINGERPRINT_BYTES} bytes"
        )));
    }
    if !value.bytes().all(|b| (0x20..=0x7e).contains(&b)) {
        return Err(invalid(format!(
            "generation fingerprint {name} must contain printable ASCII only"
        )));
    }
    Ok(())
}

fn invalid(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_fingerprint_is_bound_to_reviewed_scirust_revision() {
        let fingerprint =
            GenerationFingerprint::canonical("embed:v1", "jl:42", "f32", "simhash:256")
                .with_calibration("rcps:sha256:abc");
        fingerprint.validate().unwrap();
        assert_eq!(fingerprint.scirust_revision, SCIRUST_REVISION);
    }

    #[test]
    fn multiline_and_empty_fields_are_rejected() {
        let mut fingerprint =
            GenerationFingerprint::canonical("embed:v1", "jl:42", "f32", "simhash:256");
        fingerprint.embedding.clear();
        assert!(fingerprint.validate().is_err());
        fingerprint.embedding = "model\nforged=1".into();
        assert!(fingerprint.validate().is_err());
    }

    /// Single-source-of-truth tripwire: every SciRust git dependency pinned in
    /// Cargo.toml must target exactly [`SCIRUST_REVISION`] — that constant is
    /// what persisted generation fingerprints bind to, so a silent divergence
    /// would let stores be reopened under a different numerical foundation.
    #[test]
    fn cargo_toml_pins_match_scirust_revision() {
        let manifest = include_str!("../Cargo.toml");
        let mut found = 0;
        for line in manifest.lines() {
            let Some(idx) = line.find("rev = \"") else {
                continue;
            };
            let rest = &line[idx + "rev = \"".len()..];
            let end = rest.find('"').expect("terminated rev string literal");
            found += 1;
            assert_eq!(
                &rest[..end],
                SCIRUST_REVISION,
                "Cargo.toml pins a SciRust revision that differs from SCIRUST_REVISION"
            );
        }
        // Runtime retrieval + SIMD + two dev-deps: at least four pins today.
        assert!(
            found >= 4,
            "expected >=4 SciRust pins in Cargo.toml, found {found}"
        );
    }
}
