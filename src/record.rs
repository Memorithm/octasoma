//! Stable logical memory records for OctaSoma v0.5.
//!
//! Physical indexes may be rebuilt, quantized, sharded or compacted without
//! changing a record's logical identity. Product adapters supply policy-bearing
//! scope/provenance values; OctaSoma owns only storage/lifecycle primitives.

use std::fmt;

/// Stable logical memory identifier, independent of physical index position.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemoryId(String);

impl MemoryId {
    pub fn new(value: impl Into<String>) -> Result<Self, RecordError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(RecordError::EmptyId);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for MemoryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Product-supplied isolation scope. OctaSoma treats values as exact labels;
/// authorization remains the consumer's responsibility.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MemoryScope {
    pub tenant: Option<String>,
    pub workspace: Option<String>,
    pub agent: Option<String>,
}

/// Provenance carried with a memory observation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Provenance {
    pub source: String,
    pub source_record: Option<String>,
    /// Supplied externally. The record layer never reads a wall clock.
    pub observed_at_unix_ms: Option<u64>,
}

/// Data-sensitivity classification carried through storage and recall.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum Sensitivity {
    Public,
    #[default]
    Internal,
    Confidential,
    Restricted,
}

/// Fingerprint of the embedding/projection contract used to index a record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmbeddingFingerprint {
    pub provider: String,
    pub model: String,
    pub revision: Option<String>,
    pub dimension: usize,
    pub projection: Option<String>,
    pub quantization: Option<String>,
}

impl EmbeddingFingerprint {
    pub fn new(
        provider: impl Into<String>,
        model: impl Into<String>,
        dimension: usize,
    ) -> Result<Self, RecordError> {
        if dimension == 0 {
            return Err(RecordError::ZeroEmbeddingDimension);
        }
        Ok(Self {
            provider: provider.into(),
            model: model.into(),
            revision: None,
            dimension,
            projection: None,
            quantization: None,
        })
    }
}

/// Logical lifecycle state. Tombstones and supersession survive index rebuilds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryStatus {
    Active,
    Tombstoned { at_generation: u64 },
    Superseded {
        by: MemoryId,
        at_generation: u64,
    },
}

impl MemoryStatus {
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Active)
    }
}

/// TTL and minimum-retention controls supplied by the governing product.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Retention {
    pub expires_at_unix_ms: Option<u64>,
    pub retain_until_unix_ms: Option<u64>,
}

impl Retention {
    pub fn is_expired_at(&self, now_unix_ms: u64) -> bool {
        self.expires_at_unix_ms
            .is_some_and(|deadline| now_unix_ms >= deadline)
    }

    pub fn permits_purge_at(&self, now_unix_ms: u64) -> bool {
        self.retain_until_unix_ms
            .is_none_or(|deadline| now_unix_ms >= deadline)
    }
}

/// Evidence relation between stable records. Consumers decide its authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelationKind {
    Confirms,
    Contradicts,
    Supersedes,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryRelation {
    pub kind: RelationKind,
    pub target: MemoryId,
}

/// Canonical v0.5 logical memory record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryRecord {
    pub id: MemoryId,
    pub payload: Vec<u8>,
    pub scope: MemoryScope,
    pub provenance: Vec<Provenance>,
    /// Supplied externally; `None` is valid for deterministic/offline data.
    pub created_at_unix_ms: Option<u64>,
    /// Monotonic store generation that last changed this logical record.
    pub generation: u64,
    /// Routing hint only; never causal truth.
    pub causal_region: Option<String>,
    pub sensitivity: Sensitivity,
    pub status: MemoryStatus,
    pub retention: Retention,
    pub embedding: Option<EmbeddingFingerprint>,
    pub relations: Vec<MemoryRelation>,
}

impl MemoryRecord {
    pub fn new(id: MemoryId, payload: impl Into<Vec<u8>>, generation: u64) -> Self {
        Self {
            id,
            payload: payload.into(),
            scope: MemoryScope::default(),
            provenance: Vec::new(),
            created_at_unix_ms: None,
            generation,
            causal_region: None,
            sensitivity: Sensitivity::default(),
            status: MemoryStatus::Active,
            retention: Retention::default(),
            embedding: None,
            relations: Vec::new(),
        }
    }

    /// Whether ordinary recall should expose this record at `now_unix_ms`.
    pub fn visible_at(&self, now_unix_ms: u64) -> bool {
        self.status.is_active() && !self.retention.is_expired_at(now_unix_ms)
    }

    /// Upserts payload at a strictly newer generation while preserving identity.
    pub fn upsert_payload(
        &mut self,
        payload: impl Into<Vec<u8>>,
        generation: u64,
    ) -> Result<(), RecordError> {
        self.advance_generation(generation)?;
        self.payload = payload.into();
        self.status = MemoryStatus::Active;
        Ok(())
    }

    /// Creates a logical delete marker. Physical purge is separate.
    pub fn tombstone(&mut self, generation: u64) -> Result<(), RecordError> {
        self.advance_generation(generation)?;
        self.status = MemoryStatus::Tombstoned {
            at_generation: generation,
        };
        Ok(())
    }

    /// Marks this record superseded and records that evidence relation.
    pub fn supersede(&mut self, by: MemoryId, generation: u64) -> Result<(), RecordError> {
        if by == self.id {
            return Err(RecordError::SelfRelation);
        }
        self.advance_generation(generation)?;
        self.relations.push(MemoryRelation {
            kind: RelationKind::Supersedes,
            target: by.clone(),
        });
        self.status = MemoryStatus::Superseded {
            by,
            at_generation: generation,
        };
        Ok(())
    }

    pub fn add_relation(
        &mut self,
        kind: RelationKind,
        target: MemoryId,
    ) -> Result<(), RecordError> {
        if target == self.id {
            return Err(RecordError::SelfRelation);
        }
        self.relations.push(MemoryRelation { kind, target });
        Ok(())
    }

    /// A logical deletion can be physically purged only after its retention floor.
    pub fn purgeable_at(&self, now_unix_ms: u64) -> bool {
        !self.status.is_active() && self.retention.permits_purge_at(now_unix_ms)
    }

    fn advance_generation(&mut self, generation: u64) -> Result<(), RecordError> {
        if generation <= self.generation {
            return Err(RecordError::NonMonotonicGeneration {
                current: self.generation,
                proposed: generation,
            });
        }
        self.generation = generation;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecordError {
    EmptyId,
    ZeroEmbeddingDimension,
    SelfRelation,
    NonMonotonicGeneration { current: u64, proposed: u64 },
}

impl fmt::Display for RecordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyId => f.write_str("memory id must not be empty"),
            Self::ZeroEmbeddingDimension => {
                f.write_str("embedding fingerprint dimension must be non-zero")
            }
            Self::SelfRelation => f.write_str("a memory record cannot relate to itself"),
            Self::NonMonotonicGeneration { current, proposed } => write!(
                f,
                "record generation must increase monotonically: current={current}, proposed={proposed}"
            ),
        }
    }
}

impl std::error::Error for RecordError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(value: &str) -> MemoryId {
        MemoryId::new(value).unwrap()
    }

    #[test]
    fn ids_and_embedding_dimensions_are_validated() {
        assert_eq!(MemoryId::new("   "), Err(RecordError::EmptyId));
        assert_eq!(
            EmbeddingFingerprint::new("scirust", "encoder", 0),
            Err(RecordError::ZeroEmbeddingDimension)
        );
    }

    #[test]
    fn lifecycle_is_monotonic_and_separates_delete_from_purge() {
        let mut record = MemoryRecord::new(id("m:1"), b"v1".to_vec(), 7);
        record.retention.retain_until_unix_ms = Some(1_000);
        assert!(record.visible_at(500));
        assert_eq!(
            record.tombstone(7),
            Err(RecordError::NonMonotonicGeneration {
                current: 7,
                proposed: 7
            })
        );
        record.tombstone(8).unwrap();
        assert!(!record.visible_at(500));
        assert!(!record.purgeable_at(999));
        assert!(record.purgeable_at(1_000));
    }

    #[test]
    fn ttl_hides_active_record_without_mutating_status() {
        let mut record = MemoryRecord::new(id("m:ttl"), b"x".to_vec(), 1);
        record.retention.expires_at_unix_ms = Some(10);
        assert!(record.visible_at(9));
        assert!(!record.visible_at(10));
        assert!(record.status.is_active());
    }

    #[test]
    fn upsert_preserves_identity_and_requires_newer_generation() {
        let original = id("stable");
        let mut record = MemoryRecord::new(original.clone(), b"old".to_vec(), 2);
        record.upsert_payload(b"new".to_vec(), 3).unwrap();
        assert_eq!(record.id, original);
        assert_eq!(record.payload, b"new");
        assert_eq!(record.generation, 3);
        assert!(record.upsert_payload(b"stale".to_vec(), 2).is_err());
    }

    #[test]
    fn supersession_is_explicit_evidence_and_rejects_self_links() {
        let mut record = MemoryRecord::new(id("old"), b"old".to_vec(), 1);
        assert_eq!(
            record.add_relation(RelationKind::Confirms, id("old")),
            Err(RecordError::SelfRelation)
        );
        record.supersede(id("new"), 2).unwrap();
        assert!(matches!(record.status, MemoryStatus::Superseded { .. }));
        assert_eq!(record.relations.len(), 1);
        assert_eq!(record.relations[0].kind, RelationKind::Supersedes);
    }
}
