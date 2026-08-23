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

/// Product-supplied Tenant → Workspace → Agent isolation scope.
///
/// OctaSoma carries the exact labels but never decides authorization. All three
/// boundaries are mandatory so an omitted label cannot silently collapse two
/// product namespaces into one memory scope.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemoryScope {
    tenant: String,
    workspace: String,
    agent: String,
}

impl MemoryScope {
    pub fn new(
        tenant: impl Into<String>,
        workspace: impl Into<String>,
        agent: impl Into<String>,
    ) -> Result<Self, RecordError> {
        let tenant = tenant.into();
        let workspace = workspace.into();
        let agent = agent.into();
        if tenant.trim().is_empty() {
            return Err(RecordError::EmptyScopeComponent("tenant"));
        }
        if workspace.trim().is_empty() {
            return Err(RecordError::EmptyScopeComponent("workspace"));
        }
        if agent.trim().is_empty() {
            return Err(RecordError::EmptyScopeComponent("agent"));
        }
        Ok(Self {
            tenant,
            workspace,
            agent,
        })
    }

    pub fn tenant(&self) -> &str {
        &self.tenant
    }

    pub fn workspace(&self) -> &str {
        &self.workspace
    }

    pub fn agent(&self) -> &str {
        &self.agent
    }
}

/// Provenance carried with a memory observation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Provenance {
    source: String,
    source_record: Option<String>,
    /// Supplied externally. The record layer never reads a wall clock.
    observed_at_unix_ms: Option<u64>,
}

impl Provenance {
    pub fn new(source: impl Into<String>) -> Result<Self, RecordError> {
        let source = source.into();
        if source.trim().is_empty() {
            return Err(RecordError::EmptyProvenanceSource);
        }
        Ok(Self {
            source,
            source_record: None,
            observed_at_unix_ms: None,
        })
    }

    pub fn with_source_record(
        mut self,
        source_record: impl Into<String>,
    ) -> Result<Self, RecordError> {
        let source_record = source_record.into();
        if source_record.trim().is_empty() {
            return Err(RecordError::EmptyProvenanceRecord);
        }
        self.source_record = Some(source_record);
        Ok(self)
    }

    pub fn with_observed_at(mut self, observed_at_unix_ms: u64) -> Self {
        self.observed_at_unix_ms = Some(observed_at_unix_ms);
        self
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn source_record(&self) -> Option<&str> {
        self.source_record.as_deref()
    }

    pub fn observed_at_unix_ms(&self) -> Option<u64> {
        self.observed_at_unix_ms
    }
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
    provider: String,
    model: String,
    revision: Option<String>,
    dimension: usize,
    projection: Option<String>,
    quantization: Option<String>,
}

impl EmbeddingFingerprint {
    pub fn new(
        provider: impl Into<String>,
        model: impl Into<String>,
        dimension: usize,
    ) -> Result<Self, RecordError> {
        let provider = provider.into();
        let model = model.into();
        if provider.trim().is_empty() {
            return Err(RecordError::EmptyEmbeddingProvider);
        }
        if model.trim().is_empty() {
            return Err(RecordError::EmptyEmbeddingModel);
        }
        if dimension == 0 {
            return Err(RecordError::ZeroEmbeddingDimension);
        }
        Ok(Self {
            provider,
            model,
            revision: None,
            dimension,
            projection: None,
            quantization: None,
        })
    }

    pub fn with_revision(mut self, revision: impl Into<String>) -> Result<Self, RecordError> {
        let revision = revision.into();
        if revision.trim().is_empty() {
            return Err(RecordError::EmptyFingerprintComponent("revision"));
        }
        self.revision = Some(revision);
        Ok(self)
    }

    pub fn with_projection(mut self, projection: impl Into<String>) -> Result<Self, RecordError> {
        let projection = projection.into();
        if projection.trim().is_empty() {
            return Err(RecordError::EmptyFingerprintComponent("projection"));
        }
        self.projection = Some(projection);
        Ok(self)
    }

    pub fn with_quantization(
        mut self,
        quantization: impl Into<String>,
    ) -> Result<Self, RecordError> {
        let quantization = quantization.into();
        if quantization.trim().is_empty() {
            return Err(RecordError::EmptyFingerprintComponent("quantization"));
        }
        self.quantization = Some(quantization);
        Ok(self)
    }

    pub fn provider(&self) -> &str {
        &self.provider
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn revision(&self) -> Option<&str> {
        self.revision.as_deref()
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn projection(&self) -> Option<&str> {
        self.projection.as_deref()
    }

    pub fn quantization(&self) -> Option<&str> {
        self.quantization.as_deref()
    }
}

/// Logical lifecycle state. Tombstones and supersession survive index rebuilds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryStatus {
    Active,
    Tombstoned { at_generation: u64 },
    Superseded { by: MemoryId, at_generation: u64 },
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
    /// Read from the replacer: "this record supersedes `target`".
    Supersedes,
    /// Read from the supplanted: "this record was superseded by `target`" —
    /// the edge [`MemoryRecord::supersede`] records on itself.
    SupersededBy,
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
    /// Mandatory compatibility identity for the semantic representation.
    pub embedding: EmbeddingFingerprint,
    pub relations: Vec<MemoryRelation>,
}

impl MemoryRecord {
    /// Creates an active record with mandatory isolation, provenance and
    /// embedding-contract identity.
    pub fn new(
        id: MemoryId,
        payload: impl Into<Vec<u8>>,
        scope: MemoryScope,
        provenance: Provenance,
        embedding: EmbeddingFingerprint,
        generation: u64,
    ) -> Self {
        Self {
            id,
            payload: payload.into(),
            scope,
            provenance: vec![provenance],
            created_at_unix_ms: None,
            generation,
            causal_region: None,
            sensitivity: Sensitivity::default(),
            status: MemoryStatus::Active,
            retention: Retention::default(),
            embedding,
            relations: Vec::new(),
        }
    }

    /// Whether ordinary recall should expose this record at `now_unix_ms`.
    pub fn visible_at(&self, now_unix_ms: u64) -> bool {
        self.status.is_active() && !self.retention.is_expired_at(now_unix_ms)
    }

    /// Adds another validated provenance assertion.
    pub fn add_provenance(&mut self, provenance: Provenance) {
        self.provenance.push(provenance);
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

    /// Marks this record superseded and records that evidence relation — on
    /// this record, pointing *at* the replacement (`SupersededBy`), so the
    /// edge reads correctly from either end.
    pub fn supersede(&mut self, by: MemoryId, generation: u64) -> Result<(), RecordError> {
        if by == self.id {
            return Err(RecordError::SelfRelation);
        }
        self.advance_generation(generation)?;
        self.relations.push(MemoryRelation {
            kind: RelationKind::SupersededBy,
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

    pub(crate) fn advance_generation(&mut self, generation: u64) -> Result<(), RecordError> {
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
    EmptyScopeComponent(&'static str),
    EmptyProvenanceSource,
    EmptyProvenanceRecord,
    EmptyEmbeddingProvider,
    EmptyEmbeddingModel,
    EmptyFingerprintComponent(&'static str),
    ZeroEmbeddingDimension,
    SelfRelation,
    NonMonotonicGeneration { current: u64, proposed: u64 },
    UnknownMemoryId(String),
}

impl fmt::Display for RecordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyId => f.write_str("memory id must not be empty"),
            Self::EmptyScopeComponent(component) => {
                write!(f, "memory scope component must not be empty: {component}")
            }
            Self::EmptyProvenanceSource => f.write_str("provenance source must not be empty"),
            Self::EmptyProvenanceRecord => {
                f.write_str("provenance source record must not be empty when present")
            }
            Self::EmptyEmbeddingProvider => f.write_str("embedding provider must not be empty"),
            Self::EmptyEmbeddingModel => f.write_str("embedding model must not be empty"),
            Self::EmptyFingerprintComponent(component) => {
                write!(
                    f,
                    "embedding fingerprint component must not be empty: {component}"
                )
            }
            Self::ZeroEmbeddingDimension => {
                f.write_str("embedding fingerprint dimension must be non-zero")
            }
            Self::SelfRelation => f.write_str("a memory record cannot relate to itself"),
            Self::NonMonotonicGeneration { current, proposed } => write!(
                f,
                "record generation must increase monotonically: current={current}, proposed={proposed}"
            ),
            Self::UnknownMemoryId(id) => {
                write!(f, "no memory record with id {id:?}")
            }
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

    fn scope() -> MemoryScope {
        MemoryScope::new("tenant-a", "workspace-a", "agent-a").unwrap()
    }

    fn provenance() -> Provenance {
        Provenance::new("ccos:event-log")
            .unwrap()
            .with_source_record("event:42")
            .unwrap()
    }

    fn embedding() -> EmbeddingFingerprint {
        EmbeddingFingerprint::new("scirust", "sciagent-encoder", 768)
            .unwrap()
            .with_revision("model-rev-1")
            .unwrap()
            .with_projection("projection-head-v3")
            .unwrap()
            .with_quantization("f32")
            .unwrap()
    }

    fn record(id_value: &str, generation: u64) -> MemoryRecord {
        MemoryRecord::new(
            id(id_value),
            b"payload".to_vec(),
            scope(),
            provenance(),
            embedding(),
            generation,
        )
    }

    #[test]
    fn ids_scope_provenance_and_embedding_contract_are_validated() {
        assert_eq!(MemoryId::new("   "), Err(RecordError::EmptyId));
        assert_eq!(
            MemoryScope::new("", "workspace", "agent"),
            Err(RecordError::EmptyScopeComponent("tenant"))
        );
        assert_eq!(
            MemoryScope::new("tenant", "", "agent"),
            Err(RecordError::EmptyScopeComponent("workspace"))
        );
        assert_eq!(
            MemoryScope::new("tenant", "workspace", ""),
            Err(RecordError::EmptyScopeComponent("agent"))
        );
        assert_eq!(
            Provenance::new("  "),
            Err(RecordError::EmptyProvenanceSource)
        );
        assert_eq!(
            EmbeddingFingerprint::new("", "encoder", 768),
            Err(RecordError::EmptyEmbeddingProvider)
        );
        assert_eq!(
            EmbeddingFingerprint::new("scirust", "", 768),
            Err(RecordError::EmptyEmbeddingModel)
        );
        assert_eq!(
            EmbeddingFingerprint::new("scirust", "encoder", 0),
            Err(RecordError::ZeroEmbeddingDimension)
        );
    }

    #[test]
    fn constructor_requires_nested_scope_provenance_and_embedding_identity() {
        let record = record("m:1", 7);
        assert_eq!(record.scope.tenant(), "tenant-a");
        assert_eq!(record.scope.workspace(), "workspace-a");
        assert_eq!(record.scope.agent(), "agent-a");
        assert_eq!(record.provenance[0].source(), "ccos:event-log");
        assert_eq!(record.embedding.provider(), "scirust");
        assert_eq!(record.embedding.dimension(), 768);
        assert!(record.visible_at(0));
    }

    #[test]
    fn lifecycle_is_monotonic_and_separates_delete_from_purge() {
        let mut record = record("m:1", 7);
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
        let mut record = record("m:ttl", 1);
        record.retention.expires_at_unix_ms = Some(10);
        assert!(record.visible_at(9));
        assert!(!record.visible_at(10));
        assert!(record.status.is_active());
    }

    #[test]
    fn upsert_preserves_identity_and_requires_newer_generation() {
        let original = id("stable");
        let mut record = MemoryRecord::new(
            original.clone(),
            b"old".to_vec(),
            scope(),
            provenance(),
            embedding(),
            2,
        );
        record.upsert_payload(b"new".to_vec(), 3).unwrap();
        assert_eq!(record.id, original);
        assert_eq!(record.payload, b"new");
        assert_eq!(record.generation, 3);
        assert!(record.upsert_payload(b"stale".to_vec(), 2).is_err());
    }

    #[test]
    fn supersession_is_explicit_evidence_and_rejects_self_links() {
        let mut record = record("old", 1);
        assert_eq!(
            record.add_relation(RelationKind::Confirms, id("old")),
            Err(RecordError::SelfRelation)
        );
        record.supersede(id("new"), 2).unwrap();
        assert!(matches!(record.status, MemoryStatus::Superseded { .. }));
        assert_eq!(record.relations.len(), 1);
        // The edge lives on the supplanted record and points at the replacement.
        assert_eq!(record.relations[0].kind, RelationKind::SupersededBy);
        assert_eq!(record.relations[0].target, id("new"));
    }
}
