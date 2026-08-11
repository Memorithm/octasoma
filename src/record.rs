//! Canonical logical memory-record model for OctaSoma v0.5.
//!
//! A [`MemoryRecord`] is independent of physical index position, shard layout,
//! precision tier and Spatial/Fractal Lens coordinates. Product adapters supply
//! scope and provenance; OctaSoma stores and recalls the record without becoming
//! authoritative for tenant policy or causal truth.

use std::fmt;

/// Stable logical id, independent of physical store/index location.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemoryRecordId(String);

impl MemoryRecordId {
    /// Creates a non-empty stable id.
    pub fn new(value: impl Into<String>) -> Result<Self, RecordError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(RecordError::InvalidField("id"));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for MemoryRecordId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Product-supplied nested isolation scope.
///
/// OctaSoma carries these ids but does not decide access policy. Empty strings
/// are rejected so a missing boundary cannot silently collapse namespaces.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemoryScope {
    pub tenant_id: String,
    pub workspace_id: String,
    pub agent_id: String,
}

impl MemoryScope {
    pub fn new(
        tenant_id: impl Into<String>,
        workspace_id: impl Into<String>,
        agent_id: impl Into<String>,
    ) -> Result<Self, RecordError> {
        let scope = Self {
            tenant_id: tenant_id.into(),
            workspace_id: workspace_id.into(),
            agent_id: agent_id.into(),
        };
        if scope.tenant_id.trim().is_empty() {
            return Err(RecordError::InvalidField("tenant_id"));
        }
        if scope.workspace_id.trim().is_empty() {
            return Err(RecordError::InvalidField("workspace_id"));
        }
        if scope.agent_id.trim().is_empty() {
            return Err(RecordError::InvalidField("agent_id"));
        }
        Ok(scope)
    }
}

/// Classification supplied by the product adapter. It is metadata, not an
/// authorization decision inside OctaSoma.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum SensitivityLevel {
    #[default]
    Normal,
    Sensitive,
    Restricted,
}

/// Logical lifecycle state. Physical purge is a separate storage operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LifecycleState {
    Active,
    /// No longer recallable, but retained as an auditable logical marker.
    Tombstoned {
        generation: u64,
        reason: Option<String>,
    },
    /// Replaced by a newer logical record.
    Superseded {
        by: MemoryRecordId,
        generation: u64,
    },
    /// Expired by a product-supplied retention/TTL decision.
    Expired {
        generation: u64,
    },
}

impl LifecycleState {
    /// Only active records belong in the normal recall view.
    pub fn is_recallable(&self) -> bool {
        matches!(self, Self::Active)
    }
}

/// Primitive evidence relation between memories.
///
/// These relations describe memory evidence. A consumer such as CCOS remains
/// responsible for deciding whether they alter its own causal state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryRelationKind {
    Supersedes,
    Contradicts,
    Confirms,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryRelation {
    pub kind: MemoryRelationKind,
    pub target: MemoryRecordId,
    /// Optional provenance for the relation assertion itself.
    pub provenance: Option<String>,
}

/// Fingerprint of the representation pipeline used to index this record.
///
/// The opaque value is expected to bind at least the embedding model and any
/// learned projection/quantization configuration relevant to compatibility.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EmbeddingFingerprint(String);

impl EmbeddingFingerprint {
    pub fn new(value: impl Into<String>) -> Result<Self, RecordError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(RecordError::InvalidField("embedding_fingerprint"));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Canonical logical memory object for v0.5.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryRecord {
    pub id: MemoryRecordId,
    pub scope: MemoryScope,
    /// Text or canonical content from which the semantic representation derives.
    pub content: String,
    /// Stable product/audit provenance label or URI.
    pub provenance: String,
    /// Caller-provided event/logical timestamp; OctaSoma does not read a wall
    /// clock when constructing records.
    pub timestamp: u64,
    /// Immutable store generation in which this logical version was written.
    pub generation: u64,
    /// Causal narrowing hint supplied by the consumer; never causal authority.
    pub causal_region_id: Option<String>,
    pub sensitivity_level: SensitivityLevel,
    pub lifecycle_state: LifecycleState,
    pub embedding_fingerprint: EmbeddingFingerprint,
    pub relations: Vec<MemoryRelation>,
}

impl MemoryRecord {
    /// Constructs an active record after validating fields that must never be
    /// empty at the storage boundary.
    pub fn new(
        id: MemoryRecordId,
        scope: MemoryScope,
        content: impl Into<String>,
        provenance: impl Into<String>,
        timestamp: u64,
        generation: u64,
        embedding_fingerprint: EmbeddingFingerprint,
    ) -> Result<Self, RecordError> {
        let content = content.into();
        let provenance = provenance.into();
        if content.is_empty() {
            return Err(RecordError::InvalidField("content"));
        }
        if provenance.trim().is_empty() {
            return Err(RecordError::InvalidField("provenance"));
        }
        Ok(Self {
            id,
            scope,
            content,
            provenance,
            timestamp,
            generation,
            causal_region_id: None,
            sensitivity_level: SensitivityLevel::Normal,
            lifecycle_state: LifecycleState::Active,
            embedding_fingerprint,
            relations: Vec::new(),
        })
    }

    pub fn is_recallable(&self) -> bool {
        self.lifecycle_state.is_recallable()
    }

    /// Adds a primitive relation while preventing a meaningless self-edge.
    pub fn add_relation(&mut self, relation: MemoryRelation) -> Result<(), RecordError> {
        if relation.target == self.id {
            return Err(RecordError::SelfRelation);
        }
        self.relations.push(relation);
        Ok(())
    }

    /// Marks this logical version as tombstoned. Product policy determines when
    /// to call this; storage code determines when a later physical purge occurs.
    pub fn tombstone(&mut self, generation: u64, reason: Option<String>) {
        self.lifecycle_state = LifecycleState::Tombstoned { generation, reason };
    }

    pub fn mark_superseded(
        &mut self,
        by: MemoryRecordId,
        generation: u64,
    ) -> Result<(), RecordError> {
        if by == self.id {
            return Err(RecordError::SelfRelation);
        }
        self.lifecycle_state = LifecycleState::Superseded { by, generation };
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordError {
    InvalidField(&'static str),
    SelfRelation,
}

impl fmt::Display for RecordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidField(field) => write!(f, "invalid or empty memory record field: {field}"),
            Self::SelfRelation => write!(f, "a memory record cannot relate to or supersede itself"),
        }
    }
}

impl std::error::Error for RecordError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn record() -> MemoryRecord {
        MemoryRecord::new(
            MemoryRecordId::new("mem-1").unwrap(),
            MemoryScope::new("tenant-a", "workspace-a", "agent-a").unwrap(),
            "semantic memory content",
            "event:42",
            42,
            7,
            EmbeddingFingerprint::new("embed-v1:projection-v3:int8-none").unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn record_is_logical_and_recallable_by_default() {
        let r = record();
        assert_eq!(r.id.as_str(), "mem-1");
        assert_eq!(r.scope.tenant_id, "tenant-a");
        assert_eq!(r.generation, 7);
        assert!(r.is_recallable());
    }

    #[test]
    fn nested_scope_rejects_missing_boundary() {
        assert!(MemoryScope::new("", "workspace", "agent").is_err());
        assert!(MemoryScope::new("tenant", "", "agent").is_err());
        assert!(MemoryScope::new("tenant", "workspace", "").is_err());
    }

    #[test]
    fn tombstone_removes_record_from_normal_recall_view() {
        let mut r = record();
        r.tombstone(8, Some("retention-policy".into()));
        assert!(!r.is_recallable());
        assert!(matches!(
            r.lifecycle_state,
            LifecycleState::Tombstoned { generation: 8, .. }
        ));
    }

    #[test]
    fn primitive_relations_are_evidence_and_reject_self_edges() {
        let mut r = record();
        r.add_relation(MemoryRelation {
            kind: MemoryRelationKind::Confirms,
            target: MemoryRecordId::new("mem-2").unwrap(),
            provenance: Some("evaluator:rsi".into()),
        })
        .unwrap();
        assert_eq!(r.relations.len(), 1);
        assert!(
            r.add_relation(MemoryRelation {
                kind: MemoryRelationKind::Contradicts,
                target: r.id.clone(),
                provenance: None,
            })
            .is_err()
        );
    }

    #[test]
    fn superseding_self_is_rejected() {
        let mut r = record();
        assert!(r.mark_superseded(r.id.clone(), 9).is_err());
        r.mark_superseded(MemoryRecordId::new("mem-2").unwrap(), 9)
            .unwrap();
        assert!(!r.is_recallable());
    }
}
