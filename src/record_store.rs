//! The record layer that makes [`MemoryRecord`]s live: a validated sidecar map
//! from stable logical ids to records, with tombstones, TTL visibility, and
//! purge accounting.
//!
//! The store never reads a wall clock — every lifecycle decision takes an
//! externally supplied `now_unix_ms`, exactly like the record model. Physical
//! index entries are append-only, so `purge` removes *records*; reclaiming the
//! underlying index space is a rebuild/compaction concern (publish a new
//! generation containing only visible items).
//!
//! Persistence is a versioned, length-prefixed binary format (`RECS` v1) with
//! validate-before-allocate decoding: every declared length or count is checked
//! against the bytes actually available before it can drive an allocation.

use std::collections::BTreeMap;
use std::io::{self, Read};
use std::path::Path;

use crate::record::{
    EmbeddingFingerprint, MemoryId, MemoryRecord, MemoryRelation, MemoryScope, MemoryStatus,
    Provenance, RecordError, RelationKind, Retention, Sensitivity,
};

const MAGIC: &[u8; 4] = b"RECS";
const VERSION: u32 = 1;
/// Lower bound on the encoded size of one record: fixed-width fields only
/// (four 4-byte string headers, counts, flags, scalars) — used to reject
/// hostile record counts before any allocation they would size.
const MIN_ENCODED_RECORD_BYTES: usize = 54;

#[derive(Clone, Debug, Default)]
pub struct RecordStore {
    records: BTreeMap<String, MemoryRecord>,
}

/// The full recall predicate for the record layer: lifecycle (`now_unix_ms`),
/// optional tenant/workspace/agent scoping (`None` = wildcard at that level),
/// and a sensitivity clearance — records classified strictly above it are
/// hidden. The default filter is unscoped, fully cleared, at time zero.
///
/// **Compaction must run the exact same predicate as recall** (see
/// `ShardedHybrid::compact_filtered`): dropping an index entry a legal query
/// could still return would be silent multi-tenant data loss.
#[derive(Clone, Debug)]
pub struct RecordFilter {
    pub now_unix_ms: u64,
    pub tenant: Option<String>,
    pub workspace: Option<String>,
    pub agent: Option<String>,
    /// Records with `sensitivity > clearance` are hidden.
    pub clearance: Sensitivity,
}

impl Default for RecordFilter {
    fn default() -> Self {
        Self {
            now_unix_ms: 0,
            tenant: None,
            workspace: None,
            agent: None,
            clearance: Sensitivity::Restricted,
        }
    }
}

impl RecordFilter {
    /// A lifecycle-only filter (no scoping, full clearance).
    pub fn at(now_unix_ms: u64) -> Self {
        Self {
            now_unix_ms,
            ..Self::default()
        }
    }

    fn admits_scope(&self, tenant: &str, workspace: &str, agent: &str) -> bool {
        self.tenant.as_deref().is_none_or(|want| want == tenant)
            && self
                .workspace
                .as_deref()
                .is_none_or(|want| want == workspace)
            && self.agent.as_deref().is_none_or(|want| want == agent)
    }
}

impl RecordStore {
    /// An empty record store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of stored records (including invisible ones).
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether no records are stored.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// The stored record for `id`, if any.
    pub fn get(&self, id: &str) -> Option<&MemoryRecord> {
        self.records.get(id)
    }

    /// Whether recall may expose `id` at `now_unix_ms`. An id with **no**
    /// record is treated as visible so plain index payloads (which predate the
    /// record layer) keep flowing through filtered recalls.
    pub fn is_visible_at(&self, id: &str, now_unix_ms: u64) -> bool {
        match self.records.get(id) {
            Some(record) => record.visible_at(now_unix_ms),
            None => true,
        }
    }

    /// [`RecordStore::is_visible_at`] under a full [`RecordFilter`] — lifecycle
    /// state, tenant/workspace/agent scoping and sensitivity clearance in one
    /// predicate. Unknown ids pass through (see above); known ids must satisfy
    /// every dimension of the filter.
    pub fn admits(&self, id: &str, filter: &RecordFilter) -> bool {
        match self.records.get(id) {
            Some(record) => {
                record.visible_at(filter.now_unix_ms)
                    && filter.admits_scope(
                        record.scope.tenant(),
                        record.scope.workspace(),
                        record.scope.agent(),
                    )
                    && record.sensitivity <= filter.clearance
            }
            None => true,
        }
    }

    /// Inserts or replaces `record`. Replacing an existing id requires a
    /// strictly newer generation (`NonMonotonicGeneration` otherwise). Returns
    /// the previously stored record, if any — callers use it to roll back an
    /// upsert whose follow-up work failed.
    pub fn put(&mut self, record: MemoryRecord) -> Result<Option<MemoryRecord>, RecordError> {
        let key = record.id.as_str().to_string();
        if let Some(existing) = self.records.get(&key)
            && record.generation <= existing.generation
        {
            return Err(RecordError::NonMonotonicGeneration {
                current: existing.generation,
                proposed: record.generation,
            });
        }
        Ok(self.records.insert(key, record))
    }

    /// Marks `id` tombstoned at a strictly newer generation. The record must
    /// exist; logical deletes of unknown ids are refused rather than invented.
    pub fn tombstone(&mut self, id: &str, generation: u64) -> Result<(), RecordError> {
        let record = self
            .records
            .get_mut(id)
            .ok_or_else(|| RecordError::UnknownMemoryId(id.to_string()))?;
        record.tombstone(generation)
    }

    /// Ids whose records are purgeable at `now_unix_ms` (inactive and past the
    /// retention floor), sorted.
    pub fn purgeable_ids_at(&self, now_unix_ms: u64) -> Vec<String> {
        self.records
            .iter()
            .filter(|(_, r)| r.purgeable_at(now_unix_ms))
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Physically removes every record purgeable at `now_unix_ms` from the
    /// record layer and returns how many were removed. Index-space reclamation
    /// happens when the caller compacts/rebuilds.
    pub fn purge_purgeable_at(&mut self, now_unix_ms: u64) -> usize {
        let doomed = self.purgeable_ids_at(now_unix_ms);
        let removed = doomed.len();
        for id in doomed {
            self.records.remove(&id);
        }
        removed
    }

    /// The outgoing relation targets of `id` that the filter admits, in the
    /// record's own insertion order — one BFS step of the relation graph.
    /// Unknown ids have no outgoing edges; targets filtered out (hidden,
    /// foreign scope, above clearance) are silently skipped so traversal can
    /// never become a side channel around [`RecordFilter`].
    pub fn related_ids(&self, id: &str, filter: &RecordFilter) -> Vec<(RelationKind, String)> {
        let Some(record) = self.records.get(id) else {
            return Vec::new();
        };
        if !self.admits(id, filter) {
            // A hidden source has no traversable edges either.
            return Vec::new();
        }
        record
            .relations
            .iter()
            .filter(|relation| self.admits(relation.target.as_str(), filter))
            .map(|relation| (relation.kind, relation.target.as_str().to_string()))
            .collect()
    }

    /// Encodes the store as `RECS` v1 bytes (sorted by id — deterministic).
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(self.records.len() as u64).to_le_bytes());
        for record in self.records.values() {
            put_str(&mut out, record.id.as_str());
            put_bytes(&mut out, &record.payload);
            put_str(&mut out, record.scope.tenant());
            put_str(&mut out, record.scope.workspace());
            put_str(&mut out, record.scope.agent());

            out.extend_from_slice(&(record.provenance.len() as u32).to_le_bytes());
            for p in &record.provenance {
                put_str(&mut out, p.source());
                match p.source_record() {
                    Some(value) => {
                        out.push(1);
                        put_str(&mut out, value);
                    }
                    None => out.push(0),
                }
                match p.observed_at_unix_ms() {
                    Some(at) => {
                        out.push(1);
                        out.extend_from_slice(&at.to_le_bytes());
                    }
                    None => out.push(0),
                }
            }

            match record.created_at_unix_ms {
                Some(at) => {
                    out.push(1);
                    out.extend_from_slice(&at.to_le_bytes());
                }
                None => out.push(0),
            }
            out.extend_from_slice(&record.generation.to_le_bytes());
            match &record.causal_region {
                Some(region) => {
                    out.push(1);
                    put_str(&mut out, region);
                }
                None => out.push(0),
            }
            out.push(sensitivity_code(record.sensitivity));
            out.push(match &record.status {
                MemoryStatus::Active => 0,
                MemoryStatus::Tombstoned { .. } => 1,
                MemoryStatus::Superseded { .. } => 2,
            });
            if let MemoryStatus::Tombstoned { at_generation } = &record.status {
                out.extend_from_slice(&at_generation.to_le_bytes());
            }
            if let MemoryStatus::Superseded { by, at_generation } = &record.status {
                put_str(&mut out, by.as_str());
                out.extend_from_slice(&at_generation.to_le_bytes());
            }
            match record.retention.expires_at_unix_ms {
                Some(deadline) => {
                    out.push(1);
                    out.extend_from_slice(&deadline.to_le_bytes());
                }
                None => out.push(0),
            }
            match record.retention.retain_until_unix_ms {
                Some(deadline) => {
                    out.push(1);
                    out.extend_from_slice(&deadline.to_le_bytes());
                }
                None => out.push(0),
            }

            put_str(&mut out, record.embedding.provider());
            put_str(&mut out, record.embedding.model());
            match record.embedding.revision() {
                Some(value) => {
                    out.push(1);
                    put_str(&mut out, value);
                }
                None => out.push(0),
            }
            out.extend_from_slice(&(record.embedding.dimension() as u64).to_le_bytes());
            match record.embedding.projection() {
                Some(value) => {
                    out.push(1);
                    put_str(&mut out, value);
                }
                None => out.push(0),
            }
            match record.embedding.quantization() {
                Some(value) => {
                    out.push(1);
                    put_str(&mut out, value);
                }
                None => out.push(0),
            }

            out.extend_from_slice(&(record.relations.len() as u32).to_le_bytes());
            for relation in &record.relations {
                out.push(match relation.kind {
                    RelationKind::Confirms => 0,
                    RelationKind::Contradicts => 1,
                    RelationKind::Supersedes => 2,
                    RelationKind::SupersededBy => 3,
                });
                put_str(&mut out, relation.target.as_str());
            }
        }
        out
    }

    /// Decodes `RECS` v1 bytes written by [`RecordStore::encode`]. Every count
    /// and length is validated against the remaining input before it can size
    /// an allocation; structural violations are `InvalidData` errors, and
    /// semantically invalid records (empty ids, non-monotonic relations) fail
    /// without being inserted.
    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        let mut r: &[u8] = bytes;
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(invalid("not a RECS record store (bad magic)"));
        }
        let version = read_u32(&mut r)?;
        if version != VERSION {
            return Err(invalid(&format!("unsupported RECS version {version}")));
        }
        let count = read_u64(&mut r)?;
        crate::fileguard::guard_count(
            "RECS records",
            count as usize,
            MIN_ENCODED_RECORD_BYTES,
            r.len() as u64,
        )?;

        let mut store = RecordStore::new();
        for _ in 0..count {
            let id = read_lp_string("RECS id", &mut r)?;
            let payload = read_lp_bytes("RECS payload", &mut r)?;
            let scope = MemoryScope::new(
                read_lp_string("RECS tenant", &mut r)?,
                read_lp_string("RECS workspace", &mut r)?,
                read_lp_string("RECS agent", &mut r)?,
            )
            .map_err(|e| invalid(&e.to_string()))?;

            let prov_count = read_u32(&mut r)?;
            crate::fileguard::guard_count(
                "RECS provenances",
                prov_count as usize,
                3,
                r.len() as u64,
            )?;
            let mut provenance = Vec::with_capacity(prov_count as usize);
            for _ in 0..prov_count {
                let source = read_lp_string("RECS provenance source", &mut r)?;
                let mut p = Provenance::new(source).map_err(|e| invalid(&e.to_string()))?;
                if read_flag("RECS provenance record flag", &mut r)? {
                    p = p
                        .with_source_record(read_lp_string("RECS provenance record", &mut r)?)
                        .map_err(|e| invalid(&e.to_string()))?;
                }
                if read_flag("RECS observed flag", &mut r)? {
                    p = p.with_observed_at(read_u64(&mut r)?);
                }
                provenance.push(p);
            }

            let created_at_unix_ms = read_opt_u64("RECS created flag", &mut r)?;
            let generation = read_u64(&mut r)?;
            let causal_region = match read_flag("RECS region flag", &mut r)? {
                true => Some(read_lp_string("RECS causal region", &mut r)?),
                false => None,
            };
            let sensitivity = sensitivity_from_code(read_u8("RECS sensitivity", &mut r)?)?;
            let status = read_status(&mut r)?;
            let retention = Retention {
                expires_at_unix_ms: read_opt_u64("RECS expires flag", &mut r)?,
                retain_until_unix_ms: read_opt_u64("RECS retain flag", &mut r)?,
            };

            let provider = read_lp_string("RECS embedding provider", &mut r)?;
            let model = read_lp_string("RECS embedding model", &mut r)?;
            let revision = match read_flag("RECS revision flag", &mut r)? {
                true => Some(read_lp_string("RECS embedding revision", &mut r)?),
                false => None,
            };
            let dim = read_u64(&mut r)? as usize;
            if dim == 0 {
                return Err(invalid("RECS embedding dimension must be non-zero"));
            }
            let mut embedding = EmbeddingFingerprint::new(provider, model, dim)
                .map_err(|e| invalid(&e.to_string()))?;
            if let Some(revision) = revision {
                embedding = embedding
                    .with_revision(revision)
                    .map_err(|e| invalid(&e.to_string()))?;
            }
            if read_flag("RECS projection flag", &mut r)? {
                embedding = embedding
                    .with_projection(read_lp_string("RECS projection", &mut r)?)
                    .map_err(|e| invalid(&e.to_string()))?;
            }
            if read_flag("RECS quantization flag", &mut r)? {
                embedding = embedding
                    .with_quantization(read_lp_string("RECS quantization", &mut r)?)
                    .map_err(|e| invalid(&e.to_string()))?;
            }

            let rel_count = read_u32(&mut r)?;
            crate::fileguard::guard_count("RECS relations", rel_count as usize, 5, r.len() as u64)?;
            let mut relations = Vec::with_capacity(rel_count as usize);
            for _ in 0..rel_count {
                let kind = match read_u8("RECS relation kind", &mut r)? {
                    0 => RelationKind::Confirms,
                    1 => RelationKind::Contradicts,
                    2 => RelationKind::Supersedes,
                    3 => RelationKind::SupersededBy,
                    other => return Err(invalid(&format!("unknown RECS relation kind {other}"))),
                };
                relations.push(MemoryRelation {
                    kind,
                    target: MemoryId::new(read_lp_string("RECS relation target", &mut r)?)
                        .map_err(|e| invalid(&e.to_string()))?,
                });
            }

            let record = MemoryRecord {
                id: MemoryId::new(id.clone()).map_err(|e| invalid(&e.to_string()))?,
                payload,
                scope,
                provenance,
                created_at_unix_ms,
                generation,
                causal_region,
                sensitivity,
                status,
                retention,
                embedding,
                relations,
            };
            store.put(record).map_err(|e| invalid(&e.to_string()))?;
        }
        crate::fileguard::guard_no_trailing_bytes("RECS record store", r.len())?;
        Ok(store)
    }

    /// Writes the store to `path` (single synced file; the caller's manifest is
    /// the commit point).
    pub fn save_to_disk(&self, path: impl AsRef<Path>) -> io::Result<()> {
        std::fs::write(path, self.encode())
    }

    /// Loads a store written by [`RecordStore::save_to_disk`].
    pub fn load_from_disk(path: impl AsRef<Path>) -> io::Result<Self> {
        Self::decode(&std::fs::read(path)?)
    }
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    put_bytes(out, value.as_bytes())
}

fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}

fn read_u8(what: &str, r: &mut &[u8]) -> io::Result<u8> {
    crate::fileguard::guard_count(what, 1, 1, r.len() as u64)?;
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

fn read_u32(r: &mut &[u8]) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut &[u8]) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_flag(what: &str, r: &mut &[u8]) -> io::Result<bool> {
    match read_u8(what, r)? {
        0 => Ok(false),
        1 => Ok(true),
        other => Err(invalid(&format!("{what} must be 0 or 1, got {other}"))),
    }
}

fn read_opt_u64(what: &str, r: &mut &[u8]) -> io::Result<Option<u64>> {
    match read_flag(what, r)? {
        false => Ok(None),
        true => Ok(Some(read_u64(r)?)),
    }
}

fn read_lp_bytes(what: &str, r: &mut &[u8]) -> io::Result<Vec<u8>> {
    let len = read_u32(r)? as usize;
    crate::fileguard::guard_count(what, len, 1, r.len() as u64)?;
    let mut b = vec![0u8; len];
    r.read_exact(&mut b)?;
    Ok(b)
}

fn read_lp_string(what: &str, r: &mut &[u8]) -> io::Result<String> {
    String::from_utf8(read_lp_bytes(what, r)?).map_err(|e| invalid(&e.to_string()))
}

fn read_status(r: &mut &[u8]) -> io::Result<MemoryStatus> {
    match read_u8("RECS status", r)? {
        0 => Ok(MemoryStatus::Active),
        1 => Ok(MemoryStatus::Tombstoned {
            at_generation: read_u64(r)?,
        }),
        2 => {
            let by = MemoryId::new(read_lp_string("RECS superseded-by", r)?)
                .map_err(|e| invalid(&e.to_string()))?;
            Ok(MemoryStatus::Superseded {
                by,
                at_generation: read_u64(r)?,
            })
        }
        other => Err(invalid(&format!("unknown RECS status code {other}"))),
    }
}

fn sensitivity_code(sensitivity: Sensitivity) -> u8 {
    match sensitivity {
        Sensitivity::Public => 0,
        Sensitivity::Internal => 1,
        Sensitivity::Confidential => 2,
        Sensitivity::Restricted => 3,
    }
}

fn sensitivity_from_code(code: u8) -> io::Result<Sensitivity> {
    Ok(match code {
        0 => Sensitivity::Public,
        1 => Sensitivity::Internal,
        2 => Sensitivity::Confidential,
        3 => Sensitivity::Restricted,
        other => return Err(invalid(&format!("unknown RECS sensitivity code {other}"))),
    })
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope(name: &str) -> MemoryScope {
        MemoryScope::new("tenant-a", "workspace-a", name).unwrap()
    }

    fn provenance() -> Provenance {
        Provenance::new("ccos:event-log")
            .unwrap()
            .with_source_record("event:42")
            .unwrap()
            .with_observed_at(1234)
    }

    fn embedding(dim: usize) -> EmbeddingFingerprint {
        EmbeddingFingerprint::new("scirust", "sciagent-encoder", dim)
            .unwrap()
            .with_revision("rev-1")
            .unwrap()
            .with_quantization("f32")
            .unwrap()
    }

    fn record(id: &str, generation: u64) -> MemoryRecord {
        let mut record = MemoryRecord::new(
            MemoryId::new(id).unwrap(),
            format!("payload-of-{id}").into_bytes(),
            scope("agent-a"),
            provenance(),
            embedding(768),
            generation,
        );
        record.created_at_unix_ms = Some(100);
        record.causal_region = Some("src/db.rs".into());
        record.sensitivity = Sensitivity::Confidential;
        record.retention.expires_at_unix_ms = Some(2_000);
        record.retention.retain_until_unix_ms = Some(1_500);
        record
            .add_relation(RelationKind::Confirms, MemoryId::new("m:other").unwrap())
            .unwrap();
        record
    }

    #[test]
    fn put_get_and_monotonic_upsert_enforcement() {
        let mut store = RecordStore::new();
        assert!(store.is_empty());
        assert!(store.put(record("m:1", 5)).unwrap().is_none());
        assert_eq!(store.len(), 1);
        assert_eq!(store.get("m:1").unwrap().generation, 5);

        // Same-generation replacement is refused; newer replaces and returns old.
        assert_eq!(
            store.put(record("m:1", 5)).unwrap_err(),
            RecordError::NonMonotonicGeneration {
                current: 5,
                proposed: 5
            }
        );
        let old = store.put(record("m:1", 6)).unwrap().unwrap();
        assert_eq!(old.generation, 5);

        assert_eq!(
            store.tombstone("missing", 9).unwrap_err(),
            RecordError::UnknownMemoryId("missing".into())
        );
    }

    #[test]
    fn visibility_ttl_tombstone_and_purge_accounting() {
        let mut store = RecordStore::new();
        store.put(record("live", 1)).unwrap();
        let mut expired = record("expired", 1);
        expired.retention.expires_at_unix_ms = Some(1_000);
        store.put(expired).unwrap();
        let mut dead = record("dead", 1);
        dead.retention.retain_until_unix_ms = Some(1_000);
        store.put(dead).unwrap();
        store.tombstone("dead", 2).unwrap();

        // The shared helper gives every record a 2_000 TTL expiry and a
        // 1_500 retention floor; 'now' sits between them.
        let now = 1_500;
        assert!(store.is_visible_at("live", now));
        assert!(!store.is_visible_at("expired", now)); // TTL hid it (deadline 1_000)
        assert!(!store.is_visible_at("dead", now)); // tombstoned

        // Unknown ids pass through (plain-index back-compat contract).
        assert!(store.is_visible_at("never-stored", now));

        // 'expired' is still Active → not purgeable even though its TTL passed;
        // 'dead' is inactive and past both floors → purgeable.
        assert_eq!(store.purgeable_ids_at(now), vec!["dead".to_string()]);
        assert_eq!(store.purge_purgeable_at(now), 1);
        assert_eq!(store.len(), 2);
        assert!(store.get("dead").is_none());
    }

    #[test]
    fn recs_roundtrip_preserves_every_field() {
        let mut store = RecordStore::new();
        store.put(record("m:é-1", 3)).unwrap();
        let mut plain = MemoryRecord::new(
            MemoryId::new("m:plain").unwrap(),
            b"p".to_vec(),
            scope("agent-b"),
            Provenance::new("bare").unwrap(),
            embedding(4),
            1,
        );
        plain.tombstone(4).unwrap();
        store.put(plain).unwrap();

        let decoded = RecordStore::decode(&store.encode()).unwrap();
        assert_eq!(decoded.len(), store.len());
        for (id, original) in store.records.iter() {
            assert_eq!(
                decoded.get(id),
                Some(original),
                "roundtrip mismatch for {id}"
            );
        }
    }

    #[test]
    fn recs_rejects_hostile_counts_magic_and_trailing_bytes() {
        let good = RecordStore::new().encode();
        // Bad magic.
        let mut bad = good.clone();
        bad[0] = b'X';
        assert_eq!(
            RecordStore::decode(&bad).err().unwrap().kind(),
            io::ErrorKind::InvalidData
        );

        // A hostile record count cannot drive an allocation: the declared
        // minimum per-record footprint must exist in the file first.
        let mut hostile = Vec::new();
        hostile.extend_from_slice(MAGIC);
        hostile.extend_from_slice(&VERSION.to_le_bytes());
        hostile.extend_from_slice(&u64::MAX.to_le_bytes());
        let err = RecordStore::decode(&hostile).err().unwrap();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("RECS records"));

        // Trailing garbage after a valid body is rejected.
        let mut trailing = good.clone();
        trailing.push(0);
        assert!(RecordStore::decode(&trailing).is_err());

        // Truncated mid-record is rejected.
        assert!(RecordStore::decode(&good[..good.len() - 3]).is_err());
    }

    #[test]
    fn recs_disk_roundtrip() {
        let mut store = RecordStore::new();
        store.put(record("m:disk", 2)).unwrap();
        let path = "/tmp/octasoma_record_store_roundtrip.recs";
        store.save_to_disk(path).unwrap();
        let loaded = RecordStore::load_from_disk(path).unwrap();
        assert_eq!(loaded.get("m:disk"), store.get("m:disk"));
        std::fs::remove_file(path).ok();
    }
}
