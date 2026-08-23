//! Clustered per-theme 3-D projection — the intermediate answer to the
//! documented collapse of a single global 3-D projection past ~a dozen latent
//! themes (cluster recall@1: 100 % at 4 themes → 2.6 % at 256, see
//! `docs/evaluation.md`).
//!
//! The corpus is partitioned by deterministic full-D k-means; **each cluster
//! gets its own 3×D PCA head** trained on its members, and items are indexed in
//! their own [`FractalMemory3D`]. Queries are routed to the `top_t` nearest
//! centroids and merged across those clusters. This is ShardedMemory's
//! validated per-region-PCA trick, applied with *discovered* semantic regions
//! instead of caller-supplied keys.
//!
//! Honest limits, part of the design:
//! - cross-cluster distances come from different projections and are **not
//!   strictly comparable** (same caveat as `ShardedMemory::recall_global`);
//! - routing itself is approximate: a query near a cluster boundary may be
//!   answered from the wrong `top_t` window (mitigate with larger `top_t`);
//! - clusters are static once trained — retrain on drift, like PQ codebooks;
//! - in-memory only for now; persistence rides the same generation
//!   infrastructure as the other stores in a later step.

use crate::{Embedder, FractalMemory3D, compute_pca_projection_parallel};

const KMEANS_ITERS: usize = 15;

/// Deterministic full-D k-means over L2-normalized rows: stride initialization,
/// fixed iterations, f64 mean accumulation, lowest-index tie-breaking. Returns
/// `(assignments, centroids)` — same input, byte-equal output anywhere.
fn kmeans(vectors: &[f32], n: usize, dim: usize, k: usize) -> (Vec<usize>, Vec<f32>) {
    let k = k.clamp(1, n);
    let mut centroids: Vec<f32> = Vec::with_capacity(k * dim);
    for c in 0..k {
        // Stride sampling covers the corpus without a PRNG.
        let pick = (c as u64 * n as u64 / k as u64) as usize;
        centroids.extend_from_slice(&vectors[pick * dim..(pick + 1) * dim]);
    }

    let mut assignments = vec![0usize; n];
    for _ in 0..KMEANS_ITERS {
        // Assignment: nearest centroid by cosine (normalized rows), ties to
        // the lower index via strict `<`.
        for s in 0..n {
            let v = &vectors[s * dim..(s + 1) * dim];
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let centroid = &centroids[c * dim..(c + 1) * dim];
                let dot: f32 = v.iter().zip(centroid).map(|(&x, &y)| x * y).sum();
                let d = 1.0 - dot;
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            assignments[s] = best;
        }
        // Update: empty clusters keep their previous centroid.
        let mut means = vec![0f64; k * dim];
        let mut counts = vec![0u32; k];
        for s in 0..n {
            let c = assignments[s];
            counts[c] += 1;
            let v = &vectors[s * dim..(s + 1) * dim];
            for j in 0..dim {
                means[c * dim + j] += v[j] as f64;
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                continue;
            }
            for j in 0..dim {
                centroids[c * dim + j] = (means[c * dim + j] / counts[c] as f64) as f32;
            }
        }
    }
    (assignments, centroids)
}

/// A clustered multi-projection fractal memory (see the module docs).
pub struct ClusteredMemory<E: Embedder> {
    clusters: Vec<FractalMemory3D>,
    /// Payloads per cluster member (the 3-D layer keeps only projected points
    /// plus arena offsets; keeping payloads here mirrors that split).
    payloads: Vec<Vec<Vec<u8>>>,
    centroids: Vec<f32>,
    dim: usize,
    embedder: E,
}

impl<E: Embedder> ClusteredMemory<E> {
    /// Builds the memory from pre-computed `(payload, vector)` pairs: k-means
    /// partition over normalized vectors, then one parallel-trained PCA head
    /// per cluster (bit-identical for any thread count — proposal C2).
    ///
    /// # Panics
    /// If `items` is empty or any vector is empty / of inconsistent length.
    pub fn build(items: &[(&[u8], Vec<f32>)], num_clusters: usize, embedder: E) -> Self {
        assert!(!items.is_empty(), "build needs items");
        assert!(num_clusters > 0, "num_clusters must be non-zero");
        let dim = items[0].1.len();
        assert!(dim > 0, "vector dimension must be non-zero");
        assert!(
            items.iter().all(|(_, v)| v.len() == dim),
            "all vectors must share one dimension"
        );

        let flat: Vec<f32> = items
            .iter()
            .flat_map(|(_, v)| {
                let mut copy = v.clone();
                crate::normalize_unit(&mut copy);
                copy
            })
            .collect();
        let (assignments, centroids) =
            kmeans(&flat, items.len(), dim, num_clusters.min(items.len()));
        let k = centroids.len() / dim;

        let mut members: Vec<Vec<usize>> = vec![Vec::new(); k];
        for (s, &a) in assignments.iter().enumerate() {
            members[a].push(s);
        }

        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let mut clusters = Vec::with_capacity(k);
        let mut payloads = Vec::with_capacity(k);
        for member_ids in &members {
            if member_ids.is_empty() {
                // Index alignment: an empty cluster stays an empty memory.
                clusters.push(FractalMemory3D::new(dim, 0));
                payloads.push(Vec::new());
                continue;
            }
            let n = member_ids.len();
            let mut training = Vec::with_capacity(n * dim);
            for &s in member_ids {
                training.extend_from_slice(&flat[s * dim..(s + 1) * dim]);
            }
            let projection = compute_pca_projection_parallel(&training, n, dim, 20, threads);
            let mut memory = FractalMemory3D::new_from_calibration(dim, projection);
            let mut cluster_payloads = Vec::with_capacity(n);
            for &s in member_ids {
                let (payload, _) = &items[s];
                assert!(
                    memory
                        .insert(&flat[s * dim..(s + 1) * dim], Some(payload))
                        .is_some(),
                    "validated cluster member must project"
                );
                cluster_payloads.push(payload.to_vec());
            }
            clusters.push(memory);
            payloads.push(cluster_payloads);
        }

        Self {
            clusters,
            payloads,
            centroids,
            dim,
            embedder,
        }
    }

    /// Number of clusters.
    pub fn num_clusters(&self) -> usize {
        self.clusters.len()
    }

    /// Total items across all clusters.
    pub fn len(&self) -> usize {
        self.clusters.iter().map(FractalMemory3D::item_count).sum()
    }

    /// Whether nothing has been stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Recall routed to the `top_t` clusters whose centroids are closest to
    /// `query` (cosine), merged ascending by projected distance² and truncated
    /// to `k`. Distances across clusters come from different projections —
    /// treat cross-cluster ordering as heuristic (module docs).
    ///
    /// Returns `(payload, distance²)` pairs. Empty when `k == 0`, the query
    /// dimension mismatches, or the memory is empty.
    pub fn recall_vec(&self, query: &[f32], k: usize, top_t: usize) -> Vec<(Vec<u8>, f32)> {
        if k == 0 || self.clusters.is_empty() || query.len() != self.dim {
            return Vec::new();
        }
        let mut qn = query.to_vec();
        crate::normalize_unit(&mut qn);

        let mut order: Vec<(f32, usize)> = (0..self.clusters.len())
            .map(|c| {
                let centroid = &self.centroids[c * self.dim..(c + 1) * self.dim];
                let dot: f32 = qn.iter().zip(centroid).map(|(&x, &y)| x * y).sum();
                (-dot, c)
            })
            .collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut hits: Vec<(usize, u32, f32)> = Vec::new(); // (cluster, item_id, d2)
        for &(_, c) in order.iter().take(top_t.max(1)) {
            hits.extend(
                self.clusters[c]
                    .nearest_embedding(&qn, k)
                    .into_iter()
                    .map(|(id, d2)| (c, id, d2)),
            );
        }
        hits.sort_by(|a, b| a.2.total_cmp(&b.2));
        hits.truncate(k);
        hits.into_iter()
            .map(|(c, id, d2)| (self.payloads[c][id as usize].clone(), d2))
            .collect()
    }

    /// The embedder dimensionality.
    pub fn dim(&self) -> usize {
        self.embedder.dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HashEmbedder;

    fn vector(dim: usize, theme: usize, jitter: usize) -> Vec<f32> {
        // Near-orthogonal block-one-hot themes (same construction as the PQ
        // gate): theme t owns an 8-dim block set to 1.0. DIM must be >= 8 per
        // theme or blocks overlap into identical rows.
        assert!(dim / 8 > theme);
        (0..dim)
            .map(|d| {
                if d / 8 == theme {
                    1.0 + 0.01 * ((jitter * dim + d) as f32 * 0.7).sin()
                } else {
                    0.02 * (d as f32 * 0.7).sin()
                }
            })
            .collect()
    }

    fn corpus(themes: usize, per_theme: usize) -> Vec<(Vec<u8>, Vec<f32>)> {
        let dim = themes * 8;
        (0..themes)
            .flat_map(|t| {
                (0..per_theme).map(move |i| (format!("t{t}_{i}").into_bytes(), vector(dim, t, i)))
            })
            .collect()
    }

    fn item_refs(items: &[(Vec<u8>, Vec<f32>)]) -> Vec<(&[u8], Vec<f32>)> {
        items
            .iter()
            .map(|(p, v)| (p.as_slice(), v.clone()))
            .collect()
    }

    #[test]
    fn clustered_recall_survives_many_themes_where_global_collapses() {
        let items = corpus(16, 20); // 16 latent themes — global 3-D is at ~73 %
        let memory = ClusteredMemory::build(&item_refs(&items), 8, HashEmbedder::new(128));

        assert_eq!(memory.num_clusters(), 8);
        assert_eq!(memory.len(), items.len());

        // Every query lands back on its own theme's items at the top.
        let mut top1_correct = 0usize;
        for (t, i) in [(3, 4usize), (11, 9), (14, 0), (7, 19)] {
            let query = vector(128, t, i + 100); // unseen jitter
            let hits = memory.recall_vec(&query, 5, 2);
            assert_eq!(hits.len(), 5);
            let want = format!("t{t}_");
            if String::from_utf8_lossy(&hits[0].0).starts_with(&want) {
                top1_correct += 1;
            }
        }
        assert_eq!(top1_correct, 4, "routed top-1 missed its own theme");
    }

    #[test]
    fn routing_window_and_guards_behave() {
        let items = corpus(8, 15);
        let memory = ClusteredMemory::build(&item_refs(&items), 4, HashEmbedder::new(64));

        // top_t=1 still finds the right theme (routing, not luck).
        let query = vector(64, 6, 3 + 50);
        let hits = memory.recall_vec(&query, 3, 1);
        assert!(String::from_utf8_lossy(&hits[0].0).starts_with("t6_"));

        // Guards: bad dim, k=0, unknown-empty are all empty results.
        assert!(memory.recall_vec(&[0.0; 8], 3, 2).is_empty());
        assert!(memory.recall_vec(&query, 0, 2).is_empty());

        // Deterministic rebuild: byte-identical structure.
        let again = ClusteredMemory::build(&item_refs(&items), 4, HashEmbedder::new(64));
        assert_eq!(
            again.recall_vec(&query, 3, 1),
            memory.recall_vec(&query, 3, 1)
        );
    }
}
