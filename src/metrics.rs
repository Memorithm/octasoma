//! Standard IR ranking metrics — Recall@k, Precision@k, MRR, MAP, nDCG@k.
//!
//! Vendored from SciRust (`scirust-retrieval/src/metrics.rs`, same org and dual
//! license) — proposal D1 of `docs/scirust-improvements.md`. These replace the
//! bespoke "cluster recall" counters in the examples with the numbers the ANN /
//! retrieval literature reports, so octasoma's precision claims are comparable
//! and regressions are catchable in CI (see `tests/recall_gate.rs`).
//!
//! Each function takes a ranked list of returned item ids (best first) and a
//! notion of relevance, and returns a score in `[0, 1]`. Pure, deterministic.

use std::collections::{HashMap, HashSet};

/// Recall@k: fraction of all relevant items that appear in the top-`k`.
pub fn recall_at_k(retrieved: &[u64], relevant: &HashSet<u64>, k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let hits = retrieved
        .iter()
        .take(k)
        .filter(|id| relevant.contains(id))
        .count();
    hits as f64 / relevant.len() as f64
}

/// Precision@k: fraction of the top-`k` returned that are relevant. The
/// denominator is `min(k, |retrieved|)` so a short result list is not penalised
/// for positions that do not exist.
pub fn precision_at_k(retrieved: &[u64], relevant: &HashSet<u64>, k: usize) -> f64 {
    let depth = retrieved.len().min(k);
    if depth == 0 {
        return 0.0;
    }
    let hits = retrieved
        .iter()
        .take(k)
        .filter(|id| relevant.contains(id))
        .count();
    hits as f64 / depth as f64
}

/// Reciprocal rank: `1 / rank` of the first relevant item (rank is 1-based), or
/// `0.0` if none of the returned items are relevant.
pub fn reciprocal_rank(retrieved: &[u64], relevant: &HashSet<u64>) -> f64 {
    for (i, id) in retrieved.iter().enumerate() {
        if relevant.contains(id) {
            return 1.0 / (i as f64 + 1.0);
        }
    }
    0.0
}

/// Mean reciprocal rank over several `(ranking, relevant-set)` queries.
pub fn mean_reciprocal_rank(queries: &[(Vec<u64>, HashSet<u64>)]) -> f64 {
    if queries.is_empty() {
        return 0.0;
    }
    let total: f64 = queries
        .iter()
        .map(|(ranking, relevant)| reciprocal_rank(ranking, relevant))
        .sum();
    total / queries.len() as f64
}

/// Average precision for one query: the mean of the precision values taken at
/// each rank where a relevant item occurs, divided by the number of relevant
/// items.
pub fn average_precision(retrieved: &[u64], relevant: &HashSet<u64>) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let mut hits = 0usize;
    let mut sum = 0.0f64;
    for (i, id) in retrieved.iter().enumerate() {
        if relevant.contains(id) {
            hits += 1;
            sum += hits as f64 / (i as f64 + 1.0);
        }
    }
    sum / relevant.len() as f64
}

/// Discounted cumulative gain of the first `k` ranks: `Σ gainᵢ / log₂(rank+1)`.
fn dcg_at_k(gains: impl Iterator<Item = f64>, k: usize) -> f64 {
    gains
        .take(k)
        .enumerate()
        .map(|(i, g)| g / (i as f64 + 2.0).log2())
        .sum()
}

/// nDCG@k with graded relevance `gains` (use `1.0`/`0.0` for binary relevance):
/// the DCG of the returned ranking divided by the ideal DCG (gains sorted
/// descending). Returns `0.0` when there is no positive gain to recover.
pub fn ndcg_at_k(retrieved: &[u64], gains: &HashMap<u64, f64>, k: usize) -> f64 {
    let actual = dcg_at_k(
        retrieved
            .iter()
            .map(|id| gains.get(id).copied().unwrap_or(0.0)),
        k,
    );
    let mut ideal: Vec<f64> = gains.values().copied().filter(|&g| g > 0.0).collect();
    ideal.sort_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));
    let best = dcg_at_k(ideal.into_iter(), k);
    if best <= 0.0 {
        return 0.0;
    }
    actual / best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(ids: &[u64]) -> HashSet<u64> {
        ids.iter().copied().collect()
    }

    #[test]
    fn recall_precision_and_rr_hand_values() {
        // retrieved top-4 = [1,2,3,4]; relevant = {2,4,9}.
        let retrieved = [1, 2, 3, 4];
        let relevant = set(&[2, 4, 9]);
        assert!((recall_at_k(&retrieved, &relevant, 4) - 2.0 / 3.0).abs() < 1e-12);
        assert!((precision_at_k(&retrieved, &relevant, 4) - 0.5).abs() < 1e-12);
        assert!((reciprocal_rank(&retrieved, &relevant) - 0.5).abs() < 1e-12);
        assert_eq!(reciprocal_rank(&retrieved, &set(&[9])), 0.0);
        // Short result lists are not penalised for missing positions.
        assert!((precision_at_k(&[2], &relevant, 4) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn map_and_mrr_hand_values() {
        // Hits at ranks 2 and 4 of 3 relevant: AP = (1/2 + 2/4) / 3.
        let ap = average_precision(&[1, 2, 3, 4], &set(&[2, 4, 9]));
        assert!((ap - (0.5 + 0.5) / 3.0).abs() < 1e-12);
        let mrr = mean_reciprocal_rank(&[
            (vec![1, 2], set(&[2])), // rr = 1/2
            (vec![3, 1], set(&[3])), // rr = 1
            (vec![5, 6], set(&[7])), // rr = 0
        ]);
        assert!((mrr - (0.5 + 1.0) / 3.0).abs() < 1e-12);
    }

    #[test]
    fn ndcg_is_one_for_ideal_order_and_less_otherwise() {
        let gains: HashMap<u64, f64> = [(1u64, 3.0), (2u64, 1.0)].into_iter().collect();
        assert!((ndcg_at_k(&[1, 2, 3], &gains, 3) - 1.0).abs() < 1e-12);
        let swapped = ndcg_at_k(&[2, 1, 3], &gains, 3);
        assert!(swapped < 1.0 && swapped > 0.0);
        assert_eq!(ndcg_at_k(&[1], &HashMap::new(), 3), 0.0);
    }
}
