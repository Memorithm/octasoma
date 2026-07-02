//! Distribution-free risk control for the precision tier — the math behind
//! [`SketchIndex::certify_shortlist`](crate::SketchIndex::certify_shortlist).
//!
//! `hoeffding_ucb` and `rcps_select` are vendored from SciRust
//! (`scirust-core/src/nn/conformal.rs`, same org and dual license), ported to
//! `f64` — ~40 lines of pure, deterministic math, so octasoma's dependency
//! budget stays at one crate. RCPS is Bates et al., *Distribution-Free,
//! Risk-Controlling Prediction Sets* (2021).
//!
//! The point: `docs/precision-sketch.md` reports the shortlist/recall trade-off
//! empirically ("recall@512 = 88.7 %"). RCPS upgrades that to a **certificate**
//! — *on corpora exchangeable with the calibration queries, the expected recall
//! loss of the chosen shortlist is ≤ α, with probability ≥ 1 − δ over the draw
//! of the calibration set*. The caveat is part of the guarantee: query drift
//! (new topics, a different embedder) voids it, so re-calibrate when the
//! workload changes.

/// **Hoeffding** upper confidence bound on the mean of `n` i.i.d. losses in
/// `[0, 1]` with empirical mean `mean`, at confidence `1 − delta`:
/// `mean + √(ln(1/δ) / (2n))`. The bound holds with probability `≥ 1 − delta`.
/// `n == 0` returns `+∞` (no data certifies nothing).
///
/// # Panics
/// If `delta` is outside `(0, 1)`.
pub fn hoeffding_ucb(mean: f64, n: usize, delta: f64) -> f64 {
    assert!(delta > 0.0 && delta < 1.0, "delta must be in (0,1)");
    if n == 0 {
        return f64::INFINITY;
    }
    mean + ((1.0 / delta).ln() / (2.0 * n as f64)).sqrt()
}

/// **RCPS** — Risk-Controlling Prediction Sets (Bates et al., 2021). Given a
/// family of predictors whose risk is **non-increasing** in the index (here:
/// nested shortlists — a larger shortlist can only improve recall), `risks[i]`
/// is the empirical risk of candidate `i` on a calibration set of size `n`.
/// Returns the index of the **smallest** candidate such that the Hoeffding
/// upper bound on the risk is `≤ alpha` for it **and every larger** candidate —
/// guaranteeing true risk `≤ alpha` with probability `≥ 1 − delta`. `None` if
/// no candidate controls the risk (calibration set too small for the asked
/// `(alpha, delta)`, or the risk genuinely too high).
///
/// The right-to-left scan stops at the first violation, which keeps the
/// guarantee rigorous even if the empirical risks are not perfectly monotone
/// (sampling noise, exact-cosine ties).
pub fn rcps_select(risks: &[f64], n: usize, alpha: f64, delta: f64) -> Option<usize> {
    let mut hat = None;
    for i in (0..risks.len()).rev() {
        if hoeffding_ucb(risks[i], n, delta) <= alpha {
            hat = Some(i);
        } else {
            break;
        }
    }
    hat
}

/// **Split-conformal quantile** with the finite-sample correction (vendored from
/// `scirust-core/src/nn/conformal.rs`, ported to `f64`): the `⌈(n+1)(1−α)⌉`-th
/// smallest calibration score. Calibrating on nonconformity scores (here:
/// `1 − similarity` of confirmed-relevant recalls, see
/// [`crate::RelevanceFeedback::nonconformity`]) makes any set built as
/// "everything with nonconformity `≤ q̂`" cover the relevant item with
/// probability `≥ 1 − α` — distribution-free, assuming exchangeability with the
/// calibration workload. Returns `+∞` when the calibration set is too small for
/// the asked `alpha` (no guarantee is possible — never a fake radius).
///
/// # Panics
/// If `alpha` is outside `(0, 1)`.
pub fn conformal_quantile(scores: &[f64], alpha: f64) -> f64 {
    assert!(alpha > 0.0 && alpha < 1.0, "alpha must be in (0,1)");
    let n = scores.len();
    if n == 0 {
        return f64::INFINITY;
    }
    let mut s = scores.to_vec();
    s.sort_by(f64::total_cmp);
    let k = (((n + 1) as f64) * (1.0 - alpha)).ceil() as usize; // 1-indexed rank
    if (1..=n).contains(&k) {
        s[k - 1]
    } else {
        f64::INFINITY
    }
}

/// A certified shortlist size for the SimHash precision tier — the output of
/// [`SketchIndex::certify_shortlist`](crate::SketchIndex::certify_shortlist).
///
/// Reads as: *querying with `shortlist` keeps the expected recall loss at `k`
/// (`1 − recall@k` against the exact full-corpus rerank) at or below `alpha`,
/// with probability `≥ 1 − delta` over the draw of the `calibration_n`
/// calibration queries — for workloads exchangeable with those queries.*
#[derive(Debug, Clone, PartialEq)]
pub struct ShortlistCertificate {
    /// The smallest certified shortlist — feed it to
    /// [`SketchIndex::nearest`](crate::SketchIndex::nearest) or
    /// [`HybridMemory::with_shortlist`](crate::HybridMemory::with_shortlist).
    pub shortlist: usize,
    /// The `k` of the certified `recall@k`.
    pub k: usize,
    /// The certified risk level: expected recall loss `≤ alpha`.
    pub alpha: f64,
    /// The confidence: the guarantee holds with probability `≥ 1 − delta`.
    pub delta: f64,
    /// Calibration queries actually used (dimension-mismatched ones are dropped).
    pub calibration_n: usize,
    /// Empirical mean recall loss of `shortlist` on the calibration set.
    pub empirical_risk: f64,
    /// The Hoeffding upper bound that certified it (`≤ alpha` by construction).
    pub risk_ucb: f64,
    /// The candidate grid that was swept (doubling from `k` to the corpus size).
    pub grid: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hoeffding_ucb_shrinks_with_n_and_is_infinite_without_data() {
        assert!(hoeffding_ucb(0.1, 0, 0.1).is_infinite());
        let loose = hoeffding_ucb(0.1, 10, 0.1);
        let tight = hoeffding_ucb(0.1, 1000, 0.1);
        assert!(tight < loose && tight > 0.1);
        // Exact value: 0.1 + sqrt(ln(10) / 2000).
        let expect = 0.1 + (10.0f64.ln() / 2000.0).sqrt();
        assert!((tight - expect).abs() < 1e-12);
    }

    #[test]
    fn conformal_quantile_is_finite_sample_correct() {
        // n=4, alpha=0.5 → k = ceil(5·0.5) = 3 → 3rd smallest = 0.3.
        let q = conformal_quantile(&[0.4, 0.1, 0.3, 0.2], 0.5);
        assert!((q - 0.3).abs() < 1e-12, "q = {q}");
        // Too few calibration points for the asked coverage → +∞, never fake.
        assert!(conformal_quantile(&[0.5, 0.7], 0.1).is_infinite());
        assert!(conformal_quantile(&[], 0.5).is_infinite());
    }

    #[test]
    fn rcps_picks_the_smallest_certified_candidate_and_stops_at_a_violation() {
        // n large enough that the UCB slack is ~0.048 at delta=0.1.
        let n = 1000;
        // Monotone risks; alpha=0.2 certifies indices 2 and 3 only.
        assert_eq!(rcps_select(&[0.5, 0.3, 0.1, 0.0], n, 0.2, 0.1), Some(2));
        // A violation in the middle stops the scan: index 1's UCB > alpha, so
        // index 0 is never reached even though its risk is low.
        assert_eq!(rcps_select(&[0.0, 0.9, 0.0], n, 0.2, 0.1), Some(2));
        // Nothing certifies.
        assert_eq!(rcps_select(&[0.5, 0.4], n, 0.2, 0.1), None);
        // Too little data certifies nothing, whatever the risks.
        assert_eq!(rcps_select(&[0.0, 0.0], 3, 0.05, 0.05), None);
    }
}
