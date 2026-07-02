//! Temperature scaling of recall confidences — proposal B3 of
//! `docs/scirust-improvements.md`.
//!
//! Adapted from SciRust's multi-class `temperature_scale`
//! (`scirust-core/src/nn/calibration.rs`, same org and dual license) to
//! octasoma's **binary relevance** framing: a recall score `s ∈ (0, 1)` is read
//! as the logit `ln(s / (1 − s))` and rescaled by a fitted temperature —
//! `p = σ(logit(s) / T)` — so `T > 1` softens over-confident scores, `T < 1`
//! sharpens under-confident ones, and `s = 0.5` is a fixed point. `T` is fitted
//! by the same 60-step golden-section search on the (binary) negative
//! log-likelihood, deterministic by construction.
//!
//! The labels come from the **explicit relevance-feedback channel**
//! ([`crate::RelevanceFeedback::score_labels`]): raw cosines from different
//! stores/shards are not comparable — the documented "coarse heuristic" of the
//! cross-region merge — but calibrated probabilities are. Fit one temperature
//! per store (or per shard) and compare *probabilities*, not cosines.

/// Numerically-safe binary logit of a score clamped into `(ε, 1 − ε)`.
fn logit(score: f32) -> f32 {
    let s = score.clamp(1e-6, 1.0 - 1e-6);
    (s / (1.0 - s)).ln()
}

/// The calibrated relevance probability of a raw recall `score` under
/// temperature `t`: `σ(logit(score) / t)`.
pub fn calibrated_probability(score: f32, t: f32) -> f32 {
    let z = logit(score) / t.max(1e-6);
    1.0 / (1.0 + (-z).exp())
}

/// Binary negative log-likelihood of `(score, relevant)` pairs at temperature `t`.
fn nll(pairs: &[(f32, bool)], t: f32) -> f32 {
    let mut total = 0.0f32;
    for &(score, label) in pairs {
        let p = calibrated_probability(score, t).clamp(1e-6, 1.0 - 1e-6);
        total -= if label { p.ln() } else { (1.0 - p).ln() };
    }
    total / pairs.len() as f32
}

/// Fits the temperature minimising the binary NLL over `pairs` — the same
/// 60-step golden-section search on `T ∈ [0.05, 10]` as SciRust's
/// `temperature_scale`. Deterministic. Returns `None` when the pairs cannot
/// identify a temperature (fewer than 2 observations, or a single class —
/// fitting those would silently produce a degenerate `T`).
pub fn fit_temperature(pairs: &[(f32, bool)]) -> Option<f32> {
    let positives = pairs.iter().filter(|(_, l)| *l).count();
    if pairs.len() < 2 || positives == 0 || positives == pairs.len() {
        return None;
    }
    let (mut a, mut b) = (0.05f32, 10.0f32);
    let inv_phi = (5.0f32.sqrt() - 1.0) / 2.0; // 1/φ ≈ 0.618
    let mut c = b - (b - a) * inv_phi;
    let mut d = a + (b - a) * inv_phi;
    let mut fc = nll(pairs, c);
    let mut fd = nll(pairs, d);
    for _ in 0..60 {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - (b - a) * inv_phi;
            fc = nll(pairs, c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + (b - a) * inv_phi;
            fd = nll(pairs, d);
        }
    }
    Some(0.5 * (a + b))
}

/// Expected calibration error of `(score, relevant)` pairs under temperature
/// `t`, with 10 equal-width probability bins: `Σ (|bin| / n) · |accuracy −
/// confidence|`. Lower is better-calibrated; `0` is perfect.
pub fn expected_calibration_error(pairs: &[(f32, bool)], t: f32) -> f32 {
    const BINS: usize = 10;
    if pairs.is_empty() {
        return 0.0;
    }
    let mut count = [0usize; BINS];
    let mut conf = [0.0f32; BINS];
    let mut acc = [0.0f32; BINS];
    for &(score, label) in pairs {
        let p = calibrated_probability(score, t);
        let b = ((p * BINS as f32) as usize).min(BINS - 1);
        count[b] += 1;
        conf[b] += p;
        acc[b] += if label { 1.0 } else { 0.0 };
    }
    let n = pairs.len() as f32;
    (0..BINS)
        .filter(|&b| count[b] > 0)
        .map(|b| {
            let k = count[b] as f32;
            (k / n) * ((acc[b] / k) - (conf[b] / k)).abs()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An over-confident synthetic workload: scores near 1 but only 60%
    /// actually relevant. Fitting must soften (T > 1) and reduce the ECE.
    fn overconfident() -> Vec<(f32, bool)> {
        (0..200)
            .map(|i| {
                let score = 0.9 + 0.0004 * (i % 100) as f32;
                (score, i % 5 < 3) // 60% relevant
            })
            .collect()
    }

    #[test]
    fn fitting_softens_overconfidence_and_reduces_ece() {
        let pairs = overconfident();
        let t = fit_temperature(&pairs).expect("two classes present");
        assert!(t > 1.0, "over-confident scores need T > 1, got {t}");
        let before = expected_calibration_error(&pairs, 1.0);
        let after = expected_calibration_error(&pairs, t);
        assert!(
            after < before,
            "ECE must drop: {before} -> {after} at T = {t}"
        );
        // Calibrated probability lands near the true 60% relevance rate.
        let p = calibrated_probability(0.92, t);
        assert!((p - 0.6).abs() < 0.15, "calibrated p = {p}");
        // Deterministic.
        assert_eq!(fit_temperature(&pairs), Some(t));
    }

    #[test]
    fn degenerate_inputs_refuse_to_fit() {
        assert_eq!(fit_temperature(&[]), None);
        assert_eq!(fit_temperature(&[(0.9, true)]), None);
        assert_eq!(fit_temperature(&[(0.9, true), (0.8, true)]), None); // one class
        assert_eq!(fit_temperature(&[(0.2, false), (0.3, false)]), None);
    }

    #[test]
    fn temperature_keeps_the_half_point_and_orders_scores() {
        for t in [0.5f32, 1.0, 3.0] {
            assert!((calibrated_probability(0.5, t) - 0.5).abs() < 1e-6);
            assert!(calibrated_probability(0.9, t) > calibrated_probability(0.6, t));
        }
    }
}
