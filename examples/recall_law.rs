//! A closed-form recall law for the SimHash tier — proposal C4 of
//! `docs/scirust-improvements.md`.
//!
//! Sweeps the precision tier over a `(bits, shortlist)` grid (the D1 metrics
//! measure each point), then asks SciRust's symbolic-regression engine (a
//! dev-dependency) for a **Pareto front of formulas** `recall ≈ f(bits,
//! shortlist)` — an interpretable law that sizes a store from its parameters
//! instead of a lookup table, in the project's explainability spirit.
//!
//! The honest caveat, printed with the result: a fitted law describes the swept
//! grid on THIS corpus family; extrapolating outside it (or to a different
//! embedder) is unsafe — re-sweep and re-fit per corpus family.
//!
//! Run with: `cargo run --release --example recall_law`

use octasoma::{SketchIndex, metrics};
use scirust_symreg::discover;
use std::collections::HashSet;

const DIM: usize = 64;
const N: usize = 2000;
const CLUSTERS: usize = 40;
const K: usize = 10;
const QUERIES: usize = 32;

fn main() {
    // Deterministic corpus + fresh queries (LCG noise — see the recall gate for
    // why the queries must not be perturbed copies).
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut noise = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };
    let mut items = Vec::with_capacity(N);
    for c in 0..CLUSTERS {
        let base: Vec<f32> = (0..DIM)
            .map(|d| ((c * DIM + d) as f32 * 0.61).sin())
            .collect();
        for _ in 0..N / CLUSTERS {
            items.push(base.iter().map(|x| x + 0.3 * noise()).collect::<Vec<f32>>());
        }
    }
    let queries: Vec<Vec<f32>> = (0..QUERIES)
        .map(|_| (0..DIM).map(|_| noise()).collect())
        .collect();

    // The sweep: recall@10 vs the exact oracle at every grid point.
    let bits_grid = [64usize, 128, 256, 512, 1024];
    let shortlist_grid = [16usize, 32, 64, 128, 256, 512];
    println!(
        "[i] sweeping {}x{} grid over {N} items…",
        bits_grid.len(),
        shortlist_grid.len()
    );
    let mut data: Vec<(Vec<f64>, f64)> = Vec::new();
    for &bits in &bits_grid {
        let mut idx = SketchIndex::new(DIM, bits, 42);
        for (i, item) in items.iter().enumerate() {
            idx.insert(item, &(i as u64).to_le_bytes());
        }
        let oracle: Vec<HashSet<u64>> = queries
            .iter()
            .map(|q| {
                idx.nearest(q, K, N)
                    .into_iter()
                    .map(|(p, _)| u64::from_le_bytes(p.try_into().unwrap()))
                    .collect()
            })
            .collect();
        for &shortlist in &shortlist_grid {
            let mut recall = 0.0;
            for (q, oracle_ids) in queries.iter().zip(&oracle) {
                let got: Vec<u64> = idx
                    .nearest(q, K, shortlist)
                    .into_iter()
                    .map(|(p, _)| u64::from_le_bytes(p.try_into().unwrap()))
                    .collect();
                recall += metrics::recall_at_k(&got, oracle_ids, K);
            }
            recall /= QUERIES as f64;
            println!("[i] bits={bits:<5} shortlist={shortlist:<4} recall@{K}={recall:.4}");
            data.push((vec![bits as f64, shortlist as f64], recall));
        }
    }

    // Symbolic regression: a Pareto front (formula size vs fit) of recall laws.
    println!("\n[i] fitting recall ≈ f(bits, shortlist)…");
    let front = discover(&data, &["bits", "shortlist"], &[1, 2, 3], 200, 22, 35, 25);
    println!("\nPareto front (formula size vs mean-squared error):");
    println!("  {:>6} {:>11}   formula", "size", "mse");
    for (size, mse, expr) in &front {
        println!("  {size:>6} {mse:>11.2e}   {expr}");
    }
    println!(
        "\n[!] validity domain: bits ∈ [64, 1024], shortlist ∈ [16, 512], THIS corpus\n\
         [!] family (40 clusters, 64-d, N=2000). A fitted law extrapolates unsafely —\n\
         [!] re-sweep and re-fit per corpus family, then verify a chosen operating\n\
         [!] point with SketchIndex::certify_shortlist for an actual guarantee."
    );
}
