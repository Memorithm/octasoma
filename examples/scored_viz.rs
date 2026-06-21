//! "See your query" — heat-colour a memory by how close each item is to a query.
//!
//! Where [`kv_cache_viz`](kv_cache_viz.rs) colours points by *category*, this
//! colours them by *precision score*: the exact cosine similarity of every stored
//! memory to a query vector. Cold blue = unrelated, hot red = on-query. The hot
//! region is the answer to "what does this query actually retrieve", made visible.
//!
//! ```text
//! cargo run --release --example scored_viz                  # synthetic demo
//! cargo run --release --example scored_viz -- latents.tsv   # your real vectors
//! ```
//!
//! TSV input: one vector per line, `label<TAB>f0 f1 …`. The **first** line is used
//! as the query (and is also indexed, so it shows up scored ~1.0); the rest are the
//! memory. Then open `viewer/index.html` and drop the emitted `scored.json`.

use std::fs;

use octasoma::{DeterministicRng, HybridMemory, QueryStrategy};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out_path = "scored.json";
    let bits = 256;
    let seed = 0x05C0_4EEDu64;

    let (labels, vectors, query) = match args.first() {
        Some(path) => read_tsv(path),
        None => synth(128),
    };
    let n = vectors.len();
    assert!(n > 0, "no vectors to visualise");
    let d = vectors[0].len();

    // PCA so clusters separate in the 3-D layout; the score colouring is exact
    // (full-dim cosine), independent of the projection.
    let calib: Vec<f32> = vectors.iter().flatten().copied().collect();
    let mut mem = HybridMemory::new_with_pca(d, &calib, n.min(8_000), bits, seed);
    for (label, v) in labels.iter().zip(&vectors) {
        mem.insert(v, label.as_bytes());
    }

    fs::write(out_path, mem.export_scored_json(&query, 200_000)).expect("write json");

    // Show the precise top-5 so the heat map has a ground truth to compare against.
    println!("indexed {n} vectors ({d}-dim) → {out_path}");
    println!("top matches for the query (exact cosine rerank):");
    for (payload, score) in mem.query(&query, QueryStrategy::PrecisionSketch, 5) {
        println!("  {score:+.4}  {}", String::from_utf8_lossy(payload));
    }
    println!("open viewer/index.html and drop {out_path} (colour = cosine to query).");
}

/// Reads `label<TAB>floats` lines; the first vector becomes the query.
fn read_tsv(path: &str) -> (Vec<String>, Vec<Vec<f32>>, Vec<f32>) {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut labels = Vec::new();
    let mut vectors = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let (label, rest) = line.split_once('\t').unwrap_or(("vec", line));
        let v: Vec<f32> = rest
            .split_whitespace()
            .filter_map(|t| t.parse().ok())
            .collect();
        if !v.is_empty() {
            labels.push(label.to_string());
            vectors.push(v);
        }
    }
    assert!(!vectors.is_empty(), "no numeric vectors parsed from {path}");
    let query = vectors[0].clone();
    (labels, vectors, query)
}

/// Synthetic memory: a few semantic "themes", each a cluster. The query is the
/// first theme's centre, so one cluster lights up hot and the rest stay cold.
fn synth(dim: usize) -> (Vec<String>, Vec<Vec<f32>>, Vec<f32>) {
    let (themes, per_theme) = (6usize, 500usize);
    let mut rng = DeterministicRng::new(0x05C0_4EED);
    let centers: Vec<Vec<f32>> = (0..themes)
        .map(|_| unit((0..dim).map(|_| rng.next_f32() - 0.5).collect()))
        .collect();

    let mut labels = Vec::with_capacity(themes * per_theme);
    let mut vectors = Vec::with_capacity(themes * per_theme);
    for t in 0..themes * per_theme {
        let h = t % themes;
        let v = unit(
            centers[h]
                .iter()
                .map(|&c| c + 0.30 * (rng.next_f32() - 0.5))
                .collect(),
        );
        labels.push(format!("theme {h} item {t}"));
        vectors.push(v);
    }
    let query = centers[0].clone(); // light up theme 0
    (labels, vectors, query)
}

fn unit(mut v: Vec<f32>) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-12 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
    v
}
