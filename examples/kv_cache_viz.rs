//! "See your KV-cache" — project SLHAv2-style 128-dim tile latents to 3-D with
//! OctaSoma and emit a JSON for the viewer (`viewer/index.html`).
//!
//! Two modes:
//! ```text
//! cargo run --release --example kv_cache_viz                 # synthetic demo
//! cargo run --release --example kv_cache_viz -- latents.tsv  # your real tiles
//! ```
//!
//! TSV input: one tile per line, `label<TAB>f0 f1 … f127` — the 128 floats from
//! SLHAv2's `SciRustSlhaTile::dequant_latent()`, with a label such as
//! `"head 3 tok 12"`. Produce it from SLHAv2 with a few lines:
//! ```ignore
//! for tile in tiles {
//!     let v = tile.dequant_latent();                  // [f32; 128]
//!     print!("head {} tok {}\t", tile.head_id, tile.token_id);
//!     for x in v { print!("{x} "); }
//!     println!();
//! }
//! ```
//! Then open `viewer/index.html` and drop the emitted JSON: tiles colour by head,
//! and the spatial layout *is* the compressed KV-cache — clusters and outliers
//! become visible.

use std::fs;

use octasoma::{DeterministicRng, FractalMemory3D};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out_path = "kv_cache.json";

    let (labels, vectors) = match args.first() {
        Some(path) => read_tsv(path),
        None => synth(128),
    };
    let n = vectors.len();
    assert!(n > 0, "no tiles to visualise");
    let d = vectors[0].len();

    // PCA so heads / semantic clusters separate in the 3-D layout.
    let calib: Vec<f32> = vectors.iter().flatten().copied().collect();
    let mut mem = FractalMemory3D::new_with_pca(d, &calib, n.min(8_000));
    for (label, v) in labels.iter().zip(&vectors) {
        mem.insert(v, Some(label.as_bytes()));
    }

    fs::write(out_path, mem.export_points_json(200_000)).expect("write json");
    println!(
        "projected {n} tiles ({d}-dim latents) → {out_path}\n\
         open viewer/index.html and drop {out_path} (colour = head; drag to rotate)."
    );
}

fn read_tsv(path: &str) -> (Vec<String>, Vec<Vec<f32>>) {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut labels = Vec::new();
    let mut vectors = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let (label, rest) = line.split_once('\t').unwrap_or(("tile", line));
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
    (labels, vectors)
}

/// Synthetic KV-cache: a few attention "heads", each a cluster of 128-dim tiles.
fn synth(dim: usize) -> (Vec<String>, Vec<Vec<f32>>) {
    let (heads, per_head) = (6usize, 500usize);
    let mut rng = DeterministicRng::new(0x51AA_5EED);
    let centers: Vec<Vec<f32>> = (0..heads)
        .map(|_| unit((0..dim).map(|_| rng.next_f32()).collect()))
        .collect();

    let mut labels = Vec::with_capacity(heads * per_head);
    let mut vectors = Vec::with_capacity(heads * per_head);
    for t in 0..heads * per_head {
        let h = t % heads;
        let v = unit(
            centers[h]
                .iter()
                .map(|&c| c + 0.35 * rng.next_f32())
                .collect(),
        );
        labels.push(format!("head {h} tok {t}"));
        vectors.push(v);
    }
    (labels, vectors)
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
