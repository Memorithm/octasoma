//! OctaSoma as a **pre-filter for SLHAv2 attention** (step 3 of the pyramid).
//!
//! SLHAv2 scores every KV-cache tile against the query (`compute_score`). At long
//! context that is a lot of tiles. OctaSoma's cheap 3-D index routes attention to
//! `M` candidate tiles; an exact rerank (here a dot-product stand-in for SLHAv2's
//! `compute_score`) keeps the top-`k`. We measure how much attention work is saved
//! and how much of the true top-`k` survives.
//!
//! SYNTHETIC tiles (128-dim, like `SciRustSlhaTile::dequant_latent()`); SLHAv2's
//! real kernel replaces the rerank. Run: `cargo run --release --example slha_prefilter`

use std::time::Instant;

use octasoma::{DeterministicRng, FractalMemory3D};

fn main() {
    let (n, d) = (20_000usize, 128usize); // N tiles, 128-dim latents (SLHAv2 D_C)
    let (heads, q, m, k) = (8usize, 500usize, 200usize, 10usize);

    let mut rng = DeterministicRng::new(0x517A);
    let centers: Vec<Vec<f32>> = (0..heads).map(|_| unit(rand_vec(&mut rng, d))).collect();
    let tiles: Vec<Vec<f32>> = (0..n)
        .map(|i| unit(noisy(&centers[i % heads], 0.4, &mut rng)))
        .collect();

    // OctaSoma 3-D index over the tile latents.
    let calib: Vec<f32> = tiles[..n.min(4_000)].iter().flatten().copied().collect();
    let mut mem = FractalMemory3D::new_with_pca(d, &calib, n.min(4_000));
    for (i, t) in tiles.iter().enumerate() {
        mem.insert(t, Some(&(i as u32).to_le_bytes()));
    }

    // Attention ∝ dot product (stand-in for SLHAv2's compute_score).
    let score = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };

    let (mut full_us, mut pf_us, mut recall) = (0.0f64, 0.0f64, 0.0f64);
    let mut qrng = DeterministicRng::new(0x999);
    for _ in 0..q {
        let g = (qrng.next_u64() as usize) % n;
        let query = unit(noisy(&tiles[g], 0.1, &mut qrng));

        // Ground truth: exact attention top-k over ALL tiles.
        let t = Instant::now();
        let truth = topk_by_score(&tiles, &query, k, &score, None);
        full_us += t.elapsed().as_secs_f64() * 1e6;

        // Pre-filter: OctaSoma proposes M candidates → exact rerank to top-k.
        let t = Instant::now();
        let cand: Vec<usize> = mem
            .nearest_embedding(&query, m)
            .into_iter()
            .map(|(id, _)| id as usize)
            .collect();
        let pf = topk_by_score(&tiles, &query, k, &score, Some(&cand));
        pf_us += t.elapsed().as_secs_f64() * 1e6;

        let overlap = pf.iter().filter(|x| truth.contains(x)).count();
        recall += overlap as f64 / k as f64;
    }

    println!("SLHAv2 attention pre-filter via OctaSoma — SYNTHETIC tiles ({d}-dim, N={n})\n");
    println!(
        "tiles scored / query:   full = {n}   pre-filter = {m}   → {}× less attention work",
        n / m
    );
    println!(
        "top-{k} recall (pre-filter vs exact): {:.1}%",
        recall / q as f64 * 100.0
    );
    println!(
        "latency / query:        full {:.0} µs   pre-filter {:.0} µs   → {:.0}× faster",
        full_us / q as f64,
        pf_us / q as f64,
        full_us / pf_us
    );
    println!(
        "\nHonest read: the 3-D pre-filter recovers only a fraction of the *exact* top-k\n\
         (3-D is a coarse router), so OctaSoma is weak for precise attention selection.\n\
         Its SLHAv2 value is therefore visualization (the viewer) and coarse diversity/\n\
         eviction — not exact attention routing, which is SLHAv2's own compute_score job."
    );
}

fn topk_by_score(
    tiles: &[Vec<f32>],
    query: &[f32],
    k: usize,
    score: &impl Fn(&[f32], &[f32]) -> f32,
    subset: Option<&[usize]>,
) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = match subset {
        Some(ids) => ids.iter().map(|&i| (i, score(&tiles[i], query))).collect(),
        None => tiles
            .iter()
            .enumerate()
            .map(|(i, t)| (i, score(t, query)))
            .collect(),
    };
    scored.sort_by(|a, b| b.1.total_cmp(&a.1)); // descending: higher dot = more attention
    scored.iter().take(k).map(|x| x.0).collect()
}

fn rand_vec(rng: &mut DeterministicRng, d: usize) -> Vec<f32> {
    (0..d).map(|_| rng.next_f32()).collect()
}

fn noisy(center: &[f32], spread: f32, rng: &mut DeterministicRng) -> Vec<f32> {
    center
        .iter()
        .map(|&c| c + spread * rng.next_f32())
        .collect()
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
