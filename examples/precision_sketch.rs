//! Precision: 3-D projection vs. SimHash sketch vs. exact cosine.
//!
//! On the *same* clustered embeddings, how well does each cheap method recover the
//! true (full-dimensional) nearest neighbour? We report **recall@M** — is the exact
//! nearest within a method's M cheapest candidates? — because that is how a cheap
//! filter is actually used: produce a shortlist, then rerank it exactly.
//!
//! - **3-D PCA** — OctaSoma's coarse spatial router (rank by distance on 3 coords).
//! - **SimHash-`B`** — rank by Hamming distance on a `B`-bit sketch of the *full*
//!   embedding (popcount; safe, stable).
//! - **hybrid** — SimHash top-32 shortlist, then an exact cosine rerank (recall@1).
//! - **exact cosine** — brute force over full embeddings (the reference ranker).
//!
//! Run: `cargo run --release --example precision_sketch -- [N] [D] [C] [Q] [BITS]`

use std::time::Instant;

use octasoma::{DeterministicRng, FractalMemory3D, SimHasher, hamming};

fn main() {
    let a: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|s| s.parse().ok())
        .collect();
    let n = a.first().copied().unwrap_or(20_000);
    let d = a.get(1).copied().unwrap_or(768);
    let c = a.get(2).copied().unwrap_or(16);
    let q = a.get(3).copied().unwrap_or(500);
    let bits = a.get(4).copied().unwrap_or(256);

    eprintln!("[i] generating {n} unit embeddings in R^{d} from {c} themes …");
    let (emb, label) = gen_clustered(n, d, c, 0xC0FFEE);

    let calib: Vec<f32> = emb[..n.min(8_000)].iter().flatten().copied().collect();
    let mut tree = FractalMemory3D::new_with_pca(d, &calib, n.min(8_000));
    for v in &emb {
        tree.insert(v, None);
    }
    let pts: Vec<[f32; 3]> = tree.items.iter().map(|it| it.point).collect();

    let hasher = SimHasher::new(d, bits, 0x5117);
    let sketches: Vec<Vec<u64>> = emb.iter().map(|v| hasher.sketch(v)).collect();

    let centers = cluster_centers(d, c, 0xC0FFEE);
    let mut rng = DeterministicRng::new(0xDEC0DE);

    // recall@{1,10,32} via the true neighbour's rank in each method's ordering.
    let mut r3 = [0usize; 4];
    let mut rk = [0usize; 4];
    let (mut clus3, mut clusk, mut clusx, mut hybrid1) = (0usize, 0usize, 0usize, 0usize);
    let (mut us3, mut usk, mut usx, mut ushy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let marks = [1usize, 32, 128, 512];
    let shortlist = a.get(5).copied().unwrap_or(256);

    for _ in 0..q {
        let theme = (rng.next_u64() as usize) % c;
        let query = unit(noisy(&centers[theme], 0.9, &mut rng));

        // Reference: exact nearest by cosine (unit vectors → max dot).
        let t = Instant::now();
        let truth = exact_argmax(&emb, &query);
        usx += t.elapsed().as_secs_f64() * 1e6;
        if label[truth] == theme {
            clusx += 1;
        }

        // 3-D: the true neighbour's rank by squared distance on 3 coords.
        let t = Instant::now();
        let pq = tree.project(&query).unwrap();
        let dt = d2(pq, pts[truth]);
        let mut rank = 0usize;
        let mut top = (f32::INFINITY, 0usize);
        for (i, &p) in pts.iter().enumerate() {
            let di = d2(pq, p);
            if di < dt {
                rank += 1;
            }
            if di < top.0 {
                top = (di, i);
            }
        }
        us3 += t.elapsed().as_secs_f64() * 1e6;
        for (m, &mark) in marks.iter().enumerate() {
            if rank < mark {
                r3[m] += 1;
            }
        }
        if label[top.1] == theme {
            clus3 += 1;
        }

        // SimHash: the true neighbour's rank by Hamming, and a top-`shortlist`.
        let t = Instant::now();
        let qs = hasher.sketch(&query);
        let ht = hamming(&qs, &sketches[truth]);
        let mut rank = 0usize;
        let mut cand: Vec<(u32, usize)> = Vec::with_capacity(n);
        for (i, s) in sketches.iter().enumerate() {
            let h = hamming(&qs, s);
            if h < ht {
                rank += 1;
            }
            cand.push((h, i));
        }
        usk += t.elapsed().as_secs_f64() * 1e6;
        for (m, &mark) in marks.iter().enumerate() {
            if rank < mark {
                rk[m] += 1;
            }
        }
        // cluster@1 = label of the single smallest-Hamming item.
        let topk = cand
            .iter()
            .min_by_key(|(h, _)| *h)
            .map(|(_, i)| *i)
            .unwrap();
        if label[topk] == theme {
            clusk += 1;
        }

        // Hybrid: SimHash top-`shortlist` → exact cosine rerank → recall@1.
        let t = Instant::now();
        if cand.len() > shortlist {
            cand.select_nth_unstable_by_key(shortlist, |(h, _)| *h);
            cand.truncate(shortlist);
        }
        let mut best = (f32::NEG_INFINITY, 0usize);
        for &(_, i) in &cand {
            let dot: f32 = emb[i].iter().zip(&query).map(|(x, y)| x * y).sum();
            if dot > best.0 {
                best = (dot, i);
            }
        }
        ushy += t.elapsed().as_secs_f64() * 1e6; // rerank only; sketch-scan (usk) is shared
        if best.1 == truth {
            hybrid1 += 1;
        }
    }

    let qf = q as f64;
    let p = |x: usize| x as f64 / qf * 100.0;
    println!(
        "\nPrecision on clustered embeddings  (N={n}, D={d}, themes={c}, queries={q}, sketch={bits}-bit)\n"
    );
    println!(
        "{:<24} {:>8} {:>9} {:>10} {:>10} {:>10} {:>10}",
        "method", "recall@1", "recall@32", "recall@128", "recall@512", "cluster@1", "µs/query"
    );
    println!("{}", "-".repeat(86));
    println!(
        "{:<24} {:>7.1}% {:>8.1}% {:>9.1}% {:>9.1}% {:>9.1}% {:>10.1}",
        "3-D PCA (octree)",
        p(r3[0]),
        p(r3[1]),
        p(r3[2]),
        p(r3[3]),
        p(clus3),
        us3 / qf
    );
    println!(
        "{:<24} {:>7.1}% {:>8.1}% {:>9.1}% {:>9.1}% {:>9.1}% {:>10.1}",
        format!("SimHash-{bits} (Hamming)"),
        p(rk[0]),
        p(rk[1]),
        p(rk[2]),
        p(rk[3]),
        p(clusk),
        usk / qf
    );
    println!(
        "{:<24} {:>7.1}% {:>8} {:>10} {:>10} {:>9.1}% {:>10.1}",
        format!("  └ hybrid (rerank {shortlist})"),
        p(hybrid1),
        "—",
        "—",
        "—",
        p(clusk),
        (usk + ushy) / qf
    );
    println!(
        "{:<24} {:>7.1}% {:>8} {:>10} {:>10} {:>9.1}% {:>10.1}",
        "exact cosine (brute)",
        100.0,
        "—",
        "—",
        "—",
        p(clusx),
        usx / qf
    );
    println!(
        "\nsketch storage: {} KiB total ({} bytes/item)\n\n\
         Reading: the 3-D router barely ranks the true neighbour near the top; a {bits}-bit\n\
         SimHash does, so a SimHash shortlist + an exact rerank (hybrid) recovers most of\n\
         the precision the projection threw away — at a fraction of brute-force cost, in\n\
         100% safe stable Rust. This is the high-precision tier to add, not a Hilbert/SIMD\n\
         rewrite of the 3-D index.",
        n * bits / 8 / 1024,
        bits / 8
    );
}

fn gen_clustered(n: usize, d: usize, c: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<usize>) {
    let centers = cluster_centers(d, c, seed);
    let mut rng = DeterministicRng::new(seed ^ 0xA5A5);
    let mut emb = Vec::with_capacity(n);
    let mut label = Vec::with_capacity(n);
    for i in 0..n {
        let k = i % c;
        emb.push(unit(noisy(&centers[k], 0.9, &mut rng)));
        label.push(k);
    }
    (emb, label)
}

fn cluster_centers(d: usize, c: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = DeterministicRng::new(seed);
    (0..c)
        .map(|_| unit((0..d).map(|_| rng.next_f32()).collect()))
        .collect()
}

fn noisy(center: &[f32], spread: f32, rng: &mut DeterministicRng) -> Vec<f32> {
    // Scale by 1/sqrt(d) so `spread` is the noise norm relative to the unit center,
    // independent of dimension (otherwise high-D noise swamps the signal).
    let s = spread / (center.len() as f32).sqrt();
    center.iter().map(|&x| x + s * rng.next_f32()).collect()
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

fn exact_argmax(emb: &[Vec<f32>], q: &[f32]) -> usize {
    let mut best = (f32::NEG_INFINITY, 0usize);
    for (i, v) in emb.iter().enumerate() {
        let dot: f32 = v.iter().zip(q).map(|(a, b)| a * b).sum();
        if dot > best.0 {
            best = (dot, i);
        }
    }
    best.1
}

#[inline]
fn d2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    dx * dx + dy * dy + dz * dz
}
