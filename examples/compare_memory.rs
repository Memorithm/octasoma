//! Confronting OctaSoma's 3-D memory against other memory regimes.
//!
//! Offline, deterministic, dependency-free. On one synthetic clustered dataset we
//! measure, for each "memory type", the axes the literature uses to compare memory
//! systems: cluster (topical) recall@1, exact recall@1, query latency, and
//! coordinate footprint. Then we demonstrate the niche the comparison points to —
//! OctaSoma as a cheap **pre-filter** ahead of an exact full-D reranker.
//!
//! Run: `cargo run --release --example compare_memory -- [N D clusters queries M]`

use std::time::Instant;

use octasoma::{DeterministicRng, FractalMemory3D};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let n = arg(&a, 0, 20_000);
    let d = arg(&a, 1, 256);
    let clusters = arg(&a, 2, 16);
    let q = arg(&a, 3, 1_000);
    let m = arg(&a, 4, 50); // pre-filter candidate count

    println!("OctaSoma — comparison vs other memory regimes");
    println!("dataset: N={n}  D={d}  clusters={clusters}  queries={q}  (prefilter M={m})\n");

    // ---- synthetic clustered embeddings with known ground-truth clusters ----
    let mut rng = DeterministicRng::new(0xC0FFEE);
    let centers: Vec<Vec<f32>> = (0..clusters).map(|_| unit(rand_vec(&mut rng, d))).collect();
    let mut data: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut label: Vec<u32> = Vec::with_capacity(n);
    for i in 0..n {
        let c = i % clusters;
        data.push(unit(noisy(&centers[c], 0.35, &mut rng)));
        label.push(c as u32);
    }
    let mut qv: Vec<Vec<f32>> = Vec::with_capacity(q);
    let mut ql: Vec<u32> = Vec::with_capacity(q);
    for _ in 0..q {
        let c = (rng.next_u64() as usize) % clusters;
        qv.push(unit(noisy(&centers[c], 0.35, &mut rng)));
        ql.push(c as u32);
    }

    // ---- (1) full-D exact store: the recall ceiling (a FAISS-Flat stand-in) ----
    let t = Instant::now();
    let truth: Vec<u32> = qv.iter().map(|query| full_nn(&data, query)).collect();
    let fulld_lat = us_per(t.elapsed().as_secs_f64(), q);
    let fulld_cluster = pct(
        (0..q)
            .filter(|&i| label[truth[i] as usize] == ql[i])
            .count(),
        q,
    );

    // ---- (2) OctaSoma PCA-3D and (3) JL-3D ----
    let calib = n.min(4_000);
    let flat: Vec<f32> = data[..calib].iter().flatten().copied().collect();
    let mut pca = FractalMemory3D::new_with_pca(d, &flat, calib);
    let mut jl = FractalMemory3D::new(d, 7);
    let t = Instant::now();
    for (i, v) in data.iter().enumerate() {
        pca.insert(v, Some(&label[i].to_le_bytes()));
    }
    let pca_build = (n as f64 / t.elapsed().as_secs_f64()) as u64;
    for (i, v) in data.iter().enumerate() {
        jl.insert(v, Some(&label[i].to_le_bytes()));
    }

    // ---- (4) OctaSoma supervised-3D: PCA on the class CENTROIDS ----
    // A cheap, label-aware (LDA-like) projection: the top-3 directions of
    // between-theme variance. Reuses the PCA path on the centroid matrix.
    let mut cent = vec![0f64; clusters * d];
    let mut cnt = vec![0usize; clusters];
    for (i, v) in data.iter().enumerate() {
        let c = label[i] as usize;
        cnt[c] += 1;
        for (slot, &x) in cent[c * d..c * d + d].iter_mut().zip(v.iter()) {
            *slot += x as f64;
        }
    }
    for (chunk, &count) in cent.chunks_mut(d).zip(cnt.iter()) {
        if count > 0 {
            for slot in chunk.iter_mut() {
                *slot /= count as f64;
            }
        }
    }
    let cent_f: Vec<f32> = cent.iter().map(|&x| x as f32).collect();
    let mut sup = FractalMemory3D::new_with_pca(d, &cent_f, clusters);
    for (i, v) in data.iter().enumerate() {
        sup.insert(v, Some(&label[i].to_le_bytes()));
    }

    let eval = |mem: &FractalMemory3D| -> (f64, f64, f64) {
        let t = Instant::now();
        let (mut cl, mut ex) = (0usize, 0usize);
        for (qi, query) in qv.iter().enumerate() {
            if let Some(&(id, _)) = mem.nearest_embedding(query, 1).first() {
                if label[id as usize] == ql[qi] {
                    cl += 1;
                }
                if id == truth[qi] {
                    ex += 1;
                }
            }
        }
        (pct(cl, q), pct(ex, q), us_per(t.elapsed().as_secs_f64(), q))
    };
    let (pca_cl, pca_ex, pca_lat) = eval(&pca);
    let (jl_cl, jl_ex, jl_lat) = eval(&jl);
    let (sup_cl, sup_ex, sup_lat) = eval(&sup);

    // ---- report: the comparison table ----
    let coord_full = 4 * d; // D float32 coordinates per item
    let coord_octa = 12; // 3 float32 coordinates per item
    println!(
        "{:<22} {:>10} {:>9} {:>12} {:>14}",
        "memory type", "cluster@1", "exact@1", "latency µs", "coord B/item"
    );
    println!("{}", "-".repeat(70));
    println!(
        "{:<22} {:>9.1}% {:>8.1}% {:>12.2} {:>14}",
        "full-D exact (flat)", fulld_cluster, 100.0, fulld_lat, coord_full
    );
    println!(
        "{:<22} {:>9.1}% {:>8.1}% {:>12.2} {:>14}",
        "OctaSoma PCA-3D", pca_cl, pca_ex, pca_lat, coord_octa
    );
    println!(
        "{:<22} {:>9.1}% {:>8.1}% {:>12.2} {:>14}",
        "OctaSoma JL-3D", jl_cl, jl_ex, jl_lat, coord_octa
    );
    println!(
        "{:<22} {:>9.1}% {:>8.1}% {:>12.2} {:>14}",
        "OctaSoma supervised-3D", sup_cl, sup_ex, sup_lat, coord_octa
    );
    println!(
        "\ncoordinate footprint: OctaSoma stores 3 floats/item vs {d} → {}× smaller; \
         PCA-3D inserts ≈{pca_build}/s; octree nodes={}",
        coord_full / coord_octa,
        pca.node_count()
    );

    // ---- the useful niche: OctaSoma as a Stage-0 pre-filter + full-D rerank ----
    let t = Instant::now();
    let (mut pf_cl, mut pf_ex) = (0usize, 0usize);
    for (qi, query) in qv.iter().enumerate() {
        // 3-D octree proposes M candidates; exact full-D distance reranks them.
        let mut best = (u32::MAX, f32::INFINITY);
        for (id, _) in pca.nearest_embedding(query, m) {
            let dd = dist2(&data[id as usize], query);
            if dd < best.1 {
                best = (id, dd);
            }
        }
        if best.0 != u32::MAX {
            if label[best.0 as usize] == ql[qi] {
                pf_cl += 1;
            }
            if best.0 == truth[qi] {
                pf_ex += 1;
            }
        }
    }
    let pf_lat = us_per(t.elapsed().as_secs_f64(), q);

    println!("\nuseful niche — two-stage retrieval (OctaSoma PCA-3D pre-filter → full-D rerank):");
    println!(
        "  cluster@1 {:.1}%   exact@1 {:.1}%   latency {:.2} µs   full-D distance ops/query: {} vs {} (≈{}× fewer)",
        pf_cl as f64 / q as f64 * 100.0,
        pf_ex as f64 / q as f64 * 100.0,
        pf_lat,
        m,
        n,
        n / m.max(1)
    );
    println!(
        "\nreading: full-D wins recall but costs {}× the coordinate footprint and a linear scan;\n\
         OctaSoma is a compact, fast COARSE router (high cluster-recall for few themes, ~0 exact);\n\
         as a pre-filter it recovers most exact recall at a fraction of the full-D work.",
        coord_full / coord_octa
    );
}

// --- helpers ---------------------------------------------------------------

fn arg(a: &[String], i: usize, def: usize) -> usize {
    a.get(i).and_then(|s| s.parse().ok()).unwrap_or(def)
}

fn us_per(secs: f64, q: usize) -> f64 {
    secs * 1e6 / q as f64
}

fn pct(num: usize, den: usize) -> f64 {
    num as f64 / den as f64 * 100.0
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

fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn full_nn(data: &[Vec<f32>], query: &[f32]) -> u32 {
    let mut best = (0u32, f32::INFINITY);
    for (i, v) in data.iter().enumerate() {
        let d2 = dist2(v, query);
        if d2 < best.1 {
            best = (i as u32, d2);
        }
    }
    best.0
}
