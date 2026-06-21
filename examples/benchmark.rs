//! OctaSoma benchmark & evaluation harness.
//!
//! Produces the real numbers reported in the paper and documentation:
//!   * insertion throughput,
//!   * exact octree k-NN latency vs. brute-force latency (speed-up),
//!   * recall@k of the 3-D projected nearest neighbours against the true
//!     high-dimensional nearest neighbours — for both JL and PCA projections,
//!   * memory footprint and on-disk LZ4 compression ratio.
//!
//! Run with:  `cargo run --release --example benchmark`
//! Optional args: N D CLUSTERS QUERIES K   (e.g. `... -- 50000 256 200 1000 10`)

use std::time::Instant;

use octasoma::{DeterministicRng, FractalMemory3D, ItemId};

/// Deterministic, normalised, clustered high-dimensional dataset.
struct Dataset {
    high_dim: usize,
    centers: Vec<Vec<f32>>,
    vectors: Vec<Vec<f32>>, // length N, each of length D, L2-normalised
}

/// One sample = cluster centre + small Gaussian-ish noise, renormalised.
fn sample_near(center: &[f32], rng: &mut DeterministicRng) -> Vec<f32> {
    let mut v: Vec<f32> = center
        .iter()
        .map(|&cx| {
            // Sum of three uniforms ≈ zero-mean noise.
            let noise = (rng.next_f32() + rng.next_f32() + rng.next_f32()) / 3.0;
            cx + 0.35 * noise
        })
        .collect();
    l2_normalise(&mut v);
    v
}

impl Dataset {
    fn generate(n: usize, d: usize, clusters: usize, seed: u64) -> Self {
        let mut rng = DeterministicRng::new(seed);

        // Random unit-norm cluster centres.
        let mut centers = Vec::with_capacity(clusters);
        for _ in 0..clusters {
            let mut c: Vec<f32> = (0..d).map(|_| rng.next_f32()).collect();
            l2_normalise(&mut c);
            centers.push(c);
        }

        let mut vectors = Vec::with_capacity(n);
        for i in 0..n {
            vectors.push(sample_near(&centers[i % clusters], &mut rng));
        }
        Self {
            high_dim: d,
            centers,
            vectors,
        }
    }

    /// Fresh, held-out queries drawn from the same clusters but never inserted
    /// (different noise stream) — the honest way to measure projection recall.
    fn query_set(&self, m: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = DeterministicRng::new(seed);
        (0..m)
            .map(|i| sample_near(&self.centers[i % self.centers.len()], &mut rng))
            .collect()
    }

    fn flat_calibration(&self, samples: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(samples * self.high_dim);
        for v in self.vectors.iter().take(samples) {
            out.extend_from_slice(v);
        }
        out
    }
}

fn l2_normalise(v: &mut [f32]) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-12 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// True high-dimensional k-NN (cosine == Euclidean on unit vectors) — ground truth.
fn highd_knn(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let d2: f32 = v.iter().zip(query).map(|(a, b)| (a - b) * (a - b)).sum();
            (d2, i)
        })
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    scored.into_iter().take(k).map(|(_, i)| i).collect()
}

fn recall_at_k(predicted: &[ItemId], truth: &[usize]) -> f32 {
    let hits = predicted
        .iter()
        .filter(|&&p| truth.contains(&(p as usize)))
        .count();
    hits as f32 / truth.len() as f32
}

struct EngineEval {
    name: &'static str,
    insert_per_sec: f64,
    octree_us: f64,
    brute3d_us: f64,
    recall_at_1: f64,
    recall_at_k: f64,
    cluster_at_1: f64,
    cluster_at_k: f64,
    nodes: usize,
    items: usize,
}

fn evaluate(
    name: &'static str,
    mut mem: FractalMemory3D,
    data: &Dataset,
    queries: &[Vec<f32>],
    k: usize,
) -> EngineEval {
    // --- insertion throughput ---
    let t = Instant::now();
    for (i, v) in data.vectors.iter().enumerate() {
        mem.insert(v, Some(format!("{i}").as_bytes())).unwrap();
    }
    let insert_secs = t.elapsed().as_secs_f64();
    let insert_per_sec = data.vectors.len() as f64 / insert_secs;

    // Octree exact 3-D k-NN latency (held-out queries).
    let t = Instant::now();
    let mut octree_results = Vec::with_capacity(queries.len());
    for q in queries {
        let p = mem.project(q).unwrap();
        octree_results.push(mem.nearest(p, k));
    }
    let octree_us = t.elapsed().as_secs_f64() * 1e6 / queries.len() as f64;

    // Brute-force 3-D k-NN latency (same answers, no pruning).
    let t = Instant::now();
    for q in queries {
        let p = mem.project(q).unwrap();
        std::hint::black_box(mem.nearest_bruteforce(p, k));
    }
    let brute3d_us = t.elapsed().as_secs_f64() * 1e6 / queries.len() as f64;

    // --- recall of the 3-D projected neighbours vs. true high-D neighbours ---
    // Two notions: exact-item recall (same data point) and cluster recall
    // (same semantic cluster — the metric that matters for topical memory).
    let clusters = data.centers.len();
    let mut r1 = 0.0f64;
    let mut rk = 0.0f64;
    let mut c1 = 0.0f64;
    let mut ck = 0.0f64;
    for (qi, (q, res)) in queries.iter().zip(&octree_results).enumerate() {
        let pred: Vec<ItemId> = res.iter().map(|x| x.0).collect();
        let truth_k = highd_knn(&data.vectors, q, k); // ground truth, sorted
        if !pred.is_empty() && truth_k.first() == Some(&(pred[0] as usize)) {
            r1 += 1.0;
        }
        rk += recall_at_k(&pred, &truth_k) as f64;

        // Cluster recall: vector i belongs to cluster (i % clusters); query qi too.
        let q_cluster = qi % clusters;
        if !pred.is_empty() && pred[0] as usize % clusters == q_cluster {
            c1 += 1.0;
        }
        let same: usize = pred
            .iter()
            .filter(|&&p| p as usize % clusters == q_cluster)
            .count();
        ck += same as f64 / pred.len().max(1) as f64;
    }
    let n = queries.len() as f64;

    EngineEval {
        name,
        insert_per_sec,
        octree_us,
        brute3d_us,
        recall_at_1: r1 / n,
        recall_at_k: rk / n,
        cluster_at_1: c1 / n,
        cluster_at_k: ck / n,
        nodes: mem.node_count(),
        items: mem.item_count(),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |i: usize, def: usize| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(def);
    let n = get(0, 50_000);
    let d = get(1, 256);
    let clusters = get(2, 200);
    let queries = get(3, 1_000);
    let k = get(4, 10);

    println!("# OctaSoma benchmark");
    println!();
    println!(
        "Config: N={n} items, D={d} dims, {clusters} clusters, {queries} queries, k={k}  (machine-dependent)\n"
    );

    let data = Dataset::generate(n, d, clusters, 0xA11CE);
    let qs = data.query_set(queries, 0xC0FFEE); // held-out, never inserted

    let jl = FractalMemory3D::new(d, 42);
    let calib = data.flat_calibration(n.min(4000));
    let pca = FractalMemory3D::new_with_pca(d, &calib, n.min(4000));

    let evals = [
        evaluate("JL (random)", jl, &data, &qs, k),
        evaluate("PCA (learned)", pca, &data, &qs, k),
    ];

    println!("## Retrieval quality (exact-item vs. cluster)\n");
    println!("| Projection | recall@1 | recall@{k} | cluster@1 | cluster@{k} |");
    println!("|---|---|---|---|---|");
    for e in &evals {
        println!(
            "| {} | {:.1}% | {:.1}% | {:.1}% | {:.1}% |",
            e.name,
            e.recall_at_1 * 100.0,
            e.recall_at_k * 100.0,
            e.cluster_at_1 * 100.0,
            e.cluster_at_k * 100.0,
        );
    }

    println!("\n## Latency (exact 3-D k-NN)\n");
    println!("| Projection | octree k-NN | brute-force 3-D | speed-up |");
    println!("|---|---|---|---|");
    for e in &evals {
        println!(
            "| {} | {:.2} µs | {:.2} µs | {:.1}× |",
            e.name,
            e.octree_us,
            e.brute3d_us,
            e.brute3d_us / e.octree_us,
        );
    }

    println!("\n## Throughput & structure\n");
    println!("| Projection | insert/s | nodes | items |");
    println!("|---|---|---|---|");
    for e in &evals {
        println!(
            "| {} | {:.0} | {} | {} |",
            e.name, e.insert_per_sec, e.nodes, e.items
        );
    }

    // --- persistence / compression ---
    let mut store = FractalMemory3D::new(d, 1);
    for (i, v) in data.vectors.iter().enumerate().take(5_000) {
        // 64-byte structured payloads with redundancy, typical of agent memory.
        let payload = format!(
            "memory record #{i:05} :: cluster {} :: ok ........",
            i % clusters
        );
        store.insert(v, Some(payload.as_bytes())).unwrap();
    }
    let raw = store.arena_size();
    let compressed = lz4_flex::compress(&store.payload_arena).len();
    let path = "/tmp/octasoma_bench.frac";
    store.save_to_disk(path).unwrap();
    let on_disk = std::fs::metadata(path).unwrap().len();
    std::fs::remove_file(path).ok();

    println!("\n## Persistence (5,000 records)\n");
    println!("- payload arena raw: {raw} bytes");
    println!(
        "- payload arena LZ4: {compressed} bytes  ({:.2}× compression)",
        raw as f64 / compressed.max(1) as f64
    );
    println!("- full file on disk: {on_disk} bytes (nodes + items + projection + LZ4 arena)");
}
