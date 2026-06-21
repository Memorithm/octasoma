//! Honest, **synthetic** benchmark of the CCOS → OctaSoma pipeline *shape*: how
//! many tokens a recall injects, whether it still finds the target, and how
//! causally-relevant the returned context is — for naive / semantic-only /
//! causal-only / causal+semantic.
//!
//! THIS IS A SYNTHETIC MODEL, not CCOS's or SLHAv2's real numbers. Nodes are
//! clustered embeddings; "modules" stand in for CCOS's causal regions. It measures
//! the *mechanism* (causal pre-filter narrows the set, semantic ranks within it),
//! to sanity-check the architecture — not to assert production figures.
//!
//! Run: `cargo run --release --example pipeline_bench`

use std::time::Instant;

use octasoma::{DeterministicRng, FractalMemory3D};

fn main() {
    let (n, d) = (20_000usize, 256usize);
    let (topics, mods_per_topic) = (16usize, 8usize);
    let modules = topics * mods_per_topic;
    let (q, k, tok_per_node) = (500usize, 8usize, 50usize);

    let mut rng = DeterministicRng::new(0xA11CE);
    let topic_centers: Vec<Vec<f32>> = (0..topics).map(|_| unit(rand_vec(&mut rng, d))).collect();

    // Nodes: each in one module; a module belongs to one topic (so several modules
    // share a topic — that is where causal scope, not semantics, tells them apart).
    let mut emb: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut node_module: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        let m = i % modules;
        node_module.push(m);
        emb.push(unit(noisy(
            &topic_centers[m / mods_per_topic],
            0.45,
            &mut rng,
        )));
    }
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); modules];
    for (i, &m) in node_module.iter().enumerate() {
        members[m].push(i);
    }

    // OctaSoma as the global semantic index.
    let calib: Vec<f32> = emb[..n.min(4_000)].iter().flatten().copied().collect();
    let mut mem = FractalMemory3D::new_with_pca(d, &calib, n.min(4_000));
    for (i, v) in emb.iter().enumerate() {
        mem.insert(v, Some(&(i as u32).to_le_bytes()));
    }

    let (mut sem_found, mut tri_found) = (0usize, 0usize);
    let (mut sem_prec, mut tri_prec, mut naive_prec) = (0.0f64, 0.0f64, 0.0f64);
    let (mut sem_tok, mut causal_tok, mut tri_tok) = (0usize, 0usize, 0usize);

    let t0 = Instant::now();
    for _ in 0..q {
        let g = (rng.next_u64() as usize) % n;
        let m = node_module[g];
        // Query: a paraphrase of the specific target node g.
        let query = unit(noisy(&emb[g], 0.15, &mut rng));

        // (2) semantic-only — OctaSoma global top-k.
        let sids: Vec<usize> = mem
            .nearest_embedding(&query, k)
            .into_iter()
            .map(|(id, _)| id as usize)
            .collect();
        if sids.contains(&g) {
            sem_found += 1;
        }
        sem_prec += relevant_fraction(&sids, m, &node_module);
        sem_tok += sids.len() * tok_per_node;

        // (3) causal-only — the whole causal region (module), unranked.
        causal_tok += members[m].len() * tok_per_node;

        // (4) causal + semantic — semantic top-k *within* the causal region.
        let mut cand: Vec<(usize, f32)> = members[m]
            .iter()
            .map(|&j| (j, dist2(&emb[j], &query)))
            .collect();
        cand.sort_by(|a, b| a.1.total_cmp(&b.1));
        let topk: Vec<usize> = cand.iter().take(k).map(|x| x.0).collect();
        if topk.contains(&g) {
            tri_found += 1;
        }
        tri_prec += relevant_fraction(&topk, m, &node_module);
        tri_tok += topk.len() * tok_per_node;

        // naive precision (all nodes): region size / N.
        naive_prec += members[m].len() as f64 / n as f64;
    }
    let us = t0.elapsed().as_secs_f64() * 1e6 / q as f64;

    let naive_tok = n * tok_per_node;
    let avg = |x: usize| x as f64 / q as f64;
    let pct = |x: usize| x as f64 / q as f64 * 100.0;
    let red = |t: f64| naive_tok as f64 / t;

    println!("Pipeline benchmark — SYNTHETIC model (NOT CCOS/SLHAv2 real numbers)");
    println!(
        "N={n} nodes, {topics} topics × {mods_per_topic} modules, k={k}, {tok_per_node} tok/node\n"
    );
    println!(
        "{:<26} {:>11} {:>11} {:>16} {:>10}",
        "strategy", "tokens/turn", "target hit", "causal-relevant", "vs naive"
    );
    println!("{}", "-".repeat(78));
    println!(
        "{:<26} {:>11} {:>10.0}% {:>15.0}% {:>10}",
        "naive (inject all)",
        naive_tok,
        100.0,
        naive_prec / q as f64 * 100.0,
        "1×"
    );
    println!(
        "{:<26} {:>11.0} {:>10.1}% {:>15.0}% {:>9.0}×",
        "semantic-only (OctaSoma)",
        avg(sem_tok),
        pct(sem_found),
        sem_prec / q as f64 * 100.0,
        red(avg(sem_tok))
    );
    println!(
        "{:<26} {:>11.0} {:>10.0}% {:>15.0}% {:>9.0}×",
        "causal-only (CCOS region)",
        avg(causal_tok),
        100.0,
        100.0,
        red(avg(causal_tok))
    );
    println!(
        "{:<26} {:>11.0} {:>10.1}% {:>15.0}% {:>9.0}×",
        "causal + semantic (triad)",
        avg(tri_tok),
        pct(tri_found),
        tri_prec / q as f64 * 100.0,
        red(avg(tri_tok))
    );
    println!("\nOctaSoma global semantic recall: {us:.1} µs/query.");
    println!(
        "Reading: at the SAME tiny token budget, the triad keeps ~100% target hit AND\n\
         100% causally-relevant context, while semantic-only spends the budget on\n\
         same-topic-but-wrong-module nodes (lower relevance); causal-only is relevant\n\
         but far larger. Causal narrows (CCOS), semantic ranks within (OctaSoma).\n\
         SYNTHETIC — real gains depend on CCOS's filter and SLHAv2's kernel."
    );
}

fn relevant_fraction(set: &[usize], module: usize, node_module: &[usize]) -> f64 {
    if set.is_empty() {
        return 0.0;
    }
    let hits = set.iter().filter(|&&j| node_module[j] == module).count();
    hits as f64 / set.len() as f64
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
