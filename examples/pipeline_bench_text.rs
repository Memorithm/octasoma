//! A3 with **real text + real embeddings**: runs the CCOS→OctaSoma cascade on a
//! small realistic corpus, embedded by a real model.
//!
//! ```text
//! cargo run --release --example pipeline_bench_text                    # offline smoke (HashEmbedder)
//! cargo run --release --example pipeline_bench_text -- --url http://localhost:11434 --model nomic-embed-text --dim 768
//! ```
//!
//! Offline (`--hash`) the queries are paraphrases that do NOT match by hash, so
//! semantic hits are ~0 — that run only checks the plumbing. With Ollama the
//! paraphrases match semantically and the cascade is meaningful. Modules share
//! topics (auth+http = "web", db+cache = "storage"), so semantic-only confuses
//! modules and the causal scope (CCOS) is what disambiguates.

use octasoma::{Embedder, FractalMemory3D, HashEmbedder, OllamaEmbedder};

/// (uri, module index, content)
const NODES: &[(&str, usize, &str)] = &[
    (
        "sym:auth.rs:login",
        0,
        "authenticate a user with username and password",
    ),
    (
        "sym:auth.rs:token",
        0,
        "issue and verify JSON web tokens for sessions",
    ),
    ("sym:auth.rs:hash", 0, "hash and salt passwords with argon2"),
    (
        "sym:auth.rs:logout",
        0,
        "end the user session and clear credentials",
    ),
    (
        "sym:http.rs:route",
        1,
        "match incoming HTTP requests to handlers",
    ),
    (
        "sym:http.rs:serve",
        1,
        "start the HTTP server and accept connections",
    ),
    (
        "sym:http.rs:cors",
        1,
        "configure cross-origin resource sharing headers",
    ),
    (
        "sym:http.rs:client",
        1,
        "send outbound HTTP requests to other services",
    ),
    (
        "sym:db.rs:query",
        2,
        "build and execute SQL queries against Postgres",
    ),
    (
        "sym:db.rs:pool",
        2,
        "manage a pool of reusable database connections",
    ),
    (
        "sym:db.rs:migrate",
        2,
        "apply schema migrations to the database",
    ),
    (
        "sym:db.rs:tx",
        2,
        "run statements inside a database transaction",
    ),
    (
        "sym:cache.rs:get",
        3,
        "look up a value in the in-memory LRU cache",
    ),
    (
        "sym:cache.rs:set",
        3,
        "insert a value into the cache with a TTL",
    ),
    (
        "sym:cache.rs:evict",
        3,
        "evict least-recently-used entries when full",
    ),
    (
        "sym:cache.rs:warm",
        3,
        "preload hot keys into the cache at startup",
    ),
];

/// (query paraphrase, target uri)
const QUERIES: &[(&str, &str)] = &[
    ("how do users sign in?", "sym:auth.rs:login"),
    ("verify a JWT session token", "sym:auth.rs:token"),
    ("open a pooled connection to the database", "sym:db.rs:pool"),
    (
        "remove old entries when the cache is full",
        "sym:cache.rs:evict",
    ),
    (
        "handle cross-origin requests in the web server",
        "sym:http.rs:cors",
    ),
    ("run SQL inside a transaction", "sym:db.rs:tx"),
];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut url = String::new();
    let mut model = "nomic-embed-text".to_string();
    let mut dim = 768usize;
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--url" => url = it.next().unwrap_or_default(),
            "--model" => model = it.next().unwrap_or_default(),
            "--dim" => dim = it.next().and_then(|s| s.parse().ok()).unwrap_or(768),
            _ => {}
        }
    }

    if url.is_empty() {
        eprintln!("[i] no --url: offline HashEmbedder (plumbing smoke test; semantic hits ~0).\n");
        run(HashEmbedder::new(256));
    } else {
        eprintln!("[i] embedding via Ollama {model} at {url}\n");
        run(OllamaEmbedder::new(url, model, dim));
    }
}

fn run<E: Embedder>(embedder: E) {
    let n = NODES.len();
    let mut node_emb = Vec::with_capacity(n);
    for (_, _, content) in NODES {
        match embedder.embed(content) {
            Ok(v) => node_emb.push(v),
            Err(e) => {
                eprintln!(
                    "embed failed: {e}\nhint: start your model server, or omit --url for --hash"
                );
                std::process::exit(1);
            }
        }
    }

    let d = embedder.dim();
    let flat: Vec<f32> = node_emb.iter().flatten().copied().collect();
    let mut mem = FractalMemory3D::new_with_pca(d, &flat, n);
    for (i, v) in node_emb.iter().enumerate() {
        mem.insert(v, Some(&(i as u32).to_le_bytes()));
    }

    let tok = |i: usize| NODES[i].2.split_whitespace().count();
    let k = 5usize;
    let q = QUERIES.len();

    let (mut sem_hit, mut tri_hit) = (0usize, 0usize);
    let (mut sem_rel, mut tri_rel) = (0.0f64, 0.0f64);
    let (mut sem_tok, mut tri_tok, mut causal_tok) = (0usize, 0usize, 0usize);

    for (qtext, target) in QUERIES {
        let g = NODES.iter().position(|(u, _, _)| u == target).unwrap();
        let m = NODES[g].1;
        let qv = match embedder.embed(qtext) {
            Ok(v) => v,
            Err(_) => continue,
        };

        // semantic-only: OctaSoma global top-k.
        let sids: Vec<usize> = mem
            .nearest_embedding(&qv, k)
            .into_iter()
            .map(|(id, _)| id as usize)
            .collect();
        if sids.contains(&g) {
            sem_hit += 1;
        }
        sem_rel += rel(&sids, m);
        sem_tok += sids.iter().map(|&i| tok(i)).sum::<usize>();

        // causal scope = the target's module (assume CCOS surfaced it).
        let region: Vec<usize> = (0..n).filter(|&i| NODES[i].1 == m).collect();
        causal_tok += region.iter().map(|&i| tok(i)).sum::<usize>();

        // triad: rerank within the module by distance to the query embedding.
        let mut cand: Vec<(usize, f32)> = region
            .iter()
            .map(|&i| (i, dist2(&node_emb[i], &qv)))
            .collect();
        cand.sort_by(|a, b| a.1.total_cmp(&b.1));
        let topk: Vec<usize> = cand.iter().take(k).map(|x| x.0).collect();
        if topk.contains(&g) {
            tri_hit += 1;
        }
        tri_rel += rel(&topk, m);
        tri_tok += topk.iter().map(|&i| tok(i)).sum::<usize>();
    }

    let naive_tok: usize = (0..n).map(tok).sum();
    let pf = |x: usize| x as f64 / q as f64 * 100.0;
    let af = |x: usize| x as f64 / q as f64;

    println!("Cascade on real text ({n} nodes, {q} queries, k={k})\n");
    println!(
        "{:<26} {:>11} {:>11} {:>16}",
        "strategy", "tokens/turn", "target hit", "causal-relevant"
    );
    println!("{}", "-".repeat(68));
    println!(
        "{:<26} {:>11} {:>10.0}% {:>15.0}%",
        "naive (all nodes)",
        naive_tok,
        100.0,
        100.0 / n as f64 * region_size_avg()
    );
    println!(
        "{:<26} {:>11.1} {:>10.0}% {:>15.0}%",
        "semantic-only (OctaSoma)",
        af(sem_tok),
        pf(sem_hit),
        sem_rel / q as f64 * 100.0
    );
    println!(
        "{:<26} {:>11.1} {:>10.0}% {:>15.0}%",
        "causal-only (CCOS region)",
        af(causal_tok),
        100.0,
        100.0
    );
    println!(
        "{:<26} {:>11.1} {:>10.0}% {:>15.0}%",
        "causal + semantic (triad)",
        af(tri_tok),
        pf(tri_hit),
        tri_rel / q as f64 * 100.0
    );
    println!(
        "\nWith --url (Ollama) the paraphrase queries match semantically; the triad should\nkeep target hit high at a small token budget. Offline (--hash) hits are ~0 by design."
    );
}

fn rel(set: &[usize], module: usize) -> f64 {
    if set.is_empty() {
        return 0.0;
    }
    set.iter().filter(|&&i| NODES[i].1 == module).count() as f64 / set.len() as f64
}

fn region_size_avg() -> f64 {
    // average module size as a fraction proxy for naive relevance
    let n = NODES.len() as f64;
    let modules = NODES.iter().map(|x| x.1).max().unwrap_or(0) + 1;
    n / modules as f64
}

fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}
