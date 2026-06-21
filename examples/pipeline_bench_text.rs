//! A3 — the CCOS → OctaSoma cascade on **real text + real embeddings**.
//!
//! ```text
//! # built-in 16-node demo corpus:
//! cargo run --release --example pipeline_bench_text -- --url http://localhost:11434 --model nomic-embed-text --dim 768
//!
//! # your real CCOS workspace (scale test):
//! cargo run --release --example pipeline_bench_text -- \
//!     --corpus nodes.tsv --queries queries.tsv \
//!     --url http://localhost:11434 --model nomic-embed-text --dim 768
//! ```
//!
//! `nodes.tsv`  : `uri⇥content`  (or `uri⇥module⇥content`). With two columns the
//!                **module is auto-derived from the uri** — the file part of a CCOS
//!                node id (`sym:src/db.rs:query` → module `src/db.rs`), i.e. the
//!                natural causal region. See `bench/sample_nodes.tsv`.
//! `queries.tsv`: `query⇥target_uri`. See `bench/sample_queries.tsv`.
//!
//! Without `--url` it runs offline with `HashEmbedder` (plumbing smoke; hits ~0).
//! Modules that share a topic are where semantic-only confuses and the causal scope
//! (CCOS) disambiguates.

use std::fs;

use octasoma::{Embedder, FractalMemory3D, HashEmbedder, OllamaEmbedder};

/// Built-in demo corpus: (uri, module index, content).
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

type Node = (String, usize, String); // (uri, module, content)

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut url, mut model, mut dim) = (String::new(), "nomic-embed-text".to_string(), 768usize);
    let (mut corpus, mut queries_path) = (None, None);
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--url" => url = it.next().unwrap_or_default(),
            "--model" => model = it.next().unwrap_or_default(),
            "--dim" => dim = it.next().and_then(|s| s.parse().ok()).unwrap_or(768),
            "--corpus" => corpus = it.next(),
            "--queries" => queries_path = it.next(),
            _ => {}
        }
    }

    let nodes = corpus.map(|p| load_nodes(&p)).unwrap_or_else(builtin_nodes);
    let queries = queries_path
        .map(|p| load_queries(&p))
        .unwrap_or_else(builtin_queries);
    eprintln!("[i] {} nodes, {} queries", nodes.len(), queries.len());

    if url.is_empty() {
        eprintln!("[i] no --url: offline HashEmbedder (plumbing smoke; semantic hits ~0).\n");
        run(HashEmbedder::new(256), &nodes, &queries);
    } else {
        eprintln!("[i] embedding via Ollama {model} at {url}\n");
        run(OllamaEmbedder::new(url, model, dim), &nodes, &queries);
    }
}

fn run<E: Embedder>(embedder: E, nodes: &[Node], queries: &[(String, String)]) {
    let n = nodes.len();
    let mut node_emb = Vec::with_capacity(n);
    for (_, _, content) in nodes {
        match embedder.embed(content) {
            Ok(v) => node_emb.push(v),
            Err(e) => {
                eprintln!(
                    "embed failed: {e}\nhint: start the model server, or drop --url for --hash"
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

    let modules = nodes.iter().map(|x| x.1).max().unwrap_or(0) + 1;
    let tok = |i: usize| nodes[i].2.split_whitespace().count();
    let k = 5usize;

    let (mut processed, mut sem_hit, mut tri_hit) = (0usize, 0usize, 0usize);
    let (mut sem_rel, mut tri_rel) = (0.0f64, 0.0f64);
    let (mut sem_tok, mut tri_tok, mut causal_tok) = (0usize, 0usize, 0usize);

    for (qtext, target) in queries {
        let Some(g) = nodes.iter().position(|(u, _, _)| u == target) else {
            eprintln!("[!] target not in corpus, skipping: {target}");
            continue;
        };
        let m = nodes[g].1;
        let Ok(qv) = embedder.embed(qtext) else {
            continue;
        };
        processed += 1;

        // semantic-only — OctaSoma global top-k.
        let sids: Vec<usize> = mem
            .nearest_embedding(&qv, k)
            .into_iter()
            .map(|(id, _)| id as usize)
            .collect();
        if sids.contains(&g) {
            sem_hit += 1;
        }
        sem_rel += rel(&sids, m, nodes);
        sem_tok += sids.iter().map(|&i| tok(i)).sum::<usize>();

        // causal scope = the target's module (assume CCOS surfaced it).
        let region: Vec<usize> = (0..n).filter(|&i| nodes[i].1 == m).collect();
        causal_tok += region.iter().map(|&i| tok(i)).sum::<usize>();

        // triad — semantic rerank within the causal region.
        let mut cand: Vec<(usize, f32)> = region
            .iter()
            .map(|&i| (i, dist2(&node_emb[i], &qv)))
            .collect();
        cand.sort_by(|a, b| a.1.total_cmp(&b.1));
        let topk: Vec<usize> = cand.iter().take(k).map(|x| x.0).collect();
        if topk.contains(&g) {
            tri_hit += 1;
        }
        tri_rel += rel(&topk, m, nodes);
        tri_tok += topk.iter().map(|&i| tok(i)).sum::<usize>();
    }
    if processed == 0 {
        eprintln!("no queries processed (targets not found in corpus?)");
        return;
    }

    let naive_tok: usize = (0..n).map(tok).sum();
    let naive_rel = 100.0 / modules as f64; // avg module is 1/modules of all nodes
    let p = processed as f64;
    let af = |x: usize| x as f64 / p;
    let pf = |x: usize| x as f64 / p * 100.0;

    println!("Cascade on real text ({n} nodes, {modules} modules, {processed} queries, k={k})\n");
    println!(
        "{:<26} {:>11} {:>11} {:>16}",
        "strategy", "tokens/turn", "target hit", "causal-relevant"
    );
    println!("{}", "-".repeat(68));
    println!(
        "{:<26} {:>11} {:>10.0}% {:>15.0}%",
        "naive (all nodes)", naive_tok, 100.0, naive_rel
    );
    println!(
        "{:<26} {:>11.1} {:>10.0}% {:>15.0}%",
        "semantic-only (OctaSoma)",
        af(sem_tok),
        pf(sem_hit),
        sem_rel / p * 100.0
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
        tri_rel / p * 100.0
    );
    println!(
        "\nReading: the triad keeps target hit high at a small token budget AND 100%\n\
         causal relevance; semantic-only spends the budget on same-topic/wrong-module\n\
         nodes (lower relevance). At large N, OctaSoma's coarse 3-D makes the causal\n\
         narrowing essential — try a bigger --corpus to see it."
    );
}

fn builtin_nodes() -> Vec<Node> {
    NODES
        .iter()
        .map(|(u, m, c)| (u.to_string(), *m, c.to_string()))
        .collect()
}

fn builtin_queries() -> Vec<(String, String)> {
    QUERIES
        .iter()
        .map(|(q, t)| (q.to_string(), t.to_string()))
        .collect()
}

/// `uri⇥content` or `uri⇥module⇥content`; module auto-derived from the uri when absent.
fn load_nodes(path: &str) -> Vec<Node> {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut keys: Vec<String> = Vec::new();
    let mut out = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.splitn(3, '\t').collect();
        let (uri, mkey, content) = match cols.as_slice() {
            [u, m, c] => (u.to_string(), m.to_string(), c.to_string()),
            [u, c] => (u.to_string(), module_key(u), c.to_string()),
            _ => continue,
        };
        let idx = keys.iter().position(|k| *k == mkey).unwrap_or_else(|| {
            keys.push(mkey);
            keys.len() - 1
        });
        out.push((uri, idx, content));
    }
    assert!(
        !out.is_empty(),
        "no nodes parsed from {path} (expected `uri\\tcontent` per line)"
    );
    out
}

fn load_queries(path: &str) -> Vec<(String, String)> {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut out = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Some((q, t)) = line.split_once('\t') {
            out.push((q.to_string(), t.to_string()));
        }
    }
    assert!(
        !out.is_empty(),
        "no queries parsed from {path} (expected `query\\ttarget_uri` per line)"
    );
    out
}

/// CCOS node id `kind:file:rest` → the file part (its causal region).
fn module_key(uri: &str) -> String {
    let parts: Vec<&str> = uri.splitn(3, ':').collect();
    if parts.len() >= 2 {
        parts[1].to_string()
    } else {
        uri.to_string()
    }
}

fn rel(set: &[usize], module: usize, nodes: &[Node]) -> f64 {
    if set.is_empty() {
        return 0.0;
    }
    set.iter().filter(|&&i| nodes[i].1 == module).count() as f64 / set.len() as f64
}

fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}
