//! OctaCore **global** recall on real text — SimHash shortlist → exact cosine rerank.
//!
//! Indexes a corpus via `Cascade::index_node`, then measures recall@1 / recall@k of
//! `recall_global` against the target uris. This is the scope-free precision path
//! (no CCOS region); the 3-D router scores ~0% here, the sketch tier far more.
//!
//! ```text
//! # offline plumbing smoke (built-in corpus, HashEmbedder — exact-text only):
//! cargo run --release --example recall_global_bench
//!
//! # your real corpus with a local embedding model:
//! cargo run --release --example recall_global_bench -- \
//!   --corpus nodes.tsv --queries queries.tsv \
//!   --url http://localhost:11434 --model nomic-embed-text --dim 768 --bits 1024
//! ```
//!
//! `nodes.tsv`  : `uri⇥content` per line (e.g. from `scripts/rs_to_nodes.sh`).
//! `queries.tsv`: `query⇥target_uri` per line.

use std::fs;
use std::time::Instant;

use octacore::{Cascade, InMemoryScope};
use octasoma::{Embedder, HashEmbedder, OllamaEmbedder};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut url, mut model, mut dim, mut bits, mut k, mut shortlist) = (
        String::new(),
        "nomic-embed-text".to_string(),
        768usize,
        256usize,
        5usize,
        0usize, // 0 → recall_global's default of (k*32).max(256)
    );
    let (mut corpus, mut queries_path) = (None, None);
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--url" => url = it.next().unwrap_or_default(),
            "--model" => model = it.next().unwrap_or_default(),
            "--dim" => dim = it.next().and_then(|s| s.parse().ok()).unwrap_or(768),
            "--bits" => bits = it.next().and_then(|s| s.parse().ok()).unwrap_or(256),
            "--k" => k = it.next().and_then(|s| s.parse().ok()).unwrap_or(5),
            "--shortlist" => shortlist = it.next().and_then(|s| s.parse().ok()).unwrap_or(0),
            "--corpus" => corpus = it.next(),
            "--queries" => queries_path = it.next(),
            _ => {}
        }
    }

    let nodes = corpus.map(|p| load_pairs(&p)).unwrap_or_else(builtin_nodes);
    let queries = queries_path
        .map(|p| load_pairs(&p))
        .unwrap_or_else(builtin_queries);
    let effective_shortlist = if shortlist == 0 {
        (k * 32).max(256)
    } else {
        shortlist
    };
    eprintln!(
        "[i] {} nodes, {} queries, {bits}-bit sketch, k={k}, shortlist={effective_shortlist}",
        nodes.len(),
        queries.len()
    );
    if effective_shortlist * 2 >= nodes.len() {
        eprintln!(
            "[!] shortlist ({effective_shortlist}) covers most of the corpus ({}) → the sketch \
             reranks nearly everything, so it is near-exact and --bits barely matters here. Lower \
             --shortlist (e.g. 32) or grow the corpus to see the sketch width take effect.",
            nodes.len()
        );
    }

    if url.is_empty() {
        eprintln!("[i] no --url: offline HashEmbedder (plumbing smoke; exact-text only).\n");
        run(
            HashEmbedder::new(256),
            bits,
            &nodes,
            &queries,
            k,
            effective_shortlist,
        );
    } else {
        eprintln!("[i] embedding via Ollama {model} at {url}\n");
        run(
            OllamaEmbedder::new(url, model, dim),
            bits,
            &nodes,
            &queries,
            k,
            effective_shortlist,
        );
    }
}

fn run<E: Embedder>(
    embedder: E,
    bits: usize,
    nodes: &[(String, String)],
    queries: &[(String, String)],
    k: usize,
    shortlist: usize,
) {
    let mut core = Cascade::with_sketch_bits(InMemoryScope::new(), embedder, bits);
    let t = Instant::now();
    for (uri, content) in nodes {
        if core.index_node(uri, content).is_err() {
            eprintln!("embed failed while indexing — start the model server or drop --url");
            std::process::exit(1);
        }
    }
    let index_s = t.elapsed().as_secs_f64();

    let (mut hit1, mut hitk, mut processed, mut us) = (0usize, 0usize, 0usize, 0.0f64);
    for (q, target) in queries {
        let t = Instant::now();
        let Ok(w) = core.recall_global_shortlisted(q, k, shortlist) else {
            continue;
        };
        us += t.elapsed().as_secs_f64() * 1e6;
        processed += 1;
        let uris: Vec<&str> = w.items.iter().map(|i| i.uri.as_str()).collect();
        if uris.first() == Some(&target.as_str()) {
            hit1 += 1;
        }
        if uris.iter().any(|u| u == target) {
            hitk += 1;
        }
    }
    if processed == 0 {
        eprintln!("no queries processed");
        return;
    }
    let p = processed as f64;
    println!(
        "OctaCore recall_global — {} nodes, {bits}-bit sketch, shortlist {shortlist}\n",
        nodes.len()
    );
    println!("  recall@1 : {:.1}%", hit1 as f64 / p * 100.0);
    println!("  recall@{k} : {:.1}%", hitk as f64 / p * 100.0);
    println!(
        "  recall latency: {:.0} µs/query (includes the query embedding round-trip)",
        us / p
    );
    println!("  index time: {index_s:.2}s for {} nodes", nodes.len());
    println!(
        "\n(HashEmbedder offline = exact-text recall; with OllamaEmbedder this is semantic.\n\
         The sketch width (--bits) matters when shortlist ≪ corpus — see docs/precision-sketch.md.)"
    );
}

fn load_pairs(path: &str) -> Vec<(String, String)> {
    let text = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    let mut out = Vec::new();
    for line in text.lines() {
        if let Some((a, b)) = line.split_once('\t')
            && !a.trim().is_empty()
            && !b.trim().is_empty()
        {
            out.push((a.to_string(), b.to_string()));
        }
    }
    assert!(!out.is_empty(), "no `a\\tb` pairs parsed from {path}");
    out
}

fn builtin_nodes() -> Vec<(String, String)> {
    [
        (
            "sym:src/db.rs:query",
            "build and run SQL queries against Postgres",
        ),
        (
            "sym:src/db.rs:pool",
            "manage a pool of reusable database connections",
        ),
        (
            "sym:src/auth.rs:login",
            "authenticate a user with username and password",
        ),
        (
            "sym:src/auth.rs:token",
            "issue and verify JSON web tokens for sessions",
        ),
        (
            "sym:src/cache.rs:evict",
            "evict least-recently-used entries when full",
        ),
        (
            "sym:src/http.rs:cors",
            "configure cross-origin resource sharing headers",
        ),
    ]
    .iter()
    .map(|(u, c)| (u.to_string(), c.to_string()))
    .collect()
}

fn builtin_queries() -> Vec<(String, String)> {
    [
        (
            "manage a pool of reusable database connections",
            "sym:src/db.rs:pool",
        ),
        (
            "issue and verify JSON web tokens for sessions",
            "sym:src/auth.rs:token",
        ),
        (
            "evict least-recently-used entries when full",
            "sym:src/cache.rs:evict",
        ),
    ]
    .iter()
    .map(|(q, t)| (q.to_string(), t.to_string()))
    .collect()
}
