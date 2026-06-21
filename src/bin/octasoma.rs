//! `octasoma` — a tiny command-line semantic memory for everyone.
//!
//! No code required: store and recall memories from your shell. Memories live in
//! a single `.frac` file (default `./octasoma.frac`, override with `--store` or
//! the `OCTASOMA_STORE` env var).
//!
//! By default it embeds text with a local Ollama server; pass `--hash` to run
//! fully offline (exact-text recall only, no model needed).
//!
//! ```text
//! octasoma remember "I prefer dark mode and the metric system"
//! octasoma recall   "what are my preferences?"
//! octasoma reflect  "preferences" -k 3
//! octasoma stats
//! ```

use std::process::exit;

use octasoma::{EmbedError, Embedder, HashEmbedder, OctaSomaAgent, OllamaEmbedder};

const DEFAULT_STORE: &str = "octasoma.frac";
const DEFAULT_URL: &str = "http://localhost:11434";
const DEFAULT_MODEL: &str = "nomic-embed-text";
const DEFAULT_OLLAMA_DIM: usize = 768;
const DEFAULT_HASH_DIM: usize = 256;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || matches!(args[0].as_str(), "help" | "-h" | "--help") {
        usage();
        return;
    }

    // ---- parse flags (order-independent) ----
    let mut command = String::new();
    let mut positionals: Vec<String> = Vec::new();
    let mut store = std::env::var("OCTASOMA_STORE").unwrap_or_else(|_| DEFAULT_STORE.to_string());
    let mut use_hash = false;
    let mut url = DEFAULT_URL.to_string();
    let mut model = DEFAULT_MODEL.to_string();
    let mut dim: Option<usize> = None;
    let mut k: usize = 5;

    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--store" => store = it.next().unwrap_or_else(|| die("--store needs a path")),
            "--hash" => use_hash = true,
            "--url" => url = it.next().unwrap_or_else(|| die("--url needs a value")),
            "--model" => model = it.next().unwrap_or_else(|| die("--model needs a value")),
            "--dim" => {
                dim = Some(
                    it.next()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or_else(|| die("--dim needs an integer")),
                )
            }
            "-k" | "--top-k" => {
                k = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| die("-k needs an integer"))
            }
            _ if command.is_empty() => command = a,
            _ => positionals.push(a),
        }
    }

    let text = positionals.join(" ");

    // ---- dispatch with the chosen embedder ----
    let result = if use_hash {
        run(
            HashEmbedder::new(dim.unwrap_or(DEFAULT_HASH_DIM)),
            &command,
            &text,
            &store,
            k,
        )
    } else {
        run(
            OllamaEmbedder::new(url, model, dim.unwrap_or(DEFAULT_OLLAMA_DIM)),
            &command,
            &text,
            &store,
            k,
        )
    };

    if let Err(msg) = result {
        eprintln!("error: {msg}");
        exit(1);
    }
}

fn run<E: Embedder>(
    embedder: E,
    command: &str,
    text: &str,
    store: &str,
    k: usize,
) -> Result<(), String> {
    match command {
        "remember" | "add" | "store" => {
            if text.is_empty() {
                return Err("nothing to remember (provide some text)".into());
            }
            let mut agent = load_or_new(embedder, store)?;
            agent.perceive(text).map_err(embed_err)?;
            agent
                .save(store)
                .map_err(|e| format!("could not save {store}: {e}"))?;
            println!("remembered ({} memories in {store})", agent.len());
            Ok(())
        }
        "recall" | "query" | "search" => {
            if text.is_empty() {
                return Err("nothing to recall (provide a query)".into());
            }
            let agent = load_existing(embedder, store)?;
            let hits = agent.recall(text, k).map_err(embed_err)?;
            if hits.is_empty() {
                println!("(no memories found)");
            } else {
                for (i, m) in hits.iter().enumerate() {
                    println!("{}. {m}", i + 1);
                }
            }
            Ok(())
        }
        "reflect" | "context" => {
            if text.is_empty() {
                return Err("nothing to reflect on (provide a query)".into());
            }
            let agent = load_existing(embedder, store)?;
            let ctx = agent.reflect(text, k).map_err(embed_err)?;
            println!(
                "{}",
                if ctx.is_empty() {
                    "(no memories found)".into()
                } else {
                    ctx
                }
            );
            Ok(())
        }
        "explain" | "why" => {
            if text.is_empty() {
                return Err("nothing to explain (provide a query)".into());
            }
            let agent = load_existing(embedder, store)?;
            match agent.explain(text, k).map_err(embed_err)? {
                None => println!("(query did not project to a valid point)"),
                Some(e) => {
                    let q = e.query_point;
                    println!("query position: [{:.3}, {:.3}, {:.3}]", q[0], q[1], q[2]);
                    println!("\nzoom path (coarse → fine):");
                    for r in &e.zoom_path {
                        println!(
                            "  level {:>2}: {:>7} memories  (half_size {:.4})",
                            r.level, r.count, r.half_size
                        );
                    }
                    println!("\nnearest memories (the 'why'):");
                    for (i, nb) in e.neighbors.iter().enumerate() {
                        println!(
                            "  {}. d={:.4}  [{:.2},{:.2},{:.2}]  {}",
                            i + 1,
                            nb.distance,
                            nb.point[0],
                            nb.point[1],
                            nb.point[2],
                            String::from_utf8_lossy(&nb.payload)
                        );
                    }
                }
            }
            Ok(())
        }
        "export" => {
            if text.is_empty() {
                return Err("provide an output path, e.g. `octasoma export memory.json`".into());
            }
            let agent = load_existing(embedder, store)?;
            let json = agent.export_points_json(1_000_000);
            std::fs::write(text, &json).map_err(|e| format!("could not write {text}: {e}"))?;
            println!(
                "exported {} memories to {text} (3-D points for a viewer)",
                agent.len()
            );
            Ok(())
        }
        "stats" | "info" => {
            let agent = load_existing(embedder, store)?;
            let core = agent.core();
            println!("store:   {store}");
            println!("memories:{:>8}", agent.len());
            println!("nodes:   {:>8}", core.node_count());
            println!("arena:   {:>8} bytes", core.arena_size());
            println!("high_dim:{:>8}", core.high_dim);
            Ok(())
        }
        other => Err(format!("unknown command '{other}' (try `octasoma help`)")),
    }
}

fn load_or_new<E: Embedder>(embedder: E, store: &str) -> Result<OctaSomaAgent<E>, String> {
    if std::path::Path::new(store).exists() {
        load_existing(embedder, store)
    } else {
        Ok(OctaSomaAgent::new(embedder, 42))
    }
}

fn load_existing<E: Embedder>(embedder: E, store: &str) -> Result<OctaSomaAgent<E>, String> {
    if !std::path::Path::new(store).exists() {
        return Err(format!(
            "no memory at {store} yet — run `octasoma remember \"...\"` first"
        ));
    }
    OctaSomaAgent::from_file(embedder, store).map_err(|e| {
        format!("could not open {store}: {e} (was it created with a different embedder/dimension?)")
    })
}

fn embed_err(e: EmbedError) -> String {
    match e {
        EmbedError::Io(io) => format!(
            "embedding request failed: {io}\nhint: start your model server (e.g. `ollama serve`) \
or run offline with `--hash`"
        ),
        EmbedError::Protocol(m) => format!("embedding endpoint returned an unexpected reply: {m}"),
    }
}

fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    exit(2);
}

fn usage() {
    println!(
        "octasoma — command-line semantic memory\n\n\
USAGE:\n  \
octasoma <command> [text...] [options]\n\n\
COMMANDS:\n  \
remember <text>     store a memory\n  \
recall   <query>    list the most relevant memories\n  \
reflect  <query>    print a prompt-ready context block\n  \
explain  <query>    show WHY: zoom path + nearest memories + 3-D positions\n  \
export   <file>     dump 3-D points to JSON for a viewer\n  \
stats               show store statistics\n  \
help                show this help\n\n\
OPTIONS:\n  \
--store <path>      memory file (default: ./octasoma.frac, or $OCTASOMA_STORE)\n  \
-k, --top-k <n>     how many memories to return (default: 5)\n  \
--hash              offline embedder (no model server; exact-text recall only)\n  \
--url <url>         Ollama base URL (default: http://localhost:11434)\n  \
--model <name>      embedding model (default: nomic-embed-text)\n  \
--dim <n>           embedding dimensionality (default: 768 ollama / 256 hash)\n\n\
EXAMPLES:\n  \
octasoma remember \"I prefer dark mode\"\n  \
octasoma recall \"what do I prefer?\"\n  \
octasoma --hash remember \"works fully offline\"\n  \
octasoma --hash recall \"offline\""
    );
}
