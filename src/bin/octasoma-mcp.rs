//! `octasoma-mcp` — a stdio JSON-RPC (MCP) server exposing OctaSoma as **semantic
//! memory** for agents and the CHECKUPAUTO stack (CCOS / SLHAv2).
//!
//! Build & run (requires the `mcp` feature):
//! ```text
//! cargo run --release --features mcp --bin octasoma-mcp -- memory.store --hash
//! ```
//!
//! Speaks line-delimited JSON-RPC 2.0 (`initialize`, `tools/list`, `tools/call`).
//! Tools: `ingest`, `recall`, `explain`, `stats`.
//!
//! Memory is **region-sharded** ([`octasoma::ShardedMemory`]): one OctaSoma index
//! per *causal region*, the deployment the real-scale benchmark validated (a single
//! global 3-D index collapses at scale; per region it works). `ingest`/`recall`
//! take an optional `region`; when omitted it is derived from the CCOS-style uri
//! (`sym:src/db.rs:query` → `src/db.rs`), matching the in-process `ShardedOctaIndex`
//! adapter. The store is a **directory** of shards + a manifest.
//!
//! `recall` returns CCOS's `RecallWindow { strategy, items:[{uri,score,kind,content}],
//! tokens }` shape, so it drops straight into CCOS and any MCP-speaking agent.

use std::io::{self, BufRead, Write};

use octasoma::{Embedder, HashEmbedder, OllamaEmbedder, ShardedMemory};
use serde_json::{Value, json};

/// Unit separator packing `"uri␟content"` into one payload.
const SEP: char = '\u{1f}';

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut store = String::new();
    let mut use_hash = false;
    let mut url = "http://localhost:11434".to_string();
    let mut model = "nomic-embed-text".to_string();
    let mut dim: Option<usize> = None;

    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--hash" => use_hash = true,
            "--url" => url = it.next().unwrap_or_default(),
            "--model" => model = it.next().unwrap_or_default(),
            "--dim" => dim = it.next().and_then(|s| s.parse().ok()),
            _ if store.is_empty() => store = a,
            _ => {}
        }
    }
    if store.is_empty() {
        eprintln!("usage: octasoma-mcp <store_dir> [--hash] [--url U] [--model M] [--dim N]");
        std::process::exit(2);
    }

    if use_hash {
        serve(HashEmbedder::new(dim.unwrap_or(256)), &store);
    } else {
        serve(OllamaEmbedder::new(url, model, dim.unwrap_or(768)), &store);
    }
}

fn serve<E: Embedder>(embedder: E, store: &str) {
    // A populated store has a manifest; otherwise start fresh.
    let manifest = std::path::Path::new(store).join("manifest.osm");
    let mut mem = if manifest.exists() {
        ShardedMemory::open_dir(embedder, store).unwrap_or_else(|e| {
            eprintln!("could not open {store}: {e}");
            std::process::exit(1);
        })
    } else {
        ShardedMemory::new(embedder)
    };

    let stdin = io::stdin();
    let mut out = io::stdout().lock();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        if let Some(resp) = handle(&line, &mut mem, store) {
            let _ = writeln!(out, "{resp}");
            let _ = out.flush();
        }
    }
}

fn handle<E: Embedder>(line: &str, mem: &mut ShardedMemory<E>, store: &str) -> Option<String> {
    let req: Value = serde_json::from_str(line).ok()?;
    let id = req.get("id").cloned();
    let method = req.get("method").and_then(Value::as_str).unwrap_or("");
    match method {
        "initialize" => Some(reply(
            id,
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "octasoma", "version": env!("CARGO_PKG_VERSION") }
            }),
        )),
        "notifications/initialized" | "initialized" => None,
        "ping" => Some(reply(id, json!({}))),
        "tools/list" => Some(reply(id, json!({ "tools": tool_list() }))),
        "tools/call" => {
            let p = req.get("params").cloned().unwrap_or(Value::Null);
            let name = p.get("name").and_then(Value::as_str).unwrap_or("");
            let args = p.get("arguments").cloned().unwrap_or_else(|| json!({}));
            let (text, is_error) = match call_tool(name, &args, mem, store) {
                Ok(v) => (v.to_string(), false),
                Err(e) => (e, true),
            };
            Some(reply(
                id,
                json!({ "content": [ { "type": "text", "text": text } ], "isError": is_error }),
            ))
        }
        _ => id.map(|id| error(Some(id), -32601, "method not found")),
    }
}

fn call_tool<E: Embedder>(
    name: &str,
    args: &Value,
    mem: &mut ShardedMemory<E>,
    store: &str,
) -> Result<Value, String> {
    let arg_str = |k: &str| {
        args.get(k)
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    };
    let arg_usize = |k: &str, d: usize| {
        args.get(k)
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(d)
    };

    match name {
        "ingest" => {
            let (uri, text) = (arg_str("uri"), arg_str("text"));
            if text.is_empty() {
                return Err("ingest needs `text`".into());
            }
            // Region: explicit arg, else derived from the uri, else "default".
            let region = {
                let r = arg_str("region");
                if !r.is_empty() {
                    r
                } else if !uri.is_empty() {
                    region_of(&uri)
                } else {
                    "default".to_string()
                }
            };
            // Pack uri+content as the payload; embed the content.
            let packed = format!("{uri}{SEP}{text}");
            mem.insert(&region, &packed, &text)
                .map_err(|e| e.to_string())?;
            mem.save_dir(store)
                .map_err(|e| format!("save failed: {e}"))?;
            Ok(json!({ "uri": uri, "region": region, "nodes_added": 1 }))
        }
        "recall" => {
            let text = {
                let t = arg_str("text");
                if t.is_empty() { arg_str("anchor") } else { t }
            };
            if text.is_empty() {
                return Err("recall needs `text`".into());
            }
            let k = arg_usize("k", arg_usize("budget", 5)).max(1);
            let region = arg_str("region");

            // Scoped recall within a causal region (the validated path); else a
            // coarse cross-region merge.
            let hits = if region.is_empty() {
                mem.recall_global_scored(&text, k)
            } else {
                mem.recall_scored(&region, &text, k)
            }
            .map_err(|e| e.to_string())?;

            let mut items = Vec::new();
            let mut tokens = 0usize;
            for (packed, d2) in hits {
                let (uri, content) = split_payload(&packed);
                tokens += content.len() / 4 + 1;
                items.push(json!({
                    "uri": uri,
                    "score": 1.0 / (1.0 + d2 as f64),
                    "kind": kind_of(&uri),
                    "content": content,
                }));
            }
            let strategy = if region.is_empty() {
                "semantic-global"
            } else {
                "semantic"
            };
            Ok(json!({ "strategy": strategy, "region": region, "items": items, "tokens": tokens }))
        }
        "explain" => {
            let text = arg_str("text");
            if text.is_empty() {
                return Err("explain needs `text`".into());
            }
            let k = arg_usize("k", 5).max(1);
            // Region: explicit, else the sole region if there is exactly one.
            let region = {
                let r = arg_str("region");
                if !r.is_empty() {
                    r
                } else {
                    let keys = mem.region_keys();
                    match keys.as_slice() {
                        [only] => only.to_string(),
                        _ => {
                            return Err(format!(
                                "explain needs `region` (one of: {})",
                                keys.join(", ")
                            ));
                        }
                    }
                }
            };
            match mem.explain(&region, &text, k).map_err(|e| e.to_string())? {
                None => Err(format!("unknown region '{region}' or invalid query")),
                Some(e) => {
                    let zoom: Vec<Value> = e
                        .zoom_path
                        .iter()
                        .map(|r| json!({ "level": r.level, "count": r.count, "half_size": r.half_size }))
                        .collect();
                    let neighbors: Vec<Value> = e
                        .neighbors
                        .iter()
                        .map(|nb| {
                            let (uri, content) =
                                split_payload(&String::from_utf8_lossy(&nb.payload));
                            json!({ "uri": uri, "content": content, "distance": nb.distance, "point": nb.point })
                        })
                        .collect();
                    Ok(json!({
                        "region": region,
                        "query_point": e.query_point,
                        "zoom_path": zoom,
                        "neighbors": neighbors,
                    }))
                }
            }
        }
        "stats" => Ok(json!({
            "memories": mem.len(),
            "regions": mem.regions(),
            "region_keys": mem.region_keys(),
        })),
        other => Err(format!("unknown tool '{other}'")),
    }
}

/// Causal region (file) from a CCOS-style `kind:path[:symbol]` uri; falls back to
/// the whole uri. Mirrors `integration/ccos/octa_index.rs::region_of`.
fn region_of(uri: &str) -> String {
    let rest = uri.split_once(':').map(|(_, r)| r).unwrap_or(uri);
    if uri.starts_with("sym:")
        && let Some(i) = rest.rfind(':')
    {
        return rest[..i].to_string();
    }
    rest.to_string()
}

fn split_payload(raw: &str) -> (String, String) {
    match raw.split_once(SEP) {
        Some((u, c)) => (u.to_string(), c.to_string()),
        None => (String::new(), raw.to_string()),
    }
}

fn kind_of(uri: &str) -> String {
    uri.split(':')
        .next()
        .filter(|s| !s.is_empty())
        .unwrap_or("memory")
        .to_string()
}

fn tool_list() -> Value {
    json!([
        {
            "name": "ingest",
            "description": "Embed `text` and store it as a semantic memory under `uri`, in causal region `region` (optional; derived from a CCOS-style uri when omitted).",
            "inputSchema": { "type": "object",
                "properties": { "uri": {"type":"string"}, "text": {"type":"string"}, "region": {"type":"string"} },
                "required": ["text"] }
        },
        {
            "name": "recall",
            "description": "Semantic recall nearest `text`. With `region` it is scoped to that causal region (the validated path); without, a coarse cross-region merge. Returns {strategy, region, items:[{uri,score,kind,content}], tokens} (CCOS RecallWindow shape).",
            "inputSchema": { "type": "object",
                "properties": { "text": {"type":"string"}, "region": {"type":"string"}, "k": {"type":"integer","default":5} },
                "required": ["text"] }
        },
        {
            "name": "explain",
            "description": "Explain a recall within `region` (optional if only one region exists): the query's 3-D position, the coarse→fine zoom path, and nearest memories with distances.",
            "inputSchema": { "type": "object",
                "properties": { "text": {"type":"string"}, "region": {"type":"string"}, "k": {"type":"integer","default":5} },
                "required": ["text"] }
        },
        {
            "name": "stats",
            "description": "Memory statistics: total memories, region count, and region keys.",
            "inputSchema": { "type": "object", "properties": {} }
        }
    ])
}

fn reply(id: Option<Value>, value: Value) -> String {
    json!({ "jsonrpc": "2.0", "id": id.unwrap_or(Value::Null), "result": value }).to_string()
}

fn error(id: Option<Value>, code: i64, message: &str) -> String {
    json!({ "jsonrpc": "2.0", "id": id.unwrap_or(Value::Null), "error": { "code": code, "message": message } })
        .to_string()
}
