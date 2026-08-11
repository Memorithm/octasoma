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
//! Memory is **region-sharded and hybrid** ([`octasoma::ShardedHybrid`]): one
//! [`octasoma::HybridMemory`] per *causal region* — the explainable 3-D layer **and**
//! the SimHash precision tier over the same items. `recall` is therefore **precise**
//! (a SimHash shortlist → exact cosine rerank), with a `strategy` knob; `explain`
//! still works via the 3-D layer. `ingest`/`recall` take an optional `region` (when
//! omitted it is derived from the CCOS-style uri, `sym:src/db.rs:query` → `src/db.rs`).
//! The store is a **directory** of per-region shards + a manifest.
//!
//! `recall` returns CCOS's `RecallWindow { strategy, items:[{uri,score,kind,content}],
//! tokens }` shape (here `score` is the cosine similarity), so it drops straight into
//! CCOS and any MCP-speaking agent.

use std::io::{self, BufRead, Read, Write};

use octasoma::{Embedder, HashEmbedder, OllamaEmbedder, QueryStrategy, ShardedHybrid};
use serde_json::{Value, json};

/// Unit separator packing `"uri␟content"` into one payload.
const SEP: char = '\u{1f}';

/// Hard resource ceilings for the stdio MCP trust boundary.
///
/// Limits are expressed in UTF-8 bytes at runtime. JSON Schema `maxLength`
/// remains useful client guidance, while the server-side checks are authoritative.
const MAX_REQUEST_BYTES: usize = 1 << 20;
const MAX_TEXT_BYTES: usize = 128 << 10;
const MAX_QUERY_BYTES: usize = 32 << 10;
const MAX_URI_BYTES: usize = 4 << 10;
const MAX_REGION_BYTES: usize = 1 << 10;
const MAX_STRATEGY_BYTES: usize = 32;
const MAX_K: usize = 32;
const MAX_DIM: usize = 16_384;
const MIN_BITS: usize = 64;
const MAX_BITS: usize = 8_192;
const MAX_FEEDBACK_URIS: usize = MAX_K * 2;
const MAX_FEEDBACK_ENTRIES: usize = 1_024;
const MAX_MEMORIES: usize = 10_000;
const MAX_REGIONS: usize = 1_024;

enum InputLine {
    Eof,
    Line(String),
    TooLong,
    InvalidUtf8,
}

/// Drain the remainder of an overlong line without allocating for it.
fn discard_until_newline<R: BufRead>(reader: &mut R) -> io::Result<()> {
    loop {
        let (consume, found_newline) = {
            let buf = reader.fill_buf()?;
            if buf.is_empty() {
                return Ok(());
            }
            match buf.iter().position(|b| *b == b'\n') {
                Some(index) => (index + 1, true),
                None => (buf.len(), false),
            }
        };
        reader.consume(consume);
        if found_newline {
            return Ok(());
        }
    }
}

/// Read one newline-delimited JSON-RPC message with a strict byte ceiling.
fn read_input_line<R: BufRead>(reader: &mut R) -> io::Result<InputLine> {
    let mut bytes = Vec::new();
    let read = {
        let mut limited = (&mut *reader).take((MAX_REQUEST_BYTES + 1) as u64);
        limited.read_until(b'\n', &mut bytes)?
    };

    if read == 0 {
        return Ok(InputLine::Eof);
    }

    if bytes.len() > MAX_REQUEST_BYTES {
        if !bytes.ends_with(b"\n") {
            discard_until_newline(reader)?;
        }
        return Ok(InputLine::TooLong);
    }

    if bytes.last() == Some(&b'\n') {
        bytes.pop();
    }
    if bytes.last() == Some(&b'\r') {
        bytes.pop();
    }

    match String::from_utf8(bytes) {
        Ok(line) => Ok(InputLine::Line(line)),
        Err(_) => Ok(InputLine::InvalidUtf8),
    }
}

fn bounded_string(args: &Value, key: &str, max_bytes: usize) -> Result<String, String> {
    let Some(raw) = args.get(key) else {
        return Ok(String::new());
    };
    let value = raw
        .as_str()
        .ok_or_else(|| format!("`{key}` must be a string"))?;
    if value.len() > max_bytes {
        return Err(format!("`{key}` exceeds the {max_bytes}-byte limit"));
    }
    Ok(value.to_string())
}

fn bounded_usize(args: &Value, key: &str, default: usize, maximum: usize) -> Result<usize, String> {
    let Some(raw) = args.get(key) else {
        return Ok(default);
    };
    let value = raw
        .as_u64()
        .ok_or_else(|| format!("`{key}` must be a positive integer"))?;
    let value =
        usize::try_from(value).map_err(|_| format!("`{key}` is too large for this platform"))?;
    if value == 0 || value > maximum {
        return Err(format!("`{key}` must be between 1 and {maximum}"));
    }
    Ok(value)
}

fn bounded_string_list(args: &Value, key: &str) -> Result<Vec<String>, String> {
    let Some(raw) = args.get(key) else {
        return Ok(Vec::new());
    };
    let values = raw
        .as_array()
        .ok_or_else(|| format!("`{key}` must be an array of strings"))?;
    if values.len() > MAX_FEEDBACK_URIS {
        return Err(format!(
            "`{key}` exceeds the {MAX_FEEDBACK_URIS}-item limit"
        ));
    }

    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let uri = value
                .as_str()
                .ok_or_else(|| format!("`{key}[{index}]` must be a string"))?;
            if uri.len() > MAX_URI_BYTES {
                return Err(format!(
                    "`{key}[{index}]` exceeds the {MAX_URI_BYTES}-byte URI limit"
                ));
            }
            Ok(uri.to_string())
        })
        .collect()
}

fn validate_store_capacity(memory_count: usize, region_count: usize) -> Result<(), String> {
    if memory_count > MAX_MEMORIES {
        return Err(format!("store exceeds the {MAX_MEMORIES}-memory limit"));
    }
    if region_count > MAX_REGIONS {
        return Err(format!("store exceeds the {MAX_REGIONS}-region limit"));
    }
    Ok(())
}

fn ensure_ingest_capacity(
    memory_count: usize,
    region_count: usize,
    adds_region: bool,
) -> Result<(), String> {
    validate_store_capacity(memory_count, region_count)?;

    if memory_count >= MAX_MEMORIES {
        return Err(format!("store has reached the {MAX_MEMORIES}-memory limit"));
    }
    if adds_region && region_count >= MAX_REGIONS {
        return Err(format!("store has reached the {MAX_REGIONS}-region limit"));
    }
    Ok(())
}

fn ensure_feedback_capacity(current: usize, additional: usize) -> Result<(), String> {
    let resulting = current
        .checked_add(additional)
        .ok_or_else(|| "feedback count overflow".to_string())?;
    if resulting > MAX_FEEDBACK_ENTRIES {
        return Err(format!(
            "feedback log would exceed the {MAX_FEEDBACK_ENTRIES}-observation session limit"
        ));
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut store = String::new();
    let mut use_hash = false;
    let mut url = "http://localhost:11434".to_string();
    let mut model = "nomic-embed-text".to_string();
    let mut dim: Option<usize> = None;
    let mut bits = 256usize;

    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--hash" => use_hash = true,
            "--url" => url = it.next().unwrap_or_default(),
            "--model" => model = it.next().unwrap_or_default(),
            "--dim" => dim = it.next().and_then(|s| s.parse().ok()),
            "--bits" => bits = it.next().and_then(|s| s.parse().ok()).unwrap_or(256),
            _ if store.is_empty() => store = a,
            _ => {}
        }
    }
    if store.is_empty() {
        eprintln!(
            "usage: octasoma-mcp <store_dir> [--hash] [--url U] [--model M] [--dim N] [--bits B]"
        );
        std::process::exit(2);
    }

    let selected_dim = dim.unwrap_or(if use_hash { 256 } else { 768 });
    if selected_dim == 0 || selected_dim > MAX_DIM {
        eprintln!("--dim must be between 1 and {MAX_DIM}");
        std::process::exit(2);
    }
    if !(MIN_BITS..=MAX_BITS).contains(&bits) || bits % 64 != 0 {
        eprintln!("--bits must be a multiple of 64 between {MIN_BITS} and {MAX_BITS}");
        std::process::exit(2);
    }

    if use_hash {
        serve(HashEmbedder::new(selected_dim), &store, bits);
    } else {
        serve(OllamaEmbedder::new(url, model, selected_dim), &store, bits);
    }
}

/// Per-session relevance-feedback state: what the last `recall` returned (so
/// the `feedback` tool can label by uri) and the accumulated log — the explicit
/// channel the calibrated tiers consume (see `octasoma::feedback`).
#[derive(Default)]
struct FeedbackState {
    last_query: String,
    last_items: Vec<(String, f32)>,
    log: octasoma::RelevanceFeedback,
}

fn serve<E: Embedder>(embedder: E, store: &str, bits: usize) {
    // A populated store has a manifest; otherwise start fresh.
    let manifest = std::path::Path::new(store).join("manifest.osh");
    let mut mem = if manifest.exists() {
        ShardedHybrid::open_dir(embedder, store).unwrap_or_else(|e| {
            eprintln!("could not open {store}: {e}");
            std::process::exit(1);
        })
    } else {
        ShardedHybrid::new(embedder, bits)
    };

    if let Err(error) = validate_store_capacity(mem.len(), mem.regions()) {
        eprintln!("could not open {store}: {error}");
        std::process::exit(1);
    }

    let stdin = io::stdin();
    let mut input = stdin.lock();
    let mut out = io::stdout().lock();
    let mut fb = FeedbackState::default();

    loop {
        let line = match read_input_line(&mut input) {
            Ok(InputLine::Eof) => break,
            Ok(InputLine::TooLong) => {
                let response = error(
                    None,
                    -32600,
                    &format!("request exceeds the {MAX_REQUEST_BYTES}-byte limit"),
                );
                let _ = writeln!(out, "{response}");
                let _ = out.flush();
                continue;
            }
            Ok(InputLine::InvalidUtf8) => {
                let response = error(None, -32700, "parse error: invalid UTF-8");
                let _ = writeln!(out, "{response}");
                let _ = out.flush();
                continue;
            }
            Ok(InputLine::Line(line)) => line,
            Err(e) => {
                eprintln!("stdin read failed: {e}");
                break;
            }
        };

        if line.trim().is_empty() {
            continue;
        }
        if let Some(resp) = handle(&line, &mut mem, store, &mut fb) {
            let _ = writeln!(out, "{resp}");
            let _ = out.flush();
        }
    }
}

fn handle<E: Embedder>(
    line: &str,
    mem: &mut ShardedHybrid<E>,
    store: &str,
    fb: &mut FeedbackState,
) -> Option<String> {
    let req: Value = match serde_json::from_str(line) {
        Ok(request) => request,
        Err(e) => {
            return Some(error(None, -32700, &format!("parse error: {e}")));
        }
    };
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
            let (text, is_error) = match call_tool(name, &args, mem, store, fb) {
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
    mem: &mut ShardedHybrid<E>,
    store: &str,
    fb: &mut FeedbackState,
) -> Result<Value, String> {
    match name {
        "ingest" => {
            let uri = bounded_string(args, "uri", MAX_URI_BYTES)?;
            let text = bounded_string(args, "text", MAX_TEXT_BYTES)?;
            if text.is_empty() {
                return Err("ingest needs `text`".into());
            }
            if uri.contains(SEP) {
                return Err("`uri` contains the reserved unit separator".into());
            }
            // Region: explicit arg, else derived from the uri, else "default".
            let region = {
                let r = bounded_string(args, "region", MAX_REGION_BYTES)?;
                if !r.is_empty() {
                    r
                } else if !uri.is_empty() {
                    region_of(&uri)
                } else {
                    "default".to_string()
                }
            };
            if region.len() > MAX_REGION_BYTES {
                return Err(format!(
                    "`region` exceeds the {MAX_REGION_BYTES}-byte limit"
                ));
            }
            if region.contains(SEP) {
                return Err("`region` contains the reserved unit separator".into());
            }

            let adds_region = !mem.region_keys().iter().any(|existing| existing == &region);
            ensure_ingest_capacity(mem.len(), mem.regions(), adds_region)?;

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
                let text = bounded_string(args, "text", MAX_QUERY_BYTES)?;
                if text.is_empty() {
                    bounded_string(args, "anchor", MAX_QUERY_BYTES)?
                } else {
                    text
                }
            };
            if text.is_empty() {
                return Err("recall needs `text`".into());
            }
            let k = if args.get("k").is_some() {
                bounded_usize(args, "k", 5, MAX_K)?
            } else {
                bounded_usize(args, "budget", 5, MAX_K)?
            };
            let region = bounded_string(args, "region", MAX_REGION_BYTES)?;
            let strategy_arg = bounded_string(args, "strategy", MAX_STRATEGY_BYTES)?;
            let strategy = parse_strategy(&strategy_arg);

            // Precise recall: scoped to a region with the chosen strategy, or a
            // cosine-merged precise recall across all regions when no region given.
            let hits = if region.is_empty() {
                mem.recall_global(&text, k)
            } else {
                mem.recall_with(&region, &text, k, strategy)
            }
            .map_err(|e| e.to_string())?;

            let mut items = Vec::new();
            let mut tokens = 0usize;
            // What the `feedback` tool will label, by uri.
            fb.last_query = text.clone();
            fb.last_items.clear();
            for (packed, cosine) in hits {
                let (uri, content) = split_payload(&packed);
                tokens += content.len() / 4 + 1;
                items.push(json!({
                    "uri": uri,
                    "score": cosine as f64,
                    "kind": kind_of(&uri),
                    "content": content,
                }));
                fb.last_items.push((uri, cosine));
            }
            let strategy_label = if region.is_empty() {
                "precise-global"
            } else {
                strategy_name(strategy)
            };
            Ok(
                json!({ "strategy": strategy_label, "region": region, "items": items, "tokens": tokens }),
            )
        }
        "explain" => {
            let text = bounded_string(args, "text", MAX_QUERY_BYTES)?;
            if text.is_empty() {
                return Err("explain needs `text`".into());
            }
            let k = bounded_usize(args, "k", 5, MAX_K)?;
            // Region: explicit, else the sole region if there is exactly one.
            let region = {
                let r = bounded_string(args, "region", MAX_REGION_BYTES)?;
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
            "feedback_recorded": fb.log.len(),
            "feedback_relevant": fb.log.relevant_count(),
        })),
        "feedback" => {
            // Label the LAST recall's items by uri — the explicit relevance
            // channel (see `octasoma::feedback`; same shape CCOS's premium
            // ImprovementLoop consumes). Unknown uris are reported, not ignored
            // silently.
            if fb.last_items.is_empty() {
                return Err("feedback needs a prior recall in this session".into());
            }
            let relevant = bounded_string_list(args, "relevant")?;
            let irrelevant = bounded_string_list(args, "irrelevant")?;

            let mut observations = Vec::new();
            let mut unknown = Vec::new();

            for (list, label) in [(relevant, true), (irrelevant, false)] {
                for uri in list {
                    match fb.last_items.iter().find(|(u, _)| *u == uri) {
                        Some((matched_uri, score)) => {
                            observations.push((matched_uri.clone(), *score, label));
                        }
                        None => unknown.push(uri),
                    }
                }
            }

            ensure_feedback_capacity(fb.log.len(), observations.len())?;

            let recorded = observations.len();
            for (uri, score, label) in observations {
                fb.log.record(&fb.last_query, &uri, score, label);
            }

            Ok(json!({
                "recorded": recorded,
                "unknown_uris": unknown,
                "total_feedback": fb.log.len(),
            }))
        }
        other => Err(format!("unknown tool '{other}'")),
    }
}

/// Parse a recall `strategy` string into a [`QueryStrategy`] (default: precise).
fn parse_strategy(s: &str) -> QueryStrategy {
    match s {
        "fast" | "spatial" => QueryStrategy::FastSpatial,
        "cascade" | "hybrid" => QueryStrategy::HybridCascade,
        _ => QueryStrategy::PrecisionSketch,
    }
}

fn strategy_name(s: QueryStrategy) -> &'static str {
    match s {
        QueryStrategy::FastSpatial => "fast-spatial",
        QueryStrategy::PrecisionSketch => "precise",
        QueryStrategy::HybridCascade => "hybrid-cascade",
    }
}

/// Causal region (file) from a CCOS-style `kind:path[:symbol]` uri; falls back to
/// the whole uri. Mirrors `integration/ccos/octa_index.rs::region_of`.
fn region_of(uri: &str) -> String {
    let rest = uri.split_once(':').map(|(_, r)| r).unwrap_or(uri);
    if !uri.starts_with("sym:") {
        return rest.to_string();
    }
    if let Some(i) = rest.rfind(':') {
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
            "inputSchema": {
                "type": "object",
                "properties": {
                    "uri": {
                        "type": "string",
                        "maxLength": MAX_URI_BYTES
                    },
                    "text": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_TEXT_BYTES
                    },
                    "region": {
                        "type": "string",
                        "maxLength": MAX_REGION_BYTES
                    }
                },
                "required": ["text"]
            }
        },
        {
            "name": "recall",
            "description": "Precise semantic recall nearest `text` (SimHash shortlist → exact cosine rerank). With `region` it is scoped; without, a cosine-merged recall across regions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_QUERY_BYTES
                    },
                    "region": {
                        "type": "string",
                        "maxLength": MAX_REGION_BYTES
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["precise", "fast", "spatial", "cascade", "hybrid"],
                        "maxLength": MAX_STRATEGY_BYTES
                    },
                    "k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_K,
                        "default": 5
                    }
                },
                "required": ["text"]
            }
        },
        {
            "name": "explain",
            "description": "Explain a recall within `region`: query position, zoom path, and nearest memories.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_QUERY_BYTES
                    },
                    "region": {
                        "type": "string",
                        "maxLength": MAX_REGION_BYTES
                    },
                    "k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_K,
                        "default": 5
                    }
                },
                "required": ["text"]
            }
        },
        {
            "name": "stats",
            "description": "Memory statistics: total memories, region count, region keys, and feedback counters.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "feedback",
            "description": "Label memories from the last recall as relevant or irrelevant.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "relevant": {
                        "type": "array",
                        "maxItems": MAX_FEEDBACK_URIS,
                        "items": {
                            "type": "string",
                            "maxLength": MAX_URI_BYTES
                        }
                    },
                    "irrelevant": {
                        "type": "array",
                        "maxItems": MAX_FEEDBACK_URIS,
                        "items": {
                            "type": "string",
                            "maxLength": MAX_URI_BYTES
                        }
                    }
                }
            }
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

#[cfg(test)]
mod capacity_tests {
    use super::*;

    #[test]
    fn store_capacity_accepts_boundaries_and_rejects_growth() {
        assert!(validate_store_capacity(MAX_MEMORIES, MAX_REGIONS).is_ok());
        assert!(ensure_ingest_capacity(MAX_MEMORIES - 1, MAX_REGIONS, false).is_ok());
        assert!(ensure_ingest_capacity(MAX_MEMORIES, 1, false).is_err());
        assert!(ensure_ingest_capacity(1, MAX_REGIONS, true).is_err());
        assert!(validate_store_capacity(MAX_MEMORIES + 1, 1).is_err());
        assert!(validate_store_capacity(1, MAX_REGIONS + 1).is_err());
    }

    #[test]
    fn feedback_capacity_is_transactional_at_the_boundary() {
        assert!(
            ensure_feedback_capacity(MAX_FEEDBACK_ENTRIES - MAX_FEEDBACK_URIS, MAX_FEEDBACK_URIS,)
                .is_ok()
        );
        assert!(ensure_feedback_capacity(MAX_FEEDBACK_ENTRIES, 1).is_err());
        assert!(ensure_feedback_capacity(usize::MAX, 1).is_err());
    }
}
