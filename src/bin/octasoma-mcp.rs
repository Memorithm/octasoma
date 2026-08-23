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

/// Optional non-negative integer argument (`u64` — timestamps, generations).
fn bounded_opt_u64(args: &Value, key: &str) -> Result<Option<u64>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(raw) => raw
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("`{key}` must be a non-negative integer")),
    }
}

/// A required non-negative integer argument.
fn bounded_req_u64(args: &Value, key: &str) -> Result<u64, String> {
    bounded_opt_u64(args, key)?.ok_or_else(|| format!("`{key}` is required"))
}

/// The record-layer join key inside an MCP payload: everything before the unit
/// separator (payloads pack `id<US>text`).
fn record_key(payload: &str) -> &str {
    payload.split_once(SEP).map_or(payload, |(id, _)| id)
}

/// `Some(value)` unless the string is empty (absent-or-blank argument → wildcard).
fn non_empty(value: String) -> Option<String> {
    if value.is_empty() { None } else { Some(value) }
}

/// Generation for a lifecycle write: explicit `generation` must strictly exceed
/// the stored one; the default is exactly one past it.
fn resolve_generation(current: Option<u64>, args: &Value) -> Result<u64, String> {
    let current = current.unwrap_or(0);
    match bounded_opt_u64(args, "generation")? {
        Some(requested) if requested <= current => Err(format!(
            "`generation` must be strictly greater than the current {current}"
        )),
        Some(requested) => Ok(requested),
        None => Ok(current + 1),
    }
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
    if !(MIN_BITS..=MAX_BITS).contains(&bits) || bits & 63 != 0 {
        eprintln!("--bits must be a multiple of 64 between {MIN_BITS} and {MAX_BITS}");
        std::process::exit(2);
    }

    if use_hash {
        serve(HashEmbedder::new(selected_dim), &store, bits, "hash");
    } else {
        serve(
            OllamaEmbedder::new(url, model.clone(), selected_dim),
            &store,
            bits,
            &model,
        );
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

fn serve<E: Embedder>(embedder: E, store: &str, bits: usize, embedder_label: &str) {
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
        if let Some(resp) = handle(&line, &mut mem, store, &mut fb, embedder_label) {
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
    embedder_label: &str,
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
            let (text, is_error) = match call_tool(name, &args, mem, store, fb, embedder_label) {
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
    embedder_label: &str,
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

            // Lifecycle-aware recall: with `now_ms`, hidden records (tombstoned,
            // superseded, TTL-expired) are dropped, and optional
            // tenant/workspace/agent scoping plus sensitivity clearance apply.
            // Scoped only — global visible merging is deliberately unsupported
            // until it has a measured story.
            let visible_now = bounded_opt_u64(args, "now_ms")?;
            let tenant = bounded_string(args, "tenant", 256)?;
            let workspace = bounded_string(args, "workspace", 256)?;
            let agent = bounded_string(args, "agent", 256)?;
            let clearance_arg = bounded_string(args, "clearance", 32)?;
            let hops = bounded_usize(args, "hops", 0, 2)?;
            let max_expanded = bounded_usize(args, "max_expanded", 8, 32)?;
            let wants_filter = visible_now.is_some()
                || !tenant.is_empty()
                || !workspace.is_empty()
                || !agent.is_empty()
                || !clearance_arg.is_empty()
                || hops > 0;
            if wants_filter && region.is_empty() {
                return Err("`region` is required when `now_ms`, scoping or `hops` is set".into());
            }

            let hits = if let Some(now) = visible_now {
                let clearance = if clearance_arg.is_empty() {
                    octasoma::Sensitivity::Restricted
                } else {
                    parse_sensitivity(&clearance_arg)?
                };
                let filter = octasoma::RecordFilter {
                    now_unix_ms: now,
                    tenant: non_empty(tenant),
                    workspace: non_empty(workspace),
                    agent: non_empty(agent),
                    clearance,
                };
                if hops > 0 {
                    mem.recall_related(
                        &region,
                        &text,
                        k,
                        &filter,
                        record_key,
                        octasoma::Traversal { hops, max_expanded },
                    )
                } else {
                    mem.recall_filtered(&region, &text, k, &filter, record_key)
                        .map(|rows| {
                            rows.into_iter()
                                .map(|(payload, score)| octasoma::RelatedHit {
                                    payload,
                                    score,
                                    hop: 0,
                                    via_kind: None,
                                    via_from: None,
                                })
                                .collect()
                        })
                }
            } else if region.is_empty() {
                mem.recall_global(&text, k).map(|rows| {
                    rows.into_iter()
                        .map(|(payload, score)| octasoma::RelatedHit {
                            payload,
                            score,
                            hop: 0,
                            via_kind: None,
                            via_from: None,
                        })
                        .collect()
                })
            } else {
                mem.recall_with(&region, &text, k, strategy).map(|rows| {
                    rows.into_iter()
                        .map(|(payload, score)| octasoma::RelatedHit {
                            payload,
                            score,
                            hop: 0,
                            via_kind: None,
                            via_from: None,
                        })
                        .collect()
                })
            }
            .map_err(|e| e.to_string())?;

            let mut items = Vec::new();
            let mut tokens = 0usize;
            fb.last_query = text.clone();
            fb.last_items.clear();
            for hit in &hits {
                let (uri, content) = split_payload(&hit.payload);
                tokens += content.len() / 4 + 1;
                let mut item = json!({
                    "uri": uri,
                    "score": hit.score as f64,
                    "kind": kind_of(&uri),
                    "content": content,
                });
                if hit.hop > 0 {
                    item["via"] = json!({
                        "from": hit.via_from,
                        "relation": hit.via_kind.map(relation_name),
                        "hop": hit.hop,
                    });
                    item["inherited_score"] = json!(true);
                }
                items.push(item);
                fb.last_items.push((uri, hit.score));
            }
            let strategy_label = if region.is_empty() {
                "precise-global"
            } else if hops > 0 {
                "precise-related"
            } else {
                strategy_name(strategy)
            };
            let mut result = json!({
                "strategy": strategy_label,
                "region": region,
                "items": items,
                "tokens": tokens,
            });
            if let Some(now) = visible_now {
                result["visible_as_of"] = json!(now);
            }
            Ok(result)
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
            "records": mem.records_len(),
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
        "remember" => {
            let uri = bounded_string(args, "uri", MAX_URI_BYTES)?;
            if uri.is_empty() {
                return Err("remember needs `uri`".into());
            }
            if uri.contains(SEP) {
                return Err("`uri` contains the reserved unit separator".into());
            }
            let text = bounded_string(args, "text", MAX_TEXT_BYTES)?;
            if text.is_empty() {
                return Err("remember needs `text`".into());
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

            use octasoma::{EmbeddingFingerprint, MemoryId, MemoryRecord, MemoryScope, Provenance};
            let scope_part = |key: &str| -> Result<String, String> {
                let value = bounded_string(args, key, 256)?;
                Ok(if value.is_empty() {
                    "default".to_string()
                } else {
                    value
                })
            };
            let scope = MemoryScope::new(
                scope_part("tenant")?,
                scope_part("workspace")?,
                scope_part("agent")?,
            )
            .map_err(|e| e.to_string())?;
            let sensitivity = parse_sensitivity(&bounded_string(args, "sensitivity", 32)?)?;

            let source_value = bounded_string(args, "source", 256)?;
            let source = if source_value.is_empty() {
                "octasoma-mcp".to_string()
            } else {
                source_value
            };
            let mut provenance = Provenance::new(source).map_err(|e| e.to_string())?;
            let source_record = bounded_string(args, "source_record", MAX_URI_BYTES)?;
            if !source_record.is_empty() {
                provenance = provenance
                    .with_source_record(source_record)
                    .map_err(|e| e.to_string())?;
            }
            let created_at = bounded_opt_u64(args, "created_at_ms")?;
            if let Some(at) = created_at {
                provenance = provenance.with_observed_at(at);
            }

            let embedding = EmbeddingFingerprint::new("octasoma-mcp", embedder_label, mem.dim())
                .map_err(|e| e.to_string())?;
            let generation = resolve_generation(mem.record(&uri).map(|r| r.generation), args)?;

            let mut record = MemoryRecord::new(
                MemoryId::new(uri.clone()).map_err(|e| e.to_string())?,
                Vec::new(),
                scope,
                provenance,
                embedding,
                generation,
            );
            record.sensitivity = sensitivity;
            record.retention.expires_at_unix_ms = bounded_opt_u64(args, "expires_at_ms")?;
            record.retention.retain_until_unix_ms = bounded_opt_u64(args, "retain_until_ms")?;
            record.created_at_unix_ms = created_at;

            // The index payload keeps the recall convention (id<US>text); the
            // record's id is the join key into the record layer.
            let packed = format!("{uri}{SEP}{text}");
            mem.remember_with_payload(&region, record, packed.as_bytes(), &text)
                .map_err(|e| e.to_string())?;
            mem.save_dir(store)
                .map_err(|e| format!("save failed: {e}"))?;
            Ok(json!({
                "uri": uri,
                "region": region,
                "generation": generation,
                "records": mem.records_len(),
            }))
        }
        "tombstone" => {
            let uri = bounded_string(args, "uri", MAX_URI_BYTES)?;
            if uri.is_empty() {
                return Err("tombstone needs `uri`".into());
            }
            let current = mem.record(&uri).map(|r| r.generation);
            let generation = resolve_generation(current, args)?;
            mem.tombstone(&uri, generation).map_err(|e| e.to_string())?;
            mem.save_dir(store)
                .map_err(|e| format!("save failed: {e}"))?;
            Ok(json!({ "uri": uri, "generation": generation }))
        }
        "purge" => {
            let now_ms = bounded_req_u64(args, "now_ms")?;
            // Compact *before* dropping records: once a record is gone, its
            // index payload looks like an ordinary record-less item and the
            // unknown-ids-pass-through rule would keep it forever. Compacting
            // first reclaims every hidden entry while the record layer can
            // still vouch that it is dead — purge then becomes irreversible.
            let keys: Vec<String> = mem
                .region_keys()
                .iter()
                .map(|key| (*key).to_string())
                .collect();
            let mut index_reclaimed = 0usize;
            // Unscoped, fully-cleared lifecycle filter: an irreversible purge
            // must reclaim every hidden entry, whatever its tenant.
            let filter = octasoma::RecordFilter::at(now_ms);
            for key in &keys {
                index_reclaimed += mem
                    .compact_filtered(key, &filter, record_key)
                    .map_err(|e| e.to_string())?;
            }
            let removed = mem.purge_purgeable_at(now_ms);
            mem.save_dir(store)
                .map_err(|e| format!("save failed: {e}"))?;
            Ok(json!({
                "removed": removed,
                "index_reclaimed": index_reclaimed,
                "records": mem.records_len(),
            }))
        }
        "compact" => {
            let now_ms = bounded_req_u64(args, "now_ms")?;
            let region = bounded_string(args, "region", MAX_REGION_BYTES)?;
            let tenant = bounded_string(args, "tenant", 256)?;
            let workspace = bounded_string(args, "workspace", 256)?;
            let agent = bounded_string(args, "agent", 256)?;
            let clearance_arg = bounded_string(args, "clearance", 32)?;
            let clearance = if clearance_arg.is_empty() {
                octasoma::Sensitivity::Restricted
            } else {
                parse_sensitivity(&clearance_arg)?
            };
            // The caller's filter must mirror the predicate its recalls use:
            // compaction drops exactly what this filter can never return.
            let filter = octasoma::RecordFilter {
                now_unix_ms: now_ms,
                tenant: non_empty(tenant),
                workspace: non_empty(workspace),
                agent: non_empty(agent),
                clearance,
            };

            let mut per_region = Vec::new();
            let mut reclaimed_total = 0usize;
            if !region.is_empty() {
                let reclaimed = mem
                    .compact_filtered(&region, &filter, record_key)
                    .map_err(|e| e.to_string())?;
                reclaimed_total += reclaimed;
                per_region.push(json!({ "region": region, "reclaimed": reclaimed }));
            } else {
                let keys: Vec<String> = mem
                    .region_keys()
                    .iter()
                    .map(|key| (*key).to_string())
                    .collect();
                for key in keys {
                    let reclaimed = mem
                        .compact_filtered(&key, &filter, record_key)
                        .map_err(|e| e.to_string())?;
                    reclaimed_total += reclaimed;
                    per_region.push(json!({ "region": key, "reclaimed": reclaimed }));
                }
            }
            mem.save_dir(store)
                .map_err(|e| format!("save failed: {e}"))?;
            Ok(json!({
                "reclaimed_total": reclaimed_total,
                "regions": per_region,
                "memories": mem.len(),
            }))
        }
        "relate" => {
            let uri = bounded_string(args, "uri", MAX_URI_BYTES)?;
            if uri.is_empty() {
                return Err("relate needs `uri`".into());
            }
            let target = bounded_string(args, "target", MAX_URI_BYTES)?;
            if target.is_empty() {
                return Err("relate needs `target`".into());
            }
            let kind = parse_relation(&bounded_string(args, "relation", 32)?)?;
            let current = mem.record(&uri).map(|r| r.generation);
            let generation = resolve_generation(current, args)?;
            mem.relate(&uri, kind, &target, generation)
                .map_err(|e| e.to_string())?;
            mem.save_dir(store)
                .map_err(|e| format!("save failed: {e}"))?;
            Ok(json!({
                "uri": uri,
                "relation": relation_name(kind),
                "target": target,
                "generation": generation,
            }))
        }
        other => Err(format!("unknown tool '{other}'")),
    }
}

/// Parse a `sensitivity` string into the record enum (default: internal).
fn parse_sensitivity(s: &str) -> Result<octasoma::Sensitivity, String> {
    match s {
        "" | "internal" => Ok(octasoma::Sensitivity::Internal),
        "public" => Ok(octasoma::Sensitivity::Public),
        "confidential" => Ok(octasoma::Sensitivity::Confidential),
        "restricted" => Ok(octasoma::Sensitivity::Restricted),
        other => Err(format!(
            "`sensitivity` must be one of public, internal, confidential, restricted (got '{other}')"
        )),
    }
}

/// Wire name of a relation kind (the `relate` tool's vocabulary).
fn relation_name(kind: octasoma::RelationKind) -> &'static str {
    match kind {
        octasoma::RelationKind::Confirms => "confirms",
        octasoma::RelationKind::Contradicts => "contradicts",
        octasoma::RelationKind::Supersedes => "supersedes",
        octasoma::RelationKind::SupersededBy => "superseded_by",
    }
}

fn parse_relation(s: &str) -> Result<octasoma::RelationKind, String> {
    match s {
        "confirms" => Ok(octasoma::RelationKind::Confirms),
        "contradicts" => Ok(octasoma::RelationKind::Contradicts),
        "supersedes" => Ok(octasoma::RelationKind::Supersedes),
        "superseded_by" | "superseded-by" => Ok(octasoma::RelationKind::SupersededBy),
        other => Err(format!(
            "`relation` must be one of confirms, contradicts, supersedes, superseded_by (got '{other}')"
        )),
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
                    },
                    "now_ms": {
                        "type": "integer",
                        "description": "Lifecycle-aware recall: drop tombstoned/superseded/TTL-expired records as of this unix-ms timestamp. Requires `region`.",
                        "minimum": 0
                    },
                    "tenant": { "type": "string", "maxLength": 256 },
                    "workspace": { "type": "string", "maxLength": 256 },
                    "agent": { "type": "string", "maxLength": 256 },
                    "clearance": {
                        "type": "string",
                        "enum": ["public", "internal", "confidential", "restricted"],
                        "description": "Hide records classified strictly above this level (default: restricted = see all)."
                    },
                    "hops": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 2,
                        "default": 0,
                        "description": "Follow relation edges (confirms/contradicts/supersedes) from direct hits this many BFS levels. Requires `region` and respects the same filter; expanded items carry `via` metadata and their parent's score."
                    },
                    "max_expanded": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 32,
                        "default": 8
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
            "description": "Memory statistics: total memories, region count, region keys, record-layer size, and feedback counters.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "remember",
            "description": "Embed `text` and store it as a memory with a full logical record: tenant/workspace/agent scope, sensitivity, TTL (`expires_at_ms`) and retention floor (`retain_until_ms`). Lifecycle-aware recall sees it via `now_ms`; `tombstone`/`purge`/`compact` retire it.",
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
                    },
                    "tenant": { "type": "string", "maxLength": 256, "default": "default" },
                    "workspace": { "type": "string", "maxLength": 256, "default": "default" },
                    "agent": { "type": "string", "maxLength": 256, "default": "default" },
                    "sensitivity": {
                        "type": "string",
                        "enum": ["public", "internal", "confidential", "restricted"],
                        "default": "internal"
                    },
                    "expires_at_ms": { "type": "integer", "minimum": 0 },
                    "retain_until_ms": { "type": "integer", "minimum": 0 },
                    "created_at_ms": { "type": "integer", "minimum": 0 },
                    "source": { "type": "string", "maxLength": 256, "default": "octasoma-mcp" },
                    "source_record": { "type": "string", "maxLength": 4096 },
                    "generation": {
                        "type": "integer",
                        "description": "Must strictly exceed the stored generation for this uri (default: current + 1).",
                        "minimum": 1
                    }
                },
                "required": ["uri", "text"]
            }
        },
        {
            "name": "tombstone",
            "description": "Logical delete: mark the record for `uri` tombstoned. Subsequent lifecycle-aware recalls stop returning it; physical reclamation is `compact`'s job.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "uri": { "type": "string", "maxLength": MAX_URI_BYTES },
                    "generation": {
                        "type": "integer",
                        "description": "Must strictly exceed the stored generation (default: current + 1).",
                        "minimum": 1
                    }
                },
                "required": ["uri"]
            }
        },
        {
            "name": "relate",
            "description": "Add an evidence edge between two records: uri --relation--> target (confirms, contradicts, supersedes, superseded_by). Lifecycle-aware recall with `hops` traverses these edges within the caller's filter.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "uri": { "type": "string", "maxLength": MAX_URI_BYTES },
                    "target": { "type": "string", "maxLength": MAX_URI_BYTES },
                    "relation": {
                        "type": "string",
                        "enum": ["confirms", "contradicts", "supersedes", "superseded_by"]
                    },
                    "generation": {
                        "type": "integer",
                        "description": "Must strictly exceed the stored generation (default: current + 1).",
                        "minimum": 1
                    }
                },
                "required": ["uri", "target", "relation"]
            }
        },
        {
            "name": "purge",
            "description": "Irreversibly remove records that are inactive and past their retention floor as of `now_ms`. Compacts every region first, so hidden index entries are reclaimed while the record layer can still identify them.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "now_ms": { "type": "integer", "minimum": 0 }
                },
                "required": ["now_ms"]
            }
        },
        {
            "name": "compact",
            "description": "Rebuild region index(es) keeping only what a recall under the same filter could return at `now_ms` (lifecycle + optional tenant/workspace/agent scoping and clearance), reclaiming hidden entries. Empty `region` compacts every region. The filter must mirror your recalls — dropping an entry a legal query could still return is data loss.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "now_ms": { "type": "integer", "minimum": 0 },
                    "region": { "type": "string", "maxLength": MAX_REGION_BYTES },
                    "tenant": { "type": "string", "maxLength": 256 },
                    "workspace": { "type": "string", "maxLength": 256 },
                    "agent": { "type": "string", "maxLength": 256 },
                    "clearance": {
                        "type": "string",
                        "enum": ["public", "internal", "confidential", "restricted"]
                    }
                },
                "required": ["now_ms"]
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
