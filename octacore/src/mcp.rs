//! A minimal, dependency-light **MCP server** that exposes the OctaCore cascade
//! to any [Model Context Protocol](https://modelcontextprotocol.io) client, so an
//! AI agent (Claude Desktop, IDE assistants, custom clients) can use it as a
//! semantic memory: `remember` documents, then `recall` a token-budgeted,
//! cosine-reranked context window.
//!
//! It speaks JSON-RPC 2.0 over newline-delimited stdio (the MCP stdio transport),
//! using only `serde`/`serde_json` — no async runtime. Build the `octacore-mcp`
//! binary with `--features mcp`; see `docs/MCP.md` for client setup.
//!
//! This is the offline, deterministic cascade (the built-in keyword causal scope
//! + OctaSoma's `HashEmbedder` exact-cosine rerank): no network, no API keys. In
//! production the same [`Cascade`] wires CCOS and a real embedder.

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use octasoma::{EmbedError, HashEmbedder};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{Cascade, CausalScope, RecallWindow, ScopeItem};

const SERVER_NAME: &str = "octacore";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");
const DEFAULT_PROTOCOL: &str = "2024-11-05";

/// One stored document: its content, an id (`uri`), and optional causal keywords.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Doc {
    uri: String,
    content: String,
    #[serde(default)]
    keywords: Vec<String>,
}

/// The in-memory corpus the cascade recalls over (optionally persisted to disk).
#[derive(Default, Serialize, Deserialize)]
struct Memory {
    docs: Vec<Doc>,
}

impl Memory {
    /// Run the cascade over the current corpus for `query`.
    fn recall(
        &self,
        query: &str,
        k: usize,
        budget_tokens: usize,
        dim: usize,
    ) -> Result<RecallWindow, EmbedError> {
        let cascade = Cascade::new(ScopeView { docs: &self.docs }, HashEmbedder::new(dim));
        cascade.recall(query, k, budget_tokens)
    }
}

/// A causal-scope view over the corpus: a document is in scope if it has no
/// keywords (always eligible → pure semantic recall) or the query mentions one
/// of them (keyword-gated causal narrowing, like the toy scope in the demo).
struct ScopeView<'a> {
    docs: &'a [Doc],
}

impl CausalScope for ScopeView<'_> {
    fn scope(&self, query: &str, _budget_tokens: usize) -> Vec<ScopeItem> {
        let q = query.to_lowercase();
        self.docs
            .iter()
            .filter(|d| {
                d.keywords.is_empty() || d.keywords.iter().any(|k| q.contains(&k.to_lowercase()))
            })
            .map(|d| ScopeItem {
                uri: d.uri.clone(),
                content: d.content.clone(),
            })
            .collect()
    }
}

/// The MCP server state: the corpus, the embedding dimension, and an optional
/// JSON store path for persistence.
struct Server {
    mem: Memory,
    dim: usize,
    store_path: Option<PathBuf>,
    /// What the `feedback` tool labels, by uri: the last recall's `(uri, score)`.
    last_query: String,
    last_items: Vec<(String, f32)>,
    /// The explicit relevance-feedback log (octasoma's channel — the calibration
    /// input for the conformal/temperature tiers; see `octasoma::feedback`).
    feedback: octasoma::RelevanceFeedback,
}

impl Server {
    /// Dispatch one JSON-RPC *request* (a message carrying an `id`) and build its
    /// response value.
    fn handle_request(&mut self, method: &str, params: Value, id: Value) -> Value {
        let result: Result<Value, (i64, String)> = match method {
            "initialize" => Ok(self.initialize(&params)),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(tools_list()),
            "tools/call" => self.tools_call(&params).map_err(|m| (-32602, m)),
            // Declared capability is `tools` only; answer these defensively so
            // clients that probe them don't error.
            "resources/list" => Ok(json!({ "resources": [] })),
            "prompts/list" => Ok(json!({ "prompts": [] })),
            other => Err((-32601, format!("method not found: {other}"))),
        };
        match result {
            Ok(value) => json!({ "jsonrpc": "2.0", "id": id, "result": value }),
            Err((code, message)) => {
                json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
            }
        }
    }

    fn initialize(&self, params: &Value) -> Value {
        // Echo the client's protocol version when present (most compatible).
        let protocol = params
            .get("protocolVersion")
            .and_then(|v| v.as_str())
            .unwrap_or(DEFAULT_PROTOCOL)
            .to_string();
        json!({
            "protocolVersion": protocol,
            "capabilities": { "tools": { "listChanged": false } },
            "serverInfo": { "name": SERVER_NAME, "version": SERVER_VERSION },
            "instructions": "OctaCore recall cascade. `remember` adds documents; \
                `recall` returns a token-budgeted, cosine-reranked context window; \
                `stats`/`clear` manage the corpus."
        })
    }

    fn tools_call(&mut self, params: &Value) -> Result<Value, String> {
        let name = params
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "missing tool `name`".to_string())?;
        let args = params
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| json!({}));
        let outcome = match name {
            "recall" => self.tool_recall(&args),
            "remember" => self.tool_remember(&args),
            "feedback" => self.tool_feedback(&args),
            "stats" => Ok(self.tool_stats()),
            "clear" => Ok(self.tool_clear()),
            other => Err(format!("unknown tool: {other}")),
        };
        // Tool-level problems are reported in the result via `isError`, not as
        // JSON-RPC errors.
        Ok(match outcome {
            Ok(text) => json!({ "content": [{ "type": "text", "text": text }], "isError": false }),
            Err(text) => json!({ "content": [{ "type": "text", "text": text }], "isError": true }),
        })
    }

    fn tool_recall(&mut self, args: &Value) -> Result<String, String> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| "`query` (non-empty string) is required".to_string())?;
        let k = args.get("k").and_then(|v| v.as_u64()).unwrap_or(5).max(1) as usize;
        let budget = args
            .get("budget_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(256)
            .max(1) as usize;

        let window = self
            .mem
            .recall(query, k, budget, self.dim)
            .map_err(|e| format!("embedding error: {e:?}"))?;

        // What a subsequent `feedback` call will label, by uri.
        self.last_query = query.to_string();
        self.last_items = window
            .items
            .iter()
            .map(|it| (it.uri.clone(), it.score))
            .collect();

        if window.items.is_empty() {
            return Ok(format!(
                "No matches for {query:?} (corpus holds {} document(s)).",
                self.mem.docs.len()
            ));
        }
        let mut out = format!(
            "recall({query:?}) — strategy={}, {} item(s), ~{} token(s):\n",
            window.strategy,
            window.items.len(),
            window.tokens
        );
        for (i, it) in window.items.iter().enumerate() {
            out.push_str(&format!(
                "{}. [{:+.3}] {} — {}\n",
                i + 1,
                it.score,
                it.uri,
                it.content
            ));
        }
        Ok(out)
    }

    fn tool_remember(&mut self, args: &Value) -> Result<String, String> {
        let mut to_add: Vec<Doc> = Vec::new();
        if let Some(arr) = args.get("documents").and_then(|v| v.as_array()) {
            for d in arr {
                to_add.push(parse_doc(d)?);
            }
        } else if args.get("content").is_some() {
            to_add.push(parse_doc(args)?);
        } else {
            return Err("provide `documents` (array) or a single `content` string".to_string());
        }
        if to_add.is_empty() {
            return Err("no documents to remember".to_string());
        }
        let mut added = 0usize;
        for mut d in to_add {
            if d.uri.trim().is_empty() {
                d.uri = format!("doc:{}", self.mem.docs.len() + 1);
            }
            self.mem.docs.push(d);
            added += 1;
        }
        self.persist();
        Ok(format!(
            "Remembered {added} document(s); corpus now holds {} document(s).",
            self.mem.docs.len()
        ))
    }

    /// Label the LAST recall's items by uri — octasoma's explicit relevance
    /// channel (the same shape CCOS's premium ImprovementLoop consumes): the
    /// collected log calibrates the conformal/temperature tiers. Unknown uris are
    /// reported, never swallowed.
    fn tool_feedback(&mut self, args: &Value) -> Result<String, String> {
        if self.last_items.is_empty() {
            return Err("feedback needs a prior recall in this session".to_string());
        }
        let uris = |key: &str| -> Vec<String> {
            args.get(key)
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str())
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default()
        };
        let (mut recorded, mut unknown) = (0usize, Vec::new());
        for (list, label) in [(uris("relevant"), true), (uris("irrelevant"), false)] {
            for uri in list {
                match self.last_items.iter().find(|(u, _)| *u == uri) {
                    Some((u, score)) => {
                        self.feedback.record(&self.last_query, u, *score, label);
                        recorded += 1;
                    }
                    None => unknown.push(uri),
                }
            }
        }
        let mut out = format!(
            "Recorded {recorded} observation(s); the feedback log holds {} ({} relevant).",
            self.feedback.len(),
            self.feedback.relevant_count()
        );
        if !unknown.is_empty() {
            out.push_str(&format!(
                " Unknown uris (not in the last recall): {unknown:?}."
            ));
        }
        Ok(out)
    }

    fn tool_stats(&self) -> String {
        let store = self.store_path.as_ref().map_or_else(
            || "(in-memory only)".to_string(),
            |p| p.display().to_string(),
        );
        format!(
            "{} document(s); embedding dim={}; store={store}; feedback: {} observation(s) ({} relevant).",
            self.mem.docs.len(),
            self.dim,
            self.feedback.len(),
            self.feedback.relevant_count()
        )
    }

    fn tool_clear(&mut self) -> String {
        let n = self.mem.docs.len();
        self.mem.docs.clear();
        self.persist();
        format!("Cleared {n} document(s).")
    }

    /// Persist the corpus to `store_path` (best effort; ignored if unset).
    fn persist(&self) {
        let Some(p) = &self.store_path else {
            return;
        };
        let Ok(bytes) = serde_json::to_vec_pretty(&self.mem) else {
            return;
        };
        let _ = std::fs::write(p, bytes);
    }
}

/// The four tools the server advertises, with JSON-Schema input definitions.
fn tools_list() -> Value {
    json!({
        "tools": [
            {
                "name": "recall",
                "description": "Recall a token-budgeted, semantically reranked context window for a query, via the OctaCore cascade (causal narrowing + exact-cosine rerank).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "What to recall." },
                        "k": { "type": "integer", "minimum": 1, "description": "Max items to return (default 5)." },
                        "budget_tokens": { "type": "integer", "minimum": 1, "description": "Approximate token budget for the window (default 256)." }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "remember",
                "description": "Add documents to the recall corpus. Each has `content`, an optional `uri` (id), and optional `keywords` (causal gating; omit for pure semantic recall).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "documents": {
                            "type": "array",
                            "description": "Documents to add.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "uri": { "type": "string" },
                                    "content": { "type": "string" },
                                    "keywords": { "type": "array", "items": { "type": "string" } }
                                },
                                "required": ["content"]
                            }
                        },
                        "content": { "type": "string", "description": "Convenience: add a single document by content." },
                        "uri": { "type": "string" },
                        "keywords": { "type": "array", "items": { "type": "string" } }
                    }
                }
            },
            {
                "name": "feedback",
                "description": "After using a recall, report which returned documents were actually relevant (by uri, referring to the LAST recall of this session). This explicit relevance feedback calibrates the memory's confidence tiers — call it whenever a recalled document clearly helped or clearly did not.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "relevant": { "type": "array", "items": { "type": "string" }, "description": "uris of the last recall's items that were useful" },
                        "irrelevant": { "type": "array", "items": { "type": "string" }, "description": "uris that were not useful" }
                    }
                }
            },
            {
                "name": "stats",
                "description": "Report corpus size, embedding dimension, and store path.",
                "inputSchema": { "type": "object", "properties": {} }
            },
            {
                "name": "clear",
                "description": "Remove all documents from the corpus.",
                "inputSchema": { "type": "object", "properties": {} }
            }
        ]
    })
}

fn parse_doc(value: &Value) -> Result<Doc, String> {
    let content = value
        .get("content")
        .and_then(|c| c.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "each document needs a non-empty `content` string".to_string())?
        .to_string();
    let uri = value
        .get("uri")
        .and_then(|u| u.as_str())
        .unwrap_or_default()
        .to_string();
    let keywords = value
        .get("keywords")
        .and_then(|k| k.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Ok(Doc {
        uri,
        content,
        keywords,
    })
}

fn write_msg(out: &mut impl Write, msg: &Value) -> io::Result<()> {
    let line = serde_json::to_string(msg).unwrap_or_else(|_| "{}".to_string());
    out.write_all(line.as_bytes())?;
    out.write_all(b"\n")?;
    out.flush()
}

/// Serve the cascade over the MCP stdio transport: read newline-delimited
/// JSON-RPC requests from stdin, write responses to stdout, until EOF.
///
/// Honours two environment variables:
/// - `OCTACORE_MCP_STORE` — JSON file to load on start and save after each change
///   (omit for an in-memory corpus),
/// - `OCTACORE_MCP_DIM` — embedding dimension (default 256).
pub fn serve_stdio() -> io::Result<()> {
    let dim = std::env::var("OCTACORE_MCP_DIM")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|d| *d > 0)
        .unwrap_or(256);
    let store_path = std::env::var("OCTACORE_MCP_STORE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from);

    let mut mem = Memory::default();
    if let Some(loaded) = store_path
        .as_ref()
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|bytes| serde_json::from_slice::<Memory>(&bytes).ok())
    {
        mem = loaded;
    }
    let mut server = Server {
        mem,
        dim,
        store_path,
        last_query: String::new(),
        last_items: Vec::new(),
        feedback: octasoma::RelevanceFeedback::new(),
    };

    let stdin = io::stdin();
    let mut out = io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let msg: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                write_msg(
                    &mut out,
                    &json!({
                        "jsonrpc": "2.0", "id": Value::Null,
                        "error": { "code": -32700, "message": format!("parse error: {e}") }
                    }),
                )?;
                continue;
            }
        };
        let method = msg
            .get("method")
            .and_then(|m| m.as_str())
            .unwrap_or_default();
        let params = msg.get("params").cloned().unwrap_or(Value::Null);
        // Requests carry an `id` and get a response; notifications (no `id`) do not.
        if let Some(id) = msg.get("id") {
            let resp = server.handle_request(method, params, id.clone());
            write_msg(&mut out, &resp)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn server() -> Server {
        Server {
            mem: Memory::default(),
            dim: 128,
            store_path: None,
            last_query: String::new(),
            last_items: Vec::new(),
            feedback: octasoma::RelevanceFeedback::new(),
        }
    }

    #[test]
    fn initialize_reports_server_info_and_echoes_protocol() {
        let mut s = server();
        let resp = s.handle_request(
            "initialize",
            json!({ "protocolVersion": "2025-06-18" }),
            json!(1),
        );
        assert_eq!(resp["result"]["serverInfo"]["name"], json!("octacore"));
        assert_eq!(resp["result"]["protocolVersion"], json!("2025-06-18"));
        assert!(resp["result"]["capabilities"]["tools"].is_object());
    }

    #[test]
    fn tools_list_exposes_the_five_tools() {
        let mut s = server();
        let resp = s.handle_request("tools/list", Value::Null, json!(2));
        let names: Vec<&str> = resp["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        for expected in ["recall", "remember", "feedback", "stats", "clear"] {
            assert!(names.contains(&expected), "missing tool {expected}");
        }
    }

    /// The explicit relevance channel: labels the last recall by uri, refuses
    /// without one, reports unknown uris, and counts show up in stats.
    #[test]
    fn feedback_labels_the_last_recall_by_uri() {
        let mut s = server();
        // No prior recall → a visible tool-level error.
        let resp = s.handle_request(
            "tools/call",
            json!({ "name": "feedback", "arguments": { "relevant": ["doc:1"] } }),
            json!(10),
        );
        assert_eq!(resp["result"]["isError"], json!(true));

        // Remember + recall, then label the hit.
        s.handle_request(
            "tools/call",
            json!({ "name": "remember", "arguments": {
                "uri": "doc:pool", "content": "manage a pool of reusable database connections" } }),
            json!(11),
        );
        s.handle_request(
            "tools/call",
            json!({ "name": "recall", "arguments": {
                "query": "manage a pool of reusable database connections", "k": 1 } }),
            json!(12),
        );
        let resp = s.handle_request(
            "tools/call",
            json!({ "name": "feedback", "arguments": {
                "relevant": ["doc:pool"], "irrelevant": ["doc:ghost"] } }),
            json!(13),
        );
        assert_eq!(resp["result"]["isError"], json!(false));
        let text = resp["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("Recorded 1 observation"), "{text}");
        assert!(
            text.contains("doc:ghost"),
            "unknown uris are reported: {text}"
        );

        let stats = s.handle_request(
            "tools/call",
            json!({ "name": "stats", "arguments": {} }),
            json!(14),
        );
        let text = stats["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("1 observation(s) (1 relevant)"), "{text}");
    }

    #[test]
    fn remember_then_recall_ranks_the_relevant_document() {
        let mut s = server();
        let add = s.handle_request(
            "tools/call",
            json!({
                "name": "remember",
                "arguments": { "documents": [
                    { "uri": "db", "content": "manage a pool of reusable database connections" },
                    { "uri": "auth", "content": "authenticate a user with username and password" }
                ]}
            }),
            json!(3),
        );
        assert_eq!(add["result"]["isError"], json!(false));

        let rec = s.handle_request(
            "tools/call",
            json!({
                "name": "recall",
                "arguments": { "query": "open a pooled database connection", "k": 1 }
            }),
            json!(4),
        );
        let text = rec["result"]["content"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("db"),
            "expected the db doc to rank first: {text}"
        );
        assert!(
            !text.contains("auth"),
            "k=1 should drop the auth doc: {text}"
        );
    }

    #[test]
    fn recall_without_documents_is_empty_not_an_error() {
        let mut s = server();
        let rec = s.handle_request(
            "tools/call",
            json!({ "name": "recall", "arguments": { "query": "anything" } }),
            json!(5),
        );
        assert_eq!(rec["result"]["isError"], json!(false));
        assert!(
            rec["result"]["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("No matches")
        );
    }

    #[test]
    fn unknown_method_yields_method_not_found() {
        let mut s = server();
        let resp = s.handle_request("does/not/exist", Value::Null, json!(6));
        assert_eq!(resp["error"]["code"], json!(-32601));
    }

    #[test]
    fn unknown_tool_is_reported_as_tool_error() {
        let mut s = server();
        let resp = s.handle_request(
            "tools/call",
            json!({ "name": "nope", "arguments": {} }),
            json!(7),
        );
        assert_eq!(resp["result"]["isError"], json!(true));
    }
}
