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

use std::io::{self, BufRead, Read, Write};
use std::path::{Path, PathBuf};

use octasoma::{EmbedError, HashEmbedder};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{Cascade, CausalScope, RecallWindow, ScopeItem};

const SERVER_NAME: &str = "octacore";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");
const DEFAULT_PROTOCOL: &str = "2024-11-05";

/// Authoritative server-side limits for the MCP trust boundary.
const MAX_REQUEST_BYTES: usize = 1 << 20;
const MAX_QUERY_BYTES: usize = 32 << 10;
const MAX_CONTENT_BYTES: usize = 128 << 10;
const MAX_URI_BYTES: usize = 4 << 10;
const MAX_KEYWORD_BYTES: usize = 512;
const MAX_KEYWORDS_PER_DOC: usize = 64;
const MAX_DOCS_PER_REQUEST: usize = 64;
const MAX_CORPUS_DOCS: usize = 10_000;
const MAX_CORPUS_BYTES: usize = 16 << 20;
const MAX_STORE_BYTES: usize = 64 << 20;
const MAX_K: usize = 32;
const MAX_BUDGET_TOKENS: usize = 8_192;
const MAX_DIM: usize = 16_384;
const MAX_FEEDBACK_URIS: usize = MAX_K * 2;
const MAX_FEEDBACK_ENTRIES: usize = 1_024;
const MAX_PROTOCOL_BYTES: usize = 64;
const MAX_TOOL_NAME_BYTES: usize = 64;

enum InputLine {
    Eof,
    Line(String),
    TooLong,
    InvalidUtf8,
}

fn discard_until_newline<R: BufRead>(reader: &mut R) -> io::Result<()> {
    loop {
        let (consume, found_newline) = {
            let buf = reader.fill_buf()?;
            if buf.is_empty() {
                return Ok(());
            }
            match buf.iter().position(|byte| *byte == b'\n') {
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

fn bounded_required_string(args: &Value, key: &str, max_bytes: usize) -> Result<String, String> {
    let value = args
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("`{key}` must be a string"))?;
    let value = value.trim();
    if value.is_empty() {
        return Err(format!("`{key}` must be non-empty"));
    }
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

fn bounded_uri_list(args: &Value, key: &str) -> Result<Vec<String>, String> {
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
                    "`{key}[{index}]` exceeds the {MAX_URI_BYTES}-byte limit"
                ));
            }
            Ok(uri.to_string())
        })
        .collect()
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

/// One stored document: its content, an id (`uri`), and optional causal keywords.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Doc {
    uri: String,
    content: String,
    #[serde(default)]
    keywords: Vec<String>,
}

impl Doc {
    fn payload_bytes(&self) -> Result<usize, String> {
        let mut total = self
            .uri
            .len()
            .checked_add(self.content.len())
            .ok_or_else(|| "document size overflow".to_string())?;
        for keyword in &self.keywords {
            total = total
                .checked_add(keyword.len())
                .ok_or_else(|| "document size overflow".to_string())?;
        }
        Ok(total)
    }

    fn validate(&self) -> Result<(), String> {
        if self.content.trim().is_empty() {
            return Err("document content must be non-empty".to_string());
        }
        if self.content.len() > MAX_CONTENT_BYTES {
            return Err(format!(
                "document content exceeds the {MAX_CONTENT_BYTES}-byte limit"
            ));
        }
        if self.uri.len() > MAX_URI_BYTES {
            return Err(format!(
                "document uri exceeds the {MAX_URI_BYTES}-byte limit"
            ));
        }
        if self.keywords.len() > MAX_KEYWORDS_PER_DOC {
            return Err(format!(
                "document exceeds the {MAX_KEYWORDS_PER_DOC}-keyword limit"
            ));
        }
        for (index, keyword) in self.keywords.iter().enumerate() {
            if keyword.is_empty() {
                return Err(format!("keyword {index} must be non-empty"));
            }
            if keyword.len() > MAX_KEYWORD_BYTES {
                return Err(format!(
                    "keyword {index} exceeds the {MAX_KEYWORD_BYTES}-byte limit"
                ));
            }
        }
        Ok(())
    }
}

/// The in-memory corpus the cascade recalls over (optionally persisted to disk).
#[derive(Default, Serialize, Deserialize)]
struct Memory {
    docs: Vec<Doc>,
}

impl Memory {
    fn validate(&self) -> Result<(), String> {
        if self.docs.len() > MAX_CORPUS_DOCS {
            return Err(format!(
                "corpus exceeds the {MAX_CORPUS_DOCS}-document limit"
            ));
        }

        let mut total = 0usize;
        for (index, doc) in self.docs.iter().enumerate() {
            doc.validate()
                .map_err(|error| format!("document {index}: {error}"))?;
            total = total
                .checked_add(doc.payload_bytes()?)
                .ok_or_else(|| "corpus size overflow".to_string())?;
            if total > MAX_CORPUS_BYTES {
                return Err(format!(
                    "corpus exceeds the {MAX_CORPUS_BYTES}-byte payload limit"
                ));
            }
        }
        Ok(())
    }

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
            .and_then(Value::as_str)
            .filter(|protocol| protocol.len() <= MAX_PROTOCOL_BYTES)
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
            .and_then(Value::as_str)
            .ok_or_else(|| "missing tool `name`".to_string())?;
        if name.len() > MAX_TOOL_NAME_BYTES {
            return Err(format!(
                "tool name exceeds the {MAX_TOOL_NAME_BYTES}-byte limit"
            ));
        }
        let args = params
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| json!({}));
        let outcome = match name {
            "recall" => self.tool_recall(&args),
            "remember" => self.tool_remember(&args),
            "feedback" => self.tool_feedback(&args),
            "stats" => Ok(self.tool_stats()),
            "clear" => self.tool_clear(),
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
        let query = bounded_required_string(args, "query", MAX_QUERY_BYTES)?;
        let k = bounded_usize(args, "k", 5, MAX_K)?;
        let budget = bounded_usize(args, "budget_tokens", 256, MAX_BUDGET_TOKENS)?;

        let window = self
            .mem
            .recall(&query, k, budget, self.dim)
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
        let mut to_add = Vec::new();

        if let Some(raw_documents) = args.get("documents") {
            let documents = raw_documents
                .as_array()
                .ok_or_else(|| "`documents` must be an array".to_string())?;
            if documents.len() > MAX_DOCS_PER_REQUEST {
                return Err(format!(
                    "`documents` exceeds the {MAX_DOCS_PER_REQUEST}-item limit"
                ));
            }
            for document in documents {
                to_add.push(parse_doc(document)?);
            }
        } else if args.get("content").is_some() {
            to_add.push(parse_doc(args)?);
        } else {
            return Err("provide `documents` (array) or a single `content` string".to_string());
        }

        if to_add.is_empty() {
            return Err("no documents to remember".to_string());
        }

        let resulting_count = self
            .mem
            .docs
            .len()
            .checked_add(to_add.len())
            .ok_or_else(|| "corpus document count overflow".to_string())?;
        if resulting_count > MAX_CORPUS_DOCS {
            return Err(format!(
                "corpus would exceed the {MAX_CORPUS_DOCS}-document limit"
            ));
        }

        let original_len = self.mem.docs.len();
        for mut document in to_add {
            if document.uri.trim().is_empty() {
                document.uri = format!("doc:{}", self.mem.docs.len() + 1);
            }
            self.mem.docs.push(document);
        }

        let added = self.mem.docs.len() - original_len;
        if let Err(error) = self.mem.validate().and_then(|()| self.persist()) {
            self.mem.docs.truncate(original_len);
            return Err(error);
        }

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
        let relevant = bounded_uri_list(args, "relevant")?;
        let irrelevant = bounded_uri_list(args, "irrelevant")?;

        let mut observations = Vec::new();
        let mut unknown = Vec::new();

        for (list, label) in [(relevant, true), (irrelevant, false)] {
            for uri in list {
                match self.last_items.iter().find(|(stored, _)| *stored == uri) {
                    Some((matched_uri, score)) => {
                        observations.push((matched_uri.clone(), *score, label));
                    }
                    None => unknown.push(uri),
                }
            }
        }

        ensure_feedback_capacity(self.feedback.len(), observations.len())?;

        let recorded = observations.len();

        for (uri, score, label) in observations {
            self.feedback.record(&self.last_query, &uri, score, label);
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

    fn tool_clear(&mut self) -> Result<String, String> {
        let old_docs = std::mem::take(&mut self.mem.docs);
        let cleared = old_docs.len();

        if let Err(error) = self.persist() {
            self.mem.docs = old_docs;
            return Err(error);
        }

        Ok(format!("Cleared {cleared} document(s)."))
    }

    /// Persist the corpus by syncing a same-directory temporary file and
    /// atomically renaming it over the previous store.
    fn persist(&self) -> Result<(), String> {
        let Some(path) = &self.store_path else {
            return Ok(());
        };

        self.mem.validate()?;
        let bytes = serde_json::to_vec(&self.mem)
            .map_err(|error| format!("store serialization failed: {error}"))?;
        if bytes.len() > MAX_STORE_BYTES {
            return Err(format!(
                "serialized store exceeds the {MAX_STORE_BYTES}-byte limit"
            ));
        }

        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("store directory creation failed: {error}"))?;
        }

        let file_name = path
            .file_name()
            .ok_or_else(|| "store path has no file name".to_string())?
            .to_string_lossy();
        let temporary = path.with_file_name(format!(".{file_name}.tmp-{}", std::process::id()));

        let _ = std::fs::remove_file(&temporary);

        let write_result = (|| -> io::Result<()> {
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            std::fs::rename(&temporary, path)?;
            Ok(())
        })();

        if let Err(error) = write_result {
            let _ = std::fs::remove_file(&temporary);
            return Err(format!("store persistence failed: {error}"));
        }

        Ok(())
    }
}

/// The four tools the server advertises, with JSON-Schema input definitions.
fn tools_list() -> Value {
    json!({
        "tools": [
            {
                "name": "recall",
                "description": "Recall a token-budgeted, semantically reranked context window.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_QUERY_BYTES
                        },
                        "k": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_K,
                            "default": 5
                        },
                        "budget_tokens": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_BUDGET_TOKENS,
                            "default": 256
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "remember",
                "description": "Add bounded documents to the recall corpus.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "documents": {
                            "type": "array",
                            "maxItems": MAX_DOCS_PER_REQUEST,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "uri": {
                                        "type": "string",
                                        "maxLength": MAX_URI_BYTES
                                    },
                                    "content": {
                                        "type": "string",
                                        "minLength": 1,
                                        "maxLength": MAX_CONTENT_BYTES
                                    },
                                    "keywords": {
                                        "type": "array",
                                        "maxItems": MAX_KEYWORDS_PER_DOC,
                                        "items": {
                                            "type": "string",
                                            "minLength": 1,
                                            "maxLength": MAX_KEYWORD_BYTES
                                        }
                                    }
                                },
                                "required": ["content"]
                            }
                        },
                        "content": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_CONTENT_BYTES
                        },
                        "uri": {
                            "type": "string",
                            "maxLength": MAX_URI_BYTES
                        },
                        "keywords": {
                            "type": "array",
                            "maxItems": MAX_KEYWORDS_PER_DOC,
                            "items": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": MAX_KEYWORD_BYTES
                            }
                        }
                    }
                }
            },
            {
                "name": "feedback",
                "description": "Label documents from the last recall.",
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
            },
            {
                "name": "stats",
                "description": "Report corpus size, embedding dimension, and store path.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "clear",
                "description": "Remove all documents from the corpus.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    })
}

fn parse_doc(value: &Value) -> Result<Doc, String> {
    let content = bounded_required_string(value, "content", MAX_CONTENT_BYTES)?;

    let uri = match value.get("uri") {
        None => String::new(),
        Some(raw) => {
            let uri = raw
                .as_str()
                .ok_or_else(|| "`uri` must be a string".to_string())?;
            if uri.len() > MAX_URI_BYTES {
                return Err(format!("`uri` exceeds the {MAX_URI_BYTES}-byte limit"));
            }
            uri.to_string()
        }
    };

    let keywords = match value.get("keywords") {
        None => Vec::new(),
        Some(raw) => {
            let values = raw
                .as_array()
                .ok_or_else(|| "`keywords` must be an array of strings".to_string())?;
            if values.len() > MAX_KEYWORDS_PER_DOC {
                return Err(format!(
                    "`keywords` exceeds the {MAX_KEYWORDS_PER_DOC}-item limit"
                ));
            }

            values
                .iter()
                .enumerate()
                .map(|(index, raw_keyword)| {
                    let keyword = raw_keyword
                        .as_str()
                        .ok_or_else(|| {
                            format!("`keywords[{index}]` must be a string")
                        })?
                        .trim();
                    if keyword.is_empty() {
                        return Err(format!(
                            "`keywords[{index}]` must be non-empty"
                        ));
                    }
                    if keyword.len() > MAX_KEYWORD_BYTES {
                        return Err(format!(
                            "`keywords[{index}]` exceeds the                              {MAX_KEYWORD_BYTES}-byte limit"
                        ));
                    }
                    Ok(keyword.to_string())
                })
                .collect::<Result<Vec<_>, String>>()?
        }
    };

    let document = Doc {
        uri,
        content,
        keywords,
    };
    document.validate()?;
    Ok(document)
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn load_memory(path: &Path) -> io::Result<Memory> {
    let metadata = match std::fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Ok(Memory::default());
        }
        Err(error) => return Err(error),
    };

    if metadata.len() > MAX_STORE_BYTES as u64 {
        return Err(invalid_data(format!(
            "store exceeds the {MAX_STORE_BYTES}-byte limit"
        )));
    }

    let file = std::fs::File::open(path)?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take((MAX_STORE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)?;

    if bytes.len() > MAX_STORE_BYTES {
        return Err(invalid_data(format!(
            "store exceeds the {MAX_STORE_BYTES}-byte limit"
        )));
    }

    let memory: Memory = serde_json::from_slice(&bytes)
        .map_err(|error| invalid_data(format!("invalid store JSON: {error}")))?;
    memory.validate().map_err(invalid_data)?;
    Ok(memory)
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
    let dim = match std::env::var("OCTACORE_MCP_DIM") {
        Ok(raw) => raw
            .parse::<usize>()
            .ok()
            .filter(|dimension| (1..=MAX_DIM).contains(dimension))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("OCTACORE_MCP_DIM must be between 1 and {MAX_DIM}"),
                )
            })?,
        Err(std::env::VarError::NotPresent) => 256,
        Err(error) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid OCTACORE_MCP_DIM: {error}"),
            ));
        }
    };

    let store_path = std::env::var("OCTACORE_MCP_STORE")
        .ok()
        .filter(|value| !value.is_empty())
        .map(PathBuf::from);

    let mem = match &store_path {
        Some(path) => load_memory(path)?,
        None => Memory::default(),
    };

    let mut server = Server {
        mem,
        dim,
        store_path,
        last_query: String::new(),
        last_items: Vec::new(),
        feedback: octasoma::RelevanceFeedback::new(),
    };

    let stdin = io::stdin();
    let mut input = stdin.lock();
    let mut out = io::stdout().lock();

    loop {
        let line = match read_input_line(&mut input)? {
            InputLine::Eof => break,
            InputLine::TooLong => {
                write_msg(
                    &mut out,
                    &json!({
                        "jsonrpc": "2.0",
                        "id": Value::Null,
                        "error": {
                            "code": -32600,
                            "message": format!(
                                "request exceeds the {MAX_REQUEST_BYTES}-byte limit"
                            )
                        }
                    }),
                )?;
                continue;
            }
            InputLine::InvalidUtf8 => {
                write_msg(
                    &mut out,
                    &json!({
                        "jsonrpc": "2.0",
                        "id": Value::Null,
                        "error": {
                            "code": -32700,
                            "message": "parse error: invalid UTF-8"
                        }
                    }),
                )?;
                continue;
            }
            InputLine::Line(line) => line,
        };

        if line.trim().is_empty() {
            continue;
        }

        let msg: Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(error) => {
                write_msg(
                    &mut out,
                    &json!({
                        "jsonrpc": "2.0",
                        "id": Value::Null,
                        "error": {
                            "code": -32700,
                            "message": format!("parse error: {error}")
                        }
                    }),
                )?;
                continue;
            }
        };

        let method = msg
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let params = msg.get("params").cloned().unwrap_or(Value::Null);

        if let Some(id) = msg.get("id") {
            let response = server.handle_request(method, params, id.clone());
            write_msg(&mut out, &response)?;
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
    #[test]
    fn request_reader_rejects_long_lines_and_recovers() {
        let mut input = Vec::new();
        input.extend(std::iter::repeat_n(b'x', MAX_REQUEST_BYTES + 17));
        input.extend_from_slice(b"\n{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"ping\"}\n");

        let mut cursor = io::Cursor::new(input);
        assert!(matches!(
            read_input_line(&mut cursor).unwrap(),
            InputLine::TooLong
        ));
        match read_input_line(&mut cursor).unwrap() {
            InputLine::Line(line) => assert!(line.contains("\"id\":2")),
            _ => panic!("reader did not recover after an overlong line"),
        }
    }

    #[test]
    fn tool_arguments_and_schema_are_bounded() {
        let mut server = server();

        let too_many = (0..=MAX_DOCS_PER_REQUEST)
            .map(|index| {
                json!({
                    "uri": format!("doc:{index}"),
                    "content": "bounded"
                })
            })
            .collect::<Vec<_>>();
        let response = server.handle_request(
            "tools/call",
            json!({
                "name": "remember",
                "arguments": { "documents": too_many }
            }),
            json!(20),
        );
        assert_eq!(response["result"]["isError"], json!(true));
        assert!(
            response["result"]["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("64-item limit")
        );

        let response = server.handle_request(
            "tools/call",
            json!({
                "name": "recall",
                "arguments": {
                    "query": "bounded",
                    "k": MAX_K + 1
                }
            }),
            json!(21),
        );
        assert_eq!(response["result"]["isError"], json!(true));
        assert!(
            response["result"]["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("between 1 and 32")
        );

        let schema = tools_list().to_string();
        assert!(schema.contains("\"maximum\":32"));
        assert!(schema.contains("\"maximum\":8192"));
        assert!(schema.contains("\"maxItems\":64"));
        assert!(schema.contains("\"maxLength\":131072"));
    }

    #[test]
    fn invalid_and_oversized_stores_are_rejected() {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "octacore_invalid_store_{}_{}.json",
            std::process::id(),
            "limits"
        ));
        std::fs::remove_file(&path).ok();

        std::fs::write(&path, b"{not-json").unwrap();
        let error = load_memory(&path).err().expect("invalid JSON rejected");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);

        let file = std::fs::File::create(&path).unwrap();
        file.set_len((MAX_STORE_BYTES + 1) as u64).unwrap();
        let error = load_memory(&path)
            .err()
            .expect("oversized store rejected before allocation");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn persistence_is_atomic_and_roundtrips() {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "octacore_atomic_store_{}_{}.json",
            std::process::id(),
            "roundtrip"
        ));
        std::fs::remove_file(&path).ok();

        let mut server = Server {
            mem: Memory::default(),
            dim: 128,
            store_path: Some(path.clone()),
            last_query: String::new(),
            last_items: Vec::new(),
            feedback: octasoma::RelevanceFeedback::new(),
        };

        let response = server.handle_request(
            "tools/call",
            json!({
                "name": "remember",
                "arguments": {
                    "uri": "doc:atomic",
                    "content": "persist this document atomically"
                }
            }),
            json!(30),
        );
        assert_eq!(response["result"]["isError"], json!(false));

        let loaded = load_memory(&path).unwrap();
        assert_eq!(loaded.docs.len(), 1);
        assert_eq!(loaded.docs[0].uri, "doc:atomic");

        let file_name = path.file_name().unwrap().to_string_lossy();
        let temporary = path.with_file_name(format!(".{file_name}.tmp-{}", std::process::id()));
        assert!(!temporary.exists(), "temporary file was not cleaned up");

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn feedback_log_is_bounded_across_calls() {
        let mut server = server();

        let remember = server.handle_request(
            "tools/call",
            json!({
                "name": "remember",
                "arguments": {
                    "uri": "doc:bounded",
                    "content": "bounded relevance feedback"
                }
            }),
            json!(40),
        );
        assert_eq!(remember["result"]["isError"], json!(false));

        let recall = server.handle_request(
            "tools/call",
            json!({
                "name": "recall",
                "arguments": {
                    "query": "bounded relevance feedback",
                    "k": 1
                }
            }),
            json!(41),
        );
        assert_eq!(recall["result"]["isError"], json!(false));

        let batch = vec!["doc:bounded"; MAX_FEEDBACK_URIS];

        for id in 0..(MAX_FEEDBACK_ENTRIES / MAX_FEEDBACK_URIS) {
            let response = server.handle_request(
                "tools/call",
                json!({
                    "name": "feedback",
                    "arguments": {
                        "relevant": batch
                    }
                }),
                json!(100 + id),
            );

            assert_eq!(response["result"]["isError"], json!(false), "{response}");
        }

        assert_eq!(server.feedback.len(), MAX_FEEDBACK_ENTRIES);

        let overflow = server.handle_request(
            "tools/call",
            json!({
                "name": "feedback",
                "arguments": {
                    "relevant": ["doc:bounded"]
                }
            }),
            json!(200),
        );

        assert_eq!(overflow["result"]["isError"], json!(true));

        let error = overflow["result"]["content"][0]["text"].as_str().unwrap();

        assert!(error.contains("1024-observation session limit"), "{error}");

        assert_eq!(server.feedback.len(), MAX_FEEDBACK_ENTRIES);
    }
}
