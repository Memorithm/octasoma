//! Embedding providers for the agent layer.
//!
//! The [`Embedder`] trait abstracts "text → vector" so the agent works with any
//! source. Two implementations ship in the box, both dependency-free:
//!
//! - [`HashEmbedder`] — deterministic, offline, *non-semantic* (a hash). Ideal
//!   for tests, demos, and reproducible pipelines.
//! - [`OllamaEmbedder`] — talks to a local Ollama / OpenAI-compatible embedding
//!   endpoint over HTTP using only the standard library (no `reqwest`, no TLS;
//!   plain `http://` for a localhost model server).

use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use crate::DeterministicRng;

/// Error returned by an [`Embedder`].
#[derive(Debug)]
pub enum EmbedError {
    /// Transport / I/O failure (connection refused, timeout, …).
    Io(io::Error),
    /// The endpoint replied but the payload could not be understood.
    Protocol(String),
}

impl std::fmt::Display for EmbedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbedError::Io(e) => write!(f, "embedding I/O error: {e}"),
            EmbedError::Protocol(m) => write!(f, "embedding protocol error: {m}"),
        }
    }
}

impl std::error::Error for EmbedError {}

impl From<io::Error> for EmbedError {
    fn from(e: io::Error) -> Self {
        EmbedError::Io(e)
    }
}

/// Anything that can turn text into a fixed-length embedding vector.
pub trait Embedder {
    /// The dimensionality of the vectors this embedder produces.
    fn dim(&self) -> usize;

    /// Embed a single string.
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError>;

    /// Embed a batch (default: sequential calls to [`Embedder::embed`]).
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
}

// ---------------------------------------------------------------------------
// HashEmbedder — deterministic, dependency-free, non-semantic
// ---------------------------------------------------------------------------

/// A deterministic embedder: the same text always maps to the same unit vector,
/// derived by seeding [`DeterministicRng`] with an FNV-1a hash of the text.
///
/// It is **not semantic** — unrelated texts get unrelated vectors — but it is
/// perfect for tests, demos, and any offline pipeline that just needs stable,
/// reproducible vectors without a model server.
pub struct HashEmbedder {
    dim: usize,
}

impl HashEmbedder {
    /// Creates a hash embedder producing `dim`-dimensional unit vectors.
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "dim must be non-zero");
        Self { dim }
    }

    fn seed_of(text: &str) -> u64 {
        // FNV-1a, 64-bit.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in text.as_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }
}

impl Embedder for HashEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let mut rng = DeterministicRng::new(Self::seed_of(text));
        let mut v: Vec<f32> = (0..self.dim).map(|_| rng.next_f32()).collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
        Ok(v)
    }
}

// ---------------------------------------------------------------------------
// OllamaEmbedder — std-only HTTP client for a local model server
// ---------------------------------------------------------------------------

/// Embeds text via a local Ollama / OpenAI-compatible HTTP endpoint, e.g.
/// `nomic-embed-text` served at `http://localhost:11434/api/embeddings`.
///
/// Uses only `std::net` (plain `http://`, `Connection: close`); intended for a
/// localhost model server. For TLS or remote hosts, implement [`Embedder`] with
/// your preferred HTTP client.
pub struct OllamaEmbedder {
    base_url: String,
    model: String,
    endpoint: String,
    dim: usize,
    timeout: Duration,
}

impl OllamaEmbedder {
    /// Creates a client for `model` at `base_url`, declaring the embedding
    /// dimensionality `dim` (e.g. 768 for `nomic-embed-text`).
    pub fn new(base_url: impl Into<String>, model: impl Into<String>, dim: usize) -> Self {
        assert!(dim > 0, "dim must be non-zero");
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: model.into(),
            endpoint: "/api/embeddings".to_string(),
            dim,
            timeout: Duration::from_secs(60),
        }
    }

    /// Overrides the request path (default `/api/embeddings`).
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Overrides the read/write timeout (default 60 s).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Fallible single embedding (the method [`Embedder::embed`] delegates to).
    pub fn try_embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let body = format!(
            "{{\"model\":{},\"prompt\":{}}}",
            json_string(&self.model),
            json_string(text)
        );
        let url = format!("{}{}", self.base_url, self.endpoint);
        let response = http_post_json(&url, &body, self.timeout)?;
        let vec = extract_float_array(&response, "embedding").ok_or_else(|| {
            EmbedError::Protocol("response had no numeric \"embedding\" array".to_string())
        })?;
        if vec.is_empty() {
            return Err(EmbedError::Protocol("empty embedding".to_string()));
        }
        Ok(vec)
    }
}

impl Embedder for OllamaEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.try_embed(text)
    }
}

// ---------------------------------------------------------------------------
// Minimal std-only HTTP + JSON helpers
// ---------------------------------------------------------------------------

/// POSTs `body` as JSON to an `http://` URL and returns the response body.
fn http_post_json(url: &str, body: &str, timeout: Duration) -> io::Result<String> {
    let rest = url.strip_prefix("http://").ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "only http:// URLs are supported",
        )
    })?;
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let (host, port) = match authority.rfind(':') {
        Some(i) => (
            &authority[..i],
            authority[i + 1..].parse::<u16>().unwrap_or(80),
        ),
        None => (authority, 80u16),
    };

    let mut stream = TcpStream::connect((host, port))?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;

    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\n\
         Content-Length: {len}\r\nConnection: close\r\n\r\n{body}",
        len = body.len()
    );
    stream.write_all(request.as_bytes())?;
    stream.flush()?;

    let mut raw = String::new();
    stream.read_to_string(&mut raw)?;

    let body_start = raw
        .find("\r\n\r\n")
        .map(|i| i + 4)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "malformed HTTP response"))?;
    Ok(raw[body_start..].to_string())
}

/// Escapes a string as a JSON string literal (including surrounding quotes).
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Extracts the first JSON array of numbers that follows `"key"` in `json`.
/// Tolerant by design: it scans for `"key"`, the next `[`, and the matching `]`.
fn extract_float_array(json: &str, key: &str) -> Option<Vec<f32>> {
    let pattern = format!("\"{key}\"");
    let key_pos = json.find(&pattern)?;
    let after = &json[key_pos + pattern.len()..];
    let lb = after.find('[')?;
    let rb = after[lb..].find(']')? + lb;
    let inner = &after[lb + 1..rb];

    let mut out = Vec::new();
    for tok in inner.split(',') {
        let t = tok.trim();
        if t.is_empty() {
            continue;
        }
        match t.parse::<f32>() {
            Ok(v) => out.push(v),
            Err(_) => return None,
        }
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_embedder_is_deterministic_and_unit() {
        let e = HashEmbedder::new(64);
        let a = e.embed("hello world").unwrap();
        let b = e.embed("hello world").unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
        let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
        // Different text → different vector.
        assert_ne!(a, e.embed("goodbye").unwrap());
    }

    #[test]
    fn json_string_escapes() {
        assert_eq!(json_string("a\"b\\c\n"), "\"a\\\"b\\\\c\\n\"");
    }

    #[test]
    fn extract_embedding_array() {
        let resp = r#"{"model":"x","embedding":[0.5, -1.25, 3.0],"done":true}"#;
        let v = extract_float_array(resp, "embedding").unwrap();
        assert_eq!(v, vec![0.5, -1.25, 3.0]);
        assert!(extract_float_array(resp, "missing").is_none());
    }

    #[test]
    fn extract_rejects_non_numeric() {
        let resp = r#"{"embedding":[0.1, "oops", 0.2]}"#;
        assert!(extract_float_array(resp, "embedding").is_none());
    }
}
