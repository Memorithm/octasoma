//! End-to-end smoke tests for the `octasoma-mcp` server (sharded, stdio JSON-RPC).
//!
//! Only built under `--features mcp` (the binary's required feature); drives the
//! real binary over stdin/stdout with `HashEmbedder` (offline, deterministic).
#![cfg(feature = "mcp")]

use std::io::Write;
use std::process::{Command, Stdio};

/// Runs one server session against `store`, writing raw bytes to stdin and
/// returning everything written to stdout.
fn session_input(store: &str, input: &[u8]) -> String {
    let mut child = Command::new(env!("CARGO_BIN_EXE_octasoma-mcp"))
        .args([store, "--hash", "--dim", "128"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn octasoma-mcp");
    {
        let mut stdin = child.stdin.take().unwrap();
        stdin.write_all(input).unwrap();
        // stdin dropped here → EOF → server loop ends.
    }
    let out = child.wait_with_output().expect("wait for octasoma-mcp");
    assert!(out.status.success(), "server failed: {:?}", out.status);
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Runs one server session against `store`, piping one JSON-RPC request per line.
fn session(store: &str, requests: &[&str]) -> String {
    let mut input = requests.join("\n");
    input.push('\n');
    session_input(store, input.as_bytes())
}

fn unique_store(tag: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("octasoma_mcp_{tag}_{}", std::process::id()));
    std::fs::remove_dir_all(&p).ok();
    p.push("store");
    p
}

#[test]
fn ingest_recall_explain_stats() {
    let store = unique_store("rt");
    let s = store.to_str().unwrap();

    let out = session(
        s,
        &[
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"ingest","arguments":{"uri":"sym:src/db.rs:query","text":"build and run SQL queries"}}}"#,
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"ingest","arguments":{"uri":"sym:src/db.rs:pool","text":"a pool of db connections"}}}"#,
            r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"ingest","arguments":{"uri":"sym:src/auth.rs:login","text":"authenticate a user"}}}"#,
            r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"recall","arguments":{"text":"a pool of db connections","region":"src/db.rs","k":2}}}"#,
            r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"explain","arguments":{"text":"a pool of db connections","region":"src/db.rs","k":1}}}"#,
            r#"{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"stats"}}"#,
        ],
    );

    // initialize handshake
    assert!(
        out.contains("serverInfo"),
        "missing initialize result: {out}"
    );
    // ingest auto-derived the causal region from the CCOS uri
    assert!(out.contains("src/db.rs"), "region not derived: {out}");
    assert!(
        out.contains("src/auth.rs"),
        "auth region not derived: {out}"
    );
    // scoped recall returns the hit and is the precise (sketch → exact rerank) strategy
    assert!(
        out.contains("sym:src/db.rs:pool"),
        "recall hit missing: {out}"
    );
    assert!(out.contains("precise"), "strategy missing: {out}");
    // explain returns a 3-D point + zoom path
    assert!(out.contains("query_point"), "explain point missing: {out}");
    assert!(
        out.contains("zoom_path"),
        "explain zoom path missing: {out}"
    );
    // stats: 3 memories across 2 regions
    assert!(out.contains("\\\"memories\\\":3"), "memories count: {out}");
    assert!(out.contains("\\\"regions\\\":2"), "regions count: {out}");

    std::fs::remove_dir_all(store.parent().unwrap()).ok();
}

#[test]
fn scoped_recall_excludes_other_regions() {
    let store = unique_store("scope");
    let s = store.to_str().unwrap();

    let out = session(
        s,
        &[
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"ingest","arguments":{"uri":"sym:src/db.rs:query","text":"build and run SQL queries"}}}"#,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"ingest","arguments":{"uri":"sym:src/auth.rs:login","text":"authenticate a user"}}}"#,
            // recall scoped to auth must never surface a db node
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"recall","arguments":{"text":"build and run SQL queries","region":"src/auth.rs","k":5}}}"#,
        ],
    );

    // The id:3 recall line is the last; it must not contain the db uri.
    let recall_line = out.lines().find(|l| l.contains("\"id\":3")).unwrap_or("");
    assert!(
        !recall_line.contains("sym:src/db.rs:query"),
        "scoped recall leaked another region: {recall_line}"
    );
    assert!(
        recall_line.contains("sym:src/auth.rs:login"),
        "scoped recall missed its own region: {recall_line}"
    );

    std::fs::remove_dir_all(store.parent().unwrap()).ok();
}

#[test]
fn store_persists_across_reopen() {
    let store = unique_store("persist");
    let s = store.to_str().unwrap();

    session(
        s,
        &[
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"ingest","arguments":{"uri":"sym:src/db.rs:query","text":"build and run SQL queries"}}}"#,
        ],
    );
    // Fresh process, same store directory.
    let out = session(
        s,
        &[r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"stats"}}"#],
    );
    assert!(
        out.contains("\\\"memories\\\":1"),
        "store did not persist across reopen: {out}"
    );

    std::fs::remove_dir_all(store.parent().unwrap()).ok();
}

// -- record layer over MCP ---------------------------------------------------

/// remember → visible recall → tombstone → hidden → purge; every step through
/// the real binary, the whole lifecycle in one stdio session.
#[test]
fn record_lifecycle_over_mcp() {
    let store = unique_store("lifecycle");
    let s = store.to_str().unwrap();

    let out = session(
        s,
        &[
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#,
            // Durable memory + one whose TTL already ran out.
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"remember","arguments":{"uri":"sym:src/db.rs:pool","text":"a pool of db connections","tenant":"acme","workspace":"platform","agent":"coder","expires_at_ms":9999999999999,"created_at_ms":1000}}}"#,
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"remember","arguments":{"uri":"sym:src/db.rs:cache","text":"a stale cache note","expires_at_ms":1000}}}"#,
            // Plain ingest stays visible without any record (back-compat).
            r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"ingest","arguments":{"uri":"sym:src/db.rs:schema","text":"the database schema"}}}"#,
            r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"recall","arguments":{"text":"database connections and schema","region":"src/db.rs","k":10,"now_ms":5000}}}"#,
            r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"tombstone","arguments":{"uri":"sym:src/db.rs:pool"}}}"#,
            r#"{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"recall","arguments":{"text":"database connections and schema","region":"src/db.rs","k":10,"now_ms":5000}}}"#,
            r#"{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"stats"}}"#,
            r#"{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"purge","arguments":{"now_ms":5000}}}"#,
            r#"{"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"compact","arguments":{"now_ms":5000,"region":"src/db.rs"}}}"#,
        ],
    );

    let line = |id: u32| {
        out.lines()
            .find(|l| l.contains(&format!("\"id\":{id}")))
            .unwrap_or("")
    };

    let recall5 = line(5);
    assert!(
        recall5.contains("sym:src/db.rs:pool") && recall5.contains("sym:src/db.rs:schema"),
        "visible recall missed live items: {recall5}"
    );
    assert!(
        !recall5.contains("sym:src/db.rs:cache"),
        "expired record leaked into a now_ms recall: {recall5}"
    );
    assert!(recall5.contains("visible_as_of"), "now_ms echo missing");

    let recall7 = line(7);
    assert!(
        !recall7.contains("sym:src/db.rs:pool"),
        "tombstoned record leaked: {recall7}"
    );
    assert!(
        recall7.contains("sym:src/db.rs:schema"),
        "plain payload lost after tombstone elsewhere: {recall7}"
    );

    assert!(
        line(8).contains("\\\"records\\\":2"),
        "two remembers = two records; plain ingest adds none: {}",
        line(8)
    );
    let purge = line(9);
    // Only the tombstoned `pool` is purgeable — `cache` is TTL-hidden but its
    // status is still Active, so it stays in the record layer. Both *hidden
    // index entries* (pool + cache) are reclaimed by the pre-purge compaction.
    assert!(purge.contains("\\\"removed\\\":1"), "purge count: {purge}");
    assert!(
        purge.contains("\\\"index_reclaimed\\\":2"),
        "purge did not reclaim hidden index entries: {purge}"
    );
    let compact = line(10);
    assert!(
        compact.contains("\\\"memories\\\":1"),
        "only the plain schema item should survive: {compact}"
    );

    std::fs::remove_dir_all(store.parent().unwrap()).ok();
}

/// Non-monotonic generations are refused before anything is written.
#[test]
fn remember_rejects_stale_generations() {
    let store = unique_store("monotonic");
    let s = store.to_str().unwrap();

    let out = session(
        s,
        &[
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"remember","arguments":{"uri":"sym:r:x","text":"first version of the fact"}}}"#,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"remember","arguments":{"uri":"sym:r:x","text":"stale rewrite","generation":1}}}"#,
        ],
    );

    let stale = out.lines().find(|l| l.contains("\"id\":2")).unwrap_or("");
    assert!(
        stale.contains("isError\\\":true") || stale.contains("must be strictly greater"),
        "stale generation was not refused: {stale}"
    );
    assert!(
        stale.contains("must be strictly greater than the current 1"),
        "unexpected refusal message: {stale}"
    );

    std::fs::remove_dir_all(store.parent().unwrap()).ok();
}

#[test]
fn malformed_and_oversized_requests_are_visible_and_recoverable() {
    let store = unique_store("bounded-lines");
    let s = store.to_str().unwrap();

    let huge_request = format!(
        "{{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{{\"name\":\"ingest\",\"arguments\":{{\"text\":\"{}\"}}}}}}",
        "x".repeat((1 << 20) + 64)
    );
    let input = format!(
        "{{not-json\n{huge_request}\n{{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"ping\"}}\n"
    );

    let out = session_input(s, input.as_bytes());

    assert!(
        out.contains("\"code\":-32700"),
        "parse error missing: {out}"
    );
    assert!(
        out.contains("request exceeds the 1048576-byte limit"),
        "oversized-request error missing: {out}"
    );
    assert!(
        out.lines().any(|line| line.contains("\"id\":3")),
        "server did not recover after the oversized line: {out}"
    );

    std::fs::remove_dir_all(store.parent().unwrap()).ok();
}

#[test]
fn tool_arguments_are_bounded_before_work_is_performed() {
    let store = unique_store("tool-limits");
    let s = store.to_str().unwrap();

    let oversized_text = "x".repeat((128 << 10) + 1);
    let ingest = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "ingest",
            "arguments": { "text": oversized_text }
        }
    });
    let recall = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "recall",
            "arguments": { "text": "bounded query", "k": 33 }
        }
    });
    let input = format!("{ingest}\n{recall}\n");

    let out = session_input(s, input.as_bytes());

    assert!(
        out.contains("`text` exceeds the 131072-byte limit"),
        "text limit was not enforced: {out}"
    );
    assert!(
        out.contains("`k` must be between 1 and 32"),
        "k limit was not enforced: {out}"
    );

    std::fs::remove_dir_all(store.parent().unwrap()).ok();
}

#[test]
fn tool_schema_advertises_server_limits() {
    let store = unique_store("schema-limits");
    let s = store.to_str().unwrap();

    let out = session(s, &[r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#]);

    assert!(out.contains("\"maximum\":32"), "k maximum missing: {out}");
    assert!(
        out.contains("\"maxLength\":131072"),
        "text maxLength missing: {out}"
    );
    assert!(
        out.contains("\"maxItems\":64"),
        "feedback maxItems missing: {out}"
    );

    std::fs::remove_dir_all(store.parent().unwrap()).ok();
}

#[test]
fn feedback_log_is_bounded_across_calls() {
    let store = unique_store("feedback-capacity");
    let s = store.to_str().unwrap();

    let uri = "sym:src/db.rs:query";
    let mut requests = vec![
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "ingest",
                "arguments": {
                    "uri": uri,
                    "text": "build bounded SQL queries"
                }
            }
        })
        .to_string(),
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "recall",
                "arguments": {
                    "text": "build bounded SQL queries",
                    "k": 1
                }
            }
        })
        .to_string(),
    ];

    let full_batch = vec![uri; 64];
    for id in 3..19 {
        requests.push(
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": "tools/call",
                "params": {
                    "name": "feedback",
                    "arguments": {
                        "relevant": full_batch
                    }
                }
            })
            .to_string(),
        );
    }

    requests.push(
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 19,
            "method": "tools/call",
            "params": {
                "name": "feedback",
                "arguments": {
                    "relevant": [uri]
                }
            }
        })
        .to_string(),
    );

    requests.push(
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 20,
            "method": "tools/call",
            "params": {
                "name": "stats",
                "arguments": {}
            }
        })
        .to_string(),
    );

    let mut input = requests.join("\n");
    input.push('\n');
    let out = session_input(s, input.as_bytes());

    let overflow = out
        .lines()
        .find(|line| line.contains("\"id\":19"))
        .unwrap_or("");
    assert!(
        overflow.contains("1024-observation session limit"),
        "feedback overflow was not rejected: {overflow}"
    );

    let stats = out
        .lines()
        .find(|line| line.contains("\"id\":20"))
        .unwrap_or("");
    assert!(
        stats.contains("\\\"feedback_recorded\\\":1024"),
        "feedback log changed after rejected batch: {stats}"
    );

    std::fs::remove_dir_all(store.parent().unwrap()).ok();
}

/// Tenant scoping hides other companies' records at recall time, and compaction
/// under a tenant filter reclaims exactly that tenant's unreachable entries.
#[test]
fn scoped_recall_and_compact_isolate_tenants() {
    let store = unique_store("tenants");
    let s = store.to_str().unwrap();

    let out = session(
        s,
        &[
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"remember","arguments":{"uri":"sym:t:acme","text":"alpha fact","tenant":"acme"}}}"#,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"remember","arguments":{"uri":"sym:t:beta","text":"beta fact","tenant":"other-co"}}}"#,
            // acme's recall must not surface the other company's memory.
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"recall","arguments":{"text":"fact","region":"t","now_ms":5000,"tenant":"acme","k":10}}}"#,
            // Compaction *under the other-co filter* reclaims exactly acme's
            // entry (it is unreachable for that tenant) and nothing else.
            r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"compact","arguments":{"now_ms":5000,"tenant":"other-co"}}}"#,
            // The unscoped recall afterwards still returns beta; alpha's index
            // entry was reclaimed (its record remains in the layer).
            r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"recall","arguments":{"text":"fact","region":"t","now_ms":5000,"k":10}}}"#,
        ],
    );

    let line = |id: u32| {
        out.lines()
            .find(|l| l.contains(&format!("\"id\":{id}")))
            .unwrap_or("")
    };

    let recall3 = line(3);
    assert!(
        recall3.contains("sym:t:acme") && !recall3.contains("sym:t:beta"),
        "tenant scoping failed: {recall3}"
    );

    let compact4 = line(4);
    assert!(
        compact4.contains("\\\"reclaimed_total\\\":1"),
        "scoped compaction did not reclaim exactly one entry: {compact4}"
    );

    let recall5 = line(5);
    assert!(
        recall5.contains("sym:t:beta") && !recall5.contains("sym:t:acme"),
        "post-compaction recall wrong: {recall5}"
    );

    std::fs::remove_dir_all(store.parent().unwrap()).ok();
}
