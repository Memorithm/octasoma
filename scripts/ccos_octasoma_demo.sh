#!/usr/bin/env bash
# Decoupled CCOS ⟷ OctaSoma bridge over MCP — no crate coupling, just the server.
# CCOS ingests its node URIs into OctaSoma, then asks for a semantic recall and
# gets back a RecallWindow { strategy, items:[{uri,score,kind,content}], tokens }.
# Offline (--hash) recall is exact-text; with Ollama (--url/--model) it is semantic.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[+] building octasoma-mcp (--features mcp) ..."
cargo build --release --features mcp >/dev/null 2>&1

STORE="$(mktemp -u).frac"
trap 'rm -f "$STORE"' EXIT

echo "[+] MCP session — CCOS-style nodes in, semantic recall out:"
printf '%s\n' \
'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
'{"jsonrpc":"2.0","method":"notifications/initialized"}' \
'{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"ingest","arguments":{"uri":"sym:src/auth.rs:login","text":"user login and authentication flow"}}}' \
'{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"ingest","arguments":{"uri":"sym:src/db.rs:query","text":"SQL query builder and connection pool"}}}' \
'{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"ingest","arguments":{"uri":"mod:src/cache.rs","text":"in-memory LRU cache for hot keys"}}}' \
'{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"recall","arguments":{"text":"SQL query builder and connection pool","k":3}}}' \
'{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"stats"}}' \
  | ./target/release/octasoma-mcp "$STORE" --hash

echo
echo "[i] id:5 is a CCOS RecallWindow — drop-in for CCOS or any MCP agent."
echo "[i] swap --hash for --url/--model to get real semantic recall via Ollama."
