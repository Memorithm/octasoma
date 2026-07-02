> **MISSION ACCOMPLISHED (2026-07-02).** This runbook is retired. The
> standalone `CHECKUPAUTO/octacore` repository was published, evolved (it grew
> the MCP server), and its evolution has been **merged back** into this staging
> (PR #22) — which is now the **canonical home** of the crate. The standalone
> repository is archived (read-only); nothing publishes to it anymore. To
> materialise a standalone tree (for an archive refresh or a future re-split),
> `scripts/extract_octacore.sh <DEST>` still works and now pins the OctaSoma
> dependency to a **release tag** instead of a raw commit. The runbook below is
> kept as the historical record of the original publication procedure.

> **STATUS NOTE (2026-07-01).** This runbook was written on the `octacore`
> staging branch, whose root was the standalone crate. That branch has been
> absorbed into `master`: the crate is staged at `octacore/` in this repo
> (with a `path = ".."` dependency on OctaSoma), and the standalone layout —
> git-pinned dependency, root `LICENSE` — is materialised by
> `scripts/extract_octacore.sh <DEST>`. The `master` copy of the crate is
> newer than the old branch (it adds `recall_global_shortlisted` and the
> PolyForm Noncommercial dual-license). Read the steps below with that
> mapping in mind.

# OCTACORE — integration runbook (read me first)

You are Claude Code in a fresh session. Your job: **publish the OctaCore crate in
this branch to the `CHECKUPAUTO/octacore` repository**, autonomously. The human
should not have to do anything else. Don't ask for confirmation unless genuinely
blocked.

## What OctaCore is (keep this framing honest)

The **intelligent assembly of the CHECKUPAUTO memory triad** into one recall cascade
— not a fourth memory:

- **CCOS** (causal) narrows a query to a small causal region.
- **OctaSoma** (semantic) reranks that region by **exact cosine** — the finisher.
  For the **scope-free** case, `recall_global` uses a **SimHash sketch shortlist →
  exact cosine rerank** (`octasoma::SketchIndex`), replacing the ~0% global 3-D router.
- **SLHAv2** (attention) is the inference-time KV-cache kernel that consumes the
  window; OctaSoma serves it as a **visualisation lens**, not a text reranker.

Validated: 99% hit at ~26 tokens/turn (~137× fewer than naive).

## The crate (in this branch)

Root = the standalone crate: `Cargo.toml`, `src/lib.rs`, `examples/`, `README.md`,
`LICENSE`, `.gitignore`, `.github/workflows/ci.yml`, plus this `OCTACORE_HANDOFF.md`
(do **not** publish it — step 4).

- Default build depends only on OctaSoma. `slha`/`ccos` are optional features, their
  adapters verified against CCOS `v0.3.0` and scirust/SLHAv2 `v0.2.0`.
- The OctaSoma dependency is **pinned** to
  `rev = "3f3e7885fb1321d64ce64936a4ee00be7db871de"` (a commit on OctaSoma's
  `master`, which has `SketchIndex` etc.). **Keep this rev.** Once OctaSoma tags a
  release, it can become `branch = "master"` or a version — leave a one-line note in
  `README.md`, don't change the pin now.

## Steps

0. **Write access to `CHECKUPAUTO/octacore`** is required — check it (list branches);
   if not in this session's scope, stop and say so. `CHECKUPAUTO/octasoma` is public.
1. **Get the files.** If this file is in your checkout you have them; else:
   `git fetch https://github.com/CHECKUPAUTO/octasoma octacore && git checkout FETCH_HEAD`.
   Take the files/tree only — not OctaSoma's history.
2. **Verify:** `cargo build` · `cargo test` (unit tests + doctest) · `cargo fmt --all
   -- --check` · `cargo clippy --all-targets -- -D warnings`. Optional:
   `cargo build --features ccos,slha`. Fix before publishing if not green.
3. **Publish** these files at the repo root of `CHECKUPAUTO/octacore` as a clean
   initial commit (detect the default branch; overwrite any auto-generated
   README/LICENSE). Do **not** open a PR unless asked.
4. **Exclude this runbook** (`rm OCTACORE_HANDOFF.md` before committing).
5. **Commit trailers:**
   ```
   Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
   Claude-Session: https://claude.ai/code/session_01U1eByX3Kr7d8zXKRQsRSJn
   ```
6. **Report** the octacore URL + commit pushed, the build/test results, and anything
   blocked.
