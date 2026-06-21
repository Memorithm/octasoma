# OctaSoma — Integration kernel: giving an AI agent a memory

This guide tells you **what to do with this memory and how to wire it into an
agent's system**. There are three layers; pick the one that matches your stack:

| Layer | Type | Use it when |
|---|---|---|
| [`MemoryKernel`](#1-the-memory-kernel-rust) | Rust API | You control the agent loop in Rust |
| [System prompt + tools](#2-wiring-into-the-llm) | Prompt/JSON | You drive an LLM (any language) and want it to use memory |
| [`octasoma` CLI](#3-the-cli-for-quick-use) | Shell | Quick use / scripting / no code |

The engine retrieves *topically* relevant memories (see
[evaluation.md](evaluation.md)); treat it as a **coarse semantic router**, ideal
for a focused set of facts, preferences, tasks, or personas.

---

## 1. The memory kernel (Rust)

`MemoryKernel<E>` is the opinionated routine on top of the raw agent. It decides
**what to store**, **what to retrieve**, and **how to hand context to the model**.

```rust
use octasoma::{HashEmbedder, KernelConfig, MemoryKernel, OctaSomaAgent};

// Build a kernel (swap HashEmbedder for OllamaEmbedder in production).
let config = KernelConfig { top_k: 5, ..Default::default() };
let mut memory = MemoryKernel::new(OctaSomaAgent::new(HashEmbedder::new(768), 42), config);

// --- the cognitive loop, once per agent turn ---
let user_msg = "remind me what database we chose";

// 1. Retrieve context to prepend to the LLM prompt.
let step = memory.step(user_msg, /* remember_input = */ false)?;
let prompt = format!("{}\n\n{}\n\nUser: {}", memory.system_prompt(), step.context, user_msg);
// → send `prompt` to your LLM …

// 2. After the model replies, store any durable facts it (or the user) produced.
memory.observe("We chose PostgreSQL for the project")?;
```

### What the loop does

```
user message ─▶ kernel.step(msg)
                  ├─ recall_context(msg)  → "## Relevant memories\n- …"
                  └─ (optional) observe(msg)
              ─▶ prompt = system_prompt() + context + msg ─▶ LLM ─▶ reply
              ─▶ kernel.observe(reply or extracted facts)
```

### Policy (`KernelConfig`)

| Field | Default | Meaning |
|---|---|---|
| `top_k` | 5 | Memories retrieved per turn |
| `context_header` | `## Relevant memories` | Header for the injected block |
| `bullet` | `- ` | Per-memory prefix |
| `min_observation_chars` | 8 | Skip storing trivially short text |
| `max_context_chars` | 2000 | Hard cap on injected context |
| `autosave_path` / `autosave_every` | none / 0 | Persist every *N* stored memories |

### Key methods

- `observe(text) -> bool` — store a memory if it passes policy.
- `recall_context(query) -> String` — prompt-ready context block.
- `step(input, remember_input) -> MemoryStep` — one turn (retrieve + optional store).
- `system_prompt() -> String` — the fragment below, parameterised by your header.
- `save()` / `save_to(path)` — persist the `.frac` store.

Run it: `cargo run --release --example kernel_loop`.

---

## 2. Wiring into the LLM

### 2a. System prompt (context-injection pattern)

Prepend this to your system prompt (this is exactly what `kernel.system_prompt()`
returns). It tells the model how to treat the injected memory block:

> You have a long-term semantic memory. When relevant, recalled memories are
> provided to you in a section titled "## Relevant memories". Use them as
> background recollections: prefer information that is consistent and recent,
> treat them as fallible (they may be partial or outdated), and never invent
> memories that are not listed. If the user states a durable fact, preference, or
> decision, assume it will be remembered for future turns. Do not mention the
> memory mechanism unless asked.

Each turn, your runtime calls `recall_context(user_msg)` and inserts the returned
block before the user's message. The model never calls anything — your code does
the retrieval. This is the simplest and most reliable pattern.

### 2b. Tool calling (model-driven pattern)

If your agent uses function/tool calling, expose two tools and let the model
decide when to store and recall. The ready-made schema is
`octasoma::MEMORY_TOOL_SCHEMA_JSON` (OpenAI/Anthropic-compatible):

```json
[
  { "name": "memory_store",
    "description": "Persist a durable fact, preference, decision, or observation …",
    "input_schema": { "type": "object",
      "properties": { "text": { "type": "string" } }, "required": ["text"] } },
  { "name": "memory_recall",
    "description": "Retrieve memories relevant to a query before answering …",
    "input_schema": { "type": "object",
      "properties": { "query": { "type": "string" },
                      "top_k": { "type": "integer", "default": 5 } },
      "required": ["query"] } }
]
```

Wire the tool handlers to the kernel:

| Tool call | Handler |
|---|---|
| `memory_store{ text }` | `kernel.observe(text)` |
| `memory_recall{ query, top_k }` | `kernel.recall_context(query)` (or `agent.recall`) |

Add to your system prompt: *"Call `memory_store` whenever the user states
something worth remembering across sessions. Call `memory_recall` before
answering questions that may depend on earlier context."*

### Which pattern?

- **Context-injection (2a)** — deterministic, cheap, no extra round-trips. Best
  default. Your code decides retrieval; the model just reads.
- **Tool-calling (2b)** — flexible; the model chooses when to remember/recall.
  Costs extra turns and depends on the model's judgement.

You can combine them: auto-inject context every turn **and** expose
`memory_store` so the model can save salient facts explicitly.

---

## 3. The CLI (for quick use)

No code required — see the [README install section](../README.md#install):

```bash
octasoma remember "I prefer dark mode and the metric system"
octasoma recall   "what are my preferences?"
octasoma reflect  "preferences" -k 3      # prints a prompt-ready block
octasoma stats
```

Use `--hash` for a fully offline (non-semantic) store, or point at a local model
with `--url`/`--model`/`--dim` (defaults to Ollama `nomic-embed-text`).

---

## Lifecycle & good practice

1. **Calibrate (optional but recommended).** Build the projection with PCA from a
   representative corpus — `MemoryKernel::calibrated(embedder, &corpus, cfg)` — so
   topical recall is as high as possible. Calibration does **not** store the
   corpus; it only learns the 3-D projection.
2. **Store durable, self-contained facts.** Prefer *"The user prefers metric
   units"* over a whole transcript. The `min_observation_chars` gate filters
   noise. The text you store is exactly what is returned on recall.
3. **Persist.** Set `autosave_every`/`autosave_path`, or call `save_to(path)` at
   checkpoints. The store is a single portable `.frac` file.
4. **Mind the dimensionality.** Three projected dimensions resolve only a handful
   of dominant themes well (see [evaluation.md](evaluation.md)). Keep each store
   thematically focused; shard by domain/persona if you have many topics.
5. **Concurrency.** The kernel mutates `&mut self`; wrap it in your runtime's lock
   (or keep one kernel per agent task) if you share it across threads.

## Minimal end-to-end (pseudo-code, any language)

```text
on_startup:
    memory = load_or_create_store()
    system = SYSTEM_PROMPT                       # section 2a

on_user_message(msg):
    context = memory.recall_context(msg)         # "## Relevant memories\n- …"
    reply   = llm(system, context, msg)
    for fact in extract_durable_facts(msg, reply):
        memory.observe(fact)                      # store what matters
    persist_if_needed(memory)
    return reply
```
