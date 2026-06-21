//! A simulated agent turn driven by the [`MemoryKernel`] — 100% offline.
//!
//! Shows the three things you wire into an LLM agent:
//!   1. the system-prompt fragment (how the model should treat memory),
//!   2. the per-turn context injection (`kernel.step`),
//!   3. the tool schema for function-calling agents.
//!
//! Run with: `cargo run --release --example kernel_loop`

use octasoma::{HashEmbedder, KernelConfig, MEMORY_TOOL_SCHEMA_JSON, MemoryKernel, OctaSomaAgent};

fn main() {
    let config = KernelConfig {
        top_k: 3,
        min_observation_chars: 4,
        ..Default::default()
    };
    let mut memory = MemoryKernel::new(OctaSomaAgent::new(HashEmbedder::new(256), 42), config);

    println!("================ 1. SYSTEM PROMPT (give this to the LLM) ================\n");
    println!("{}\n", memory.system_prompt());

    // Durable facts observed over time (e.g. extracted from past conversations).
    for fact in [
        "The user is named Sam and is based in Berlin.",
        "The user prefers concise, direct answers.",
        "The current project is a Rust octree memory engine called OctaSoma.",
    ] {
        memory.observe(fact).unwrap();
    }
    println!("(stored {} durable memories)\n", memory.len());

    // ---- one agent turn ----
    let user_message = "The user prefers concise, direct answers.";
    let step = memory
        .step(user_message, /* remember_input = */ false)
        .unwrap();

    println!("================ 2. CONTEXT INJECTED THIS TURN ================\n");
    println!("user message: {user_message:?}\n");
    println!(
        "{}",
        if step.context.is_empty() {
            "(no relevant memories)".into()
        } else {
            step.context.clone()
        }
    );
    println!("→ prepend the block above to your LLM prompt, then add the user message.\n");

    println!("================ 3. TOOL SCHEMA (for function-calling agents) ================\n");
    println!("{MEMORY_TOOL_SCHEMA_JSON}");
    println!(
        "\nWire `memory_store` → kernel.observe(text), `memory_recall` → kernel.recall_context(query)."
    );
}
