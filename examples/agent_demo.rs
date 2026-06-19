//! OctaSoma agent demo — 100% Rust, fully offline.
//!
//! Mirrors the perceive / reflect loop an LLM agent would run. It uses the
//! built-in deterministic [`HashEmbedder`] so it needs no model server; swap in
//! `OllamaEmbedder::new("http://localhost:11434", "nomic-embed-text", 768)` to
//! use a real embedding model with no other change.
//!
//! Run with: `cargo run --release --example agent_demo`

use octasoma::{HashEmbedder, OctaSomaAgent, OllamaEmbedder};

fn main() {
    println!("OctaSoma agent demo (offline, HashEmbedder)\n");

    // 1. Calibrate a PCA projection from a small corpus, then build the agent.
    let corpus = [
        "Rust's ownership system guarantees memory safety at compile time.",
        "Python is widely used for data science and rapid prototyping.",
        "Fractal geometry describes patterns that repeat at every scale.",
        "Octrees subdivide 3-D space into eight octants recursively.",
        "Machine learning turns raw data into predictive models.",
    ];
    let mut agent = OctaSomaAgent::calibrate(HashEmbedder::new(256), &corpus).expect("calibration");

    // 2. Perception loop — store observations as they arrive.
    for obs in [
        "Async Rust with the tokio runtime is highly performant.",
        "PyO3 bridges Rust and Python, but this project is pure Rust now.",
        "A loose octree expands node bounds to catch boundary points.",
    ] {
        agent.perceive(obs).expect("perceive");
    }
    println!("stored memories: {}", agent.len());

    // 3. Reflection loop — retrieve context for a prompt.
    let query = "Async Rust with the tokio runtime is highly performant.";
    let context = agent.reflect(query, 3).expect("reflect");
    println!(
        "\nquery:   {query}\ncontext:\n  {}",
        context.replace('\n', "\n  ")
    );

    // 4. Persist and reload.
    let path = "/tmp/octasoma_agent_demo.frac";
    agent.save(path).expect("save");
    let reloaded = OctaSomaAgent::from_file(HashEmbedder::new(256), path).expect("load");
    println!("\nreloaded {} memories from {path}", reloaded.len());
    std::fs::remove_file(path).ok();

    // 5. The same agent, against a real model server (requires Ollama running):
    //    let mut agent = OctaSomaAgent::new(
    //        OllamaEmbedder::new("http://localhost:11434", "nomic-embed-text", 768), 42);
    let _ = OllamaEmbedder::new("http://localhost:11434", "nomic-embed-text", 768);
    println!("\ndone.");
}
