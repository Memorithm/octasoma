//! Integration tests for the agent + memory-kernel layers.
use octasoma::{
    Embedder, HashEmbedder, KernelConfig, MEMORY_TOOL_SCHEMA_JSON, MemoryKernel, OctaSomaAgent,
    OllamaEmbedder,
};

#[test]
fn agent_perceive_recall_roundtrip() {
    let mut a = OctaSomaAgent::new(HashEmbedder::new(128), 1);
    for t in [
        "alpha beta gamma",
        "delta epsilon zeta",
        "octrees subdivide space",
    ] {
        a.perceive(t).unwrap();
    }
    assert_eq!(a.len(), 3);
    // Deterministic embedder ⇒ querying stored text recalls itself.
    assert_eq!(
        a.recall("octrees subdivide space", 1).unwrap(),
        vec!["octrees subdivide space".to_string()]
    );
}

#[test]
fn agent_save_and_reload() {
    let mut a = OctaSomaAgent::new(HashEmbedder::new(96), 4);
    for t in [
        "one memory here",
        "another memory there",
        "a third recollection",
    ] {
        a.perceive(t).unwrap();
    }
    let path = format!("/tmp/octasoma_agent_{}.frac", std::process::id());
    a.save(&path).unwrap();
    let b = OctaSomaAgent::from_file(HashEmbedder::new(96), &path).unwrap();
    assert_eq!(b.len(), 3);
    assert_eq!(
        b.recall("another memory there", 1).unwrap(),
        vec!["another memory there".to_string()]
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn agent_calibrate_does_not_store_corpus() {
    let corpus = ["topic a", "topic b", "topic c", "topic d", "topic e"];
    let agent = OctaSomaAgent::calibrate(HashEmbedder::new(64), &corpus).unwrap();
    assert!(agent.is_empty());
    assert_eq!(agent.core().high_dim, 64);
}

#[test]
fn kernel_observe_policy_and_step() {
    let mut k = MemoryKernel::with_defaults(HashEmbedder::new(128), 7);
    assert!(!k.observe("tiny").unwrap()); // below default min length
    assert!(k.observe("the user is based in Europe").unwrap());
    assert!(k.observe("the user prefers metric units").unwrap());

    let step = k.step("the user prefers metric units", false).unwrap();
    assert!(step.context.contains("Relevant memories"));
    assert!(step.context.contains("metric"));
    assert!(!step.stored_input);

    let step2 = k.step("a fresh durable statement to keep", true).unwrap();
    assert!(step2.stored_input);
    assert_eq!(k.len(), 3);
}

#[test]
fn kernel_context_truncation_and_empty() {
    let mut k = MemoryKernel::with_defaults(HashEmbedder::new(64), 1);
    assert_eq!(k.recall_context("nothing yet").unwrap(), "");
    k.config_mut().max_context_chars = 50;
    k.config_mut().min_observation_chars = 1;
    for i in 0..20 {
        k.observe(&format!("memory {i} padded with extra words here"))
            .unwrap();
    }
    let ctx = k
        .recall_context("memory 3 padded with extra words here")
        .unwrap();
    assert!(
        ctx.len() <= 50,
        "context not truncated: {} bytes",
        ctx.len()
    );
}

#[test]
fn kernel_autosave_writes_file() {
    let path = format!("/tmp/octasoma_kernel_autosave_{}.frac", std::process::id());
    let cfg = KernelConfig {
        autosave_path: Some(path.clone()),
        autosave_every: 3,
        min_observation_chars: 1,
        ..KernelConfig::default()
    };
    let mut k = MemoryKernel::new(OctaSomaAgent::new(HashEmbedder::new(64), 0), cfg);
    for i in 0..3 {
        k.observe(&format!("observation {i}")).unwrap();
    }
    assert!(
        std::path::Path::new(&path).exists(),
        "autosave did not write the store"
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn kernel_exposes_system_prompt_and_tools() {
    let k = MemoryKernel::with_defaults(HashEmbedder::new(32), 0);
    assert!(k.system_prompt().contains("Relevant memories"));
    assert!(MEMORY_TOOL_SCHEMA_JSON.contains("memory_store"));
    assert!(MEMORY_TOOL_SCHEMA_JSON.contains("memory_recall"));
    assert!(MEMORY_TOOL_SCHEMA_JSON.contains("memory_explain"));
}

#[test]
fn agent_zoom_and_path() {
    let mut a = OctaSomaAgent::new(HashEmbedder::new(128), 9);
    for i in 0..500 {
        a.perceive(&format!("memory number {i}")).unwrap();
    }
    let q = "memory number 123";
    // Root zoom covers everything.
    let root = a.zoom(q, 0, 1).unwrap().unwrap();
    assert_eq!(root.count, a.len());
    // The path narrows from the whole memory toward the query.
    let path = a.zoom_path(q, 16, 1).unwrap();
    assert_eq!(path[0].count, a.len());
    assert!(path.last().unwrap().count <= path[0].count);
}

#[test]
fn kernel_explain_is_auditable() {
    let mut k = MemoryKernel::with_defaults(HashEmbedder::new(128), 3);
    k.observe("the user lives in Berlin").unwrap();
    k.observe("the user prefers metric units").unwrap();
    let e = k
        .explain("the user prefers metric units", 2)
        .unwrap()
        .unwrap();
    assert!(!e.neighbors.is_empty());
    // The deterministic embedder makes the queried text its own nearest memory.
    assert_eq!(
        String::from_utf8_lossy(&e.neighbors[0].payload),
        "the user prefers metric units"
    );
    assert!(e.neighbors[0].distance < 1e-4);
}

#[test]
fn ollama_embedder_unreachable_errors_without_panicking() {
    // Port 1 on loopback is refused immediately — exercises the error path
    // (not the happy path, which needs a running model server).
    let e = OllamaEmbedder::new("http://127.0.0.1:1", "model", 8);
    assert!(e.embed("hello").is_err());
}
