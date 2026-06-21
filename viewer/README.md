# OctaSoma memory viewer

A dependency-free, **offline** 3-D viewer for an exported memory — no server, no
build, no network.

1. Export your store: `octasoma export memory.json`
   (or in Rust: `std::fs::write("memory.json", mem.export_points_json(1_000_000))`)
2. Open `index.html` in any browser.
3. Drop `memory.json` onto the page (or use the file picker).

**Controls:** drag to rotate · scroll to zoom · hover a point to read the memory.

Points are coloured by payload prefix, so same-theme memories share a hue, and the
spatial layout *is* the projection — you are literally looking at OctaSoma's 3-D
memory. Very large stores are sub-sampled for smooth rotation.

## Colour by precision score

A *scored* export colours each point by its exact cosine similarity to a query
instead of by category — cold blue (unrelated) → hot red (on-query) — so the
region a query actually retrieves lights up. The legend becomes a score gradient
and hovering shows the per-memory score.

```bash
cargo run --release --example scored_viz          # synthetic demo → scored.json
cargo run --release --example scored_viz -- vecs.tsv   # first row is the query
```

In Rust it is one call on a [`HybridMemory`]:

```rust
std::fs::write("scored.json", mem.export_scored_json(&query, 1_000_000));
```

Drop `scored.json` here exactly like any other export — the viewer detects the
`"scored"` flag and switches to the heat map automatically.

## See an SLHAv2 KV-cache

`cargo run --release --example kv_cache_viz` projects 128-dim tile latents (à la
SLHAv2) to 3-D and writes `kv_cache.json` — drop it here to *see your KV-cache*,
coloured by attention head. Feed your real tiles as TSV:

```bash
cargo run --release --example kv_cache_viz -- latents.tsv
```

one tile per line, `label⇥f0 f1 … f127`, where the 128 floats come from
`SciRustSlhaTile::dequant_latent()` and the label is e.g. `"head 3 tok 12"`.
