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

## See an SLHAv2 KV-cache

`cargo run --release --example kv_cache_viz` projects 128-dim tile latents (à la
SLHAv2) to 3-D and writes `kv_cache.json` — drop it here to *see your KV-cache*,
coloured by attention head. Feed your real tiles as TSV:

```bash
cargo run --release --example kv_cache_viz -- latents.tsv
```

one tile per line, `label⇥f0 f1 … f127`, where the 128 floats come from
`SciRustSlhaTile::dequant_latent()` and the label is e.g. `"head 3 tok 12"`.
