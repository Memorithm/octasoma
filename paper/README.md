# OctaSoma — Paper sources

arXiv-style sources for the OctaSoma paper, in English and French.

```
paper/
├── en/main.tex   # English (authoritative source)
├── fr/main.tex   # French translation
├── refs.bib      # shared bibliography (optional; the .tex files also embed it)
└── README.md
```

## Building

The documents are **self-contained**: each `main.tex` embeds its bibliography in
a `thebibliography` environment, so no `.bib` or `bibtex` run is required.

With a TeX distribution (TeX Live / MiKTeX):

```bash
cd paper/en && pdflatex main.tex && pdflatex main.tex   # twice: refs + figure
cd ../fr   && pdflatex main.tex && pdflatex main.tex
```

Or with `latexmk`:

```bash
latexmk -pdf paper/en/main.tex
latexmk -pdf paper/fr/main.tex
```

The figure uses `pgfplots`; the French version additionally uses
`babel`'s `french` option. Both packages ship with standard TeX distributions
and are supported on arXiv. Only standard packages are used, so the sources are
arXiv-ready as-is.

## Reproducing the numbers

Every figure in the paper comes from the engine's benchmark harness:

```bash
cargo run --release --example benchmark -- 50000 256 16 500 10    # Table 1
for C in 4 16 64 256; do
  cargo run --release --example benchmark -- 20000 128 $C 500 10  # Table 2 / Figure 1
done
```

The cascade (Table 3, the inference-pyramid section) is reproduced on a real
codebase with a local embedding model:

```bash
bash scripts/rs_to_nodes.sh <SRC_DIR> > nodes.tsv                 # e.g. ~/CCOS/src
grep '^sym:' nodes.tsv | awk -F'\t' '{n=$1; sub(/.*:/,"",n); print "what does " n " do?\t" $1}' > queries.tsv
cargo run --release --example pipeline_bench_text -- \
  --corpus nodes.tsv --queries queries.tsv \
  --url http://localhost:11434 --model nomic-embed-text --dim 768
```

Latency and throughput are machine-dependent; synthetic recall figures are
deterministic given the seeds in the harness. The cascade figures depend on the
corpus and embedding model (the paper reports a 795-node run). See
[`../docs/evaluation.md`](../docs/evaluation.md) and
[`../docs/integration-ecosystem.md`](../docs/integration-ecosystem.md).

## Before submission

- Set `\author{...}` and affiliation (currently a placeholder).
- If you prefer external bibliography management, the entries are mirrored in
  `refs.bib`; switch each `thebibliography` block to `\bibliography{../refs}` and
  run `bibtex`.
