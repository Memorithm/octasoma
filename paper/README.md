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

Latency and throughput are machine-dependent; recall figures are deterministic
given the seeds in the harness. See [`../docs/evaluation.md`](../docs/evaluation.md).

## Before submission

- Set `\author{...}` and affiliation (currently a placeholder).
- If you prefer external bibliography management, the entries are mirrored in
  `refs.bib`; switch each `thebibliography` block to `\bibliography{../refs}` and
  run `bibtex`.
