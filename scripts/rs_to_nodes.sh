#!/usr/bin/env bash
# Generate a pipeline_bench_text corpus (uri<TAB>content) from a Rust source tree,
# mimicking CCOS node ids (file:… and sym:…). The module is auto-derived from the
# uri (the file path) by the harness, so two columns are enough.
#
#   scripts/rs_to_nodes.sh <SRC_DIR> > nodes.tsv
#   # e.g. scripts/rs_to_nodes.sh ~/CCOS/src > nodes.tsv
#
# Then write a few `query<TAB>target_uri` lines into queries.tsv and run:
#   cargo run --release --example pipeline_bench_text -- \
#     --corpus nodes.tsv --queries queries.tsv \
#     --url http://localhost:11434 --model nomic-embed-text --dim 768
set -u
root="${1:-.}"
[ -d "$root" ] || { echo "rs_to_nodes.sh: no such directory: $root" >&2; exit 1; }

find "$root" -name '*.rs' -type f | while read -r f; do
  rel="${f#"$root"/}"

  # file node: content = the module-level doc (//!) if any, else the path.
  doc="$(grep -m1 -E '^[[:space:]]*//!' "$f" 2>/dev/null | sed -E 's#^[[:space:]]*//!\s?##' || true)"
  printf 'file:%s\t%s\n' "$rel" "${doc:-Rust source file $rel}"

  # symbol nodes: fn / struct / trait / enum → content = the trimmed signature line.
  grep -nE '^[[:space:]]*(pub[[:space:]]+)?(async[[:space:]]+)?(fn|struct|trait|enum)[[:space:]]+[A-Za-z_]' "$f" 2>/dev/null \
  | while IFS= read -r hit; do
      line="${hit#*:}"
      name="$(printf '%s' "$line" | sed -E 's/.*\b(fn|struct|trait|enum)[[:space:]]+([A-Za-z_][A-Za-z0-9_]*).*/\2/')"
      sig="$(printf '%s' "$line" | sed -E 's/^[[:space:]]+//; s/[[:space:]]*\{.*$//; s/\t/ /g')"
      [ -n "$name" ] && printf 'sym:%s:%s\t%s\n' "$rel" "$name" "$sig"
    done
done
