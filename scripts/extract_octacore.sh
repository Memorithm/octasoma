#!/usr/bin/env bash
# Materialise the staged `octacore/` crate into a standalone crate directory, ready
# to push to https://github.com/CHECKUPAUTO/octacore.
#
#   scripts/extract_octacore.sh <DEST_DIR>
#
# It copies the crate (minus build artifacts) and rewrites the OctaSoma dependency
# from the local path to a git dependency pinned to the commit this crate is verified
# against, then prints the git commands to publish it.
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
src="$here/octacore"
dest="${1:?usage: scripts/extract_octacore.sh <DEST_DIR>}"

# OctaSoma commit octacore is pinned to — a commit on `master` (the API octacore
# needs, incl. SketchIndex, is now merged). Bump when octacore needs a newer API;
# switch to a released version/tag once OctaSoma publishes one.
rev="3f3e7885fb1321d64ce64936a4ee00be7db871de"
octasoma_url="https://github.com/CHECKUPAUTO/octasoma"
octacore_url="https://github.com/CHECKUPAUTO/octacore"

[ -d "$src" ] || { echo "no octacore/ crate at $src" >&2; exit 1; }
mkdir -p "$dest"

for f in Cargo.toml src examples README.md .gitignore .github LICENSE; do
  [ -e "$src/$f" ] && cp -r "$src/$f" "$dest/"
done

# local path dependency -> pinned git dependency
sed -i.bak \
  -e "s|^octasoma = { path = \"\.\.\" }|octasoma = { git = \"$octasoma_url\", rev = \"$rev\" }|" \
  "$dest/Cargo.toml"
rm -f "$dest/Cargo.toml.bak"

echo "OctaCore standalone crate written to: $dest"
echo
echo "Verify and publish:"
echo "  cd \"$dest\""
echo "  cargo build && cargo test          # sanity (pulls octasoma @ $rev)"
echo "  git init -b main && git add . && git commit -m 'OctaCore: initial crate'"
echo "  git remote add origin $octacore_url.git"
echo "  git push -u origin main"
