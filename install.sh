#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# OctaSoma installer — builds and installs the `octasoma` command-line memory.
#
#   ./install.sh            # build, test, and install the `octasoma` CLI
#   ./install.sh --no-test  # skip the test step
#   ./install.sh --build    # just build (no install)
# ---------------------------------------------------------------------------
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[-]${NC} $*" >&2; exit 1; }

RUN_TESTS=1
DO_INSTALL=1
for arg in "$@"; do
  case "$arg" in
    --no-test) RUN_TESTS=0 ;;
    --build)   DO_INSTALL=0 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) err "unknown option: $arg (try --help)" ;;
  esac
done

# 1. Toolchain check.
if ! command -v cargo >/dev/null 2>&1; then
  err "Rust/Cargo not found. Install it from https://rustup.rs :
      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
fi
log "using $(cargo --version)"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 2. Build.
log "building (release) ..."
cargo build --release

# 3. Test (optional).
if [ "$RUN_TESTS" -eq 1 ]; then
  log "running the test suite ..."
  cargo test --release
fi

# 4. Install the CLI.
if [ "$DO_INSTALL" -eq 1 ]; then
  log "installing the 'octasoma' CLI to ~/.cargo/bin ..."
  cargo install --path . --force
  echo ""
  log "============================================================"
  log " OctaSoma installed."
  if ! command -v octasoma >/dev/null 2>&1; then
    warn "~/.cargo/bin is not on your PATH. Add this to your shell profile:"
    echo '      export PATH="$HOME/.cargo/bin:$PATH"'
  fi
  log " Try it:"
  log '   octasoma --hash remember "I prefer dark mode"'
  log '   octasoma --hash recall   "what do I prefer?"'
  log "   octasoma help"
  log "============================================================"
else
  log "build complete (skipped install). Binary at ./target/release/octasoma"
fi
