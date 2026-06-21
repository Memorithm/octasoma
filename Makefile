# OctaSoma — common developer & user commands.
.PHONY: help build test stress lint fmt fmt-check bench demo kernel cli install doc clean

help:           ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS=":.*?## "} {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

build:          ## Build the library, CLI, and examples (release)
	cargo build --release --bins --examples

test:           ## Run the full test suite
	cargo test --release

stress:         ## Run the heavy soak tests (1M inserts, churn)
	cargo test --release --test stress -- --ignored --nocapture

lint:           ## Clippy with warnings denied
	cargo clippy --all-targets -- -D warnings

fmt:            ## Format the code
	cargo fmt --all

fmt-check:      ## Check formatting (CI)
	cargo fmt --all -- --check

bench:          ## Run the evaluation benchmark
	cargo run --release --example benchmark

demo:           ## Run the offline agent demo
	cargo run --release --example agent_demo

kernel:         ## Run the memory-kernel agent-loop demo
	cargo run --release --example kernel_loop

cli:            ## Build just the CLI
	cargo build --release --bin octasoma

install:        ## Install the `octasoma` CLI to ~/.cargo/bin
	cargo install --path . --force

doc:            ## Build and open the API docs
	cargo doc --no-deps --open

clean:          ## Remove build artifacts
	cargo clean
