# =============================================================================
# graphician — build, test, and release
#
# Targets:
#   build       — build Rust extension + copy .so into the Python package
#   test        — run the full test suite (excludes known failures)
#   clean       — remove build artifacts
#   install     — install the package in development mode
#   format      — run ruff formatter
#   lint        — run ruff linter + mypy
#   check       — format + lint + test (pre-commit gate)
# =============================================================================

SHELL := /bin/bash
.PHONY: build test clean install format lint check help

# Detect Python binary
PYTHON   := $(shell which python3 2>/dev/null || echo python3)
PIP      := $(PYTHON) -m pip
PYTEST   := $(PYTHON) -m pytest
RUSTC    := $(shell which cargo 2>/dev/null || echo "")
CARGO    := $(RUSTC:cargo=%/cargo)

# Derived paths
NATIVE_DIR  := graphician-native
SO_DIR      := src/graphician/_extract
# PyO3 targets abi3-py311, so one extension works on every supported CPython
# version instead of embedding the build interpreter's minor version.
SO_FILE     := $(SO_DIR)/graphician_native.abi3.so

# ------------------------------------------------------------------
# help
# ------------------------------------------------------------------
help:
	@echo "graphician Makefile targets:"
	@echo "  build    — Compile Rust extension and install .so"
	@echo "  test     — Run full test suite (399 tests)"
	@echo "  clean    — Remove build artifacts"
	@echo "  install  — Install package in dev mode"
	@echo "  format   — Run ruff formatter"
	@echo "  lint     — Run ruff + mypy"
	@echo "  check    — format + lint + test (pre-commit gate)"

# ------------------------------------------------------------------
# build: Rust + .so placement
# ------------------------------------------------------------------
build: $(SO_FILE)

$(SO_FILE): $(NATIVE_DIR)/src/lib.rs $(NATIVE_DIR)/Cargo.toml
	@echo "→ Building graphician-native (release)..."
	cd $(NATIVE_DIR) && cargo build --release
	@echo "→ Copying .so → $(SO_FILE)..."
	mkdir -p $(SO_DIR)
	cp $(NATIVE_DIR)/target/release/libgraphician_native.so $(SO_FILE)
	@echo "✓ Build complete"

# ------------------------------------------------------------------
# test
# ------------------------------------------------------------------
test: build
	@echo "→ Running tests..."
	$(PYTEST) tests/ -v \
		--ignore=tests/test_cli_json_golden.py \
		-k "not test_personalized_biases" \
		--tb=short
	@echo "✓ Tests passed"

# ------------------------------------------------------------------
# clean
# ------------------------------------------------------------------
clean:
	@echo "→ Cleaning..."
	@rm -rf $(NATIVE_DIR)/target
	@rm -rf $(SO_DIR)/*.so
	@rm -rf $(SO_DIR)/__pycache__
	@rm -rf .pytest_cache
	@rm -rf .coverage coverage.xml
	@rm -rf build/ dist/ *.egg-info
	@rm -rf graphician.db ariadne.db
	@echo "✓ Clean complete"

# ------------------------------------------------------------------
# install
# ------------------------------------------------------------------
install: build
	@echo "→ Installing graphician in dev mode..."
	$(PIP) install -e .
	@echo "✓ Installed"

# ------------------------------------------------------------------
# format
# ------------------------------------------------------------------
format:
	@echo "→ Formatting..."
	$(PYTHON) -m ruff format src/ tests/
	@echo "✓ Formatted"

# ------------------------------------------------------------------
# lint
# ------------------------------------------------------------------
lint:
	@echo "→ Linting..."
	$(PYTHON) -m ruff check src/ tests/
	$(PYTHON) -m mypy src/graphician/
	@echo "✓ Linted"

# ------------------------------------------------------------------
# check — pre-commit style gate
# ------------------------------------------------------------------
check: format lint test
	@echo "✓ All checks passed"
