# Graphician

A **local-first code graph** for navigating, reviewing, and reasoning about a codebase.

Graphician parses source code across multiple languages, builds a typed property graph, and exposes 40+ operations for structural analysis — all running locally with no external dependencies required.

## Pipeline

![Graphician pipeline](static/graphician.svg)

Source code is parsed by tree-sitter into an AST, assembled into a property graph (~10K nodes, ~25K edges), and exposed through a CLI or JSON tool API for structural analysis.

## Features

- **Multi-language parsing** — C/C++, Rust, Java, TypeScript, JavaScript, Python (via tree-sitter)
- **Code graph** — nodes for functions, classes, modules, calls, imports, data flows, and more
- **Structural analysis** — call graphs, data flows, communities, centrality, cycles, bridges, dead code
- **Change analysis** — git-aware diff detection, risk scoring, affected flows, test coverage gaps
- **Semantic search** — BM25 + embedding-based search with reciprocal rank fusion
- **Embeddings** — local sentence-transformers or external API backends
- **Pattern detection** — framework-level patterns (React, Spring DI, SQLAlchemy, FastAPI, etc.)
- **CLI & tool interface** — structured JSON commands for scripting and agent integration
- **Graph watcher** — incremental rebuilds on file changes

## Installation

```bash
pip install graphician
```

### Optional extras

| Extra | Adds |
|-------|------|
| `embeddings` | Local embedding model (`sentence-transformers`) |
| `jedi` | Jedi-based Python call enrichment |
| `dev` | Test and lint tooling |

```bash
pip install graphician[embeddings,jedi,dev]
```

## Quick Start

### Build a graph

```bash
graphician build /path/to/project
```

### Inspect the result

```bash
graphician status
graphician flows --top 10
graphician communities
graphician god-nodes --top 10
```

### Search and analyze

```bash
# Semantic search
graphician search "authentication" --detail standard

# Impact analysis
graphician impact "src/auth.rs::User"

# Caller / callee chains
graphician callers "login"
graphician callees "render"

# Execution paths
graphician paths "main" "process_payment"

# Extraction, call-resolution, connectivity, and test-link coverage
graphician coverage
```

### Review changes

```bash
# Detect what changed since a revision
graphician detect-changes --base HEAD~3

# Risk score for a symbol
graphician risk "src/payment.py::charge"

# Test coverage gaps
graphician test-coverage
```

### Tool interface (JSON)

```bash
graphician tool --operation minimal_context --target "src/main.rs::App" --detail minimal
```

## CLI Reference

| Subcommand | Description |
|------------|-------------|
| `build` | Full extraction and graph build |
| `update` | Incremental update from last build |
| `status` | Graph statistics (nodes, edges, call resolution) |
| `coverage` | Graph extraction and relationship coverage |
| `search` | Hybrid semantic + keyword search |
| `tool` | Generic tool operation interface |
| `impact` | Symbol change blast radius |
| `paths` | Execution paths between two symbols |
| `callers` / `callees` | Caller / callee chains |
| `flows` | Ranked execution flows |
| `architecture` | Communities, coupling, bottlenecks |
| `communities` | Community detection (Louvain, Leiden, Infomap) |
| `bridge-nodes` | Chokepoint / bridge nodes |
| `god-nodes` | Highest PageRank hub nodes |
| `cycles` | Dependency cycles |
| `articulation` | Articulation points |
| `large-functions` | Oversized functions |
| `dead-code` | Unreachable symbols |
| `detect-changes` | Diff-aware change detection |
| `risk` | Change risk scoring |
| `test-coverage` | Test coverage gaps |
| `graph-diff` | Graph-level diff |
| `snapshot-diff` | Snapshot comparison |
| `embed` | Build local embeddings |
| `embed-external` | Fetch embeddings from external API |
| `rebuild-fts` | Rebuild full-text search index |
| `watch` | Watch mode — incremental rebuilds |

## Architecture

```
src/graphician/
├── analysis/          ← Structural analysis: impact, flows, communities, search, changes
│   ├── communities/   ← Louvain, Leiden, Infomap detection
│   ├── search/        ← BM25, fuzzy, fusion, vocabulary
│   ├── changes/       ← Git-aware diff, risk, coverage
│   └── dedup/         ← MinHash, LSH, union-find deduplication
├── core/              ← Graph, Node, Edge, EdgeId, NodeId
├── extraction/        ← Language parsers, pipeline, call resolution
│   ├── documents/     ← HTML, Markdown, SVG parsers
│   ├── languages/     ← tree-sitter language bindings
│   ├── patterns/      ← Framework pattern detection
│   └── flows/         ← Entry point discovery
├── interfaces/        ← CLI, MCP transport, daemon, watcher
└── persistence/       ← SQLite store, embeddings, FTS5 index
```

## Graph Stats

Output from `graphician status` on this repository:

```
Graph stats: 16489 nodes, 35621 edges

Node kinds: 12756 variable, 1326 function, 576 method, 285 module, 253 class, 226 file, 928 flow
Edge kinds: 13482 data_flow, 10147 member_of, 6034 calls, 2110 defines, 1709 tested_by, 1112 imports
```

## Benchmarks

Native (Rust/PyO3) vs. pure-Python implementations, measured on this repository:

| Operation | Python | Native | Speedup |
|---|---:|---:|---:|
| Full CLI build and SQLite save | 2.06 s | 1.392 s (median) | ~1.5x |
| Type resolution (1,500 unique placeholders) | 740.081 ms | 21.538 ms | 34.4x |
| Rust extraction (500 functions) | 41.62 ms | 9.14 ms | 4.55x |
| Call resolution (3,000 ambiguous calls) | 37.786 ms | 18.840 ms | 2.01x |
| Full SQLite load (5K nodes / 15K edges) | 55.197 ms | 37.424 ms | 1.48x |
| Incremental SQLite save (1% nodes changed) | 55.438 ms | 39.119 ms | 1.42x |
| Full SQLite save (5K nodes / 15K edges) | 60.351 ms | 49.473 ms | 1.22x |
| Flow materialization | 0.230 s | 0.125 s | 1.84x |
| Native snapshot access (10K nodes / 30K edges, reused) | 40.952 ms (cold) | 22.266 ms | 1.84x |

Native extraction is enabled for Rust, TypeScript, JavaScript, Java, and C++, with
automatic fallback to the Python implementation (`GRAPHICIAN_NATIVE_EXTRACTORS=0`
forces the fallback explicitly). The installed wheel loads as `cp311-abi3` across
supported CPython versions.

### Coverage baseline

| Metric | Value |
|---|---:|
| File extraction | 225 / 225 (100%) |
| Definition source locations | 100% |
| Call resolution | 96.25% |
| Functions with callers | 29.87% |
| Functions with callees | 80.11% |
| Static production functions linked to tests | 20.87% |
| Connected graph nodes | 99.94% |

All six supported languages — Python, Rust, TypeScript, JavaScript, Java, and C++
— have parity fixtures requiring complete source-location coverage and resolution
of an unambiguous local call.

## Configuration

Graphician auto-discovers `graphician.db` in the current directory. Store file path:

```bash
graphician build --db /custom/path/graphician.db
```

## Development

```bash
# Install in editable mode
pip install -e ".[dev,embeddings,jedi]"

# Run tests
pytest

# Lint
ruff check

# Type check
mypy graphician
```

## License

MIT
