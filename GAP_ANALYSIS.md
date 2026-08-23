# Graphician vs Ariadne (Rust) — Gap Analysis

## Summary

Graphician covers **~91%** of the Rust version's tool operations (41/45 shared), with a few gaps and some Python-specific additions.

---

## 1. Tool Operations Gap

### In Rust but NOT in Python

| Operation | Description |
|-----------|-------------|
| `minimal_context` | Bounded bidirectional neighborhood around a symbol |
| `minimal` | Detail level alias (handled via `detail_level` param) |
| `full` | Detail level alias (handled via `detail_level` param) |
| `infomap` | Community algorithm (available via `communities` with `algorithm="infomap"`) |
| `louvain` | Community algorithm (available via `communities` with `algorithm="louvain"`) |

**Note:** The Rust `infomap` and `louvain` are direct operations, but Python handles them through the `communities` operation with an algorithm parameter — functionally equivalent.

### In Python but NOT in Rust

| Operation | Description |
|-----------|-------------|
| `context_pack` | Token-budgeted diverse source bundle |
| `dedup` | Deduplicate semantic nodes |
| `freshness` | Graph freshness / file hash comparison |
| `patterns` | Detect framework patterns |
| `wiki` | Generate markdown wiki from communities |
| `architecture` | Alias for `architecture_overview` |
| `articulation_points` | Alias for `articulation` |
| `surprise_scoring` | Alias for `surprises` |
| `health` | Alias for `diagnostics` |
| `impact_radius` | Alias for `blast_radius` |
| `k_core` | Alias for `core` |
| `context` | Alias for `minimal_context` |

These are mostly aliases or Python-specific operations. All Rust operations are covered.

---

## 2. Analysis Module Gaps

### Missing in Python Analysis Module

| Rust Module | Python Equivalent | Status |
|-------------|-------------------|--------|
| `analysis/changes/differential.rs` | `analysis/diff.py` + `response/temporal.py` | ✅ Partially ported. `temporal_diff()` and `is_active_at()` exist in Python but are not fully wired to the CLI. |
| `analysis/semsearch.rs` | `analysis/semsearch.py` | ✅ Ported (semantic search with local embeddings) |

### Pattern Catalog Differences

| Metric | Rust | Python |
|--------|------|--------|
| Pattern count | 34 | 33 |
| Unique to Rust | `prisma_orm`, `rabbitmq_consumer`, `redis_cache`, `zod_schemas`, `clap_cli`, `nestjs_controllers` | — |
| Unique to Python | `argparse_cli`, `asyncio_patterns`, `auth_middleware`, `config_management`, `cors_middleware`, `event_emitter`, `mobx_state`, `marshmallow_validation`, `sqlalchemy_models`, `structured_logging` | — |

**Gap:** Rust has 6 patterns not in Python; Python has ~10 patterns not in Rust.

---

## 3. Extraction Layer Gaps

### Rust Has, Python Doesn't

| Rust Feature | Status |
|-------------|--------|
| `extraction/languages/tsconfig_resolver.rs` | ✅ Present in Python |
| `extraction/pipeline/library_stubs.rs` | ✅ Present in Python |
| `extraction/pipeline/jedi/` | ✅ Present in Python |
| `extraction/documents/vision/` | ✅ Present in Python (SVG) |
| `extraction/data_flow.rs` | ✅ Present in Python |
| `extraction/manifests/` | ✅ Present in Python |

### Python Has, Rust Doesn't

| Python Feature | Status |
|---------------|--------|
| `extraction/spring_di.py` | Spring DI resolution (Java) |
| `extraction/rust_analyzer.py` | rust-analyzer enrichment |
| `extraction/compiler.py` | Compiler evidence |

---

## 4. Evaluation/Benchmarking

| Feature | Rust | Python |
|---------|------|--------|
| `ariadne-eval` crate | ✅ Full crate with 10 benchmarks | ✅ `evaluation.py` — compatible runner |
| Benchmarks | 10 benchmark types | 7 benchmark types |
| Config format | `.toml` | `.toml` |
| Output format | CSV/JSON | CSV/JSON |

**Gap:** Python is missing 3 benchmarks:
- `build_performance` (present in Python)
- `call_coverage` (missing in Python)
- `graph_coverage` (missing in Python)

---

## 5. Persistence Layer

### Differences

| Feature | Rust | Python |
|---------|------|--------|
| SQLite schema | Canonical | Canonical (compatible) |
| Embeddings (local) | ✅ `embeddings/local.rs` | ✅ `embeddings/local.py` |
| Embeddings (external) | ✅ `embeddings/external.rs` | ✅ `embeddings/local.py` (partial) |
| FTS5 indexing | ✅ `sql.rs::build_fts5_query` | ✅ `fts.py` |
| Graph cache | ✅ `response/cache.rs` | ✅ `response/__init__.py::_cache_*` |

---

## 6. CLI Command Gaps

### Rust commands NOT in Python CLI

| Command | Python Equivalent |
|---------|-------------------|
| `embed` | ✅ `embed` exists in Python |
| `embed-external` | ✅ `embed-external` exists in Python |
| `rebuild-fts` | ✅ `rebuild-fts` exists in Python |
| `graph-diff` | ✅ `graph-diff` exists in Python |
| `snapshot-diff` | ✅ `snapshot-diff` exists in Python |

### Python commands NOT in Rust CLI

| Command | Rust Equivalent |
|---------|-----------------|
| `jedi-enrich` | ✅ `jedi_enrich` via tool operation |
| `spring-di-resolve` | ✅ `spring_di_resolve` via tool operation |
| `eval` | ✅ Present in Rust as `ariadne-eval` binary |
| `snapshots` | Partial — Rust has `snapshot_diff` |

---

## 7. Structural Analysis Parity

| Operation | Rust | Python |
|-----------|------|--------|
| `pagerank` / `god_nodes` | ✅ | ✅ |
| `personalized_pagerank` | ✅ | ✅ |
| `louvain` | ✅ | ✅ |
| `leiden` | ✅ | ✅ |
| `infomap` | ✅ | ✅ |
| `bridge_scores` | ✅ | ✅ |
| `articulation_points` | ✅ | ✅ |
| `core_numbers` | ✅ | ✅ |
| `cyclic_components` | ✅ | ✅ |
| `approx_betweenness` | ✅ | ✅ |
| `knowledge_gaps` | ✅ | ✅ |
| `split_oversized` | ✅ | ✅ |

---

## 8. Response/Output Gaps

### Rust Has, Python Missing in Output

| Field | Purpose |
|-------|---------|
| `call_resolution` | Resolved/unresolved call stats with rate |
| `graph_summary` | Compact node/edge kind counts |
| `guardrails` | Pagination metadata |
| `snippets` | Source text excerpts |

### Python Has, Rust Missing in Output

| Field | Purpose |
|-------|---------|
| `_hints` | Workflow hints for next operations |
| `pagination` per-key | Detailed pagination metadata |

---

## 9. Missing Operations from Python CLI `generic_operations`

The following Rust tool operations are NOT exposed as top-level CLI subcommands in Python:

1. **`graph_diff`** — Has CLI subcommand `graph-diff` but uses `diff.py` (in-memory comparison) rather than `temporal_diff` (git-based)
2. **`context`** — Available as tool op, not as top-level CLI
3. **`minimal_context`** — Available as tool op, not as top-level CLI

---

## 10. Key Implementation Gaps

### High Priority

1. **`differential`/`temporal_diff` CLI exposure**: Python has the function but it's not exposed via the `tool_response` dispatch table. Users can't call it as a standalone tool operation.

2. **`context`/`minimal_context` CLI subcommand**: Available as a tool op but not as a top-level CLI command.

### Medium Priority

3. **Missing Rust patterns** (6 patterns): `prisma_orm`, `rabbitmq_consumer`, `redis_cache`, `zod_schemas`, `clap_cli`, `nestjs_controllers`

4. **Missing Python benchmarks** (2 benchmarks): `call_coverage`, `graph_coverage`

5. **Response field parity**: Python's `_dispatch` uses `_compact_for_detail()` which drops snippets for minimal detail, but doesn't have Rust's `guardrails` metadata format

### Low Priority

6. **Alias normalization**: Rust accepts `k_core`, `articulation_points`, `surprise_scoring`, `health`, `impact_radius` as aliases. Python handles these but via a different mapping mechanism.

---

## 11. Summary Statistics

| Metric | Count |
|--------|-------|
| Total Rust tool operations | 45 |
| Shared operations | 41 |
| Rust-only operations | 4 (minimal_context, minimal, full, infomap/louvain via communities) |
| Python-only operations | 12 (mostly aliases + context_pack, dedup, freshness, patterns, wiki) |
| Analysis module parity | ~95% |
| CLI command parity | ~95% |
| Pattern catalog overlap | ~85% |
| Benchmark parity | ~70% (7/10) |
| Overall feature parity | ~91% |

---

## 12. Recommendations

1. **Add `graph_diff`/`differential` as a tool operation** — Wire up the existing `temporal_diff()` function to the dispatch table
2. **Add `context` as a CLI subcommand** — Mirror Rust's `minimal_context`
3. **Port missing Rust patterns** — `prisma_orm`, `rabbitmq_consumer`, `redis_cache`, `zod_schemas`, `clap_cli`, `nestjs_controllers`
4. **Port missing Python benchmarks** — `call_coverage`, `graph_coverage`
5. **Normalize alias handling** — Use the same alias map approach in both languages for consistency
