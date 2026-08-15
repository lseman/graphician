# TODO — Analysis Module Stubs & Remaining Gaps

The `analysis/` module has been reorganized from a flat structure into a nested
package structure matching the Rust `ariadne-graph` crate layout.

## Directory Structure (Updated)

```
analysis/
├── __init__.py          # Re-exports public API
├── centrality.py        # Flat: pagerank, personalized_pagerank, is_rank_noise
├── paths.py             # Flat: path analysis utilities
├── semsearch.py         # Flat: semantic search with embeddings
├── context_pack.py      # Flat: context pack builder
├── export.py            # Flat: GraphML export
├── diff.py              # Flat: graph diff utilities
│
├── changes/             # Split from changes.py
│   ├── __init__.py
│   ├── types.py         # Change, RiskScore
│   ├── detection.py     # detect_changes()
│   ├── risk.py          # compute_risk()
│   ├── coverage.py      # compute_test_coverage()
│
├── communities/         # Split from communities.py
│   ├── __init__.py
│   ├── louvain.py       # Louvain, Leiden, Infomap
│   ├── nodes.py         # Bridge/hub/god nodes
│   ├── quality.py       # community_cohesion, community_quality
│   ├── gaps.py          # knowledge_gaps()
│   ├── split.py         # split_oversized()
│   └── utils.py         # Shared utilities
│
├── dedup/               # Split from dedup.py
│   ├── __init__.py
│   ├── types.py         # DedupOptions, DedupResult
│   ├── normalize.py     # normalize_label, entropy gate
│   ├── minhash.py       # MinHash signatures
│   ├── lsh.py           # LshIndex, lsh_candidate_pairs
│   ├── similarity.py    # Jaro-Winkler
│   └── union_find.py    # Union-Find, deduplicate_nodes
│
├── flows/               # Split from flows.py
│   ├── __init__.py
│   ├── types.py         # FlowOptions
│   ├── detection.py     # compute_flows, detect_entry_points
│   ├── entry_points.py  # Framework entry detection
│   ├── trace.py         # Flow tracing, criticality scoring
│
├── impact/              # Split from impact.py
│   ├── __init__.py
│   ├── types.py         # ImpactQuery, ImpactHit, ImpactResult
│   └── engine.py        # find_impact(), compute_impact()
│
├── motifs/              # Split from motifs.py
│   ├── __init__.py
│   ├── dsl.py           # Motif, MotifBuilder, NamePattern
│   ├── engine.py        # VF2 subgraph isomorphism
│   └── builtins.py      # security_audit, diamond_inheritance, doc_function_triangle
│
├── patterns/            # Split from patterns.py
│   ├── __init__.py
│   ├── types.py         # PatternCategory, FrameworkPattern, PatternMatch
│   ├── builtin.py       # Built-in pattern catalog
│   └── matcher.py       # detect_patterns(), _match_pattern()
│
├── search/              # Split from search.py
│   ├── __init__.py
│   ├── types.py         # SearchIntent, SearchHit
│   ├── vocabulary.py    # Tokenization, stopwords
│   ├── fuzzy.py         # Levenshtein, fuzzy_score
│   ├── search.py        # ranked_search, task_aware_search, fts_ranked_search
│   └── utils.py         # Graph summary
│
├── structure.py         # Flat: structural analysis utilities
│
├── refactoring/         # ✅ Rename preview, dead code detection
│   ├── __init__.py
│   ├── types.py         # RenameEdit, RenamePreview, RenameStats
│   └── engine.py        # rename_preview(), find_dead_code()
```

## Implemented ✅

| Module | Status | Notes |
|--------|--------|-------|
| `changes/` | Complete | Change detection, risk scores, test-coverage gaps |
| `communities/` | Complete | Louvain/Leiden/Infomap, bridge/hub/god nodes, knowledge_gaps(), split_oversized() |
| `dedup/` | Complete | 6-pass dedup: normalize → entropy gate → MinHash/LSH → Jaro-Winkler → community boost → union-find |
| `flows/` | Complete | Flow detection with BFS tracing, criticality scoring |
| `impact/` | Complete | Reverse BFS impact with `ImpactQuery`/`ImpactHit`, edge-kind costs, node-kind boosts |
| `motifs/` | Complete | Full VF2 subgraph isomorphism engine, `Motif` DSL, `NamePattern`, built-in motifs |
| `patterns/` | Complete | ~30 framework pattern definitions, scoring matching engine |
| `search/` | Complete | `SearchIntent` enum, `ranked_search()`, `search_by_name()`, `task_aware_search()`, `fts_ranked_search()` |
| `refactoring/` | Complete | `RenameEdit`, `RenamePreview`, `RenameStats`, `rename_preview()`, `find_dead_code()` |
| `centrality.py` | Flat | `pagerank()`, `personalized_pagerank()`, `is_rank_noise()` |
| `paths.py` | Flat | `find_paths()`, `find_top_paths()`, `callers_of()`, `callees_of()`, `max_depth_from()` |
| `structure.py` | Flat | `cyclic_components()`, `core_numbers()`, `bridge_scores()`, `approx_betweenness()` |

## Stubs / Future Expansion

### `analysis/refactoring/` — ✅ Implemented
- ✅ `RenameEdit` — single rename edit suggestion (file, line, old, new, confidence)
- ✅ `RenamePreview` — preview result with edits list and stats
- ✅ `RenameStats` — count of high/medium/low confidence edits
- ✅ `rename_preview()` — finds definition site, call sites, import sites, bare-name refs
- ✅ `find_dead_code()` — detects functions/classes with no callers, no test refs, no importers
  - Excludes entry points (name patterns: main, test_, Handle, serve, run, etc.)
  - Excludes test files (paths containing __tests__, .spec., .test., /test_)
  - Excludes framework-inherited classes (name suffixes: Stack, Construct, Resource, Model, App)
- ⏳ **Not yet implemented:** extraction suggestions, type-change propagation (suggested by Rust `mod.rs` but no Rust code exists yet)

## Extraction Reorganization (Partial)

The `extraction/` package has been partially reorganized:

```
extraction/
├── pipeline/            # Call resolution, type resolution, Spring DI
├── patterns/            # Framework pattern detection
├── documents/           # Document extraction
├── languages/parsers/   # Language-specific parsers
├── flows/               # Split from flows.py (NEW)
│   ├── __init__.py
│   ├── types.py
│   ├── entry_points.py
│   └── trace.py
├── manifests/           # Manifest extraction
├── compiler.py          # Flat: compiler evidence
├── data_flow.py         # Flat: data flow extraction
├── rust_analyzer.py     # Flat: Rust analyzer integration
├── test_detect.py       # Flat: test detection
├── pipeline.py          # Flat: extraction pipeline
└── call_resolution.py   # Flat (will move to pipeline/)
└── type_resolution.py   # Flat (will move to pipeline/)
```

## Operational Parity

- ✅ Rust and Python read and write the same canonical Rust SQLite schema. Opening the
  retired Python-only layout destructively resets it and marks it for a rebuild.
- ✅ Rust's grouped CLI forms (`analysis`, `git`, `structure`, `advanced`, `agent`,
  `maintenance`, and `utility`) are accepted alongside Python's flat aliases.
- ✅ `eval`, `jedi-enrich`, and `spring-di-resolve` are exposed from the Python CLI.
- ✅ The seven Rust evaluation benchmarks and CSV/JSON result formats are available.
- ✅ Graph self-loops and parallel edges use Rust/petgraph semantics, allowing lossless
  Rust-to-Python database loading.

## Rust Reference Gaps

Functions present in the Rust reference but not yet ported:

| Rust Function | Location | Python Status | Priority |
|---------------|----------|---------------|----------|
| `refactoring/` module | `ariadne-graph/src/analysis/refactoring/` | **Implemented** ✅ | — |
| Community options (enum/struct) | `communities/mod.rs` | Not needed — Python uses keyword args | Low |
