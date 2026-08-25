# Graphician Hardening Report

This report records the correctness, coverage, packaging, and performance gates
used for `graphician-native` and the Python integration. Measurements use the
Graphician repository itself on the same local machine and an isolated temporary
SQLite database.

## Current coverage baseline

| Metric | Before | Current |
|---|---:|---:|
| File extraction | Not measurable | 225 / 225 (100%) |
| Definition source locations | 100% | 100% |
| Call resolution | 66.82% | 96.25% |
| Functions with callers | 27.92% | 29.87% |
| Functions with callees | 79.78% | 80.11% |
| Static production functions linked to tests | 17.91% | 20.87% |
| Connected graph nodes | 99.94% | 99.94% |

`test_links` is static relationship coverage derived from `tested_by` edges. It
is not runtime statement or branch coverage.

All six supported language baselines—Python, Rust, TypeScript, JavaScript,
Java, and C++—have parity fixtures requiring complete source-location coverage
and resolution of an unambiguous local call.

The behavioral suite has 464 passing tests. Repository-wide branch coverage is
58.12%; the existing 60% pytest gate is therefore still red. The new end-to-end
build/coverage test improved this from 58.02%, but the remaining gap is broader
legacy coverage (not a failure in the graph-coverage command) and should be
closed with focused tests for the currently untested service and fallback paths.

## Performance

| Operation | Before | Current | Result |
|---|---:|---:|---:|
| Full CLI build and SQLite save | ~2.06 s | 1.392 s median | ~32% faster |
| Document mention resolution (profiled) | 2.16 s | 0.016 s | Removed quadratic scan |
| Flow materialization on repository graph | 0.230 s Python | 0.125 s native | 1.84x faster |
| Native snapshot access (10K nodes / 30K edges) | 40.952 ms cold | 22.266 ms reused | 1.84x faster |
| Type resolution (1,500 unique placeholders) | 740.081 ms Python | 21.538 ms native | 34.4x faster |
| Call resolution (3,000 ambiguous calls) | 37.786 ms Python | 18.840 ms native | 2.01x faster |
| Full SQLite save (5K nodes / 15K edges) | 60.351 ms Python | 49.473 ms native | 1.22x faster |
| Full SQLite load (5K nodes / 15K edges) | 55.197 ms Python | 37.424 ms native | 1.48x faster |
| Incremental SQLite save (1% nodes changed) | 55.438 ms Python | 39.119 ms native | 1.42x faster |
| Rust extraction (500 functions) | 41.62 ms Python | 9.14 ms native | 4.55x faster |

The full-build comparison includes additional calls recovered by the improved
resolver, so the current graph performs more relationship work than the baseline.

Implemented optimizations:

- one indexed symbol table for the document mention post-pass;
- top-down directory pruning before walking dependency/build trees;
- batched SQLite node, edge, and file-state insertion;
- native batch flow tracing for graphs with at least 1,000 nodes and 32 entry
  points, where snapshot conversion is amortized.
- a graph-owned persistent native snapshot shared across analysis operations,
  invalidated by structural mutations and direct edge kind/confidence edits.
- native whole-graph call and type resolution planners, with Python-owned
  mutation application and automatic reference-implementation fallbacks.
- native canonical SQLite full save/load and stable-ID incremental
  synchronization for filesystem-backed stores without persisted embeddings.
- native extraction dispatch for Rust, TypeScript, JavaScript, Java, and C++,
  committed to the destination graph only after successful extraction, with
  Python fallback. Python extraction remains on its dedicated implementation
  until relative-import metadata reaches parity.

## Correctness fixes

- Library-stub resolution now rewrites placeholder edges instead of retaining a
  duplicate unresolved edge.
- Stub selection respects Python, Rust, JavaScript/TypeScript, and C++ caller
  dialects and is idempotent.
- Suppressed runtime calls are pruned only when strong project-symbol evidence
  is absent; project functions named `get`, `load`, `parse`, `add`, or `len`
  remain resolvable.
- Python `from` imports retain module, original symbol, and local alias metadata,
  allowing relative and aliased imports to disambiguate duplicate definitions.
- Fluent constructors such as `Node.new(...).with_property(...)` participate in
  receiver-type resolution.
- Manifest inputs now produce source-backed file nodes and provenance edges.
- Coverage compares persisted build-manifest paths with extracted document,
  diagram, manifest, and source roots, including absolute-to-relative matching.

## Native gates

Enabled native paths have parity tests for PageRank, traversal, paths, impact,
cycles, core numbers, articulation points, Louvain/Leiden dispatch, MinHash/LSH,
motifs, community quality, fuzzy/ranked search, data flow, and batch flow tracing.
The installed wheel must load as `cp311-abi3` on supported CPython versions.

Native Infomap remains gated because its partition fragmentation fails parity.
Five native language extractors are enabled after graph-level parity fixtures for
calls, scopes, ownership, and symbol kinds. Python implementations remain the
automatic fallback, and `GRAPHICIAN_NATIVE_EXTRACTORS=0` provides an explicit
opt-out. A native implementation is not enabled solely because its internal
kernel is faster; public end-to-end timing includes Python/Rust conversion.

`cargo fmt --check` and `cargo check --all-targets --all-features` pass. A raw
`cargo test` executable cannot link under PyO3's `extension-module` feature
because Python symbols are intentionally supplied by the interpreter at module
load time. Native behavior is therefore exercised through the built ABI3 module,
the Python parity suite, and an isolated wheel-install smoke test.
