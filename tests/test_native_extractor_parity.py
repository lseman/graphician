"""Parity contracts for the native language extractors enabled by the pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from graphician._extract import HAS_RUST
from graphician._extract import extractors as native
from graphician.core.graph import Graph
from graphician.extraction.languages import Language
from graphician.extraction.languages.parsers import cpp, java, javascript, rust, typescript

CASES = [
    (
        "sample.rs",
        "trait T { fn t(&self); } struct A; "
        "impl A { fn m(&self) { helper(); } } fn helper() {}",
        rust.extract_file,
        native.extract_rust_file,
    ),
    (
        "sample.ts",
        "class A { m() { helper(); } } function helper() {}",
        typescript.extract_file,
        native.extract_typescript_file,
    ),
    (
        "sample.js",
        "class A { m() { helper(); } } function helper() {}",
        javascript.extract_file,
        native.extract_javascript_file,
    ),
    (
        "Sample.java",
        "class A { void m() { helper(); } void helper() {} }",
        java.extract_file,
        native.extract_java_file,
    ),
    (
        "sample.cpp",
        "class A { void m() { helper(); } void helper() {} };",
        cpp.extract_file,
        native.extract_cpp_file,
    ),
]


def _signature(graph: Graph):
    nodes = sorted((node.kind, node.qualified_name) for _, node in graph.nodes())
    edges = sorted(
        (
            graph.node(src).qualified_name,
            edge.kind,
            graph.node(dst).qualified_name,
        )
        for _, src, dst, edge in graph.edges()
    )
    return nodes, edges


@pytest.mark.skipif(not HAS_RUST, reason="native extension unavailable")
@pytest.mark.parametrize(("filename", "source", "python_extract", "native_extract"), CASES)
def test_native_extractor_matches_python_graph(
    tmp_path: Path, filename, source, python_extract, native_extract
):
    path = tmp_path / filename
    path.write_text(source, encoding="utf-8")
    file_qn = f"file::{filename}"
    expected = Graph()
    actual = Graph()

    python_extract(path, expected, file_qn=file_qn)
    native_extract(path, actual, file_qn=file_qn)

    assert _signature(actual) == _signature(expected)


@pytest.mark.skipif(not HAS_RUST, reason="native extension unavailable")
def test_pipeline_native_failure_falls_back_atomically(tmp_path: Path, monkeypatch):
    from graphician.extraction import pipeline

    path = tmp_path / "sample.rs"
    path.write_text("fn recovered() {}", encoding="utf-8")

    def fail_after_mutation(_path, fragment, **_kwargs):
        rust.extract_file(path, fragment, file_qn="file::partial.rs")
        raise RuntimeError("native failure")

    monkeypatch.setattr(native, "extract_rust_file", fail_after_mutation)
    extractor = pipeline._dedicated_extractors()[Language.RUST]
    graph = Graph()
    extractor(path, graph, file_qn="file::sample.rs")

    assert graph.find_by_qname("file::sample.rs::recovered") is not None
    assert graph.find_by_qname("file::partial.rs") is None
