from __future__ import annotations

from pathlib import Path

from graphician.core import EdgeKind, Graph, NodeKind
from graphician.extraction.languages import LanguageRegistry
from graphician.extraction.languages.parsers.rust import extract_file
from graphician.extraction.pipeline import ExtractionPipeline


def _extract(tmp_path: Path, source: str) -> Graph:
    path = tmp_path / "lib.rs"
    path.write_text(source)
    graph = Graph()
    extract_file(path, graph)
    return graph


def _qnames(graph: Graph) -> set[str]:
    return {node.qualified_name for _, node in graph.nodes()}


def _edges(graph: Graph) -> set[tuple[str, EdgeKind, str]]:
    return {
        (graph.node(src).qualified_name, edge.kind, graph.node(dst).qualified_name)
        for _, src, dst, edge in graph.edges()
    }


def test_rust_traits_impls_async_calls_and_nested_functions(tmp_path: Path) -> None:
    graph = _extract(
        tmp_path,
        """
trait Runner { fn run(&self); }
struct Job;
impl Runner for Job {
    fn run(&self) { helper(); self.tick(); }
}
impl Job {
    async fn tick(&self) {
        fetch().await;
        fn nested() { deep(); }
        nested();
    }
}
""",
    )
    qnames = _qnames(graph)
    # IMPL nodes are NOT created (matching ariadne-rust behavior).
    # Methods are extracted directly under their containing class.
    assert {"lib::Runner", "lib::Job"} <= qnames
    assert "lib::Job::tick::nested" in qnames
    assert graph.node(graph.find_by_qname("lib::Runner")).kind is NodeKind.TRAIT
    # No IMPL node exists — matching ariadne-rust
    assert graph.find_by_qname("lib::Job::Runner::impl") is None
    assert {"call::helper", "call::tick", "call::fetch", "call::nested", "call::deep"} <= qnames
    assert any(
        edge.kind is EdgeKind.CALLS and graph.node(dst).qualified_name == "call::fetch"
        for _, _, dst, edge in graph.edges()
    )


def test_rust_use_trees_macros_and_test_modules(tmp_path: Path) -> None:
    graph = _extract(
        tmp_path,
        """
use std::{fmt, sync::Arc};
use crate::util::{helper, nested::Thing};
fn login() -> bool { true }
fn exercise() { assert!(login()); }
#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn checks_login() { exercise().await; }
}
mod extra;
""",
    )
    qnames = _qnames(graph)
    assert {
        "module::std::fmt",
        "module::std::sync::Arc",
        "module::crate::util::helper",
        "module::crate::util::nested::Thing",
        "module::super::*",
        "lib::tests",
        "lib::extra",
    } <= qnames
    test_id = graph.find_by_qname("lib::tests::checks_login")
    assert test_id is not None
    assert graph.node(test_id).properties["is_test"] is True
    assert ("lib::exercise", EdgeKind.CALLS, "call::login") in _edges(graph)


def test_rust_nested_use_trees_expand_and_preserve_source_alias(tmp_path: Path) -> None:
    graph = _extract(
        tmp_path,
        "use std::{self, collections::{HashMap, HashSet}, io as sio};\n",
    )
    imports = {
        node.qualified_name
        for _, _, dst, edge in graph.edges()
        if edge.kind is EdgeKind.IMPORTS
        for node in [graph.node(dst)]
    }
    assert imports == {
        "module::std",
        "module::std::collections::HashMap",
        "module::std::collections::HashSet",
        "module::std::io",
    }


def test_macro_contained_calls_use_normal_pipeline_resolution(tmp_path: Path) -> None:
    (tmp_path / "lib.rs").write_text(
        "pub fn login() -> bool { true }\n"
        "pub fn exercise() { assert!(login()); }\n"
    )

    graph = ExtractionPipeline(LanguageRegistry()).build(tmp_path)

    login = next(
        node_id
        for node_id, node in graph.nodes()
        if node.qualified_name.endswith("::login")
        and not node.qualified_name.startswith("call::")
    )
    exercise = next(
        node_id
        for node_id, node in graph.nodes()
        if node.qualified_name.endswith("::exercise")
    )
    assert any(
        destination == login and edge.kind is EdgeKind.CALLS
        for destination, edge in graph.out_neighbors(exercise)
    )
    assert graph.find_by_qname("call::login") is None


def test_pipeline_keeps_equal_rust_file_stems_distinct(tmp_path: Path) -> None:
    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    (tmp_path / "first" / "lib.rs").write_text("pub fn first() {}\n")
    (tmp_path / "second" / "lib.rs").write_text("pub fn second() {}\n")

    graph = ExtractionPipeline(LanguageRegistry(), workers=1).build(tmp_path)

    assert graph.find_by_qname("file::first/lib.rs::first") is not None
    assert graph.find_by_qname("file::second/lib.rs::second") is not None
