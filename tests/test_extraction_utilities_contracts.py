from __future__ import annotations

import pytest

from graphician.analysis.type_resolution import resolve_type_placeholders
from graphician.core import Edge, EdgeKind, Graph, Node, NodeKind
from graphician.extraction.exclusions import default_ignored_name
from graphician.extraction.suppress_list import SuppressList, should_suppress_call_placeholder
from graphician.extraction.test_detect import is_test_file_path, is_test_name


def test_type_placeholder_resolution_rewires_unique_types_and_removes_orphan() -> None:
    graph = Graph()
    child = graph.add_node(Node.new(NodeKind.CLASS, "app::Child"))
    real = graph.add_node(Node.new(NodeKind.CLASS, "base::Base"))
    placeholder = graph.add_node(Node.new(NodeKind.CLASS, "type::Base"))
    graph.add_edge(child, placeholder, Edge.extracted(EdgeKind.INHERITS))

    assert resolve_type_placeholders(graph) == 1
    assert graph.find_by_qname("type::Base") is None
    assert any(dst == real and edge.kind is EdgeKind.INHERITS for dst, edge in graph.out_neighbors(child))


def test_type_placeholder_resolution_preserves_ambiguous_types() -> None:
    graph = Graph()
    child = graph.add_node(Node.new(NodeKind.CLASS, "app::Child"))
    graph.add_node(Node.new(NodeKind.CLASS, "one::Base"))
    graph.add_node(Node.new(NodeKind.TRAIT, "two::Base"))
    placeholder = graph.add_node(Node.new(NodeKind.CLASS, "type::Base"))
    graph.add_edge(child, placeholder, Edge.extracted(EdgeKind.IMPLEMENTS))
    assert resolve_type_placeholders(graph) == 0
    assert graph.find_by_qname("type::Base") == placeholder


@pytest.mark.parametrize(
    "path",
    ["tests/test_app.py", "src/__tests__/app.js", "pkg/app_test.go", "AppTest.java", "view.spec.ts"],
)
def test_test_file_detection_cross_language_conventions(path: str) -> None:
    assert is_test_file_path(path)


@pytest.mark.parametrize("name", ["test_login", "should_work", "describe_api", "setup_method"])
def test_test_name_detection_conventions(name: str) -> None:
    assert is_test_name(name)
    assert not is_test_name("production_handler")
    assert is_test_name("run", {"decorators": ["@pytest.mark.asyncio"]})


def test_suppress_list_and_default_exclusions_contract() -> None:
    suppress = SuppressList(patterns=[r"^call::vendor::", "[invalid"], qname_prefixes=["call::tmp"], min_source_length=4)
    assert suppress.should_suppress("call::vendor::log", "src/a.py")
    assert suppress.should_suppress("call::tmp_helper", "src/a.py")
    assert suppress.should_suppress("call::other", None)
    assert not suppress.should_suppress("call::other", "src/a.py")
    suppress.add_pattern(r"^call::generated")
    suppress.add_pattern("[")
    assert suppress.should_suppress("call::generated_fn", "long")
    assert suppress.to_dict()["min_source_length"] == 4
    assert should_suppress_call_placeholder("call::std::fmt")
    assert default_ignored_name("node_modules")
    assert default_ignored_name(".git")
    assert not default_ignored_name("src")
