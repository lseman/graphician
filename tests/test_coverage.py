"""Regression tests for graph-derived test coverage."""

from graphician.analysis.changes.coverage import compute_test_coverage
from graphician.core.edge import Edge, EdgeKind
from graphician.core.graph import Graph
from graphician.core.node import Node, NodeKind


def test_coverage_counts_tested_and_untested_symbols() -> None:
    graph = Graph()
    tested = graph.add_node(Node.new(NodeKind.FUNCTION, "pkg::tested"))
    graph.add_node(Node.new(NodeKind.FUNCTION, "pkg::untested"))
    test_node = Node.new(NodeKind.FUNCTION, "tests::test_tested")
    test_node.with_property("is_test", True)
    test = graph.add_node(test_node)
    graph.add_edge(tested, test, Edge.extracted(EdgeKind.TESTED_BY))

    result = compute_test_coverage(graph)

    assert result["coverage"] == 0.5
    assert result["tested_count"] == 1
    assert result["untested_count"] == 1
    assert [item["qualified_name"] for item in result["untested"]] == ["pkg::untested"]


def test_coverage_handles_an_empty_graph() -> None:
    result = compute_test_coverage(Graph())

    assert result["coverage"] == 0.0
    assert result["tested_count"] == 0
    assert result["untested_count"] == 0
    assert result["untested"] == []
