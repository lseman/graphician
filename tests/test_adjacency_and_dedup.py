"""Tests for adjacency matrix builder and graph analysis utilities."""

from __future__ import annotations

from graphician.analysis.adjacency import (
    AdjacencyConfig,
    build_adjacency_matrix,
    build_transitions_matrix,
    build_weighted_adjacency_matrix,
    compute_out_degree_matrix,
)
from graphician.analysis.dedup.normalize import normalize_label, passes_entropy_gate
from graphician.analysis.search.utils import _graph_summary
from graphician.core.edge import Edge, EdgeKind
from graphician.core.graph import Graph
from graphician.core.node import Node, NodeKind


class TestAdjacencyConfig:
    """Tests for AdjacencyConfig."""

    def test_default_weights(self) -> None:
        config = AdjacencyConfig()
        assert config.weights is not None
        assert config.weights[EdgeKind.CALLS] == 1.0
        assert config.weights[EdgeKind.DEFINES] == 0.7
        assert config.weights[EdgeKind.INHERITS] == 1.15

    def test_edge_weight_excludes_ambiguous(self) -> None:
        config = AdjacencyConfig()
        calls_edge = Edge.ambiguous(EdgeKind.CALLS)
        w = config.edge_weight(calls_edge)
        assert w == 0.0

    def test_edge_weight_with_extracted(self) -> None:
        config = AdjacencyConfig()
        extracted_edge = Edge.extracted(EdgeKind.CALLS)
        w = config.edge_weight(extracted_edge)
        assert w > 0

    def test_exclude_ambiguous_true(self) -> None:
        config = AdjacencyConfig(exclude_ambiguous=True)
        ambiguous_edge = Edge.ambiguous(EdgeKind.CALLS)
        assert config.edge_weight(ambiguous_edge) == 0.0

    def test_exclude_ambiguous_false(self) -> None:
        config = AdjacencyConfig(exclude_ambiguous=False)
        ambiguous_edge = Edge.ambiguous(EdgeKind.CALLS)
        w = config.edge_weight(ambiguous_edge)
        assert w > 0

    def test_custom_weights(self) -> None:
        config = AdjacencyConfig(weights={EdgeKind.CALLS: 2.0})
        edge = Edge.extracted(EdgeKind.CALLS)
        w = config.edge_weight(edge)
        assert w == 2.0


class TestBuildAdjacencyMatrix:
    """Tests for build_adjacency_matrix."""

    def test_empty_graph(self) -> None:
        graph = Graph()
        _, _, _, nodes, mapping = build_adjacency_matrix(graph)
        assert len(nodes) == 0
        assert len(mapping) == 0

    def test_graph_with_nodes_no_edges(self) -> None:
        graph = Graph()
        graph.add_node(Node.new(NodeKind.FUNCTION, "foo"))
        graph.add_node(Node.new(NodeKind.FUNCTION, "bar"))
        _, _, _, nodes, mapping = build_adjacency_matrix(graph)
        assert len(nodes) == 2
        assert len(mapping) == 2

    def test_graph_with_edges(self) -> None:
        graph = Graph()
        graph.add_node(Node.new(NodeKind.FUNCTION, "caller"))
        graph.add_node(Node.new(NodeKind.FUNCTION, "callee"))
        graph.add_edge(
            graph.find_by_qname("caller"),
            graph.find_by_qname("callee"),
            Edge.extracted(EdgeKind.CALLS),
        )
        _, _, _, nodes, mapping = build_adjacency_matrix(graph)
        assert len(nodes) == 2
        assert len(mapping) == 2

    def test_exclude_ambiguous_edges(self) -> None:
        graph = Graph()
        graph.add_node(Node.new(NodeKind.FUNCTION, "caller"))
        graph.add_node(Node.new(NodeKind.FUNCTION, "callee"))
        graph.add_edge(
            graph.find_by_qname("caller"),
            graph.find_by_qname("callee"),
            Edge.ambiguous(EdgeKind.CALLS),
        )
        config = AdjacencyConfig(exclude_ambiguous=True)
        _, _, weights, _, _ = build_adjacency_matrix(graph, config)
        assert len(weights) == 0

    def test_multiple_edge_kinds(self) -> None:
        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "foo"))
        n2 = graph.add_node(Node.new(NodeKind.FUNCTION, "bar"))
        graph.add_edge(n1, n2, Edge.extracted(EdgeKind.CALLS))
        graph.add_edge(n1, n2, Edge.extracted(EdgeKind.DEFINES))
        _, _, weights, _, _ = build_adjacency_matrix(graph)
        assert len(weights) == 2


class TestBuildWeightedAdjacencyMatrix:
    """Tests for build_weighted_adjacency_matrix."""

    def test_degree_computation(self) -> None:
        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "foo"))
        n2 = graph.add_node(Node.new(NodeKind.FUNCTION, "bar"))
        graph.add_edge(n1, n2, Edge.extracted(EdgeKind.CALLS))
        _, _, _, degree, _, _ = build_weighted_adjacency_matrix(graph)
        assert len(degree) == 2
        assert degree[0] > 0
        assert degree[1] == 0


class TestComputeOutDegreeMatrix:
    """Tests for compute_out_degree_matrix."""

    def test_degree_values(self) -> None:
        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "foo"))
        n2 = graph.add_node(Node.new(NodeKind.FUNCTION, "bar"))
        graph.add_edge(n1, n2, Edge.extracted(EdgeKind.CALLS))
        degrees = compute_out_degree_matrix(graph)
        assert len(degrees) == 2
        assert degrees[0] > 0
        assert degrees[1] == 0


class TestBuildTransitionsMatrix:
    """Tests for build_transitions_matrix."""

    def test_transition_probs_sum_to_one(self) -> None:
        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "foo"))
        n2 = graph.add_node(Node.new(NodeKind.FUNCTION, "bar"))
        n3 = graph.add_node(Node.new(NodeKind.FUNCTION, "baz"))
        graph.add_edge(n1, n2, Edge.extracted(EdgeKind.CALLS))
        graph.add_edge(n1, n3, Edge.extracted(EdgeKind.CALLS))
        _, _, probs, _ = build_transitions_matrix(graph)
        assert abs(probs.sum() - 1.0) < 0.01


class TestNormalizeLabel:
    """Tests for normalize_label."""

    def test_uppercase(self) -> None:
        assert normalize_label("FooBar") == "foobar"
        assert normalize_label("FOO") == "foo"

    def test_with_underscore(self) -> None:
        assert normalize_label("Foo_Bar") == "foo_bar"

    def test_empty_string(self) -> None:
        assert normalize_label("") == ""

    def test_preserves_numbers(self) -> None:
        result = normalize_label("test123")
        assert result == "test"


class TestPassesEntropyGate:
    """Tests for passes_entropy_gate."""

    def test_high_entropy(self) -> None:
        assert passes_entropy_gate("xkcd", 1.0)

    def test_low_entropy(self) -> None:
        assert not passes_entropy_gate("aaaa", 1.0)

    def test_default_threshold(self) -> None:
        assert passes_entropy_gate("hello_world", 0.5)


class TestGraphSummary:
    """Tests for _graph_summary."""

    def test_basic_summary(self) -> None:
        graph = Graph()
        graph.add_node(Node.new(NodeKind.FUNCTION, "foo"))
        graph.add_node(Node.new(NodeKind.CLASS, "Bar"))
        summary = _graph_summary(graph)
        assert summary["total_nodes"] == 2
        assert summary["total_edges"] == 0
        assert "function" in summary["node_kinds"]
        assert "class" in summary["node_kinds"]

    def test_with_edges(self) -> None:
        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "foo"))
        n2 = graph.add_node(Node.new(NodeKind.FUNCTION, "bar"))
        graph.add_edge(n1, n2, Edge.extracted(EdgeKind.CALLS))
        summary = _graph_summary(graph)
        assert summary["total_nodes"] == 2
        assert summary["total_edges"] == 1
