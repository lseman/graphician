"""Tests for motif DSL and path analysis."""

from __future__ import annotations

from graphician.analysis.motifs.dsl import (
    Motif,
    MotifEdge,
    MotifNode,
    NamePattern,
    _glob_to_regex,
)
from graphician.analysis.paths import (
    PathQuery,
    callees_of,
    callers_of,
    find_paths,
    find_top_paths,
    max_depth_from,
)
from graphician.core.edge import Edge, EdgeKind
from graphician.core.graph import Graph
from graphician.core.id import NodeId
from graphician.core.node import Node, NodeKind

# ---------------------------------------------------------------------------
# NamePattern tests
# ---------------------------------------------------------------------------

class TestNamePattern:
    """Tests for NamePattern matching."""

    def test_exact_match(self) -> None:
        pat = NamePattern.exact("foo")
        assert pat.matches("foo")
        assert not pat.matches("bar")
        assert not pat.matches("FOO")

    def test_exact_no_match(self) -> None:
        pat = NamePattern.exact("foo")
        assert not pat.matches("foobar")

    def test_contains_match(self) -> None:
        pat = NamePattern.contains("foo")
        assert pat.matches("foo_bar")
        assert pat.matches("bar_foo")
        assert pat.matches("FOO_BAR")

    def test_contains_no_match(self) -> None:
        pat = NamePattern.contains("foo")
        assert not pat.matches("bar_baz")

    def test_glob_match(self) -> None:
        pat = NamePattern.glob("test_*")
        assert pat.matches("test_func")
        assert pat.matches("test_class")

    def test_glob_no_match(self) -> None:
        pat = NamePattern.glob("test_*")
        assert not pat.matches("other_func")

    def test_regex_match(self) -> None:
        pat = NamePattern.regex(r"^\d+_test$")
        assert pat.matches("1_test")
        assert pat.matches("42_test")

    def test_regex_no_match(self) -> None:
        pat = NamePattern.regex(r"^\d+_test$")
        assert not pat.matches("test_1")

    def test_invalid_regex_fallback(self) -> None:
        pat = NamePattern.regex(r"[invalid")
        assert not pat.matches("anything")

    def test_invalid_glob_fallback(self) -> None:
        pat = NamePattern.glob("[invalid")
        assert not pat.matches("anything")

    def test_glob_with_colons(self) -> None:
        pat = NamePattern.glob("test_*")
        assert not pat.matches("test::inner")

    def test_constructors(self) -> None:
        p1 = NamePattern.exact("foo")
        assert p1.kind == "exact"
        assert p1.pattern == "foo"

        p2 = NamePattern.contains("bar")
        assert p2.kind == "contains"

        p3 = NamePattern.glob("*.py")
        assert p3.kind == "glob"

        p4 = NamePattern.regex(r"\d+")
        assert p4.kind == "regex"


class TestGlobToRegex:
    """Tests for _glob_to_regex helper."""

    def test_simple_glob(self) -> None:
        assert _glob_to_regex("test*") == "^test[^:]*$"

    def test_multiple_wildcards(self) -> None:
        assert _glob_to_regex("a*b*c") == "^a[^:]*b[^:]*c$"

    def test_special_chars_escaped(self) -> None:
        assert _glob_to_regex("test.py") == "^test\\.py$"


# ---------------------------------------------------------------------------
# MotifNode tests
# ---------------------------------------------------------------------------

class TestMotifNode:
    """Tests for MotifNode dataclass."""

    def test_default_values(self) -> None:
        node = MotifNode()
        assert node.id == 0
        assert node.kind is None
        assert node.name is None
        assert node.min_degree is None

    def test_with_values(self) -> None:
        node = MotifNode(id=1, kind="function", name=NamePattern.exact("foo"))
        assert node.id == 1
        assert node.kind == "function"
        assert node.name.matches("foo")


# ---------------------------------------------------------------------------
# Motif tests
# ---------------------------------------------------------------------------

class TestMotif:
    """Tests for Motif."""

    def test_empty_motif(self) -> None:
        motif = Motif()
        ok, _ = motif.validate()
        assert ok

    def test_valid_motif(self) -> None:
        motif = Motif(
            nodes=[MotifNode(id=0), MotifNode(id=1)],
            edges=[MotifEdge(from_id=0, to_id=1, kind=EdgeKind.CALLS)],
        )
        ok, _ = motif.validate()
        assert ok

    def test_duplicate_node_id(self) -> None:
        motif = Motif(nodes=[MotifNode(id=0), MotifNode(id=0)])
        ok, msg = motif.validate()
        assert not ok
        assert "duplicate" in msg

    def test_edge_from_missing_node(self) -> None:
        motif = Motif(
            nodes=[MotifNode(id=0)],
            edges=[MotifEdge(from_id=1, to_id=0, kind=EdgeKind.CALLS)],
        )
        ok, _ = motif.validate()
        assert not ok

    def test_edge_to_missing_node(self) -> None:
        motif = Motif(
            nodes=[MotifNode(id=0)],
            edges=[MotifEdge(from_id=0, to_id=1, kind=EdgeKind.CALLS)],
        )
        ok, _ = motif.validate()
        assert not ok

    def test_builder_empty(self) -> None:
        motif = Motif.builder().build()
        assert len(motif.nodes) == 0
        assert len(motif.edges) == 0

    def test_builder_with_nodes(self) -> None:
        motif = (
            Motif.builder()
            .add_node(lambda b: b.kind(NodeKind.FUNCTION))
            .add_node(lambda b: b.kind(NodeKind.CLASS))
            .build()
        )
        assert len(motif.nodes) == 2
        assert motif.nodes[0].kind == NodeKind.FUNCTION

    def test_builder_with_edges(self) -> None:
        motif = (
            Motif.builder()
            .add_node(lambda b: b.kind(NodeKind.FUNCTION))
            .add_node(lambda b: b.kind(NodeKind.FUNCTION))
            .add_edge(0, 1, EdgeKind.CALLS)
            .build()
        )
        assert len(motif.edges) == 1
        assert motif.edges[0].from_id == 0
        assert motif.edges[0].to_id == 1

    def test_builder_node_name(self) -> None:
        motif = (
            Motif.builder()
            .add_node(lambda b: b.kind(NodeKind.FUNCTION).name_exact("foo"))
            .build()
        )
        assert motif.nodes[0].name.matches("foo")

    def test_builder_node_contains(self) -> None:
        motif = (
            Motif.builder()
            .add_node(lambda b: b.name_contains("handler"))
            .build()
        )
        assert motif.nodes[0].name.matches("my_handler")

    def test_builder_node_glob(self) -> None:
        motif = (
            Motif.builder()
            .add_node(lambda b: b.name_glob("test_*"))
            .build()
        )
        assert motif.nodes[0].name.matches("test_foo")

    def test_builder_node_regex(self) -> None:
        motif = (
            Motif.builder()
            .add_node(lambda b: b.name_regex(r"^\w+_v2$"))
            .build()
        )
        assert motif.nodes[0].name.matches("foo_v2")

    def test_builder_node_min_degree(self) -> None:
        motif = (
            Motif.builder()
            .add_node(lambda b: b.min_degree(5))
            .build()
        )
        assert motif.nodes[0].min_degree == 5

    def test_builder_complex_motif(self) -> None:
        motif = (
            Motif.builder()
            .add_node(lambda b: b.kind(NodeKind.FUNCTION).name_contains("process"))
            .add_node(lambda b: b.kind(NodeKind.CLASS).name_regex(r".*Controller"))
            .add_edge(0, 1, EdgeKind.CALLS)
            .build()
        )
        ok, _ = motif.validate()
        assert ok
        assert len(motif.nodes) == 2
        assert len(motif.edges) == 1


# ---------------------------------------------------------------------------
# Path analysis tests
# ---------------------------------------------------------------------------

class TestFindPaths:
    """Tests for find_paths."""

    def test_direct_path(self) -> None:
        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "foo"))
        n2 = graph.add_node(Node.new(NodeKind.FUNCTION, "bar"))
        graph.add_edge(n1, n2, Edge.extracted(EdgeKind.CALLS))

        q = PathQuery(from_id=n1, to_id=n2)
        paths = find_paths(graph, q)
        assert len(paths) >= 1

    def test_no_path(self) -> None:
        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "foo"))
        n2 = graph.add_node(Node.new(NodeKind.FUNCTION, "bar"))

        q = PathQuery(from_id=n1, to_id=n2)
        paths = find_paths(graph, q)
        assert len(paths) == 0

    def test_indirect_path(self) -> None:
        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "a"))
        n2 = graph.add_node(Node.new(NodeKind.FUNCTION, "b"))
        n3 = graph.add_node(Node.new(NodeKind.FUNCTION, "c"))
        graph.add_edge(n1, n2, Edge.extracted(EdgeKind.CALLS))
        graph.add_edge(n2, n3, Edge.extracted(EdgeKind.CALLS))

        q = PathQuery(from_id=n1, to_id=n3)
        paths = find_paths(graph, q)
        assert len(paths) >= 1

    def test_max_hops(self) -> None:
        graph = Graph()
        for i in range(5):
            n = graph.add_node(Node.new(NodeKind.FUNCTION, f"n{i}"))
            if i > 0:
                graph.add_edge(
                    NodeId(i - 1), n, Edge.extracted(EdgeKind.CALLS)
                )

        # With max_hops=3, should NOT reach node 4 from node 0
        q = PathQuery(from_id=NodeId(0), to_id=NodeId(4), max_hops=3)
        paths = find_paths(graph, q)
        assert len(paths) == 0

        # With max_hops=10, should reach
        q2 = PathQuery(from_id=NodeId(0), to_id=NodeId(4), max_hops=10)
        paths2 = find_paths(graph, q2)
        assert len(paths2) >= 1


class TestCalleesOf:
    """Tests for callees_of."""

    def test_single_call(self) -> None:
        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "foo"))
        n2 = graph.add_node(Node.new(NodeKind.FUNCTION, "bar"))
        graph.add_edge(n1, n2, Edge.extracted(EdgeKind.CALLS))

        callees = callees_of(graph, n1)
        assert len(callees) == 1
        assert n2 in callees

    def test_multiple_callees(self) -> None:
        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "foo"))
        n2 = graph.add_node(Node.new(NodeKind.FUNCTION, "bar"))
        n3 = graph.add_node(Node.new(NodeKind.FUNCTION, "baz"))
        graph.add_edge(n1, n2, Edge.extracted(EdgeKind.CALLS))
        graph.add_edge(n1, n3, Edge.extracted(EdgeKind.CALLS))

        callees = callees_of(graph, n1)
        assert len(callees) == 2
        assert n2 in callees
        assert n3 in callees

    def test_no_callees(self) -> None:
        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "foo"))

        callees = callees_of(graph, n1)
        assert callees == []

    def test_max_hops(self) -> None:
        graph = Graph()
        for i in range(4):
            n = graph.add_node(Node.new(NodeKind.FUNCTION, f"n{i}"))
            if i > 0:
                graph.add_edge(
                    NodeId(i - 1), n, Edge.extracted(EdgeKind.CALLS)
                )

        callees_2hop = callees_of(graph, NodeId(0), max_hops=2)
        assert len(callees_2hop) == 2  # n1 and n2

        callees_3hop = callees_of(graph, NodeId(0), max_hops=3)
        assert len(callees_3hop) == 3  # n1, n2, n3


class TestCallersOf:
    """Tests for callers_of."""

    def test_single_caller(self) -> None:
        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "foo"))
        n2 = graph.add_node(Node.new(NodeKind.FUNCTION, "bar"))
        graph.add_edge(n1, n2, Edge.extracted(EdgeKind.CALLS))

        callers = callers_of(graph, n2)
        assert len(callers) == 1
        assert n1 in callers

    def test_multiple_callers(self) -> None:
        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "foo"))
        n2 = graph.add_node(Node.new(NodeKind.FUNCTION, "bar"))
        n3 = graph.add_node(Node.new(NodeKind.FUNCTION, "baz"))
        graph.add_edge(n1, n3, Edge.extracted(EdgeKind.CALLS))
        graph.add_edge(n2, n3, Edge.extracted(EdgeKind.CALLS))

        callers = callers_of(graph, n3)
        assert len(callers) == 2
        assert n1 in callers
        assert n2 in callers

    def test_no_callers(self) -> None:
        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "foo"))

        callers = callers_of(graph, n1)
        assert callers == []


class TestMaxDepthFrom:
    """Tests for max_depth_from."""

    def test_no_outgoing(self) -> None:
        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "foo"))

        depth = max_depth_from(graph, n1)
        assert depth == 0

    def test_linear_chain(self) -> None:
        graph = Graph()
        for i in range(4):
            n = graph.add_node(Node.new(NodeKind.FUNCTION, f"n{i}"))
            if i > 0:
                graph.add_edge(
                    NodeId(i - 1), n, Edge.extracted(EdgeKind.CALLS)
                )

        depth = max_depth_from(graph, NodeId(0))
        assert depth == 3

    def test_diamond(self) -> None:
        # a -> b -> d, a -> c -> d
        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "a"))
        n2 = graph.add_node(Node.new(NodeKind.FUNCTION, "b"))
        n3 = graph.add_node(Node.new(NodeKind.FUNCTION, "c"))
        n4 = graph.add_node(Node.new(NodeKind.FUNCTION, "d"))
        graph.add_edge(n1, n2, Edge.extracted(EdgeKind.CALLS))
        graph.add_edge(n1, n3, Edge.extracted(EdgeKind.CALLS))
        graph.add_edge(n2, n4, Edge.extracted(EdgeKind.CALLS))
        graph.add_edge(n3, n4, Edge.extracted(EdgeKind.CALLS))

        depth = max_depth_from(graph, n1)
        assert depth == 2  # a -> b -> d (or a -> c -> d)

    def test_max_hops_limit(self) -> None:
        graph = Graph()
        for i in range(6):
            n = graph.add_node(Node.new(NodeKind.FUNCTION, f"n{i}"))
            if i > 0:
                graph.add_edge(
                    NodeId(i - 1), n, Edge.extracted(EdgeKind.CALLS)
                )

        # With max_hops=2, depth should be 2
        depth = max_depth_from(graph, NodeId(0), max_hops=2)
        assert depth == 2

        # With max_hops=10, depth should be 5 (full chain)
        depth = max_depth_from(graph, NodeId(0), max_hops=10)
        assert depth == 5


class TestFindTopPaths:
    """Tests for find_top_paths."""

    def test_single_path(self) -> None:
        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "foo"))
        n2 = graph.add_node(Node.new(NodeKind.FUNCTION, "bar"))
        graph.add_edge(n1, n2, Edge.extracted(EdgeKind.CALLS))

        q = PathQuery(from_id=n1, to_id=n2)
        top = find_top_paths(graph, q, limit=5)
        assert len(top) >= 1

    def test_no_path(self) -> None:
        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "foo"))
        n2 = graph.add_node(Node.new(NodeKind.FUNCTION, "bar"))

        q = PathQuery(from_id=n1, to_id=n2)
        top = find_top_paths(graph, q, limit=5)
        assert top == []

    def test_weighted_paths(self) -> None:
        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "foo"))
        n2 = graph.add_node(Node.new(NodeKind.FUNCTION, "bar"))
        n3 = graph.add_node(Node.new(NodeKind.FUNCTION, "baz"))

        graph.add_edge(n1, n2, Edge.extracted(EdgeKind.CALLS))
        graph.add_edge(n1, n3, Edge.extracted(EdgeKind.DEFINES))

        q = PathQuery(from_id=n1)
        top = find_top_paths(graph, q, limit=5)
        assert len(top) >= 1

    def test_limit(self) -> None:
        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "foo"))
        n2 = graph.add_node(Node.new(NodeKind.FUNCTION, "bar"))
        n3 = graph.add_node(Node.new(NodeKind.FUNCTION, "baz"))
        graph.add_edge(n1, n2, Edge.extracted(EdgeKind.CALLS))
        graph.add_edge(n1, n3, Edge.extracted(EdgeKind.CALLS))

        q = PathQuery(from_id=n1)
        top = find_top_paths(graph, q, limit=1)
        assert len(top) <= 1
