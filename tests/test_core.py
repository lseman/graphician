"""Tests for core graph types."""

from graphician.core.edge import Edge, EdgeKind, Confidence
from graphician.core.graph import Graph
from graphician.core.id import NodeId
from graphician.core.node import Node, NodeKind


class TestNodeId:
    def test_creation(self):
        nid = NodeId(42)
        assert nid.value == 42

    def test_frozen(self):
        nid = NodeId(1)
        try:
            nid.value = 2
            assert False, "Should be immutable"
        except AttributeError:
            pass


class TestNode:
    def test_new(self):
        node = Node.new(NodeKind.FUNCTION, "m::f")
        assert node.kind == NodeKind.FUNCTION
        assert node.name == "f"
        assert node.qualified_name == "m::f"

    def test_with_source(self):
        node = Node.new(NodeKind.FUNCTION, "m::f")
        node = node.with_source("file.py", 10, 20)
        assert node.source_uri == "file.py"
        assert node.line_start == 10
        assert node.line_end == 20

    def test_with_source_text_truncation(self):
        node = Node.new(NodeKind.FUNCTION, "m::f")
        big_text = "x" * 20000
        node = node.with_source_text(big_text)
        assert node.source_text is not None
        assert len(node.source_text.encode("utf-8")) <= 10000

    def test_with_property(self):
        node = Node.new(NodeKind.FUNCTION, "m::f")
        node = node.with_property("key", "value")
        assert node.properties == {"key": "value"}


class TestEdge:
    def test_extracted(self):
        edge = Edge.extracted(EdgeKind.CALLS)
        assert edge.kind == EdgeKind.CALLS
        assert edge.confidence == Confidence.EXTRACTED

    def test_inferred(self):
        edge = Edge.inferred(EdgeKind.MENTIONS, 0.8)
        assert edge.kind == EdgeKind.MENTIONS
        assert edge.confidence == Confidence.INFERRED

    def test_ambiguous(self):
        edge = Edge.ambiguous(EdgeKind.MENTIONS)
        assert edge.confidence == Confidence.AMBIGUOUS


class TestGraph:
    def test_add_and_traverse(self):
        g = Graph()
        a = g.add_node(Node.new(NodeKind.FUNCTION, "m::f"))
        b = g.add_node(Node.new(NodeKind.FUNCTION, "m::g"))
        g.add_edge(a, b, Edge.extracted(EdgeKind.CALLS))
        assert g.node_count() == 2
        assert g.edge_count() == 1
        callees = list(g.out_neighbors(a))
        assert len(callees) == 1
        nid, _ = callees[0]
        assert nid == b

    def test_find_by_qname(self):
        g = Graph()
        a = g.add_node(Node.new(NodeKind.FUNCTION, "m::f"))
        assert g.find_by_qname("m::f") == a

    def test_duplicate_qname_updates(self):
        g = Graph()
        a = g.add_node(Node.new(NodeKind.FUNCTION, "m::f"))
        a2 = g.add_node(Node.new(NodeKind.FUNCTION, "m::f"))
        assert a == a2
        assert g.node_count() == 1

    def test_remove_node(self):
        g = Graph()
        a = g.add_node(Node.new(NodeKind.FUNCTION, "m::f"))
        b = g.add_node(Node.new(NodeKind.FUNCTION, "m::g"))
        g.add_edge(a, b, Edge.extracted(EdgeKind.CALLS))
        g.remove_node(a)
        assert g.node_count() == 1
        assert g.edge_count() == 0

    def test_rename_node(self):
        g = Graph()
        a = g.add_node(Node.new(NodeKind.FUNCTION, "m::f"))
        g.rename_node(a, "n::f", "f")
        assert g.find_by_qname("n::f") == a
        assert g.find_by_qname("m::f") is None

    def test_rename_into_existing_merges(self):
        g = Graph()
        parent = g.add_node(Node.new(NodeKind.SECTION, "doc::f::a"))
        s1 = g.add_node(Node.new(NodeKind.SECTION, "doc::f::section-0"))
        s2 = g.add_node(Node.new(NodeKind.SECTION, "doc::f::section-1"))
        g.add_edge(parent, s1, Edge.extracted(EdgeKind.DEFINES))
        g.add_edge(parent, s2, Edge.extracted(EdgeKind.DEFINES))

        r1 = g.rename_node(s1, "doc::f::dup", "dup")
        assert r1 == s1
        r2 = g.rename_node(s2, "doc::f::dup", "dup")
        assert r2 == s1
        assert g.node_count() == 2

    def test_merge_graphs(self):
        g1 = Graph()
        a = g1.add_node(Node.new(NodeKind.FUNCTION, "m::f"))
        b = g1.add_node(Node.new(NodeKind.FUNCTION, "m::g"))
        g1.add_edge(a, b, Edge.extracted(EdgeKind.CALLS))

        g2 = Graph()
        c = g2.add_node(Node.new(NodeKind.FUNCTION, "n::h"))
        d = g2.add_node(Node.new(NodeKind.FUNCTION, "n::i"))
        g2.add_edge(c, d, Edge.extracted(EdgeKind.CALLS))

        g1.merge(g2)
        assert g1.node_count() == 4
        assert g1.edge_count() == 2

    def test_merge_deduplicates(self):
        g1 = Graph()
        a = g1.add_node(Node.new(NodeKind.FUNCTION, "m::f"))
        g1.add_node(Node.new(NodeKind.FILE, "file::a.rs"))

        g2 = Graph()
        _ = g2.add_node(Node.new(NodeKind.FUNCTION, "m::f"))
        g2.add_node(Node.new(NodeKind.FILE, "file::b.rs"))

        g1.merge(g2)
        assert g1.node_count() == 3

    def test_merge_deduplicates_edges(self):
        g1 = Graph()
        a = g1.add_node(Node.new(NodeKind.FUNCTION, "a"))
        b = g1.add_node(Node.new(NodeKind.FUNCTION, "b"))
        g1.add_edge(a, b, Edge.extracted(EdgeKind.CALLS))

        g2 = Graph()
        a2 = g2.add_node(Node.new(NodeKind.FUNCTION, "a"))
        b2 = g2.add_node(Node.new(NodeKind.FUNCTION, "b"))
        g2.add_edge(a2, b2, Edge.extracted(EdgeKind.CALLS))

        g1.merge(g2)
        assert g1.edge_count() == 1

    def test_self_loop_preserved_like_rust_graph(self):
        g = Graph()
        a = g.add_node(Node.new(NodeKind.FUNCTION, "m::f"))
        eid = g.add_edge(a, a, Edge.extracted(EdgeKind.CALLS))
        assert eid.value == 0
        assert g.edge_count() == 1

    def test_parallel_edge_preserved_like_rust_graph(self):
        g = Graph()
        a = g.add_node(Node.new(NodeKind.FUNCTION, "m::f"))
        b = g.add_node(Node.new(NodeKind.FUNCTION, "m::g"))
        g.add_edge(a, b, Edge.extracted(EdgeKind.CALLS))
        eid = g.add_edge(a, b, Edge.extracted(EdgeKind.CALLS))
        assert eid.value == 1
        assert g.edge_count() == 2

    def test_iteration(self):
        g = Graph()
        a = g.add_node(Node.new(NodeKind.FUNCTION, "m::f"))
        b = g.add_node(Node.new(NodeKind.FUNCTION, "m::g"))
        g.add_edge(a, b, Edge.extracted(EdgeKind.CALLS))

        nodes = list(g.nodes())
        edges = list(g.edges())
        assert len(nodes) == 2
        assert len(edges) == 1

    def test_in_neighbors(self):
        g = Graph()
        a = g.add_node(Node.new(NodeKind.FUNCTION, "m::f"))
        b = g.add_node(Node.new(NodeKind.FUNCTION, "m::g"))
        g.add_edge(a, b, Edge.extracted(EdgeKind.CALLS))
        callers = list(g.in_neighbors(b))
        assert len(callers) == 1
        nid, _ = callers[0]
        assert nid == a
