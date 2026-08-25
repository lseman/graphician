"""Tests for the type-placeholder resolver."""

from graphician.core.edge import Edge, EdgeKind
from graphician.core.graph import Graph
from graphician.core.node import Node, NodeKind
from graphician.extraction import type_resolution as type_resolution_module
from graphician.extraction.type_resolution import (
    resolve_type_placeholders,
)


class TestResolveTypePlaceholders:
    def test_rewires_implements_edge(self):
        g = Graph()
        placeholder_id = g.add_node(
            Node.new(NodeKind.CLASS, "type::PaymentGateway"))
        owner_id = g.add_node(Node.new(NodeKind.CLASS, "StripeGateway"))
        g.add_edge(owner_id, placeholder_id,
                   Edge.extracted(EdgeKind.IMPLEMENTS))

        real_iface_id = g.add_node(
            Node.new(NodeKind.TRAIT,
                     "file::PaymentGateway.java::PaymentGateway"))

        rewired = resolve_type_placeholders(g)
        assert rewired == 1
        # Owner should now point at the real interface
        assert any(
            dst == real_iface_id and e.kind == EdgeKind.IMPLEMENTS
            for dst, e in g.out_neighbors(owner_id)
        )
        # Placeholder should be gone (no remaining edges)
        assert g.find_by_qname("type::PaymentGateway") is None

    def test_rewires_inherits_edge(self):
        g = Graph()
        placeholder_id = g.add_node(
            Node.new(NodeKind.CLASS, "type::BaseModel"))
        child_id = g.add_node(Node.new(NodeKind.CLASS, "UserModel"))
        g.add_edge(child_id, placeholder_id,
                   Edge.extracted(EdgeKind.INHERITS))

        real_base_id = g.add_node(
            Node.new(NodeKind.CLASS,
                     "file::models.py::BaseModel"))

        rewired = resolve_type_placeholders(g)
        assert rewired == 1
        assert any(
            dst == real_base_id and e.kind == EdgeKind.INHERITS
            for dst, e in g.out_neighbors(child_id)
        )

    def test_skips_ambiguous_names(self):
        g = Graph()
        placeholder_id = g.add_node(Node.new(NodeKind.CLASS, "type::Base"))
        owner_id = g.add_node(Node.new(NodeKind.CLASS, "Derived"))
        g.add_edge(owner_id, placeholder_id,
                   Edge.extracted(EdgeKind.INHERITS))

        g.add_node(Node.new(NodeKind.CLASS, "pkg_a::Base"))
        g.add_node(Node.new(NodeKind.CLASS, "pkg_b::Base"))

        rewired = resolve_type_placeholders(g)
        assert rewired == 0
        assert g.find_by_qname("type::Base") is not None

    def test_is_idempotent(self):
        g = Graph()
        placeholder_id = g.add_node(Node.new(NodeKind.CLASS, "type::Base"))
        owner_id = g.add_node(Node.new(NodeKind.CLASS, "Derived"))
        g.add_edge(owner_id, placeholder_id,
                   Edge.extracted(EdgeKind.INHERITS))
        g.add_node(Node.new(NodeKind.CLASS, "pkg::Base"))

        first = resolve_type_placeholders(g)
        second = resolve_type_placeholders(g)
        assert first == 1
        assert second == 0

    def test_leaves_non_supertype_edges_alone(self):
        g = Graph()
        placeholder_id = g.add_node(Node.new(NodeKind.CLASS, "type::Foo"))
        owner_id = g.add_node(Node.new(NodeKind.CLASS, "Bar"))
        # A Defines edge (not Inherits/Implements) should not be touched.
        g.add_edge(owner_id, placeholder_id,
                   Edge.extracted(EdgeKind.DEFINES))
        g.add_node(Node.new(NodeKind.CLASS, "pkg::Foo"))

        rewired = resolve_type_placeholders(g)
        assert rewired == 0
        assert g.find_by_qname("type::Foo") is not None

    def test_no_candidates_noop(self):
        g = Graph()
        placeholder_id = g.add_node(Node.new(NodeKind.CLASS, "type::Missing"))
        owner_id = g.add_node(Node.new(NodeKind.CLASS, "Something"))
        g.add_edge(owner_id, placeholder_id,
                   Edge.extracted(EdgeKind.INHERITS))

        rewired = resolve_type_placeholders(g)
        assert rewired == 0
        assert g.find_by_qname("type::Missing") is not None

    def test_native_plan_matches_python_fallback(self, monkeypatch):
        def make_graph():
            graph = Graph()
            placeholder = graph.add_node(Node.new(NodeKind.CLASS, "type::Gateway"))
            owner = graph.add_node(Node.new(NodeKind.CLASS, "pkg::Owner"))
            graph.add_edge(owner, placeholder, Edge.extracted(EdgeKind.IMPLEMENTS))
            graph.add_node(Node.new(NodeKind.TRAIT, "pkg::Gateway"))
            ambiguous = graph.add_node(Node.new(NodeKind.CLASS, "type::Base"))
            graph.add_edge(owner, ambiguous, Edge.extracted(EdgeKind.INHERITS))
            graph.add_node(Node.new(NodeKind.CLASS, "a::Base"))
            graph.add_node(Node.new(NodeKind.CLASS, "b::Base"))
            return graph

        native_graph = make_graph()
        fallback_graph = make_graph()
        assert resolve_type_placeholders(native_graph) == 1

        monkeypatch.setattr(type_resolution_module, "plan_type_resolution", None)
        assert resolve_type_placeholders(fallback_graph) == 1

        def topology(graph):
            return sorted(
                (
                    graph.node(source).qualified_name,
                    graph.node(target).qualified_name,
                    edge.kind.value,
                    edge.confidence.value,
                )
                for _, source, target, edge in graph.edges()
            )

        assert topology(native_graph) == topology(fallback_graph)

    def test_native_failure_uses_python_fallback(self, monkeypatch):
        graph = Graph()
        placeholder = graph.add_node(Node.new(NodeKind.CLASS, "type::Base"))
        owner = graph.add_node(Node.new(NodeKind.CLASS, "pkg::Owner"))
        graph.add_edge(owner, placeholder, Edge.extracted(EdgeKind.INHERITS))
        real = graph.add_node(Node.new(NodeKind.CLASS, "pkg::Base"))

        def fail(*_args):
            raise RuntimeError("native unavailable")

        monkeypatch.setattr(type_resolution_module, "plan_type_resolution", fail)

        assert resolve_type_placeholders(graph) == 1
        assert any(
            target == real and edge.kind == EdgeKind.INHERITS
            for target, edge in graph.out_neighbors(owner)
        )
