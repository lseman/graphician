"""Tests for the type-placeholder resolver."""

import pytest

from ariadne_py.core.edge import Edge, EdgeKind
from ariadne_py.core.graph import Graph
from ariadne_py.core.id import NodeId
from ariadne_py.core.node import Node, NodeKind
from ariadne_py.extraction.type_resolution import (
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
