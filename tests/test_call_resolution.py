"""Tests for the 6-tier call-placeholder resolver."""

import pytest

from graphician.core.edge import Edge, EdgeKind
from graphician.core.graph import Graph
from graphician.core.id import NodeId
from graphician.core.node import Node, NodeKind
from graphician.extraction.call_resolution import (
    _infer_type_from_let_bindings,
    _infer_type_from_receiver_expression,
    common_prefix_len,
    module_stem,
    resolve_call_placeholders,
    should_suppress_call_placeholder,
)


def _make_fn(graph: Graph, qname: str, uri: str | None = None, line: int = 0) -> NodeId:
    node = Node.new(NodeKind.FUNCTION, qname)
    if uri is not None:
        node = node.with_source(uri, line, line + 1)
    return graph.add_node(node)


def _make_method(graph: Graph, qname: str, uri: str | None = None) -> NodeId:
    node = Node.new(NodeKind.METHOD, qname)
    if uri is not None:
        node = node.with_source(uri, 0, 1)
    return graph.add_node(node)


class TestSuppression:
    def test_known_builtin_suppressed(self):
        assert should_suppress_call_placeholder("len")
        assert should_suppress_call_placeholder("to_string_lossy")
        assert should_suppress_call_placeholder("edges_directed")
        assert should_suppress_call_placeholder("unwrap_or")
        assert should_suppress_call_placeholder("printf")
        assert should_suppress_call_placeholder("clone")
        assert should_suppress_call_placeholder("collect")
        assert should_suppress_call_placeholder("map_err")

    def test_project_name_not_suppressed(self):
        assert not should_suppress_call_placeholder("resolve_call_placeholders")
        assert not should_suppress_call_placeholder("extract_file")
        assert not should_suppress_call_placeholder("custom_handler")

    def test_empty_name_suppressed(self):
        assert should_suppress_call_placeholder("")

    def test_suppressed_placeholder_is_pruned_from_graph(self):
        graph = Graph()
        caller = _make_fn(graph, "file::caller", "main.py")
        placeholder = _make_fn(graph, "call::len")
        graph.add_edge(caller, placeholder, Edge.ambiguous(EdgeKind.CALLS))

        assert resolve_call_placeholders(graph) == 0
        assert graph.find_by_qname("call::len") is None
        assert list(graph.out_neighbors(caller)) == []


class TestHelpers:
    def test_common_prefix_len_exact(self):
        assert common_prefix_len("a::b::c", "a::b::c") == 3

    def test_common_prefix_len_partial(self):
        assert common_prefix_len("a::b::c", "a::b::d") == 2

    def test_common_prefix_len_none(self):
        assert common_prefix_len("a::b", "x::y") == 0
        assert common_prefix_len("", "a::b") == 0

    def test_module_stem_simple(self):
        assert module_stem("src/auth.rs") == "auth"

    def test_module_stem_container_mod(self):
        assert module_stem("src/auth/mod.rs") == "auth"
        assert module_stem("pkg/auth/__init__.py") == "auth"
        assert module_stem("web/auth/index.ts") == "auth"
        # lib/main.rs → "main" is a container name → return parent "lib"
        assert module_stem("lib/main.rs") == "lib"

    def test_python_classmethod_constructor_infers_receiver_type(self):
        source = "node = Node.new(NodeKind.FUNCTION, name)\nnode.with_property('x', 1)"
        assert _infer_type_from_let_bindings(source, "node") == "Node"
        assert _infer_type_from_receiver_expression("Node.new(kind, qn).with_source(uri)") == "Node"


class TestTier1UniqueName:
    def test_single_candidate_resolves(self):
        g = Graph()
        caller = _make_fn(g, "file::caller", "main.py")
        ph = _make_fn(g, "call::login")
        g.add_edge(caller, ph, Edge.ambiguous(EdgeKind.CALLS))
        real = _make_fn(g, "pkg/auth.py::login", "pkg/auth.py")
        resolved = resolve_call_placeholders(g)
        assert resolved == 1
        points = list(g.out_neighbors(caller))
        assert any(dst == real for dst, _ in points)

    def test_no_candidate_unchanged(self):
        g = Graph()
        caller = _make_fn(g, "file::caller", "main.py")
        ph = _make_fn(g, "call::nonexistent")
        g.add_edge(caller, ph, Edge.ambiguous(EdgeKind.CALLS))
        resolved = resolve_call_placeholders(g)
        assert resolved == 0


class TestTier2FileLocal:
    def test_file_local_wins_over_global(self):
        g = Graph()
        caller_a = _make_fn(g, "file::entry_a", "src/a.rs")
        ph = _make_fn(g, "call::shared")
        g.add_edge(caller_a, ph, Edge.ambiguous(EdgeKind.CALLS))
        shared_a = _make_fn(g, "src/a.rs::shared", "src/a.rs")
        shared_b = _make_fn(g, "src/b.rs::shared", "src/b.rs")

        resolved = resolve_call_placeholders(g)
        assert resolved == 1
        points = list(g.out_neighbors(caller_a))
        assert any(dst == shared_a for dst, _ in points)
        assert not any(dst == shared_b for dst, _ in points)


class TestTier3Scoped:
    def test_scoped_call_resolves(self):
        g = Graph()
        caller = _make_fn(g, "file::entry", "src/lib.rs")
        ph = _make_fn(g, "call::shared")
        edge = Edge.ambiguous(EdgeKind.CALLS)
        edge.properties["call_scope"] = "beta"
        g.add_edge(caller, ph, edge)

        _make_fn(g, "src/lib.rs::alpha::shared", "src/lib.rs")
        beta = _make_fn(g, "src/lib.rs::beta::shared", "src/lib.rs")

        resolved = resolve_call_placeholders(g)
        assert resolved == 1
        points = list(g.out_neighbors(caller))
        assert any(dst == beta for dst, _ in points)


class TestTier4Receiver:
    @pytest.mark.parametrize(
        ("annotation", "expected_type"),
        [
            ("graph: &mut dyn GraphMut", "GraphMut"),
            ("store: &crate::persistence::Store", "Store"),
        ],
    )
    def test_parameter_annotation_resolves_receiver(self, annotation, expected_type):
        g = Graph()
        receiver = annotation.split(":", 1)[0]
        caller_node = Node.new(NodeKind.FUNCTION, "file::src/main.rs::run")
        caller_node.with_source("src/main.rs", 0, 1)
        caller_node.with_source_text(
            f"fn run({annotation}) {{ {receiver}.persist(); }}"
        )
        caller = g.add_node(caller_node)
        ph = _make_fn(g, "call::persist")
        edge = Edge.ambiguous(EdgeKind.CALLS)
        edge.properties["call_receiver"] = receiver
        g.add_edge(caller, ph, edge)
        expected = _make_method(g, f"file::src/types.rs::{expected_type}::persist")
        _make_method(g, "file::src/types.rs::Other::persist")

        assert resolve_call_placeholders(g) == 1

        assert any(dst == expected for dst, edge in g.out_neighbors(caller))

    def test_self_receiver_uses_surrounding_impl(self):
        g = Graph()
        source = """impl Store {
    fn save(&self) {
        self.persist();
    }
}
"""
        caller_node = Node.new(NodeKind.FUNCTION, "file::src/store.rs::save")
        caller_node.with_source("src/store.rs", 2, 3).with_source_text(source)
        caller = g.add_node(caller_node)
        ph = _make_fn(g, "call::persist")
        edge = Edge.ambiguous(EdgeKind.CALLS)
        edge.properties["call_receiver"] = "self"
        g.add_edge(caller, ph, edge)
        expected = _make_method(g, "file::src/store.rs::Store::persist")
        _make_method(g, "file::src/other.rs::Other::persist")

        assert resolve_call_placeholders(g) == 1

        assert any(dst == expected for dst, edge in g.out_neighbors(caller))


class TestTier5ImportScoped:
    def test_import_scoped_resolves(self):
        g = Graph()
        # Caller file
        caller_file = _make_fn(g, "file::main.py", "main.py", 0)
        # Module import: main.py imports pkg.auth
        module = _make_fn(g, "module::pkg.auth", "pkg/auth.py")
        g.add_edge(caller_file, module, Edge.extracted(EdgeKind.IMPORTS))

        # Caller function
        caller = _make_fn(g, "main.py::entry", "main.py")
        ph = _make_fn(g, "call::login")
        g.add_edge(caller, ph, Edge.ambiguous(EdgeKind.CALLS))

        # Two candidates
        auth_login = _make_fn(g, "pkg/auth.py::login", "pkg/auth.py")
        _make_fn(g, "pkg/billing.py::login", "pkg/billing.py")

        resolved = resolve_call_placeholders(g)
        assert resolved >= 1
        # auth_login should be the resolved target (import_scoped)
        points = list(g.out_neighbors(caller))
        call_targets = [dst for dst, e in points if e.kind == EdgeKind.CALLS]
        assert auth_login in call_targets


class TestTier6SameDir:
    def test_same_dir_wins(self):
        g = Graph()
        pipeline_fn = _make_fn(
            g, "file::src/pipeline/mod::process", "src/pipeline/mod.rs"
        )
        io_fn = _make_fn(
            g, "file::src/io/mod::process", "src/io/mod.rs"
        )

        caller = _make_fn(
            g, "file::src/pipeline/runner::run", "src/pipeline/runner.rs"
        )
        ph = _make_fn(g, "call::process")
        g.add_edge(caller, ph, Edge.ambiguous(EdgeKind.CALLS))

        resolved = resolve_call_placeholders(g)
        assert resolved >= 1
        points = list(g.out_neighbors(caller))
        call_targets = [dst for dst, e in points if e.kind == EdgeKind.CALLS]
        assert pipeline_fn in call_targets
        assert io_fn not in call_targets


class TestTier7FreqPrior:
    def test_most_called_wins(self):
        g = Graph()
        a = _make_fn(g, "file::src/a::render")
        b = _make_fn(g, "file::src/b::render")

        caller1 = _make_fn(g, "file::src/x::caller1")
        caller2 = _make_fn(g, "file::src/x::caller2")
        g.add_edge(caller1, b, Edge.extracted(EdgeKind.CALLS))
        g.add_edge(caller2, b, Edge.extracted(EdgeKind.CALLS))

        new_caller = _make_fn(g, "file::src/y::new_caller")
        ph = _make_fn(g, "call::render")
        g.add_edge(new_caller, ph, Edge.ambiguous(EdgeKind.CALLS))

        resolved = resolve_call_placeholders(g)
        assert resolved >= 1
        points = list(g.out_neighbors(new_caller))
        call_targets = [dst for dst, e in points if e.kind == EdgeKind.CALLS]
        assert b in call_targets
        assert a not in call_targets


class TestEdgeProperties:
    def test_resolved_edge_has_tag(self):
        g = Graph()
        caller = _make_fn(g, "file::caller", "main.py")
        ph = _make_fn(g, "call::login")
        g.add_edge(caller, ph, Edge.ambiguous(EdgeKind.CALLS))
        real = _make_fn(g, "pkg/auth.py::login", "pkg/auth.py")

        resolve_call_placeholders(g)

        points = list(g.out_neighbors(caller))
        for dst, edge in points:
            if dst == real:
                assert edge.properties.get("resolved_from") is not None
                break


class TestStaleRemoval:
    def test_placeholder_edge_removed_after_resolution(self):
        g = Graph()
        caller = _make_fn(g, "file::caller", "main.py")
        ph = _make_fn(g, "call::login")
        g.add_edge(caller, ph, Edge.ambiguous(EdgeKind.CALLS))
        _make_fn(g, "pkg/auth.py::login", "pkg/auth.py")

        resolve_call_placeholders(g)

        # The placeholder node should be removed if it has no edges
        assert g.node(ph) is None

    def test_placeholder_kept_when_no_candidates(self):
        g = Graph()
        caller = _make_fn(g, "file::caller", "main.py")
        ph = _make_fn(g, "call::nonexistent")
        g.add_edge(caller, ph, Edge.ambiguous(EdgeKind.CALLS))

        resolve_call_placeholders(g)

        assert g.node(ph) is not None
