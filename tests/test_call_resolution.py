"""Tests for the 6-tier call-placeholder resolver."""

from pathlib import Path

import pytest

from graphician.core.edge import Edge, EdgeKind
from graphician.core.graph import Graph
from graphician.core.id import NodeId
from graphician.core.node import Node, NodeKind
from graphician.extraction import call_resolution as call_resolution_module
from graphician.extraction.call_resolution import (
    _infer_type_from_let_bindings,
    _infer_type_from_receiver_expression,
    _resolve_call_placeholders_python,
    common_prefix_len,
    module_stem,
    resolve_call_placeholders,
    should_suppress_call_placeholder,
)
from graphician.extraction.languages import LanguageRegistry
from graphician.extraction.pipeline import ExtractionPipeline


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
        # len, clone, collect were removed from suppression - they now have stub coverage
        # and should be resolved by the stub resolver instead of being suppressed
        assert not should_suppress_call_placeholder("len")
        assert not should_suppress_call_placeholder("clone")
        assert not should_suppress_call_placeholder("collect")
        assert should_suppress_call_placeholder("to_string_lossy")
        assert should_suppress_call_placeholder("edges_directed")
        assert should_suppress_call_placeholder("unwrap_or")
        assert should_suppress_call_placeholder("printf")
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
        # clone, collect, len were removed from suppression (now have stub coverage)
        placeholder = _make_fn(graph, "call::to_string_lossy")
        graph.add_edge(caller, placeholder, Edge.ambiguous(EdgeKind.CALLS))

        assert resolve_call_placeholders(graph) == 0
        assert graph.find_by_qname("call::to_string_lossy") is None
        assert list(graph.out_neighbors(caller)) == []

    @pytest.mark.parametrize("name", ["get", "load", "parse", "add", "len"])
    def test_suppressed_name_still_resolves_to_project_definition(self, name):
        graph = Graph()
        caller = _make_fn(graph, "file::caller", "main.py")
        placeholder = _make_fn(graph, f"call::{name}")
        graph.add_edge(caller, placeholder, Edge.ambiguous(EdgeKind.CALLS))
        target = _make_fn(graph, f"project::{name}", "main.py")

        assert resolve_call_placeholders(graph) == 1
        assert [(node_id, edge.kind) for node_id, edge in graph.out_neighbors(caller)] == [
            (target, EdgeKind.CALLS)
        ]

    def test_ambiguous_suppressed_name_without_strong_evidence_is_pruned(self):
        # edges_directed is still suppressed (no stub coverage)
        # Note: get was removed from suppression (now has stub coverage)
        graph = Graph()
        caller = _make_fn(graph, "file::caller", "app/main.py")
        placeholder = _make_fn(graph, "call::edges_directed")
        graph.add_edge(caller, placeholder, Edge.ambiguous(EdgeKind.CALLS))
        _make_fn(graph, "first::edges_directed", "lib/first.py")
        _make_fn(graph, "second::edges_directed", "vendor/second.py")

        assert resolve_call_placeholders(graph) == 0
        assert graph.find_by_qname("call::edges_directed") is None


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


def test_relative_import_and_alias_select_correct_duplicate_definition(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "other").mkdir()
    (tmp_path / "pkg" / "types.py").write_text("class Choice:\n    pass\n")
    (tmp_path / "other" / "types.py").write_text("class Choice:\n    pass\n")
    (tmp_path / "pkg" / "caller.py").write_text(
        "from .types import Choice as Selected\n"
        "def choose():\n"
        "    return Selected()\n"
    )

    graph = ExtractionPipeline(LanguageRegistry(), strict=True).build(tmp_path)
    caller = graph.find_by_qname("file::pkg/caller.py::choose")
    expected = graph.find_by_qname("file::pkg/types.py::Choice")
    other = graph.find_by_qname("file::other/types.py::Choice")

    assert caller is not None and expected is not None and other is not None
    targets = [
        target
        for target, edge in graph.out_neighbors(caller)
        if edge.kind == EdgeKind.CALLS
    ]
    assert targets == [expected]


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


def test_native_resolution_matches_python_fallback_across_tiers():
    def make_graph():
        graph = Graph()
        unique_caller = _make_fn(graph, "app::unique_caller", "src/main.py")
        unique_placeholder = _make_fn(graph, "call::unique_target")
        graph.add_edge(
            unique_caller, unique_placeholder, Edge.ambiguous(EdgeKind.CALLS)
        )
        _make_fn(graph, "lib::unique_target", "src/lib.py")

        local_caller = _make_fn(graph, "app::local_caller", "src/local.py")
        local_placeholder = _make_fn(graph, "call::local_target")
        graph.add_edge(
            local_caller, local_placeholder, Edge.ambiguous(EdgeKind.CALLS)
        )
        _make_fn(graph, "local::local_target", "src/local.py")
        _make_fn(graph, "remote::local_target", "other/local.py")

        receiver_node = Node.new(NodeKind.FUNCTION, "app::receiver_caller")
        receiver_node.with_source("src/receiver.py", 0, 1)
        receiver_node.with_source_text("store: Store\nstore.persist()")
        receiver_caller = graph.add_node(receiver_node)
        receiver_placeholder = _make_fn(graph, "call::persist")
        receiver_edge = Edge.ambiguous(EdgeKind.CALLS)
        receiver_edge.properties["call_receiver"] = "store"
        graph.add_edge(receiver_caller, receiver_placeholder, receiver_edge)
        _make_method(graph, "types::Store::persist", "types.py")
        _make_method(graph, "types::Other::persist", "types.py")

        noise_caller = _make_fn(graph, "app::noise", "src/main.py")
        noise_placeholder = _make_fn(graph, "call::len")
        graph.add_edge(noise_caller, noise_placeholder, Edge.ambiguous(EdgeKind.CALLS))
        return graph

    native = make_graph()
    fallback = make_graph()
    assert resolve_call_placeholders(native) == 3
    assert _resolve_call_placeholders_python(fallback) == 3

    def topology(graph):
        return sorted(
            (
                graph.node(source).qualified_name,
                graph.node(target).qualified_name,
                edge.kind.value,
                edge.confidence.value,
                edge.properties.get("resolved_from"),
            )
            for _, source, target, edge in graph.edges()
        )

    assert topology(native) == topology(fallback)


def test_native_resolution_failure_uses_python_fallback(monkeypatch):
    graph = Graph()
    caller = _make_fn(graph, "app::caller", "main.py")
    placeholder = _make_fn(graph, "call::target")
    graph.add_edge(caller, placeholder, Edge.ambiguous(EdgeKind.CALLS))
    target = _make_fn(graph, "app::target", "target.py")

    def fail(*_args):
        raise RuntimeError("native unavailable")

    monkeypatch.setattr(call_resolution_module, "plan_call_resolution", fail)
    assert resolve_call_placeholders(graph) == 1
    assert any(node_id == target for node_id, _edge in graph.out_neighbors(caller))
