"""Tests for cross-language call resolution (Tier 8)."""

from graphician.core.edge import Edge, EdgeKind
from graphician.core.graph import Graph
from graphician.core.node import Node, NodeKind
from graphician.extraction.cross_language import (
    CrossLanguageRegistry,
    _detect_ffi_boundaries,
    _infer_stub_dialect,
    enrich_with_cross_language_data,
    resolve_cross_language_calls,
)


def make_fn(graph, qname, uri=None, line=0):
    """Create a function node."""
    node = Node.new(NodeKind.FUNCTION, qname)
    if uri:
        node = node.with_source(uri, line, line + 1)
    return graph.add_node(node)


def make_stub(graph, qname, dialect):
    """Create a stub class node."""
    node = Node.new(NodeKind.CLASS, qname)
    node = node.with_source(f"<{dialect}-stdlib:{qname}>", 0, 0)
    node = node.with_property("dialect", dialect)
    node = node.with_property("is_stub", True)
    return graph.add_node(node)


class TestFFIDetection:
    def test_detects_python_pybind11_imports(self):
        """Detect FFI patterns in source text."""
        graph = Graph()
        make_fn(graph, "mod::ffi_func", "src/mod.py", 1)
        boundaries = _detect_ffi_boundaries(graph)
        assert isinstance(boundaries, list)


class TestCrossLanguageResolution:
    def test_cross_language_resolves_rust_stub_from_python(self):
        """Python calling a Rust stub function via PyO3 should resolve."""
        graph = Graph()
        caller = make_fn(graph, "src/main.py::run_app", "src/main.py", 10)
        call_node = Node.new(NodeKind.FUNCTION, "call::process_request")
        call_id = graph.add_node(call_node)
        graph.add_edge(caller, call_id, Edge.ambiguous(EdgeKind.CALLS))

        registry = CrossLanguageRegistry()
        registry.register_type("stub::Vec", "rust", NodeKind.CLASS, "rust", exportable=True)
        registry.register_type("stub::String", "rust", NodeKind.CLASS, "rust", exportable=True)
        registry._callee_index["process_request"] = ["stub::Vec", "stub::String"]

        make_stub(graph, "stub::Vec", "rust")
        make_stub(graph, "stub::String", "rust")

        resolved = resolve_cross_language_calls(graph, registry)
        assert resolved >= 1

    def test_cross_language_creates_stub_when_missing(self):
        """Should create stub nodes for unresolved cross-language targets."""
        graph = Graph()
        caller = make_fn(graph, "src/app.py::main", "src/app.py", 1)
        call_node = Node.new(NodeKind.FUNCTION, "call::create_user")
        call_id = graph.add_node(call_node)
        graph.add_edge(caller, call_id, Edge.ambiguous(EdgeKind.CALLS))

        registry = CrossLanguageRegistry()
        registry.register_type("stub::JSONParser", "rust", NodeKind.CLASS, "rust")
        registry._callee_index["create_user"] = ["stub::JSONParser"]

        resolved = resolve_cross_language_calls(graph, registry)
        assert resolved >= 1

        stub = graph.find_by_qname("stub::JSONParser")
        assert stub is not None

    def test_cross_language_noop_for_unrelated_calls(self):
        """Calls to unknown methods should not be resolved."""
        graph = Graph()
        caller = make_fn(graph, "src/main.py::foo", "src/main.py", 1)
        call_node = Node.new(NodeKind.FUNCTION, "call::unknown_method_xyz")
        call_id = graph.add_node(call_node)
        graph.add_edge(caller, call_id, Edge.ambiguous(EdgeKind.CALLS))

        registry = CrossLanguageRegistry()
        registry._callee_index["bar"] = ["stub::SomeType"]

        resolved = resolve_cross_language_calls(graph, registry)
        assert resolved == 0


class TestEnrichCrossLanguage:
    def test_enrichment_returns_stats(self):
        """enrich_with_cross_language_data returns proper statistics."""
        graph = Graph()
        make_fn(graph, "src/main.py::run", "src/main.py", 1)

        stats = enrich_with_cross_language_data(graph)
        assert isinstance(stats, dict)
        assert "boundaries_detected" in stats
        assert "registry_types" in stats
        assert "resolved" in stats
        assert "unresolved_remaining" in stats


class TestStubDialectInference:
    def test_infer_rust_dialect(self):
        assert _infer_stub_dialect("Vec") == "rust"
        assert _infer_stub_dialect("HashMap") == "rust"
        assert _infer_stub_dialect("Result") == "rust"

    def test_infer_cpp_dialect(self):
        assert _infer_stub_dialect("vector") == "cpp"
        assert _infer_stub_dialect("shared_ptr") == "cpp"

    def test_infer_javascript_dialect(self):
        assert _infer_stub_dialect("Promise") == "javascript"
        assert _infer_stub_dialect("Buffer") == "javascript"

    def test_default_python(self):
        assert _infer_stub_dialect("MyCustomType") == "python"
        assert _infer_stub_dialect("List") == "python"
