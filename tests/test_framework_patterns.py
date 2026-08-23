"""Tests for the framework pattern detection engine."""

import pytest

from graphician.core.edge import Edge, EdgeKind
from graphician.core.graph import Graph
from graphician.core.node import Node, NodeKind
from graphician.extraction.patterns.framework_patterns import (
    FrameworkPattern,
    PatternCategory,
    PatternMatch,
    _built_in_patterns,
    detect_patterns,
    load_patterns_from_file,
    merged_catalog,
    _match_single_pattern,
)


class TestBuiltInCatalog:
    def test_has_minimum_patterns(self):
        patterns = _built_in_patterns()
        assert len(patterns) >= 30, (
            f"Expected at least 30 patterns, got {len(patterns)}"
        )

    def test_unique_ids(self):
        patterns = _built_in_patterns()
        ids = [p.id for p in patterns]
        assert len(ids) == len(set(ids)), "Duplicate pattern IDs found"

    def test_valid_confidence_values(self):
        patterns = _built_in_patterns()
        for p in patterns:
            assert 0.0 <= p.min_confidence <= 1.0

    def test_has_all_categories(self):
        patterns = _built_in_patterns()
        categories = {p.category for p in patterns}
        expected = {
            PatternCategory.LIFECYCLE,
            PatternCategory.ROUTING,
            PatternCategory.MIDDLEWARE,
            PatternCategory.DEPENDENCY_INJECTION,
            PatternCategory.TESTING,
            PatternCategory.DATA_MAPPING,
            PatternCategory.STATE_MANAGEMENT,
            PatternCategory.VALIDATION,
            PatternCategory.COMMAND_LINE,
        }
        for cat in expected:
            assert cat in categories, f"Missing category: {cat}"


class TestDetectPatterns:
    def test_empty_graph(self):
        g = Graph()
        matches = detect_patterns(g)
        assert matches == []

    def test_fastapi_detected(self):
        """FastAPI pattern matches when signature names and import patterns match."""
        g = Graph()
        # Function names contain exact signature strings (case-sensitive).
        g.add_node(
            Node.new(NodeKind.FUNCTION, "app.py::APIRouter_routes")
                .with_source("fastapi_app.py", 0, 10)
        )
        g.add_node(
            Node.new(NodeKind.FUNCTION, "app.py::Depends_inject")
                .with_source("fastapi_app.py", 5, 15)
        )
        g.add_node(
            Node.new(NodeKind.FUNCTION, "app.py::Query_param")
                .with_source("fastapi_app.py", 10, 20)
        )
        # Each function needs an outgoing Defines edge for pattern matching.
        file_node = Node.new(NodeKind.FILE, "app.py")
        file_id = g.add_node(file_node)
        for nid, _ in g.nodes():
            if nid != file_id:
                # Edge FROM function TO file (out_neighbors checked by matcher)
                g.add_edge(nid, file_id, Edge.extracted(EdgeKind.DEFINES))

        matches = detect_patterns(g)
        fastapi_matches = [m for m in matches if m.pattern_id == "fastapi_routes"]
        assert len(fastapi_matches) >= 1

    def test_jest_detected(self):
        """Jest pattern matches when signature names and import patterns match."""
        g = Graph()
        g.add_node(
            Node.new(NodeKind.FUNCTION, "jest.test.spec.js::describe_block")
                .with_source("jest.test.spec.js", 0, 5)
        )
        g.add_node(
            Node.new(NodeKind.FUNCTION, "jest.test.spec.js::it_test_case")
                .with_source("jest.test.spec.js", 0, 5)
        )
        g.add_node(
            Node.new(NodeKind.FUNCTION, "jest.test.spec.js::expect_assertion")
                .with_source("jest.test.spec.js", 0, 5)
        )
        file_node = Node.new(NodeKind.FILE, "jest.test.spec.js")
        file_id = g.add_node(file_node)
        for nid, _ in g.nodes():
            if nid != file_id:
                g.add_edge(nid, file_id, Edge.extracted(EdgeKind.DEFINES))

        matches = detect_patterns(g)
        jest_matches = [m for m in matches if m.pattern_id == "jest_tests"]
        assert len(jest_matches) >= 1

    def test_no_false_positive_low_confidence(self):
        g = Graph()
        g.add_node(
            Node.new(NodeKind.FUNCTION, "app.py::useState")
                .with_source("app.py", 0, 1)
        )

        matches = detect_patterns(g)
        react_matches = [m for m in matches if m.pattern_id == "react_hooks"]
        assert len(react_matches) == 0


class TestMatchSinglePattern:
    def test_basic_match(self):
        g = Graph()
        func_id = g.add_node(
            Node.new(NodeKind.FUNCTION, "app.py::handle_request")
                .with_source("myapp.py", 0, 10)
        )
        g.add_node(
            Node.new(NodeKind.FILE, "app.py")
        )
        # Edge FROM function TO file (out_neighbors checked by matcher)
        file_id = g.find_by_qname("app.py")
        if file_id:
            g.add_edge(func_id, file_id, Edge.extracted(EdgeKind.DEFINES))

        pattern = FrameworkPattern(
            id="test_pattern",
            display_name="Test",
            description="Test",
            framework="test",
            category=PatternCategory.GENERIC,
            min_confidence=0.0,
            required_node_kinds=[NodeKind.FUNCTION],
            required_edge_kinds=[EdgeKind.DEFINES],
            signature_names=["handle"],
            import_patterns=["myapp"],
            min_nodes=1,
            max_nodes=10,
        )

        matches = _match_single_pattern(g, pattern)
        assert len(matches) == 1
        assert matches[0].pattern_id == "test_pattern"
        assert matches[0].confidence > 0

    def test_no_match_missing_signature(self):
        g = Graph()
        g.add_node(
            Node.new(NodeKind.FUNCTION, "app.py::something_else")
        )
        pattern = FrameworkPattern(
            id="test_no_match",
            display_name="Test",
            description="Test",
            framework="test",
            category=PatternCategory.GENERIC,
            min_confidence=0.0,
            required_node_kinds=[NodeKind.FUNCTION],
            signature_names=["unique_name_xyz"],
        )

        matches = _match_single_pattern(g, pattern)
        assert matches == []

    def test_no_match_missing_import(self):
        g = Graph()
        g.add_node(
            Node.new(NodeKind.FUNCTION, "app.py::useState")
                .with_source("app.py", 0, 1)
        )
        pattern = FrameworkPattern(
            id="test_import",
            display_name="Test",
            description="Test",
            framework="test",
            category=PatternCategory.GENERIC,
            min_confidence=0.0,
            required_node_kinds=[NodeKind.FUNCTION],
            signature_names=["useState"],
            import_patterns=["react"],
        )

        matches = _match_single_pattern(g, pattern)
        assert matches == []


class TestPatternMatch:
    def test_frozen(self):
        m = PatternMatch(
            pattern_id="test",
            display_name="Test",
            framework="test",
            category="generic",
            matched_node_ids=["1", "2"],
            matched_node_names=["foo", "bar"],
            matched_edge_kinds=["defines"],
            confidence=0.75,
            source_uris=["app.py"],
        )
        with pytest.raises(Exception):
            m.pattern_id = "other"


class TestCatalogManagement:
    def test_merged_catalog_includes_custom(self):
        custom = [
            FrameworkPattern(
                id="my_pattern",
                display_name="My Custom Pattern",
                description="Custom",
                framework="custom",
                category=PatternCategory.GENERIC,
            ),
        ]
        catalog = merged_catalog(custom)
        ids = [p.id for p in catalog]
        assert "my_pattern" in ids

    def test_merged_catalog_removes_duplicate_builtin(self):
        custom = [
            FrameworkPattern(
                id="react_hooks",
                display_name="My React Hooks",
                description="Overrides built-in",
                framework="react",
                category=PatternCategory.LIFECYCLE,
            ),
        ]
        catalog = merged_catalog(custom)
        ids = [p.id for p in catalog]
        assert ids.count("react_hooks") == 1
        idx = ids.index("react_hooks")
        assert catalog[idx].display_name == "My React Hooks"

    def test_load_patterns_from_file_raises_on_missing(self):
        with pytest.raises(FileNotFoundError):
            load_patterns_from_file("/nonexistent/path.toml")
