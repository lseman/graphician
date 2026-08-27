"""Tests for LSH dedup and pattern matching."""

from __future__ import annotations

from graphician.analysis.dedup.lsh import LshIndex, lsh_candidate_pairs
from graphician.analysis.dedup.minhash import MinHash, shingle
from graphician.analysis.dedup.types import DedupOptions
from graphician.analysis.patterns.matcher import detect_patterns
from graphician.analysis.patterns.types import (
    FrameworkPattern,
    PatternCategory,
    PatternMatch,
)
from graphician.core.edge import Edge, EdgeKind
from graphician.core.graph import Graph
from graphician.core.node import Node, NodeKind

# ---------------------------------------------------------------------------
# LSH index tests
# ---------------------------------------------------------------------------

class TestLshIndex:
    """Tests for LshIndex."""

    def test_add_and_get_candidates(self) -> None:
        sig_a = MinHash.from_iter(shingle("foo", 2), 10)
        sig_b = MinHash.from_iter(shingle("foo", 2), 10)
        sig_c = MinHash.from_iter(shingle("bar", 2), 10)

        index = LshIndex(num_bands=3, row_length=3)
        index.add(sig_a, "a")
        index.add(sig_b, "b")
        index.add(sig_c, "c")

        # a and b share bands → should be candidates for each other
        cands_for_a = index.get_candidates(sig_a)
        assert "b" in cands_for_a  # b shares band with a
        assert "c" not in cands_for_a  # c is too different

    def test_empty_index(self) -> None:
        index = LshIndex(num_bands=3, row_length=3)
        sig = MinHash.from_iter(shingle("x", 2), 10)
        assert index.get_candidates(sig) == set()

    def test_single_band_overlap(self) -> None:
        sig_a = MinHash.from_iter(shingle("test_func", 2), 10)
        sig_b = MinHash.from_iter(shingle("test_func", 2), 10)
        sig_c = MinHash.from_iter(shingle("xyz", 2), 10)

        index = LshIndex(num_bands=5, row_length=2)
        index.add(sig_a, "a")
        index.add(sig_c, "c")

        candidates = index.get_candidates(sig_b)
        assert "a" in candidates  # b is identical to a


class TestLshCandidatePairs:
    """Tests for lsh_candidate_pairs."""

    def test_same_name_pairs(self) -> None:
        nodes = [
            Node.new(NodeKind.FUNCTION, "foo"),
            Node.new(NodeKind.FUNCTION, "foo"),
            Node.new(NodeKind.FUNCTION, "bar"),
        ]
        from graphician.core.id import NodeId
        node_ids = [NodeId(i) for i in range(len(nodes))]

        options = DedupOptions(
            shingle_size=2, num_permutations=10,
            num_bands=3, row_length=3, jaccard_threshold=0.5,
        )
        pairs = lsh_candidate_pairs(nodes, node_ids, options)
        # The two "foo" nodes should be paired
        assert len(pairs) >= 1

    def test_different_names_no_pairs(self) -> None:
        nodes = [
            Node.new(NodeKind.FUNCTION, "xyz"),
            Node.new(NodeKind.FUNCTION, "abc"),
            Node.new(NodeKind.FUNCTION, "def"),
        ]
        from graphician.core.id import NodeId
        node_ids = [NodeId(i) for i in range(len(nodes))]

        options = DedupOptions(
            shingle_size=2, num_permutations=10,
            num_bands=3, row_length=3, jaccard_threshold=0.5,
        )
        pairs = lsh_candidate_pairs(nodes, node_ids, options)
        # Very different names should not pair
        assert len(pairs) == 0


# ---------------------------------------------------------------------------
# Pattern matcher tests
# ---------------------------------------------------------------------------

class TestDetectPatterns:
    """Tests for detect_patterns."""

    def test_no_patterns(self) -> None:
        graph = Graph()
        matches = detect_patterns(graph)
        assert isinstance(matches, list)

    def test_custom_pattern_matches(self) -> None:
        graph = Graph()
        graph.add_node(Node.new(NodeKind.FUNCTION, "process_request"))
        graph.add_node(Node.new(NodeKind.FUNCTION, "process_response"))

        custom_pattern = FrameworkPattern(
            id="test_pattern",
            display_name="Test Pattern",
            description="A test pattern",
            framework="test",
            category=PatternCategory.GENERIC,
            signature_names=["process"],
            min_nodes=2,
            max_nodes=10,
        )

        matches = detect_patterns(graph, [custom_pattern])
        assert len(matches) >= 1
        assert any(m.pattern_id == "test_pattern" for m in matches)

    def test_pattern_no_match_insufficient_nodes(self) -> None:
        graph = Graph()
        graph.add_node(Node.new(NodeKind.FUNCTION, "process_request"))

        custom_pattern = FrameworkPattern(
            id="req_pattern",
            display_name="Request Pattern",
            description="Requests pattern",
            framework="test",
            category=PatternCategory.GENERIC,
            signature_names=["request"],
            min_nodes=2,
            max_nodes=10,
        )

        matches = detect_patterns(graph, [custom_pattern])
        assert all(m.pattern_id != "req_pattern" for m in matches)

    def test_pattern_sorted_by_confidence(self) -> None:
        graph = Graph()
        for i in range(5):
            graph.add_node(Node.new(NodeKind.FUNCTION, f"process_{i}"))

        custom_pattern = FrameworkPattern(
            id="process_pattern",
            display_name="Process Pattern",
            description="Process pattern",
            framework="test",
            category=PatternCategory.GENERIC,
            signature_names=["process"],
            min_nodes=2,
            max_nodes=10,
        )

        matches = detect_patterns(graph, [custom_pattern])
        if len(matches) > 1:
            confidences = [m.confidence for m in matches]
            assert confidences == sorted(confidences, reverse=True)

    def test_pattern_with_node_kind_filter(self) -> None:
        graph = Graph()
        graph.add_node(Node.new(NodeKind.FUNCTION, "login_handler"))
        graph.add_node(Node.new(NodeKind.CLASS, "AuthController"))

        custom_pattern = FrameworkPattern(
            id="controller_pattern",
            display_name="Controller Pattern",
            description="Controllers",
            framework="test",
            category=PatternCategory.ROUTING,
            signature_names=["handler", "controller"],
            required_node_kinds=[NodeKind.CLASS],
            min_nodes=1,
            max_nodes=5,
        )

        matches = detect_patterns(graph, [custom_pattern])
        if matches:
            assert all(
                n["kind"] == "class"
                for m in matches
                for n in m.matched_nodes
            )

    def test_pattern_with_import_filter(self) -> None:
        graph = Graph()
        graph.add_node(
            Node.new(NodeKind.FUNCTION, "view_func").with_source("django/views.py", 0, 1)
        )
        graph.add_node(
            Node.new(NodeKind.FUNCTION, "other_func").with_source("flask/app.py", 0, 1)
        )

        custom_pattern = FrameworkPattern(
            id="django_pattern",
            display_name="Django Pattern",
            description="Django views",
            framework="django",
            category=PatternCategory.ROUTING,
            signature_names=["view"],
            import_patterns=["django"],
            min_nodes=1,
            max_nodes=5,
        )

        matches = detect_patterns(graph, [custom_pattern])
        if matches:
            assert all(
                "django" in n.get("source_uri", "")
                for m in matches
                for n in m.matched_nodes
            )

    def test_pattern_with_required_edges(self) -> None:
        graph = Graph()
        graph.add_node(Node.new(NodeKind.FUNCTION, "process"))
        graph.add_node(Node.new(NodeKind.FUNCTION, "validate"))

        # Use node IDs directly for edges
        nid1 = graph.find_by_qname("process")
        nid2 = graph.find_by_qname("validate")
        if nid1 and nid2:
            graph.add_edge(nid1, nid2, Edge.ambiguous(EdgeKind.CALLS))

        custom_pattern = FrameworkPattern(
            id="pipeline_pattern",
            display_name="Pipeline Pattern",
            description="Pipeline",
            framework="test",
            category=PatternCategory.GENERIC,
            signature_names=["process", "validate"],
            min_nodes=2,
            max_nodes=5,
            required_edge_kinds=[EdgeKind.CALLS],
        )

        matches = detect_patterns(graph, [custom_pattern])
        assert any(m.pattern_id == "pipeline_pattern" for m in matches)

    def test_pattern_max_nodes_exceeded(self) -> None:
        graph = Graph()
        for i in range(20):
            graph.add_node(Node.new(NodeKind.FUNCTION, f"process_{i}"))

        custom_pattern = FrameworkPattern(
            id="limited_pattern",
            display_name="Limited Pattern",
            description="Limited",
            framework="test",
            category=PatternCategory.GENERIC,
            signature_names=["process"],
            min_nodes=1,
            max_nodes=5,
        )

        matches = detect_patterns(graph, [custom_pattern])
        assert all(m.pattern_id != "limited_pattern" for m in matches)

    def test_pattern_min_nodes_not_met(self) -> None:
        graph = Graph()
        for i in range(2):
            graph.add_node(Node.new(NodeKind.FUNCTION, f"process_{i}"))

        custom_pattern = FrameworkPattern(
            id="high_min_pattern",
            display_name="High Min Pattern",
            description="High min",
            framework="test",
            category=PatternCategory.GENERIC,
            signature_names=["process"],
            min_nodes=10,
            max_nodes=50,
        )

        matches = detect_patterns(graph, [custom_pattern])
        assert all(m.pattern_id != "high_min_pattern" for m in matches)


class TestPatternMatch:
    """Tests for PatternMatch data structure."""

    def test_pattern_match_creation(self) -> None:
        match = PatternMatch(
            pattern_id="test",
            display_name="Test",
            framework="test",
            category=PatternCategory.GENERIC.value,
            confidence=0.8,
            matched_nodes=[{"id": 1, "name": "foo"}],
            matched_edges=[],
        )
        assert match.pattern_id == "test"
        assert match.confidence == 0.8
        assert len(match.matched_nodes) == 1


class TestConfidenceCalculation:
    """Tests for confidence calculation in pattern matching."""

    def test_confidence_bounds(self) -> None:
        graph = Graph()
        for i in range(10):
            graph.add_node(Node.new(NodeKind.FUNCTION, f"handler_{i}"))

        custom_pattern = FrameworkPattern(
            id="handler_pattern",
            display_name="Handler Pattern",
            description="Handlers",
            framework="test",
            category=PatternCategory.ROUTING,
            signature_names=["handler"],
            min_nodes=1,
            max_nodes=20,
        )

        matches = detect_patterns(graph, [custom_pattern])
        assert matches
        match = matches[0]
        assert 0 < match.confidence <= 1.0

    def test_confidence_with_edges(self) -> None:
        graph = Graph()
        graph.add_node(Node.new(NodeKind.FUNCTION, "handler_1"))
        graph.add_node(Node.new(NodeKind.FUNCTION, "handler_2"))

        nid1 = graph.find_by_qname("handler_1")
        nid2 = graph.find_by_qname("handler_2")
        if nid1 and nid2:
            graph.add_edge(nid1, nid2, Edge.ambiguous(EdgeKind.CALLS))

        custom_pattern = FrameworkPattern(
            id="edge_pattern",
            display_name="Edge Pattern",
            description="With edges",
            framework="test",
            category=PatternCategory.ROUTING,
            signature_names=["handler"],
            min_nodes=1,
            max_nodes=10,
            required_edge_kinds=[EdgeKind.CALLS],
        )

        matches = detect_patterns(graph, [custom_pattern])
        assert matches
        assert matches[0].confidence > 0.5

    def test_confidence_no_edges_fails_required_edges(self) -> None:
        graph = Graph()
        graph.add_node(Node.new(NodeKind.FUNCTION, "handler_1"))
        graph.add_node(Node.new(NodeKind.FUNCTION, "handler_2"))

        custom_pattern = FrameworkPattern(
            id="no_edges_pattern",
            display_name="No Edges",
            description="No edges",
            framework="test",
            category=PatternCategory.ROUTING,
            signature_names=["handler"],
            min_nodes=2,
            max_nodes=5,
            required_edge_kinds=[EdgeKind.CALLS],
        )

        matches = detect_patterns(graph, [custom_pattern])
        # Should not match because there are no edges between candidates
        assert all(m.pattern_id != "no_edges_pattern" for m in matches)
