"""Tests for community quality metrics, split logic, context packing, and graph diff."""

from __future__ import annotations

from pathlib import Path

import pytest

from graphician.core.edge import Edge, EdgeKind
from graphician.core.graph import Graph
from graphician.core.id import NodeId
from graphician.core.node import Node, NodeKind


# ── community quality ────────────────────────────────────────────────


class TestCommunityCohesion:
    """Tests for community_cohesion."""

    def test_empty_communities(self) -> None:
        from graphician.analysis.communities.quality import community_cohesion

        graph = Graph()
        assert community_cohesion(graph, {}) == {}

    def test_single_node_community(self) -> None:
        from graphician.analysis.communities.quality import community_cohesion

        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "app::foo"))
        communities = {n1.value: 1}
        result = community_cohesion(graph, communities)
        assert result == {1: 1.0}

    def test_two_connected_nodes_same_community(self) -> None:
        from graphician.analysis.communities.quality import community_cohesion

        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "app::foo"))
        n2 = graph.add_node(Node.new(NodeKind.FUNCTION, "app::bar"))
        graph.add_edge(n1, n2, Edge.extracted(EdgeKind.CALLS))
        communities = {n1.value: 1, n2.value: 1}
        result = community_cohesion(graph, communities)
        # n=2, possible=1, actual=1 -> cohesion=1.0
        assert result == {1: 1.0}

    def test_two_connected_nodes_diff_communities(self) -> None:
        from graphician.analysis.communities.quality import community_cohesion

        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "app::foo"))
        n2 = graph.add_node(Node.new(NodeKind.FUNCTION, "app::bar"))
        graph.add_edge(n1, n2, Edge.extracted(EdgeKind.CALLS))
        communities = {n1.value: 1, n2.value: 2}
        result = community_cohesion(graph, communities)
        assert result == {1: 1.0, 2: 1.0}

    def test_three_nodes_triangle(self) -> None:
        from graphician.analysis.communities.quality import community_cohesion

        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "app::a"))
        n2 = graph.add_node(Node.new(NodeKind.FUNCTION, "app::b"))
        n3 = graph.add_node(Node.new(NodeKind.FUNCTION, "app::c"))
        graph.add_edge(n1, n2, Edge.extracted(EdgeKind.CALLS))
        graph.add_edge(n2, n3, Edge.extracted(EdgeKind.CALLS))
        graph.add_edge(n3, n1, Edge.extracted(EdgeKind.CALLS))
        communities = {n1.value: 1, n2.value: 1, n3.value: 1}
        result = community_cohesion(graph, communities)
        # n=3, possible=3, actual=3 -> cohesion=1.0
        assert result == {1: 1.0}

    def test_three_nodes_sparse(self) -> None:
        from graphician.analysis.communities.quality import community_cohesion

        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "app::a"))
        n2 = graph.add_node(Node.new(NodeKind.FUNCTION, "app::b"))
        n3 = graph.add_node(Node.new(NodeKind.FUNCTION, "app::c"))
        graph.add_edge(n1, n2, Edge.extracted(EdgeKind.CALLS))
        # n3 is isolated in the community
        communities = {n1.value: 1, n2.value: 1, n3.value: 1}
        result = community_cohesion(graph, communities)
        # n=3, possible=3, actual=1 -> cohesion=1/3
        assert abs(result[1] - 1.0 / 3.0) < 0.01


class TestCommunityQuality:
    """Tests for community_quality."""

    def test_empty(self) -> None:
        from graphician.analysis.communities.quality import community_quality

        graph = Graph()
        result = community_quality(graph, {})
        assert result.community_count == 0
        assert result.score == 0.0
        assert result.singleton_count == 0

    def test_single_community(self) -> None:
        from graphician.analysis.communities.quality import community_quality

        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "app::a"))
        n2 = graph.add_node(Node.new(NodeKind.FUNCTION, "app::b"))
        graph.add_edge(n1, n2, Edge.extracted(EdgeKind.CALLS))
        communities = {n1.value: 1, n2.value: 1}
        result = community_quality(graph, communities)
        assert result.community_count == 1
        assert result.min_size == 2
        assert result.max_size == 2
        assert result.mean_size == 2.0
        assert result.singleton_count == 0

    def test_mixed_singletons(self) -> None:
        from graphician.analysis.communities.quality import community_quality

        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "app::a"))
        n2 = graph.add_node(Node.new(NodeKind.FUNCTION, "app::b"))
        n3 = graph.add_node(Node.new(NodeKind.FUNCTION, "app::c"))
        communities = {n1.value: 1, n2.value: 2, n3.value: 3}
        result = community_quality(graph, communities)
        assert result.community_count == 3
        assert result.singleton_count == 3
        assert result.min_size == 1
        assert result.max_size == 1

    def test_resolution_scaling(self) -> None:
        from graphician.analysis.communities.quality import community_quality

        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "app::a"))
        n2 = graph.add_node(Node.new(NodeKind.FUNCTION, "app::b"))
        communities = {n1.value: 1, n2.value: 2}
        r1 = community_quality(graph, communities, resolution=0.5)
        r2 = community_quality(graph, communities, resolution=2.0)
        assert r2.score >= r1.score


# ── split oversized ─────────────────────────────────────────────────


class TestSplitOversized:
    """Tests for split_oversized."""

    def test_split_empty_graph(self) -> None:
        from graphician.analysis.communities.split import split_oversized

        graph = Graph()
        result = split_oversized(graph, threshold_pct=0.5)
        assert result["operation"] == "split_oversized"
        assert result["final_count"] == 0

    def test_split_small_graph(self) -> None:
        from graphician.analysis.communities.split import split_oversized

        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "app::a"))
        n2 = graph.add_node(Node.new(NodeKind.FUNCTION, "app::b"))
        result = split_oversized(graph)
        assert result["operation"] == "split_oversized"
        assert result["final_count"] >= 0

    def test_split_returns_expected_keys(self) -> None:
        from graphician.analysis.communities.split import split_oversized

        graph = Graph()
        for i in range(10):
            graph.add_node(Node.new(NodeKind.FUNCTION, f"app::func_{i}"))
        result = split_oversized(graph)
        assert "operation" in result
        assert "threshold" in result
        assert "original_count" in result
        assert "split_count" in result
        assert "final_count" in result
        assert "splits" in result


# ── context pack ─────────────────────────────────────────────────────


class TestContextPack:
    """Tests for build_context_pack and helpers."""

    def test_context_pack_empty_graph(self) -> None:
        from graphician.analysis.context_pack import build_context_pack

        graph = Graph()
        result = build_context_pack(graph, "test_query")
        assert "context_pack" in result
        assert result["note"] == "No relevant symbols found"

    def test_context_pack_with_nodes(self) -> None:
        from graphician.analysis.context_pack import build_context_pack

        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "app::search").with_source("/tmp/app.py", 1, 1))
        graph.add_edge(n1, n1, Edge.extracted(EdgeKind.CALLS))
        result = build_context_pack(graph, "search")
        assert "context_pack" in result
        assert "total_tokens" in result
        assert "budget" in result

    def test_context_pack_exceeds_budget(self) -> None:
        from graphician.analysis.context_pack import build_context_pack

        graph = Graph()
        n1 = graph.add_node(Node.new(NodeKind.FUNCTION, "app::search"))
        result = build_context_pack(graph, "search", token_budget=1)
        assert "context_pack" in result

    def test_context_pack_max_items(self) -> None:
        from graphician.analysis.context_pack import build_context_pack

        graph = Graph()
        for i in range(20):
            graph.add_node(Node.new(NodeKind.FUNCTION, f"app::func_{i}"))
        result = build_context_pack(graph, "func", max_items=5)
        assert result["total_items"] <= 5


class TestContextHelpers:
    """Tests for _count_tokens, _compute_relevance, _compute_diversity_penalty."""

    def test_count_tokens_empty(self) -> None:
        from graphician.analysis.context_pack import _count_tokens
        assert _count_tokens("") == 1  # minimum 1

    def test_count_tokens_short(self) -> None:
        from graphician.analysis.context_pack import _count_tokens
        assert _count_tokens("hello") == 1  # 5//4 = 1

    def test_count_tokens_long(self) -> None:
        from graphician.analysis.context_pack import _count_tokens
        assert _count_tokens("x" * 100) == 25  # 100//4 = 25

    def test_compute_relevance_name_match(self) -> None:
        from graphician.analysis.context_pack import ContextItem, _compute_relevance
        item = ContextItem(
            qualified_name="app::other", kind="function", name="search",
            source_uri=None, content="",
        )
        score = _compute_relevance(item, "search")
        assert score == 3.0

    def test_compute_relevance_qname_match(self) -> None:
        from graphician.analysis.context_pack import ContextItem, _compute_relevance
        item = ContextItem(
            qualified_name="app::search_module", kind="function", name="x",
            source_uri=None, content="",
        )
        score = _compute_relevance(item, "search")
        assert score == 2.0

    def test_compute_relevance_content_match(self) -> None:
        from graphician.analysis.context_pack import ContextItem, _compute_relevance
        item = ContextItem(
            qualified_name="app::other", kind="function", name="other",
            source_uri=None, content="this implements search logic",
        )
        score = _compute_relevance(item, "search")
        assert score == 1.0

    def test_compute_relevance_no_match(self) -> None:
        from graphician.analysis.context_pack import ContextItem, _compute_relevance
        item = ContextItem(
            qualified_name="app::other", kind="function", name="other",
            source_uri=None, content="nothing related",
        )
        assert _compute_relevance(item, "search") == 0.0

    def test_compute_diversity_penalty_no_penalty(self) -> None:
        from graphician.analysis.context_pack import ContextItem, _compute_diversity_penalty
        item = ContextItem(
            qualified_name="app::a", kind="function", name="a",
            source_uri="/tmp/a.py", content="",
        )
        assert _compute_diversity_penalty(item, set(), set()) == 1.0

    def test_compute_diversity_penalty_file_duplicate(self) -> None:
        from graphician.analysis.context_pack import ContextItem, _compute_diversity_penalty
        item = ContextItem(
            qualified_name="app::b", kind="function", name="b",
            source_uri="/tmp/a.py", content="",
        )
        penalty = _compute_diversity_penalty(item, {"/tmp/a.py"}, set())
        assert penalty == 0.5

    def test_compute_diversity_penalty_kind_duplicate(self) -> None:
        from graphician.analysis.context_pack import ContextItem, _compute_diversity_penalty
        item = ContextItem(
            qualified_name="app::b", kind="function", name="b",
            source_uri="/tmp/b.py", content="",
        )
        penalty = _compute_diversity_penalty(item, set(), {"function"})
        assert penalty == 0.8

    def test_compute_diversity_penalty_both_duplicates(self) -> None:
        from graphician.analysis.context_pack import ContextItem, _compute_diversity_penalty
        item = ContextItem(
            qualified_name="app::b", kind="function", name="b",
            source_uri="/tmp/a.py", content="",
        )
        penalty = _compute_diversity_penalty(item, {"/tmp/a.py"}, {"function"})
        assert penalty == 0.4  # 0.5 * 0.8


class TestContextItem:
    """Tests for ContextItem dataclass."""

    def test_context_item_defaults(self) -> None:
        from graphician.analysis.context_pack import ContextItem

        item = ContextItem(
            qualified_name="app::foo", kind="function", name="foo",
            source_uri="/tmp/foo.py", content="def foo(): pass",
        )
        assert item.evidence == []
        assert item.token_count == 0


# ── graph diff ───────────────────────────────────────────────────────


class TestGraphDiff:
    """Tests for graph_diff."""

    def test_diff_empty_graphs(self) -> None:
        from graphician.analysis.diff import graph_diff

        base = Graph()
        head = Graph()
        result = graph_diff(base, head)
        assert result["added_nodes"] == []
        assert result["removed_nodes"] == []
        assert result["modified_nodes"] == []
        assert result["added_edges"] == []
        assert result["removed_edges"] == []
        assert result["community_changes"] == []

    def test_diff_added_node(self) -> None:
        from graphician.analysis.diff import graph_diff

        base = Graph()
        base.add_node(Node.new(NodeKind.FUNCTION, "app::old"))

        head = Graph()
        head.add_node(Node.new(NodeKind.FUNCTION, "app::old"))
        head.add_node(Node.new(NodeKind.FUNCTION, "app::new"))

        result = graph_diff(base, head)
        assert len(result["added_nodes"]) == 1
        assert result["added_nodes"][0]["qualified_name"] == "app::new"
        assert len(result["removed_nodes"]) == 0

    def test_diff_removed_node(self) -> None:
        from graphician.analysis.diff import graph_diff

        base = Graph()
        base.add_node(Node.new(NodeKind.FUNCTION, "app::old"))
        base.add_node(Node.new(NodeKind.FUNCTION, "app::gone"))

        head = Graph()
        head.add_node(Node.new(NodeKind.FUNCTION, "app::old"))

        result = graph_diff(base, head)
        assert len(result["removed_nodes"]) == 1
        assert result["removed_nodes"][0]["qualified_name"] == "app::gone"

    def test_diff_modified_node(self) -> None:
        from graphician.analysis.diff import graph_diff

        base = Graph()
        b = base.add_node(Node.new(NodeKind.FUNCTION, "app::foo").with_source("/tmp/app.py", 1, 1))

        head = Graph()
        h = head.add_node(Node.new(NodeKind.FUNCTION, "app::foo").with_source("/tmp/app.py", 5, 10))

        result = graph_diff(base, head)
        assert len(result["modified_nodes"]) == 1
        assert result["modified_nodes"][0]["qualified_name"] == "app::foo"

    def test_diff_unchanged_node(self) -> None:
        from graphician.analysis.diff import graph_diff

        base = Graph()
        base.add_node(Node.new(NodeKind.FUNCTION, "app::foo").with_source("/tmp/app.py", 1, 1))

        head = Graph()
        head.add_node(Node.new(NodeKind.FUNCTION, "app::foo").with_source("/tmp/app.py", 1, 1))

        result = graph_diff(base, head)
        assert result["modified_nodes"] == []

    def test_diff_added_edge(self) -> None:
        from graphician.analysis.diff import graph_diff

        base = Graph()
        b1 = base.add_node(Node.new(NodeKind.FUNCTION, "app::a"))
        b2 = base.add_node(Node.new(NodeKind.FUNCTION, "app::b"))

        head = Graph()
        h1 = head.add_node(Node.new(NodeKind.FUNCTION, "app::a"))
        h2 = head.add_node(Node.new(NodeKind.FUNCTION, "app::b"))
        head.add_edge(h1, h2, Edge.extracted(EdgeKind.CALLS))

        result = graph_diff(base, head)
        assert len(result["added_edges"]) == 1

    def test_diff_removed_edge(self) -> None:
        from graphician.analysis.diff import graph_diff

        base = Graph()
        b1 = base.add_node(Node.new(NodeKind.FUNCTION, "app::a"))
        b2 = base.add_node(Node.new(NodeKind.FUNCTION, "app::b"))
        base.add_edge(b1, b2, Edge.extracted(EdgeKind.CALLS))

        head = Graph()
        head.add_node(Node.new(NodeKind.FUNCTION, "app::a"))
        head.add_node(Node.new(NodeKind.FUNCTION, "app::b"))

        result = graph_diff(base, head)
        assert len(result["removed_edges"]) == 1

    def test_diff_unchanged_edges(self) -> None:
        from graphician.analysis.diff import graph_diff

        base = Graph()
        b1 = base.add_node(Node.new(NodeKind.FUNCTION, "app::a"))
        b2 = base.add_node(Node.new(NodeKind.FUNCTION, "app::b"))
        base.add_edge(b1, b2, Edge.extracted(EdgeKind.CALLS))

        head = Graph()
        h1 = head.add_node(Node.new(NodeKind.FUNCTION, "app::a"))
        h2 = head.add_node(Node.new(NodeKind.FUNCTION, "app::b"))
        head.add_edge(h1, h2, Edge.extracted(EdgeKind.CALLS))

        result = graph_diff(base, head)
        assert result["added_edges"] == []
        assert result["removed_edges"] == []

    def test_diff_community_changes(self) -> None:
        from graphician.analysis.diff import graph_diff

        base = Graph()
        b1 = base.add_node(Node.new(NodeKind.FUNCTION, "app::foo"))
        base._nodes[b1.value] = base.node(b1).with_property("community", "c1")

        head = Graph()
        h1 = head.add_node(Node.new(NodeKind.FUNCTION, "app::foo"))
        head._nodes[h1.value] = head.node(h1).with_property("community", "c2")

        result = graph_diff(base, head)
        assert len(result["community_changes"]) == 1
        assert result["community_changes"][0]["old_community"] == "c1"
        assert result["community_changes"][0]["new_community"] == "c2"

    def test_diff_no_community_changes(self) -> None:
        from graphician.analysis.diff import graph_diff

        base = Graph()
        b1 = base.add_node(Node.new(NodeKind.FUNCTION, "app::foo"))
        base._nodes[b1.value] = base.node(b1).with_property("community", "c1")

        head = Graph()
        h1 = head.add_node(Node.new(NodeKind.FUNCTION, "app::foo"))
        head._nodes[h1.value] = head.node(h1).with_property("community", "c1")

        result = graph_diff(base, head)
        assert result["community_changes"] == []


# ── diff dataclasses ─────────────────────────────────────────────────


class TestDiffDataClasses:
    """Tests for DiffNode, DiffEdge, CommunityChange, GraphDiff."""

    def test_diff_node_defaults(self) -> None:
        from graphician.analysis.diff import DiffNode

        dn = DiffNode(id=1, qualified_name="app::foo", kind="function")
        assert dn.source is None

    def test_diff_edge_defaults(self) -> None:
        from graphician.analysis.diff import DiffEdge

        de = DiffEdge(id=1, src=2, dst=3, kind="CALLS")
        assert de.id == 1

    def test_community_change_defaults(self) -> None:
        from graphician.analysis.diff import CommunityChange

        cc = CommunityChange(node_id=1, qualified_name="app::foo",
                             old_community="c1", new_community="c2")
        assert cc.node_id == 1

    def test_graph_diff_defaults(self) -> None:
        from graphician.analysis.diff import GraphDiff

        gd = GraphDiff()
        assert gd.added_nodes == []
        assert gd.removed_nodes == []
        assert gd.modified_nodes == []
        assert gd.added_edges == []
        assert gd.removed_edges == []
        assert gd.community_changes == []
