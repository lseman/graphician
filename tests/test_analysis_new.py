"""Tests for newly implemented analysis modules.

Covers: paths, impact, search, centrality — all aligned with the Rust
reference implementation at ariadne-graph/src/analysis/.
"""

from graphician.analysis.centrality import (
    is_rank_noise,
    pagerank,
    personalized_pagerank,
)
from graphician.analysis.impact import ImpactQuery, find_impact
from graphician.analysis.paths import (
    PathQuery,
    WeightedPath,
    callees_of,
    callers_of,
    find_paths,
    find_top_paths,
    max_depth_from,
)
from graphician.analysis.search import (
    SearchIntent,
    fts_ranked_search,
    ranked_search,
    search_by_name,
    task_aware_search,
)
from graphician.core.edge import Confidence, Edge, EdgeKind
from graphician.core.graph import Graph
from graphician.core.id import NodeId
from graphician.core.node import Node, NodeKind

# ── Helpers ──────────────────────────────────────────────────────────

def _make_graph() -> Graph:
    """Build a small test graph: a -> b -> c, plus b -> d."""
    g = Graph()
    a = g.add_node(Node.new(NodeKind.FUNCTION, "main"))
    b = g.add_node(Node.new(NodeKind.FUNCTION, "process"))
    c = g.add_node(Node.new(NodeKind.FUNCTION, "save"))
    d = g.add_node(Node.new(NodeKind.CLASS, "Database"))
    g.add_edge(a, b, Edge.extracted(EdgeKind.CALLS))
    g.add_edge(b, c, Edge.extracted(EdgeKind.CALLS))
    g.add_edge(b, d, Edge.extracted(EdgeKind.CALLS))
    return g


# ── Paths ────────────────────────────────────────────────────────────

class TestCalleesOf:
    def test_returns_all_call_targets(self):
        g = _make_graph()
        callees = callees_of(g, NodeId(0))
        assert len(callees) == 3

    def test_empty_for_leaf(self):
        g = _make_graph()
        callees = callees_of(g, NodeId(2))
        assert callees == []


class TestCallersOf:
    def test_returns_all_callers(self):
        g = _make_graph()
        callers = callers_of(g, NodeId(2))
        assert len(callers) == 2

    def test_empty_for_root(self):
        g = _make_graph()
        callers = callers_of(g, NodeId(0))
        assert callers == []


class TestMaxDepthFrom:
    def test_linear_chain(self):
        g = _make_graph()
        assert max_depth_from(g, NodeId(0)) == 2

    def test_leaf_is_zero(self):
        g = _make_graph()
        assert max_depth_from(g, NodeId(2)) == 0


class TestFindPaths:
    def test_returns_all_paths(self):
        g = _make_graph()
        result = find_paths(g, PathQuery(from_id=NodeId(0), max_hops=5))
        assert len(result) >= 3

    def test_with_target(self):
        g = _make_graph()
        result = find_paths(g, PathQuery(
            from_id=NodeId(0),
            to_id=NodeId(2),
            max_hops=5,
        ))
        assert len(result) >= 1
        # Every path should end at node 2
        for path in result:
            assert path[-1].value == 2


class TestFindTopPaths:
    def test_returns_weighted_paths(self):
        g = _make_graph()
        paths = find_top_paths(
            g,
            PathQuery(from_id=NodeId(0), max_hops=5),
            limit=5,
        )
        assert len(paths) >= 1
        assert isinstance(paths[0], WeightedPath)
        assert paths[0].cost > 0

    def test_uses_cost_priority_before_early_limit(self):
        g = Graph()
        start = g.add_node(Node.new(NodeKind.FUNCTION, "start"))
        target = g.add_node(Node.new(NodeKind.FUNCTION, "target"))

        # A FIFO worklist fills the internal candidate limit with these
        # expensive routes before it discovers the cheaper two-hop route.
        for index in range(3):
            direct = g.add_node(Node.new(NodeKind.FUNCTION, f"direct_{index}"))
            g.add_edge(start, direct, Edge.extracted(EdgeKind.SIMILAR_TO))
            g.add_edge(direct, target, Edge.extracted(EdgeKind.SIMILAR_TO))

        helper = g.add_node(Node.new(NodeKind.FUNCTION, "helper"))
        g.add_edge(start, helper, Edge.extracted(EdgeKind.DEFINES))
        g.add_edge(helper, target, Edge.extracted(EdgeKind.DEFINES))

        paths = find_top_paths(
            g,
            PathQuery(from_id=start, to_id=target, max_hops=2),
            limit=1,
        )

        assert len(paths) == 1
        assert paths[0].nodes == [start, helper, target]
        assert paths[0].cost == 0.7


# ── Impact ───────────────────────────────────────────────────────────

class TestFindImpact:
    def test_returns_reverse_reachables(self):
        g = _make_graph()
        hits = find_impact(g, ImpactQuery(seed_id=NodeId(2), max_hops=5, limit=10))
        assert len(hits) >= 2
        # Nodes should be sorted by score descending
        for i in range(len(hits) - 1):
            assert hits[i].score >= hits[i + 1].score

    def test_ignores_ambiguous_edges(self):
        g = Graph()
        a = g.add_node(Node.new(NodeKind.FUNCTION, "a"))
        b = g.add_node(Node.new(NodeKind.FUNCTION, "b"))
        g.add_edge(a, b, Edge.extracted(EdgeKind.CALLS))
        g.add_edge(a, b, Edge(EdgeKind.CALLS, Confidence.AMBIGUOUS))
        hits = find_impact(g, ImpactQuery(seed_id=b, max_hops=5, limit=10))
        assert len(hits) == 1  # Only 'a' should be a hit

    def test_respects_max_hops(self):
        g = Graph()
        a = g.add_node(Node.new(NodeKind.FUNCTION, "a"))
        b = g.add_node(Node.new(NodeKind.FUNCTION, "b"))
        c = g.add_node(Node.new(NodeKind.FUNCTION, "c"))
        g.add_edge(a, b, Edge.extracted(EdgeKind.CALLS))
        g.add_edge(b, c, Edge.extracted(EdgeKind.CALLS))
        hits = find_impact(g, ImpactQuery(seed_id=c, max_hops=1, limit=10))
        assert len(hits) == 1  # Only b (dist=1), not a (dist=2)

    def test_includes_direct_dependencies_below_reverse_dependants(self):
        g = Graph()
        caller = g.add_node(Node.new(NodeKind.FUNCTION, "caller"))
        seed = g.add_node(Node.new(NodeKind.FUNCTION, "seed"))
        dependency = g.add_node(Node.new(NodeKind.FUNCTION, "dependency"))
        g.add_edge(caller, seed, Edge.extracted(EdgeKind.CALLS))
        g.add_edge(seed, dependency, Edge.extracted(EdgeKind.CALLS))

        hits = find_impact(g, ImpactQuery(seed_id=seed, max_hops=1, limit=10))

        assert [hit.id for hit in hits] == [caller, dependency]
        assert hits[0].score > hits[1].score
        assert hits[0].via == [EdgeKind.CALLS]
        assert hits[1].via == [EdgeKind.CALLS]


# ── Search ───────────────────────────────────────────────────────────

class TestRankedSearch:
    def test_exact_match(self):
        g = _make_graph()
        hits = ranked_search(g, "main", limit=10)
        assert len(hits) >= 1
        assert hits[0].node.name == "main"

    def test_substring_match(self):
        g = _make_graph()
        hits = ranked_search(g, "process", limit=10)
        assert len(hits) >= 1
        assert hits[0].node.name == "process"

    def test_no_match(self):
        g = _make_graph()
        hits = ranked_search(g, "nonexistent_xyz", limit=10)
        assert hits == []


class TestSearchByName:
    def test_exact(self):
        g = _make_graph()
        hits = search_by_name(g, "main", exact=True)
        assert len(hits) == 1

    def test_substring(self):
        g = _make_graph()
        hits = search_by_name(g, "pro", exact=False)
        assert len(hits) >= 1


class TestTaskAwareSearch:
    def test_classifies_and_ranks(self):
        g = _make_graph()
        hits = task_aware_search(
            g, "main", intent=SearchIntent.IMPACT, limit=10
        )
        assert len(hits) >= 1


class TestSearchIntent:
    def test_classify_debug(self):
        assert SearchIntent.classify("fix the bug in login") == SearchIntent.DEBUG

    def test_classify_lookup(self):
        assert SearchIntent.classify("find the auth function") == SearchIntent.LOOKUP

    def test_classify_default(self):
        assert SearchIntent.classify("hello world") == SearchIntent.LOOKUP


# ── Centrality ───────────────────────────────────────────────────────

class TestPageRank:
    def test_concentrates_on_sinks(self):
        g = Graph()
        a = g.add_node(Node.new(NodeKind.FUNCTION, "a"))
        b = g.add_node(Node.new(NodeKind.FUNCTION, "b"))
        c = g.add_node(Node.new(NodeKind.FUNCTION, "c"))
        g.add_edge(a, c, Edge.extracted(EdgeKind.CALLS))
        g.add_edge(b, c, Edge.extracted(EdgeKind.CALLS))
        ranks = pagerank(g, 0.85, 30)
        assert ranks[c] > ranks[a]
        assert ranks[c] > ranks[b]

    def test_personalized_biases_toward_seed(self):
        g = Graph()
        a = g.add_node(Node.new(NodeKind.FUNCTION, "a"))
        b = g.add_node(Node.new(NodeKind.FUNCTION, "b"))
        c = g.add_node(Node.new(NodeKind.FUNCTION, "c"))
        g.add_edge(a, b, Edge.extracted(EdgeKind.CALLS))
        g.add_edge(c, b, Edge.extracted(EdgeKind.CALLS))
        ranks = personalized_pagerank(g, [(a, 1.0)], 0.85, 30)
        assert ranks[a] > ranks[c]


class TestIsRankNoise:
    def test_filters_file(self):
        assert is_rank_noise(Node.new(NodeKind.FILE, "lib.rs"))

    def test_filters_flow(self):
        assert is_rank_noise(Node.new(NodeKind.FLOW, "flow::main"))

    def test_filters_placeholder(self):
        assert is_rank_noise(
            Node.new(NodeKind.FUNCTION, "call::external")
        )

    def test_does_not_filter_function(self):
        assert not is_rank_noise(
            Node.new(NodeKind.FUNCTION, "src::login")
        )


class TestFtsRankedSearch:
    def test_fallback_to_ranked_search(self):
        g = _make_graph()
        hits = fts_ranked_search(g, "process", limit=10)
        assert len(hits) >= 1
