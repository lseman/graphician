from __future__ import annotations

import pytest

from ariadne_py.analysis.communities.core import CommunityOptions, WorkingGraph
from ariadne_py.analysis.communities.infomap import (
    LcgRng,
    _compute_lmdl,
    _entropy_term,
    _labels_for_original,
    _random_walk_init,
    detect_communities,
    infomap,
    infomap_with_options,
)
from ariadne_py.core import Edge, EdgeKind, Graph, Node, NodeKind


def _add_nodes(graph: Graph, count: int):
    return [graph.add_node(Node.new(NodeKind.FUNCTION, f"n{i}")) for i in range(count)]


def _connect(graph: Graph, left, right, *, both: bool = True) -> None:
    graph.add_edge(left, right, Edge.extracted(EdgeKind.CALLS))
    if both:
        graph.add_edge(right, left, Edge.extracted(EdgeKind.CALLS))


@pytest.mark.parametrize("bidirectional", [False, True])
def test_infomap_splits_disconnected_pairs(bidirectional: bool) -> None:
    graph = Graph()
    a, b, c, d = _add_nodes(graph, 4)
    _connect(graph, a, b, both=bidirectional)
    _connect(graph, c, d, both=bidirectional)

    communities = infomap(graph)

    assert communities[a] == communities[b]
    assert communities[c] == communities[d]
    assert communities[a] != communities[c]


def test_infomap_separates_dense_triangles_joined_by_bridge() -> None:
    graph = Graph()
    nodes = _add_nodes(graph, 6)
    for chunk in (nodes[:3], nodes[3:]):
        for i, left in enumerate(chunk):
            for right in chunk[i + 1 :]:
                _connect(graph, left, right)
    _connect(graph, nodes[2], nodes[3], both=False)

    communities = infomap(graph)

    assert len({communities[node] for node in nodes[:3]}) == 1
    assert len({communities[node] for node in nodes[3:]}) == 1
    assert communities[nodes[0]] != communities[nodes[3]]


def test_infomap_handles_empty_edgeless_and_ambiguous_placeholder_graphs() -> None:
    assert infomap(Graph()) == {}

    graph = Graph()
    a, b, c, d, placeholder = _add_nodes(graph, 5)
    _connect(graph, a, b)
    _connect(graph, c, d)
    graph.add_edge(a, placeholder, Edge.ambiguous(EdgeKind.CALLS))
    graph.add_edge(c, placeholder, Edge.ambiguous(EdgeKind.CALLS))
    communities = infomap(graph)
    assert communities[a] != communities[c]

    edgeless = Graph()
    isolated = _add_nodes(edgeless, 3)
    labels = infomap(edgeless)
    assert len({labels[node] for node in isolated}) == 3


def test_infomap_is_deterministic_across_option_variants() -> None:
    graph = Graph()
    nodes = _add_nodes(graph, 12)
    for chunk in (nodes[:6], nodes[6:]):
        for i, left in enumerate(chunk):
            for right in chunk[i + 1 :]:
                _connect(graph, left, right)
    _connect(graph, nodes[5], nodes[6], both=False)

    shallow = infomap_with_options(
        graph, CommunityOptions(max_passes=5, max_levels=2, well_connectedness=0)
    )
    refined = infomap_with_options(
        graph, CommunityOptions(max_passes=20, max_levels=5, well_connectedness=1)
    )

    for left in nodes:
        for right in nodes:
            assert (shallow[left] == shallow[right]) == (refined[left] == refined[right])


def test_infomap_numeric_helpers_are_bounded_and_deterministic() -> None:
    first = LcgRng(42)
    second = LcgRng(42)
    assert [first.gen_range(2, 10) for _ in range(5)] == [
        second.gen_range(2, 10) for _ in range(5)
    ]
    assert all(0 <= first.gen_f32() < 1 for _ in range(10))
    assert _entropy_term(0) == 0
    assert _entropy_term(0.5) == pytest.approx(-0.5)

    graph = Graph()
    nodes = _add_nodes(graph, 2)
    _connect(graph, *nodes)
    working = WorkingGraph.from_graph(graph)
    labels = _random_walk_init(working)
    assert len(labels) == 2
    assert _compute_lmdl(working, labels, 2 * working.total_weight) >= 0
    mapping = {node: index for index, node in enumerate(nodes)}
    assert _labels_for_original(working, mapping) == [0, 1]


def test_infomap_public_response_reports_partition_quality_and_cross_edges() -> None:
    graph = Graph()
    a, b, c, d = _add_nodes(graph, 4)
    _connect(graph, a, b)
    _connect(graph, c, d)
    _connect(graph, b, c, both=False)

    result = detect_communities(graph)

    assert result["algorithm"] == "infomap"
    assert result["community_count"] == len(result["communities"])
    assert result["quality"] >= 0
    assert isinstance(result["cross_community_edges"], list)
