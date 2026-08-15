from __future__ import annotations

from ariadne_py.analysis.structure import (
    approx_betweenness,
    bridge_scores,
    call_resolution_stats,
    compute_surprise_scoring,
    core_numbers,
    cyclic_components,
    find_articulation_points,
    find_counterfactual,
    find_cycles,
    find_god_nodes,
    find_large_functions,
    find_motifs,
)
from ariadne_py.core import Edge, EdgeKind, Graph, Node, NodeKind


def _node(graph: Graph, name: str, kind: NodeKind = NodeKind.FUNCTION, **props):
    node = Node.new(kind, f"app::{name}")
    node.properties.update(props)
    return graph.add_node(node)


def _edge(graph: Graph, source, target, kind=EdgeKind.CALLS):
    graph.add_edge(source, target, Edge.extracted(kind))


def test_structural_cycles_cores_bridges_and_articulation_points() -> None:
    graph = Graph()
    a, b, c, tail = [_node(graph, name) for name in ("a", "b", "c", "tail")]
    _edge(graph, a, b)
    _edge(graph, b, c)
    _edge(graph, c, a)
    _edge(graph, c, tail)

    components = cyclic_components(graph)
    assert len(components) == 1
    assert set(components[0].nodes) == {a, b, c}
    assert find_cycles(graph)["total"] == 1
    assert find_articulation_points(graph)["articulation_points"][0]["qualified_name"] == "app::c"
    assert core_numbers(graph)[tail] == 1
    assert approx_betweenness(graph)[c] > approx_betweenness(graph).get(tail, 0)
    bridges = bridge_scores(graph, {a: 0, b: 0, c: 0, tail: 1})
    assert bridges[0].articulation
    assert bridges[0].node == c


def test_structural_self_loop_empty_and_ranked_queries() -> None:
    graph = Graph()
    self_loop = _node(graph, "loop")
    _edge(graph, self_loop, self_loop)
    large = _node(graph, "large", line_count=120)
    _node(graph, "small", line_count=5)
    _edge(graph, large, self_loop)

    # Rust petgraph semantics preserve self-loops as one-node cyclic components.
    assert [component.nodes for component in cyclic_components(graph)] == [[self_loop]]
    assert find_large_functions(graph, min_lines=30)["large_functions"][0]["line_count"] == 120
    assert find_god_nodes(graph, top=1)["total"] == 1
    assert find_god_nodes(Graph())["god_nodes"] == []


def test_motif_detection_covers_diamond_feedback_fan_in_and_fan_out() -> None:
    graph = Graph()
    a, b, c, d, e = [_node(graph, name) for name in ("a", "b", "c", "d", "e")]
    for source, target in ((a, b), (a, c), (b, d), (c, d), (b, a), (e, d), (a, e)):
        _edge(graph, source, target)

    assert find_motifs(graph, "diamond")["total"] >= 1
    assert find_motifs(graph, "feedback")["total"] == 1
    assert find_motifs(graph, "fan_in")["motifs"][0]["qualified_name"] == "app::d"
    assert find_motifs(graph, "fan_out")["motifs"][0]["qualified_name"] == "app::a"
    assert "Unknown pattern" in find_motifs(graph, "unknown")["error"]


def test_surprise_counterfactual_and_call_resolution_contracts() -> None:
    graph = Graph()
    source = _node(graph, "source")
    bridge = _node(graph, "bridge")
    target = _node(graph, "target")
    placeholder = graph.add_node(Node.new(NodeKind.FUNCTION, "call::missing"))
    _edge(graph, source, bridge)
    _edge(graph, bridge, target)
    _edge(graph, source, placeholder)

    surprises = compute_surprise_scoring(
        graph, {0: {source.value, bridge.value}, 1: {target.value, placeholder.value}}
    )
    assert surprises["total"] == 2
    assert surprises["surprises"][0]["from_community"] == 0
    counterfactual = find_counterfactual(graph, "app::bridge")
    assert counterfactual["broken_incoming_edges"]
    assert counterfactual["broken_outgoing_edges"]
    assert "Node not found" in find_counterfactual(graph, "missing")["error"]
    assert call_resolution_stats(graph) == {
        "resolved": 2, "unresolved": 1, "total": 3, "rate": 0.6667
    }
    assert call_resolution_stats(Graph())["rate"] == 1.0
