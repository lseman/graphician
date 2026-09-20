"""Node analysis: bridge, hub, god nodes, and centrality."""

from __future__ import annotations

from typing import Any

import networkx as nx

from ...core.graph import Graph
from ...core.id import NodeId
from ...core.node import Node, NodeKind
from ..centrality import _BUILTIN_NOISE, _JSON_KEY_NOISE
from .utils import _qualified_name_or_fallback, _to_networkx


def find_bridge_nodes(
    graph: Graph,
    top: int = 20,
) -> dict[str, Any]:
    """Find bridge/chokepoint nodes.

    Nodes whose removal would disconnect the graph or increase
    the number of weakly connected components.
    """
    nx_graph = _to_networkx(graph)
    ug = nx_graph.to_undirected()

    # Find articulation points
    articulation = set(nx.articulation_points(ug))

    # Compute betweenness centrality
    betweenness = nx.betweenness_centrality(ug)

    # Bridge nodes: high betweenness + articulation point
    bridges: list[dict[str, Any]] = []
    for nid in articulation:
        node = graph.node(NodeId(nid))
        if node:
            bridges.append({
                "qualified_name": node.qualified_name,
                "kind": node.kind.value,
                "betweenness": round(betweenness.get(nid, 0), 4),
                "degree": betweenness.get(nid, 0),
            })

    # Also include high-betweenness nodes even if not articulation points
    for nid, score in sorted(betweenness.items(), key=lambda x: x[1], reverse=True):
        if nid not in articulation:
            node = graph.node(NodeId(nid))
            if node and score > 0.1:
                bridges.append({
                    "qualified_name": node.qualified_name,
                    "kind": node.kind.value,
                    "betweenness": round(score, 4),
                    "degree": score,
                })

    bridges.sort(key=lambda x: x["betweenness"], reverse=True)
    return {
        "bridge_nodes": bridges[:top],
        "total": len(bridges),
    }


def find_hub_nodes(
    graph: Graph,
    top: int = 20,
) -> dict[str, Any]:
    """Find hub nodes by degree centrality."""
    nx_graph = _to_networkx(graph)
    degree = nx.degree_centrality(nx_graph)

    hubs: list[dict[str, Any]] = []
    for nid, score in sorted(degree.items(), key=lambda x: x[1], reverse=True)[:top]:
        node = graph.node(NodeId(nid))
        if node:
            hubs.append({
                "qualified_name": node.qualified_name,
                "kind": node.kind.value,
                "degree_centrality": round(score, 4),
                "in_degree": sum(1 for _ in graph.in_neighbors(NodeId(nid))),
                "out_degree": sum(1 for _ in graph.out_neighbors(NodeId(nid))),
            })

    return {"hub_nodes": hubs, "total": len(hubs)}


def find_god_nodes(
    graph: Graph,
    top: int = 20,
) -> dict[str, Any]:
    """Find top nodes by PageRank, filtered to exclude synthetic noise."""
    from ..centrality import pagerank

    scores = pagerank(graph, damping=0.85)

    gods: list[dict[str, Any]] = []
    for nid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        if len(gods) >= top:
            break
        node = graph.node(nid)
        if node and not is_rank_noise(node):
            gods.append({
                "qualified_name": node.qualified_name,
                "kind": node.kind.value,
                "pagerank": round(score, 6),
            })

    return {"god_nodes": gods, "total": len(gods)}


def compute_centrality(
    graph: Graph,
) -> dict[str, Any]:
    """Compute all centrality measures."""
    nx_graph = _to_networkx(graph)

    from ..centrality import pagerank

    pagerank_scores = pagerank(graph, damping=0.85)
    return {
        "degree_centrality": {
            _qualified_name_or_fallback(graph, NodeId(nid)): round(score, 4)
            for nid, score in sorted(
                nx.degree_centrality(nx_graph).items(),
                key=lambda x: x[1],
                reverse=True,
            )[:30]
        },
        "betweenness_centrality": {
            _qualified_name_or_fallback(graph, NodeId(nid)): round(score, 4)
            for nid, score in sorted(
                nx.betweenness_centrality(nx_graph).items(),
                key=lambda x: x[1],
                reverse=True,
            )[:30]
        },
        "pagerank": {
            _qualified_name_or_fallback(graph, nid): round(score, 6)
            for nid, score in sorted(
                pagerank_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:30]
        },
    }


def is_rank_noise(node: Node) -> bool:
    """True for nodes that inflate god-node rankings without representing a real symbol.

    Filters out file containers, synthetic flow and hyperedge nodes, unresolved call
    placeholders, concept nodes, builtin type names, method stubs, JSON-key
    identifiers, and file-level hubs.
    """
    if node.kind in (NodeKind.FILE, NodeKind.FLOW, NodeKind.HYPEREDGE):
        return True
    if node.qualified_name.startswith("call::"):
        return True
    if node.kind == NodeKind.CONCEPT:
        return True
    if len(node.name) == 1 or (len(node.name) > 0 and node.name.isdigit()):
        return True
    # Method stubs: synthetic AST nodes like ".method_name()" or "function_name()"
    if node.name.startswith(".") or (node.name.endswith("()") and node.kind in (NodeKind.FUNCTION, NodeKind.METHOD)):
        return True
    # File-level hub: node name matches the source filename
    if node.source_uri:
        basename = node.source_uri.rsplit("/", 1)[-1]
        if basename and node.name == basename.rsplit(".", 1)[0]:
            return True
    # JSON key nodes
    if (
        node.source_uri
        and node.source_uri.lower().endswith(".json")
        and node.name in _JSON_KEY_NOISE
    ):
        return True
    return node.name in _BUILTIN_NOISE
