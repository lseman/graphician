"""Node analysis: bridge, hub, god nodes, and centrality."""

from __future__ import annotations

from typing import Any

import networkx as nx

from ...core.graph import Graph
from ...core.id import NodeId
from ...core.node import Node, NodeKind
from .utils import _to_networkx


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
    """Find top nodes by PageRank."""
    nx_graph = _to_networkx(graph)
    pagerank = nx.pagerank(nx_graph, alpha=0.85)

    gods: list[dict[str, Any]] = []
    for nid, score in sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top]:
        node = graph.node(NodeId(nid))
        if node:
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

    return {
        "degree_centrality": {
            graph.node(NodeId(nid)).qualified_name if graph.node(NodeId(nid)) else f"node:{nid}":
            round(score, 4)
            for nid, score in sorted(
                nx.degree_centrality(nx_graph).items(),
                key=lambda x: x[1],
                reverse=True,
            )[:30]
        },
        "betweenness_centrality": {
            graph.node(NodeId(nid)).qualified_name if graph.node(NodeId(nid)) else f"node:{nid}":
            round(score, 4)
            for nid, score in sorted(
                nx.betweenness_centrality(nx_graph).items(),
                key=lambda x: x[1],
                reverse=True,
            )[:30]
        },
        "pagerank": {
            graph.node(NodeId(nid)).qualified_name if graph.node(NodeId(nid)) else f"node:{nid}":
            round(score, 6)
            for nid, score in sorted(
                nx.pagerank(nx_graph).items(),
                key=lambda x: x[1],
                reverse=True,
            )[:30]
        },
    }
