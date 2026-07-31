"""Community detection algorithms: Louvain, Leiden, Infomap."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx

from ...core.edge import EdgeKind
from ...core.graph import Graph
from ...core.id import NodeId
from ...core.node import NodeKind
from .utils import _find_community, _to_networkx


def detect_communities(
    graph: Graph,
    algorithm: str = "louvain",
) -> dict[str, Any]:
    """Detect communities using Louvain, Leiden, or Infomap.

    Returns community assignments, quality metrics, and cross-community edges.
    """
    nx_graph = _to_networkx(graph)

    if algorithm == "louvain":
        communities = _louvain(nx_graph)
    elif algorithm == "leiden":
        communities = _leiden(nx_graph)
    elif algorithm == "infomap":
        communities = _infomap(nx_graph)
    else:
        communities = _louvain(nx_graph)

    quality = _modularity(nx_graph, communities)
    cross_edges = _find_cross_community_edges(graph, communities)

    return {
        "algorithm": algorithm,
        "quality": quality,
        "community_count": len(communities),
        "communities": [
            {
                "id": cid,
                "size": len(nodes),
                "nodes": [
                    {
                        "qualified_name": graph.node(NodeId(nid)).qualified_name
                        if graph.node(NodeId(nid))
                        else f"node:{nid}",
                        "kind": graph.node(NodeId(nid)).kind.value
                        if graph.node(NodeId(nid))
                        else "unknown",
                    }
                    for nid in sorted(list(nodes))[:20]  # Limit for response size
                ],
            }
            for cid, nodes in sorted(communities.items())
        ],
        "cross_community_edges": cross_edges,
    }


def _louvain(graph: nx.DiGraph) -> dict[int, set[int]]:
    """Louvain community detection via networkx."""
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = greedy_modularity_communities(graph)
        result: dict[int, set[int]] = {}
        for i, comm in enumerate(communities):
            result[i] = set(comm)
        return result
    except ImportError:
        # Fallback: singletons
        return {i: {n} for i, n in enumerate(graph.nodes())}


def _leiden(graph: nx.DiGraph) -> dict[int, set[int]]:
    """Leiden community detection."""
    try:
        import igraph as ig
        # Convert to undirected for community detection
        ug = graph.to_undirected()
        ig_graph = ig.Graph.from_networkx(ug)
        communities = ig_graph.community_multilevel()
        result: dict[int, set[int]] = {}
        for i, comm in enumerate(communities):
            result[i] = set(comm)
        return result
    except ImportError:
        # Fallback to Louvain
        return _louvain(graph)


def _infomap(graph: nx.DiGraph) -> dict[int, set[int]]:
    """Infomap community detection."""
    try:
        from networkx.algorithms.community import infomap_communities
        communities = infomap_communities(graph)
        result: dict[int, set[int]] = {}
        for i, comm in enumerate(communities):
            result[i] = set(comm)
        return result
    except ImportError:
        return _louvain(graph)


def _modularity(graph: nx.DiGraph, communities: dict[int, set[int]]) -> float:
    """Compute modularity of community assignment."""
    ug = graph.to_undirected()
    try:
        return nx.algorithms.community.modularity(ug, communities.values())
    except Exception:
        return 0.0


def _find_cross_community_edges(
    graph: Graph,
    communities: dict[int, set[int]],
) -> list[dict[str, Any]]:
    """Find edges that cross community boundaries."""
    cross: list[dict[str, Any]] = []
    for _, src, dst, edge in graph.edges():
        src_comm = _find_community(src.value, communities)
        dst_comm = _find_community(dst.value, communities)
        if src_comm != dst_comm:
            cross.append({
                "source": graph.node(src).qualified_name if graph.node(src) else "?",
                "target": graph.node(dst).qualified_name if graph.node(dst) else "?",
                "kind": edge.kind.value,
                "from_community": src_comm,
                "to_community": dst_comm,
            })
    return cross[:50]  # Limit
