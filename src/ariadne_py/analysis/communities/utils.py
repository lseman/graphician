"""Shared utilities for communities module."""

from __future__ import annotations

import networkx as nx
from typing import Any

from ...core.graph import Graph
from ...core.id import NodeId
from ...core.node import NodeKind


def _find_community(node_id: int, communities: dict[int, set[int]]) -> int:
    """Find which community a node belongs to."""
    for cid, nodes in communities.items():
        if node_id in nodes:
            return cid
    return -1


def _to_networkx(graph: Graph) -> nx.DiGraph:
    """Convert Ariadne Graph to NetworkX DiGraph."""
    nx_graph = nx.DiGraph()

    for nid, node in graph.nodes():
        nx_graph.add_node(nid.value, **{
            "qualified_name": node.qualified_name,
            "kind": node.kind.value,
            "name": node.name,
        })

    for _, src, dst, edge in graph.edges():
        nx_graph.add_edge(src.value, dst.value, **{
            "kind": edge.kind.value,
            "confidence": edge.confidence.value,
        })

    return nx_graph


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
