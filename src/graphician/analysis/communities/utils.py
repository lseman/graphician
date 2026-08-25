"""Shared utilities for communities module."""

from __future__ import annotations

from typing import Any

import networkx as nx

from ...core.graph import Graph
from ...core.id import NodeId


def _node_summary(graph: Graph, node_id: int) -> dict[str, Any]:
    """Build a compact {qualified_name, kind} summary for a node id.

    Falls back to a placeholder if the node is missing from the graph.
    """
    node = graph.node(NodeId(node_id))
    if node is None:
        return {"qualified_name": f"node:{node_id}", "kind": "unknown"}
    return {"qualified_name": node.qualified_name, "kind": node.kind.value}


def _qualified_name_or_fallback(graph: Graph, node_id: NodeId) -> str:
    """Resolve a node's qualified name, or a `node:<id>` placeholder if missing."""
    node = graph.node(node_id)
    return node.qualified_name if node is not None else f"node:{node_id.value}"


def _find_community(node_id: int, communities: dict[int, set[int]]) -> int:
    """Find which community a node belongs to."""
    for cid, nodes in communities.items():
        if node_id in nodes:
            return cid
    return -1


def _to_networkx(graph: Graph) -> nx.DiGraph:
    """Convert the code graph to NetworkX DiGraph."""
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
    except Exception:  # noqa: BLE001 -- networkx raises varied errors on degenerate partitions
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
            src_node = graph.node(src)
            dst_node = graph.node(dst)
            cross.append({
                "source": src_node.qualified_name if src_node else "?",
                "target": dst_node.qualified_name if dst_node else "?",
                "kind": edge.kind.value,
                "from_community": src_comm,
                "to_community": dst_comm,
            })
    return cross[:50]  # Limit
