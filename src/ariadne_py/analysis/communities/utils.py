"""Shared utilities for communities module."""

from __future__ import annotations

import networkx as nx

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
