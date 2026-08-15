"""Split oversized communities — recursively subdivide large groups."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx

from ...core.graph import Graph
from ...core.id import NodeId
from ...core.node import Node, NodeKind
from ...core.edge import Edge, EdgeKind
from .leiden import leiden_with_options
from .core import CommunityOptions


def split_oversized(
    graph: Graph,
    threshold_pct: float = 0.05,
    min_size: int = 3,
) -> dict[str, Any]:
    """Split oversized communities by recursively subdividing.

    Uses Leiden on subgraphs of communities that exceed the threshold.
    Returns updated community assignments and split details.
    """
    # Get initial communities
    initial_communities = leiden_with_options(graph, CommunityOptions(max_passes=10, max_levels=5))
    total_nodes = graph.nodes().__length_hint__() if hasattr(graph.nodes(), '__length_hint__') else sum(1 for _ in graph.nodes())
    threshold = max(int(total_nodes * threshold_pct), min_size)

    # Build community membership (node_id -> cid)
    size_map: dict[int, list[NodeId]] = defaultdict(list)
    for nid, cid in initial_communities.items():
        size_map[cid].append(nid)

    # Assign new IDs for split communities
    max_cid = max(initial_communities.values()) if initial_communities else 0
    next_id = max_cid + 1000

    # Result starts as initial assignment
    result: dict[NodeId, int] = dict(initial_communities)

    splits: list[dict[str, Any]] = []

    for cid, members in size_map.items():
        if len(members) <= threshold:
            continue

        # Build subgraph from members
        sub_graph = _build_subgraph(graph, members)

        # Run Leiden on subgraph
        sub_communities = leiden_with_options(sub_graph, CommunityOptions(max_passes=10, max_levels=5))

        # Map sub-community IDs to new global IDs
        for sub_cid in sorted(sub_communities.values()):
            sub_members = [nid for nid, c in sub_communities.items() if c == sub_cid]
            if len(sub_members) >= min_size:
                new_cid = next_id
                next_id += 1
            else:
                new_cid = cid
            for nid in sub_members:
                result[nid] = new_cid

        splits.append({
            "original_community": cid,
            "original_size": len(members),
            "sub_communities": len(set(sub_communities.values())),
        })

    # Convert result to set-based format
    new_communities: dict[int, set[NodeId]] = defaultdict(set)
    for nid, cid in result.items():
        new_communities[cid].add(nid)

    return {
        "operation": "split_oversized",
        "threshold": threshold,
        "original_count": len(initial_communities),
        "split_count": len(splits),
        "final_count": len(new_communities),
        "splits": splits,
    }


def _build_subgraph(graph: Graph, member_ids: list[NodeId]) -> Graph:
    """Build a subgraph containing only the specified members and edges between them."""
    member_set = set(nid.value for nid in member_ids)
    node_map: dict[NodeId, NodeId] = {}

    sub_graph = Graph()

    # Add nodes
    for nid in member_ids:
        node = graph.node(nid)
        if node is not None:
            new_nid = sub_graph.add_node(node)
            node_map[nid] = new_nid

    # Add edges between members
    for nid in member_ids:
        for dst, edge in graph.out_neighbors(nid):
            if dst.value in member_set:
                if nid in node_map and dst in node_map:
                    sub_graph.add_edge(node_map[nid], node_map[dst], edge)

    return sub_graph
