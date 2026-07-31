"""Split oversized communities — recursively subdivide large groups."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx

from ...core.graph import Graph
from ...core.id import NodeId
from .louvain import _leiden


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
    nx_graph = _to_networkx(graph)
    initial_communities: dict[int, set[int]] = _leiden(nx_graph)
    total_nodes = graph.nodes().__length_hint__() if hasattr(graph.nodes(), '__length_hint__') else sum(1 for _ in graph.nodes())
    threshold = max(int(total_nodes * threshold_pct), min_size)

    # Build community membership
    size_map: dict[int, list[int]] = defaultdict(list)
    for cid, nodes in initial_communities.items():
        for nid in nodes:
            size_map[cid].append(nid)

    # Assign new IDs for split communities
    max_cid = max(initial_communities.keys()) if initial_communities else 0
    next_id = max_cid + 1000

    # Result starts as initial assignment
    result: dict[int, int] = {}
    for cid, nodes in initial_communities.items():
        for nid in nodes:
            result[nid] = cid

    splits: list[dict[str, Any]] = []

    for cid, members in size_map.items():
        if len(members) <= threshold:
            continue

        # Build subgraph and run Leiden again
        sub_nx = nx_graph.subgraph(members).copy()
        sub_communities: dict[int, set[int]] = _leiden(sub_nx)

        for sub_cid, sub_members in sub_communities.items():
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
            "sub_communities": len(sub_communities),
        })

    # Convert result to dict format for response
    new_communities: dict[int, list[int]] = defaultdict(list)
    for nid, cid in result.items():
        new_communities[cid].append(nid)

    return {
        "operation": "split_oversized",
        "threshold": threshold,
        "original_count": len(initial_communities),
        "split_count": len(splits),
        "final_count": len(new_communities),
        "splits": splits,
    }


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
