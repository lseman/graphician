"""Louvain algorithm — standard multi-level modularity optimization.

Mirrors the Rust ``louvain.rs`` module. Implements:
- Full multi-level modularity optimization (not a thin NetworkX wrapper)
- Edge kind weights and ambiguous edge down-weighting
- Modularity gain with degree/size tracking for O(E) per pass
- Configurable resolution and objectives
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ...core.edge import EdgeKind
from ...core.graph import Graph
from ...core.id import NodeId
from .core import (
    CommunityOptions,
    WorkingGraph,
    aggregate,
    densify,
    identity_labels,
    relabel,
)
from .utils import _find_community, _to_networkx, _modularity, _find_cross_community_edges


def louvain(
    graph: Graph,
    options: CommunityOptions | None = None,
) -> dict[int, set[int]]:
    """Louvain community detection with full multi-level optimization.

    Mirrors Rust ``louvain`` (louvain.rs:5-7).

    Args:
        graph: The code graph.
        options: Detection options. Defaults to CommunityOptions().

    Returns:
        Mapping from node index to community id (set-based).
    """
    if options is None:
        options = CommunityOptions()
    return louvain_with_options(graph, options)


def louvain_with_options(
    graph: Graph,
    options: CommunityOptions,
) -> dict[int, set[int]]:
    """Louvain with explicit options.

    Mirrors Rust ``louvain_with_options`` (louvain.rs:9-20).
    """
    working = WorkingGraph.from_graph(graph)
    if working.total_weight <= 0.0:
        nodes = list(working.original_nodes())
        return {nid: i for i, nid in enumerate(nodes)}

    final_labels = _run_multilevel_louvain(working, options)
    return {nid: label for nid, label in relabel(final_labels).items()}


def _run_multilevel_louvain(
    working: WorkingGraph,
    options: CommunityOptions,
) -> dict[NodeId, int]:
    """Multi-level Louvain: local-move → densify → aggregate → repeat.

    Mirrors Rust ``run_multilevel_louvain`` (louvain.rs:22-54).
    """
    current: dict[NodeId, int] = {
        nid: i for i, nid in enumerate(working.original_nodes())
    }

    for _ in range(options.max_levels):
        partition = _local_move(working, options)
        dense = densify(partition)
        moved = len(set(dense)) < working.len()

        for nid in current:
            # Find current super-node index for this original node
            for super_idx, members in enumerate(working.members):
                if nid in members:
                    current[nid] = dense[super_idx]
                    break

        if not moved:
            return current

        working = aggregate(working, dense)
        if working.len() <= 1:
            break

    return current


def _local_move(working: WorkingGraph, options: CommunityOptions) -> list[int]:
    """One multi-level pass of local move for modularity optimization.

    Mirrors Rust ``local_move`` (louvain.rs:56-120).

    Uses incremental tracking of comm_degree and comm_size for O(E) per pass.
    """
    n = working.len()
    comm: list[int] = list(range(n))
    comm_degree: list[float] = list(working.degree)
    comm_size: list[float] = [len(m) for m in working.members]
    two_m = 2.0 * working.total_weight

    if two_m <= 0.0:
        return comm

    for _ in range(options.max_passes):
        moved = False
        for u in range(n):
            current = comm[u]
            node_degree = working.degree[u]
            if node_degree == 0.0:
                continue

            node_mass = float(len(working.members[u]))

            # Remove u from its current community for gain calculation
            comm_degree[current] -= node_degree
            comm_size[current] -= node_mass

            # Compute weight from u to each neighboring community
            weight_to_comm: dict[int, float] = defaultdict(float)
            for v, w in working.adj[u]:
                weight_to_comm[comm[v]] += w

            # Find best community to move u to
            best = current
            best_gain = options.min_modularity_gain

            # Stay gain
            stay_weight = weight_to_comm.get(current, 0.0)
            stay_gain = stay_weight - options.resolution * node_degree * comm_degree[current] / two_m
            if stay_gain > best_gain:
                best_gain = stay_gain
                best = current

            # Try each candidate
            for candidate, edge_weight in weight_to_comm.items():
                if candidate == current:
                    continue
                gain = edge_weight - options.resolution * node_degree * comm_degree[candidate] / two_m
                if gain > best_gain:
                    best_gain = gain
                    best = candidate

            # Update
            comm[u] = best
            comm_degree[best] += node_degree
            comm_size[best] += node_mass
            if best != current:
                moved = True

        if not moved:
            break

    return comm


# ── Public API ───────────────────────────────────────────────────────


def detect_communities(
    graph: Graph,
    algorithm: str = "louvain",
) -> dict[str, Any]:
    """Detect communities using Louvain, Leiden, or Infomap.

    Returns community assignments, quality metrics, and cross-community edges.
    """
    if algorithm == "louvain":
        communities = louvain_with_options(graph, CommunityOptions())
    elif algorithm == "leiden":
        from .leiden import leiden_with_options
        communities = leiden_with_options(graph, CommunityOptions())
    elif algorithm == "infomap":
        try:
            from .infomap import infomap_with_options
            communities = infomap_with_options(graph, CommunityOptions())
        except ImportError:
            communities = louvain_with_options(graph, CommunityOptions())
    else:
        communities = louvain_with_options(graph, CommunityOptions())

    # Convert to set-based format for backward compatibility
    # communities is dict[NodeId, int] → set-based dict[int, set[int]]
    set_communities: dict[int, set[int]] = {}
    for nid, cid in communities.items():
        set_communities.setdefault(cid, set()).add(nid.value)

    ug = _to_networkx(graph).to_undirected()
    quality = _modularity(ug, set_communities)
    cross_edges = _find_cross_community_edges(graph, set_communities)

    return {
        "algorithm": algorithm,
        "quality": quality,
        "community_count": len(set_communities),
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
                    for nid in sorted(nodes)[:20]
                ],
            }
            for cid, nodes in sorted(set_communities.items())
        ],
        "cross_community_edges": cross_edges,
    }
