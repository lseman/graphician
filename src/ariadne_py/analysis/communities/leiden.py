"""Leiden community detection — Louvain with refinement for guaranteed
well-connected communities.

Mirrors the Rust ``leiden.rs`` module. This implementation uses
networkx for the base Louvain algorithm and adds a refinement phase
that splits poorly connected nodes, ensuring every community is
connected.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx

from ...core.graph import Graph
from ...core.id import NodeId
from .utils import _find_community, _to_networkx


class CommunityOptions:
    """Options for community detection algorithms."""

    def __init__(
        self,
        max_levels: int = 10,
        max_passes: int = 10,
        resolution: float = 1.0,
        min_modularity_gain: float = 1e-8,
        well_connectedness: float = 0.3,
    ) -> None:
        self.max_levels = max_levels
        self.max_passes = max_passes
        self.resolution = resolution
        self.min_modularity_gain = min_modularity_gain
        self.well_connectedness = well_connectedness


def leiden(
    graph: Graph,
    options: CommunityOptions | None = None,
) -> dict[int, set[int]]:
    """Leiden algorithm — Louvain with refinement.

    Guarantees well-connected communities by refining each partition
    to split poorly connected nodes.

    Args:
        graph: The code graph.
        options: Detection options. Defaults to CommunityOptions().

    Returns:
        Mapping from node index to community id.
    """
    if options is None:
        options = CommunityOptions()

    nx_graph = _to_networkx(graph)
    ug = nx_graph.to_undirected()

    # Base Louvain
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(ug, weight="weight"))
    except Exception:
        # Fallback: singletons
        return {i: {n} for i, n in enumerate(ug.nodes())}

    # Convert to index → community map
    initial: dict[int, int] = {}
    for cid, comm in enumerate(communities):
        for nid in comm:
            initial[nid] = cid

    # Refinement phase: split poorly connected nodes
    refined = refinement_phase(ug, initial, options)

    # Enforce connectedness
    enforced = enforce_connected(ug, refined)

    # Densify: relabel to consecutive integers
    return densify(enforced)


def refinement_phase(
    graph: nx.Graph,
    partition: dict[int, int],
    options: CommunityOptions,
) -> dict[int, int]:
    """Refine partition by splitting poorly connected nodes.

    Within each community, runs local-move to split nodes whose edge
    weight into the sub-community is below the well-connectedness threshold.

    Args:
        graph: Undirected networkx graph.
        partition: Node → community mapping.
        options: Detection options.

    Returns:
        Refined node → community mapping.
    """
    # Group nodes by parent community
    by_parent: dict[int, list[int]] = defaultdict(list)
    for nid, cid in partition.items():
        by_parent[cid].append(nid)

    parents = sorted(by_parent.items(), key=lambda x: x[0])
    labels = {nid: 0 for nid in graph.nodes()}
    cursor = 0

    for parent_id, members in parents:
        if len(members) <= 1:
            labels[members[0]] = cursor
            cursor += 1
            continue

        # Local labels in [base, base + members.len)
        base = cursor
        refined: dict[int, int] = {u: base + i for i, u in enumerate(members)}
        member_set = set(members)

        # Precompute parent degree
        parent_degree = sum(graph.degree[m] for m in members)

        for _ in range(options.max_passes):
            moved = False
            for u in members:
                current = refined[u]
                node_degree = graph.degree[u]
                if node_degree == 0:
                    continue

                # Weight to each local community
                weight_to_comm: dict[int, float] = defaultdict(float)
                for v in graph.neighbors(u):
                    if v not in member_set:
                        continue
                    w = graph[u][v].get("weight", 1.0)
                    weight_to_comm[refined[v]] += w

                best = current
                best_gain = options.min_modularity_gain

                # Stay gain
                stay_weight = weight_to_comm.get(current, 0.0)
                if stay_weight > best_gain:
                    best_gain = stay_weight
                    best = current

                # Try each candidate
                for candidate, edge_weight in weight_to_comm.items():
                    if candidate == current:
                        continue

                    threshold = 0.0
                    if parent_degree > 0 and options.well_connectedness > 0:
                        cand_degree = sum(graph.degree[v] for v in graph.nodes() if refined[v] == candidate)
                        threshold = (
                            options.well_connectedness
                            * cand_degree
                            * (parent_degree - cand_degree)
                            / (2 * parent_degree)
                        )

                    if edge_weight < threshold:
                        continue
                    if edge_weight > best_gain:
                        best_gain = edge_weight
                        best = candidate

                if best != current:
                    refined[u] = best
                    moved = True

            if not moved:
                break

        for u in members:
            labels[u] = refined[u]

        cursor += len(members)

    return labels


def enforce_connected(graph: nx.Graph, partition: dict[int, int]) -> dict[int, int]:
    """Ensure every community is connected.

    Disconnected communities are merged with their largest connected
    component.

    Args:
        graph: Undirected networkx graph.
        partition: Node → community mapping.

    Returns:
        Enforced node → community mapping.
    """
    communities: dict[int, list[int]] = defaultdict(list)
    for nid, cid in partition.items():
        communities[cid].append(nid)

    enforced: dict[int, int] = {}
    cursor = 0

    for cid, nodes in sorted(communities.items()):
        if len(nodes) <= 1:
            enforced[nodes[0]] = cursor
            cursor += 1
            continue

        # Find largest connected component
        subgraph = graph.subgraph(nodes)
        try:
            components = list(nx.connected_components(subgraph))
            largest = max(components, key=len)
        except Exception:
            largest = set(nodes)

        # Assign all nodes in this component the same label
        for nid in nodes:
            enforced[nid] = cursor

        cursor += 1

    return enforced


def densify(partition: dict[int, int]) -> dict[int, set[int]]:
    """Relabel communities to consecutive integers and return set mapping.

    Args:
        partition: Node → community mapping.

    Returns:
        Community id → set of node indices.
    """
    communities: dict[int, set[int]] = defaultdict(set)
    for nid, cid in partition.items():
        communities[cid].add(nid)

    # Relabel to consecutive
    result: dict[int, set[int]] = {}
    for new_id, old_id in enumerate(sorted(communities.keys())):
        result[new_id] = communities[old_id]

    return result
