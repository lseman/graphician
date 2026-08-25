"""Leiden algorithm — Louvain with refinement for guaranteed
well-connected communities.

Mirrors the Rust ``leiden.rs`` module. This is a full multi-level
implementation with:
- Multi-level Louvain as base
- Leiden-style refinement: local-move within communities with
  well-connectedness threshold
- Connectivity enforcement via undirected BFS
- Modularity gain tracking for O(E) per pass
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ...core.graph import Graph
from ...core.id import NodeId
from .core import (
    CommunityOptions,
    WorkingGraph,
    aggregate,
    densify,
    enforce_connected,
    relabel,
)
from .numba_accel import _local_move_csr, build_csr_from_working, has_numba
from .utils import _find_cross_community_edges, _modularity, _node_summary, _to_networkx


def leiden(
    graph: Graph,
    options: CommunityOptions | None = None,
) -> dict[NodeId, int]:
    """Leiden algorithm — Louvain with refinement.

    Guarantees well-connected communities by refining each partition
    to split poorly connected nodes.

    Mirrors Rust ``leiden`` (leiden.rs:5-7).

    Args:
        graph: The code graph.
        options: Detection options. Defaults to CommunityOptions().

    Returns:
        Mapping from node id to community label.
    """
    if options is None:
        options = CommunityOptions()
    return leiden_with_options(graph, options)


def leiden_with_options(
    graph: Graph,
    options: CommunityOptions,
) -> dict[NodeId, int]:
    """Leiden with explicit options.

    Mirrors Rust ``leiden_with_options`` (leiden.rs:9-20).
    """
    from .native import detect_native

    native = detect_native(graph, options, "leiden")
    if native is not None:
        return native

    working = WorkingGraph.from_graph(graph)
    if working.total_weight <= 0.0:
        nodes = list(working.original_nodes())
        return {nid: i for i, nid in enumerate(nodes)}

    final_labels = _run_multilevel_leiden(working, options)
    return {nid: label for nid, label in relabel(final_labels).items()}


def _run_multilevel_leiden(
    working: WorkingGraph,
    options: CommunityOptions,
) -> dict[NodeId, int]:
    """Multi-level Leiden: local-move → refinement → aggregate → repeat.

    Mirrors Rust ``run_multilevel_leiden`` (leiden.rs:22-55).
    """
    current: dict[NodeId, int] = {
        nid: i for i, nid in enumerate(working.original_nodes())
    }

    for _ in range(options.max_levels):
        partition = _local_move(working, options)
        moved = len(set(partition)) < working.len()

        aggregation_partition = _refinement_phase(working, partition, options)

        member_owner = {
            nid: super_idx
            for super_idx, members in enumerate(working.members)
            for nid in members
        }
        for nid in current:
            super_idx = member_owner.get(nid)
            if super_idx is not None:
                current[nid] = aggregation_partition[super_idx]

        if not moved:
            return current

        working = aggregate(working, aggregation_partition)
        if working.len() <= 1:
            break

    return current


def _local_move(working: WorkingGraph, options: CommunityOptions) -> list[int]:
    """Multi-level local move for modularity optimization.

    Identical to Louvain's local_move since Leiden uses the same
    modularity gain calculation — the difference is in the refinement
    phase and connectivity enforcement.

    Uses numba-accelerated CSR loops when available.

    Mirrors Rust ``local_move`` (leiden.rs:57-120).
    """
    if has_numba():
        row_ptr, col_idx, edge_weight = build_csr_from_working(working)
        degree = np.array(working.degree, dtype=np.float64)
        result = _local_move_csr(
            np.intp(working.len()),
            row_ptr, col_idx, edge_weight, degree,
            options.resolution,
            options.max_passes,
            options.min_modularity_gain,
            42,
        )
        return result.tolist()

    # Pure Python fallback
    n = working.len()
    comm: list[int] = list(range(n))
    comm_degree: list[float] = list(working.degree)
    comm_size: list[float] = [float(len(m)) for m in working.members]
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

            comm_degree[current] -= node_degree
            comm_size[current] -= node_mass

            weight_to_comm: dict[int, float] = defaultdict(float)
            for v, w in working.adj[u]:
                weight_to_comm[comm[v]] += w

            best = current
            best_gain = options.min_modularity_gain

            # Stay gain
            stay_weight = weight_to_comm.get(current, 0.0)
            stay_gain = stay_weight - options.resolution * node_degree * comm_degree[current] / two_m
            if stay_gain > best_gain:
                best_gain = stay_gain
                best = current

            for candidate, edge_weight in weight_to_comm.items():
                if candidate == current:
                    continue
                gain = edge_weight - options.resolution * node_degree * comm_degree[candidate] / two_m
                if gain > best_gain:
                    best_gain = gain
                    best = candidate

            comm[u] = best
            comm_degree[best] += node_degree
            comm_size[best] += node_mass
            if best != current:
                moved = True

        if not moved:
            break

    return comm


def _refinement_phase(
    working: WorkingGraph,
    partition: list[int],
    options: CommunityOptions,
) -> list[int]:
    """Leiden-style refinement: split poorly connected nodes.

    Mirrors Rust ``refinement_phase`` (leiden.rs:122-229).

    Within each community, runs local-move to split nodes whose edge
    weight into the sub-community is below the well-connectedness
    threshold.

    Args:
        working: The working graph.
        partition: Current community assignment.
        options: Detection options.

    Returns:
        Refined partition as dense label list.
    """
    n = working.len()
    two_m = 2.0 * working.total_weight

    if two_m <= 0.0:
        return list(partition)

    # Group nodes by parent community
    by_parent: dict[int, list[int]] = defaultdict(list)
    for u, c in enumerate(partition):
        by_parent[c].append(u)

    parents = sorted(by_parent.items(), key=lambda kv: kv[0])

    # Precompute parent degrees
    parent_degree: dict[int, float] = defaultdict(float)
    for u, c in enumerate(partition):
        parent_degree[c] += working.degree[u]

    # Allocate label ranges
    label_base: list[int] = []
    cursor = 0
    for _, members in parents:
        label_base.append(cursor)
        cursor += len(members)

    total_labels = cursor

    # Refine each parent community
    per_parent_labels: list[list[int]] = []

    for idx, (parent, members) in enumerate(parents):
        base = label_base[idx]
        parent_total = parent_degree.get(parent, 0.0)

        if len(members) <= 1:
            per_parent_labels.append([base])
            continue

        member_set = set(members)

        # Local refined labels: [base, base + len(members))
        refined: dict[int, int] = {u: base + i for i, u in enumerate(members)}
        local_degree: dict[int, float] = {base + i: working.degree[u] for i, u in enumerate(members)}
        local_size: dict[int, float] = {base + i: float(len(working.members[u])) for i, u in enumerate(members)}

        for _ in range(options.max_passes):
            moved = False
            for u in members:
                current_label = refined[u]
                node_degree = working.degree[u]
                node_mass = float(len(working.members[u]))
                if node_degree == 0.0:
                    continue

                # Remove u from current local community
                local_degree[current_label] -= node_degree
                local_size[current_label] -= node_mass

                # Compute weight to each local community
                weight_to_comm: dict[int, float] = defaultdict(float)
                for v, w in working.adj[u]:
                    if v not in member_set:
                        continue
                    weight_to_comm[refined[v]] += w

                best = current_label
                best_gain = options.min_modularity_gain

                # Stay gain
                stay_weight = weight_to_comm.get(current_label, 0.0)
                stay_gain = stay_weight - options.resolution * node_degree * local_degree[current_label] / two_m
                if stay_gain > best_gain:
                    best_gain = stay_gain
                    best = current_label

                for target, weight in weight_to_comm.items():
                    if target == current_label:
                        continue

                    gain = weight - options.resolution * node_degree * local_degree[target] / two_m

                    # Well-connectedness threshold
                    if options.well_connectedness > 0.0 and parent_total > 0.0 and local_degree[target] > 0.0:
                        w_ratio = weight / local_degree[target]
                        wc_threshold = options.well_connectedness * (
                            stay_weight / two_m
                            - node_degree * local_degree[current_label] / two_m
                            + node_mass * local_size[current_label] / two_m
                        )
                        if not (gain > best_gain and w_ratio >= wc_threshold):
                            continue

                    if gain > best_gain:
                        best_gain = gain
                        best = target

                if best != current_label:
                    refined[u] = best
                    moved = True

                # Restore u's contribution
                local_degree[current_label] += node_degree
                local_size[current_label] += node_mass

            if not moved:
                break

        per_parent_labels.append([refined[u] for u in members])

    # Assemble global result
    result: list[int] = [total_labels] * n
    for idx, (_, members) in enumerate(parents):
        for u, label in zip(members, per_parent_labels[idx], strict=False):
            result[u] = label

    # Enforce connectivity (Leiden guarantee)
    enforce_connected(working, result)
    result = densify(result)

    return result


# ── Public API ───────────────────────────────────────────────────────


def detect_communities(
    graph: Graph,
    algorithm: str = "leiden",
) -> dict[str, Any]:
    """Detect communities using Leiden algorithm.

    Returns community assignments, quality metrics, and cross-community edges.
    """
    communities = leiden_with_options(graph, CommunityOptions())

    set_communities: dict[int, set[int]] = {}
    for node_id, community_id in communities.items():
        set_communities.setdefault(community_id, set()).add(node_id.value)

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
                    _node_summary(graph, nid)
                    for nid in sorted(list(nodes))[:20]
                ],
            }
            for cid, nodes in sorted(set_communities.items())
        ],
        "cross_community_edges": cross_edges,
    }
