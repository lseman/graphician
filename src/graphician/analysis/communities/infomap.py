"""Infomap algorithm — LMDL description-length optimization via random walks.

Mirrors the Rust ``infomap.rs`` module. This is the most sophisticated
algorithm in the suite:
- Multi-level aggregation with LMDL acceptance/rejection
- Random-walk initialization with deterministic LCG
- LMDL-based local-move with incremental delta computation
- Leiden-style refinement within communities
- O(degree) per-node LMDL delta (only two communities re-evaluated)
"""

from __future__ import annotations

import math
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
from .numba_accel import _random_walk_init_csr, build_csr_from_working, has_numba
from .utils import _find_cross_community_edges, _modularity, _node_summary, _to_networkx

# ── LCG RNG ─────────────────────────────────────────────────────────


class LcgRng:
    """Linear congruential generator for deterministic random walks.

    Mirrors Rust ``LcgRng`` (infomap.rs:382-394).
    Uses Numerical Recipes constants.
    """

    def __init__(self, seed: int = 0x5DEECE66D) -> None:
        self.state: int = seed

    def gen_range(self, low: int, high: int) -> int:
        """Uniform integer in [low, high)."""
        self.state = (self.state * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
        return low + (self.state % (high - low))

    def gen_f32(self) -> float:
        """Uniform float in [0, 1)."""
        self.state = (self.state * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
        return (((self.state >> 11) & 0x1FFFFF) / 0x200000)


# ── LMDL helpers ─────────────────────────────────────────────────────


def _entropy_term(probability: float) -> float:
    """Compute -p * log2(p), returning 0 if p <= 0.

    Mirrors Rust ``entropy_term`` (infomap.rs:256-260).
    """
    if probability > 0.0:
        return probability * math.log2(probability)
    return 0.0


# ── Infomap algorithm ───────────────────────────────────────────────


def infomap(
    graph: Graph,
    options: CommunityOptions | None = None,
) -> dict[NodeId, int]:
    """Infomap community detection with LMDL optimization.

    Mirrors Rust ``infomap`` (infomap.rs:6-8).

    Runs multi-level Infomap with random-walk initialization,
    LMDL-based local-move, and Leiden-style refinement.

    Args:
        graph: The code graph.
        options: Detection options. Defaults to CommunityOptions().

    Returns:
        Mapping from node id to community label.
    """
    if options is None:
        options = CommunityOptions()
    return infomap_with_options(graph, options)


def infomap_with_options(
    graph: Graph,
    options: CommunityOptions,
) -> dict[NodeId, int]:
    """Infomap with explicit options.

    Mirrors Rust ``infomap_with_options`` (infomap.rs:10-29).
    """
    from .native import detect_native

    native = detect_native(graph, options, "infomap")
    if native is not None:
        return native

    working = WorkingGraph.from_graph(graph)
    if working.total_weight <= 0.0:
        nodes = list(working.original_nodes())
        return {nid: i for i, nid in enumerate(nodes)}

    final_labels = _run_infomap_multilevel(working, options)
    return {nid: label for nid, label in relabel(final_labels).items()}


def _run_infomap_multilevel(
    working: WorkingGraph,
    options: CommunityOptions,
) -> dict[NodeId, int]:
    """Multi-level Infomap: random walk init → LMDL local-move → refine → aggregate.

    Mirrors Rust ``run_infomap_multilevel`` (infomap.rs:31-140).
    """
    # Clone the original working graph for LMDL computation on the original graph.
    original_nodes = list(working.original_nodes())
    original_members = [list(m) for m in working.members]
    original_adj = [list(a) for a in working.adj]
    original_self_loop = list(working.self_loop)
    original_degree = list(working.degree)
    original_two_m = 2.0 * working.total_weight

    if original_two_m <= 0.0:
        return {nid: i for i, nid in enumerate(original_nodes)}

    # Build original working graph (deep copy)
    original_working = WorkingGraph(
        members=original_members,
        adj=original_adj,
        self_loop=original_self_loop,
        degree=original_degree,
        total_weight=working.total_weight,
    )

    # Original node → super-node mapping
    original_to_super: dict[NodeId, int] = {
        nid: i for i, nid in enumerate(original_nodes)
    }

    best_mapping: dict[NodeId, int] = dict(original_to_super)
    best_lmdl = _compute_lmdl(
        original_working,
        _labels_for_original(original_working, best_mapping),
        original_two_m,
    )

    for _level in range(options.max_levels):
        two_m = 2.0 * working.total_weight
        if two_m <= 0.0:
            break

        # Random-walk initialization
        labels = _random_walk_init(working)

        # Greedy local-move to minimize LMDL
        prev_pass_lmdl = float("inf")
        for _pass in range(options.max_passes):
            new_labels, lmdl = _infomap_local_move(working, labels, two_m, options.max_passes)
            labels = new_labels

            # Convergence check
            improved = abs(prev_pass_lmdl - lmdl) > 1e-8
            prev_pass_lmdl = lmdl
            if not improved and _pass >= 2:
                break

        # Leiden-style refinement
        if options.well_connectedness > 0.0:
            aggregation_partition = _infomap_refinement(working, labels, options)
        else:
            aggregation_partition = densify(labels)

        # Check if partition changed
        moved = any(label != aggregation_partition[i] for i, label in enumerate(labels))

        # Update original-node mapping: map each original node's current super-node
        # to the aggregated community. Mirrors Rust:
        #   for super_node in candidate_mapping.values_mut() {
        #       *super_node = aggregation_partition[*super_node];
        #   }
        # This uses the CURRENT super-node index as a position into the
        # aggregation_partition vector (which has the same length as the
        # current working graph).
        candidate_mapping: dict[NodeId, int] = dict(original_to_super)
        for nid in candidate_mapping:
            current_super = candidate_mapping[nid]
            candidate_mapping[nid] = aggregation_partition[current_super]

        candidate_lmdl = _compute_lmdl(
            original_working,
            _labels_for_original(original_working, candidate_mapping),
            original_two_m,
        )

        if candidate_lmdl + 1e-6 < best_lmdl:
            best_lmdl = candidate_lmdl
            best_mapping = candidate_mapping
            original_to_super = candidate_mapping
        else:
            # Reject: stop to avoid index mismatch
            return best_mapping

        if not moved:
            return best_mapping

        working = aggregate(working, aggregation_partition)
        if working.len() <= 1:
            break

    return best_mapping


def _labels_for_original(
    working: WorkingGraph, mapping: dict[NodeId, int]
) -> list[int]:
    """Get labels for original nodes in graph order."""
    return [mapping.get(nid, 0xFFFFFFFF) for nid in working.original_nodes()]


# ── Random walk initialization ───────────────────────────────────────


def _random_walk_init(working: WorkingGraph) -> list[int]:
    """Random-walk initialization for community labels.

    Mirrors Rust ``random_walk_init`` (infomap.rs:142-193).

    Runs random walks and assigns each node the label of its
    most-visited neighbor.
    Uses numba-accelerated CSR loops when available.
    """
    n = working.len()
    walk_steps = max(n, 10) * 5
    walk_count = max(n, 10)

    if has_numba():
        row_ptr, col_idx, edge_weight = build_csr_from_working(working)
        degree = np.array(working.degree, dtype=np.float64)
        self_loop = np.array(working.self_loop, dtype=np.float64)
        labels = _random_walk_init_csr(
            n, row_ptr, col_idx, edge_weight, degree, self_loop,
            walk_steps, walk_count, 42,
        )
        return labels.tolist()

    # Pure Python fallback
    rng = LcgRng()

    # Compute degree for random walk selection
    degree = [
        sum(w for _, w in working.adj[u]) + 2.0 * working.self_loop[u]
        for u in range(n)
    ]

    # Run random walks and count visits
    visits: list[int] = [0] * n
    for _ in range(walk_count):
        node = rng.gen_range(0, n)
        for _ in range(walk_steps):
            visits[node] += 1
            total = degree[node]
            if total <= 0.0:
                break
            r = rng.gen_f32() * total
            next_node = node
            for v, w in working.adj[node]:
                r -= w
                if r <= 0.0:
                    next_node = v
                    break
            node = next_node

    # Assign label: neighbor with highest visit count
    labels = []
    for u in range(n):
        best_neighbor = u
        best_visits = visits[u]
        for v, _ in working.adj[u]:
            if visits[v] > best_visits:
                best_visits = visits[v]
                best_neighbor = v
        labels.append(best_neighbor)

    return labels


# ── LMDL computation ────────────────────────────────────────────────


def _compute_lmdl(
    working: WorkingGraph,
    labels: list[int],
    two_m: float,
) -> float:
    """Compute the two-level map-equation description length (LMDL).

    Mirrors Rust ``compute_lmdl`` (infomap.rs:195-225).

    LMDL = H(q_total) - 2*sum_c H(exit_c) + sum_c H(p_c + exit_c) - sum_i H(p_i)

    Args:
        working: The working graph.
        labels: Current community assignment.
        two_m: 2 * total_weight.

    Returns:
        LMDL value (non-negative).
    """
    n = working.len()
    if n == 0 or two_m <= 0.0:
        return 0.0

    flow = _compute_community_flow(labels, working, two_m)

    # Iterate in sorted order for deterministic floating-point summation
    keys = sorted(flow.keys())

    q_total: float = sum(flow[c].exit_probability for c in keys)
    length: float = _entropy_term(q_total)
    for c in keys:
        length -= _entropy_term(flow[c].exit_probability)

    for c in keys:
        p_circle = flow[c].node_probability + flow[c].exit_probability
        length += _entropy_term(p_circle)
        length -= _entropy_term(flow[c].exit_probability)
        for node_prob in flow[c].node_probabilities:
            length -= _entropy_term(node_prob)

    return max(length, 0.0)


class _CommunityFlow:
    """Per-community flow statistics."""
    __slots__ = ("exit_probability", "node_probabilities", "node_probability")

    def __init__(self) -> None:
        self.node_probability: float = 0.0
        self.exit_probability: float = 0.0
        self.node_probabilities: list[float] = []


def _compute_community_flow(
    labels: list[int],
    working: WorkingGraph,
    two_m: float,
) -> dict[int, _CommunityFlow]:
    """Compute flow statistics for each community.

    Mirrors Rust ``compute_community_flow`` (infomap.rs:227-250).
    """
    flow: dict[int, _CommunityFlow] = {}

    for u, label in enumerate(labels):
        if label not in flow:
            flow[label] = _CommunityFlow()

        entry = flow[label]
        node_probability = working.degree[u] / two_m
        entry.node_probability += node_probability
        entry.node_probabilities.append(node_probability)

        for v, w in working.adj[u]:
            if labels[v] != label:
                entry.exit_probability += w / two_m

    return flow


# ── Incremental LMDL delta ──────────────────────────────────────────


class _CommunityStats:
    """Per-community flow stats for incremental computation."""
    __slots__ = ("exit_probability", "h_p_sum", "node_probability")

    def __init__(
        self,
        node_probability: float = 0.0,
        exit_probability: float = 0.0,
        h_p_sum: float = 0.0,
    ) -> None:
        self.node_probability = node_probability
        self.exit_probability = exit_probability
        self.h_p_sum = h_p_sum


def _precompute_incremental(
    labels: list[int],
    working: WorkingGraph,
    two_m: float,
) -> tuple[list[_CommunityStats], list[dict[int, float]]]:
    """Precompute per-community stats and incoming weights for LMDL delta.

    Mirrors Rust ``precompute_incremental`` (infomap.rs:252-306).

    Returns:
        (stats, incoming_to) where:
        - stats[c] = (node_probability, exit_probability, h_p_sum)
        - incoming_to[u][c] = sum of w(v,u) for v in community c (raw weights)
    """
    n = working.len()
    max_label = max(labels) if labels else 0

    stats: list[_CommunityStats] = [
        _CommunityStats() for _ in range(max_label + 1)
    ]

    for u, label in enumerate(labels):
        p = working.degree[u] / two_m
        entry = stats[label]
        entry.node_probability += p
        entry.h_p_sum += _entropy_term(p)

    incoming_to: list[dict[int, float]] = [defaultdict(float) for _ in range(n)]

    for v in range(n):
        lv = labels[v]
        for u_idx, w in working.adj[v]:
            # incoming_to tracks weight to node u from community lv
            # Note: in Rust, v is the source and u is the target
            u = u_idx
            incoming_to[u][lv] += w
            # If edge crosses communities, add to exit probability
            if labels[u_idx] != lv:
                stats[lv].exit_probability += w / two_m

    # Convert defaultdicts to regular dicts
    incoming_to = [dict(d) for d in incoming_to]

    return stats, incoming_to


def _infomap_lmdl_delta(
    labels: list[int],
    old: int,
    new: int,
    u: int,
    working: WorkingGraph,
    two_m: float,
    stats: list[_CommunityStats],
    incoming_to: list[dict[int, float]],
) -> float:
    """Compute LMDL delta for moving node u from old to new community.

    O(degree(u)) — only two communities re-evaluated.
    Returns negative if moving improves the partition.

    Mirrors Rust ``infomap_lmdl_delta`` (infomap.rs:308-375).
    """
    if old == new:
        return float("inf")

    p_old = stats[old].node_probability
    exit_old = stats[old].exit_probability
    p_new = stats[new].node_probability
    exit_new = stats[new].exit_probability
    p_u = working.degree[u] / two_m

    # Outgoing weights from u to old and new communities
    w_out_old = 0.0
    w_out_new = 0.0
    w_out_other = 0.0
    for v_idx, w in working.adj[u]:
        label_v = labels[v_idx]
        w_m = w / two_m
        if label_v == old:
            w_out_old += w_m
        elif label_v == new:
            w_out_new += w_m
        else:
            w_out_other += w_m

    # Incoming weights from old/new communities to u
    w_in_old = incoming_to[u].get(old, 0.0) / two_m
    w_in_new = incoming_to[u].get(new, 0.0) / two_m

    # Exit probability deltas
    delta_exit_old = w_in_old - w_out_new - w_out_other
    delta_exit_new = w_out_old + w_out_other - w_in_new
    delta_q_total = delta_exit_old + delta_exit_new

    # LMDL delta
    q_total = sum(s.exit_probability for s in stats)
    q_total_after = q_total + delta_q_total

    q_old_before = p_old + exit_old
    q_old_after = (p_old - p_u) + (exit_old + delta_exit_old)

    q_new_before = p_new + exit_new
    q_new_after = (p_new + p_u) + (exit_new + delta_exit_new)

    return (
        _entropy_term(q_total_after) - _entropy_term(q_total)
        - 2.0 * (_entropy_term(exit_old + delta_exit_old) - _entropy_term(exit_old))
        - 2.0 * (_entropy_term(exit_new + delta_exit_new) - _entropy_term(exit_new))
        + (_entropy_term(q_old_after) - _entropy_term(q_old_before))
        + (_entropy_term(q_new_after) - _entropy_term(q_new_before))
    )


# ── Local move ──────────────────────────────────────────────────────


def _infomap_local_move(
    working: WorkingGraph,
    labels: list[int],
    two_m: float,
    max_passes: int,
) -> tuple[list[int], float]:
    """One greedy local-move pass to minimize LMDL.

    Mirrors Rust ``infomap_local_move`` (infomap.rs:377-426).

    Returns:
        (new_labels, lmdl) tuple.
    """
    n = working.len()
    current = list(labels)
    best_lmdl = _compute_lmdl(working, current, two_m)

    for _ in range(max_passes):
        improved = False
        # Precompute stats once per pass — O(E)
        stats, incoming_to = _precompute_incremental(current, working, two_m)

        for u in range(n):
            old = current[u]
            # Collect neighbor communities, sorted for deterministic tie-break
            neighbor_comms = sorted(set(labels[v_idx] for v_idx, _ in working.adj[u]))
            best_new = old
            best_delta = 0.0

            for cand in neighbor_comms:
                if cand == old:
                    continue
                delta = _infomap_lmdl_delta(
                    current, old, cand, u, working, two_m, stats, incoming_to
                )
                if delta < best_delta:
                    best_delta = delta
                    best_new = cand

            if best_new != old:
                current[u] = best_new
                improved = True

        if not improved:
            break

        best_lmdl = _compute_lmdl(working, current, two_m)

    return current, best_lmdl


# ── Refinement ──────────────────────────────────────────────────────


def _infomap_refinement(
    working: WorkingGraph,
    partition: list[int],
    options: CommunityOptions,
) -> list[int]:
    """Leiden-style refinement for Infomap.

    Mirrors Rust ``infomap_refinement`` (infomap.rs:428-520).

    Within each community, runs local-move to split poorly connected nodes
    and enforces connectivity.
    """
    n = working.len()
    two_m = 2.0 * working.total_weight

    # Group nodes by parent community
    by_parent: dict[int, list[int]] = defaultdict(list)
    for u, c in enumerate(partition):
        by_parent[c].append(u)

    parents = sorted(by_parent.items(), key=lambda kv: kv[0])

    # Label bases
    label_base: list[int] = []
    cursor = 0
    for _, members in parents:
        label_base.append(cursor)
        cursor += len(members)
    total_labels = max(cursor, 1)

    # Refine each parent
    per_parent_labels: list[list[int]] = []

    for idx, (parent, members) in enumerate(parents):
        base = label_base[idx]
        parent_total = sum(working.degree[i] for i in range(n) if partition[i] == parent)

        if len(members) <= 1:
            per_parent_labels.append([base])
            continue

        member_set = set(members)

        # Local refined labels
        refined: dict[int, int] = {u: base + i for i, u in enumerate(members)}
        label_degree: dict[int, float] = {
            base + i: working.degree[u] for i, u in enumerate(members)
        }

        for _ in range(options.max_passes):
            moved = False
            for u in members:
                current_label = refined[u]
                node_degree = working.degree[u]
                if node_degree == 0.0:
                    continue

                weight_to_comm: dict[int, float] = defaultdict(float)
                for v_idx, w in working.adj[u]:
                    v = v_idx
                    if v not in member_set:
                        continue
                    weight_to_comm[refined[v]] += w

                best = current_label
                best_gain = options.min_modularity_gain

                # Stay gain
                stay_weight = weight_to_comm.get(current_label, 0.0)
                if stay_weight > best_gain:
                    best_gain = stay_weight
                    best = current_label

                for candidate, edge_weight in weight_to_comm.items():
                    if candidate == current_label:
                        continue

                    # Well-connectedness threshold
                    cand_degree = label_degree.get(candidate, 0.0)
                    threshold = (
                        options.well_connectedness * cand_degree * (parent_total - cand_degree)
                        / (two_m * parent_total)
                    ) if parent_total > 0.0 else 0.0

                    if edge_weight < threshold:
                        continue
                    if edge_weight > best_gain:
                        best_gain = edge_weight
                        best = candidate

                refined[u] = best
                if best != current_label:
                    moved = True

            if not moved:
                break

        per_parent_labels.append([refined[u] for u in members])

    # Assemble global
    result: list[int] = [total_labels] * n
    for idx, (_, members) in enumerate(parents):
        for u, label in zip(members, per_parent_labels[idx], strict=True):
            result[u] = label

    # Enforce connectivity
    enforce_connected(working, result)
    result = densify(result)

    return result


# ── Public API ───────────────────────────────────────────────────────


def detect_communities(
    graph: Graph,
    algorithm: str = "infomap",
) -> dict[str, Any]:
    """Detect communities using Infomap algorithm.

    Returns community assignments, quality metrics, and cross-community edges.
    """
    assignments = infomap_with_options(graph, CommunityOptions())
    set_communities: dict[int, set[int]] = {}
    for node_id, community_id in assignments.items():
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
