"""Shared infrastructure for community detection algorithms.

Mirrors the Rust ``mod.rs`` module: WorkingGraph, CommunityOptions,
edge kind weights, graph aggregation, connectivity enforcement,
and relabeling helpers.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from ...core.edge import Confidence, EdgeKind
from ...core.graph import Graph
from ...core.id import NodeId


# ── Options ──────────────────────────────────────────────────────────


@dataclass
class CommunityOptions:
    """Options for community detection algorithms.

    Mirrors Rust ``CommunityOptions`` (mod.rs:14-24).
    """

    resolution: float = 1.0
    max_passes: int = 50
    max_levels: int = 10
    well_connectedness: float = 1.0
    min_modularity_gain: float = 1e-7


# ── Edge kind weights ────────────────────────────────────────────────


def edge_kind_weight(kind: EdgeKind) -> float:
    """Weight for an edge kind in community detection.

    Mirrors Rust ``edge_kind_weight`` (mod.rs:195-207).
    Ambiguous edges get weight 0.15 (handled in WorkingGraph).
    """
    return {
        EdgeKind.INHERITS: 1.25,
        EdgeKind.IMPLEMENTS: 1.25,
        EdgeKind.DEFINES: 0.7,
        EdgeKind.CALLS: 0.55,
        EdgeKind.DATA_FLOW: 0.65,
        EdgeKind.READS_WRITES: 0.85,
        EdgeKind.MENTIONS: 0.75,
        EdgeKind.DESCRIBES: 0.75,
        EdgeKind.DOCUMENTED_BY: 0.75,
        EdgeKind.TESTED_BY: 0.6,
        EdgeKind.IMPORTS: 0.45,
        EdgeKind.DEPENDS_ON: 0.45,
        EdgeKind.SIMILAR_TO: 0.55,
        EdgeKind.RATIONALE_FOR: 0.55,
        EdgeKind.ILLUSTRATES: 0.55,
        EdgeKind.MEMBER_OF: 0.1,
        EdgeKind.ENTRY_OF: 0.1,
    }.get(kind, 0.55)


# ── WorkingGraph ─────────────────────────────────────────────────────


class WorkingGraph:
    """Internal working graph for community algorithms.

    Each index represents a "super-node" that may contain multiple
    original nodes. Supports multi-level aggregation.

    Mirrors Rust ``WorkingGraph`` (mod.rs:28-46).
    """

    __slots__ = ("members", "adj", "self_loop", "degree", "total_weight")

    def __init__(
        self,
        members: list[list[NodeId]],
        adj: list[list[tuple[int, float]]],
        self_loop: list[float],
        degree: list[float],
        total_weight: float,
    ) -> None:
        self.members = members
        self.adj = adj
        self.self_loop = self_loop
        self.degree = degree
        self.total_weight = total_weight

    @classmethod
    def from_graph(cls, graph: Graph) -> "WorkingGraph":
        """Build a WorkingGraph from an Ariadne Graph.

        Mirrors Rust ``WorkingGraph::from_graph`` (mod.rs:48-97).
        """
        nodes: list[NodeId] = [nid for nid, _ in graph.nodes()]
        n = len(nodes)

        # Initialize: each original node is its own super-node
        members: list[list[NodeId]] = [[nid] for nid in nodes]
        adj_map: dict[int, dict[int, float]] = defaultdict(dict)
        self_loop: list[float] = [0.0] * n

        for _, src, dst, edge in graph.edges():
            try:
                src_idx = nodes.index(src)
            except ValueError:
                continue
            try:
                dst_idx = nodes.index(dst)
            except ValueError:
                continue

            weight: float
            if edge.confidence == Confidence.AMBIGUOUS:
                weight = 0.15
            else:
                weight = edge_kind_weight(edge.kind)

            adj_map[src_idx][dst_idx] = adj_map[src_idx].get(dst_idx, 0.0) + weight
            if src == dst:
                self_loop[src_idx] += weight * 0.5

        # Build sorted adjacency: sort by neighbor index for deterministic
        # floating-point summation order.
        adj: list[list[tuple[int, float]]] = []
        for u in range(n):
            neighbors = adj_map.get(u, {})
            adj.append(sorted(neighbors.items(), key=lambda kv: kv[0]))

        degree: list[float] = [
            sum(w for _, w in adj[u]) + 2.0 * self_loop[u]
            for u in range(n)
        ]
        total_weight = sum(degree) / 2.0

        return cls(members, adj, self_loop, degree, total_weight)

    def len(self) -> int:
        return len(self.members)

    def original_nodes(self):
        """Iterator over all original NodeIds."""
        for members in self.members:
            yield from members


# ── Graph operations ─────────────────────────────────────────────────


def identity_labels(nodes) -> dict:
    """Map each node to its index as its own community."""
    return {nid: i for i, nid in enumerate(nodes)}


def densify(labels: list[int]) -> list[int]:
    """Relabel community labels to consecutive integers [0, k).

    Mirrors Rust ``densify`` (mod.rs:124-131).
    """
    mapping: dict[int, int] = {}
    next_id = 0
    result: list[int] = []
    for l in labels:
        if l not in mapping:
            mapping[l] = next_id
            next_id += 1
        result.append(mapping[l])
    return result


def relabel(labels: dict) -> dict:
    """Stable relabel of community values to [0, k).

    Mirrors Rust ``relabel`` (mod.rs:113-121).
    """
    mapping: dict[int, int] = {}
    next_id = 0
    result: dict = {}
    for nid, l in labels.items():
        if l not in mapping:
            mapping[l] = next_id
            next_id += 1
        result[nid] = mapping[l]
    return result


def aggregate(prev: WorkingGraph, partition: list[int]) -> WorkingGraph:
    """Aggregate graph by partition into super-nodes.

    Mirrors Rust ``aggregate`` (mod.rs:133-176).

    Args:
        prev: The working graph at the current level.
        partition: Current community assignment (dense labels).

    Returns:
        A new WorkingGraph with super-nodes.
    """
    dense = densify(partition)
    new_n = max(dense) + 1 if dense else 0

    new_members: list[list[NodeId]] = [[] for _ in range(new_n)]
    for u, members in enumerate(prev.members):
        new_members[dense[u]].extend(members)

    adj_map: dict[int, dict[int, float]] = defaultdict(dict)
    self_loop: list[float] = [0.0] * new_n

    for u in range(prev.len()):
        cu = dense[u]
        self_loop[cu] += prev.self_loop[u]
        for v, w in prev.adj[u]:
            cv = dense[v]
            if cu == cv:
                self_loop[cu] += w * 0.5
            else:
                adj_map[cu][cv] = adj_map[cu].get(cv, 0.0) + w

    adj: list[list[tuple[int, float]]] = []
    for u in range(new_n):
        neighbors = adj_map.get(u, {})
        adj.append(sorted(neighbors.items(), key=lambda kv: kv[0]))

    degree: list[float] = [
        sum(w for _, w in adj[u]) + 2.0 * self_loop[u]
        for u in range(new_n)
    ]
    total_weight = sum(degree) / 2.0

    return WorkingGraph(new_members, adj, self_loop, degree, total_weight)


def enforce_connected(working: WorkingGraph, labels: list[int]) -> list[int]:
    """Ensure every community is connected via undirected BFS.

    Mirrors Rust ``enforce_connected`` (mod.rs:209-272).

    Builds an undirected neighbor list (since working.adj is directed)
    and splits disconnected communities via BFS.

    Uses an unseen_set to prevent re-visiting nodes during BFS,
    matching the Rust implementation exactly.

    Args:
        working: The working graph.
        labels: Current community assignment to modify in place.

    Returns:
        The modified labels list (same object, mutated).
    """
    n = working.len()

    # Build undirected neighbor list
    undirected: list[list[int]] = [[] for _ in range(n)]
    for u in range(n):
        for v, _ in working.adj[u]:
            undirected[u].append(v)
            undirected[v].append(u)

    # Group nodes by label
    by_label: dict[int, list[int]] = defaultdict(list)
    for u, c in enumerate(labels):
        by_label[c].append(u)

    # Sort for deterministic assignment
    label_groups = sorted(by_label.items(), key=lambda kv: kv[0])

    next_label = max(labels) + 1 if labels else 0
    new_labels: list[int | None] = [None] * n

    for _, members in label_groups:
        member_set = set(members)
        # Track unseen nodes for BFS (mirrors Rust's unseen_set)
        unseen_set = set(members)

        first_component = True
        while unseen_set:
            # Find first unseen node
            start = min(unseen_set)

            component_label = labels[start] if first_component else next_label
            if not first_component:
                next_label += 1
            first_component = False

            # BFS from start within this community
            queue: deque[int] = deque([start])
            unseen_set.discard(start)

            while queue:
                u = queue.popleft()
                new_labels[u] = component_label
                for v in undirected[u]:
                    if v in member_set and v in unseen_set:
                        unseen_set.discard(v)
                        new_labels[v] = component_label
                        queue.append(v)

    # Apply new labels
    for u in range(n):
        if new_labels[u] is not None:
            labels[u] = new_labels[u]

    return labels
