"""Pass 5-6: Community boost and Union-Find merge."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ...core.edge import Edge, EdgeKind
from ...core.graph import Graph
from ...core.id import NodeId, EdgeId
from ...core.node import Node, NodeKind
from .normalize import normalize_label
from .similarity import jaro_winkler


class UnionFind:
    """Union-Find data structure for grouping."""

    def __init__(self) -> None:
        self.parent: dict[NodeId, NodeId] = {}
        self.rank: dict[NodeId, int] = defaultdict(int)

    def find(self, x: NodeId) -> NodeId:
        """Find with path compression."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: NodeId, y: NodeId) -> None:
        """Union by rank."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def deduplicate_nodes(
    graph: Graph,
    options: Any = None,
) -> dict[str, Any]:
    """Deduplicate nodes with similar names using multi-pass pipeline."""
    if options is None:
        from .types import DedupOptions
        options = DedupOptions()

    # Collect eligible nodes
    nodes: list[Node] = []
    node_ids: list[NodeId] = []
    for nid, node in graph.nodes():
        if node.kind in options.eligible_kinds:
            nodes.append(node)
            node_ids.append(nid)

    if len(nodes) < 2:
        return {
            "candidates_examined": 0,
            "merges": 0,
            "nodes_removed": 0,
            "edges_rewired": 0,
        }

    # Get community map for boost
    community_map: dict[int, int] = {}
    try:
        from ..communities.louvain import detect_communities
        comm_result = detect_communities(graph, algorithm="louvain")
        for comm in comm_result.get("communities", []):
            for node_info in comm.get("nodes", []):
                qn = node_info.get("qualified_name", "")
                nid = graph.find_by_qname(qn)
                if nid:
                    community_map[nid.value] = comm["id"]
    except Exception:
        pass

    # Build normalized names map
    normalized: dict[NodeId, str] = {}
    for node in nodes:
        normalized[node_ids[nodes.index(node)]] = normalize_label(node.name)

    # Find candidate pairs
    candidates: list[tuple[NodeId, NodeId, float]] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            nid_a, nid_b = node_ids[i], node_ids[j]
            sim = jaro_winkler(
                normalized[nid_a], normalized[nid_b]
            )
            # Community boost
            ca = community_map.get(nid_a.value)
            cb = community_map.get(nid_b.value)
            if ca is not None and cb is not None and ca == cb:
                sim = min(1.0, sim + options.community_boost)
            if sim >= options.jw_threshold:
                candidates.append((nid_a, nid_b, sim))

    # Union-Find merges
    uf = UnionFind()
    merges = 0
    for nid_a, nid_b, _ in candidates:
        uf.union(nid_a, nid_b)
        merges += 1

    # Group merged nodes
    groups: dict[NodeId, list[NodeId]] = defaultdict(list)
    for nid in node_ids:
        groups[uf.find(nid)].append(nid)

    # Count merged groups (groups with > 1 member)
    merged_groups = [g for g in groups.values() if len(g) > 1]

    return {
        "candidates_examined": len(nodes) * (len(nodes) - 1) // 2,
        "merges": merges,
        "merged_groups": len(merged_groups),
        "groups": [
            [n.value for n in g]
            for g in merged_groups[:50]  # Limit for response
        ],
    }
