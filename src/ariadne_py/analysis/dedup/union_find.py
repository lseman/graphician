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
    """Deduplicate nodes with similar names using multi-pass pipeline.

    Pipeline:
    1. Normalize labels and filter by entropy gate.
    2. Use MinHash/LSH to find candidate pairs efficiently.
    3. Apply Jaro-Winkler on candidates with community boost.
    4. Union-Find to merge groups.
    """
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

    # Build normalized names map
    normalized: dict[NodeId, str] = {}
    for node in nodes:
        normalized[node_ids[nodes.index(node)]] = normalize_label(node.name)

    # Entropy gate: filter out low-entropy (likely noise) labels
    eligible_ids: set[NodeId] = set()
    for nid, name in normalized.items():
        if passes_entropy_gate(name, options.entropy_gate):
            eligible_ids.add(nid)

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

    # Step 1: MinHash/LSH candidate pair generation
    lsh_candidates = lsh_candidate_pairs(
        [n for n, nid in zip(nodes, node_ids) if nid in eligible_ids],
        [nid for nid in node_ids if nid in eligible_ids],
        options,
    )

    # Step 2: Jaro-Winkler refinement on LSH candidates
    candidates: list[tuple[NodeId, NodeId, float]] = []
    seen_ids: set[NodeId] = set(nid for nid in node_ids if nid in eligible_ids)

    for id_a, id_b, _jaccard in lsh_candidates:
        if id_a not in eligible_ids or id_b not in eligible_ids:
            continue
        sim = jaro_winkler(normalized[id_a], normalized[id_b])
        # Community boost
        ca = community_map.get(id_a.value)
        cb = community_map.get(id_b.value)
        if ca is not None and cb is not None and ca == cb:
            sim = min(1.0, sim + options.community_boost)
        if sim >= options.jw_threshold:
            candidates.append((id_a, id_b, sim))

    # Step 3: Union-Find merges
    uf = UnionFind()
    merges = 0
    for nid_a, nid_b, _ in candidates:
        uf.union(nid_a, nid_b)
        merges += 1

    # Group merged nodes
    groups: dict[NodeId, list[NodeId]] = defaultdict(list)
    for nid in seen_ids:
        groups[uf.find(nid)].append(nid)

    # Count merged groups (groups with > 1 member)
    merged_groups = [g for g in groups.values() if len(g) > 1]

    return {
        "candidates_examined": len(lsh_candidates) + len(candidates),
        "merges": merges,
        "nodes_removed": sum(len(g) - 1 for g in merged_groups),
        "edges_rewired": 0,
        "merged_groups": len(merged_groups),
        "groups": [
            [n.value for n in g]
            for g in merged_groups[:50]  # Limit for response
        ],
    }
