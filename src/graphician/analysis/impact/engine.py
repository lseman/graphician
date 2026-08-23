"""Impact analysis engine: reverse graph walk and scoring."""

from __future__ import annotations

import heapq
from typing import Any

from ...core.edge import EdgeKind
from ...core.graph import Graph
from ...core.id import NodeId
from ...core.node import Node, NodeKind
from .types import ImpactHit, ImpactQuery


def find_impact(graph: Graph, query: ImpactQuery) -> list[ImpactHit]:
    """Walk the reverse graph from *query.seed_id* and rank nodes that
    can reach it.

    Returns hits sorted by score descending (highest impact first).
    """
    heap: list[tuple[float, int, int, tuple[EdgeKind, ...]]] = [
        (0.0, 0, query.seed_id.value, ()),
    ]
    best: dict[int, tuple[float, int, list[EdgeKind]]] = {}

    while heap:
        cost, distance, nid_val, via = heapq.heappop(heap)

        if distance > query.max_hops:
            continue

        seen = best.get(nid_val)
        if seen and seen[0] <= cost:
            continue
        best[nid_val] = (cost, distance, list(via))

        if distance == query.max_hops:
            continue

        for prev, edge in graph.in_neighbors(NodeId(nid_val)):
            pval = prev.value if isinstance(prev, NodeId) else prev
            if pval in best and best[pval][0] <= cost + _impact_cost(edge):
                continue
            new_cost = cost + _impact_cost(edge)
            next_via = via + (edge.kind,)
            heapq.heappush(heap, (new_cost, distance + 1, pval, next_via))

        # A changed symbol can also require coordinated edits in its direct
        # dependencies. Follow only structural forward edges and penalize them
        # more heavily so reverse dependants continue to rank first.
        for next_id, edge in graph.out_neighbors(NodeId(nid_val)):
            forward_cost = _forward_impact_cost(edge)
            if forward_cost is None:
                continue
            next_val = next_id.value if isinstance(next_id, NodeId) else next_id
            new_cost = cost + forward_cost
            if next_val in best and best[next_val][0] <= new_cost:
                continue
            next_via = via + (edge.kind,)
            heapq.heappush(heap, (new_cost, distance + 1, next_val, next_via))

    hits: list[ImpactHit] = []
    for nid_val, (cost, distance, via) in best.items():
        if nid_val == query.seed_id.value:
            continue
        node = graph.node(NodeId(nid_val))
        if node is None:
            continue
        score = _compute_score(node, NodeId(nid_val), cost, distance)
        hits.append(ImpactHit(
            id=NodeId(nid_val),
            score=score,
            distance=distance,
            via=via,
            node=node,
        ))

    hits.sort(key=lambda h: (-h.score, h.distance))
    return hits[:query.limit]


def _impact_cost(edge: Any) -> float:
    """Cost to traverse an edge in reverse impact walk."""
    base = {
        EdgeKind.CALLS: 1.0,
        EdgeKind.DEFINES: 1.25,
        EdgeKind.IMPORTS: 1.6,
        EdgeKind.DEPENDS_ON: 1.6,
        EdgeKind.INHERITS: 0.75,
        EdgeKind.IMPLEMENTS: 0.75,
        EdgeKind.DATA_FLOW: 0.8,
        EdgeKind.READS_WRITES: 0.9,
        EdgeKind.TESTED_BY: 1.1,
        EdgeKind.MEMBER_OF: 5.0,
        EdgeKind.ENTRY_OF: 5.0,
        EdgeKind.DESCRIBES: 1.2,
        EdgeKind.DOCUMENTED_BY: 1.2,
        EdgeKind.MENTIONS: 1.8,
        EdgeKind.ILLUSTRATES: 1.8,
        EdgeKind.SIMILAR_TO: 2.0,
        EdgeKind.RATIONALE_FOR: 2.0,
    }.get(edge.kind, 1.5)
    return base / max(edge.confidence.score(), 0.05)


def _forward_impact_cost(edge: Any) -> float | None:
    """Cost for conservative forward traversal from a changed symbol."""
    base = {
        EdgeKind.CALLS: 2.25,
        EdgeKind.IMPORTS: 2.5,
        EdgeKind.DEPENDS_ON: 2.5,
        EdgeKind.INHERITS: 1.75,
        EdgeKind.IMPLEMENTS: 1.75,
        EdgeKind.DATA_FLOW: 2.0,
        EdgeKind.READS_WRITES: 2.0,
    }.get(edge.kind)
    if base is None:
        return None
    return base / max(edge.confidence.score(), 0.05)


def _node_kind_boost(kind: NodeKind) -> float:
    """Impact ranking boost for node kinds."""
    return {
        NodeKind.FUNCTION: 1.3,
        NodeKind.METHOD: 1.3,
        NodeKind.CLASS: 1.3,
        NodeKind.TYPE: 1.3,
        NodeKind.TRAIT: 1.2,
        NodeKind.IMPL: 1.2,
        NodeKind.FILE: 0.95,
        NodeKind.MODULE: 0.95,
        NodeKind.DOCUMENT: 0.85,
        NodeKind.SECTION: 0.85,
        NodeKind.CONCEPT: 0.85,
        NodeKind.DIAGRAM: 0.75,
        NodeKind.IMAGE: 0.75,
        NodeKind.VARIABLE: 0.7,
        NodeKind.COMMIT: 0.7,
        NodeKind.AUTHOR: 0.7,
        NodeKind.HYPEREDGE: 0.7,
        NodeKind.FLOW: 0.4,
        NodeKind.PACKAGE: 0.95,
    }.get(kind, 0.8)


def _compute_score(node: Node, nid: NodeId, cost: float, distance: int) -> float:
    """Compute impact score from cost and node kind."""
    return _node_kind_boost(node.kind) / (1.0 + cost)


# ── Legacy API ─────────────────────────────────────────────────────

def compute_impact(
    graph: Graph,
    target_qname: str,
    max_hops: int = 4,
    limit: int = 25,
) -> dict[str, Any]:
    """Compute impact of changes to a target symbol.

    BFS from target, rank reachable nodes by blast radius signals.

    Legacy API — prefers ``ImpactQuery`` + ``find_impact`` for new code.
    """
    target_id = graph.find_by_qname(target_qname)
    if target_id is None:
        return {"error": f"Symbol not found: {target_qname}", "results": []}

    query = ImpactQuery(seed_id=target_id, max_hops=max_hops, limit=limit)
    hits = find_impact(graph, query)

    # Build path info
    path_info: dict[int, list[str]] = {}
    for hit in hits:
        path_info[hit.id.value] = [hit.node.qualified_name]

    return {
        "target": target_qname,
        "max_hops": max_hops,
        "total_affected": len(hits),
        "results": [
            {
                "node_id": h.id.value,
                "qualified_name": h.node.qualified_name,
                "kind": h.node.kind.value,
                "name": h.node.name,
                "score": round(h.score, 4),
                "distance": h.distance,
                "via": [e.value for e in h.via],
                "path": path_info.get(h.id.value, []),
            }
            for h in hits
        ],
    }
