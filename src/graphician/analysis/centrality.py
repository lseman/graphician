"""Centrality metrics for the code graph.

``pagerank`` runs a weighted random-walk-with-damping iteration on the
directed graph. Edge kind and confidence shape transition probability, and
``personalized_pagerank`` biases the teleport distribution around supplied
seed nodes.

Edges with :attr:`Confidence.AMBIGUOUS` are skipped: those are the
unresolved call-site placeholders pointing at ``call::<name>`` synthetic
nodes, and including them distorts rank toward common function names like
``new``, ``len``, ``clone``.

:attr:`is_rank_noise` identifies nodes that inflate god-node rankings
without representing a real symbol: file containers, synthetic flow
nodes, and unresolved call placeholders.

Optimized versions use numpy/numba for performance-critical paths.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.edge import Confidence, EdgeKind
from ..core.graph import Graph
from ..core.id import NodeId
from ..core.node import Node, NodeKind
from .adjacency import AdjacencyConfig, build_adjacency_matrix

# ── Edge kind weights ────────────────────────────────────────────────

def _edge_weight(kind: EdgeKind) -> float:
    """Transition weight for an edge kind."""
    weights: dict[EdgeKind, float] = {
        EdgeKind.DEFINES: 0.7,
        EdgeKind.CALLS: 1.0,
        EdgeKind.IMPORTS: 0.55,
        EdgeKind.DEPENDS_ON: 0.55,
        EdgeKind.INHERITS: 1.15,
        EdgeKind.IMPLEMENTS: 1.15,
        EdgeKind.DATA_FLOW: 0.8,
        EdgeKind.READS_WRITES: 0.9,
        EdgeKind.MENTIONS: 0.75,
        EdgeKind.DESCRIBES: 0.75,
        EdgeKind.DOCUMENTED_BY: 0.75,
        EdgeKind.SIMILAR_TO: 0.6,
        EdgeKind.RATIONALE_FOR: 0.6,
        EdgeKind.ILLUSTRATES: 0.6,
        # Production→test edge: low weight so tests don't pull rank away.
        EdgeKind.TESTED_BY: 0.3,
        # Flow bookkeeping — overlay-only; don't let it skew rank.
        EdgeKind.MEMBER_OF: 0.05,
        EdgeKind.ENTRY_OF: 0.05,
    }
    return weights.get(kind, 0.5)


# ── PageRank ─────────────────────────────────────────────────────────

def pagerank(
    graph: Graph,
    damping: float = 0.85,
    iterations: int = 30,
) -> dict[NodeId, float]:
    """Run PageRank on *graph* and return a ``{node_id: rank}`` mapping.

    Uses edge-kind-dependent weights and skips ambiguous (unresolved)
    placeholder edges.
    Uses numba-accelerated algorithm when available.
    """
    return _weighted_pagerank(graph, damping, iterations, {})


def personalized_pagerank(
    graph: Graph,
    seeds: list[tuple[NodeId, float]],
    damping: float = 0.85,
    iterations: int = 30,
) -> dict[NodeId, float]:
    """Run personalized PageRank biased toward *seeds*."""
    total = sum(w for _, w in seeds)
    if total > 0.0:
        personalization: dict[NodeId, float] = {
            nid: max(w, 0.0) / total for nid, w in seeds
        }
    else:
        personalization = {}
    return _weighted_pagerank(graph, damping, iterations, personalization)


def _weighted_pagerank(
    graph: Graph,
    damping: float,
    iterations: int,
    personalization: dict[NodeId, float],
) -> dict[NodeId, float]:
    from .native import native_graph

    snapshot = native_graph(graph)
    if snapshot is not None:
        seeds = [(nid.value, weight) for nid, weight in personalization.items()]
        ranks = snapshot.pagerank(damping, iterations, seeds or None)
        return {NodeId(node_id): float(rank) for node_id, rank in ranks.items()}

    nodes = [nid for nid, _ in graph.nodes()]
    n = len(nodes)
    if n == 0:
        return {}

    # Build adjacency matrix with weighted edges
    config = AdjacencyConfig(
        weights={
            EdgeKind.DEFINES: 0.7,
            EdgeKind.CALLS: 1.0,
            EdgeKind.IMPORTS: 0.55,
            EdgeKind.DEPENDS_ON: 0.55,
            EdgeKind.INHERITS: 1.15,
            EdgeKind.IMPLEMENTS: 1.15,
            EdgeKind.DATA_FLOW: 0.8,
            EdgeKind.READS_WRITES: 0.9,
            EdgeKind.MENTIONS: 0.75,
            EdgeKind.DESCRIBES: 0.75,
            EdgeKind.DOCUMENTED_BY: 0.75,
            EdgeKind.SIMILAR_TO: 0.6,
            EdgeKind.RATIONALE_FOR: 0.6,
            EdgeKind.ILLUSTRATES: 0.6,
            EdgeKind.TESTED_BY: 0.3,
            EdgeKind.MEMBER_OF: 0.05,
            EdgeKind.ENTRY_OF: 0.05,
        },
        min_confidence=0.0,
        exclude_ambiguous=True,
    )

    row_ptr, col_idx, edge_weight, node_list, node_to_idx = build_adjacency_matrix(graph, config)

    # Compute out-degree
    out_degree = np.zeros(n, dtype=np.float64)
    for u in range(n):
        start = row_ptr[u]
        end = row_ptr[u + 1]
        out_degree[u] = edge_weight[start:end].sum()

    # Build personalization vector if needed
    personalization_vec = None
    has_personalization = bool(personalization)
    if has_personalization:
        personalization_vec = np.zeros(n, dtype=np.float64)
        for nid, weight in personalization.items():
            idx = node_to_idx.get(nid)
            if idx is not None:
                personalization_vec[idx] = max(weight, 0.0)
        # Normalize
        total = personalization_vec.sum()
        if total > 0:
            personalization_vec /= total
        else:
            personalization_vec = np.ones(n, dtype=np.float64) / n

    # Use numba-accelerated PageRank if available
    from .numba_pagerank import pagerank_csr as _pagerank_csr
    try:
        ranks_arr = _pagerank_csr(
            row_ptr, col_idx, edge_weight, out_degree,
            damping, iterations, personalization_vec,
        )
        return {node_list[i]: float(ranks_arr[i]) for i in range(n)}
    except Exception:
        # Fallback to pure Python
        pass

    # Pure Python fallback (original implementation)
    node_index = {nid: idx for idx, nid in enumerate(nodes)}
    init = 1.0 / n
    ranks = [init] * n

    transitions = _weighted_transitions(graph, nodes, node_index)

    for _ in range(iterations):
        if has_personalization:
            personalization_vec = [
                personalization.get(nodes[i], 0.0) for i in range(n)
            ]
            next_ranks = [
                (1.0 - damping) * p for p in personalization_vec
            ]
        else:
            uniform = 1.0 / n
            next_ranks = [(1.0 - damping) * uniform] * n

        dangling_mass = 0.0
        for idx, out_edges in enumerate(transitions):
            if not out_edges["edges"]:
                dangling_mass += ranks[idx]
                continue
            for neighbor_idx, weight in out_edges["edges"]:
                next_ranks[neighbor_idx] += (
                    damping * ranks[idx] * weight / out_edges["total"]
                )

        for idx in range(n):
            p = personalization.get(nodes[idx], 0.0) if has_personalization else 1.0 / n
            next_ranks[idx] += damping * dangling_mass * p

        ranks = next_ranks

    return dict(zip(nodes, ranks, strict=True))


# ── Weighted transitions ─────────────────────────────────────────────

def _weighted_transitions(
    graph: Graph,
    nodes: list[NodeId],
    node_index: dict[NodeId, int],
) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    for nid in nodes:
        edges: list[tuple[int, float]] = []
        for neighbor, edge in graph.out_neighbors(nid):
            if edge.confidence == Confidence.AMBIGUOUS:
                continue
            if neighbor not in node_index:
                continue
            idx = node_index[neighbor]
            w = _edge_weight(edge.kind) * max(edge.confidence.score(), 0.05)
            edges.append((idx, w))
        total = sum(w for _, w in edges)
        transitions.append({"edges": edges, "total": total})
    return transitions


# ── Noise detection ──────────────────────────────────────────────────

def is_rank_noise(node: Node) -> bool:
    """Return ``True`` for nodes that inflate god-node rankings without
    representing a real symbol.

    Filters file containers, synthetic flow nodes, hyperedges, and
    unresolved call placeholders (``call::<name>``).
    """
    if node.kind in (NodeKind.FILE, NodeKind.FLOW):
        return True
    return node.qualified_name.startswith("call::")
