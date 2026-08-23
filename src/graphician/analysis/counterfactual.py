"""Counterfactual reasoning: clone graph with edges removed.

Answers: "if I delete this function / sever this dependency,
what stops being reachable?" Uses actual reachability math
rather than conservative blast-radius approximation.
"""

from __future__ import annotations

from ..core.graph import Graph
from ..core.id import EdgeId


def run_without_edges(graph: Graph, drop: list[EdgeId]) -> Graph:
    """Return a clone of graph with the supplied edges removed.

    The returned graph has its own by_qname index rebuilt from the
    remaining nodes, so symbol resolution works correctly after
    removal.
    """
    clone = graph.clone()
    indices: list[int] = []
    for edge_id in drop:
        idx = clone.edge_index(edge_id)
        if idx is not None:
            indices.append(idx)
    indices.sort(reverse=True)
    for idx in indices:
        clone.remove_edge(EdgeId(idx))
    return clone
