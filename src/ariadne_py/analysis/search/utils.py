"""Search utility functions."""

from __future__ import annotations

from typing import Any


def _graph_summary(graph) -> dict[str, Any]:
    """Generate a brief summary of the graph structure."""
    node_counts: dict[str, int] = {}
    for _, node in graph.nodes():
        kind = node.kind.value
        node_counts[kind] = node_counts.get(kind, 0) + 1

    edge_count = sum(1 for _ in graph.edges())

    return {
        "total_nodes": graph.node_count(),
        "total_edges": edge_count,
        "node_kinds": node_counts,
    }
