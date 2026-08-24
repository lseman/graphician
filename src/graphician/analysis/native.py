"""Adapter from Graphician's Python graph to the optional native snapshot."""

from __future__ import annotations

from .._extract import HAS_RUST, NativeGraph
from ..core.graph import Graph


def native_graph(graph: Graph):
    """Build a native snapshot, or return ``None`` without the extension."""
    if not HAS_RUST or NativeGraph is None:
        return None
    return NativeGraph(
        [node_id.value for node_id, _ in graph.nodes()],
        [
            (source.value, target.value, edge.kind.value, edge.confidence.value)
            for _, source, target, edge in graph.edges()
        ],
    )
