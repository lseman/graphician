"""Adapter from Graphician's Python graph to the optional native snapshot."""

from __future__ import annotations

from .._extract import HAS_RUST, NativeGraph
from ..core.graph import Graph


def _enum_value(value: object) -> str:
    """Accept both Graphician enums and their legacy string equivalents."""
    return str(getattr(value, "value", value))


def native_graph(graph: Graph):
    """Return the graph's persistent native snapshot when available.

    The snapshot is rebuilt only after a structural graph mutation.  This lets
    successive native algorithms share one indexed Rust graph while preserving
    Graphician's mutable Python interface.
    """
    if not HAS_RUST or NativeGraph is None:
        return None
    # Structural mutations (add/remove nodes/edges) bump _native_revision.
    # Edge mutation (kind/confidence edits) is not tracked in the graph,
    # so we accept the risk: callers that mutate edges should rebuild the
    # graph or call _invalidate_native_snapshot themselves.
    if graph._native_snapshot is not None and graph._native_snapshot_key == graph._native_revision:
        return graph._native_snapshot

    snapshot = NativeGraph(
        [node_id.value for node_id, _ in graph.nodes()],
        [
            (
                source.value,
                target.value,
                _enum_value(edge.kind),
                _enum_value(edge.confidence),
            )
            for _, source, target, edge in graph.edges()
        ],
    )
    graph._native_snapshot = snapshot
    graph._native_snapshot_key = graph._native_revision
    return snapshot
