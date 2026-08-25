"""Adapter from Graphician's Python graph to the optional native snapshot."""

from __future__ import annotations

from .._extract import HAS_RUST, NativeGraph
from ..core.graph import Graph


def _enum_value(value: object) -> str:
    """Accept both Graphician enums and their legacy string equivalents."""
    return str(getattr(value, "value", value))


def _edge_state_fingerprint(graph: Graph) -> tuple[int, int]:
    """Fingerprint mutable edge fields consumed by ``NativeGraph``.

    Structural mutations are covered by ``Graph._native_revision``.  Edge
    instances are intentionally exposed by Graphician and callers may edit
    ``kind`` or ``confidence`` directly, so include those fields in a compact
    double fingerprint rather than allowing a cached snapshot to go stale.
    """
    left = 0xCBF29CE484222325
    right = 0x9E3779B185EBCA87
    mask = (1 << 64) - 1
    for edge_id, _source, _target, edge in graph.edges():
        value = hash((edge_id.value, _enum_value(edge.kind), _enum_value(edge.confidence))) & mask
        left = ((left ^ value) * 0x100000001B3) & mask
        right = (right + value + ((right << 6) & mask) + (right >> 2)) & mask
    return left, right


def native_graph(graph: Graph):
    """Return the graph's persistent native snapshot when available.

    The snapshot is rebuilt only after a structural graph mutation or a direct
    edit to an edge's kind/confidence.  This lets successive native algorithms
    share one indexed Rust graph while preserving Graphician's mutable Python
    interface.
    """
    if not HAS_RUST or NativeGraph is None:
        return None
    cache_key = (graph._native_revision, _edge_state_fingerprint(graph))
    if graph._native_snapshot is not None and graph._native_snapshot_key == cache_key:
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
    graph._native_snapshot_key = cache_key
    return snapshot
