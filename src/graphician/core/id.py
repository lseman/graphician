"""Unique identifiers for graph nodes and edges."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NodeId:
    """Opaque handle for a node in the graph."""
    value: int


@dataclass(frozen=True)
class EdgeId:
    """Opaque handle for an edge in the graph."""
    value: int
