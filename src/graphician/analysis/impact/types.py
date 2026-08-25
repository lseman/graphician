"""Types for impact analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...core.edge import EdgeKind
from ...core.id import NodeId
from ...core.node import Node


@dataclass
class ImpactQuery:
    """Parameters for impact analysis."""
    seed_id: NodeId
    max_hops: int = 4
    limit: int = 25


@dataclass
class ImpactHit:
    """A single impact analysis hit."""
    id: NodeId
    score: float
    distance: int
    via: list[EdgeKind]
    node: Node = field(repr=False)


@dataclass
class ImpactResult:
    """A single impact analysis result."""
    node_id: NodeId
    node: Node
    score: float
    reachable_count: int
    hop_distance: int
    reasons: list[str] = field(default_factory=list)
