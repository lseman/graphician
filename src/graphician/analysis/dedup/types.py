"""Types for deduplication pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...core.node import NodeKind


@dataclass
class DedupOptions:
    """Tuning parameters for the deduplication pipeline."""
    entropy_gate: float = 0.5
    shingle_size: int = 3
    num_permutations: int = 64
    num_bands: int = 12
    row_length: int = 5
    jaccard_threshold: float = 0.7
    jw_threshold: float = 0.92
    community_boost: float = 0.05
    eligible_kinds: frozenset[NodeKind] = field(
        default_factory=lambda: frozenset({
            NodeKind.CONCEPT,
            NodeKind.DOCUMENT,
            NodeKind.SECTION,
            NodeKind.DIAGRAM,
            NodeKind.IMAGE,
            NodeKind.HYPEREDGE,
        })
    )


@dataclass
class DedupResult:
    """Result of a deduplication pass."""
    candidates_examined: int = 0
    merges: int = 0
    nodes_removed: int = 0
    edges_rewired: int = 0
