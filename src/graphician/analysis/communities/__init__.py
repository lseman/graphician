"""Community detection and structural analysis.

Full Rust-port implementations:
- Multi-level Louvain (modularity optimization with degree tracking)
- Multi-level Leiden (Louvain + refinement + connectivity enforcement)
- Multi-level Infomap (LMDL-based random walks + Leiden refinement)
- PageRank (god nodes)
- Degree centrality (hub nodes)
- Bridge node detection
- Articulation points

Shared infrastructure in core.py: WorkingGraph, edge kind weights,
aggregation, connectivity enforcement.
"""

from __future__ import annotations

from .core import (
    CommunityOptions,
    WorkingGraph,
    aggregate,
    densify,
    edge_kind_weight,
    enforce_connected,
)
from .gaps import knowledge_gaps
from .leiden import (
    leiden,
    leiden_with_options,
)
from .louvain import (
    detect_communities,
    louvain,
    louvain_with_options,
)
from .nodes import (
    compute_centrality,
    find_bridge_nodes,
    find_god_nodes,
    find_hub_nodes,
    is_rank_noise,
)
from .quality import (
    LOW_COHESION_THRESHOLD,
    CommunityQuality,
    community_cohesion,
    community_quality,
)
from .split import split_oversized
from .utils import _find_community, _to_networkx

__all__ = [
    "LOW_COHESION_THRESHOLD",
    "CommunityOptions",
    "CommunityQuality",
    "WorkingGraph",
    "_find_community",
    "_to_networkx",
    "aggregate",
    "community_cohesion",
    "community_quality",
    "compute_centrality",
    "densify",
    "detect_communities",
    "edge_kind_weight",
    "enforce_connected",
    "find_bridge_nodes",
    "find_god_nodes",
    "find_hub_nodes",
    "is_rank_noise",
    "knowledge_gaps",
    "leiden",
    "leiden_with_options",
    "louvain",
    "louvain_with_options",
    "split_oversized",
]
