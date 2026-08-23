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
    edge_kind_weight,
    aggregate,
    densify,
    enforce_connected,
)
from .louvain import (
    detect_communities,
    louvain,
    louvain_with_options,
)
from .leiden import (
    leiden,
    leiden_with_options,
)
from .nodes import (
    find_bridge_nodes,
    find_hub_nodes,
    find_god_nodes,
    compute_centrality,
    is_rank_noise,
)
from .quality import (
    CommunityQuality,
    community_cohesion,
    community_quality,
    LOW_COHESION_THRESHOLD,
)
from .gaps import knowledge_gaps
from .split import split_oversized
from .utils import _find_community, _to_networkx

__all__ = [
    "detect_communities",
    "louvain",
    "louvain_with_options",
    "leiden",
    "leiden_with_options",
    "find_bridge_nodes",
    "find_hub_nodes",
    "find_god_nodes",
    "compute_centrality",
    "is_rank_noise",
    "CommunityOptions",
    "WorkingGraph",
    "community_cohesion",
    "community_quality",
    "CommunityQuality",
    "LOW_COHESION_THRESHOLD",
    "knowledge_gaps",
    "split_oversized",
    "_find_community",
    "_to_networkx",
    "aggregate",
    "densify",
    "enforce_connected",
]
