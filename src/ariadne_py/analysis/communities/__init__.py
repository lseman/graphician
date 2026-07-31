"""Community detection and structural analysis.

Uses NetworkX for graph algorithms:
- Louvain community detection
- Leiden refinement
- PageRank (god nodes)
- Degree centrality (hub nodes)
- Bridge node detection
- Articulation points
"""

from __future__ import annotations

from .louvain import (
    detect_communities,
    _louvain,
    _leiden,
    _infomap,
    _modularity,
    _find_cross_community_edges,
)
from .nodes import (
    find_bridge_nodes,
    find_hub_nodes,
    find_god_nodes,
    compute_centrality,
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
    "find_bridge_nodes",
    "find_hub_nodes",
    "find_god_nodes",
    "compute_centrality",
    "community_cohesion",
    "community_quality",
    "CommunityQuality",
    "LOW_COHESION_THRESHOLD",
    "knowledge_gaps",
    "split_oversized",
    "_find_community",
    "_to_networkx",
]
