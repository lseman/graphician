"""Community quality metrics and cohesion analysis."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

LOW_COHESION_THRESHOLD = 0.15


@dataclass
class CommunityQuality:
    """Aggregate quality metrics for a community partition."""
    community_count: int = 0
    singleton_count: int = 0
    min_size: int = 0
    max_size: int = 0
    mean_size: float = 0.0
    score: float = 0.0
    disconnected_communities: int = 0
    mean_conductance: float = 0.0
    max_conductance: float = 0.0
    mean_cohesion: float = 0.0
    low_cohesion_communities: int = 0


def community_cohesion(
    graph,
    communities: dict[int, int],
) -> dict[int, float]:
    """Compute cohesion for each community.

    Cohesion = actual internal edges / possible internal edges.
    A community with n nodes has n*(n-1)/2 possible directed edges.
    Returns {community_id: cohesion_score}.
    """
    # Count internal edges per community
    internal: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for _, src, dst, _ in graph.edges():
        if src == dst:
            continue
        sc = communities.get(src.value)
        dc = communities.get(dst.value)
        if sc is not None and dc is not None and sc == dc:
            pair = (min(src.value, dst.value), max(src.value, dst.value))
            internal[sc].add(pair)

    # Count sizes
    sizes: dict[int, int] = defaultdict(int)
    for cid in communities.values():
        sizes[cid] += 1

    result: dict[int, float] = {}
    for cid, n in sizes.items():
        if n <= 1:
            result[cid] = 1.0
        else:
            actual = len(internal.get(cid, set()))
            possible = n * (n - 1) // 2
            result[cid] = actual / possible if possible > 0 else 0.0

    return result


def community_quality(
    graph,
    communities: dict[int, int],
    resolution: float = 1.0,
) -> CommunityQuality:
    """Compute aggregate quality metrics for a community partition.

    Args:
        graph: The graph to analyze.
        communities: Mapping from node_id -> community_id.
        resolution: Resolution parameter for the quality score.

    Returns:
        CommunityQuality with aggregate metrics.
    """
    cohesion = community_cohesion(graph, communities)

    # Count sizes
    sizes: dict[int, int] = defaultdict(int)
    for cid in communities.values():
        sizes[cid] += 1

    community_count = len(sizes)
    sizes_list = list(sizes.values())
    min_size = min(sizes_list) if sizes_list else 0
    max_size = max(sizes_list) if sizes_list else 0
    mean_size = sum(sizes_list) / community_count if community_count > 0 else 0.0
    singleton_count = sum(1 for s in sizes_list if s == 1)

    # Quality score: normalized silhouette-like metric
    score = 0.0
    if community_count > 0:
        total_nodes = graph.node_count()
        if total_nodes > 0:
            sum_sq = sum((s / total_nodes) ** 2 for s in sizes_list)
            score = (sum_sq * resolution) ** 0.5

    # Conductance per community
    conductances: list[float] = []
    for cid, size in sizes.items():
        if size <= 1:
            conductances.append(0.0)
            continue
        internal_edges = 0
        external_edges = 0
        for _, src, dst, _ in graph.edges():
            sc = communities.get(src.value)
            dc = communities.get(dst.value)
            if sc == cid and dc == cid:
                internal_edges += 1
            elif (sc == cid) != (dc == cid):
                external_edges += 1
        total = internal_edges + external_edges
        conductances.append(external_edges / total if total > 0 else 0.0)

    mean_conductance = (
        sum(conductances) / len(conductances) if conductances else 0.0
    )
    max_conductance = max(conductances) if conductances else 0.0

    # Low cohesion communities
    low_cohesion = sum(
        1 for cid, c in cohesion.items()
        if sizes.get(cid, 0) > 1 and c < LOW_COHESION_THRESHOLD
    )

    # Mean cohesion (only for communities with > 1 node)
    cohesion_vals = [
        c for cid, c in cohesion.items() if sizes.get(cid, 0) > 1
    ]
    mean_cohesion = (
        sum(cohesion_vals) / len(cohesion_vals) if cohesion_vals else 0.0
    )

    return CommunityQuality(
        community_count=community_count,
        singleton_count=singleton_count,
        min_size=min_size,
        max_size=max_size,
        mean_size=round(mean_size, 2),
        score=round(score, 4),
        disconnected_communities=0,
        mean_conductance=round(mean_conductance, 4),
        max_conductance=round(max_conductance, 4),
        mean_cohesion=round(mean_cohesion, 4),
        low_cohesion_communities=low_cohesion,
    )
