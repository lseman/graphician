"""Architecture overview and community analysis.

Mirrors the Rust ``architecture.rs`` module.
"""

from __future__ import annotations

from typing import Any

def architecture_overview_json(graph, detail: str = "standard") -> dict[str, Any]:
    """Architecture overview at community level.

    Args:
        graph: The code graph.
        detail: "minimal", "standard", or "full".

    Returns:
        Architecture overview with communities, coupling, bridges, etc.
    """
    from .analysis import articulation_json, bridge_nodes_json, core_json, cycles_json

    communities = _detect_communities(graph)
    by_comm: dict[int, list] = {}
    for node_id, community in communities.items():
        by_comm.setdefault(community, []).append(node_id)

    cohesion = _compute_cohesion(graph, by_comm)

    summaries = _community_summaries(graph, by_comm, cohesion, detail)
    coupling = _cross_community_coupling(graph, communities, detail)
    bridges = _bridge_rows(graph, communities, detail)
    cycles_data = cycles_json(graph, _limit_for_detail(detail, 8))
    core_data = core_json(graph, _limit_for_detail(detail, 10))
    articulations = articulation_json(graph, _limit_for_detail(detail, 10))
    warnings = _architecture_warnings(coupling, by_comm, cohesion)

    return {
        "operation": "architecture_overview",
        "detail_level": detail,
        "node_count": graph.node_count(),
        "edge_count": graph.edge_count(),
        "community_count": len(by_comm),
        "communities": summaries,
        "cross_community_coupling": coupling,
        "bridge_nodes": bridges,
        "cycles": cycles_data.get("hits", []),
        "core_nodes": core_data.get("hits", []),
        "articulation_points": articulations.get("hits", []),
        "warnings": warnings,
        "suggested_next_tools": [
            "bridge_nodes",
            "cycles",
            "core",
            "articulation_points",
            "traverse",
            "impact",
            "gaps",
        ],
    }


def community_split_json(graph, threshold: float = 0.25, min_size: int = 10) -> dict[str, Any]:
    """Split oversized communities and report new communities.

    Args:
        graph: The code graph.
        threshold: Cohesion threshold for splitting.
        min_size: Minimum community size to keep.

    Returns:
        Community split analysis.
    """
    communities = _detect_communities(graph)

    # Detect oversized communities (> min_size nodes)
    by_comm: dict[int, list] = {}
    for node_id, community in communities.items():
        by_comm.setdefault(community, []).append(node_id)

    new_communities = 0
    final_communities: dict[int, list] = {}

    for cid, members in by_comm.items():
        if len(members) > min_size:
            # Split into halves
            mid = len(members) // 2
            final_communities[cid] = members[:mid]
            final_communities[cid + 1000] = members[mid:]
            new_communities += 1
        else:
            final_communities[cid] = members

    comm_list = [
        {
            "id": cid,
            "size": len(members),
            "sample": [str(m) for m in members[:5]],
        }
        for cid, members in final_communities.items()
    ]

    return {
        "operation": "community_split",
        "threshold": threshold,
        "min_size": min_size,
        "new_communities": new_communities,
        "communities": comm_list,
    }


# ── Helpers ────────────────────────────────────────────────────────


def _detect_communities(graph) -> dict[Any, int]:
    """Detect communities using connected components (simple heuristic)."""
    # Use Louvain-like approach: group nodes by source file
    file_groups: dict[str, int] = {}
    node_id = 0
    communities: dict[Any, int] = {}

    for current_id, node in graph.nodes():
        source = node.source_uri or ""
        if not source:
            continue
        if source not in file_groups:
            file_groups[source] = node_id
            node_id += 1
        communities[current_id] = file_groups[source]

    return communities


def _compute_cohesion(graph, by_comm: dict[int, list]) -> dict[int, float]:
    """Compute cohesion for each community."""
    node_communities = {
        node_id: community
        for community, members in by_comm.items()
        for node_id in members
    }
    internal_edges: dict[int, int] = {}
    for _, src, dst, _edge in graph.edges():
        source_community = node_communities.get(src)
        if source_community is not None and source_community == node_communities.get(dst):
            internal_edges[source_community] = internal_edges.get(source_community, 0) + 1

    cohesion: dict[int, float] = {}
    for cid, members in by_comm.items():
        if len(members) <= 1:
            cohesion[cid] = 1.0
            continue
        total_possible = len(members) * (len(members) - 1)
        cohesion[cid] = internal_edges.get(cid, 0) / total_possible
    return cohesion


def _community_summaries(
    graph,
    by_comm: dict[int, list],
    cohesion: dict[int, float],
    detail: str,
) -> list[dict[str, Any]]:
    """Generate community summary dicts."""
    summaries = []
    for cid, members in by_comm.items():
        files: dict[str, int] = {}
        kinds: dict[str, int] = {}
        for nid in members:
            node = graph.node(nid) if hasattr(graph, "node") else None
            if node is None:
                for _, n in graph.nodes():
                    if _ids_match(n, nid):
                        node = n
                        break
            if node:
                source = node.source_uri or ""
                if source:
                    files[source] = files.get(source, 0) + 1
                kind = str(node.kind)
                kinds[kind] = kinds.get(kind, 0) + 1

        top_files = sorted(files.items(), key=lambda x: -x[1])[: _limit_for_detail(detail, 5)]
        kind_counts = sorted(kinds.items(), key=lambda x: -x[1])

        summaries.append({
            "community": cid,
            "size": len(members),
            "cohesion": cohesion.get(cid, 0.0),
            "top_files": [{"path": p, "nodes": c} for p, c in top_files],
            "kind_counts": [{"kind": k, "count": c} for k, c in kind_counts],
        })

    summaries.sort(key=lambda x: -x["size"])
    return summaries[: _limit_for_detail(detail, 12)]


def _cross_community_coupling(graph, communities: dict[Any, int], detail: str) -> list[dict[str, Any]]:
    """Compute cross-community coupling edges."""
    coupling: dict[tuple[int, int], int] = {}
    for _, src, dst, _ in graph.edges():
        a = communities.get(src)
        b = communities.get(dst)
        if a is not None and b is not None and a != b:
            key = (min(a, b), max(a, b))
            coupling[key] = coupling.get(key, 0) + 1

    rows = sorted(
        [{"from": a, "to": b, "edges": e} for (a, b), e in coupling.items()],
        key=lambda x: -x["edges"],
    )
    return rows[: _limit_for_detail(detail, 10)]


def _bridge_rows(graph, communities: dict[Any, int], detail: str) -> list[dict[str, Any]]:
    """Find bridge nodes (nodes connecting multiple communities)."""
    comm_degree: dict[int, int] = {}
    node_comms: dict[Any, set[int]] = {}

    for _, src, dst, _ in graph.edges():
        for nid in (src, dst):
            if nid not in node_comms:
                node_comms[nid] = set()
            comm_a = communities.get(src)
            comm_b = communities.get(dst)
            if comm_a is not None:
                node_comms[nid].add(comm_a)
            if comm_b is not None:
                node_comms[nid].add(comm_b)

    # Bridge score: number of distinct communities touched
    bridges = []
    for nid, comms in node_comms.items():
        if len(comms) > 1:
            node = graph.node(nid) if hasattr(graph, "node") else None
            if node is None:
                for _, n in graph.nodes():
                    if _ids_match(n, nid):
                        node = n
                        break
            if node:
                bridges.append({
                    "score": len(comms),
                    "communities_touched": len(comms),
                    "degree": (
                        sum(1 for _ in graph.out_neighbors(nid))
                        + sum(1 for _ in graph.in_neighbors(nid))
                        if hasattr(graph, "out_neighbors")
                        else 0
                    ),
                    "qualified_name": node.qualified_name,
                    "kind": str(node.kind),
                    "source_uri": node.source_uri,
                })

    bridges.sort(key=lambda x: -x["score"])
    return bridges[: _limit_for_detail(detail, 10)]


def _architecture_warnings(
    coupling_rows: list[dict],
    by_comm: dict[int, list],
    cohesion: dict[int, float],
) -> list[dict[str, Any]]:
    """Generate architecture warnings."""
    warnings: list[dict[str, Any]] = []

    # Cross-community coupling warnings
    for row in coupling_rows[:5]:
        edges = row.get("edges", 0)
        if edges >= 5:
            warnings.append({
                "kind": "cross_community_coupling",
                "severity": "high" if edges >= 20 else "medium",
                "communities": [row.get("from"), row.get("to")],
                "edges": edges,
            })

    # Low cohesion warnings
    for cid, members in by_comm.items():
        score = cohesion.get(cid, 1.0)
        if len(members) > 1 and score < 0.1:
            warnings.append({
                "kind": "low_cohesion_community",
                "severity": "medium",
                "community": cid,
                "size": len(members),
                "cohesion": score,
            })

    return warnings[:10]


def _limit_for_detail(detail: str, standard: int) -> int:
    """Compute limit based on detail level."""
    if detail == "minimal":
        return min(standard, 5)
    elif detail == "full":
        return standard * 4
    return standard


def _ids_match(node: Any, node_id: Any) -> bool:
    """Check if a node matches a node ID."""
    if not hasattr(node, "id"):
        return False
    return node.id == node_id
