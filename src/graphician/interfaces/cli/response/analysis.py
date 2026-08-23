"""Analysis operations: bridge nodes, cycles, core, articulation, gaps, etc.

Mirrors the Rust ``analysis.rs`` module.
"""

from __future__ import annotations

from typing import Any

from ....core.edge import Confidence, EdgeKind


def bridge_nodes_json(graph, limit: int = 25) -> dict[str, Any]:
    """Find cross-community bridge nodes.

    Args:
        graph: The code graph.
        limit: Max bridge nodes to return.

    Returns:
        Bridge nodes with scores.
    """
    # Detect communities via Leiden.
    communities = _detect_communities(graph)

    # Find nodes connecting multiple communities
    node_comms: dict[int, set] = {}
    for _, src, dst, _ in graph.edges():
        comm_a = communities.get(src)
        comm_b = communities.get(dst)
        if comm_a is not None:
            node_comms.setdefault(src, set()).add(comm_a)
            node_comms.setdefault(dst, set()).add(comm_a)
        if comm_b is not None:
            node_comms.setdefault(dst, set()).add(comm_b)
            node_comms.setdefault(src, set()).add(comm_b)

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
                    "qualified_name": node.qualified_name,
                    "kind": str(node.kind),
                    "source_uri": node.source_uri,
                    "communities": len(comms),
                })

    bridges.sort(key=lambda x: -x["score"])
    return {
        "operation": "bridge_nodes",
        "hits": bridges[:limit],
        "total": len(bridges),
    }


def cycles_json(graph, limit: int = 25) -> dict[str, Any]:
    """Detect dependency cycles.

    Args:
        graph: The code graph.
        limit: Max cycles to return.

    Returns:
        Cycles with member nodes.
    """
    from ....analysis.structure import cyclic_components

    components = sorted(cyclic_components(graph), key=lambda item: -len(item.nodes))
    cycles = []
    for component in components[:limit]:
        nodes = [
            graph.node(node_id)
            for node_id in component.nodes
            if graph.node(node_id) is not None
        ]
        cycles.append({
            "size": len(nodes),
            "nodes": [
                {
                    "qualified_name": node.qualified_name,
                    "kind": node.kind.value,
                    "source_uri": node.source_uri,
                }
                for node in nodes
            ],
        })
    return {
        "operation": "cycles",
        "hits": cycles,
        "total": len(components),
    }


def core_json(graph, limit: int = 25) -> dict[str, Any]:
    """Find k-core nodes (high connectivity).

    Args:
        graph: The code graph.
        limit: Max core nodes to return.

    Returns:
        Core nodes with degree info.
    """
    from ....analysis.structure import core_numbers

    core_nodes = []
    ranked = sorted(core_numbers(graph).items(), key=lambda item: -item[1])
    for nid, coreness in ranked[:limit]:
        node = graph.node(nid)
        if node:
            core_nodes.append({
                "core": coreness,
                "degree": sum(1 for _ in graph.in_neighbors(nid))
                + sum(1 for _ in graph.out_neighbors(nid)),
                "qualified_name": node.qualified_name,
                "kind": node.kind.value,
                "source_uri": node.source_uri,
            })

    return {
        "operation": "core",
        "hits": core_nodes,
        "total": len(core_nodes),
    }


def articulation_json(graph, limit: int = 25) -> dict[str, Any]:
    """Find articulation points (single points of failure).

    Args:
        graph: The code graph.
        limit: Max articulation points to return.

    Returns:
        Articulation points with info.
    """
    from ....analysis.structure import find_articulation_points

    raw = find_articulation_points(graph)
    points = []
    for item in raw["articulation_points"]:
        node_id = graph.find_by_qname(item["qualified_name"])
        if node_id is None:
            continue
        points.append({
            **item,
            "degree": sum(1 for _ in graph.in_neighbors(node_id))
            + sum(1 for _ in graph.out_neighbors(node_id)),
        })
    points.sort(key=lambda item: -item["degree"])

    return {
        "operation": "articulation_points",
        "hits": points[:limit],
        "total": raw["total"],
    }


def gaps_json(graph, limit: int = 25) -> dict[str, Any]:
    """Find structural gaps (isolated or weakly connected nodes).

    Args:
        graph: The code graph.
        limit: Max gap nodes to return.

    Returns:
        Gap nodes with weakness info.
    """
    connected: set[int] = set()
    for _, src, dst, _ in graph.edges():
        connected.add(src)
        connected.add(dst)

    disconnected = []
    for nid, node in graph.nodes():
        if nid not in connected:
            disconnected.append({
                "qualified_name": node.qualified_name,
                "kind": str(node.kind),
                "source_uri": node.source_uri,
            })

    # Also find nodes with very few edges
    degrees: dict[int, int] = {}
    for _, src, dst, _ in graph.edges():
        degrees[src] = degrees.get(src, 0) + 1
        degrees[dst] = degrees.get(dst, 0) + 1

    for nid, degree in degrees.items():
        if degree <= 1 and len(disconnected) < limit:
            node = graph.node(nid) if hasattr(graph, "node") else None
            if node is None:
                for _, n in graph.nodes():
                    if _ids_match(n, nid):
                        node = n
                        break
            if node:
                disconnected.append({
                    "qualified_name": node.qualified_name,
                    "kind": str(node.kind),
                    "source_uri": node.source_uri,
                    "weakness": "low_degree",
                    "degree": degree,
                })

    return {
        "operation": "gaps",
        "hits": disconnected[:limit],
        "total": len(disconnected),
    }


def surprises_json(graph, limit: int = 25) -> dict[str, Any]:
    """Find surprising cross-community connections.

    Args:
        graph: The code graph.
        limit: Max surprises to return.

    Returns:
        Surprising connections.
    """
    communities = _detect_communities(graph)
    coupling: dict[tuple[str, str], int] = {}

    for _, src, dst, _ in graph.edges():
        a = str(communities.get(src, "unknown"))
        b = str(communities.get(dst, "unknown"))
        if a != b:
            key = (min(a, b), max(a, b))
            coupling[key] = coupling.get(key, 0) + 1

    surprises = []
    for (a, b), count in sorted(coupling.items(), key=lambda x: -x[1])[:limit]:
        surprises.append({
            "src": a,
            "dst": b,
            "score": count,
        })

    return {
        "operation": "surprises",
        "hits": surprises,
        "total": len(surprises),
    }


def large_functions_json(graph, min_lines: int = 80, limit: int = 50) -> dict[str, Any]:
    """Find large functions/methods.

    Args:
        graph: The code graph.
        min_lines: Minimum line count to qualify as large.
        limit: Max results to return.

    Returns:
        Large functions with line counts.
    """
    large = []
    for _, node in graph.nodes():
        if node.kind in ("function", "method") or str(node.kind) in ("FUNCTION", "METHOD"):
            length = 0
            if node.line_start is not None and node.line_end is not None:
                length = node.line_end - node.line_start
            if length >= min_lines:
                large.append({
                    "qualified_name": node.qualified_name,
                    "kind": str(node.kind),
                    "source_uri": node.source_uri,
                    "lines": length,
                    "line_start": node.line_start,
                    "line_end": node.line_end,
                })

    large.sort(key=lambda x: -x["lines"])
    return {
        "operation": "large_functions",
        "hits": large[:limit],
        "total": len(large),
    }


def diagnostics_json(db_path: str, limit: int = 25) -> dict[str, Any]:
    """Graph health diagnostics.

    Args:
        db_path: Path to the database.
        limit: Limit for sub-queries.

    Returns:
        Health diagnostics with warnings and metrics.
    """
    from ....persistence.store import GraphStore
    from .impact import hub_nodes_json

    store = GraphStore(db_path)
    try:
        graph = store.load_graph()
        from ....analysis.structure import call_resolution_stats

        calls = call_resolution_stats(graph)
        warnings: list[dict[str, Any]] = []
        health = "healthy"

        # Check for dead code
        dead = _find_dead_code(graph, 100)
        if len(dead) > 10:
            warnings.append({
                "kind": "dead_code",
                "message": f"{len(dead)} potentially dead nodes detected",
            })

        # Check for large functions
        large = large_functions_json(graph, min_lines=200, limit=limit)
        if large["total"] > 0:
            warnings.append({
                "kind": "large_functions",
                "message": f"{large['total']} functions exceed 200 lines",
            })

        # Check for god nodes
        god = hub_nodes_json(graph, limit)
        if god["hits"]:
            warnings.append({
                "kind": "high_degree_nodes",
                "message": f"{len(god['hits'])} nodes with degree > 50",
            })

        if warnings:
            health = "warnings"
            if any(w.get("severity") == "high" for w in warnings):
                health = "degraded"

        return {
            "operation": "diagnostics",
            "health": health,
            "node_count": graph.node_count(),
            "edge_count": graph.edge_count(),
            "warnings": warnings,
            "confidence_mix": {
                "extracted": sum(
                    1
                    for _, _, _, e in graph.edges()
                    if e.confidence is Confidence.EXTRACTED
                ),
                "inferred": sum(
                    1
                    for _, _, _, e in graph.edges()
                    if e.confidence is Confidence.INFERRED
                ),
                "ambiguous": sum(
                    1
                    for _, _, _, e in graph.edges()
                    if e.confidence is Confidence.AMBIGUOUS
                ),
            },
            "call_resolution": calls,
        }
    finally:
        store.close()


# ── Helpers ────────────────────────────────────────────────────────


def _detect_communities(graph) -> dict[Any, int]:
    """Detect well-connected communities with Leiden."""
    from ....analysis.communities.core import CommunityOptions
    from ....analysis.communities.leiden import leiden_with_options

    return leiden_with_options(graph, CommunityOptions())


def _find_cycles(graph, limit: int = 100) -> list[dict[str, Any]]:
    """Find simple cycles using DFS."""
    adj: dict[Any, set] = {}
    for _, src, dst, _ in graph.edges():
        adj.setdefault(src, set()).add(dst)

    cycles: list[dict[str, Any]] = []
    visited: set[Any] = set()

    def _dfs(node: Any, path: list[Any]) -> None:
        if len(cycles) >= limit:
            return
        for neighbor in adj.get(node, set()):
            if neighbor in path:
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:]
                members = []
                for nid in cycle:
                    n = graph.node(nid) if hasattr(graph, "node") else None
                    if n is None:
                        for _, nd in graph.nodes():
                            if _ids_match(nd, nid):
                                n = nd
                                break
                    if n:
                        members.append(n.qualified_name)
                if members:
                    cycles.append({
                        "length": len(cycle),
                        "members": members,
                    })
            elif neighbor not in visited:
                visited.add(neighbor)
                _dfs(neighbor, path + [neighbor])

    for nid, _node in graph.nodes():
        if nid not in visited:
            visited.add(nid)
            _dfs(nid, [nid])
            if len(cycles) >= limit:
                break

    return cycles


def _find_dead_code(graph, limit: int = 100) -> list[dict[str, Any]]:
    """Find nodes with no incoming or outgoing edges."""
    connected: set[Any] = set()
    for _, src, dst, _ in graph.edges():
        connected.add(src)
        connected.add(dst)

    dead = []
    for nid, node in graph.nodes():
        if nid not in connected and node.kind not in ("file",):
            dead.append({
                "qualified_name": node.qualified_name,
                "kind": str(node.kind),
                "source_uri": node.source_uri,
            })
            if len(dead) >= limit:
                break
    return dead


def _ids_match(node: Any, node_id: Any) -> bool:
    """Check if a node matches a node ID."""
    if not hasattr(node, "id"):
        return False
    return node.id == node_id
