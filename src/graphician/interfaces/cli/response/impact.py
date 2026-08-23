"""Impact analysis and god node detection.

Mirrors the Rust ``impact.rs`` module.
"""

from __future__ import annotations

from collections import deque
from typing import Any


def handle_impact(graph, params: dict[str, Any]) -> dict[str, Any]:
    """Compute impact of a node on the rest of the graph.

    Args:
        graph: The code graph.
        params: Parameters (target, max_hops, direction).

    Returns:
        Impact analysis with impacted nodes and scores.
    """
    target = params.get("target", params.get("seed", ""))
    max_hops = int(params.get("max_hops", params.get("max_depth", 3)))
    direction = params.get("direction", "out")

    if not target:
        return {"operation": "impact", "impacted": [], "total": 0}

    nid = _resolve(graph, target)
    if nid is None:
        return {
            "operation": "impact",
            "impacted": [],
            "total": 0,
            "error": f"target not found: {target}",
        }

    impacted = []
    seen: set = {nid}
    queue = deque([(nid, 0)])

    while queue:
        current, depth = queue.popleft()
        if depth >= max_hops:
            continue

        neighbors = []
        if direction in ("out", "both"):
            neighbors.extend(_out_ids(graph, current))
        if direction in ("in", "both"):
            neighbors.extend(_in_ids(graph, current))

        for next_id in neighbors:
            if next_id not in seen:
                seen.add(next_id)
                node = graph.node(next_id) if hasattr(graph, "node") else None
                if node is None:
                    for _, n in graph.nodes():
                        if _ids_match(n, next_id):
                            node = n
                            break
                if node:
                    score = 1.0 / (depth + 1)
                    impacted.append({
                        "score": round(score, 4),
                        "distance": depth + 1,
                        "qualified_name": node.qualified_name,
                        "kind": str(node.kind),
                        "source_uri": node.source_uri,
                    })
                    queue.append((next_id, depth + 1))

    impacted.sort(key=lambda x: -x["score"])
    return {
        "operation": "impact",
        "target": target,
        "impacted": impacted,
        "total": len(impacted),
    }


def handle_god_nodes(graph, params: dict[str, Any]) -> dict[str, Any]:
    """Find god nodes (high-degree nodes).

    Args:
        graph: The code graph.
        params: Parameters (limit, min_degree).

    Returns:
        God nodes with degree info.
    """
    min_degree = int(params.get("min_degree", 20))
    limit = int(params.get("limit", 25))

    degrees: dict[int, tuple[int, int]] = {}  # node_id -> (in_degree, out_degree)
    for _, src, dst, _ in graph.edges():
        if src not in degrees:
            degrees[src] = (0, 0)
        degrees[src] = (degrees[src][0], degrees[src][1] + 1)
        if dst not in degrees:
            degrees[dst] = (0, 0)
        degrees[dst] = (degrees[dst][0] + 1, degrees[dst][1])

    god_nodes = []
    for nid, (in_d, out_d) in degrees.items():
        total = in_d + out_d
        if total >= min_degree:
            node = graph.node(nid) if hasattr(graph, "node") else None
            if node is None:
                for _, n in graph.nodes():
                    if _ids_match(n, nid):
                        node = n
                        break
            if node:
                god_nodes.append({
                    "score": total,
                    "in_degree": in_d,
                    "out_degree": out_d,
                    "qualified_name": node.qualified_name,
                    "kind": str(node.kind),
                    "source_uri": node.source_uri,
                })

    god_nodes.sort(key=lambda x: -x["score"])
    return {
        "operation": "god_nodes",
        "hits": god_nodes[:limit],
        "total": len(god_nodes),
    }


def hub_nodes_json(graph, limit: int = 25) -> dict[str, Any]:
    """Find hub nodes (high-degree, similar to god nodes).

    Args:
        graph: The code graph.
        limit: Max results.

    Returns:
        Hub nodes.
    """
    degrees: dict[int, int] = {}
    for _, src, dst, _ in graph.edges():
        degrees[src] = degrees.get(src, 0) + 1
        degrees[dst] = degrees.get(dst, 0) + 1

    hubs = []
    for nid, degree in sorted(degrees.items(), key=lambda x: -x[1])[:limit]:
        node = graph.node(nid) if hasattr(graph, "node") else None
        if node is None:
            for _, n in graph.nodes():
                if _ids_match(n, nid):
                    node = n
                    break
        if node:
            hubs.append({
                "score": degree,
                "qualified_name": node.qualified_name,
                "kind": str(node.kind),
                "source_uri": node.source_uri,
            })

    return {
        "operation": "hub_nodes",
        "hits": hubs,
        "total": len(hubs),
    }


# ── Helpers ────────────────────────────────────────────────────────


def _resolve(graph, target: str) -> Any:
    """Resolve a target string to a node ID."""
    nid = graph.find_by_qname(target) if hasattr(graph, "find_by_qname") else None
    if nid is not None:
        return nid

    try:
        return int(target)
    except (ValueError, TypeError):
        pass

    for node_id, node in graph.nodes():
        if node.qualified_name == target or node.name == target:
            return node_id

    return None


def _out_ids(graph, node_id: Any) -> list[Any]:
    """Get outgoing neighbor IDs."""
    if hasattr(graph, "out_neighbors"):
        return [n for n, _ in graph.out_neighbors(node_id)]
    return []


def _in_ids(graph, node_id: Any) -> list[Any]:
    """Get incoming neighbor IDs."""
    if hasattr(graph, "in_neighbors"):
        return [n for n, _ in graph.in_neighbors(node_id)]
    return []


def _ids_match(node: Any, node_id: Any) -> bool:
    """Check if a node matches a node ID."""
    if not hasattr(node, "id"):
        return False
    return node.id == node_id
