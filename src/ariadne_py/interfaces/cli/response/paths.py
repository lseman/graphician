"""Path query operations: between-two-symbols path finding.

Mirrors the Rust ``paths.rs`` module.
"""

from __future__ import annotations

from typing import Any

from ....core.edge import EdgeKind


def handle_paths(graph, params: dict[str, Any]) -> dict[str, Any]:
    """Find paths between two symbols using weighted path enumeration.

    Args:
        graph: The code graph.
        params: Parameters (from, to, max_hops, limit).

    Returns:
        Paths with costs and node lists.
    """
    from_id = _resolve(graph, params.get("from", ""))
    to_id = _resolve(graph, params.get("to", ""))

    if from_id is None or to_id is None:
        return {
            "operation": "paths",
            "paths": [],
            "error": "Both 'from' and 'to' must be valid qualified names",
        }

    max_hops = int(params.get("max_hops", 5))
    limit = int(params.get("limit", 10))

    # Use find_top_paths from the paths module
    try:
        from ...analysis.paths import find_top_paths, PathQuery

        paths = find_top_paths(
            graph,
            PathQuery(from_id=from_id, to_id=to_id, max_hops=max_hops),
            limit,
        )
    except Exception:
        # Fallback: simple BFS paths
        paths = _simple_paths(graph, from_id, to_id, max_hops, limit)

    result_paths: list[dict[str, Any]] = []
    for path in paths:
        nodes = []
        for nid in path.nodes:
            node = graph.node(nid)
            if node is None:
                for _, n in graph.nodes():
                    if _ids_match(n, nid):
                        node = n
                        break
            if node:
                nodes.append({
                    "qualified_name": node.qualified_name,
                    "kind": str(node.kind),
                })
        result_paths.append({
            "cost": path.cost,
            "nodes": nodes,
        })

    return {
        "operation": "paths",
        "paths": result_paths,
    }


def _resolve(graph, target: str) -> Any | None:
    """Resolve a target string to a node ID."""
    nid = graph.find_by_qname(target) if hasattr(graph, "find_by_qname") else None
    if nid is not None:
        return nid
    try:
        return int(target)
    except (ValueError, TypeError):
        pass
    for _, node in graph.nodes():
        if node.qualified_name == target or node.name == target:
            return node.id if hasattr(node, "id") else None
    return None


def _ids_match(node: Any, node_id: Any) -> bool:
    """Check if a node matches a node ID."""
    if not hasattr(node, "id"):
        return False
    return node.id == node_id


def _simple_paths(graph, from_id, to_id, max_hops, limit):
    """Simple BFS path finding between two nodes.

    Args:
        graph: The code graph.
        from_id: Starting node ID.
        to_id: Target node ID.
        max_hops: Maximum number of hops.
        limit: Maximum number of paths to return.

    Returns:
        List of WeightedPath objects.
    """
    from ...analysis.paths import WeightedPath

    results: list[WeightedPath] = []
    # BFS queue: (current_id, path_nodes, total_cost)
    queue: list[tuple[Any, list, float]] = [(from_id, [from_id], 0.0)]
    seen_paths: set[tuple] = set()

    while queue and len(results) < limit:
        current, path_nodes, cost = queue.pop(0)
        depth = len(path_nodes) - 1

        if current == to_id and len(path_nodes) > 1:
            results.append(WeightedPath(nodes=path_nodes, cost=cost))
            continue
        if depth >= max_hops:
            continue

        neighbors = []
        if hasattr(graph, "out_neighbors"):
            neighbors.extend(graph.out_neighbors(current))
        if hasattr(graph, "in_neighbors"):
            neighbors.extend(graph.in_neighbors(current))

        for neighbor, edge in neighbors:
            if neighbor in path_nodes:
                continue
            path_key = tuple(str(n) for n in path_nodes + [neighbor])
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)

            edge_cost = 1.0
            if edge and hasattr(edge, "confidence"):
                edge_cost = 1.0 / max(edge.confidence, 0.05)
            elif edge and hasattr(edge, "kind"):
                base_costs = {
                    EdgeKind.DEFINES: 0.35,
                    EdgeKind.CALLS: 1.0,
                    EdgeKind.IMPORTS: 1.35,
                    EdgeKind.DEPENDS_ON: 1.35,
                }
                edge_cost = base_costs.get(edge.kind, 1.5)

            new_path = path_nodes + [neighbor]
            queue.append((neighbor, new_path, cost + edge_cost))

    results.sort(key=lambda p: p.cost)
    return results[:limit]
