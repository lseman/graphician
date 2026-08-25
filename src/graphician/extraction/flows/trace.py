"""Flow tracing and criticality scoring."""

from __future__ import annotations

from collections import deque

from ...core.edge import EdgeKind
from ...core.node import Node


def _is_test_node(node: Node) -> bool:
    """Check if a node is a test node."""
    val = node.properties.get("is_test")
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes")
    return False


def trace_flow(graph, entry, options) -> list:
    """Forward BFS from entry through Calls edges."""
    safety_ceiling = max(options.max_nodes_per_flow * 10, 500)
    visited = {entry}
    members = []
    queue = deque([(entry, 0)])

    while queue:
        node, depth = queue.popleft()
        members.append((node, depth))
        if len(members) >= safety_ceiling:
            break
        if depth >= options.max_depth:
            continue
        for next_id, edge in graph.out_neighbors(node):
            if edge.kind != EdgeKind.CALLS:
                continue
            if edge.confidence == "ambiguous":
                continue
            dst_node = graph.node(next_id)
            if dst_node is None:
                continue
            if dst_node.qualified_name.startswith("call::"):
                continue
            if next_id not in visited:
                visited.add(next_id)
                queue.append((next_id, depth + 1))

    if len(members) <= options.max_nodes_per_flow:
        return members

    member_ids = {nid for nid, _ in members}
    scored = []
    for nid, depth in members:
        in_flow_fanin = sum(
            1 for src, e in graph.in_neighbors(nid)
            if e.kind == EdgeKind.CALLS and src in member_ids
        )
        closeness = 1.0 / (depth + 1)
        fanin_bonus = min(in_flow_fanin / 10.0, 0.5)
        score = closeness + fanin_bonus
        scored.append((nid, depth, score))

    scored.sort(key=lambda x: -x[2])
    scored = scored[: options.max_nodes_per_flow]
    return [(nid, depth) for nid, depth, _ in scored]


def _compute_criticality(graph, members, entry_name, is_test_entry):
    """Criticality score in [0, 1]. Higher = more important."""
    size = len(members)
    size_score = min(max(size * 0.1, 0.0), 0.6) if size > 1 else 0.0

    total_fanin = 0
    for nid, _ in members:
        total_fanin += sum(1 for _, e in graph.in_neighbors(nid) if e.kind == EdgeKind.CALLS)
    avg_fanin = total_fanin / size if size > 0 else 0.0
    reuse_score = min(avg_fanin / 8.0, 0.25)

    name = entry_name.lower()
    shape_bonus = 0.0
    if name in ("main", "__main__"):
        shape_bonus = 0.1
    elif name.startswith("handle") or name.startswith("on_") or name.startswith("serve"):
        shape_bonus = 0.05

    test_penalty = 0.2 if is_test_entry else 0.0

    return max(min(size_score + reuse_score + shape_bonus - test_penalty, 1.0), 0.0)
