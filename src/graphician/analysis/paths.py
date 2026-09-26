"""Constrained path enumeration and traversal helpers.

``find_paths`` enumerates simple paths between two nodes (or from a
source to any node) under three constraints:

- ``max_hops`` — drop paths longer than this (BFS bound).
- ``edge_kinds`` — restrict traversal to specific edge kinds.
- ``min_confidence`` — drop edges whose confidence score is below this
  threshold.

``find_top_paths`` uses Dijkstra-style priority search with edge-kind
weights and diversity scoring to return the best weighted paths.

``callees_of`` / ``callers_of`` walk the graph following outgoing
``Calls`` / incoming ``Calls`` edges, returning all reachable nodes.

``max_depth_from`` computes the greatest hop count reachable from a
node, optionally restricted to specific edge kinds.

Optimized versions use numpy for batch operations.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ..core.edge import EdgeKind
from ..core.graph import Graph
from ..core.id import NodeId
from ..core.node import NodeKind


@dataclass
class PathQuery:
    """Parameters for constrained path enumeration."""
    from_id: NodeId
    to_id: NodeId | None = None
    max_hops: int = 6
    edge_kinds: list[EdgeKind] | None = None
    min_confidence: float = 0.0


@dataclass
class WeightedPath:
    """A path with a cumulative traversal cost."""
    nodes: list[NodeId]
    cost: float
    edges: list[tuple[EdgeKind, float]] = field(default_factory=list)

    @property
    def hop_count(self) -> int:
        return len(self.nodes) - 1


# ── Edge kind weights ────────────────────────────────────────────────

_PATH_EDGE_COST: dict[EdgeKind, float] = {
    EdgeKind.DEFINES: 0.35,
    EdgeKind.CALLS: 1.0,
    EdgeKind.IMPORTS: 1.35,
    EdgeKind.DEPENDS_ON: 1.35,
    EdgeKind.INHERITS: 0.8,
    EdgeKind.IMPLEMENTS: 0.8,
    EdgeKind.DATA_FLOW: 0.5,
    EdgeKind.READS_WRITES: 0.9,
    EdgeKind.TESTED_BY: 1.8,
    EdgeKind.MEMBER_OF: 3.0,
    EdgeKind.ENTRY_OF: 3.0,
    EdgeKind.DESCRIBES: 1.1,
    EdgeKind.DOCUMENTED_BY: 1.1,
    EdgeKind.MENTIONS: 1.7,
    EdgeKind.ILLUSTRATES: 1.7,
    EdgeKind.SIMILAR_TO: 2.0,
    EdgeKind.RATIONALE_FOR: 2.0,
}

_IMPACT_EDGE_COST: dict[EdgeKind, float] = {
    EdgeKind.CALLS: 1.0,
    EdgeKind.DEFINES: 1.25,
    EdgeKind.IMPORTS: 1.6,
    EdgeKind.DEPENDS_ON: 1.6,
    EdgeKind.INHERITS: 0.75,
    EdgeKind.IMPLEMENTS: 0.75,
    EdgeKind.DATA_FLOW: 0.8,
    EdgeKind.READS_WRITES: 0.9,
    EdgeKind.TESTED_BY: 1.1,
    EdgeKind.MEMBER_OF: 5.0,
    EdgeKind.ENTRY_OF: 5.0,
    EdgeKind.DESCRIBES: 1.2,
    EdgeKind.DOCUMENTED_BY: 1.2,
    EdgeKind.MENTIONS: 1.8,
    EdgeKind.ILLUSTRATES: 1.8,
    EdgeKind.SIMILAR_TO: 2.0,
    EdgeKind.RATIONALE_FOR: 2.0,
}

_NODE_KIND_BOOST: dict[NodeKind, float] = {
    NodeKind.FUNCTION: 1.3,
    NodeKind.METHOD: 1.3,
    NodeKind.CLASS: 1.3,
    NodeKind.TYPE: 1.3,
    NodeKind.TRAIT: 1.2,
    NodeKind.IMPL: 1.2,
    NodeKind.FILE: 0.95,
    NodeKind.MODULE: 0.95,
    NodeKind.DOCUMENT: 0.85,
    NodeKind.SECTION: 0.85,
    NodeKind.CONCEPT: 0.85,
    NodeKind.DIAGRAM: 0.75,
    NodeKind.IMAGE: 0.75,
    NodeKind.VARIABLE: 0.7,
    NodeKind.COMMIT: 0.7,
    NodeKind.AUTHOR: 0.7,
    NodeKind.HYPEREDGE: 0.7,
    NodeKind.FLOW: 0.4,
    NodeKind.PACKAGE: 0.95,
}


def _path_edge_cost(kind: EdgeKind, confidence: float) -> float:
    """Cost to traverse an edge (lower = more preferred for top paths)."""
    return _PATH_EDGE_COST.get(kind, 1.5) / max(confidence, 0.05)


def _impact_edge_cost(kind: EdgeKind, confidence: float) -> float:
    """Cost for impact analysis (higher = more cost to traverse)."""
    return _IMPACT_EDGE_COST.get(kind, 1.5) / max(confidence, 0.05)


def _node_kind_boost(kind: NodeKind) -> float:
    """Impact ranking boost for node kinds."""
    return _NODE_KIND_BOOST.get(kind, 0.8)


# ── Path enumeration ─────────────────────────────────────────────────

def find_paths(graph: Graph, q: PathQuery) -> list[list[NodeId]]:
    """Enumerate simple paths from ``q.from_id`` to ``q.to_id`` (or any
    node if ``to_id`` is ``None``) under the query's constraints.

    Returns paths as lists of node IDs.
    """
    from .native import native_graph

    snapshot = native_graph(graph)
    if snapshot is not None:
        paths = snapshot.paths(
            q.from_id.value,
            q.to_id.value if q.to_id is not None else None,
            q.max_hops,
            [kind.value for kind in q.edge_kinds] if q.edge_kinds else None,
            q.min_confidence,
        )
        return [[NodeId(node_id) for node_id in path] for path in paths]

    results: list[list[NodeId]] = []
    queue: deque[list[NodeId]] = deque([[q.from_id]])

    while queue:
        path = queue.popleft()
        if len(path) > q.max_hops + 1:
            continue
        last = path[-1]
        # Record path when at target (or anywhere if no target specified)
        if len(path) > 1 and (q.to_id is None or last == q.to_id):
            results.append(list(path))
            if q.to_id is not None and len(path) > q.max_hops:
                continue
        for neighbor, edge in graph.out_neighbors(last):
            if neighbor in path:
                continue
            if q.edge_kinds and edge.kind not in q.edge_kinds:
                continue
            if edge.confidence.score() < q.min_confidence:
                continue
            queue.append([*path, neighbor])

    return results


def find_top_paths(
    graph: Graph,
    q: PathQuery,
    limit: int = 10,
    expansion_budget: int = 200_000,
) -> list[WeightedPath]:
    """Dijkstra-style path search returning the best weighted paths.

    Uses edge-kind-dependent costs and diversity scoring to avoid
    returning multiple nearly-identical paths.

    The search is bounded: each node is expanded at most ``limit``
    times and total expansions never exceed ``expansion_budget``, so
    the function terminates quickly even on dense graphs where the
    number of simple paths is exponential.
    """
    from .native import native_graph

    snapshot = native_graph(graph)
    if snapshot is not None:
        edge_kinds = [k.value for k in q.edge_kinds] if q.edge_kinds else None
        target_id = q.to_id.value if q.to_id is not None else None
        raw = snapshot.top_paths(
            q.from_id.value,
            target_id,
            q.max_hops,
            edge_kinds,
            q.min_confidence,
            limit,
            expansion_budget,
        )
        results: list[WeightedPath] = []
        for node_ids, cost, edge_info in raw:
            edges: list[tuple[EdgeKind, float]] = [
                (EdgeKind(e), c) for e, c in edge_info
            ]
            results.append(WeightedPath(
                nodes=[NodeId(nid) for nid in node_ids],
                cost=float(cost),
                edges=edges,
            ))
        return results

    # Fallback: Python implementation (original logic)
    if limit <= 0:
        return []

    import heapq

    class _Candidate:
        __slots__ = ("cost", "nodes")

        def __init__(self, nodes: list[NodeId], cost: float) -> None:
            self.nodes = nodes
            self.cost = cost

        def __lt__(self, other: _Candidate) -> bool:
            return self.cost < other.cost

    heap: list[_Candidate] = [_Candidate([q.from_id], 0.0)]
    chosen: list[WeightedPath] = []
    node_expansions: dict[NodeId, int] = {}
    expansions = 0

    while heap and len(chosen) < limit * 3:
        cand = heapq.heappop(heap) if heap else None
        if cand is None:
            break

        if len(cand.nodes) > q.max_hops + 1:
            continue

        last = cand.nodes[-1]

        # Record path when at target (or anywhere if no target specified)
        if len(cand.nodes) > 1:
            should_record = q.to_id is None or last == q.to_id
            if should_record:
                # Check diversity with previously chosen paths
                is_diverse = True
                for chosen_path in chosen:
                    overlap = _node_overlap(cand.nodes, chosen_path.nodes)
                    if overlap > 0.7:
                        is_diverse = False
                        break
                if is_diverse or len(chosen) < 3:
                    edges: list[tuple[EdgeKind, float]] = []
                    for i in range(len(cand.nodes) - 1):
                        src = cand.nodes[i]
                        for nbr, edge in graph.out_neighbors(src):
                            if nbr == cand.nodes[i + 1]:
                                edges.append((edge.kind, edge.confidence.score()))
                                break
                    chosen.append(WeightedPath(
                        nodes=list(cand.nodes),
                        cost=cand.cost,
                        edges=edges,
                    ))
                    if q.to_id is not None and len(cand.nodes) > q.max_hops:
                        continue
                    continue

        # Expand (bounded: per-node cap + total budget)
        if expansions >= expansion_budget:
            continue
        if node_expansions.get(last, 0) >= limit:
            continue
        node_expansions[last] = node_expansions.get(last, 0) + 1
        expansions += 1
        if len(heap) > 4 * expansion_budget:
            break
        for neighbor, edge in graph.out_neighbors(last):
            if neighbor in cand.nodes:
                continue
            if q.edge_kinds and edge.kind not in q.edge_kinds:
                continue
            if edge.confidence.score() < q.min_confidence:
                continue
            new_cost = cand.cost + _path_edge_cost(edge.kind, edge.confidence.score())
            heapq.heappush(heap, _Candidate([*cand.nodes, neighbor], new_cost))

    # Sort and limit
    chosen.sort(key=lambda p: p.cost)
    return chosen[:limit]


def _node_overlap(a: list[NodeId], b: list[NodeId]) -> float:
    """Jaccard overlap between two node lists."""
    if not a or not b:
        return 0.0
    set_a = {n.value for n in a}
    set_b = {n.value for n in b}
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ── Traversal helpers ────────────────────────────────────────────────

def callees_of(graph: Graph, node_id: NodeId, max_hops: int = 6) -> list[NodeId]:
    """Return all nodes reachable from *node_id* via outgoing ``Calls``
    edges, up to *max_hops*."""
    from .native import native_graph

    snapshot = native_graph(graph)
    if snapshot is not None:
        return [NodeId(value) for value in snapshot.traverse(
            node_id.value, EdgeKind.CALLS.value, False, max_hops
        )]
    return _traverse(graph, node_id, EdgeKind.CALLS, max_hops)


def callers_of(
    graph: Graph,
    node_id: NodeId,
    max_hops: int = 6,
    min_confidence: float = 0.5,
) -> list[NodeId]:
    """Return all nodes that can reach *node_id* via ``Calls`` edges
    (incoming ``Calls``).

    Args:
        graph: The code graph.
        node_id: The target node to find callers for.
        max_hops: Maximum traversal depth.
        min_confidence: Minimum edge confidence to follow.  Ambiguous
            edges (confidence ≈ 0) are always excluded.
    """
    from .native import native_graph

    snapshot = native_graph(graph)
    if snapshot is not None:
        return [NodeId(value) for value in snapshot.traverse(
            node_id.value, EdgeKind.CALLS.value, True, max_hops, min_confidence
        )]

    visited: set[int] = {node_id.value}
    queue: deque[int] = deque()
    for prev, edge in graph.in_neighbors(node_id):
        if edge.kind == EdgeKind.CALLS and edge.confidence.score() >= min_confidence:
            queue.append(prev.value)
    results: list[NodeId] = []
    while queue:
        nid = queue.popleft()
        if nid in visited:
            continue
        visited.add(nid)
        results.append(NodeId(nid))
        for prev, edge in graph.in_neighbors(NodeId(nid)):
            pval = prev.value if isinstance(prev, NodeId) else prev
            if (edge.kind == EdgeKind.CALLS
                    and edge.confidence.score() >= min_confidence
                    and pval not in visited):
                queue.append(pval)
    return results


def max_depth_from(graph: Graph, node_id: NodeId, max_hops: int = 20) -> int:
    """Return the greatest hop count reachable from *node_id* via
    outgoing edges (0 if no outgoing edges)."""
    from .native import native_graph

    snapshot = native_graph(graph)
    if snapshot is not None:
        return snapshot.max_depth(node_id.value, max_hops)

    visited: dict[int, int] = {node_id.value: 0}
    queue: deque[int] = deque([node_id.value])
    max_depth = 0
    while queue:
        current = queue.popleft()
        current_depth = visited[current]
        for neighbor, _ in graph.out_neighbors(NodeId(current)):
            nval = neighbor.value if isinstance(neighbor, NodeId) else neighbor
            if nval not in visited:
                new_depth = current_depth + 1
                visited[nval] = new_depth
                max_depth = max(max_depth, new_depth)
                if new_depth < max_hops:
                    queue.append(nval)
    return max_depth


def _traverse(
    graph: Graph,
    start: NodeId,
    edge_kind: EdgeKind,
    max_hops: int,
) -> list[NodeId]:
    """BFS following edges of a specific kind.

    Uses the graph's adjacency lists directly — no full-edge iteration.
    """
    visited: set[int] = {start.value}
    results: list[NodeId] = []
    queue: list[tuple[int, int]] = [(start.value, 0)]
    head = 0

    while head < len(queue):
        nid, depth = queue[head]
        head += 1
        if depth >= max_hops:
            continue
        for neighbor, _ in graph.out_neighbors(NodeId(nid)):
            nval = neighbor.value if isinstance(neighbor, NodeId) else neighbor
            if nval in visited:
                continue
            visited.add(nval)
            results.append(NodeId(nval))
            queue.append((nval, depth + 1))

    return results


def _bfs_optimized(
    graph: Graph,
    start: NodeId,
    edge_kind: EdgeKind | None = None,
    max_hops: int = 20,
    filter_confidence: float = 0.0,
) -> list[NodeId]:
    """Optimized BFS traversal using the graph's adjacency lists.

    No full-edge iteration — walks directly from the start node.
    """
    visited: set[int] = {start.value}
    results: list[NodeId] = []
    queue: list[tuple[int, int]] = [(start.value, 0)]
    head = 0

    while head < len(queue):
        nid, depth = queue[head]
        head += 1
        if depth >= max_hops:
            continue
        for neighbor, edge in graph.out_neighbors(NodeId(nid)):
            if edge.confidence.score() < filter_confidence:
                continue
            if edge_kind is not None and edge.kind != edge_kind:
                continue
            nval = neighbor.value if isinstance(neighbor, NodeId) else neighbor
            if nval in visited:
                continue
            visited.add(nval)
            results.append(NodeId(nval))
            queue.append((nval, depth + 1))

    return results
