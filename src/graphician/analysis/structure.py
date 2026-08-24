"""Structural analysis: cycles, dead code, counterfactual, motifs, surprises.

Graph algorithms for understanding architectural properties.

Optimized versions use numpy/numba for performance-critical paths.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterator

import networkx as nx
import numpy as np

from ..core.edge import Edge, EdgeKind
from ..core.graph import Graph
from ..core.id import NodeId
from ..core.node import Node, NodeKind
from .adjacency import build_adjacency_matrix
from .numba_graph_algs import has_numba as has_numba_algs
from .export import export_graphml


@dataclass
class Component:
    """A strongly connected component (cycle group)."""
    nodes: list[NodeId]


@dataclass
class CoreNumber:
    """K-core decomposition result for a node."""
    node: NodeId
    core: int


@dataclass
class BridgeScore:
    """Bridge node scoring result."""
    node: NodeId
    score: float
    communities_touched: int
    degree: int
    approx_betweenness: float
    articulation: bool = False


@dataclass
class HubNode:
    """A hub node — highest total degree (in + out), excluding File nodes."""
    node: NodeId
    name: str
    qualified_name: str
    kind: str
    file: str
    in_degree: int
    out_degree: int
    total_degree: int
    community_id: int | None = None


def cyclic_components(graph: Graph) -> list[Component]:
    """Find cyclic (strongly connected) components via Tarjan's algorithm.

    Returns only components with more than one node, or single-node
    self-loops. This is the structural-analysis version of find_cycles;
    it returns typed Component objects rather than dicts.
    """
    from .native import native_graph

    snapshot = native_graph(graph)
    if snapshot is not None:
        return [
            Component(nodes=[NodeId(node_id) for node_id in component])
            for component in snapshot.cyclic_components()
        ]

    # Tarjan's SCC
    index_counter = [0]
    stack: list[NodeId] = []
    on_stack: set[NodeId] = set()
    indices: dict[NodeId, int] = {}
    lowlinks: dict[NodeId, int] = {}
    components: list[list[NodeId]] = []
    adjacency: dict[NodeId, list[NodeId]] = defaultdict(list)
    self_loop_nodes: set[NodeId] = set()
    for _, source, target, _edge in graph.edges():
        adjacency[source].append(target)
        if source == target:
            self_loop_nodes.add(source)

    def strong_connect(v: NodeId) -> None:
        indices[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for neighbor in adjacency.get(v, []):
            if neighbor not in indices:
                strong_connect(neighbor)
                lowlinks[v] = min(lowlinks[v], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[v] = min(lowlinks[v], indices[neighbor])

        if lowlinks[v] == indices[v]:
            component: list[NodeId] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component.append(w)
                if w == v:
                    break
            components.append(component)

    for nid, _ in graph.nodes():
        if nid not in indices:
            strong_connect(nid)

    # Filter to only cyclic components (size > 1 or self-loop)
    return [
        Component(nodes=c)
        for c in components
        if len(c) > 1 or (len(c) == 1 and c[0] in self_loop_nodes)
    ]


def core_numbers(graph: Graph) -> dict[NodeId, int]:
    """K-core decomposition of the undirected view of the graph.

    Uses numba-accelerated algorithm when available, falling back to
    the pure-Python heap-based implementation.

    Returns a mapping from NodeId to its k-core number.
    """
    from .native import native_graph

    snapshot = native_graph(graph)
    if snapshot is not None:
        return {
            NodeId(node_id): int(core)
            for node_id, core in snapshot.core_numbers().items()
        }

    # Build undirected adjacency as CSR
    # First, collect all unique edges (undirected)
    edge_set: set[tuple[int, int]] = set()
    for _, src, dst, _edge in graph.edges():
        if src != dst:
            edge_set.add((min(src.value, dst.value), max(src.value, dst.value)))

    all_nodes: list[int] = [nid.value for nid, _ in graph.nodes()]
    n = len(all_nodes)
    node_to_idx: dict[int, int] = {v: i for i, v in enumerate(all_nodes)}

    # Build CSR from edge set
    adj_map: dict[int, set[int]] = defaultdict(set)
    for u, v in edge_set:
        u_idx = node_to_idx.get(u)
        v_idx = node_to_idx.get(v)
        if u_idx is not None and v_idx is not None:
            adj_map[u_idx].add(v_idx)
            adj_map[v_idx].add(u_idx)

    # Build CSR arrays
    nnz = sum(len(v) for v in adj_map.values())
    row_ptr = np.zeros(n + 1, dtype=np.intp)
    col_idx = np.zeros(nnz, dtype=np.intp)

    pos = 0
    for i in range(n):
        row_ptr[i + 1] = row_ptr[i] + len(adj_map.get(i, set()))

    for i in range(n):
        start = row_ptr[i]
        for j, v in enumerate(sorted(adj_map.get(i, set()))):
            col_idx[start + j] = v

    # Use numba-accelerated k-core if available
    if has_numba_algs():
        from .numba_graph_algs import kcore_csr
        cores = kcore_csr(row_ptr, col_idx)
        return {NodeId(all_nodes[i]): int(cores[i]) for i in range(n)}

    # Fallback: pure Python heap-based
    remaining: set[int] = set(range(n))
    degree: dict[int, int] = {i: len(adj_map.get(i, set())) for i in range(n)}
    core_map: dict[NodeId, int] = {}
    current_core = 0
    heap = [(d, i) for i, d in degree.items()]
    heapq.heapify(heap)

    while heap:
        min_degree, node = heapq.heappop(heap)
        if node not in remaining or min_degree != degree.get(node, 0):
            continue
        current_core = max(current_core, min_degree)
        core_map[NodeId(all_nodes[node])] = current_core
        remaining.discard(node)

        for neighbor in adj_map.get(node, set()):
            if neighbor in remaining:
                degree[neighbor] = max(0, degree[neighbor] - 1)
                heapq.heappush(heap, (degree[neighbor], neighbor))

    return core_map


def approx_betweenness(graph: Graph, max_sources: int = 64) -> dict[NodeId, float]:
    """Approximate betweenness centrality via BFS from up to max_sources nodes.

    Uses the undirected view of the graph for shortest paths.
    Uses numba-accelerated implementation when available.
    """
    # Build undirected adjacency as CSR
    edge_set: set[tuple[int, int]] = set()
    for _, src, dst, _edge in graph.edges():
        if src != dst:
            edge_set.add((min(src.value, dst.value), max(src.value, dst.value)))

    all_nodes: list[int] = [nid.value for nid, _ in graph.nodes()]
    n = len(all_nodes)
    node_to_idx: dict[int, int] = {v: i for i, v in enumerate(all_nodes)}

    adj_map: dict[int, set[int]] = defaultdict(set)
    for u, v in edge_set:
        u_idx = node_to_idx.get(u)
        v_idx = node_to_idx.get(v)
        if u_idx is not None and v_idx is not None:
            adj_map[u_idx].add(v_idx)
            adj_map[v_idx].add(u_idx)

    # Build CSR arrays
    nnz = sum(len(v) for v in adj_map.values())
    row_ptr = np.zeros(n + 1, dtype=np.intp)
    col_idx = np.zeros(nnz, dtype=np.intp)

    for i in range(n):
        row_ptr[i + 1] = row_ptr[i] + len(adj_map.get(i, set()))

    for i in range(n):
        start = row_ptr[i]
        for j, v in enumerate(sorted(adj_map.get(i, set()))):
            col_idx[start + j] = v

    sources = all_nodes[:max_sources]
    source_indices = np.array([node_to_idx[s] for s in sources if s in node_to_idx], dtype=np.intp)

    # Use numba-accelerated betweenness if available
    if has_numba_algs():
        from .numba_graph_algs import betweenness_approx_csr
        scores_arr = betweenness_approx_csr(row_ptr, col_idx, source_indices, max_sources)
        return {NodeId(all_nodes[i]): float(scores_arr[i]) for i in range(n)}

    # Fallback: pure Python BFS
    sources = all_nodes[:max_sources]
    scores: dict[NodeId, float] = defaultdict(float)

    for source in sources:
        # BFS from source
        queue = [source]
        pred: dict[int, list[int]] = defaultdict(list)
        dist: dict[int, int] = {source: 0}
        order: list[int] = []

        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            order.append(node)
            next_dist = dist[node] + 1
            for neighbor in adj_map.get(node_to_idx.get(node, -1), set()):
                neighbor_node = all_nodes[neighbor]
                if neighbor_node not in dist:
                    dist[neighbor_node] = next_dist
                    queue.append(neighbor_node)
                if dist.get(neighbor_node) == next_dist:
                    pred[neighbor_node].append(node)

        # Dependency accumulation (reverse order)
        dependency: dict[int, float] = defaultdict(float)
        for node in reversed(order):
            coeff = (1.0 + dependency.get(node, 0.0)) / max(len(pred[node]), 1)
            for p in pred[node]:
                dependency[p] += coeff
            if node != source:
                scores[NodeId(node)] += dependency.get(node, 0.0)

    return dict(scores)


def bridge_scores(
    graph: Graph,
    communities: dict[NodeId, int],
    limit: int = 20,
) -> list[BridgeScore]:
    """Bridge node scoring.

    Combines community touchpoints, degree, approximate betweenness, and
    articulation-point bonus into a single score per node.
    """
    articulation_set: set[NodeId] = {
        NodeId(node_id) for node_id in _articulation_points(graph)
    }
    between = approx_betweenness(graph, max_sources=64)

    touched_by_node: dict[NodeId, set[int]] = defaultdict(set)
    degree_by_node: dict[NodeId, int] = defaultdict(int)
    for _, source, target, _edge in graph.edges():
        degree_by_node[source] += 1
        degree_by_node[target] += 1
        if target in communities:
            touched_by_node[source].add(communities[target])
        if source in communities:
            touched_by_node[target].add(communities[source])

    rows: list[BridgeScore] = []
    for nid, _ in graph.nodes():
        touched = touched_by_node.get(nid, set())
        degree = degree_by_node.get(nid, 0)
        approx_bw_val = between.get(nid, 0.0)
        is_art = nid in articulation_set

        score = (
            len(touched) * 3.0
            + degree
            + approx_bw_val ** 0.5
            + (12.0 if is_art else 0.0)
        )

        if score > 0.0:
            rows.append(BridgeScore(
                node=nid,
                score=float(score),
                communities_touched=len(touched),
                degree=degree,
                approx_betweenness=approx_bw_val,
                articulation=is_art,
            ))

    rows.sort(key=lambda r: r.score, reverse=True)
    return rows[:limit]


def _articulation_points(graph: Graph) -> list[int]:
    """Internal: articulation points on undirected view (returns node IDs).
    
    Uses numba-accelerated iterative algorithm when available.
    """
    from .native import native_graph

    snapshot = native_graph(graph)
    if snapshot is not None:
        return list(snapshot.articulation_points())

    # Build undirected adjacency as CSR
    edge_set: set[tuple[int, int]] = set()
    for _, src, dst, _edge in graph.edges():
        if src != dst:
            edge_set.add((min(src.value, dst.value), max(src.value, dst.value)))

    all_nodes: list[int] = [nid.value for nid, _ in graph.nodes()]
    n = len(all_nodes)
    node_to_idx: dict[int, int] = {v: i for i, v in enumerate(all_nodes)}

    adj_map: dict[int, set[int]] = defaultdict(set)
    for u, v in edge_set:
        u_idx = node_to_idx.get(u)
        v_idx = node_to_idx.get(v)
        if u_idx is not None and v_idx is not None:
            adj_map[u_idx].add(v_idx)
            adj_map[v_idx].add(u_idx)

    # Build CSR arrays
    nnz = sum(len(v) for v in adj_map.values())
    row_ptr = np.zeros(n + 1, dtype=np.intp)
    col_idx = np.zeros(nnz, dtype=np.intp)

    for i in range(n):
        row_ptr[i + 1] = row_ptr[i] + len(adj_map.get(i, set()))

    for i in range(n):
        start = row_ptr[i]
        for j, v in enumerate(sorted(adj_map.get(i, set()))):
            col_idx[start + j] = v

    # Use numba-accelerated articulation points if available
    if has_numba_algs():
        from .numba_graph_algs import articulation_points_csr
        is_art = articulation_points_csr(row_ptr, col_idx)
        return [all_nodes[i] for i in range(n) if is_art[i]]

    # Fallback: iterative DFS (avoiding recursion limit)
    adj: dict[int, set[int]] = defaultdict(set)
    for _, src, dst, _edge in graph.edges():
        if src != dst:
            adj[src.value].add(dst.value)
            adj[dst.value].add(src.value)

    visited: set[int] = set()
    discovery: dict[int, int] = {}
    low: dict[int, int] = {}
    time_counter = [0]
    points: set[int] = set()

    # Iterative DFS with explicit stack
    for nid, _ in graph.nodes():
        nid_val = nid.value
        if nid_val in visited:
            continue

        # Stack entries: (node, parent, neighbor_index, children_count)
        stack: list[tuple[int, int | None, int, int]] = [(nid_val, None, 0, 0)]
        child_count: dict[int, int] = {nid_val: 0}

        while stack:
            node, parent, nidx, _ = stack[-1]

            if node not in visited:
                visited.add(node)
                discovery[node] = time_counter[0]
                low[node] = time_counter[0]
                time_counter[0] += 1

            neighbors = adj.get(node, set())
            neighbor_list = list(neighbors)

            if nidx < len(neighbor_list):
                # Update stack with next neighbor
                stack[-1] = (node, parent, nidx + 1, child_count.get(node, 0))
                neighbor = neighbor_list[nidx]

                if neighbor == parent:
                    continue

                if neighbor not in visited:
                    child_count[node] = child_count.get(node, 0) + 1
                    stack.append((neighbor, node, 0, 0))
                else:
                    # Back edge
                    low[node] = min(low[node], discovery[neighbor])
            else:
                # All neighbors processed, pop stack
                stack.pop()

                # Update parent's low value
                if parent is not None:
                    low[parent] = min(low[parent], low[node])

                    # Check articulation point condition
                    if low[node] >= discovery[parent]:
                        child_count[parent] = child_count.get(parent, 0) + 1
                        if parent != nid_val and child_count[parent] > 1:
                            points.add(parent)
                        elif parent == nid_val and child_count[parent] > 1:
                            points.add(parent)

    return list(points)


def find_cycles(
    graph: Graph,
    max_cycles: int = 50,
) -> dict[str, Any]:
    """Find dependency cycles via SCC (strongly connected components).

    SCCs with more than one node indicate cycles.
    """
    nx_graph = _to_nx(graph)
    sccs = list(nx.strongly_connected_components(nx_graph))

    cycles: list[dict[str, Any]] = []
    for scc in sccs:
        if len(scc) < 2:
            continue

        # Extract cycle edges
        cycle_nodes = []
        for nid in scc:
            node = graph.node(NodeId(nid))
            if node:
                cycle_nodes.append({
                    "qualified_name": node.qualified_name,
                    "kind": node.kind.value,
                })

        # Find edges within the SCC
        cycle_edges = []
        for src, dst, data in nx_graph.subgraph(scc).edges(data=True):
            src_node = graph.node(NodeId(src))
            dst_node = graph.node(NodeId(dst))
            if src_node and dst_node:
                cycle_edges.append({
                    "source": src_node.qualified_name,
                    "target": dst_node.qualified_name,
                    "kind": data.get("kind", "unknown"),
                })

        cycles.append({
            "size": len(scc),
            "nodes": cycle_nodes[:20],
            "edges": cycle_edges[:30],
        })

    cycles.sort(key=lambda c: c["size"], reverse=True)
    return {
        "cycles": cycles[:max_cycles],
        "total": len(cycles),
    }


def find_articulation_points(
    graph: Graph,
) -> dict[str, Any]:
    """Find articulation points (cut vertices in undirected view)."""
    nx_graph = _to_nx(graph)
    ug = nx_graph.to_undirected()
    points = nx.articulation_points(ug)

    result: list[dict[str, Any]] = []
    for nid in points:
        node = graph.node(NodeId(nid))
        if node:
            result.append({
                "qualified_name": node.qualified_name,
                "kind": node.kind.value,
                "name": node.name,
            })

    return {"articulation_points": result, "total": len(result)}


def find_god_nodes(
    graph: Graph,
    top: int = 20,
) -> dict[str, Any]:
    """Top nodes by PageRank."""
    from .centrality import pagerank

    scores = pagerank(graph, damping=0.85)

    gods: list[dict[str, Any]] = []
    for nid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top]:
        node = graph.node(nid)
        if node:
            gods.append({
                "qualified_name": node.qualified_name,
                "kind": node.kind.value,
                "pagerank": round(score, 6),
            })

    return {"god_nodes": gods, "total": len(gods)}


def find_large_functions(
    graph: Graph,
    min_lines: int = 30,
) -> dict[str, Any]:
    """Find functions exceeding a line threshold."""
    large: list[dict[str, Any]] = []
    for nid, node in graph.nodes():
        if node.kind in (NodeKind.FUNCTION, NodeKind.METHOD):
            lines = node.properties.get("line_count", 0)
            if isinstance(lines, (int, float)) and lines >= min_lines:
                large.append({
                    "qualified_name": node.qualified_name,
                    "kind": node.kind.value,
                    "name": node.name,
                    "line_count": lines,
                    "source_uri": node.source_uri,
                })

    large.sort(key=lambda x: x["line_count"], reverse=True)
    return {"large_functions": large, "total": len(large)}


def find_counterfactual(
    graph: Graph,
    target_qname: str,
) -> dict[str, Any]:
    """What breaks if a node or edge is removed?

    Simulates removal and reports:
    - Nodes that become unreachable
    - Broken paths
    - Affected flows
    """
    target_id = graph.find_by_qname(target_qname)
    if target_id is None:
        return {"error": f"Node not found: {target_qname}"}

    node = graph.node(target_id)
    if node is None:
        return {"error": f"Node not found: {target_qname}"}

    nx_graph = _to_nx(graph)
    original_reachable = set(nx_graph.nodes())

    # Remove the node and find what becomes unreachable
    nx_graph.remove_node(target_id.value)
    if original_reachable:
        new_reachable = set(nx_graph.nodes())
        affected = original_reachable - new_reachable
    else:
        affected = set()

    # Find incoming edges that are broken
    broken_incoming: list[dict[str, Any]] = []
    for _, src, dst, edge in graph.edges():
        if dst.value == target_id.value:
            src_node = graph.node(src)
            broken_incoming.append({
                "source": src_node.qualified_name if src_node else "?",
                "edge_kind": edge.kind.value,
            })

    # Find outgoing edges that are broken
    broken_outgoing: list[dict[str, Any]] = []
    for _, src, dst, edge in graph.edges():
        if src.value == target_id.value:
            dst_node = graph.node(dst)
            broken_outgoing.append({
                "target": dst_node.qualified_name if dst_node else "?",
                "edge_kind": edge.kind.value,
            })

    return {
        "target": target_qname,
        "node_kind": node.kind.value,
        "affected_node_count": len(affected),
        "affected_nodes": [
            graph.node(NodeId(nid)).qualified_name
            if graph.node(NodeId(nid))
            else f"node:{nid}"
            for nid in sorted(affected)[:50]
        ],
        "broken_incoming_edges": broken_incoming,
        "broken_outgoing_edges": broken_outgoing,
    }


def find_motifs(
    graph: Graph,
    pattern: str = "diamond",
) -> dict[str, Any]:
    """Subgraph motif matching.

    Patterns:
    - diamond: A→B, A→C, B→D, C→D (diamond inheritance)
    - feedback: A→B→A (mutual dependency)
    - fan_in: Multiple sources to one target
    - fan_out: One source to multiple targets
    """
    nx_graph = _to_nx(graph)

    if pattern == "diamond":
        return _find_diamonds(nx_graph, graph)
    elif pattern == "feedback":
        return _find_feedback(nx_graph, graph)
    elif pattern == "fan_in":
        return _find_fan_in(nx_graph, graph, threshold=3)
    elif pattern == "fan_out":
        return _find_fan_out(nx_graph, graph, threshold=3)
    else:
        return {"error": f"Unknown pattern: {pattern}"}


def _find_diamonds(nx_graph: nx.DiGraph, graph: Graph) -> dict[str, Any]:
    """Find diamond patterns: A→B, A→C, B→D, C→D."""
    diamonds: list[dict[str, Any]] = []

    for a in nx_graph.nodes():
        successors_a = list(nx_graph.successors(a))
        if len(successors_a) < 2:
            continue

        for i in range(len(successors_a)):
            for j in range(i + 1, len(successors_a)):
                b, c = successors_a[i], successors_a[j]
                # Find common successors of b and c
                successors_b = set(nx_graph.successors(b))
                successors_c = set(nx_graph.successors(c))
                common = successors_b & successors_c

                for d in common:
                    if d == a:
                        continue
                    nodes = [
                        graph.node(NodeId(n)).qualified_name
                        if graph.node(NodeId(n))
                        else f"node:{n}"
                        for n in [a, b, c, d]
                    ]
                    diamonds.append({
                        "pattern": "diamond",
                        "nodes": nodes,
                        "edges": [
                            f"{nodes[0]}→{nodes[1]}",
                            f"{nodes[0]}→{nodes[2]}",
                            f"{nodes[1]}→{nodes[3]}",
                            f"{nodes[2]}→{nodes[3]}",
                        ],
                    })

    return {"motifs": diamonds[:50], "pattern": "diamond", "total": len(diamonds)}


def _find_feedback(nx_graph: nx.DiGraph, graph: Graph) -> dict[str, Any]:
    """Find feedback loops: A→B→A."""
    feedback: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()

    for a, b in nx_graph.edges():
        if b in nx_graph.successors(a) and (b, a) in nx_graph.edges():
            pair = tuple(sorted([a, b]))
            if pair not in seen:
                seen.add(pair)
                nodes = [
                    graph.node(NodeId(n)).qualified_name
                    if graph.node(NodeId(n))
                    else f"node:{n}"
                    for n in [a, b]
                ]
                feedback.append({
                    "pattern": "feedback",
                    "nodes": nodes,
                    "edges": [f"{nodes[0]}→{nodes[1]}", f"{nodes[1]}→{nodes[0]}"],
                })

    return {"motifs": feedback[:50], "pattern": "feedback", "total": len(feedback)}


def _find_fan_in(nx_graph: nx.DiGraph, graph: Graph, threshold: int = 3) -> dict[str, Any]:
    """Find nodes with many incoming edges (fan-in)."""
    fan_in: list[dict[str, Any]] = []
    for nid in nx_graph.nodes():
        in_degree = nx_graph.in_degree(nid)
        if in_degree >= threshold:
            node = graph.node(NodeId(nid))
            if node:
                fan_in.append({
                    "pattern": "fan_in",
                    "qualified_name": node.qualified_name,
                    "kind": node.kind.value,
                    "in_degree": in_degree,
                })

    fan_in.sort(key=lambda x: x["in_degree"], reverse=True)
    return {"motifs": fan_in[:50], "pattern": "fan_in", "total": len(fan_in)}


def _find_fan_out(nx_graph: nx.DiGraph, graph: Graph, threshold: int = 3) -> dict[str, Any]:
    """Find nodes with many outgoing edges (fan-out)."""
    fan_out: list[dict[str, Any]] = []
    for nid in nx_graph.nodes():
        out_degree = nx_graph.out_degree(nid)
        if out_degree >= threshold:
            node = graph.node(NodeId(nid))
            if node:
                fan_out.append({
                    "pattern": "fan_out",
                    "qualified_name": node.qualified_name,
                    "kind": node.kind.value,
                    "out_degree": out_degree,
                })

    fan_out.sort(key=lambda x: x["out_degree"], reverse=True)
    return {"motifs": fan_out[:50], "pattern": "fan_out", "total": len(fan_out)}


def compute_surprise_scoring(
    graph: Graph,
    communities: dict[int, set[int]] | None = None,
) -> dict[str, Any]:
    """Unexpected cross-community / cross-language edges.

    Edges between nodes in different communities are "surprising" —
    they indicate architectural coupling that may be intentional
    (cross-cutting concern) or accidental (tight coupling).
    """
    if communities is None:
        from .communities.louvain import detect_communities
        comm_result = detect_communities(graph)
        communities = {}
        for comm in comm_result.get("communities", []):
            for node_info in comm.get("nodes", []):
                qn = node_info.get("qualified_name", "")
                nid = graph.find_by_qname(qn)
                if nid:
                    communities.setdefault(comm["id"], set()).add(nid.value)

    nx_graph = _to_nx(graph)
    surprises: list[dict[str, Any]] = []

    for _, src, dst, edge in graph.edges():
        src_comm = _find_community(src.value, communities)
        dst_comm = _find_community(dst.value, communities)

        if src_comm != dst_comm and src_comm >= 0 and dst_comm >= 0:
            src_node = graph.node(src)
            dst_node = graph.node(dst)
            if src_node and dst_node:
                surprises.append({
                    "source": src_node.qualified_name,
                    "source_kind": src_node.kind.value,
                    "target": dst_node.qualified_name,
                    "target_kind": dst_node.kind.value,
                    "edge_kind": edge.kind.value,
                    "from_community": src_comm,
                    "to_community": dst_comm,
                    "surprise_score": 1.0 / abs(src_comm - dst_comm) if src_comm != dst_comm else 0.0,
                })

    surprises.sort(key=lambda s: s["surprise_score"], reverse=True)
    return {"surprises": surprises[:50], "total": len(surprises)}


def _find_community(node_id: int, communities: dict[int, set[int]]) -> int:
    """Find which community a node belongs to."""
    for cid, nodes in communities.items():
        if node_id in nodes:
            return cid
    return -1


# Refactoring functions moved to refactoring/engine.py
from .refactoring.engine import rename_preview as _refactor_rename_preview
from .refactoring.engine import find_dead_code as _refactor_find_dead_code
from .refactoring.types import RenameEdit
from .refactoring.types import RenamePreview as _RenamePreview
from .refactoring.types import RenameStats as _RenameStats


def rename_preview(graph: Graph, qname: str, new_name: str) -> dict[str, Any] | None:
    """Preview rename of a symbol without modifying the graph.

    Backward-compatible wrapper around refactoring.rename_preview.
    Returns dict format for API compatibility.
    """
    result = _refactor_rename_preview(graph, qname, new_name)
    if result is None:
        return None
    return result.to_dict()


def find_dead_code(graph: Graph, limit: int = 100) -> dict[str, Any]:
    """Find dead code: functions/classes with no callers, no test refs, no importers.

    Backward-compatible wrapper around refactoring.find_dead_code.
    Returns dict format for API compatibility.
    """
    dead = _refactor_find_dead_code(graph, limit=limit)
    return {"dead_code": dead, "total": len(dead)}


def call_resolution_stats(graph: Graph) -> dict[str, Any]:
    """Call-resolution coverage: how many Calls edges land on real
    definitions versus unresolved call::* placeholders.

    Returns a dict with resolved, unresolved, total, and rate.
    A low rate means many call sites dead-end at placeholders.
    """
    resolved = 0
    unresolved = 0
    for _, src, dst, edge in graph.edges():
        if edge.kind != EdgeKind.CALLS:
            continue
        dst_node = graph.node(dst)
        is_placeholder = (
            dst_node is not None
            and dst_node.qualified_name.startswith("call::")
        )
        if is_placeholder:
            unresolved += 1
        else:
            resolved += 1

    total = resolved + unresolved
    rate = resolved / total if total > 0 else 1.0

    return {
        "resolved": resolved,
        "unresolved": unresolved,
        "total": total,
        "rate": round(rate, 4),
    }


def _to_nx(graph: Graph) -> nx.DiGraph:
    """Convert the code graph to NetworkX DiGraph."""
    nx_graph = nx.DiGraph()
    for nid, node in graph.nodes():
        nx_graph.add_node(nid.value, **{
            "qualified_name": node.qualified_name,
            "kind": node.kind.value,
        })
    for _, src, dst, edge in graph.edges():
        nx_graph.add_edge(src.value, dst.value, **{
            "kind": edge.kind.value,
        })
    return nx_graph
