"""VF2-style subgraph isomorphism engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...core.edge import EdgeKind
from ...core.graph import Graph
from ...core.id import NodeId
from .dsl import Motif


@dataclass
class MotifMatch:
    """A single match found by the engine."""
    node_map: dict[int, NodeId] = field(default_factory=dict)
    edges: list[tuple[str, str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_map": {k: v.value for k, v in self.node_map.items()},
            "edges": self.edges,
        }


def find_motifs(graph: Graph, motif: Motif, limit: int = 50) -> list[MotifMatch]:
    """Find all motif matches in the graph using VF2-style backtracking."""
    valid, msg = motif.validate()
    if not valid or not motif.nodes:
        return []

    # Pre-compute pattern adjacency: node_id → [(target_id, edge_kind)]
    pattern_adj: dict[int, list[tuple[int, EdgeKind]]] = {}
    for e in motif.edges:
        if e.kind is not None:
            pattern_adj.setdefault(e.from_id, []).append((e.to_id, e.kind))

    edge_kinds: dict[tuple[NodeId, NodeId], set[EdgeKind]] = {}
    for _edge_id, source, target, edge in graph.edges():
        edge_kinds.setdefault((source, target), set()).add(edge.kind)

    # Pre-filter candidates for each pattern node
    candidate_sets: list[list[NodeId]] = []
    for node_constraint in motif.nodes:
        candidates: list[NodeId] = []
        for nid, node in graph.nodes():
            if node_constraint.kind is not None and node.kind != node_constraint.kind:
                continue
            if node_constraint.name is not None and not node_constraint.name.matches(node.name):
                continue
            if node_constraint.min_degree is not None:
                deg = graph.out_degree(nid) + graph.in_degree(nid)
                if deg < node_constraint.min_degree:
                    continue
            candidates.append(nid)
        candidate_sets.append(candidates)

    # Run backtracking search
    results: list[MotifMatch] = []
    current_map: dict[int, NodeId] = {}

    def backtrack(depth: int) -> None:
        if len(results) >= limit:
            return
        if depth == len(motif.nodes):
            if _validate_edges(motif, current_map, edge_kinds):
                edges_out: list[tuple[str, str, str]] = []
                for e in motif.edges:
                    src_id = current_map.get(e.from_id)
                    dst_id = current_map.get(e.to_id)
                    if src_id and dst_id:
                        src_node = graph.node(src_id)
                        dst_node = graph.node(dst_id)
                        if src_node and dst_node:
                            edges_out.append((
                                src_node.qualified_name,
                                e.kind.value if e.kind else "any",
                                dst_node.qualified_name,
                            ))
                results.append(MotifMatch(
                    node_map=dict(current_map),
                    edges=edges_out,
                ))
            return

        pattern_node_idx = depth
        for graph_node_id in candidate_sets[pattern_node_idx]:
            if len(results) >= limit:
                return
            if any(v == graph_node_id for v in current_map.values()):
                continue
            if not _check_consistency(
                pattern_adj,
                pattern_node_idx,
                graph_node_id,
                current_map,
                edge_kinds,
            ):
                continue
            current_map[pattern_node_idx] = graph_node_id
            backtrack(depth + 1)
            del current_map[pattern_node_idx]

    backtrack(0)
    return results


def _check_consistency(
    pattern_adj: dict[int, list[tuple[int, EdgeKind]]],
    pattern_node_idx: int,
    graph_node_id: NodeId,
    current_map: dict[int, NodeId],
    edge_kinds: dict[tuple[NodeId, NodeId], set[EdgeKind]],
) -> bool:
    """Check consistency of assigning graph_node_id to pattern_node_idx."""
    for assigned_pidx, assigned_gid in current_map.items():
        pattern_has_edge = (
            any(t == assigned_pidx for t, _ in pattern_adj.get(pattern_node_idx, []))
            or any(t == pattern_node_idx for t, _ in pattern_adj.get(assigned_pidx, []))
        )
        if pattern_has_edge:
            if (
                (assigned_gid, graph_node_id) not in edge_kinds
                and (graph_node_id, assigned_gid) not in edge_kinds
            ):
                return False
    return True


def _validate_edges(
    motif: Motif,
    current_map: dict[int, NodeId],
    edge_kinds: dict[tuple[NodeId, NodeId], set[EdgeKind]],
) -> bool:
    """Validate that all motif edges exist in the graph for the current mapping."""
    for e in motif.edges:
        src_id = current_map.get(e.from_id)
        dst_id = current_map.get(e.to_id)
        if src_id is None or dst_id is None:
            return False
        kinds = edge_kinds.get((src_id, dst_id), set())
        if e.kind is not None and e.kind not in kinds:
            return False
        if e.kind is None and not kinds:
            return False
    return True
