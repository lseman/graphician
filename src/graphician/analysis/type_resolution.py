"""Resolution of ``type::<Name>`` placeholder nodes.

``type::<Name>`` placeholder nodes are left by supertype extraction (e.g.
Java's ``emit_superclass`` / ``emit_interfaces``). This pass runs after
the merged graph is assembled: for every placeholder with a unique
same-named ``Class`` / ``Trait`` node elsewhere in the graph, every
``Inherits`` / ``Implements`` edge pointing at the placeholder is rewired
to the real node instead.

Ambiguous names (multiple candidates) are left as placeholders —
matching the project's conservative-resolution philosophy used by
``call_resolution.py``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..core.edge import Edge, EdgeKind
from ..core.graph import Graph
from ..core.id import EdgeId, NodeId
from ..core.node import NodeKind


def resolve_type_placeholders(graph: Graph) -> int:
    """Resolve ``type::<Name>`` placeholders left by supertype extraction.

    Returns the number of edges rewired.
    """
    by_name = build_real_type_by_name(graph)

    # Find all placeholder nodes
    placeholders: list[tuple[int, str]] = []
    for nid, node in graph.nodes():
        if node.qualified_name.startswith("type::") and node.kind == NodeKind.CLASS:
            name = node.qualified_name[len("type::") :]
            placeholders.append((nid.value, name))

    rewired = 0
    edges_to_remove: list[EdgeId] = []
    edges_to_add: list[tuple[int, int, EdgeKind]] = []

    for placeholder_id, name in placeholders:
        candidates = by_name.get(name, [])
        if len(candidates) != 1:
            continue  # Ambiguous — leave the placeholder alone

        real_id = candidates[0]
        if real_id == placeholder_id:
            continue

        # Find edges pointing at this placeholder
        for edge_id, src_id, dst_id, edge in iter_edges(graph):
            if dst_id != placeholder_id:
                continue
            if edge.kind not in (EdgeKind.INHERITS, EdgeKind.IMPLEMENTS):
                continue

            edges_to_remove.append(edge_id)
            edges_to_add.append((src_id, real_id, edge.kind))

    # Apply edge changes
    for src_id, dst_id, kind in edges_to_add:
        graph.add_edge(
            NodeId(src_id),
            NodeId(dst_id),
            Edge.extracted(kind),
        )
        rewired += 1

    graph.remove_edges_by_id(edges_to_remove)

    # Drop orphaned placeholders
    orphaned = [
        placeholder_id
        for placeholder_id, _ in placeholders
        if has_no_edges(graph, placeholder_id)
    ]
    for nid in orphaned:
        graph.remove_node(NodeId(nid))

    return rewired


def build_real_type_by_name(graph: Graph) -> dict[str, list[int]]:
    """Map bare type name → every non-placeholder Class/Trait node."""
    by_name: dict[str, list[int]] = defaultdict(list)
    for nid, node in graph.nodes():
        if node.kind in (NodeKind.CLASS, NodeKind.TRAIT):
            if not node.qualified_name.startswith("type::"):
                by_name[node.name].append(nid.value)
    return dict(by_name)


def has_no_edges(graph: Graph, node_id: int) -> bool:
    """Check if a node has no incoming or outgoing edges."""
    try:
        for _ in graph.in_neighbors(NodeId(node_id)):
            return False
        for _ in graph.out_neighbors(NodeId(node_id)):
            return False
    except Exception:
        pass
    return True


def iter_edges(graph: Graph):
    """Iterate over all edges as (src_id, dst_id, edge)."""
    try:
        for edge_id, src, dst, edge in graph.edges():
            yield edge_id, src.value, dst.value, edge
    except Exception:
        yield from []
