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

from ..core.graph import Graph
from ..core.id import NodeId
from ..core.node import NodeKind


def resolve_type_placeholders(graph: Graph) -> int:
    """Resolve ``type::<Name>`` placeholders left by supertype extraction.

    Returns the number of edges rewired.
    """
    # Keep this historical import path as a shallow compatibility adapter;
    # extraction.type_resolution owns the native dispatch and Python fallback.
    from ..extraction.type_resolution import resolve_type_placeholders as resolve

    return resolve(graph)


def build_real_type_by_name(graph: Graph) -> dict[str, list[int]]:
    """Map bare type name → every non-placeholder Class/Trait node."""
    by_name: dict[str, list[int]] = defaultdict(list)
    for nid, node in graph.nodes():
        if (
            node.kind in (NodeKind.CLASS, NodeKind.TRAIT)
            and not node.qualified_name.startswith("type::")
        ):
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
