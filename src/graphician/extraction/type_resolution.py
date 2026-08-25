"""Resolution of ``type::<Name>`` placeholder nodes.

Some language extractors (notably Java's superclass / interface emission)
create a fresh ``type::<Name>`` placeholder node for every extends/implements
target, because resolving the reference to the actual defined class or
interface would require whole-graph knowledge at single-file extraction
time.  This mirrors ``call::<name>`` placeholders for calls — except
nothing previously resolved these.

This pass runs after the merged graph is assembled: for every
``type::<Name>`` placeholder with a unique same-named ``Class``/``Trait``
node elsewhere in the graph, every ``Inherits``/``Implements`` edge
pointing at the placeholder is rewired to point at the real node instead,
and the placeholder is dropped once it has no remaining edges.

Ambiguous names (multiple candidates) are left as placeholders, matching
the project's conservative-resolution philosophy used by
``call_resolution``.
"""

from __future__ import annotations

import logging

from .._extract import plan_type_resolution
from ..core.edge import Edge, EdgeKind
from ..core.graph import Graph
from ..core.id import EdgeId, NodeId
from ..core.node import NodeKind

logger = logging.getLogger(__name__)


def resolve_type_placeholders(graph: Graph) -> int:
    """Resolve ``type::<Name>`` placeholders left by supertype extraction.

    Returns the number of edges rewired (not removed).
    """
    if plan_type_resolution is not None:
        try:
            rewires, orphaned = plan_type_resolution(
                [
                    (nid.value, node.kind.value, node.name, node.qualified_name)
                    for nid, node in graph.nodes()
                ],
                [
                    (eid.value, src.value, dst.value, edge.kind.value)
                    for eid, src, dst, edge in graph.edges()
                ],
            )
            for _edge_id, src, dst, kind in rewires:
                graph.add_edge(NodeId(src), NodeId(dst), Edge.extracted(EdgeKind(kind)))
            graph.remove_edges_by_id([EdgeId(edge_id) for edge_id, *_ in rewires])
            for node_id in orphaned:
                graph.remove_node(NodeId(node_id))
            return len(rewires)
        except Exception:
            logger.warning("Native type resolution failed; using Python fallback", exc_info=True)

    return _resolve_type_placeholders_python(graph)


def _resolve_type_placeholders_python(graph: Graph) -> int:
    """Reference implementation and fallback for native resolution."""
    by_name = _build_real_type_by_name(graph)

    # Collect all type:: placeholders (they are emitted as Class nodes).
    placeholders: list[tuple[NodeId, str]] = []
    for nid, node in graph.nodes():
        if node.qualified_name.startswith("type::") and node.kind == NodeKind.CLASS:
            bare = node.qualified_name[len("type::"):]
            placeholders.append((nid, bare))

    edges_to_remove: list[EdgeId] = []
    edges_to_add: list[tuple[NodeId, NodeId, EdgeKind]] = []

    for placeholder_id, name in placeholders:
        candidates = by_name.get(name)
        if candidates is None or len(candidates) != 1:
            continue  # Ambiguous or missing — leave the placeholder alone.
        real_id = candidates[0]
        if real_id == placeholder_id:
            continue  # Already the real node.

        for edge_id, src, dst, edge in graph.edges():
            if dst != placeholder_id or edge.kind not in (
                EdgeKind.INHERITS,
                EdgeKind.IMPLEMENTS,
            ):
                continue
            edges_to_remove.append(edge_id)
            edges_to_add.append((src, real_id, edge.kind))

    for src, dst, kind in edges_to_add:
        graph.add_edge(src, dst, Edge.extracted(kind))

    if edges_to_remove:
        graph.remove_edges_by_id(edges_to_remove)

    # Drop placeholders that now have no remaining edges.
    for placeholder_id, _name in placeholders:
        in_nbrs = list(graph.in_neighbors(placeholder_id))
        out_nbrs = list(graph.out_neighbors(placeholder_id))
        if not in_nbrs and not out_nbrs:
            graph.remove_node(placeholder_id)

    return len(edges_to_add)


def _build_real_type_by_name(graph: Graph) -> dict[str, list[NodeId]]:
    """Map bare type name → every non-placeholder Class/Trait node with that name."""
    by_name: dict[str, list[NodeId]] = {}
    for nid, node in graph.nodes():
        if (
            node.kind in (NodeKind.CLASS, NodeKind.TRAIT)
            and not node.qualified_name.startswith("type::")
        ):
            by_name.setdefault(node.name, []).append(nid)
    return by_name
