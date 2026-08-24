"""Graph snapshot comparison.

Compares two Graph states and reports added/removed/modified nodes and
edges, plus community membership changes when community data is present.

Nodes are matched by qualified name. Edges are matched by
(source_qualified_name, target_qualified_name, kind).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DiffNode:
    """A node in a diff."""
    id: int
    qualified_name: str
    kind: str
    source: str | None = None


@dataclass
class DiffEdge:
    """An edge in a diff."""
    id: int
    src: int
    dst: int
    kind: str


@dataclass
class CommunityChange:
    """Community membership change for a node."""
    node_id: int
    qualified_name: str
    old_community: str | None
    new_community: str | None


@dataclass
class GraphDiff:
    """Diff between two graph snapshots."""
    added_nodes: list[DiffNode] = field(default_factory=list)
    removed_nodes: list[DiffNode] = field(default_factory=list)
    modified_nodes: list[DiffNode] = field(default_factory=list)
    added_edges: list[DiffEdge] = field(default_factory=list)
    removed_edges: list[DiffEdge] = field(default_factory=list)
    community_changes: list[CommunityChange] = field(default_factory=list)


def graph_diff(base: Any, head: Any) -> dict[str, Any]:
    """Compute a diff between two graph snapshots.

    Args:
        base: The older graph snapshot.
        head: The newer graph snapshot.

    Returns:
        Dict with added_nodes, removed_nodes, modified_nodes,
        added_edges, removed_edges, community_changes.
    """
    diff = GraphDiff()

    # Build lookup maps by qualified name
    base_by_qname: dict[str, int] = {}
    head_by_qname: dict[str, int] = {}
    for nid, node in base.nodes():
        base_by_qname[node.qualified_name] = nid.value
    for nid, node in head.nodes():
        head_by_qname[node.qualified_name] = nid.value

    base_qnames = set(base_by_qname.keys())
    head_qnames = set(head_by_qname.keys())

    # Added nodes: in head but not in base
    for qname, head_id in head_by_qname.items():
        if qname not in base_qnames:
            node = head.node_by_value(head_id)
            if node is not None:
                diff.added_nodes.append(DiffNode(
                    id=head_id,
                    qualified_name=node.qualified_name,
                    kind=node.kind.value,
                    source=node.source_uri,
                ))

    # Removed nodes: in base but not in head
    for qname, base_id in base_by_qname.items():
        if qname not in head_qnames:
            node = base.node_by_value(base_id)
            if node is not None:
                diff.removed_nodes.append(DiffNode(
                    id=base_id,
                    qualified_name=node.qualified_name,
                    kind=node.kind.value,
                    source=node.source_uri,
                ))

    # Modified nodes: in both but different
    for qname in base_qnames & head_qnames:
        base_id = base_by_qname[qname]
        head_id = head_by_qname[qname]
        base_node = base.node_by_value(base_id)
        head_node = head.node_by_value(head_id)
        if base_node is None or head_node is None:
            continue
        if (base_node.kind != head_node.kind
                or base_node.name != head_node.name
                or base_node.source_uri != head_node.source_uri
                or base_node.line_start != head_node.line_start
                or base_node.line_end != head_node.line_end
                or base_node.properties != head_node.properties):
            diff.modified_nodes.append(DiffNode(
                id=head_id,
                qualified_name=head_node.qualified_name,
                kind=head_node.kind.value,
                source=head_node.source_uri,
            ))

    # Build edge keys by (src_qname, dst_qname, kind)
    base_edge_keys: dict[tuple[str, str, str], list[tuple[int, int, int]]] = {}
    head_edge_keys: dict[tuple[str, str, str], list[tuple[int, int, int]]] = {}

    for edge_id, src, dst, edge in head.edges():
        src_node = head.node(src)
        dst_node = head.node(dst)
        src_qn = src_node.qualified_name if src_node else ""
        dst_qn = dst_node.qualified_name if dst_node else ""
        key = (src_qn, dst_qn, edge.kind.value)
        head_edge_keys.setdefault(key, []).append((edge_id.value, src.value, dst.value))

    for edge_id, src, dst, edge in base.edges():
        src_node = base.node(src)
        dst_node = base.node(dst)
        src_qn = src_node.qualified_name if src_node else ""
        dst_qn = dst_node.qualified_name if dst_node else ""
        key = (src_qn, dst_qn, edge.kind.value)
        base_edge_keys.setdefault(key, []).append((edge_id.value, src.value, dst.value))

    base_key_set = set(base_edge_keys.keys())
    head_key_set = set(head_edge_keys.keys())

    # Added edges: in head but not in base
    for key in head_key_set - base_key_set:
        for edge_id, src_id, dst_id in head_edge_keys[key]:
            diff.added_edges.append(DiffEdge(
                id=edge_id,
                src=src_id,
                dst=dst_id,
                kind=key[2],
            ))

    # Removed edges: in base but not in head
    for key in base_key_set - head_key_set:
        for edge_id, src_id, dst_id in base_edge_keys[key]:
            diff.removed_edges.append(DiffEdge(
                id=edge_id,
                src=src_id,
                dst=dst_id,
                kind=key[2],
            ))

    # Community changes
    for qname in base_qnames & head_qnames:
        base_id = base_by_qname[qname]
        head_id = head_by_qname[qname]
        base_node = base.node_by_value(base_id)
        head_node = head.node_by_value(head_id)
        if base_node is None or head_node is None:
            continue
        old_comm = base_node.properties.get("community")
        new_comm = head_node.properties.get("community")
        if old_comm != new_comm:
            diff.community_changes.append(CommunityChange(
                node_id=head_id,
                qualified_name=head_node.qualified_name,
                old_community=str(old_comm) if old_comm is not None else None,
                new_community=str(new_comm) if new_comm is not None else None,
            ))

    return _diff_to_dict(diff)


def _diff_to_dict(diff: GraphDiff) -> dict[str, Any]:
    """Convert GraphDiff to serializable dict."""
    return {
        "added_nodes": [
            {"id": n.id, "qualified_name": n.qualified_name,
             "kind": n.kind, "source": n.source}
            for n in diff.added_nodes
        ],
        "removed_nodes": [
            {"id": n.id, "qualified_name": n.qualified_name,
             "kind": n.kind, "source": n.source}
            for n in diff.removed_nodes
        ],
        "modified_nodes": [
            {"id": n.id, "qualified_name": n.qualified_name,
             "kind": n.kind, "source": n.source}
            for n in diff.modified_nodes
        ],
        "added_edges": [
            {"id": e.id, "src": e.src, "dst": e.dst, "kind": e.kind}
            for e in diff.added_edges
        ],
        "removed_edges": [
            {"id": e.id, "src": e.src, "dst": e.dst, "kind": e.kind}
            for e in diff.removed_edges
        ],
        "community_changes": [
            {"node_id": c.node_id, "qualified_name": c.qualified_name,
             "old_community": c.old_community,
             "new_community": c.new_community}
            for c in diff.community_changes
        ],
    }
