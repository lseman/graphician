"""Execution flow detection."""

from __future__ import annotations

from ...core.node import NodeKind
from .entry_points import detect_entry_points
from .trace import _compute_criticality, _is_test_node, trace_flow
from .types import FlowOptions


def compute_flows(graph, options=None):
    """Detect entry points, trace flows, and materialise them into the graph.

    Returns the number of flows produced.
    """
    from ...core.edge import Edge, EdgeKind
    from ...core.id import NodeId
    from ...core.node import Node, NodeKind

    opts = options or FlowOptions()
    entries = detect_entry_points(graph)
    native_members = None
    if graph.node_count() >= 1_000 and len(entries) >= 32:
        try:
            from ...analysis.native import native_graph

            snapshot = native_graph(graph)
            if snapshot is not None:
                placeholders = [
                    node_id.value
                    for node_id, node in graph.nodes()
                    if node.qualified_name.startswith("call::")
                ]
                traced = snapshot.trace_flows(
                    [entry.value for entry in entries],
                    placeholders,
                    opts.max_depth,
                    opts.max_nodes_per_flow,
                )
                native_members = [
                    [(NodeId(node_id), depth) for node_id, depth in members]
                    for members in traced
                ]
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
            native_members = None
    produced = 0

    for entry_index, entry in enumerate(entries):
        entry_node = graph.node(entry)
        if entry_node is None:
            continue

        entry_qn = entry_node.qualified_name
        entry_name = entry_node.name
        is_test_entry = _is_test_node(entry_node)

        members = (
            native_members[entry_index]
            if native_members is not None
            else trace_flow(graph, entry, opts)
        )
        if len(members) < opts.min_flow_size:
            continue

        member_count = len(members)
        depth_reached = max((d for _, d in members), default=0)
        criticality = _compute_criticality(graph, members, entry_name, is_test_entry)

        flow_qn = f"flow::{entry_qn}"
        flow_node = (
            Node.new(NodeKind.FLOW, flow_qn)
            .with_property("entry_qualified_name", entry_qn)
            .with_property("entry_name", entry_name)
            .with_property("depth", depth_reached)
            .with_property("node_count", member_count)
            .with_property("criticality", round(criticality, 4))
            .with_property("is_test_flow", is_test_entry)
        )
        flow_id = graph.add_node(flow_node)

        existing = set()
        for src, edge in graph.in_neighbors(flow_id):
            if edge.kind in (EdgeKind.MEMBER_OF, EdgeKind.ENTRY_OF):
                existing.add((src, edge.kind))

        if (entry, EdgeKind.ENTRY_OF) not in existing:
            graph.add_edge(entry, flow_id, Edge.extracted(EdgeKind.ENTRY_OF))

        for member, _depth in members:
            if member == entry:
                continue
            if (member, EdgeKind.MEMBER_OF) not in existing:
                graph.add_edge(member, flow_id, Edge.extracted(EdgeKind.MEMBER_OF))

        produced += 1

    return produced


def all_flows(graph) -> list:
    """Return all Flow nodes in the graph."""
    return [nid for nid, node in graph.nodes() if node.kind == NodeKind.FLOW]


def flows_through(graph, node):
    """Return all flows that pass through a given node."""
    result = []
    for nid, flow_node in graph.nodes():
        if flow_node.kind != NodeKind.FLOW:
            continue
        for member, _ in graph.in_neighbors(nid):
            if member == node:
                result.append(nid)
                break
    return result


def affected_flows(graph, changed):
    """Return all flows affected by a list of changed node IDs."""
    changed_set = set(changed)
    result = []
    for nid, flow_node in graph.nodes():
        if flow_node.kind != NodeKind.FLOW:
            continue
        for member, _ in graph.in_neighbors(nid):
            if member in changed_set:
                result.append(nid)
                break
    return result
