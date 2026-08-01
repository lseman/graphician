"""Flow detection: compute_flows and entry point detection."""

from __future__ import annotations

from collections import deque
from typing import Any

from ...core.edge import Edge, EdgeKind
from ...core.id import NodeId
from ...core.node import Node, NodeKind
from .types import FlowOptions
from .entry_points import _is_python_framework_entry
from .trace import _trace_flow, _compute_criticality, _is_test_node


def compute_flows_with_options(
    graph,
    options: FlowOptions,
) -> int:
    """Detect entry points, trace flows, and materialise them into the graph.

    Returns the number of flows produced.
    """
    opts = options or FlowOptions()
    entries = _detect_entry_points(graph)
    produced = 0

    for entry in entries:
        entry_node = graph.node(entry)
        if entry_node is None:
            continue

        entry_qn = entry_node.qualified_name
        entry_name = entry_node.name
        is_test_entry = _is_test_node(entry_node)

        members = _trace_flow(graph, entry, opts)
        if len(members) < opts.min_flow_size:
            continue

        member_count = len(members)
        depth_reached = max((d for _, d in members), default=0)
        criticality = _compute_criticality(graph, members, entry_name, is_test_entry)

        # Identity: flow:: prefix + entry qname. Stable across re-runs.
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

        # Idempotency: prune old MemberOf / EntryOf edges into this flow.
        existing: set[tuple[NodeId, EdgeKind]] = set()
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


def _detect_entry_points(graph) -> list[NodeId]:
    """Detect entry points: test nodes, main-like names, framework-decorated, orphans."""
    entries: list[NodeId] = []
    for nid, node in graph.nodes():
        if node.kind not in (NodeKind.FUNCTION, NodeKind.METHOD):
            continue
        if node.qualified_name.startswith("call::"):
            continue
        if _is_test_node(node):
            entries.append(nid)
            continue
        if node.name in ("main", "__main__"):
            entries.append(nid)
            continue
        if _is_framework_entry(node):
            entries.append(nid)
            continue
        # Orphan: zero incoming Calls edges.
        has_caller = any(
            e.kind == EdgeKind.CALLS for _, e in graph.in_neighbors(nid)
        )
        if not has_caller:
            entries.append(nid)
    return entries


def all_flows(graph) -> list[NodeId]:
    """Read-side helper: collect IDs of all Flow nodes, sorted by criticality desc."""
    flows: list[tuple[NodeId, float, str]] = []
    for nid, node in graph.nodes():
        if node.kind != NodeKind.FLOW:
            continue
        crit = node.properties.get("criticality", 0.0)
        if isinstance(crit, str):
            try:
                crit = float(crit)
            except (ValueError, TypeError):
                crit = 0.0
        flows.append((nid, crit, node.qualified_name))
    flows.sort(key=lambda x: (-x[1], x[2]))
    return [nid for nid, _, _ in flows]


def flows_through(graph, node: NodeId) -> list[NodeId]:
    """Read-side helper: flows that contain `node`. Returns flow node ids."""
    result: list[NodeId] = []
    for flow_id, edge in graph.out_neighbors(node):
        if edge.kind in (EdgeKind.MEMBER_OF, EdgeKind.ENTRY_OF):
            result.append(flow_id)
    return result


def affected_flows(graph, changed: list[NodeId]) -> list[NodeId]:
    """Read-side helper: union of flows_through over changed nodes, ranked by criticality."""
    seen: set[NodeId] = set()
    hits: list[tuple[NodeId, float, str]] = []
    for n in changed:
        for flow in flows_through(graph, n):
            if flow in seen:
                continue
            seen.add(flow)
            flow_node = graph.node(flow)
            if flow_node is None:
                continue
            crit = flow_node.properties.get("criticality", 0.0)
            if isinstance(crit, str):
                try:
                    crit = float(crit)
                except (ValueError, TypeError):
                    crit = 0.0
            hits.append((flow, crit, flow_node.qualified_name))
    hits.sort(key=lambda x: (-x[1], x[2]))
    return [nid for nid, _, _ in hits]
