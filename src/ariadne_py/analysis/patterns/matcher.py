"""Pattern matching engine: detect_patterns and _match_pattern."""

from __future__ import annotations

from typing import Any

from ...core.edge import EdgeKind
from ...core.id import NodeId
from ...core.node import NodeKind
from .builtin import _builtin_patterns
from .types import FrameworkPattern, PatternMatch


def detect_patterns(
    graph,
    patterns: list[FrameworkPattern] | None = None,
) -> list[PatternMatch]:
    """Run all framework pattern detections against the graph.

    Returns a list of PatternMatch objects for patterns that matched.
    """
    if patterns is None:
        patterns = _builtin_patterns()

    matches: list[PatternMatch] = []
    for pattern in patterns:
        result = _match_pattern(graph, pattern)
        if result is not None:
            matches.append(result)

    matches.sort(key=lambda m: -m.confidence)
    return matches


def _match_pattern(graph, pattern: FrameworkPattern) -> PatternMatch | None:
    """Check if a single pattern matches the graph."""
    # Collect candidate nodes matching signature names
    candidate_nodes: list[dict[str, Any]] = []
    for nid, node in graph.nodes():
        # Check signature names
        if pattern.signature_names:
            name_matches = any(
                sig.lower() in node.name.lower() or sig.lower() in node.qualified_name.lower()
                for sig in pattern.signature_names
            )
            if not name_matches:
                continue
        # Check import patterns
        if pattern.import_patterns:
            import_matches = any(
                imp in (node.source_uri or "").lower()
                for imp in pattern.import_patterns
            )
            if not import_matches:
                continue
        # Check node kind filter
        if pattern.required_node_kinds and node.kind not in pattern.required_node_kinds:
            continue
        candidate_nodes.append({
            "id": nid.value,
            "qualified_name": node.qualified_name,
            "name": node.name,
            "kind": node.kind.value,
            "source_uri": node.source_uri,
        })

    if len(candidate_nodes) < pattern.min_nodes:
        return None
    if len(candidate_nodes) > pattern.max_nodes:
        return None

    # Check required edge kinds among candidates
    candidate_ids = {n["id"] for n in candidate_nodes}
    matched_edges: list[dict[str, Any]] = []
    if pattern.required_edge_kinds:
        for _, src, dst, edge in graph.edges():
            if src.value in candidate_ids and dst.value in candidate_ids:
                if edge.kind in pattern.required_edge_kinds:
                    matched_edges.append({
                        "source_id": src.value,
                        "target_id": dst.value,
                        "kind": edge.kind.value,
                    })
        if not matched_edges and pattern.required_edge_kinds:
            return None

    # Confidence: based on node count and edge coverage
    confidence = min(0.5 + len(candidate_nodes) * 0.05 + len(matched_edges) * 0.03, 1.0)

    return PatternMatch(
        pattern_id=pattern.id,
        display_name=pattern.display_name,
        framework=pattern.framework,
        category=pattern.category.value,
        confidence=round(confidence, 3),
        matched_nodes=candidate_nodes,
        matched_edges=matched_edges,
    )
