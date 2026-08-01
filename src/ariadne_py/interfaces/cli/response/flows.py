"""Execution flow operations: list flows, affected flows, blast radius, test coverage.

Mirrors the Rust ``flows.rs`` module.
"""

from __future__ import annotations

import logging
from typing import Any

from ....core.edge import EdgeKind
from ....core.node import NodeKind

logger = logging.getLogger(__name__)


def handle_flows(graph, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """List all execution flows in the graph.

    Args:
        graph: The code graph.
        params: Optional parameters (limit, etc.).

    Returns:
        Flow listing with hits and totals.
    """
    if params is None:
        params = {}
    limit = int(params.get("limit", 25))

    hits: list[dict[str, Any]] = []
    for nid, node in graph.nodes():
        if node.kind != NodeKind.FLOW:
            continue
        props = node.properties
        hits.append({
            "qualified_name": node.qualified_name,
            "entry_qualified_name": props.get("entry_qualified_name"),
            "entry_name": props.get("entry_name"),
            "criticality": props.get("criticality"),
            "node_count": props.get("node_count"),
            "depth": props.get("depth"),
            "is_test_flow": props.get("is_test_flow"),
        })

    hits.sort(key=lambda x: (x.get("criticality") or 0, x.get("node_count") or 0), reverse=True)
    return {
        "operation": "flows",
        "hits": hits[:limit],
        "total": len(hits),
        "truncated": len(hits) > limit,
    }


def handle_affected_flows(graph, db_path: str, params: dict[str, Any]) -> dict[str, Any]:
    """Find flows affected by recent changes.

    Args:
        graph: The code graph.
        db_path: Path to the database.
        params: Parameters (base, limit).

    Returns:
        Affected flows with base and hits.
    """
    from .temporal import detect_changes_json

    base = params.get("base", "HEAD~1")
    limit = int(params.get("limit", 10))

    analysis = detect_changes_json(graph, base, max_depth=2)
    payload = analysis.get("affected_flows", {
        "hits": [],
        "total": 0,
        "truncated": False,
    })

    truncated_hits = payload.get("hits", [])[:limit]
    return {
        "operation": "affected_flows",
        "base": base,
        "hits": truncated_hits,
        "total": payload.get("total", len(truncated_hits)),
    }


def handle_blast_radius(graph, db_path: str, params: dict[str, Any]) -> dict[str, Any]:
    """Compute blast radius of changes from a git base.

    Args:
        graph: The code graph.
        db_path: Path to the database.
        params: Parameters (base, max_depth, limit).

    Returns:
        Blast radius analysis with changed files, symbols, impacted nodes.
    """
    from .temporal import detect_changes_json

    base = params.get("base", "HEAD~1")
    max_depth = int(params.get("max_depth", 2))
    limit = int(params.get("limit", 25))

    analysis = detect_changes_json(graph, base, max_depth=max_depth)

    changed_files = analysis.get("changed_files", [])[:limit]
    changed_symbols = analysis.get("changed_symbols", analysis.get("changed_nodes", []))[:limit]
    impacted = analysis.get("impacted", [])[:limit]
    test_cov = analysis.get("test_coverage", {
        "covered": [], "missing": [], "missing_count": 0,
    })

    return {
        "operation": "blast_radius",
        "base": base,
        "risk": analysis.get("risk", "low"),
        "risk_score": analysis.get("risk_score", 0.0),
        "changed_files": changed_files,
        "changed_files_count": len(changed_files),
        "changed_symbols": changed_symbols,
        "changed_symbols_count": len(changed_symbols),
        "impacted": impacted,
        "impacted_count": len(impacted),
        "test_coverage": test_cov,
    }


def handle_test_coverage(graph, db_path: str, params: dict[str, Any]) -> dict[str, Any]:
    """Test coverage for changed files or a specific target.

    Args:
        graph: The code graph.
        db_path: Path to the database.
        params: Parameters (base, target).

    Returns:
        Test coverage with covered/missing nodes.
    """
    from .temporal import detect_changes_json

    base = params.get("base")
    target = params.get("target")

    if base:
        # Coverage for changed files
        analysis = detect_changes_json(graph, base, max_depth=2)
        result = analysis.get("test_coverage", {
            "covered": [], "missing": [], "missing_count": 0,
        })
    elif target:
        # Coverage for a specific target
        nid = _resolve(graph, target)
        covered: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []

        if nid is not None:
            node = graph.node(nid)
            if node is None:
                for _, n in graph.nodes():
                    if _ids_match(n, nid):
                        node = n
                        break
            if node and node.kind in (NodeKind.FUNCTION, NodeKind.METHOD):
                tests = []
                if hasattr(graph, "out_neighbors"):
                    for test_id, edge in graph.out_neighbors(nid):
                        if edge.kind == EdgeKind.TESTED_BY:
                            test_node = graph.node(test_id)
                            if test_node:
                                tests.append({
                                    "qualified_name": test_node.qualified_name,
                                    "source_uri": test_node.source_uri,
                                })
                elif hasattr(graph, "edges"):
                    for _, src, dst, edge in graph.edges():
                        if src == nid and edge.kind == EdgeKind.TESTED_BY:
                            test_node = graph.node(dst)
                            if test_node:
                                tests.append({
                                    "qualified_name": test_node.qualified_name,
                                    "source_uri": test_node.source_uri,
                                })

                entry = {
                    "qualified_name": node.qualified_name,
                    "kind": str(node.kind),
                    "source_uri": node.source_uri,
                    "tests": tests,
                }
                if tests:
                    covered.append(entry)
                else:
                    missing.append(entry)

        result = {
            "target": target,
            "covered": covered,
            "missing": missing,
            "missing_count": len(missing),
        }
    else:
        result = {"covered": [], "missing": [], "missing_count": 0}

    return {"operation": "test_coverage", "result": result}


# ── Helpers ────────────────────────────────────────────────────────


def _resolve(graph, target: str) -> Any | None:
    """Resolve a target string to a node ID."""
    nid = graph.find_by_qname(target) if hasattr(graph, "find_by_qname") else None
    if nid is not None:
        return nid
    try:
        return int(target)
    except (ValueError, TypeError):
        pass
    for _, node in graph.nodes():
        if node.qualified_name == target or node.name == target:
            return node.id if hasattr(node, "id") else None
    return None


def _ids_match(node: Any, node_id: Any) -> bool:
    """Check if a node matches a node ID."""
    if not hasattr(node, "id"):
        return False
    return node.id == node_id
