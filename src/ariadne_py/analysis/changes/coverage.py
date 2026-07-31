"""Test coverage analysis for change analysis."""

from __future__ import annotations

from typing import Any

from ...core.edge import EdgeKind
from ...core.graph import Graph
from ...core.node import NodeKind


def compute_test_coverage(
    graph: Graph,
    base: str | None = None,
) -> dict[str, Any]:
    """Compute test coverage for symbols.

    Identifies symbols without test coverage (no tested_by edge).
    """
    # Find all tested symbols
    tested: set[str] = set()
    for _, src, dst, edge in graph.edges():
        if edge.kind == EdgeKind.TESTED_BY:
            src_node = graph.node(src)
            if src_node:
                tested.add(src_node.qualified_name)

    # Find untested symbols
    untested: list[dict[str, Any]] = []
    for nid, node in graph.nodes():
        if node.kind in (NodeKind.FUNCTION, NodeKind.METHOD, NodeKind.CLASS):
            if node.qualified_name not in tested:
                untested.append({
                    "qualified_name": node.qualified_name,
                    "kind": node.kind.value,
                    "name": node.name,
                    "source_uri": node.source_uri,
                })

    untested.sort(key=lambda x: x["qualified_name"])

    total = len(untested) + len(tested)
    coverage = len(tested) / total if total > 0 else 0.0

    return {
        "coverage": round(coverage, 4),
        "tested_count": len(tested),
        "untested_count": total,
        "untested": untested[:100],
    }
