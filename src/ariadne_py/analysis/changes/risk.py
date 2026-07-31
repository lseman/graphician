"""Risk scoring for change analysis."""

from __future__ import annotations

from typing import Any

from ...core.edge import EdgeKind
from ...core.graph import Graph
from ...core.id import NodeId
from ...core.node import NodeKind
from .types import RiskScore


def _structural_risk(graph: Graph, nid: NodeId) -> float:
    """Compute structural risk (0-1)."""
    in_deg = sum(1 for _ in graph.in_neighbors(nid))
    out_deg = sum(1 for _ in graph.out_neighbors(nid))

    # High degree = high risk (more things depend on this)
    total = in_deg + out_deg
    if total == 0:
        return 0.0

    # Normalize to 0-1
    risk = min(total / 20.0, 1.0)  # 20+ edges = max risk
    return risk


def _test_coverage_risk(graph: Graph, nid: NodeId) -> float:
    """Compute test coverage risk (0-1)."""
    # Check if tested
    has_test = False
    for _, src, dst, edge in graph.edges():
        if edge.kind == EdgeKind.TESTED_BY and dst.value == nid.value:
            has_test = True
            break

    return 0.0 if has_test else 1.0


def _security_risk(node) -> float:
    """Compute security risk based on name patterns (0-1)."""
    security_keywords = [
        "auth", "password", "token", "secret", "key", "encrypt",
        "decrypt", "hash", "permission", "grant", "admin", "root",
    ]
    name_lower = node.name.lower()
    score = 0.0
    for kw in security_keywords:
        if kw in name_lower:
            score += 0.3
    return min(score, 1.0)


def _risk_reasons(
    structural: float,
    test: float,
    security: float,
    node,
) -> list[str]:
    """Generate human-readable risk reasons."""
    reasons = []
    if structural > 0.5:
        reasons.append("high_connectivity")
    if test > 0.5:
        reasons.append("no_test_coverage")
    if security > 0.3:
        reasons.append("security_sensitive")
    if not reasons:
        reasons.append("low_risk")
    return reasons


def compute_risk(
    graph: Graph,
    base: str | None = None,
    top: int = 25,
) -> dict[str, Any]:
    """Compute risk scores for symbols.

    Combines structural risk (degree centrality, bridge nodes),
    test coverage gaps, and security-sensitive patterns.
    """
    risk_scores: list[RiskScore] = []

    for nid, node in graph.nodes():
        if node.kind not in (NodeKind.FUNCTION, NodeKind.METHOD, NodeKind.CLASS):
            continue

        structural = _structural_risk(graph, nid)
        test = _test_coverage_risk(graph, nid)
        security = _security_risk(node)

        overall = 0.4 * structural + 0.35 * test + 0.25 * security
        reasons = _risk_reasons(structural, test, security, node)

        risk_scores.append(RiskScore(
            qualified_name=node.qualified_name,
            kind=node.kind.value,
            overall=overall,
            structural=structural,
            test=test,
            security=security,
            reasons=reasons,
        ))

    risk_scores.sort(key=lambda r: r.overall, reverse=True)

    return {
        "risk_scores": [
            {
                "qualified_name": r.qualified_name,
                "kind": r.kind,
                "overall": round(r.overall, 4),
                "structural": round(r.structural, 4),
                "test": round(r.test, 4),
                "security": round(r.security, 4),
                "reasons": r.reasons,
            }
            for r in risk_scores[:top]
        ],
        "total": len(risk_scores),
    }
