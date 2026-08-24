"""Multi-factor risk scoring for changed symbols.

CRG-style risk scoring: each changed function gets a 0.0-1.0 score
combining five weighted factors:

1. **Flow participation** (sum of flow criticalities, cap 0.25)
2. **Community crossing** (0.05 per cross-community caller, cap 0.15)
3. **Test coverage** (0.30 untested → 0.05 with 5+ tests)
4. **Security sensitivity** (0.20 if name matches security keywords)
5. **Caller count** (callers/20, cap 0.10)

This is a complementary analysis to impact scoring, which answers
"what depends on this?" Risk scoring answers "how risky is it to
change this?"
"""

from __future__ import annotations

from typing import Any

from ...core.edge import EdgeKind
from ...core.graph import Graph
from ...core.id import NodeId
from ...core.node import Node, NodeKind
from .types import RiskScore, RiskFactor


# ── Security-sensitive keywords ──────────────────────────────────────

_SECURITY_KEYWORDS: list[str] = [
    "auth", "password", "token", "secret", "encrypt", "decrypt",
    "hash", "sign", "verify", "permission", "access", "credential",
    "login", "logout", "session", "csrf", "xss", "sanitize",
    "validate", "injection", "privilege", "sudo", "root", "admin",
    "key", "certificate", "https", "tls", "ssl", "payment",
    "billing", "invoice", "transaction", "debit", "credit",
    "grant",
]


def _normalize_name(name: str) -> str:
    """Normalize a name for keyword matching: lowercase, non-alpha → space."""
    return "".join(c if c.isalnum() else " " for c in name.lower())


# ── Factor 1: Flow participation ─────────────────────────────────────

_FLOW_WEIGHT = 0.25


def _flow_participation(graph: Graph, nid: NodeId) -> float:
    """Sum of flow criticalities for flows this node belongs to, capped at 0.25."""
    total = 0.0
    for flow_id, edge in graph.out_neighbors(nid):
        if edge.kind in (EdgeKind.MEMBER_OF, EdgeKind.ENTRY_OF):
            flow_node = graph.node(flow_id)
            if flow_node is not None:
                crit = flow_node.properties.get("criticality")
                if crit is not None and isinstance(crit, (int, float)):
                    total += crit
    return min(total, _FLOW_WEIGHT)


# ── Factor 2: Community crossing ─────────────────────────────────────

_COMMUNITY_WEIGHT = 0.05
_COMMUNITY_MAX = 0.15


def _community_crossing(graph: Graph, nid: NodeId) -> float:
    """0.05 per cross-file caller, capped at 0.15."""
    changed_node = graph.node(nid)
    if changed_node is None:
        return 0.0

    changed_file = changed_node.source_uri or ""
    caller_count = 0

    for caller_id, _edge in graph.in_neighbors(nid):
        if _edge.kind != EdgeKind.CALLS:
            continue
        caller_node = graph.node(caller_id)
        if caller_node is None:
            continue
        caller_file = caller_node.source_uri or ""
        if caller_file and changed_file and caller_file != changed_file:
            caller_count += 1

    return min(caller_count * _COMMUNITY_WEIGHT, _COMMUNITY_MAX)


# ── Factor 3: Test coverage ──────────────────────────────────────────

_TEST_SCORES: dict[int, float] = {
    0: 0.30,  # untested → highest risk
    1: 0.25,
    2: 0.20,
    3: 0.15,
    4: 0.10,
}
_TEST_MAX_TESTS = 5
_TEST_MIN_SCORE = 0.05


def _test_coverage(graph: Graph, nid: NodeId) -> float:
    """Graduated test coverage risk: 0.30 untested → 0.05 with 5+ tests."""
    # Check TESTED_BY edges (src → test) and CALLS edges (test → src)
    test_ids: set[NodeId] = set()

    # TESTED_BY: nid --TESTED_BY--> test_node
    for test_node_id, edge in graph.out_neighbors(nid):
        if edge.kind == EdgeKind.TESTED_BY:
            test_node = graph.node(test_node_id)
            if test_node is not None:
                is_test = test_node.properties.get("is_test")
                if is_test is True:
                    test_ids.add(test_node_id)

    # CALLS: test_node --CALLS--> nid
    for caller_id, edge in graph.in_neighbors(nid):
        if edge.kind != EdgeKind.CALLS:
            continue
        caller_node = graph.node(caller_id)
        if caller_node is not None:
            is_test = caller_node.properties.get("is_test")
            if is_test is True:
                test_ids.add(caller_id)

    test_count = len(test_ids)

    if test_count <= 4:
        return _TEST_SCORES.get(test_count, 0.05)
    return _TEST_MIN_SCORE


# ── Factor 4: Security sensitivity ───────────────────────────────────

_SECURITY_WEIGHT = 0.20


def _security_sensitivity(node: Node) -> float:
    """0.20 if normalized name contains any security keyword as substring."""
    norm = _normalize_name(node.name)
    return _SECURITY_WEIGHT if any(kw in norm for kw in _SECURITY_KEYWORDS) else 0.0


# ── Factor 5: Caller count ───────────────────────────────────────────

_CALLER_WEIGHT = 1.0 / 20.0
_CALLER_MAX = 0.10


def _caller_count(graph: Graph, nid: NodeId) -> float:
    """callers/20, capped at 0.10."""
    count = sum(1 for _ in graph.in_neighbors(nid))
    return min(count * _CALLER_WEIGHT, _CALLER_MAX)


# ── Public API ───────────────────────────────────────────────────────


def compute_risk(
    graph: Graph,
    base: str | None = None,
    top: int = 25,
) -> dict[str, Any]:
    """Compute multi-factor risk scores for all nodes in the graph.

    Uses the CRG-style 5-factor model:
    1. Flow participation (0.25 max)
    2. Community crossing (0.15 max)
    3. Test coverage (0.30 max)
    4. Security sensitivity (0.20 max)
    5. Caller count (0.10 max)

    Args:
        graph: The code graph.
        base: Unused (kept for API compatibility).
        top: Maximum risk scores to return.

    Returns:
        Dict with "risk_scores" list and "total" count.
    """
    results: list[dict[str, Any]] = []

    for nid, node in graph.nodes():
        if node.kind not in (NodeKind.FUNCTION, NodeKind.METHOD, NodeKind.CLASS):
            continue

        flow = _flow_participation(graph, nid)
        community = _community_crossing(graph, nid)
        test = _test_coverage(graph, nid)
        security = _security_sensitivity(node)
        callers = _caller_count(graph, nid)

        overall = round(flow + community + test + security + callers, 4)

        # Build factor breakdown for response
        factors: list[dict[str, Any]] = [
            {"name": "flow_participation", "weight": _FLOW_WEIGHT, "score": round(flow, 4)},
            {"name": "community_crossing", "weight": _COMMUNITY_MAX, "score": round(community, 4)},
            {"name": "test_coverage", "weight": 0.30, "score": round(test, 4)},
            {"name": "security_sensitivity", "weight": _SECURITY_WEIGHT, "score": round(security, 4)},
            {"name": "caller_count", "weight": _CALLER_MAX, "score": round(callers, 4)},
        ]

        # Build human-readable reasons
        reasons = []
        if flow > 0:
            reasons.append("in_critical_flow")
        if community > 0:
            reasons.append("cross_module_calls")
        if test >= 0.20:
            reasons.append("low_test_coverage")
        elif test > 0:
            reasons.append("some_test_coverage")
        if security > 0:
            reasons.append("security_sensitive")
        if callers > 0:
            reasons.append("many_callers")
        if not reasons:
            reasons.append("low_risk")

        # Risk level classification
        if overall >= 0.85:
            level = "CRITICAL"
        elif overall >= 0.70:
            level = "HIGH"
        elif overall >= 0.40:
            level = "MEDIUM"
        else:
            level = "LOW"

        # Legacy keys for backward compatibility
        # test: 0 if tested (1+ tests), 1 if not (old semantics)
        legacy_test = 0.0 if test < 0.30 else 1.0
        # structural: callers/20 scaled to old-style range (old used in+out degree)
        legacy_structural = min((callers + flow + community) / 0.4 * 0.10, 1.0) if callers > 0 else 0.0
        # security: keep original per-keyword style for compatibility
        legacy_security = security

        results.append({
            "qualified_name": node.qualified_name,
            "kind": node.kind.value,
            "overall": overall,
            "level": level,
            "factors": factors,
            "reasons": reasons,
            # Backward compatibility keys
            "test": legacy_test,
            "structural": legacy_structural,
            "security": legacy_security,
        })

    results.sort(key=lambda r: r["overall"], reverse=True)

    return {
        "risk_scores": results[:top],
        "total": len(results),
    }
