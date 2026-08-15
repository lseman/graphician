from __future__ import annotations

from ariadne_py.analysis.changes.coverage import compute_test_coverage
from ariadne_py.analysis.changes.detection import _parse_diff, detect_changes
from ariadne_py.analysis.changes.risk import compute_risk
from ariadne_py.core import Edge, EdgeKind, Graph, Node, NodeKind


def _located(kind: NodeKind, qname: str, path: str, start: int, end: int, **props) -> Node:
    node = Node.new(kind, qname).with_source(path, start, end)
    node.properties.update(props)
    return node


def test_diff_detection_maps_hunks_symbols_risk_and_flows() -> None:
    graph = Graph()
    changed = graph.add_node(_located(NodeKind.FUNCTION, "app::authenticate", "src/app.py", 8, 20))
    caller = graph.add_node(_located(NodeKind.FUNCTION, "app::main", "src/app.py", 1, 6))
    flow = graph.add_node(Node.new(NodeKind.FLOW, "flow::app::main").with_property("entry", "app::main"))
    graph.add_edge(caller, changed, Edge.extracted(EdgeKind.CALLS))
    graph.add_edge(changed, flow, Edge.extracted(EdgeKind.MEMBER_OF))
    diff = """diff --git a/src/app.py b/src/app.py
--- a/src/app.py
+++ b/src/app.py
@@ -9,2 +9,3 @@
-    return old_token
+    token = issue_token()
+    return token
"""

    hunks = _parse_diff(diff)
    assert len(hunks) == 1
    assert hunks[0].file_path == "src/app.py"
    result = detect_changes(graph, diff)
    assert result["total_changes"] == 1
    assert result["changes"][0]["affected_symbols"] == ["app::authenticate"]
    assert result["changes"][0]["risk_score"] == 1.5
    assert result["affected_flows"][0]["flow"] == "flow::app::main"


def test_risk_and_test_coverage_use_tested_by_source_semantics() -> None:
    graph = Graph()
    secure = graph.add_node(Node.new(NodeKind.FUNCTION, "auth::password_token_admin"))
    helper = graph.add_node(Node.new(NodeKind.FUNCTION, "app::helper"))
    test = graph.add_node(Node.new(NodeKind.FUNCTION, "tests::test_secure").with_property("is_test", True))
    graph.add_edge(secure, test, Edge.extracted(EdgeKind.TESTED_BY))
    for index in range(12):
        caller = graph.add_node(Node.new(NodeKind.FUNCTION, f"app::caller_{index}"))
        graph.add_edge(caller, secure, Edge.extracted(EdgeKind.CALLS))

    coverage = compute_test_coverage(graph)
    assert coverage["tested_count"] == 1
    assert all(item["qualified_name"] != "auth::password_token_admin" for item in coverage["untested"])

    risks = compute_risk(graph)["risk_scores"]
    secure_risk = next(item for item in risks if item["qualified_name"] == "auth::password_token_admin")
    helper_risk = next(item for item in risks if item["qualified_name"] == "app::helper")
    assert secure_risk["test"] == 0
    assert secure_risk["security"] > 0.3
    assert "high_connectivity" in secure_risk["reasons"]
    assert helper_risk["reasons"] == ["no_test_coverage"]


def test_empty_change_analysis_contracts() -> None:
    assert detect_changes(Graph(), "not a diff") == {
        "changes": [], "affected_flows": [], "total_changes": 0, "max_risk": 0.0
    }
    assert compute_risk(Graph()) == {"risk_scores": [], "total": 0}
    assert compute_test_coverage(Graph())["coverage"] == 0
