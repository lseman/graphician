from __future__ import annotations

from pathlib import Path

from graphician.core import Edge, EdgeKind, Graph, Node, NodeKind
from graphician.interfaces.cli.response import tool_response
from graphician.interfaces.cli.response.analysis import gaps_json
from graphician.persistence.store import GraphStore


def _write_graph(path: Path) -> tuple[str, str]:
    graph = Graph()
    caller = Node.new(NodeKind.FUNCTION, "pkg::sql_query").with_source("pkg/a.py", 1, 3)
    callee = Node.new(NodeKind.FUNCTION, "pkg::save").with_source("pkg/b.py", 4, 8)
    test = Node.new(NodeKind.FUNCTION, "tests::test_sql_query").with_source(
        "tests/test_a.py", 1, 2
    )
    test.with_property("is_test", True)
    caller_id = graph.add_node(caller)
    callee_id = graph.add_node(callee)
    test_id = graph.add_node(test)
    graph.add_edge(caller_id, callee_id, Edge.extracted(EdgeKind.CALLS))
    graph.add_edge(caller_id, test_id, Edge.extracted(EdgeKind.TESTED_BY))
    with GraphStore(path) as store:
        store.save_graph(graph)
    return caller.qualified_name, callee.qualified_name


def test_agent_context_neighbors_coverage_and_motifs_are_real(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.db"
    caller, callee = _write_graph(db_path)

    context = tool_response(str(db_path), "minimal_context", {"target": caller})
    callees = tool_response(str(db_path), "callees_of", {"target": caller})
    callers = tool_response(str(db_path), "callers_of", {"target": callee})
    coverage = tool_response(str(db_path), "test_coverage", {"target": caller})
    counterfactual = tool_response(
        str(db_path), "counterfactual", {"target": caller, "direction": "out"}
    )
    motifs = tool_response(
        str(db_path), "motifs", {"built_in": "security_audit"}
    )

    assert [node["qualified_name"] for node in context["nodes"]] == [
        caller,
        callee,
        "tests::test_sql_query",
    ]
    assert [item["qualified_name"] for item in callees["callees"]] == [callee]
    assert [item["qualified_name"] for item in callers["callers"]] == [caller]
    assert coverage["result"]["covered"][0]["qualified_name"] == caller
    assert counterfactual["unreachable_count"] == 2
    assert motifs["match_count"] == 1


def test_topology_diagnostics_and_graphml_use_stable_node_ids(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.db"
    caller, _callee = _write_graph(db_path)
    output = tmp_path / "graph.graphml"

    diagnostics = tool_response(str(db_path), "diagnostics", {})
    related = tool_response(str(db_path), "find_related", {"target": caller})
    exported = tool_response(
        str(db_path), "export_graphml", {"output": str(output)}
    )

    assert "error" not in diagnostics
    assert diagnostics["call_resolution"]["rate"] == 1.0
    assert related["total"] == 2
    assert exported["written"] is True
    assert "<edge source=" in output.read_text()


def test_gaps_reports_only_actual_disconnected_nodes() -> None:
    graph = Graph()
    connected = graph.add_node(Node.new(NodeKind.FUNCTION, "pkg::connected"))
    peer = graph.add_node(Node.new(NodeKind.FUNCTION, "pkg::peer"))
    graph.add_node(Node.new(NodeKind.FUNCTION, "pkg::isolated"))
    graph.add_edge(connected, peer, Edge.extracted(EdgeKind.CALLS))

    response = gaps_json(graph)
    names = {item["qualified_name"] for item in response["hits"]}

    assert "pkg::isolated" in names
    assert "pkg::connected" in names  # Connected but intentionally low-degree.


def test_remaining_agent_operations_are_wired_and_write_outputs(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.db"
    _write_graph(db_path)
    report_path = tmp_path / "report.md"
    wiki_path = tmp_path / "wiki"

    patterns = tool_response(str(db_path), "patterns", {})
    dedup = tool_response(str(db_path), "dedup", {})
    report = tool_response(str(db_path), "report", {"output": str(report_path)})
    wiki = tool_response(str(db_path), "wiki", {"output": str(wiki_path)})

    assert "error" not in patterns
    assert dedup["operation"] == "dedup"
    assert report["written"] is True
    assert report_path.read_text().startswith("# Graphician Graph Report")
    assert wiki["operation"] == "wiki"
    assert (wiki_path / "index.md").is_file()


def test_graph_merge_preserves_edges_from_sparse_node_ids() -> None:
    destination = Graph()
    existing = destination.add_node(Node.new(NodeKind.FUNCTION, "pkg::existing"))

    source = Graph()
    removed = source.add_node(Node.new(NodeKind.FUNCTION, "pkg::removed"))
    caller = source.add_node(Node.new(NodeKind.FUNCTION, "pkg::caller"))
    target = source.add_node(Node.new(NodeKind.FUNCTION, "pkg::target"))
    source.add_edge(caller, target, Edge.extracted(EdgeKind.CALLS))
    source.remove_node(removed)

    destination.merge(source)

    merged_caller = destination.find_by_qname("pkg::caller")
    merged_target = destination.find_by_qname("pkg::target")
    assert merged_caller is not None and merged_target is not None
    assert existing != merged_caller
    assert [node_id for node_id, _edge in destination.out_neighbors(merged_caller)] == [
        merged_target
    ]
