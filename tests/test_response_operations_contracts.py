from __future__ import annotations

from pathlib import Path

from ariadne_py.core import Edge, EdgeKind, Graph, Node, NodeKind
from ariadne_py.interfaces.cli.response.flows import handle_flows, handle_test_coverage
from ariadne_py.interfaces.cli.response.impact import handle_god_nodes, handle_impact, hub_nodes_json
from ariadne_py.interfaces.cli.response.paths import _simple_paths, handle_paths
from ariadne_py.interfaces.cli.response.search import (
    _fuzzy_match,
    _score_match,
    find_related_json,
    handle_context_pack,
    handle_search,
)
from ariadne_py.interfaces.cli.response.token_savings import (
    approx_tokens,
    format_panel,
    per_file_tokens,
    token_savings_for_graph,
)


def _graph(tmp_path: Path):
    source_path = tmp_path / "app.py"
    source_path.write_text("def main():\n    helper()\n" * 30)
    graph = Graph()
    main = graph.add_node(Node.new(NodeKind.FUNCTION, "app::main").with_source(str(source_path), 1, 2))
    helper = graph.add_node(Node.new(NodeKind.FUNCTION, "app::helper").with_source(str(source_path), 3, 4))
    test = graph.add_node(Node.new(NodeKind.FUNCTION, "tests::test_main").with_source(str(source_path), 5, 6))
    graph.add_edge(main, helper, Edge.extracted(EdgeKind.CALLS))
    graph.add_edge(main, test, Edge.extracted(EdgeKind.TESTED_BY))
    flow = graph.add_node(
        Node.new(NodeKind.FLOW, "flow::app::main")
        .with_property("entry_qualified_name", "app::main")
        .with_property("entry_name", "main")
        .with_property("criticality", 0.8)
        .with_property("node_count", 2)
        .with_property("depth", 1)
        .with_property("is_test_flow", False)
    )
    return graph, main, helper, test, flow


def test_path_response_resolves_names_and_returns_weighted_nodes(tmp_path: Path) -> None:
    graph, main, helper, *_ = _graph(tmp_path)
    result = handle_paths(graph, {"from": "app::main", "to": "app::helper", "max_hops": 2})
    assert result["operation"] == "paths"
    assert result["paths"][0]["nodes"][0]["qualified_name"] == "app::main"
    assert result["paths"][0]["nodes"][-1]["qualified_name"] == "app::helper"
    assert _simple_paths(graph, main, helper, 2, 5)
    assert "valid qualified names" in handle_paths(graph, {"from": "missing", "to": "also-missing"})["error"]


def test_impact_hub_and_god_node_responses_cover_directions_and_limits(tmp_path: Path) -> None:
    graph, main, helper, *_ = _graph(tmp_path)
    out = handle_impact(graph, {"target": "app::main", "direction": "out", "max_hops": 2})
    assert out["total"] == 2
    incoming = handle_impact(graph, {"target": "app::helper", "direction": "in"})
    assert incoming["impacted"][0]["qualified_name"] == "app::main"
    assert handle_impact(graph, {})["total"] == 0
    assert "target not found" in handle_impact(graph, {"target": "missing"})["error"]
    assert handle_god_nodes(graph, {"min_degree": 1, "limit": 2})["hits"]
    assert hub_nodes_json(graph, limit=1)["total"] == 1


def test_flow_and_target_test_coverage_responses(tmp_path: Path) -> None:
    graph, *_ = _graph(tmp_path)
    flows = handle_flows(graph, {"limit": 1})
    assert flows["total"] == 1
    assert flows["hits"][0]["entry_name"] == "main"
    covered = handle_test_coverage(graph, "unused.db", {"target": "app::main"})
    assert covered["result"]["covered"][0]["tests"][0]["qualified_name"] == "tests::test_main"
    missing = handle_test_coverage(graph, "unused.db", {"target": "app::helper"})
    assert missing["result"]["missing_count"] == 1
    assert handle_test_coverage(graph, "unused.db", {})["result"]["missing_count"] == 0


def test_token_savings_measures_real_files_and_formats_cli_panel(tmp_path: Path) -> None:
    graph, *_ = _graph(tmp_path)
    files = per_file_tokens(graph)
    assert len(files) == 1
    totals = token_savings_for_graph(graph, mode="cli", include_files=True)
    assert totals["raw_context_tokens"] > totals["graph_context_tokens"]
    assert totals["files"][0]["file"].endswith("app.py")
    assert "Token Savings" in totals["panel"]
    assert approx_tokens("") == 1
    assert "Saved:" in format_panel(100, 25, 75)
    assert token_savings_for_graph(Graph())["files_measured"] == 0


def test_search_response_scores_exact_prefix_substring_and_fuzzy_matches(tmp_path: Path) -> None:
    graph, *_ = _graph(tmp_path)
    exact = handle_search(graph, {"query": "helper", "limit": 1})
    assert exact["hits"][0]["qualified_name"] == "app::helper"
    assert exact["hits"][0]["score"] == 1.0
    assert handle_search(graph, {"query": "app::"})["hits"]
    assert handle_search(graph, {"query": "elp"})["hits"]
    assert handle_search(graph, {"query": "hlpr"})["hits"]
    assert handle_search(graph, {"query": ""})["total"] == 0
    assert _fuzzy_match("hlpr", "helper")
    assert not _fuzzy_match("xyz", "helper")


def test_context_pack_and_related_responses_follow_graph_neighborhood(tmp_path: Path) -> None:
    graph, *_ = _graph(tmp_path)
    context = handle_context_pack(
        graph, {"target": "app::main", "max_files": 5, "token_budget": 100}
    )
    assert context["target"] == "app::main"
    assert any(path.endswith("app.py") for path in context["files"])
    assert handle_context_pack(graph, {})["files"] == []
    assert "target not found" in handle_context_pack(graph, {"target": "missing"})["error"]

    related = find_related_json(graph, "app::main", limit=10)
    assert related["total"] == 2
    assert {item["qualified_name"] for item in related["related"]} == {
        "app::helper", "tests::test_main"
    }
    assert find_related_json(graph, "missing")["total"] == 0
