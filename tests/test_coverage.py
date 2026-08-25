from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from graphician.analysis.coverage import graph_coverage
from graphician.core.edge import Edge, EdgeKind
from graphician.core.graph import Graph
from graphician.core.node import Node, NodeKind
from graphician.interfaces.cli import main
from graphician.persistence.store import GraphStore


def _coverage_graph() -> Graph:
    graph = Graph()
    first = graph.add_node(
        Node.new(NodeKind.FUNCTION, "pkg::first").with_source("pkg/a.py", 1, 2)
    )
    second = graph.add_node(Node.new(NodeKind.FUNCTION, "pkg::second"))
    test = graph.add_node(
        Node.new(NodeKind.FUNCTION, "tests::test_first")
        .with_source("tests/test_a.py", 1, 2)
        .with_property("is_test", True)
    )
    placeholder = graph.add_node(Node.new(NodeKind.FUNCTION, "call::external"))
    graph.add_edge(first, second, Edge.extracted(EdgeKind.CALLS))
    graph.add_edge(second, placeholder, Edge.ambiguous(EdgeKind.CALLS))
    graph.add_edge(first, test, Edge.extracted(EdgeKind.TESTED_BY))
    return graph


def test_graph_coverage_reports_rates_and_breakdowns() -> None:
    result = graph_coverage(_coverage_graph(), example_limit=5)

    assert result["summary"] == {
        "health_score": 0.6667,
        "nodes": 4,
        "edges": 3,
        "source_files": 2,
    }
    assert result["source_location"] == {
        "covered": 2,
        "total": 3,
        "rate": 0.6667,
        "missing_examples": ["pkg::second"],
    }
    assert result["call_resolution"] == {
        "resolved": 1,
        "unresolved": 1,
        "total": 2,
        "rate": 0.5,
    }
    assert result["function_connectivity"]["caller_rate"] == 0.3333
    assert result["function_connectivity"]["callee_rate"] == 0.6667
    assert result["test_links"]["rate"] == 0.5
    assert result["connectivity"]["rate"] == 1.0
    assert result["node_kinds"] == {"function": 4}
    assert result["edge_kinds"] == {"calls": 2, "tested_by": 1}


def test_graph_coverage_empty_graph_is_well_defined() -> None:
    result = graph_coverage(Graph())

    assert result["summary"]["health_score"] == 0.0
    assert result["summary"]["nodes"] == 0
    assert result["call_resolution"]["rate"] == 0.0
    assert result["connectivity"]["rate"] == 0.0


def test_graph_coverage_rejects_negative_example_limit() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        graph_coverage(Graph(), example_limit=-1)


def test_graph_coverage_compares_file_nodes_with_build_manifest() -> None:
    graph = Graph()
    graph.add_node(
        Node.new(NodeKind.FILE, "file::src/app.py").with_source(
            "/repo/src/app.py", 1, 2
        )
    )

    result = graph_coverage(
        graph,
        expected_files=["src/app.py", "src/missing.py"],
        example_limit=5,
    )

    assert result["file_extraction"] == {
        "covered": 1,
        "total": 2,
        "rate": 0.5,
        "missing_examples": ["src/missing.py"],
        "manifest_available": True,
    }


def test_coverage_cli_reads_built_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    db_path = tmp_path / "graphician.db"
    with GraphStore(db_path) as store:
        store.save_graph(_coverage_graph())

    monkeypatch.setattr(sys, "argv", ["graphician", "-d", str(db_path), "coverage", "--top", "0"])
    main()

    result = json.loads(capsys.readouterr().out)
    assert result["operation"] == "coverage"
    assert result["source_location"]["missing_examples"] == []
    assert result["summary"]["nodes"] == 4


def test_build_then_coverage_cli_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text(
        "def helper():\n    return 1\n\ndef main():\n    return helper()\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "graphician.db"

    monkeypatch.setattr(
        sys, "argv", ["graphician", "-d", str(db_path), "build", str(project)]
    )
    main()
    build_output = capsys.readouterr()
    assert "Built:" in build_output.err

    monkeypatch.setattr(
        sys, "argv", ["graphician", "-d", str(db_path), "coverage", "--top", "5"]
    )
    main()
    result = json.loads(capsys.readouterr().out)

    assert result["file_extraction"]["rate"] == 1.0
    assert result["source_location"]["rate"] == 1.0
    assert result["call_resolution"] == {
        "resolved": 1,
        "unresolved": 0,
        "total": 1,
        "rate": 1.0,
    }
