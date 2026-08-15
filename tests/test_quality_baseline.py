from pathlib import Path

import pytest

from ariadne_py.analysis.quality_baseline import (
    _percentile,
    benchmark_store,
    evaluate_cochange,
    impact_accuracy,
)
from ariadne_py.core import Edge, EdgeKind, Graph, Node, NodeKind
from ariadne_py.persistence.store import GraphStore


def test_impact_accuracy_reports_precision_recall_and_f1() -> None:
    result = impact_accuracy(["a.py", "b.py"], ["b.py", "c.py"])

    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.false_negatives == 1
    assert result.precision == 0.5
    assert result.recall == 0.5
    assert result.f1 == 0.5


def test_cochange_benchmark_uses_real_graph_impact_predictions() -> None:
    graph = Graph()
    callee = graph.add_node(
        Node(NodeKind.FUNCTION, "callee", "callee", source_uri="src/callee.py")
    )
    caller = graph.add_node(
        Node(NodeKind.FUNCTION, "caller", "caller", source_uri="src/caller.py")
    )
    graph.add_node(
        Node(NodeKind.FUNCTION, "other", "other", source_uri="src/other.py")
    )
    graph.add_edge(caller, callee, Edge(kind=EdgeKind.CALLS))

    result = evaluate_cochange(graph, ["src/callee.py", "src/caller.py"])

    assert result.true_positives == 2
    assert result.false_positives == 0
    assert result.false_negatives == 0


def test_operational_baseline_reports_store_and_resolution_metrics(tmp_path: Path) -> None:
    graph = Graph()
    caller = graph.add_node(Node.new(NodeKind.FUNCTION, "caller"))
    callee = graph.add_node(Node.new(NodeKind.FUNCTION, "callee"))
    graph.add_edge(caller, callee, Edge.extracted(EdgeKind.CALLS))

    ticks = iter(float(value) for value in range(12))
    with GraphStore(tmp_path / "graph.db") as store:
        store.save_graph(graph)
        result = benchmark_store(store, samples=3, timer=lambda: next(ticks))

    assert result.node_count == 2
    assert result.edge_count == 1
    assert result.store_load_p50_ms == 1000.0
    assert result.search_p95_ms == 1000.0
    assert result.call_resolution_rate == 1.0
    assert result.sample_count == 3


def test_operational_baseline_validates_sample_count(tmp_path: Path) -> None:
    with (
        GraphStore(tmp_path / "graph.db") as store,
        pytest.raises(ValueError, match="samples must be at least 1"),
    ):
        benchmark_store(store, samples=0)


def test_percentile_uses_nearest_rank() -> None:
    assert _percentile([0.1, 0.2, 0.3, 0.4, 0.5], 50) == 0.3
    assert _percentile([0.1, 0.2, 0.3, 0.4, 0.5], 95) == 0.5
