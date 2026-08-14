from ariadne_py.analysis.quality_baseline import evaluate_cochange, impact_accuracy
from ariadne_py.core import Edge, EdgeKind, Graph, Node, NodeKind


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
