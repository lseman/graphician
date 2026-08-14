"""Regression tests for Rust parser extraction details."""

from ariadne_py.core.edge import EdgeKind
from ariadne_py.core.graph import Graph
from ariadne_py.extraction.languages.parsers.rust import extract_file


def test_method_call_captures_positional_receiver(tmp_path) -> None:
    source = tmp_path / "receiver.rs"
    source.write_text(
        "fn populate(graph: &mut Graph) { graph.add_node(node); }",
        encoding="utf-8",
    )
    graph = Graph()

    extract_file(source, graph)

    caller = graph.find_by_qname("receiver::populate")
    placeholder = graph.find_by_qname("call::add_node")
    assert caller is not None and placeholder is not None
    edges = [
        edge
        for target, edge in graph.out_neighbors(caller)
        if target == placeholder and edge.kind == EdgeKind.CALLS
    ]
    assert len(edges) == 1
    assert edges[0].properties["call_receiver"] == "graph"
