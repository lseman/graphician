from __future__ import annotations

import pytest

from graphician.analysis.structure import call_resolution_stats
from graphician.core.edge import Edge, EdgeKind
from graphician.core.graph import Graph
from graphician.core.node import Node, NodeKind
from graphician.extraction.library_stubs import (
    resolve_library_stubs,
    resolve_library_stubs_batch,
)


@pytest.mark.parametrize(
    ("source_uri", "expected_stub"),
    [
        ("src/app.py", "stub::list"),
        ("src/lib.rs", "stub::Vec"),
        ("src/app.ts", "stub::URLSearchParams"),
        ("src/app.cpp", "stub::string"),
    ],
)
def test_stub_resolution_respects_caller_language(
    source_uri: str, expected_stub: str
) -> None:
    graph = Graph()
    caller = graph.add_node(
        Node.new(NodeKind.FUNCTION, f"{source_uri}::run").with_source(source_uri, 1, 2)
    )
    placeholder = graph.add_node(Node.new(NodeKind.FUNCTION, "call::append"))
    graph.add_edge(caller, placeholder, Edge.ambiguous(EdgeKind.CALLS))

    assert resolve_library_stubs(graph) == 1
    stub = graph.find_by_qname(expected_stub)
    assert stub is not None
    assert graph.find_by_qname("call::append") is None
    assert [(target, edge.kind) for target, edge in graph.out_neighbors(caller)] == [
        (stub, EdgeKind.CALLS)
    ]
    assert call_resolution_stats(graph)["rate"] == 1.0
    assert resolve_library_stubs(graph) == 0


def test_stub_batch_statistics_count_edges_and_actual_rewrites() -> None:
    graph = Graph()
    placeholder = graph.add_node(Node.new(NodeKind.FUNCTION, "call::append"))
    for index in range(2):
        caller = graph.add_node(
            Node.new(NodeKind.FUNCTION, f"pkg::caller_{index}").with_source(
                f"pkg/file_{index}.py", 1, 2
            )
        )
        graph.add_edge(caller, placeholder, Edge.ambiguous(EdgeKind.CALLS))

    assert resolve_library_stubs_batch(graph) == {
        "operation": "library_stubs",
        "total_unresolved": 2,
        "resolved": 2,
        "unresolved_remaining": 0,
        "resolution_rate": 1.0,
    }
