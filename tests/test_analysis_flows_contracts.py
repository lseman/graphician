from __future__ import annotations

import pytest

from graphician.analysis.flows.detection import (
    _detect_entry_points,
    affected_flows,
    all_flows,
    compute_flows_with_options,
    flows_through,
)
from graphician.analysis.flows.entry_points import (
    _is_framework_entry,
    _is_generic_event_entry,
    _is_java_framework_entry,
    _is_js_ts_framework_entry,
    _is_python_framework_entry,
)
from graphician.analysis.flows.trace import _compute_criticality, _is_test_node, _trace_flow
from graphician.analysis.flows.types import FlowOptions
from graphician.core import Edge, EdgeKind, Graph, Node, NodeKind


def _node(name: str, **properties) -> Node:
    return Node(
        kind=NodeKind.FUNCTION,
        name=name,
        qualified_name=f"app::{name}",
        properties=properties,
    )


@pytest.mark.parametrize(
    ("node", "detector"),
    [
        (_node("route_users"), _is_python_framework_entry),
        (_node("endpoint", decorators=["@app.route('/users')"]), _is_python_framework_entry),
        (_node("handle_click"), _is_js_ts_framework_entry),
        (_node("render", decorators=["@Component"]), _is_js_ts_framework_entry),
        (_node("execute", annotations=["@Transactional"]), _is_java_framework_entry),
        (_node("on_message"), _is_generic_event_entry),
    ],
)
def test_framework_entry_detectors_cover_language_families(node: Node, detector) -> None:
    assert detector(node)
    assert _is_framework_entry(node)


def test_entry_detectors_reject_unmarked_helpers_and_malformed_metadata() -> None:
    assert not _is_framework_entry(_node("helper"))
    assert not _is_python_framework_entry(_node("helper", decorators="@route"))
    assert not _is_java_framework_entry(_node("helper", annotations="@Controller"))
    assert not _is_generic_event_entry(_node("on_"))


def test_flow_materialization_is_idempotent_ranked_and_queryable() -> None:
    graph = Graph()
    main = graph.add_node(_node("main"))
    handle = graph.add_node(_node("handle_request"))
    persist = graph.add_node(_node("persist"))
    test = graph.add_node(_node("test_main", is_test="yes"))
    graph.add_edge(main, handle, Edge.extracted(EdgeKind.CALLS))
    graph.add_edge(handle, persist, Edge.extracted(EdgeKind.CALLS))
    graph.add_edge(test, handle, Edge.extracted(EdgeKind.CALLS))
    options = FlowOptions(max_depth=5, max_nodes_per_flow=10, min_flow_size=2)

    assert compute_flows_with_options(graph, options) == 3
    first_edge_count = graph.edge_count()
    assert compute_flows_with_options(graph, options) == 3
    assert graph.edge_count() == first_edge_count

    flows = all_flows(graph)
    assert len(flows) == 3
    assert flows_through(graph, persist)
    assert affected_flows(graph, [persist]) == flows_through(graph, persist)
    assert main in _detect_entry_points(graph)
    assert test in _detect_entry_points(graph)


def test_trace_filters_ambiguous_placeholders_and_honors_depth_and_cap() -> None:
    graph = Graph()
    entry = graph.add_node(_node("main"))
    first = graph.add_node(_node("first"))
    second = graph.add_node(_node("second"))
    placeholder = graph.add_node(Node.new(NodeKind.FUNCTION, "call::missing"))
    graph.add_edge(entry, first, Edge.extracted(EdgeKind.CALLS))
    graph.add_edge(first, second, Edge.extracted(EdgeKind.CALLS))
    graph.add_edge(entry, placeholder, Edge.extracted(EdgeKind.CALLS))
    ambiguous = Edge.extracted(EdgeKind.CALLS)
    ambiguous.confidence = "ambiguous"
    graph.add_edge(entry, second, ambiguous)

    shallow = _trace_flow(graph, entry, FlowOptions(max_depth=1, max_nodes_per_flow=10))
    assert shallow == [(entry, 0), (first, 1)]
    capped = _trace_flow(graph, entry, FlowOptions(max_depth=5, max_nodes_per_flow=2))
    assert len(capped) == 2
    assert entry in {nid for nid, _ in capped}


def test_flow_criticality_and_test_detection_boundaries() -> None:
    graph = Graph()
    main = graph.add_node(_node("main"))
    helper = graph.add_node(_node("helper"))
    graph.add_edge(main, helper, Edge.extracted(EdgeKind.CALLS))

    normal = _compute_criticality(graph, [(main, 0), (helper, 1)], "main", False)
    test = _compute_criticality(graph, [(main, 0), (helper, 1)], "main", True)
    assert 0 <= test < normal <= 1
    assert _compute_criticality(graph, [], "empty", False) == 0
    assert _is_test_node(_node("test_it", is_test=True))
    assert not _is_test_node(_node("helper", is_test="no"))
