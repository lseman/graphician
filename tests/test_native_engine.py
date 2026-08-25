from __future__ import annotations

import pytest

from graphician._extract import (
    HAS_RUST,
    NativeGraph,
    dedup_candidate_pairs,
    fuzzy_score_matrix,
)
from graphician._extract.python import extract_data_flow as native_extract_data_flow
from graphician.analysis.communities import quality as community_quality_module
from graphician.analysis.dedup.minhash import MinHash, shingle
from graphician.analysis.motifs import engine as motif_engine
from graphician.analysis.motifs.dsl import Motif
from graphician.analysis.native import native_graph
from graphician.analysis.search import search as search_module
from graphician.analysis.search.fuzzy import _fuzzy_score
from graphician.core.edge import Edge, EdgeKind
from graphician.core.graph import Graph
from graphician.core.node import Node, NodeKind
from graphician.extraction.data_flow import extract_data_flow as python_extract_data_flow
from graphician.extraction.flows import compute_flows
from graphician.extraction.flows.trace import trace_flow
from graphician.extraction.flows.types import FlowOptions
from graphician.extraction.library_stubs import resolve_library_stubs

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust extension is not built")


def test_native_graph_validates_ids_and_preserves_ambiguous_edges() -> None:
    graph = NativeGraph(
        [10, 20, 30],
        [
            (10, 20, "calls", "extracted"),
            (10, 30, "calls", "ambiguous"),
        ],
    )

    assert graph.node_count == 3
    assert graph.edge_count == 2
    assert graph.traverse(10, max_hops=1) == [20, 30]
    with pytest.raises(ValueError, match="unique"):
        NativeGraph([10, 10], [])
    with pytest.raises(ValueError, match="not present"):
        NativeGraph([10], [(10, 20, "calls", "extracted")])


def test_native_graph_personalized_pagerank_biases_seed() -> None:
    graph = NativeGraph(
        [0, 1, 2],
        [
            (0, 1, "calls", "extracted"),
            (2, 1, "calls", "extracted"),
        ],
    )

    ranks = graph.pagerank(seeds=[(0, 1.0)])

    assert ranks[0] > ranks[2]
    assert sum(ranks.values()) == pytest.approx(1.0)


def test_native_graph_traversal_and_structure_share_snapshot() -> None:
    graph = NativeGraph(
        [0, 1, 2, 3],
        [
            (0, 1, "calls", "extracted"),
            (1, 2, "calls", "extracted"),
            (2, 0, "calls", "extracted"),
            (2, 3, "calls", "extracted"),
        ],
    )

    assert graph.traverse(0, "calls", max_hops=1) == [1]
    assert graph.traverse(2, "calls", reverse=True, max_hops=2) == [1, 0]
    assert graph.paths(0, target=3, max_hops=3) == [[0, 1, 2, 3]]
    assert graph.max_depth(0) == 3
    assert graph.cyclic_components() == [[0, 1, 2]]
    assert graph.core_numbers() == {0: 2, 1: 2, 2: 2, 3: 1}
    assert graph.articulation_points() == [2]


def test_python_graph_reuses_native_snapshot_until_structural_mutation() -> None:
    graph = Graph()
    source = graph.add_node(Node.new(NodeKind.FUNCTION, "app::source"))
    target = graph.add_node(Node.new(NodeKind.FUNCTION, "app::target"))
    graph.add_edge(source, target, Edge.extracted(EdgeKind.CALLS))

    first = native_graph(graph)
    assert first is not None
    assert native_graph(graph) is first

    added = graph.add_node(Node.new(NodeKind.FUNCTION, "app::added"))
    graph.add_edge(target, added, Edge.extracted(EdgeKind.CALLS))
    second = native_graph(graph)

    assert second is not first
    assert second.traverse(source.value, "calls", max_hops=2) == [target.value, added.value]
    assert native_graph(graph) is second


def test_native_snapshot_detects_direct_edge_state_mutation() -> None:
    graph = Graph()
    source = graph.add_node(Node.new(NodeKind.FUNCTION, "app::source"))
    target = graph.add_node(Node.new(NodeKind.FUNCTION, "app::target"))
    edge_id = graph.add_edge(source, target, Edge.extracted(EdgeKind.CALLS))
    first = native_graph(graph)
    assert first is not None

    edge = graph.edge(edge_id)
    assert edge is not None
    edge.confidence = "ambiguous"
    second = native_graph(graph)

    assert second is not first
    assert second.paths(source.value, target.value, min_confidence=0.5) == []


def test_native_graph_impact_prioritizes_reverse_dependants() -> None:
    graph = NativeGraph(
        [0, 1, 2],
        [
            (0, 1, "calls", "extracted"),
            (1, 2, "calls", "extracted"),
        ],
    )

    hits = graph.impact(1, max_hops=1)

    assert [hit[0] for hit in hits] == [0, 2]
    assert hits[0][1] < hits[1][1]


def test_native_batch_flow_trace_matches_python_with_cap_and_placeholders() -> None:
    graph = Graph()
    node_ids = [
        graph.add_node(Node.new(NodeKind.FUNCTION, f"pkg::fn_{index}"))
        for index in range(12)
    ]
    placeholder = graph.add_node(Node.new(NodeKind.FUNCTION, "call::external"))
    for index in range(11):
        graph.add_edge(node_ids[index], node_ids[index + 1], Edge.extracted(EdgeKind.CALLS))
    graph.add_edge(node_ids[0], node_ids[5], Edge.extracted(EdgeKind.CALLS))
    graph.add_edge(node_ids[0], placeholder, Edge.ambiguous(EdgeKind.CALLS))
    options = FlowOptions(max_depth=10, max_nodes_per_flow=6)

    expected = trace_flow(graph, node_ids[0], options)
    snapshot = native_graph(graph)
    assert snapshot is not None
    actual = snapshot.trace_flows(
        [node_ids[0].value],
        [placeholder.value],
        options.max_depth,
        options.max_nodes_per_flow,
    )[0]

    assert actual == [(node_id.value, depth) for node_id, depth in expected]


def test_large_flow_materialization_dispatches_native_with_python_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = Graph()
    for chain in range(32):
        chain_ids = [
            graph.add_node(Node.new(NodeKind.FUNCTION, f"chain_{chain}::fn_{index}"))
            for index in range(32)
        ]
        for source, target in zip(chain_ids, chain_ids[1:], strict=False):
            graph.add_edge(source, target, Edge.extracted(EdgeKind.CALLS))

    native_value = graph.clone()
    python_value = graph.clone()
    original_native_graph = native_graph
    dispatches = 0

    class TrackedSnapshot:
        def __init__(self, snapshot):
            self.snapshot = snapshot

        def trace_flows(self, *args, **kwargs):
            nonlocal dispatches
            dispatches += 1
            return self.snapshot.trace_flows(*args, **kwargs)

    monkeypatch.setattr(
        "graphician.analysis.native.native_graph",
        lambda value: TrackedSnapshot(original_native_graph(value)),
    )
    assert compute_flows(native_value) == 32
    assert dispatches == 1

    monkeypatch.setattr("graphician.analysis.native.native_graph", lambda _value: None)
    assert compute_flows(python_value) == 32

    def flow_topology(value: Graph):
        return sorted(
            (
                node.qualified_name,
                node.properties,
                sorted(
                    (value.node(source).qualified_name, edge.kind.value)
                    for source, edge in value.in_neighbors(node_id)
                ),
            )
            for node_id, node in value.nodes()
            if node.kind == NodeKind.FLOW
        )

    assert flow_topology(native_value) == flow_topology(python_value)


def test_native_dedup_candidates_match_python_minhash() -> None:
    labels = [
        (10, "request handler"),
        (20, "request handlers"),
        (30, "database adapter"),
        (40, "request handler"),
    ]
    parameters = (3, 64, 12, 5, 0.0)

    native = dedup_candidate_pairs(labels, *parameters)
    signatures = {
        node_id: MinHash.from_iter(shingle(label, parameters[0]), parameters[1])
        for node_id, label in labels
    }
    expected = []
    for index, (left_id, _) in enumerate(labels):
        for right_id, _ in labels[index + 1:]:
            score = signatures[left_id].jaccard(signatures[right_id])
            shares_band = any(
                signatures[left_id].signature[start:start + parameters[3]]
                == signatures[right_id].signature[start:start + parameters[3]]
                for start in range(0, parameters[2] * parameters[3], parameters[3])
            )
            if shares_band and score >= parameters[4]:
                expected.append((left_id, right_id, score))

    assert native == expected


def test_native_dedup_candidates_validate_parameters() -> None:
    with pytest.raises(ValueError, match="shingle_size"):
        dedup_candidate_pairs([(1, "node")], 0, 8, 2, 4, 0.5)
    with pytest.raises(ValueError, match="unique"):
        dedup_candidate_pairs([(1, "a"), (1, "b")], 1, 8, 2, 4, 0.5)


def test_native_motif_matching_matches_python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = Graph()
    source = graph.add_node(Node.new(NodeKind.FUNCTION, "app::source"))
    call_target = graph.add_node(Node.new(NodeKind.FUNCTION, "app::call_target"))
    import_target = graph.add_node(Node.new(NodeKind.MODULE, "app::import_target"))
    graph.add_edge(source, call_target, Edge.extracted(EdgeKind.CALLS))
    graph.add_edge(source, import_target, Edge.extracted(EdgeKind.IMPORTS))
    motif = (
        Motif.builder()
        .add_node(lambda node: node.kind(NodeKind.FUNCTION))
        .add_node(lambda node: node.kind(NodeKind.FUNCTION))
        .add_edge(0, 1, EdgeKind.CALLS)
        .build()
    )

    native = motif_engine.find_motifs(graph, motif)
    monkeypatch.setattr(motif_engine, "native_graph", lambda _graph: None)
    python = motif_engine.find_motifs(graph, motif)

    assert native == python


def test_native_community_quality_matches_python_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = Graph()
    nodes = [
        graph.add_node(Node.new(NodeKind.FUNCTION, f"app::node_{index}"))
        for index in range(5)
    ]
    graph.add_edge(nodes[0], nodes[1], Edge.extracted(EdgeKind.CALLS))
    graph.add_edge(nodes[1], nodes[0], Edge.extracted(EdgeKind.CALLS))
    graph.add_edge(nodes[1], nodes[2], Edge.extracted(EdgeKind.CALLS))
    graph.add_edge(nodes[2], nodes[2], Edge.extracted(EdgeKind.CALLS))
    graph.add_edge(nodes[3], nodes[4], Edge.extracted(EdgeKind.IMPORTS))
    assignments = {
        nodes[0].value: 10,
        nodes[1].value: 10,
        nodes[2].value: 20,
        nodes[3].value: 30,
        nodes[4].value: 30,
    }

    native = community_quality_module.community_quality(graph, assignments, resolution=1.5)
    native_cohesion = community_quality_module.community_cohesion(graph, assignments)
    monkeypatch.setattr(community_quality_module, "native_graph", lambda _graph: None)
    python = community_quality_module.community_quality(graph, assignments, resolution=1.5)
    python_cohesion = community_quality_module.community_cohesion(graph, assignments)

    assert native == python
    assert native_cohesion == python_cohesion


def test_native_fuzzy_scores_match_python_signals() -> None:
    queries = ["request handler", "rh", "database_adapter", "café"]
    targets = ["request handler", "handle request", "database adapter", "café service"]

    native = fuzzy_score_matrix(queries, targets)
    python = [[_fuzzy_score(query, target) for target in targets] for query in queries]

    for native_row, python_row in zip(native, python, strict=True):
        assert native_row == pytest.approx(python_row)


def test_native_ranked_search_matches_python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = Graph()
    for qname in (
        "app::request_handler",
        "app::handle_request",
        "db::database_adapter",
        "docs::request_handling_guide",
    ):
        graph.add_node(Node.new(NodeKind.FUNCTION, qname))

    native = search_module.ranked_search(graph, "request handler", limit=10)
    monkeypatch.setattr(search_module, "fuzzy_score_matrix", None)
    python = search_module.ranked_search(graph, "request handler", limit=10)

    assert [hit.id for hit in native] == [hit.id for hit in python]
    assert [hit.score for hit in native] == pytest.approx([hit.score for hit in python])


def test_native_data_flow_matches_python_fallback() -> None:
    source = "def total(left, right):\n    value = left + right\n    return value\n"
    native_graph_value = Graph().clone()
    native_function = native_graph_value.add_node(Node.new(NodeKind.FUNCTION, "app::total"))
    python_graph_value = Graph()
    python_function = python_graph_value.add_node(Node.new(NodeKind.FUNCTION, "app::total"))

    native_extract_data_flow(native_graph_value, native_function, source)
    python_extract_data_flow(python_graph_value, python_function, source)

    def topology(graph: Graph) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
        nodes = sorted((node.qualified_name, node.kind.value) for _, node in graph.nodes())
        edges = sorted(
            (
                graph.node(source_id).qualified_name,
                edge.kind.value,
                graph.node(target_id).qualified_name,
            )
            for _, source_id, target_id, edge in graph.edges()
        )
        return nodes, edges

    assert topology(native_graph_value) == topology(python_graph_value)


def test_library_stub_resolution_creates_typed_node() -> None:
    graph = Graph()
    caller = graph.add_node(Node.new(NodeKind.FUNCTION, "app::caller"))
    placeholder = graph.add_node(Node.new(NodeKind.FUNCTION, "call::append"))
    graph.add_edge(caller, placeholder, Edge.ambiguous(EdgeKind.CALLS))

    assert resolve_library_stubs(graph) == 1
    stub_id = graph.find_by_qname("stub::list")
    assert stub_id is not None
    stub = graph.node(stub_id)
    assert stub is not None
    assert stub.kind is NodeKind.CLASS
    assert stub.properties["is_stub"] is True
