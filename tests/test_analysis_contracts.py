from __future__ import annotations

from ariadne_py.analysis.context_pack import (
    ContextItem,
    _compute_diversity_penalty,
    _compute_relevance,
    _count_tokens,
    _get_node_content,
    _select_diverse,
)
from ariadne_py.analysis.counterfactual import run_without_edges
from ariadne_py.analysis.dedup import deduplicate_nodes
from ariadne_py.analysis.dedup.lsh import LshIndex, lsh_candidate_pairs
from ariadne_py.analysis.dedup.minhash import MinHash, shingle
from ariadne_py.analysis.dedup.normalize import normalize_label, passes_entropy_gate, shannon_entropy
from ariadne_py.analysis.dedup.similarity import jaro_winkler
from ariadne_py.analysis.dedup.types import DedupOptions
from ariadne_py.analysis.dedup.union_find import UnionFind
from ariadne_py.analysis.quality import community_cohesion, community_quality
from ariadne_py.analysis.semsearch import EmbeddingIndex, _cosine_similarity, semantic_search
from ariadne_py.core import Edge, EdgeKind, Graph, Node, NodeKind


def _graph() -> tuple[Graph, list]:
    graph = Graph()
    nodes = [
        graph.add_node(Node.new(NodeKind.FUNCTION, name).with_source_text(text))
        for name, text in (
            ("app::handle_request", "def handle_request(): return validate()"),
            ("app::validate", "def validate(): return persist()"),
            ("app::persist", "def persist(): return True"),
            ("test::test_handle", "def test_handle(): handle_request()"),
        )
    ]
    graph.add_edge(nodes[0], nodes[1], Edge.extracted(EdgeKind.CALLS))
    graph.add_edge(nodes[1], nodes[2], Edge.extracted(EdgeKind.CALLS))
    graph.add_edge(nodes[0], nodes[3], Edge.extracted(EdgeKind.TESTED_BY))
    return graph, nodes


def test_community_quality_handles_connected_singleton_and_empty_partitions() -> None:
    graph, ids = _graph()
    communities = {ids[0].value: 1, ids[1].value: 1, ids[2].value: 2}

    assert community_cohesion(graph, communities) == {1: 1.0, 2: 1.0}
    quality = community_quality(graph, communities, resolution=2.0)
    assert quality.community_count == 2
    assert quality.singleton_count == 1
    assert quality.min_size == 1
    assert quality.max_size == 2
    assert quality.mean_conductance > 0
    assert community_quality(graph, {}).community_count == 0


def test_counterfactual_clones_graph_and_drops_only_requested_edges() -> None:
    graph, _ = _graph()
    edge_ids = [eid for eid, *_ in graph.edges()]

    result = run_without_edges(graph, [edge_ids[0]])

    assert result is not graph
    assert result.node_count() == graph.node_count()
    assert result.edge_count() == graph.edge_count() - 1
    assert graph.edge_count() == 3


def test_dedup_primitives_and_pipeline_group_near_duplicate_labels() -> None:
    assert normalize_label("HTTP_Request-Handler") == "http_request_handler"
    assert shannon_entropy("aaaa") == 0.0
    assert passes_entropy_gate("request handler", 0.5)
    assert jaro_winkler("request handler", "request handlers") > 0.95
    assert shingle("abcd", 3) == ["abc", "bcd"]
    assert shingle("ab", 3) == ["ab"]

    signature = MinHash.from_iter(["abc", "bcd"], 8)
    assert signature.jaccard(signature) == 1.0
    index = LshIndex(2, 4)
    graph, ids = _graph()
    index.add(signature, ids[0])
    assert ids[0] in index.get_candidates(signature)

    options = DedupOptions(
        entropy_gate=0,
        shingle_size=2,
        num_permutations=16,
        num_bands=8,
        row_length=2,
        jaccard_threshold=0,
        jw_threshold=0.8,
        eligible_kinds=frozenset({NodeKind.CONCEPT}),
    )
    dedup_graph = Graph()
    node_ids = [
        dedup_graph.add_node(Node.new(NodeKind.CONCEPT, qn))
        for qn in ("docs::request_handler", "docs::request_handlers")
    ]
    nodes = [dedup_graph.node(nid) for nid in node_ids]
    assert lsh_candidate_pairs(nodes, node_ids, options)
    result = deduplicate_nodes(dedup_graph, options)
    assert result["merges"] == 1
    assert result["nodes_removed"] == 1

    uf = UnionFind()
    uf.union(node_ids[0], node_ids[1])
    assert uf.find(node_ids[0]) == uf.find(node_ids[1])


def test_context_pack_helpers_rank_diversity_and_respect_budget() -> None:
    graph, ids = _graph()
    content = _get_node_content(graph, ids[0])
    assert "handle_request" in content
    assert _get_node_content(graph, type(ids[0])(999)) == ""

    relevant = ContextItem(
        "app::handle_request", "function", "handle_request", "app.py", content,
        token_count=_count_tokens(content),
    )
    other = ContextItem(
        "db::persist", "function", "persist", "db.py", "write record", token_count=3,
    )
    assert _compute_relevance(relevant, "handle") > _compute_relevance(other, "handle")
    assert _compute_diversity_penalty(relevant, {"app.py"}, {"function"}) == 0.4
    selected = _select_diverse(
        {relevant.qualified_name: relevant, other.qualified_name: other},
        "handle",
        relevant.token_count,
        5,
    )
    assert selected == [relevant]


class _Vector:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _Model:
    def encode(self, value, normalize_embeddings=True):
        if isinstance(value, list):
            return [_Vector([1.0, float(i)]) for i, _ in enumerate(value)]
        return _Vector([1.0, 0.0] if "request" in value else [0.0, 1.0])


def test_semantic_index_covers_uninitialized_index_batch_search_and_clear() -> None:
    graph, ids = _graph()
    index = EmbeddingIndex()
    index.index_node(ids[0], "ignored")
    assert index.search("request") == []

    index._initialized = True
    index._model = _Model()
    index.index_graph(graph)
    assert index.search("request", limit=1)[0][0] == ids[0].value
    result = semantic_search(graph, "request", index, limit=2)
    assert result["total"] == 2
    assert result["results"][0]["qualified_name"] == "app::handle_request"
    assert _cosine_similarity([1, 0], [1, 0]) == 1.0
    assert _cosine_similarity([1], [1, 0]) == 0.0
    assert _cosine_similarity([0, 0], [1, 0]) == 0.0
    index.clear()
    assert index.search("request") == []
