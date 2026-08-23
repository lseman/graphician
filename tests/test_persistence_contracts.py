from __future__ import annotations

import sqlite3
import gc
import warnings

import pytest

from graphician.core import Confidence, Edge, EdgeKind, Graph, Node, NodeKind
from graphician.persistence import embeddings
from graphician.persistence.embeddings import ExternalEmbeddingConfig
from graphician.persistence.fts import FTSIndex, build_fts5_query
from graphician.persistence.store import (
    GraphStore,
    _graph_from_payload,
    _graph_to_payload,
    edge_identity,
    parse_confidence,
)


def test_graph_store_context_owns_and_closes_connection(tmp_path) -> None:
    store = GraphStore(tmp_path / "owned.db")
    connection = store._conn
    with store:
        connection.execute("SELECT 1").fetchone()

    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")
    store.close()  # Explicit cleanup is idempotent.


def test_graph_store_finalizer_closes_abandoned_connection(tmp_path) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        store = GraphStore(tmp_path / "abandoned.db")
        connection = store._conn
        del store
        gc.collect()

    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")
    assert not [warning for warning in caught if warning.category is ResourceWarning]


def test_fts_index_supports_crud_safe_search_rebuild_and_optimize() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE nodes (node_id INTEGER, kind TEXT, name TEXT, "
        "qualified_name TEXT, source_text TEXT)"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE nodes_fts USING fts5("
        "node_id UNINDEXED, kind, name, qualified_name, source_text)"
    )
    fts = FTSIndex(conn)
    fts.add_node(1, "handle_request", "app::handle_request", "validate user request")
    assert fts.count("request") == 1
    assert fts.search("request")[0].qualified_name == "app::handle_request"
    assert fts.search("   ") == []
    assert fts.search_safe('request OR "broken') == []

    fts.update_node(1, "persist", "app::persist", "write database record")
    assert not fts.search_safe("request")
    assert fts.search_safe("database")[0].node_id == 1
    fts.remove_node(1)
    assert fts.count("database") == 0

    conn.execute(
        "INSERT INTO nodes VALUES (2, 'function', 'render_page', 'web::render_page', 'HTML response')"
    )
    fts.rebuild()
    assert fts.search_safe("render-page")[0].qualified_name == "web::render_page"
    fts.optimize()
    assert build_fts5_query('hello + "world" OR x:y') == 'hello* world* "OR"* x* y*'
    assert build_fts5_query("!!!") == ""
    conn.close()


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (ExternalEmbeddingConfig("openai-embedding", dimension=0), "dimension"),
        (ExternalEmbeddingConfig("openai-embedding"), "api_key"),
        (ExternalEmbeddingConfig("google-embedding"), "api_key"),
        (ExternalEmbeddingConfig("unknown"), "unsupported"),
    ],
)
def test_external_embedding_config_validation(config, message: str) -> None:
    valid, error = embeddings.validate_config(config)
    assert not valid
    assert message in error


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _Client:
    responses = []
    requests = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def post(self, url, **kwargs):
        self.requests.append((url, kwargs))
        return _Response(self.responses.pop(0))


class _Httpx:
    Client = _Client


def test_single_external_embedding_providers_and_empty_input(monkeypatch) -> None:
    monkeypatch.setattr(embeddings, "_require_httpx", lambda: _Httpx)
    assert embeddings.external_embedding_from_config(
        ExternalEmbeddingConfig("ollama-embedding", dimension=3), " "
    ) == [0.0, 0.0, 0.0]

    _Client.responses = [
        {"data": [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}]},
        {"embedding": {"values": [0.25, 0.75]}},
        {"embeddings": [[0.4, 0.6]]},
        {"embeddings": []},
    ]
    assert embeddings.external_embedding_from_config(
        ExternalEmbeddingConfig("openai-embedding", api_key="key"), "hello"
    ) == [1, 0]
    assert embeddings.external_embedding_from_config(
        ExternalEmbeddingConfig("google-embedding", api_key="key"), "hello"
    ) == [0.25, 0.75]
    assert embeddings.external_embedding_from_config(
        ExternalEmbeddingConfig("ollama-embedding"), "hello"
    ) == [0.4, 0.6]
    assert embeddings.external_embedding_from_config(
        ExternalEmbeddingConfig("ollama-embedding", dimension=2), "hello"
    ) == [0.0, 0.0]


def test_batch_external_embeddings_and_provider_errors(monkeypatch) -> None:
    monkeypatch.setattr(embeddings, "_require_httpx", lambda: _Httpx)
    graph = Graph()
    graph.add_node(Node.new(NodeKind.FUNCTION, "app::one").with_source_text("one"))
    graph.add_node(Node.new(NodeKind.FUNCTION, "app::two").with_source_text("two"))
    graph.add_node(Node.new(NodeKind.FUNCTION, "app::empty"))
    assert embeddings.embeddable_texts(graph) == {"app::one": "one", "app::two": "two"}

    _Client.responses = [{"embeddings": [[1, 0], [0, 1]]}]
    result = embeddings.build_external_embeddings(
        graph, provider="ollama", model="demo", batch_size=2
    )
    assert result == {"app::one": [1.0, 0.0], "app::two": [0.0, 1.0]}
    with pytest.raises(ValueError, match="batch_size"):
        embeddings.build_external_embeddings(graph, provider="ollama", model="demo", batch_size=0)
    with pytest.raises(ValueError, match="unsupported"):
        embeddings._embed_batch(
            _Client(), provider="unknown", model="x", texts=["x"], api_key=None, base_url=None
        )


def test_store_helpers_and_payload_round_trip_preserve_graph_contract() -> None:
    graph = Graph()
    source = graph.add_node(
        Node.new(NodeKind.FUNCTION, "app::source")
        .with_source("app.py", 1, 3)
        .with_source_text("def source(): pass")
        .with_property("async", True)
    )
    target = graph.add_node(Node.new(NodeKind.FUNCTION, "app::target"))
    edge = Edge.ambiguous(EdgeKind.CALLS)
    edge.properties["receiver"] = "service"
    graph.add_edge(source, target, edge)

    restored = _graph_from_payload(_graph_to_payload(graph))
    assert restored.node_count() == 2
    assert restored.edge_count() == 1
    restored_source = restored.node(restored.find_by_qname("app::source"))
    assert restored_source.source_text == "def source(): pass"
    assert next(restored.edges())[3].confidence is Confidence.AMBIGUOUS
    assert edge_identity("a", "b", EdgeKind.CALLS).to_string() == "a\x1fb\x1fcalls"
    assert parse_confidence("extracted", 1) is Confidence.EXTRACTED
    assert parse_confidence("ambiguous", 0.5) is Confidence.AMBIGUOUS
    assert parse_confidence("other", 0.5) is Confidence.INFERRED

    with GraphStore.open_in_memory() as store:
        store.save_graph(restored)
        assert store.status()["node_count"] == 2
        assert store.conn is not None


def test_temporal_archive_source_reads_and_deletion_round_trip() -> None:
    graph = Graph()
    source_node = (
        Node.new(NodeKind.FUNCTION, "app::source")
        .with_source("src/app.py", 1, 3)
        .with_source_text("def source(): pass")
    )
    source = graph.add_node(source_node)
    target_node = Node.new(NodeKind.FUNCTION, "app::target").with_source("src/lib.py", 1, 2)
    target = graph.add_node(target_node)
    edge = Edge.extracted(EdgeKind.CALLS)
    edge.properties["site"] = 2
    graph.add_edge(source, target, edge)

    with GraphStore.open_in_memory() as store:
        store.save_graph(graph, {"src/app.py": "hash-a", "src/lib.py": "hash-b"})
        active_nodes = store.active_nodes_for_sources(["src/app.py", "src/app.py"])
        active_edges = store.active_edges_for_sources(["src/app.py", "src/app.py"])
        assert [node.qualified_name for node in active_nodes] == ["app::source"]
        assert len(active_edges) == 1

        store.archive_nodes([source_node], "2026-01-02T00:00:00Z")
        store.archive_edges(active_edges, "2026-01-02T00:00:00Z")
        assert any(node.qualified_name == "app::source" for node in store.temporal_nodes())
        assert any(item[0] == "app::source" for item in store.temporal_edges())
        temporal = store.load_temporal()
        assert temporal.find_by_qname("app::source") is not None
        assert temporal.edge_count() == 1

        store.delete_sources(["src/app.py"])
        assert store.load_graph().find_by_qname("app::source") is None
        assert store.get_file_hashes() == {"src/lib.py": "hash-b"}
        # Archived history remains available after deleting active source rows.
        assert any(node.qualified_name == "app::source" for node in store.temporal_nodes())


def test_temporal_helpers_are_noops_for_empty_inputs() -> None:
    with GraphStore.open_in_memory() as store:
        store.archive_nodes([], "now")
        store.archive_edges([], "now")
        store.delete_sources([])
        assert store.active_nodes_for_sources([]) == []
        assert store.active_edges_for_sources([]) == []
