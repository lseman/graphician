from __future__ import annotations

import io
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from ariadne_py.core.edge import Confidence, Edge, EdgeKind
from ariadne_py.core.graph import Graph
from ariadne_py.core.node import Node, NodeKind
from ariadne_py.extraction.compiler import (
    CompilerEdgeEvidence,
    CompilerEvidenceFile,
    apply_compiler_evidence,
    load_compiler_evidence,
)
from ariadne_py.extraction.languages import LanguageRegistry
from ariadne_py.extraction.pipeline import ExtractionPipeline
from ariadne_py.extraction.rust_analyzer import _file_uri_to_path, _read_message
from ariadne_py.interfaces.cli.git import collect_file_hashes, graph_freshness
from ariadne_py.interfaces.cli.response import tool_response
from ariadne_py.persistence.store import GraphStore, IncompatibleDatabaseError
from ariadne_py.analysis.diff import graph_diff
from ariadne_py.interfaces.transport.mcp import AriadneMCP


def test_tool_status_reports_call_resolution(tmp_path: Path) -> None:
    graph = Graph()
    caller = graph.add_node(Node.new(NodeKind.FUNCTION, "pkg::caller"))
    callee = graph.add_node(Node.new(NodeKind.FUNCTION, "pkg::callee"))
    graph.add_edge(caller, callee, Edge.extracted(EdgeKind.CALLS))
    db_path = tmp_path / "graph.db"
    with GraphStore(db_path) as store:
        store.save_graph(graph)

    response = tool_response(str(db_path), "status", {})

    assert response["call_resolution"] == {
        "resolved": 1,
        "unresolved": 0,
        "rate": 1.0,
    }


def test_tool_response_accepts_operations_without_hints(tmp_path: Path) -> None:
    graph = Graph()
    first = Node.new(NodeKind.FUNCTION, "pkg::first").with_source("pkg/a.py", 1, 2)
    second = Node.new(NodeKind.FUNCTION, "pkg::second").with_source("pkg/b.py", 1, 2)
    first_id = graph.add_node(first)
    second_id = graph.add_node(second)
    graph.add_edge(first_id, second_id, Edge.extracted(EdgeKind.CALLS))
    db_path = tmp_path / "graph.db"
    with GraphStore(db_path) as store:
        store.save_graph(graph)

    response = tool_response(str(db_path), "architecture", {})

    assert response["operation"] == "architecture_overview"
    assert response["community_count"] == 2
    assert "_hints" not in response


def test_compiler_evidence_adds_and_upgrades_edges_with_provenance() -> None:
    graph = Graph()
    caller = graph.add_node(Node.new(NodeKind.FUNCTION, "pkg::caller"))
    callee = graph.add_node(Node.new(NodeKind.FUNCTION, "pkg::callee"))
    graph.add_edge(caller, callee, Edge.inferred(EdgeKind.CALLS, 0.5))
    graph.add_node(Node.new(NodeKind.FUNCTION, "pkg::other"))
    evidence = CompilerEvidenceFile(
        1,
        "rust-analyzer",
        (
            CompilerEdgeEvidence(
                "pkg::caller", "pkg::callee", EdgeKind.CALLS, "definition response"
            ),
            CompilerEdgeEvidence("pkg::caller", "pkg::other", EdgeKind.CALLS),
            CompilerEdgeEvidence("pkg::missing", "pkg::other", EdgeKind.CALLS),
        ),
    )

    report = apply_compiler_evidence(graph, evidence)

    assert (report.added, report.upgraded, report.unresolved) == (1, 1, 1)
    edges = list(graph.edges())
    upgraded = edges[0][3]
    assert upgraded.confidence is Confidence.EXTRACTED
    assert upgraded.properties["provenance"] == "compiler"
    assert upgraded.properties["provider"] == "rust-analyzer"
    assert upgraded.properties["evidence_detail"] == "definition response"


def test_load_compiler_evidence_validates_version_and_provider(tmp_path: Path) -> None:
    path = tmp_path / "compiler-evidence.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "provider": "pyright",
                "edges": [
                    {"source": "a", "target": "b", "kind": "calls", "detail": "resolved"}
                ],
            }
        )
    )
    loaded = load_compiler_evidence(path)
    assert loaded.provider == "pyright"
    assert loaded.edges[0].kind is EdgeKind.CALLS

    path.write_text('{"version":2,"provider":"pyright"}')
    with pytest.raises(ValueError, match="unsupported compiler evidence version"):
        load_compiler_evidence(path)


def test_pipeline_records_relative_file_hashes(tmp_path: Path) -> None:
    source = tmp_path / "example.py"
    source.write_text("def hello():\n    return 1\n")
    pipeline = ExtractionPipeline(LanguageRegistry())
    pipeline.build(tmp_path)
    assert list(pipeline._file_hashes) == ["example.py"]
    assert pipeline._file_hashes == collect_file_hashes(tmp_path)


def test_graph_freshness_detects_added_modified_and_deleted_sources(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    first = root / "first.py"
    first.write_text("value = 1\n")
    db_path = tmp_path / "graph.db"
    with GraphStore(db_path) as store:
        store.save_graph(Graph(), collect_file_hashes(root))
        store.set_metadata("repository_root", str(root))
        assert graph_freshness(store)["state"] == "fresh"

        first.write_text("value = 2\n")
        second = root / "second.py"
        second.write_text("value = 3\n")
        dirty = graph_freshness(store)
        assert dirty["state"] == "dirty"
        assert dirty["changes"]["added"] == 1
        assert dirty["changes"]["modified"] == 1

        first.unlink()
        dirty = graph_freshness(store)
        assert dirty["changes"]["deleted"] == 1


def test_graph_freshness_is_unknown_for_legacy_database(tmp_path: Path) -> None:
    with GraphStore(tmp_path / "legacy.db") as store:
        assert graph_freshness(store)["state"] == "unknown"


def test_store_uses_canonical_schema_marker(tmp_path: Path) -> None:
    with GraphStore(tmp_path / "graph.db") as store:
        assert store.get_metadata("schema_version") == "2"
        assert store.get_metadata("implementation") is None


def test_store_rejects_rust_database_layout_without_mutating_it(tmp_path: Path) -> None:
    db_path = tmp_path / "rust.db"
    with closing(sqlite3.connect(db_path)) as connection:
        with connection:
            connection.execute("CREATE TABLE nodes (id INTEGER PRIMARY KEY, name TEXT)")

    with pytest.raises(IncompatibleDatabaseError, match="canonical Rust Ariadne schema"):
        GraphStore(db_path)

    with closing(sqlite3.connect(db_path)) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert tables == {"nodes"}


def test_lsp_framing_and_file_uri_round_trip(tmp_path: Path) -> None:
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "result": []}).encode()
    message = io.BytesIO(f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload)
    assert _read_message(message)["result"] == []

    path = tmp_path / "file with spaces.rs"
    assert _file_uri_to_path(path.as_uri()) == path


def test_pipeline_integrates_ignores_documents_manifests_and_data_flow(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("ignored.py\n")
    (tmp_path / "ignored.py").write_text("def ignored():\n    pass\n")
    (tmp_path / "app.py").write_text("def run(value):\n    result = value\n    return result\n")
    (tmp_path / "test_app.py").write_text("def test_run():\n    run(1)\n")
    (tmp_path / "README.md").write_text("# API\nUse `run` to start.\n")
    (tmp_path / "diagram.svg").write_text("<svg><text>run</text></svg>")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "demo"\ndependencies = ["httpx>=0.27"]\n'
    )

    graph = ExtractionPipeline(LanguageRegistry()).build(tmp_path)

    assert not any(node.name == "ignored" for _, node in graph.nodes())
    assert any(node.kind is NodeKind.DOCUMENT for _, node in graph.nodes())
    assert any(node.kind is NodeKind.DIAGRAM for _, node in graph.nodes())
    assert graph.find_by_qname("package::demo") is not None
    assert any(edge.kind is EdgeKind.DATA_FLOW for _, _, _, edge in graph.edges())
    assert any(edge.kind is EdgeKind.TESTED_BY for _, _, _, edge in graph.edges())


def test_parallel_pipeline_matches_sequential_pipeline(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def first():\n    second()\n")
    (tmp_path / "b.py").write_text("def second():\n    return 2\n")
    (tmp_path / "README.md").write_text("# API\nUse `first` and `second`.\n")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "parallel-demo"\ndependencies = ["httpx>=0.27"]\n'
    )

    sequential_pipeline = ExtractionPipeline(LanguageRegistry(), workers=1)
    parallel_pipeline = ExtractionPipeline(LanguageRegistry(), workers=4)
    sequential = sequential_pipeline.build(tmp_path)
    parallel = parallel_pipeline.build(tmp_path)

    def signature(graph: Graph) -> tuple[set[tuple], set[tuple]]:
        nodes = {
            (node.qualified_name, node.kind.value, node.source_uri)
            for _, node in graph.nodes()
        }
        edges = {
            (
                graph.node(source).qualified_name,
                graph.node(target).qualified_name,
                edge.kind.value,
                edge.confidence.value,
            )
            for _, source, target, edge in graph.edges()
        }
        return nodes, edges

    assert signature(parallel) == signature(sequential)
    assert parallel_pipeline._file_hashes == sequential_pipeline._file_hashes


def test_incremental_update_replaces_changed_file_and_preserves_other_file(tmp_path: Path) -> None:
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("def old_name():\n    return 1\n")
    second.write_text("def stable():\n    return 2\n")
    initial = ExtractionPipeline(LanguageRegistry()).build(tmp_path)
    stable_qname = next(node.qualified_name for _, node in initial.nodes() if node.name == "stable")

    first.write_text("def new_name():\n    return 3\n")
    pipeline = ExtractionPipeline(LanguageRegistry())
    updated = pipeline.update(tmp_path, initial, ["first.py"], [])

    assert not any(node.name == "old_name" for _, node in updated.nodes())
    assert any(node.name == "new_name" for _, node in updated.nodes())
    assert updated.find_by_qname(stable_qname) is not None


def test_snapshots_embeddings_and_temporal_diff_round_trip(tmp_path: Path) -> None:
    graph = Graph()
    graph.add_node(Node.new(NodeKind.FUNCTION, "pkg::old"))
    with GraphStore(tmp_path / "graph.db") as store:
        store.save_graph(graph)
        store.create_snapshot("commit-old")
        graph.add_node(Node.new(NodeKind.FUNCTION, "pkg::new"))
        store.save_graph(graph)
        store.set_metadata("indexed_commit", "commit-new")
        store.save_embeddings("test-model", {"pkg::new": [0.1, 0.2]})

        old = store.load_graph_at("commit-old")
        current = store.load_graph_at("commit-new")
        diff = graph_diff(old, current)
        assert [node["qualified_name"] for node in diff["added_nodes"]] == ["pkg::new"]
        assert store.load_embeddings("test-model") == {"pkg::new": [0.1, 0.2]}
        assert store.rebuild_fts() == 2


def test_incremental_save_preserves_stable_rows_and_updates_indexes(tmp_path: Path) -> None:
    initial = Graph()
    node_a = Node.new(NodeKind.FUNCTION, "pkg::a").with_source_text("def a(): pass")
    a = initial.add_node(node_a)
    b = initial.add_node(Node.new(NodeKind.FUNCTION, "pkg::b"))
    initial.add_edge(a, b, Edge.extracted(EdgeKind.CALLS))

    with GraphStore(tmp_path / "graph.db") as store:
        store.save_graph(initial)
        store.rebuild_embeddings()
        before = store._conn.execute(
            """SELECT n.id AS node_db_id, e.id AS edge_db_id, v.vector
               FROM nodes n
               JOIN edges e ON e.src_id=n.id
               JOIN embeddings v ON v.node_id=n.id
               WHERE n.qualified_name='pkg::a'"""
        ).fetchone()
        assert before is not None

        store.save_graph_incremental(initial)
        unchanged = store._conn.execute(
            """SELECT n.id AS node_db_id, e.id AS edge_db_id, v.vector
               FROM nodes n
               JOIN edges e ON e.src_id=n.id
               JOIN embeddings v ON v.node_id=n.id
               WHERE n.qualified_name='pkg::a'"""
        ).fetchone()
        assert tuple(unchanged) == tuple(before)

        updated = Graph()
        changed_a = Node.new(NodeKind.FUNCTION, "pkg::a").with_source_text("def renamed(): pass")
        changed_a.name = "renamed"
        a2 = updated.add_node(changed_a)
        c = updated.add_node(Node.new(NodeKind.CLASS, "pkg::c"))
        updated.add_edge(a2, c, Edge.extracted(EdgeKind.DEFINES))
        store.save_graph_incremental(updated, {"pkg.py": "new-hash"})

        loaded = store.load_graph()
        loaded_a_id = loaded.find_by_qname("pkg::a")
        assert loaded_a_id is not None
        assert loaded.node(loaded_a_id).name == "renamed"
        assert loaded.find_by_qname("pkg::b") is None
        assert loaded.find_by_qname("pkg::c") is not None
        persisted_a = store._conn.execute(
            "SELECT id FROM nodes WHERE qualified_name='pkg::a'"
        ).fetchone()
        assert persisted_a["id"] == before["node_db_id"]
        assert store.get_file_hashes() == {"pkg.py": "new-hash"}
        assert any(hit[0] == "pkg::a" for hit in store.fts_search("renamed", 10))
        assert store.get_embedding_stats()[0] == 2


def test_extended_tool_operations_share_mcp_dispatch() -> None:
    graph = Graph()
    graph.add_node(Node.new(NodeKind.FUNCTION, "pkg::entry"))
    server = AriadneMCP()
    server.graph = graph
    server._initialized = True

    traversal = server._execute_operation(
        "traverse", {"target": "pkg::entry", "token_budget": 20}
    )
    assert traversal["items"][0]["qualified_name"] == "pkg::entry"
    assert "orphan_nodes" in server._execute_operation("diagnostics", {})
    assert "patterns" in server._execute_operation("patterns", {})


def test_legacy_mcp_search_matches_current_hybrid_search_signature() -> None:
    graph = Graph()
    graph.add_node(Node.new(NodeKind.FUNCTION, "pkg::entry"))
    server = AriadneMCP()
    server.graph = graph
    server._initialized = True

    result = server._execute_operation("search", {"query": "entry"})

    assert result["results"][0]["qualified_name"] == "pkg::entry"


def test_strict_pipeline_reports_invalid_source(tmp_path: Path) -> None:
    source = tmp_path / "invalid.rs"
    source.write_bytes(b"\xff\xfe")
    pipeline = ExtractionPipeline(LanguageRegistry(), strict=True)

    with pytest.raises(UnicodeError):
        pipeline.build(tmp_path)
