from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from graphician.core import Edge, EdgeKind, Graph, Node, NodeKind
from graphician.evaluation import BENCHMARKS, load_config
from graphician.extraction.spring_di import resolve_spring_injections
from graphician.interfaces.cli import _normalize_grouped_argv
from graphician.persistence.store import GraphStore


def _create_rust_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    with connection:
        connection.executescript("""
            CREATE TABLE nodes (
                id INTEGER PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL,
                qualified_name TEXT NOT NULL UNIQUE, source_uri TEXT,
                line_start INTEGER, line_end INTEGER,
                properties TEXT NOT NULL DEFAULT '{}', valid_from TEXT,
                valid_to TEXT, source_text TEXT
            );
            CREATE TABLE edges (
                id INTEGER PRIMARY KEY, src_id INTEGER NOT NULL,
                dst_id INTEGER NOT NULL, kind TEXT NOT NULL,
                confidence REAL NOT NULL, conf_class TEXT NOT NULL,
                properties TEXT NOT NULL DEFAULT '{}', valid_from TEXT, valid_to TEXT
            );
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO nodes(id, kind, name, qualified_name, properties)
                VALUES (1, 'function', 'caller', 'demo::caller', '{}'),
                       (2, 'function', 'callee', 'demo::callee', '{}');
            INSERT INTO edges(id, src_id, dst_id, kind, confidence, conf_class, properties)
                VALUES (1, 1, 2, 'calls', 1.0, 'extracted', '{}');
        """)
    connection.close()


def test_python_reads_and_updates_rust_database(tmp_path: Path) -> None:
    database = tmp_path / "shared.db"
    _create_rust_database(database)

    with GraphStore(database) as store:
        graph = store.load_graph()
        assert graph.node_count() == 2
        assert graph.edge_count() == 1
        graph.add_node(Node.new(NodeKind.FUNCTION, "demo::added_by_python"))
        store.save_graph_incremental(graph)

    connection = sqlite3.connect(database)
    try:
        assert connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 3
        assert connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 1
    finally:
        connection.close()


def test_python_database_exposes_rust_identifiers(tmp_path: Path) -> None:
    database = tmp_path / "shared.db"
    graph = Graph()
    source = graph.add_node(Node.new(NodeKind.FUNCTION, "demo::source"))
    target = graph.add_node(Node.new(NodeKind.FUNCTION, "demo::target"))
    graph.add_edge(source, target, Edge.extracted(EdgeKind.CALLS))

    with GraphStore(database) as store:
        store.save_graph(graph)

    connection = sqlite3.connect(database)
    try:
        node_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(nodes)")
        }
        edge_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(edges)")
        }
        assert "id" in node_columns and "node_id" not in node_columns
        assert {"id", "src_id", "dst_id"} <= edge_columns
        assert not {"edge_id", "source_id", "target_id"} & edge_columns
        assert connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 1
        metadata = connection.execute(
            "SELECT value FROM meta WHERE key='node_count'"
        ).fetchone()
        assert metadata[0] == "2"
    finally:
        connection.close()


def test_legacy_python_database_is_destructively_reset(tmp_path: Path) -> None:
    database = tmp_path / "legacy.db"
    connection = sqlite3.connect(database)
    with connection:
        connection.executescript("""
            CREATE TABLE nodes (
                node_id INTEGER PRIMARY KEY, kind TEXT, name TEXT,
                qualified_name TEXT, properties TEXT
            );
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);
            INSERT INTO nodes VALUES (1, 'function', 'old', 'demo::old', '{}');
            INSERT INTO metadata VALUES ('old_marker', 'present');
        """)
    connection.close()

    with GraphStore(database) as store:
        assert store.status()["node_count"] == 0
        assert store.get_metadata("requires_rebuild") == "1"
        columns = {row[1] for row in store.conn.execute("PRAGMA table_info(nodes)")}
        assert "id" in columns and "node_id" not in columns
        assert store.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata'"
        ).fetchone() is None


@pytest.mark.parametrize(
    ("rust_args", "flat_command"),
    [
        (["analysis", "paths", "a", "b"], "paths"),
        (["git", "risk"], "risk"),
        (["structure", "diagnostics"], "diagnostics"),
        (["advanced", "patterns"], "patterns"),
        (["agent", "eval"], "eval"),
        (["maintenance", "rebuild-fts"], "rebuild-fts"),
        (["utility", "jedi-enrich"], "jedi-enrich"),
        (["build", "update", "."], "update"),
    ],
)
def test_rust_command_groups_are_accepted(rust_args: list[str], flat_command: str) -> None:
    normalized = _normalize_grouped_argv(rust_args)
    assert normalized[0] == flat_command


def test_evaluation_config_and_registry_match_rust(tmp_path: Path) -> None:
    config = tmp_path / "demo.toml"
    config.write_text(
        'name = "demo"\nurl = "https://example.test/demo"\ncommit = "abc"\n'
        '[[test_commits]]\nsha = "abc"\n',
        encoding="utf-8",
    )
    assert load_config(tmp_path, "demo")["name"] == "demo"
    assert BENCHMARKS == (
        "token_efficiency",
        "flow_completeness",
        "impact_accuracy",
        "impact_accuracy_learned",
        "search_quality",
        "build_performance",
        "multi_hop_retrieval",
        "agent_baseline",
        "test_coverage",
        "graph_coverage",
        "call_coverage",
    )


def test_spring_di_module_is_importable_from_public_extraction_namespace() -> None:
    assert resolve_spring_injections(Graph()) == 0
