"""SQLite persistence for the Ariadne graph.

Stores nodes, edges, file hashes, and metadata in a single SQLite database.
Supports incremental updates via file hash comparison.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

from ..core.edge import Edge, EdgeKind, Confidence
from ..core.graph import Graph
from ..core.id import EdgeId, NodeId
from ..core.node import Node, NodeKind

logger = logging.getLogger(__name__)


class GraphStore:
    """SQLite-backed graph persistence.

    Schema:
    - nodes: node_id, kind, name, qualified_name, source_uri, line_start,
      line_end, properties, source_text, valid_from, valid_to
    - edges: edge_id, kind, confidence, properties, valid_from, valid_to,
      source_id, target_id
    - file_hashes: file_path, hash
    - metadata: key, value
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                node_id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                name TEXT NOT NULL,
                qualified_name TEXT UNIQUE NOT NULL,
                source_uri TEXT,
                line_start INTEGER,
                line_end INTEGER,
                properties TEXT DEFAULT '{}',
                source_text TEXT,
                valid_from TEXT,
                valid_to TEXT
            );

            CREATE TABLE IF NOT EXISTS edges (
                edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                confidence TEXT NOT NULL DEFAULT 'extracted',
                properties TEXT DEFAULT '{}',
                valid_from TEXT,
                valid_to TEXT,
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                source_qname TEXT NOT NULL,
                target_qname TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_edges_kind ON edges(kind);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
            CREATE INDEX IF NOT EXISTS idx_nodes_qname ON nodes(qualified_name);
            CREATE INDEX IF NOT EXISTS idx_nodes_kind ON nodes(kind);

            CREATE TABLE IF NOT EXISTS file_hashes (
                file_path TEXT PRIMARY KEY,
                hash TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS graph_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                qualified_name TEXT NOT NULL,
                model TEXT NOT NULL,
                vector TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (qualified_name, model)
            );

            -- Standalone FTS5 table for full-text search.
            -- Synced from nodes table during save/rebuild.
        """)
        fts_columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(nodes_fts)").fetchall()
        }
        expected_columns = {"node_id", "kind", "name", "qualified_name", "source_text"}
        if fts_columns and fts_columns != expected_columns:
            self._conn.execute("DROP TABLE nodes_fts")
        self._conn.execute(
            """CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                node_id UNINDEXED,
                kind UNINDEXED,
                name,
                qualified_name,
                source_text
            )"""
        )

    # ── Save graph ───────────────────────────────────────────────────

    def save_graph(self, graph: Graph, file_hashes: dict[str, str] | None = None) -> None:
        """Save a complete graph to the database.

        Clears existing data and re-inserts everything.
        """
        logger.info("Saving graph with %d nodes, %d edges", graph.node_count(), graph.edge_count())

        with self._conn:
            # Clear existing data
            self._conn.execute("DELETE FROM edges")
            self._conn.execute("DELETE FROM nodes")

            # Insert nodes, build qname → db_id mapping
            qname_to_db_id: dict[str, int] = {}
            for nid, node in graph.nodes():
                db_id = self._insert_node(node)
                qname_to_db_id[node.qualified_name] = db_id

            # Insert edges using qname mapping
            for eid, src, dst, edge in graph.edges():
                src_node = graph.node(src)
                dst_node = graph.node(dst)
                if src_node and dst_node:
                    src_db = qname_to_db_id.get(src_node.qualified_name)
                    dst_db = qname_to_db_id.get(dst_node.qualified_name)
                    if src_db is not None and dst_db is not None:
                        self._insert_edge(
                            edge, src_db, dst_db,
                            src_node.qualified_name,
                            dst_node.qualified_name,
                        )

            # Sync FTS5 index from nodes table
            self._conn.execute("DELETE FROM nodes_fts")
            self._conn.execute(
                "INSERT INTO nodes_fts (node_id, kind, name, qualified_name, source_text) "
                "SELECT node_id, kind, name, qualified_name, source_text FROM nodes"
            )

            # Update file hashes
            if file_hashes:
                self._conn.execute("DELETE FROM file_hashes")
                for path, hash_val in file_hashes.items():
                    self._conn.execute(
                        "INSERT OR REPLACE INTO file_hashes (file_path, hash) VALUES (?, ?)",
                        (path, hash_val),
                    )

            # Update metadata
            self._set_metadata("node_count", str(graph.node_count()))
            self._set_metadata("edge_count", str(graph.edge_count()))
            self._set_metadata("last_updated", _now_iso())

    def _insert_node(self, node: Node) -> int:
        """Insert a node and return its database ID."""
        cursor = self._conn.execute(
            """INSERT OR IGNORE INTO nodes
               (kind, name, qualified_name, source_uri, line_start, line_end,
                properties, source_text, valid_from, valid_to)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                node.kind.value,
                node.name,
                node.qualified_name,
                node.source_uri,
                node.line_start,
                node.line_end,
                json.dumps(node.properties),
                node.source_text,
                node.valid_from,
                node.valid_to,
            ),
        )
        node_id = cursor.lastrowid
        if node_id is None:
            # Node already exists (same qualified_name)
            row = self._conn.execute(
                "SELECT node_id FROM nodes WHERE qualified_name = ?",
                (node.qualified_name,),
            ).fetchone()
            node_id = row["node_id"] if row else 0

        # Update the node
        self._conn.execute(
            """UPDATE nodes SET
               kind=?, name=?, source_uri=?, line_start=?, line_end=?,
               properties=?, source_text=?, valid_from=?, valid_to=?
               WHERE node_id=?""",
            (
                node.kind.value,
                node.name,
                node.source_uri,
                node.line_start,
                node.line_end,
                json.dumps(node.properties),
                node.source_text,
                node.valid_from,
                node.valid_to,
                node_id,
            ),
        )

        return node_id

    def _insert_edge(
        self,
        edge: Edge,
        src_db_id: int,
        dst_db_id: int,
        source_qname: str,
        target_qname: str,
    ) -> None:
        """Insert an edge."""
        self._conn.execute(
            """INSERT INTO edges
               (kind, confidence, properties, valid_from, valid_to, source_id, target_id, source_qname, target_qname)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                edge.kind.value,
                edge.confidence.value,
                json.dumps(edge.properties),
                edge.valid_from,
                edge.valid_to,
                src_db_id,
                dst_db_id,
                source_qname,
                target_qname,
            ),
        )

    # ── Load graph ───────────────────────────────────────────────────

    def load_graph(self) -> Graph:
        """Load a graph from the database."""
        graph = Graph()

        # Load nodes, build qname → NodeId mapping
        qname_to_id: dict[str, NodeId] = {}
        rows = self._conn.execute("SELECT * FROM nodes").fetchall()
        for row in rows:
            node = self._row_to_node(row)
            nid = graph.add_node(node)
            qname_to_id[node.qualified_name] = nid

        # Load edges using qname mapping
        edge_rows = self._conn.execute("SELECT * FROM edges").fetchall()
        for row in edge_rows:
            edge = self._edge_row_to_edge(row)
            src_qn = row["source_qname"]
            dst_qn = row["target_qname"]
            src_id = qname_to_id.get(src_qn)
            dst_id = qname_to_id.get(dst_qn)
            if src_id and dst_id:
                graph.add_edge(src_id, dst_id, edge)

        logger.info("Loaded graph: %d nodes, %d edges", graph.node_count(), graph.edge_count())
        return graph

    def _row_to_node(self, row: sqlite3.Row) -> Node:
        """Convert a database row to a Node."""
        return Node(
            kind=NodeKind(row["kind"]),
            name=row["name"],
            qualified_name=row["qualified_name"],
            source_uri=row["source_uri"],
            line_start=row["line_start"],
            line_end=row["line_end"],
            properties=json.loads(row["properties"]) if row["properties"] else {},
            source_text=row["source_text"],
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
        )

    def _edge_row_to_edge(self, row: sqlite3.Row) -> Edge:
        """Convert a database row to an Edge."""
        confidence = Confidence(row["confidence"])
        return Edge(
            kind=EdgeKind(row["kind"]),
            confidence=confidence,
            properties=json.loads(row["properties"]) if row["properties"] else {},
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
        )

    # ── File hashes ──────────────────────────────────────────────────

    def get_file_hashes(self) -> dict[str, str]:
        """Get all stored file hashes."""
        rows = self._conn.execute("SELECT file_path, hash FROM file_hashes").fetchall()
        return {row["file_path"]: row["hash"] for row in rows}

    def set_file_hash(self, path: str, hash_val: str) -> None:
        """Store a file hash for incremental update detection."""
        self._conn.execute(
            "INSERT OR REPLACE INTO file_hashes (file_path, hash) VALUES (?, ?)",
            (path, hash_val),
        )

    def get_changed_files(self, current_hashes: dict[str, str]) -> tuple[list[str], list[str]]:
        """Compare current file hashes with stored hashes.

        Returns (changed_files, deleted_files).
        """
        stored = self.get_file_hashes()
        changed = [p for p, h in current_hashes.items() if stored.get(p) != h]
        deleted = [p for p in stored if p not in current_hashes]
        return changed, deleted

    # ── Metadata ─────────────────────────────────────────────────────

    def get_metadata(self, key: str) -> str | None:
        """Get a metadata value."""
        row = self._conn.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def _set_metadata(self, key: str, value: str) -> None:
        """Set a metadata value."""
        self._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, value),
        )

    def set_metadata(self, key: str, value: str) -> None:
        """Persist repository or adapter metadata."""
        with self._conn:
            self._set_metadata(key, value)

    # ── Immutable snapshots ─────────────────────────────────────────

    def create_snapshot(self, label: str) -> None:
        """Persist the current active graph under a stable label."""
        if not label.strip():
            raise ValueError("snapshot label must not be empty")
        graph = self.load_graph()
        payload = json.dumps(_graph_to_payload(graph), separators=(",", ":"))
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO graph_snapshots(label, created_at, payload) VALUES (?, ?, ?)",
                (label, _now_iso(), payload),
            )

    def list_snapshots(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT label, created_at FROM graph_snapshots ORDER BY snapshot_id DESC"
        ).fetchall()
        return [{"label": row["label"], "created_at": row["created_at"]} for row in rows]

    def load_snapshot(self, label: str) -> Graph:
        row = self._conn.execute(
            "SELECT payload FROM graph_snapshots WHERE label = ?", (label,)
        ).fetchone()
        if row is None:
            raise KeyError(f"snapshot not found: {label}")
        return _graph_from_payload(json.loads(row["payload"]))

    def load_graph_at(self, revision: str) -> Graph:
        """Load a graph snapshot by commit/revision label."""
        if self.get_metadata("indexed_commit") == revision:
            return self.load_graph()
        return self.load_snapshot(revision)

    # ── Durable embeddings ──────────────────────────────────────────

    def save_embeddings(self, model: str, vectors: dict[str, list[float]]) -> None:
        if not model.strip():
            raise ValueError("embedding model must not be empty")
        with self._conn:
            self._conn.execute("DELETE FROM embeddings WHERE model = ?", (model,))
            self._conn.executemany(
                "INSERT INTO embeddings(qualified_name, model, vector, updated_at) VALUES (?, ?, ?, ?)",
                [
                    (qname, model, json.dumps(vector), _now_iso())
                    for qname, vector in vectors.items()
                ],
            )

    def load_embeddings(self, model: str) -> dict[str, list[float]]:
        rows = self._conn.execute(
            "SELECT qualified_name, vector FROM embeddings WHERE model = ?", (model,)
        ).fetchall()
        return {row["qualified_name"]: json.loads(row["vector"]) for row in rows}

    def clear_embeddings(self, model: str | None = None) -> None:
        with self._conn:
            if model is None:
                self._conn.execute("DELETE FROM embeddings")
            else:
                self._conn.execute("DELETE FROM embeddings WHERE model = ?", (model,))

    # ── Status ───────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Get database status information."""
        node_count = self._conn.execute("SELECT COUNT(*) as cnt FROM nodes").fetchone()["cnt"]
        edge_count = self._conn.execute("SELECT COUNT(*) as cnt FROM edges").fetchone()["cnt"]
        kind_dist = self._conn.execute(
            "SELECT kind, COUNT(*) as cnt FROM nodes GROUP BY kind"
        ).fetchall()
        edge_dist = self._conn.execute(
            "SELECT kind, COUNT(*) as cnt FROM edges GROUP BY kind"
        ).fetchall()
        last_updated = self.get_metadata("last_updated")

        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "kind_distribution": {r["kind"]: r["cnt"] for r in kind_dist},
            "edge_distribution": {r["kind"]: r["cnt"] for r in edge_dist},
            "last_updated": last_updated,
            "snapshots": len(self.list_snapshots()),
        }

    def rebuild_fts(self) -> int:
        """Rebuild the durable FTS index and return indexed row count."""
        with self._conn:
            self._conn.execute("DELETE FROM nodes_fts")
            self._conn.execute(
                "INSERT INTO nodes_fts (node_id, kind, name, qualified_name, source_text) "
                "SELECT node_id, kind, name, qualified_name, source_text FROM nodes"
            )
        return int(self._conn.execute("SELECT COUNT(*) FROM nodes_fts").fetchone()[0])

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> GraphStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def _now_iso() -> str:
    """Get current time as ISO string."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _graph_to_payload(graph: Graph) -> dict[str, Any]:
    nodes = []
    for _, node in graph.nodes():
        nodes.append({
            "kind": node.kind.value,
            "name": node.name,
            "qualified_name": node.qualified_name,
            "source_uri": node.source_uri,
            "line_start": node.line_start,
            "line_end": node.line_end,
            "properties": node.properties,
            "source_text": node.source_text,
            "valid_from": node.valid_from,
            "valid_to": node.valid_to,
        })
    edges = []
    for _, source, target, edge in graph.edges():
        source_node = graph.node(source)
        target_node = graph.node(target)
        if source_node is None or target_node is None:
            continue
        edges.append({
            "source": source_node.qualified_name,
            "target": target_node.qualified_name,
            "kind": edge.kind.value,
            "confidence": edge.confidence.value,
            "properties": edge.properties,
            "valid_from": edge.valid_from,
            "valid_to": edge.valid_to,
        })
    return {"nodes": nodes, "edges": edges}


def _graph_from_payload(payload: dict[str, Any]) -> Graph:
    graph = Graph()
    for item in payload.get("nodes", []):
        graph.add_node(Node(
            kind=NodeKind(item["kind"]),
            name=item["name"],
            qualified_name=item["qualified_name"],
            source_uri=item.get("source_uri"),
            line_start=item.get("line_start"),
            line_end=item.get("line_end"),
            properties=item.get("properties", {}),
            source_text=item.get("source_text"),
            valid_from=item.get("valid_from"),
            valid_to=item.get("valid_to"),
        ))
    for item in payload.get("edges", []):
        source = graph.find_by_qname(item["source"])
        target = graph.find_by_qname(item["target"])
        if source is None or target is None:
            continue
        graph.add_edge(source, target, Edge(
            kind=EdgeKind(item["kind"]),
            confidence=Confidence(item["confidence"]),
            properties=item.get("properties", {}),
            valid_from=item.get("valid_from"),
            valid_to=item.get("valid_to"),
        ))
    return graph
