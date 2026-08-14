"""SQLite persistence for the Ariadne graph.

Stores nodes, edges, file hashes, and metadata in a single SQLite database.
Supports incremental updates via file hash comparison.
Includes schema migrations, versioned tables for temporal tracking,
and confidence class tracking.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.edge import Edge, EdgeKind, Confidence
from ..core.graph import Graph
from ..core.id import EdgeId, NodeId
from ..core.node import Node, NodeKind
from .embeddings.local import (
    cosine_similarity,
    encode_embedding,
    embedding_source_text,
    semantic_embedding,
)

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL: str = "ariadne-hash-v2"
DEFAULT_EMBEDDING_DIM: int = 384


@dataclass(frozen=True)
class EdgeIdentity:
    """Unique identity string for an edge (src_qname, dst_qname, kind)."""
    src_qname: str
    dst_qname: str
    kind: EdgeKind

    def to_string(self) -> str:
        return f"{self.src_qname}\x1f{self.dst_qname}\x1f{self.kind.value}"


def edge_identity(src_qname: str, dst_qname: str, kind: EdgeKind) -> EdgeIdentity:
    """Create a unique edge identity string (src_qname, dst_qname, kind)."""
    return EdgeIdentity(src_qname, dst_qname, kind)


def parse_confidence(conf_class: str, confidence: float) -> Confidence:
    """Parse confidence class from DB string to Confidence enum."""
    if conf_class == "extracted":
        return Confidence.EXTRACTED
    elif conf_class == "ambiguous":
        return Confidence.AMBIGUOUS
    else:
        return Confidence.INFERRED


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

    def __init__(self, db_path: str | Path, read_only: bool = False) -> None:
        self.db_path = Path(db_path)
        flags = sqlite3.URI if str(db_path).startswith("file:") else 0
        self._conn = sqlite3.connect(str(self.db_path), uri=(flags != 0))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()
        self._migrate_v1()

    def _migrate_v1(self) -> None:
        """Apply schema migrations for older databases.

        Migration v1→v2: adds source_text column to nodes and node_versions,
        and updates schema_version metadata.
        """
        try:
            has_column = self._conn.execute(
                "SELECT COUNT(*) FROM pragma_table_info('nodes') WHERE name='source_text'"
            ).fetchone()[0]
        except Exception:
            has_column = 0
        if has_column == 0:
            try:
                self._conn.execute("ALTER TABLE nodes ADD COLUMN source_text TEXT")
            except Exception:
                pass
            try:
                self._conn.execute("ALTER TABLE node_versions ADD COLUMN source_text TEXT")
            except Exception:
                pass
            try:
                self._conn.execute("UPDATE metadata SET value='2' WHERE key='schema_version'")
            except Exception:
                pass
            self._set_metadata("schema_version", "2")

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                node_id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                name TEXT NOT NULL,
                qualified_name TEXT NOT NULL UNIQUE,
                source_uri TEXT,
                line_start INTEGER,
                line_end INTEGER,
                properties TEXT NOT NULL DEFAULT '{}',
                valid_from TEXT,
                valid_to TEXT,
                source_text TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_nodes_kind ON nodes(kind);
            CREATE INDEX IF NOT EXISTS idx_nodes_qname ON nodes(qualified_name);
            CREATE INDEX IF NOT EXISTS idx_nodes_source ON nodes(source_uri);
            CREATE INDEX IF NOT EXISTS idx_nodes_valid ON nodes(valid_from, valid_to);

            CREATE TABLE IF NOT EXISTS edges (
                edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                conf_class TEXT NOT NULL DEFAULT 'extracted',
                properties TEXT NOT NULL DEFAULT '{}',
                valid_from TEXT,
                valid_to TEXT,
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                source_qname TEXT NOT NULL,
                target_qname TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(target_id);
            CREATE INDEX IF NOT EXISTS idx_edges_kind ON edges(kind);
            CREATE INDEX IF NOT EXISTS idx_edges_valid ON edges(valid_from, valid_to);

            CREATE TABLE IF NOT EXISTS node_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                name TEXT NOT NULL,
                qualified_name TEXT NOT NULL,
                source_uri TEXT,
                line_start INTEGER,
                line_end INTEGER,
                properties TEXT NOT NULL DEFAULT '{}',
                valid_from TEXT,
                valid_to TEXT,
                source_text TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_node_versions_qname ON node_versions(qualified_name);
            CREATE INDEX IF NOT EXISTS idx_node_versions_source ON node_versions(source_uri);
            CREATE INDEX IF NOT EXISTS idx_node_versions_valid ON node_versions(valid_from, valid_to);

            CREATE TABLE IF NOT EXISTS edge_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                src_qname TEXT NOT NULL,
                dst_qname TEXT NOT NULL,
                kind TEXT NOT NULL,
                confidence REAL NOT NULL,
                conf_class TEXT NOT NULL,
                properties TEXT NOT NULL DEFAULT '{}',
                source_uri TEXT,
                valid_from TEXT,
                valid_to TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_edge_versions_src ON edge_versions(src_qname);
            CREATE INDEX IF NOT EXISTS idx_edge_versions_dst ON edge_versions(dst_qname);
            CREATE INDEX IF NOT EXISTS idx_edge_versions_kind ON edge_versions(kind);
            CREATE INDEX IF NOT EXISTS idx_edge_versions_valid ON edge_versions(valid_from, valid_to);

            CREATE TABLE IF NOT EXISTS embeddings (
                node_id INTEGER PRIMARY KEY,
                model TEXT NOT NULL,
                vector BLOB NOT NULL
            );

            -- File state with index timestamp.
            CREATE TABLE IF NOT EXISTS file_state (
                path TEXT PRIMARY KEY,
                hash TEXT NOT NULL,
                indexed_at_unix INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            -- Durable snapshots.
            CREATE TABLE IF NOT EXISTS graph_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );

            -- FTS5 index for full-text search.
            CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                kind,
                name,
                qualified_name
            );
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
                self._conn.execute("DELETE FROM file_state")
                now_unix = int(_now_unix())
                for path, hash_val in file_hashes.items():
                    self._conn.execute(
                        "INSERT OR REPLACE INTO file_state (path, hash, indexed_at_unix) VALUES (?, ?, ?)",
                        (path, hash_val, now_unix),
                    )

            # Update metadata
            self._set_metadata("node_count", str(graph.node_count()))
            self._set_metadata("edge_count", str(graph.edge_count()))
            self._set_metadata("last_updated", _now_iso())

    def save_graph_incremental(
        self,
        graph: Graph,
        file_hashes: dict[str, str] | None = None,
    ) -> None:
        """Synchronize the active graph while preserving unchanged rows.

        Stable node and edge IDs are retained, FTS rows are refreshed only
        for changed nodes, and local embeddings are regenerated only for the
        affected nodes. Embeddings from external providers are invalidated for
        changed nodes because rebuilding them may require network access.
        """
        existing_nodes = {
            row["qualified_name"]: row
            for row in self._conn.execute("SELECT * FROM nodes").fetchall()
        }
        target_qnames = {node.qualified_name for _, node in graph.nodes()}
        removed_node_ids = {
            int(row["node_id"])
            for qname, row in existing_nodes.items()
            if qname not in target_qnames
        }
        embedding_model_row = self._conn.execute(
            "SELECT model FROM embeddings LIMIT 1"
        ).fetchone()
        embedding_model = embedding_model_row["model"] if embedding_model_row else None
        changed_node_ids: set[int] = set()
        qname_to_db_id: dict[str, int] = {}

        with self._conn:
            for _, node in graph.nodes():
                properties = json.dumps(node.properties)
                values = (
                    node.kind.value,
                    node.name,
                    node.source_uri,
                    node.line_start,
                    node.line_end,
                    properties,
                    node.source_text,
                    node.valid_from,
                    node.valid_to,
                )
                existing = existing_nodes.get(node.qualified_name)
                if existing is None:
                    cursor = self._conn.execute(
                        """INSERT INTO nodes
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
                            properties,
                            node.source_text,
                            node.valid_from,
                            node.valid_to,
                        ),
                    )
                    db_id = int(cursor.lastrowid)
                    changed_node_ids.add(db_id)
                else:
                    db_id = int(existing["node_id"])
                    stored = (
                        existing["kind"],
                        existing["name"],
                        existing["source_uri"],
                        existing["line_start"],
                        existing["line_end"],
                        existing["properties"],
                        existing["source_text"],
                        existing["valid_from"],
                        existing["valid_to"],
                    )
                    if stored != values:
                        self._conn.execute(
                            """UPDATE nodes SET
                               kind=?, name=?, source_uri=?, line_start=?, line_end=?,
                               properties=?, source_text=?, valid_from=?, valid_to=?
                               WHERE node_id=?""",
                            (*values, db_id),
                        )
                        changed_node_ids.add(db_id)
                qname_to_db_id[node.qualified_name] = db_id

            existing_edges: dict[tuple[Any, ...], list[int]] = {}
            for row in self._conn.execute("SELECT * FROM edges").fetchall():
                key = (
                    row["source_qname"],
                    row["target_qname"],
                    row["kind"],
                    float(row["confidence"]),
                    row["conf_class"],
                    row["properties"],
                    row["valid_from"],
                    row["valid_to"],
                )
                existing_edges.setdefault(key, []).append(int(row["edge_id"]))

            for _, src, dst, edge in graph.edges():
                src_node = graph.node(src)
                dst_node = graph.node(dst)
                if src_node is None or dst_node is None:
                    continue
                confidence = edge.confidence.score()
                conf_class = _confidence_class_name(edge.confidence)
                properties = json.dumps(edge.properties)
                key = (
                    src_node.qualified_name,
                    dst_node.qualified_name,
                    edge.kind.value,
                    confidence,
                    conf_class,
                    properties,
                    edge.valid_from,
                    edge.valid_to,
                )
                retained = existing_edges.get(key)
                if retained:
                    retained.pop()
                    continue
                self._insert_edge(
                    edge,
                    qname_to_db_id[src_node.qualified_name],
                    qname_to_db_id[dst_node.qualified_name],
                    src_node.qualified_name,
                    dst_node.qualified_name,
                )

            stale_edge_ids = [
                edge_id for ids in existing_edges.values() for edge_id in ids
            ]
            if stale_edge_ids:
                self._conn.executemany(
                    "DELETE FROM edges WHERE edge_id=?",
                    [(edge_id,) for edge_id in stale_edge_ids],
                )

            invalidated_ids = changed_node_ids | removed_node_ids
            if invalidated_ids:
                self._conn.executemany(
                    "DELETE FROM nodes_fts WHERE node_id=?",
                    [(node_id,) for node_id in invalidated_ids],
                )
                self._conn.executemany(
                    "DELETE FROM embeddings WHERE node_id=?",
                    [(node_id,) for node_id in invalidated_ids],
                )

            if removed_node_ids:
                self._conn.executemany(
                    "DELETE FROM nodes WHERE node_id=?",
                    [(node_id,) for node_id in removed_node_ids],
                )

            if changed_node_ids:
                placeholders = ",".join("?" for _ in changed_node_ids)
                self._conn.execute(
                    f"""INSERT INTO nodes_fts
                        (node_id, kind, name, qualified_name, source_text)
                        SELECT node_id, kind, name, qualified_name, source_text
                        FROM nodes WHERE node_id IN ({placeholders})""",
                    tuple(sorted(changed_node_ids)),
                )

            if file_hashes is not None:
                self._conn.execute("DELETE FROM file_state")
                now_unix = int(_now_unix())
                self._conn.executemany(
                    "INSERT INTO file_state (path, hash, indexed_at_unix) VALUES (?, ?, ?)",
                    [(path, value, now_unix) for path, value in file_hashes.items()],
                )

            self._set_metadata("node_count", str(graph.node_count()))
            self._set_metadata("edge_count", str(graph.edge_count()))
            self._set_metadata("last_updated", _now_iso())

        if embedding_model == DEFAULT_EMBEDDING_MODEL and changed_node_ids:
            self._rebuild_local_embeddings_for_ids(changed_node_ids)

    def _rebuild_local_embeddings_for_ids(self, node_ids: set[int]) -> None:
        """Regenerate local embeddings for the selected persisted nodes."""
        if not node_ids:
            return
        placeholders = ",".join("?" for _ in node_ids)
        rows = self._conn.execute(
            f"""SELECT node_id, kind, name, qualified_name, source_uri, source_text
                FROM nodes
                WHERE node_id IN ({placeholders})
                  AND qualified_name NOT LIKE 'call::%'""",
            tuple(sorted(node_ids)),
        ).fetchall()
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embeddings (node_id, model, vector) VALUES (?, ?, ?)",
                [
                    (
                        row["node_id"],
                        DEFAULT_EMBEDDING_MODEL,
                        encode_embedding(
                            semantic_embedding(
                                embedding_source_text(
                                    row["kind"],
                                    row["name"],
                                    row["qualified_name"],
                                    row["source_uri"],
                                    row["source_text"],
                                )
                            )
                        ),
                    )
                    for row in rows
                ],
            )

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
        conf_class = edge.confidence.class_name() if hasattr(edge.confidence, 'class_name') else _confidence_class_name(edge.confidence)
        conf_score = edge.confidence.score() if hasattr(edge.confidence, 'score') else 1.0
        self._conn.execute(
            """INSERT INTO edges
               (kind, confidence, conf_class, properties, valid_from, valid_to, source_id, target_id, source_qname, target_qname)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                edge.kind.value,
                conf_score,
                conf_class,
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
        conf_class = row["conf_class"] if row["conf_class"] else "extracted"
        confidence = parse_confidence(conf_class, row["confidence"])
        return Edge(
            kind=EdgeKind(row["kind"]),
            confidence=confidence,
            properties=json.loads(row["properties"]) if row["properties"] else {},
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
        )

    # ── File hashes / file_state ─────────────────────────────────────

    def get_file_hashes(self) -> dict[str, str]:
        """Get all stored file hashes (legacy alias)."""
        rows = self._conn.execute("SELECT path, hash FROM file_state").fetchall()
        return {row["path"]: row["hash"] for row in rows}

    def get_file_state(self) -> dict[str, dict[str, Any]]:
        """Get all file state entries with timestamps."""
        rows = self._conn.execute("SELECT path, hash, indexed_at_unix FROM file_state").fetchall()
        return {
            row["path"]: {
                "hash": row["hash"],
                "indexed_at": row["indexed_at_unix"],
            }
            for row in rows
        }

    def set_file_hash(self, path: str, hash_val: str) -> None:
        """Store a file hash for incremental update detection."""
        now_unix = int(_now_unix())
        self._conn.execute(
            "INSERT OR REPLACE INTO file_state (path, hash, indexed_at_unix) VALUES (?, ?, ?)",
            (path, hash_val, now_unix),
        )

    def set_file_hashes(self, hashes: dict[str, str]) -> None:
        """Bulk store file hashes."""
        now_unix = int(_now_unix())
        self._conn.execute("DELETE FROM file_state")
        self._conn.executemany(
            "INSERT OR REPLACE INTO file_state (path, hash, indexed_at_unix) VALUES (?, ?, ?)",
            [(p, h, now_unix) for p, h in hashes.items()],
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
        """Save embeddings keyed by qualified name.

        The vectors are stored as BLOBs (struct-packed floats).
        """
        if not model.strip():
            raise ValueError("embedding model must not be empty")
        from ..persistence.embeddings.local import encode_embedding

        with self._conn:
            self._conn.execute("DELETE FROM embeddings WHERE model = ?", (model,))
            self._conn.executemany(
                "INSERT INTO embeddings(node_id, model, vector) VALUES (?, ?, ?)",
                [
                    (self._resolve_qname(qname), model, encode_embedding(vector))
                    for qname, vector in vectors.items()
                    if self._resolve_qname(qname) is not None
                ],
            )

    def load_embeddings(self, model: str) -> dict[str, list[float]]:
        """Load embeddings for a given model, keyed by qualified name."""
        from ..persistence.embeddings.local import decode_embedding

        rows = self._conn.execute(
            "SELECT node_id, vector FROM embeddings WHERE model = ?", (model,)
        ).fetchall()

        # Map node_id back to qualified_name
        node_map: dict[int, str] = {}
        for node_id, qname in self._conn.execute(
            "SELECT node_id, qualified_name FROM nodes"
        ).fetchall():
            node_map[node_id] = qname

        result: dict[str, list[float]] = {}
        for row in rows:
            nid = row["node_id"]
            qname = node_map.get(nid)
            if qname is not None:
                vec = decode_embedding(row["vector"])
                if vec is not None:
                    result[qname] = vec
        return result

    def clear_embeddings(self, model: str | None = None) -> None:
        with self._conn:
            if model is None:
                self._conn.execute("DELETE FROM embeddings")
            else:
                self._conn.execute("DELETE FROM embeddings WHERE model = ?", (model,))

    def get_embedding_stats(self) -> tuple[int, str | None] | None:
        """Return (count, model) or None if no embeddings."""
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt, model FROM embeddings LIMIT 1"
        ).fetchone()
        if row is None or row["cnt"] == 0:
            return None
        return (row["cnt"], row["model"])

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

    # ── FTS5 search / stats ────────────────────────────────────────

    def fts_stats(self) -> int:
        """Return the number of rows in the FTS5 index."""
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM nodes_fts").fetchone()
        return row["cnt"] if row else 0

    def fts_search(self, query: str, limit: int = 20) -> list[tuple[str, float]]:
        """Full-text search via the FTS5 ``nodes_fts`` virtual table.

        Returns ``(qualified_name, bm25_score)`` pairs ordered by relevance
        (bm25 is negative; we negate so higher = better).
        Returns an empty list if the FTS table is not yet populated or the
        query contains no indexable tokens.

        Mirrors the Rust ``fts_search`` from ``persistence/database/mod.rs``.
        """
        if not query or not query.strip() or limit <= 0:
            return []

        from .fts import build_fts5_query

        safe_query = build_fts5_query(query)
        if not safe_query:
            return []

        try:
            sql = (
                "SELECT n.qualified_name, bm25(nodes_fts) "
                "FROM nodes_fts "
                "JOIN nodes n ON nodes_fts.node_id = n.node_id "
                "WHERE nodes_fts MATCH ? "
                "ORDER BY bm25(nodes_fts) "
                "LIMIT ?"
            )
            rows = self._conn.execute(sql, (safe_query, limit)).fetchall()
        except Exception:
            return []

        results: list[tuple[str, float]] = []
        for qname, score in rows:
            results.append((qname, score))
        return results

    # ── Temporal / versioned queries ───────────────────────────────

    def temporal_nodes(self) -> list[Node]:
        """Union of active nodes and archived (closed-out) rows.

        Mirrors the Rust ``temporal_nodes``.
        """
        out: list[Node] = []
        seen_qnames: set[str] = set()
        rows = self._conn.execute(
            "SELECT name, kind, qualified_name, source_uri, line_start, line_end, "
            "properties, valid_from, valid_to, source_text "
            "FROM nodes "
            "UNION ALL "
            "SELECT name, kind, qualified_name, source_uri, line_start, line_end, "
            "properties, valid_from, valid_to, source_text "
            "FROM node_versions"
        ).fetchall()
        # Sort so archived rows (valid_to is not None) come before active rows.
        # This way active rows overwrite archived ones (same qname).
        rows.sort(key=lambda r: r["valid_to"] is not None)
        for row in rows:
            qname = row["qualified_name"]
            if qname not in seen_qnames:
                seen_qnames.add(qname)
                out.append(self._row_to_node(row))
        return out

    def temporal_edges(self) -> list[tuple[str, str, Edge, str | None]]:
        """Union of active edges and archived rows (via JOIN + UNION).

        Returns list of ``(src_qname, dst_qname, Edge, source_uri | None)``.
        Mirrors the Rust ``temporal_edges``.
        """
        out: list[tuple[str, str, Edge, str | None]] = []
        seen: set[tuple[str, str, EdgeKind]] = set()

        rows = self._conn.execute(
            "SELECT s.qualified_name, d.qualified_name, e.kind, e.confidence, "
            "e.conf_class, e.properties, e.valid_from, e.valid_to, "
            "COALESCE(s.source_uri, d.source_uri) "
            "FROM edges e "
            "JOIN nodes s ON e.source_id = s.node_id "
            "JOIN nodes d ON e.target_id = d.node_id "
            "UNION ALL "
            "SELECT src_qname, dst_qname, kind, confidence, conf_class, "
            "properties, valid_from, valid_to, source_uri "
            "FROM edge_versions"
        ).fetchall()
        for row in rows:
            src_qn = row[0]
            dst_qn = row[1]
            kind_str = row[2]
            conf_score = row[3]
            conf_class = row[4]
            props_str = row[5]
            valid_from = row[6]
            valid_to = row[7]
            source_uri = row[8]

            try:
                kind = EdgeKind(kind_str)
            except ValueError:
                continue
            confidence = parse_confidence(conf_class, conf_score)
            edge = Edge(
                kind=kind,
                confidence=confidence,
                properties=json.loads(props_str) if props_str else {},
                valid_from=valid_from,
                valid_to=valid_to,
            )
            identity = (src_qn, dst_qn, kind)
            if identity not in seen:
                seen.add(identity)
                out.append((src_qn, dst_qn, edge, source_uri))
        return out

    def load_temporal(self) -> Graph:
        """Load a graph including archived (closed-out) rows from the version
        tables, so temporal diffs can see nodes/edges that are no longer
        active.

        Archived rows are inserted first and active rows last, so for a
        qualified name present in both the active state wins.

        Mirrors the Rust ``load_temporal``.
        """
        graph = Graph()

        # Load nodes: sorted so archived rows (valid_to set) come before
        # active rows — active rows overwrite.
        nodes = self.temporal_nodes()
        # Sort: archived (valid_to is not None) first, then active
        nodes.sort(key=lambda n: n.valid_to is not None)
        for node in nodes:
            graph.add_node(node)

        # Load edges: sorted so archived rows come first
        edges = self.temporal_edges()
        edges.sort(key=lambda e: e[2].valid_to is not None)
        for src_qn, dst_qn, edge, _ in edges:
            src = graph.find_by_qname(src_qn)
            dst = graph.find_by_qname(dst_qn)
            if src is not None and dst is not None:
                graph.add_edge(src, dst, edge)

        return graph

    def archive_nodes(self, nodes: list[Node], valid_to: str) -> None:
        """Archive nodes to the version table with a ``valid_to`` timestamp.

        Mirrors the Rust ``archive_nodes``.
        """
        if not nodes:
            return
        with self._conn:
            for node in nodes:
                props = json.dumps(node.properties)
                vf = node.valid_from or valid_to
                self._conn.execute(
                    "INSERT INTO node_versions "
                    "(kind, name, qualified_name, source_uri, line_start, "
                    "line_end, properties, valid_from, valid_to) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        node.kind.value,
                        node.name,
                        node.qualified_name,
                        node.source_uri,
                        node.line_start,
                        node.line_end,
                        props,
                        vf,
                        valid_to,
                    ),
                )

    def archive_edges(
        self,
        edges: list[tuple[str, str, Edge, str | None]],
        valid_to: str,
    ) -> None:
        """Archive edges to the version table with a ``valid_to`` timestamp.

        Args:
            edges: List of ``(src_qname, dst_qname, Edge, source_uri)``.
            valid_to: Timestamp to set on archived rows.

        Mirrors the Rust ``archive_edges``.
        """
        if not edges:
            return
        with self._conn:
            for src_qn, dst_qn, edge, source_uri in edges:
                props = json.dumps(edge.properties)
                vf = edge.valid_from or valid_to
                conf_class = edge.confidence.class_name() if hasattr(
                    edge.confidence, 'class_name'
                ) else _confidence_class_name(edge.confidence)
                conf_score = edge.confidence.score() if hasattr(
                    edge.confidence, 'score'
                ) else 1.0
                self._conn.execute(
                    "INSERT INTO edge_versions "
                    "(src_qname, dst_qname, kind, confidence, conf_class, "
                    "properties, source_uri, valid_from, valid_to) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        src_qn,
                        dst_qn,
                        edge.kind.value,
                        conf_score,
                        conf_class,
                        props,
                        source_uri,
                        vf,
                        valid_to,
                    ),
                )

    # ── Source-filtered reads ──────────────────────────────────────

    def active_nodes_for_sources(self, sources: list[str]) -> list[Node]:
        """Read active nodes whose ``source_uri`` matches any of the given sources.

        Returns unique nodes by qualified name (first occurrence wins).
        Mirrors the Rust ``active_nodes_for_sources``.
        """
        out: list[Node] = []
        seen: set[str] = set()
        sql = (
            "SELECT name, kind, qualified_name, source_uri, line_start, line_end, "
            "properties, valid_from, valid_to, source_text "
            "FROM nodes WHERE source_uri = ?"
        )
        for source in sources:
            rows = self._conn.execute(sql, (source,)).fetchall()
            for row in rows:
                node = self._row_to_node(row)
                if node.qualified_name not in seen:
                    seen.add(node.qualified_name)
                    out.append(node)
        return out

    def active_edges_for_sources(self, sources: list[str]) -> list[tuple[str, str, Edge, str | None]]:
        """Read active edges where at least one endpoint matches a given source.

        Returns unique edges by ``(src_qname, dst_qname, kind)``.
        Mirrors the Rust ``active_edges_for_sources``.
        """
        out: list[tuple[str, str, Edge, str | None]] = []
        seen: set[tuple[str, str, EdgeKind]] = set()
        sql = (
            "SELECT s.qualified_name, d.qualified_name, e.kind, e.confidence, "
            "e.conf_class, e.properties, e.valid_from, e.valid_to, "
            "COALESCE(s.source_uri, d.source_uri) "
            "FROM edges e "
            "JOIN nodes s ON e.source_id = s.node_id "
            "JOIN nodes d ON e.target_id = d.node_id "
            "WHERE s.source_uri = ? OR d.source_uri = ?"
        )
        for source in sources:
            rows = self._conn.execute(sql, (source, source)).fetchall()
            for row in rows:
                src_qn = row[0]
                dst_qn = row[1]
                try:
                    kind = EdgeKind(row[2])
                except ValueError:
                    continue
                confidence = parse_confidence(row[4], row[3])
                edge = Edge(
                    kind=kind,
                    confidence=confidence,
                    properties=json.loads(row[5]) if row[5] else {},
                    valid_from=row[6],
                    valid_to=row[7],
                )
                source_uri = row[8]
                identity = (src_qn, dst_qn, kind)
                if identity not in seen:
                    seen.add(identity)
                    out.append((src_qn, dst_qn, edge, source_uri))
        return out

    # ── Source deletion ────────────────────────────────────────────

    def delete_sources(self, sources: list[str]) -> None:
        """Delete all nodes, edges, FTS, embeddings, and file_state for
        the given source URIs.

        Mirrors the Rust ``delete_sources``.
        """
        with self._conn:
            for source in sources:
                self._conn.execute(
                    "DELETE FROM edges "
                    "WHERE source_id IN (SELECT node_id FROM nodes WHERE source_uri = ?) "
                    "   OR target_id IN (SELECT node_id FROM nodes WHERE source_uri = ?)",
                    (source, source),
                )
                self._conn.execute(
                    "DELETE FROM nodes_fts WHERE node_id IN "
                    "(SELECT node_id FROM nodes WHERE source_uri = ?)",
                    (source,),
                )
                self._conn.execute(
                    "DELETE FROM embeddings WHERE node_id IN "
                    "(SELECT node_id FROM nodes WHERE source_uri = ?)",
                    (source,),
                )
                self._conn.execute(
                    "DELETE FROM nodes WHERE source_uri = ?",
                    (source,),
                )
                self._conn.execute(
                    "DELETE FROM file_state WHERE path = ?",
                    (source,),
                )

    # ── Semantic search ────────────────────────────────────────────

    def semantic_search(
        self,
        query: str,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        """Semantic search over stored embeddings.

        Returns ``(qualified_name, cosine_score)`` pairs ordered descending.
        Mirrors the Rust ``semantic_search``.
        """
        if not query or not query.strip() or limit <= 0:
            return []

        from ..persistence.embeddings.local import (
            decode_embedding,
            semantic_embedding,
        )

        stats = self.get_embedding_stats()
        if stats is None or stats[0] == 0:
            return []

        count, model = stats
        model = model or DEFAULT_EMBEDDING_MODEL

        # Build query embedding using the same model
        query_vector = semantic_embedding(query)

        # Query embeddings
        rows = self._conn.execute(
            "SELECT n.qualified_name, e.vector "
            "FROM embeddings e "
            "JOIN nodes n ON e.node_id = n.node_id "
            "WHERE n.qualified_name NOT LIKE 'call::%'"
        ).fetchall()

        results: list[tuple[str, float]] = []
        for qname, blob in rows:
            vector = decode_embedding(blob)
            if vector is None:
                continue
            score = cosine_similarity(query_vector, vector)
            if score >= 0.20:
                results.append((qname, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    # ── Embedding rebuild ──────────────────────────────────────────

    def rebuild_embeddings(self) -> int:
        """Rebuild all embeddings from node data using the local model.

        Deletes existing embeddings, then regenerates them for every
        non-placeholder node. Returns the number of embeddings stored.

        Mirrors the Rust ``rebuild_embeddings``.
        """
        from ..persistence.embeddings.local import (
            decode_embedding,
            embedding_source_text,
            semantic_embedding,
            encode_embedding,
        )

        nodes_data = self._fetch_embeddable_nodes()

        with self._conn:
            self._conn.execute("DELETE FROM embeddings")
            self._conn.executemany(
                "INSERT INTO embeddings (node_id, model, vector) VALUES (?, ?, ?)",
                [
                    (
                        row["node_id"],
                        DEFAULT_EMBEDDING_MODEL,
                        encode_embedding(
                            semantic_embedding(
                                embedding_source_text(
                                    row["kind"],
                                    row["name"],
                                    row["qualified_name"],
                                    row["source_uri"],
                                    row["source_text"],
                                )
                            )
                        ),
                    )
                    for row in nodes_data
                ],
            )

        stats = self.get_embedding_stats()
        return stats[0] if stats else 0

    def rebuild_external_embeddings(
        self,
        config: dict[str, Any] | None = None,
    ) -> int:
        """Rebuild embeddings using an external provider (OpenAI / Google / Ollama).

        Args:
            config: Dict with keys ``provider`` (one of ``openai-embedding``,
                ``google-embedding``, ``ollama-embedding``) and optionally
                ``api_key`` and ``dimension``. If ``None`` falls back to the
                ``build_external_embeddings`` helper from the embeddings module.

        Returns the number of embeddings stored.

        Mirrors the Rust ``rebuild_external_embeddings``.
        """
        from ..persistence.embeddings import (
            ExternalEmbeddingConfig,
            external_embedding_from_config,
        )

        if config is None:
            config = {}

        ext_config = ExternalEmbeddingConfig(**config)
        model_name = ext_config.model or (
            {
                "openai-embedding": "text-embedding-3-small",
                "google-embedding": "text-embedding-004",
                "ollama-embedding": "nomic-embed-text",
            }.get(ext_config.provider, "unknown")
        )
        storage_model = f"{ext_config.provider}:{model_name}"

        # Build vectors before opening the write transaction.
        nodes_data = self._fetch_embeddable_nodes()
        vectors: dict[str, list[float]] = {}
        for row in nodes_data:
            text = embedding_source_text(
                row["kind"],
                row["name"],
                row["qualified_name"],
                row["source_uri"],
                row["source_text"],
            )
            try:
                vec = external_embedding_from_config(ext_config, text)
            except Exception as e:
                logger.warning("failed to embed %s: %s", row["qualified_name"], e)
                continue
            if len(vec) != ext_config.dimension:
                logger.warning(
                    "dimension mismatch for %s: expected %d, got %d",
                    row["qualified_name"],
                    ext_config.dimension,
                    len(vec),
                )
                continue
            vectors[row["qualified_name"]] = vec

        # Store in a single transaction.
        with self._conn:
            self._conn.execute("DELETE FROM embeddings")
            self.save_embeddings(storage_model, vectors)

        stats = self.get_embedding_stats()
        return stats[0] if stats else 0

    def _fetch_embeddable_nodes(self) -> list[sqlite3.Row]:
        """Fetch node data for embedding generation.

        Returns rows with columns: node_id, kind, name, qualified_name,
        source_uri, source_text. Mirrors the Rust ``fetch_embeddable_nodes``.
        """
        return self._conn.execute(
            "SELECT node_id, kind, name, qualified_name, source_uri, source_text "
            "FROM nodes WHERE qualified_name NOT LIKE 'call::%'"
        ).fetchall()

    @property
    def conn(self) -> sqlite3.Connection:
        """Raw SQLite connection for temporal and differential queries.

        Mirrors the Rust ``conn`` accessor.
        """
        return self._conn

    @classmethod
    def open_in_memory(cls) -> GraphStore:
        """Open an in-memory SQLite database for testing.

        Mirrors the Rust ``open_in_memory``.
        """
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        store = GraphStore.__new__(GraphStore)
        store._conn = conn
        store.db_path = Path(":memory:")
        store._init_schema()
        store._migrate_v1()
        return store

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> GraphStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _resolve_qname(self, qname: str) -> int | None:
        """Resolve a qualified name to a node_id."""
        row = self._conn.execute(
            "SELECT node_id FROM nodes WHERE qualified_name = ?", (qname,)
        ).fetchone()
        return row["node_id"] if row else None


def _now_iso() -> str:
    """Get current time as ISO string."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _now_unix() -> int:
    """Get current time as Unix timestamp."""
    from datetime import datetime, timezone
    return int(datetime.now(timezone.utc).timestamp())


def _confidence_class_name(confidence: Confidence) -> str:
    """Get the confidence class string for an Edge."""
    if hasattr(confidence, 'value'):
        return confidence.value
    return str(confidence)


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
