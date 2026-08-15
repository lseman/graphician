"""FTS5 index wrapper for semantic search.

Provides full-text search over the SQLite FTS5 virtual table.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FTSResult:
    """A single FTS search result."""
    node_id: int
    qualified_name: str
    kind: str
    name: str
    score: float = 0.0
    snippet: str = ""


class FTSIndex:
    """Wrapper around SQLite FTS5 virtual table.

    Provides search, snippet generation, and rebuild functionality.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def search(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0,
    ) -> list[FTSResult]:
        """Search the FTS5 index.

        Uses BM25 scoring (built into FTS5).
        """
        # Sanitize query - FTS5 supports boolean operators
        query = query.strip()
        if not query:
            return []

        rows = self.conn.execute(
            """SELECT node_id, name, qualified_name, kind,
                      snippet(nodes_fts, 2, '[', ']', '...', 8) as snippet,
                      rank
               FROM nodes_fts
               WHERE nodes_fts MATCH ?
               ORDER BY rank
               LIMIT ? OFFSET ?""",
            (query, limit, offset),
        ).fetchall()

        return [
            FTSResult(
                node_id=row["node_id"],
                qualified_name=row["qualified_name"],
                kind=row["kind"],
                name=row["name"],
                score=abs(row["rank"]) if row["rank"] else 0.0,
                snippet=row["snippet"] or "",
            )
            for row in rows
        ]

    def count(self, query: str) -> int:
        """Count matching results."""
        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM nodes_fts WHERE nodes_fts MATCH ?",
            (query,),
        ).fetchone()
        return row["cnt"] if row else 0

    def add_node(self, node_id: int, name: str, qualified_name: str, source_text: str | None = None) -> None:
        """Add a node to the FTS index."""
        self.conn.execute(
            "INSERT INTO nodes_fts (node_id, name, qualified_name, source_text) VALUES (?, ?, ?, ?)",
            (node_id, name, qualified_name, source_text or ""),
        )

    def update_node(self, node_id: int, name: str, qualified_name: str, source_text: str | None = None) -> None:
        """Update a node in the FTS index."""
        self.conn.execute(
            "DELETE FROM nodes_fts WHERE node_id=?",
            (node_id,),
        )
        self.add_node(node_id, name, qualified_name, source_text)

    def remove_node(self, node_id: int) -> None:
        """Remove a node from the FTS index."""
        self.conn.execute("DELETE FROM nodes_fts WHERE node_id=?", (node_id,))

    def rebuild(self) -> None:
        """Rebuild the FTS index from the nodes table."""
        self.conn.execute("DELETE FROM nodes_fts")
        self.conn.execute(
            "INSERT INTO nodes_fts (node_id, kind, name, qualified_name, source_text) "
            "SELECT node_id, kind, name, qualified_name, source_text FROM nodes"
        )
        logger.info("FTS index rebuilt")

    def optimize(self) -> None:
        """Optimize the FTS index."""
        self.conn.execute("INSERT INTO nodes_fts(nodes_fts) VALUES('optimize')")

    def search_safe(self, query: str, limit: int = 20, offset: int = 0) -> list[FTSResult]:
        """Search the FTS5 index with a safe, parameterized query.

        Uses ``build_fts5_query`` internally to sanitize the raw input.
        """
        safe_query = build_fts5_query(query)
        if not safe_query:
            return []
        return self.search(safe_query, limit=limit, offset=offset)


def build_fts5_query(raw: str) -> str:
    """Build a safe FTS5 MATCH expression from a raw user query.

    Each whitespace/punctuation-separated token becomes a prefix term
    (``token*``).  Special FTS5 syntax characters are stripped to
    prevent query parse errors.

    Mirrors the Rust ``build_fts5_query`` from ``persistence/sql.rs``.
    """
    import re

    # Split on non-alphanumeric/non-underscore boundaries
    raw_tokens = re.split(r'[^a-zA-Z0-9_]+', raw)
    tokens: list[str] = []
    for raw_token in raw_tokens:
        # Strip special FTS5 characters and collect non-empty tokens
        clean = re.sub(r'[^a-zA-Z0-9_]', '', raw_token)
        if clean:
            # FTS5 treats these bare words as operators. Quote them so raw
            # user input remains a literal, valid prefix query.
            if clean.upper() in {"AND", "OR", "NOT", "NEAR"}:
                tokens.append(f'"{clean}"*')
            else:
                tokens.append(f"{clean}*")
    return " ".join(tokens)
