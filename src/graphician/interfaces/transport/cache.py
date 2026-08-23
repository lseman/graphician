"""Process-lifetime graph cache for long-lived servers (MCP server, serve).

Not used by the one-shot ``graphician tool`` CLI path — the cached path
is only activated when the server stays running across multiple queries.

Uses a file fingerprint (main DB + WAL sidecar mtime/size) to detect
when the database has changed and needs reloading.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from ...core.graph import Graph
from ...persistence.store import GraphStore

# ── Fingerprint ────────────────────────────────────────────────────────


class _DbFingerprint:
    """Fingerprint of the on-disk DB state a cached graph was loaded from."""

    __slots__ = ("main_mtime", "main_len", "wal_mtime", "wal_len")

    def __init__(
        self,
        main_mtime: float | None = None,
        main_len: int = 0,
        wal_mtime: float | None = None,
        wal_len: int = 0,
    ) -> None:
        self.main_mtime = main_mtime
        self.main_len = main_len
        self.wal_mtime = wal_mtime
        self.wal_len = wal_len

    @classmethod
    def capture(cls, db_path: str | Path) -> "_DbFingerprint":
        db_path = Path(db_path)
        try:
            main_stat = db_path.stat()
            main_mtime = main_stat.st_mtime
            main_len = main_stat.st_size
        except OSError:
            main_mtime = None
            main_len = 0

        wal_path = db_path.parent / f"{db_path.name}-wal"
        try:
            wal_stat = wal_path.stat()
            wal_mtime = wal_stat.st_mtime
            wal_len = wal_stat.st_size
        except OSError:
            wal_mtime = None
            wal_len = 0

        return cls(main_mtime, main_len, wal_mtime, wal_len)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _DbFingerprint):
            return False
        return (
            self.main_mtime == other.main_mtime
            and self.main_len == other.main_len
            and self.wal_mtime == other.wal_mtime
            and self.wal_len == other.wal_len
        )


# ── Cache slot ─────────────────────────────────────────────────────────


class _CachedGraph:
    __slots__ = ("db_path", "fingerprint", "graph", "loaded_at")

    def __init__(self, db_path: str, fingerprint: _DbFingerprint, graph: Graph) -> None:
        self.db_path = db_path
        self.fingerprint = fingerprint
        self.graph = graph
        self.loaded_at = time.monotonic()


# Global process-lifetime cache (one slot per db_path via dict)
_cache: dict[str, _CachedGraph] = {}


def load_cached(db_path: str, store: GraphStore | None = None) -> Graph:
    """Load the graph, reusing the process-lifetime cache when the DB
    fingerprint hasn't changed since it was last cached.

    Opens a fresh GraphStore every call (cheap: just a SQLite connection
    open) but skips the expensive full table scan when the fingerprint
    matches.

    Args:
        db_path: Path to the graph database.
        store: Optional pre-opened GraphStore. If None, a new one is created.

    Returns:
        The loaded (or cached) Graph.
    """
    fp = _DbFingerprint.capture(db_path)
    cached = _cache.get(db_path)

    if cached is not None and cached.fingerprint == fp:
        return cached.graph

    # Cache miss or stale — load fresh
    if store is None:
        store = GraphStore(db_path)

    graph = store.load_graph()
    _cache[db_path] = _CachedGraph(db_path, fp, graph)
    return graph


def clear_cache() -> None:
    """Clear the entire process-lifetime cache. Called on graph rebuild."""
    _cache.clear()


def cache_stats() -> dict[str, Any]:
    """Return cache statistics for diagnostics."""
    now = time.monotonic()
    entries = []
    for db_path, entry in _cache.items():
        age = now - entry.loaded_at
        entries.append({
            "db_path": db_path,
            "age_seconds": round(age, 1),
        })
    return {
        "cache_size": len(_cache),
        "entries": entries,
    }
