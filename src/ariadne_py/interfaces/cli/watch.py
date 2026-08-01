"""File watcher for incremental graph updates.

Watches a project directory and triggers incremental updates when source
files change. Uses OS-level file events (via ``watchfiles`` when available)
with a short debounce, falling back to polling every `interval` seconds
when no watcher is available.
"""

from __future__ import annotations

import hashlib
import logging
import sys
import time
from pathlib import Path
from typing import Any

from ...persistence.store import GraphStore
from ...extraction.languages import LanguageRegistry
from ...extraction.pipeline import ExtractionPipeline

logger = logging.getLogger(__name__)

# Debounce window in seconds — editor save bursts and branch switches
# collapse into a single rebuild.
WATCH_DEBOUNCE: float = 0.5

# Poll interval in seconds (fallback when OS watcher unavailable).
POLL_INTERVAL: int = 5

# Source file extensions to watch.
WATCHED_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".rs", ".ts", ".tsx", ".js", ".jsx", ".java", ".c", ".cpp",
    ".h", ".hpp", ".hh", ".hxx", ".go", ".rb", ".kt", ".swift", ".scala",
    ".cs", ".php", ".md", ".html", ".svg", ".toml", ".json",
})

# Directories to ignore.
IGNORED_DIRS: frozenset[str] = frozenset({
    ".git", "__pycache__", "node_modules", "target", "build", ".venv",
    "venv", ".tox", ".mypy_cache", ".pytest_cache", ".idea", ".vscode",
    "dist", "out", ".next", "coverage",
})


def _is_relevant_source(file_path: str) -> bool:
    """Return True if this file is a relevant source file to watch."""
    path = Path(file_path)

    # Check extension
    if path.suffix.lower() not in WATCHED_EXTENSIONS:
        return False

    # Check that no path component is in ignored dirs
    for part in path.parts:
        if part in IGNORED_DIRS:
            return False

    # Skip hidden files/directories (except dotfiles that are source)
    for part in path.parts[1:]:  # skip root
        if part.startswith(".") and part not in (".venv", ".git"):
            # Allow dotfiles like .eslintrc, .prettierrc but skip .git, .idea
            if part in IGNORED_DIRS:
                return False

    return True


def _file_hash(file_path: str) -> str | None:
    """Compute SHA-256 hash of a file, returning None on error."""
    try:
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def cmd_watch(db_path: str, path: str, interval: int = POLL_INTERVAL) -> None:
    """Watch a path and incrementally update the graph when files change.

    Args:
        db_path: Path to the Ariadne SQLite database.
        path: Project root directory to watch.
        interval: Poll interval in seconds (fallback mode).
    """
    root = Path(path).resolve()
    if not root.exists():
        print(f"Error: {root} does not exist", file=sys.stderr)
        sys.exit(1)

    store = GraphStore(db_path)
    registry = LanguageRegistry()
    pipeline = ExtractionPipeline(registry)

    print(f"watching {root} for Ariadne graph updates (debounce {WATCH_DEBOUNCE}s)")

    # Initial update to catch up on anything changed while not watching.
    try:
        _run_update(store, pipeline, root)
    except Exception as e:
        logger.warning("initial update failed for %s: %s", root, e)

    # Try OS-level file watching first.
    success = _watch_event_driven(store, pipeline, root)
    if not success:
        print(
            f"watching {root} for Ariadne graph updates every {interval}s (polling)",
            file=sys.stderr,
        )
        _watch_polling(store, pipeline, root, interval)


def _run_update(
    store: GraphStore,
    pipeline: ExtractionPipeline,
    root: Path,
) -> None:
    """Run an incremental update and print stats."""
    existing = store.load_graph()
    files = pipeline.discover_files(root)
    current_hashes: dict[str, str] = {}
    for f in files:
        rel = f.relative_to(root)
        try:
            content = f.read_bytes()
            current_hashes[rel.as_posix()] = hashlib.sha256(content).hexdigest()
        except OSError:
            pass

    changed: list[str] = []
    deleted: list[str] = []
    stored_hashes = store.get_file_hashes()
    for path_str, new_hash in current_hashes.items():
        old_hash = stored_hashes.get(path_str)
        if old_hash != new_hash:
            changed.append(path_str)
    for path_str in stored_hashes:
        if path_str not in current_hashes:
            deleted.append(path_str)

    if not changed and not deleted:
        print("no changes detected", file=sys.stderr)
        return

    graph = pipeline.update(root, existing, changed, deleted)
    store.save_graph(graph, pipeline._file_hashes or current_hashes)
    print(
        f"updated: {graph.node_count()} nodes, {graph.edge_count()} edges",
        file=sys.stderr,
    )


def _watch_event_driven(
    store: "GraphStore",
    pipeline: "ExtractionPipeline",
    root: Path,
) -> bool:
    """Try OS-level file watching. Returns True if successful."""
    try:
        import watchfiles

        # Build initial file hash map for changed detection
        files = pipeline.discover_files(root)
        current_hashes: dict[str, str] = {}
        for f in files:
            rel = f.relative_to(root)
            try:
                content = f.read_bytes()
                current_hashes[str(rel)] = hashlib.sha256(content).hexdigest()
            except OSError:
                pass

        print(
            f"watching {len(current_hashes)} files via OS events (debounce {WATCH_DEBOUNCE}s)",
            file=sys.stderr,
        )

        for change_type, file_path in watchfiles.awatch(
            str(root),
            watch_filter=watchfiles.filters.PythonFilter(),
            stop_event=None,
            debounce=WATCH_DEBOUNCE,
            max_events=100,
        ):
            rel = Path(file_path).relative_to(root)
            rel_str = str(rel)

            if not _is_relevant_source(rel_str):
                continue

            # Check if file actually changed
            new_hash = _file_hash(file_path)
            old_hash = current_hashes.get(rel_str)
            if new_hash == old_hash:
                continue
            current_hashes[rel_str] = new_hash if new_hash else ""

            try:
                _run_update(store, pipeline, root)
            except Exception as e:
                logger.warning("update failed for %s: %s", file_path, e)

    except ImportError:
        # watchfiles not available
        return False
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.warning("OS watcher failed (%s); falling back to polling", e)
        return False

    return True


def _watch_polling(
    store: "GraphStore",
    pipeline: "ExtractionPipeline",
    root: Path,
    interval: int,
) -> None:
    """Poll-based file watching fallback."""
    files = pipeline.discover_files(root)
    last_hashes: dict[str, str] = {}
    for f in files:
        rel = f.relative_to(root)
        try:
            content = f.read_bytes()
            last_hashes[str(rel)] = hashlib.sha256(content).hexdigest()
        except OSError:
            pass

    # Main polling loop
    try:
        while True:
            time.sleep(interval)
            changed = False

            files = pipeline.discover_files(root)
            current_hashes: dict[str, str] = {}
            for f in files:
                rel = f.relative_to(root)
                rel_str = str(rel)
                try:
                    content = f.read_bytes()
                    current_hashes[rel_str] = hashlib.sha256(content).hexdigest()
                except OSError:
                    continue

                if _is_relevant_source(rel_str):
                    old = last_hashes.get(rel_str)
                    new = current_hashes[rel_str]
                    if old != new:
                        changed = True
                        last_hashes[rel_str] = new

            # Detect deleted files
            for rel_str in list(last_hashes):
                if rel_str not in current_hashes:
                    changed = True
                    del last_hashes[rel_str]

            if changed:
                try:
                    _run_update(store, pipeline, root)
                except Exception as e:
                    logger.warning("update failed: %s", e)

    except KeyboardInterrupt:
        pass
    finally:
        store.close()
