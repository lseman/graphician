"""File watcher for incremental graph updates.

Watches a project directory and triggers incremental updates when source
files change. Uses OS-level file events (via ``watchfiles`` when available)
with a short debounce, falling back to polling every `interval` seconds
when no watcher is available.

Integrates with git hooks and the rebuild lock:
- Post-commit / post-merge / post-checkout hooks queue changed files
- The lock holder drains the pending queue and applies all changes in one pass
- The graph stores the git HEAD commit hash for provenance
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import sys
import time
from pathlib import Path

from ...extraction.languages import LanguageRegistry
from ...extraction.pipeline import ExtractionPipeline
from ...persistence.store import GraphStore
from .rebuild import (
    apply_resource_limits,
    drain_pending_changes,
    get_git_head,
    queue_pending_changes,
    rebuild_lock,
    write_build_config,
)

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

# Git-tracked extensions — only consider these when using git diff.
GIT_DIFF_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".rs", ".ts", ".tsx", ".js", ".jsx", ".java", ".c", ".cpp",
    ".h", ".hpp", ".hh", ".hxx", ".go", ".rb", ".kt", ".swift", ".scala",
    ".cs", ".php", ".md", ".html", ".svg", ".toml", ".json",
    ".yaml", ".yml", ".xml", ".cfg", ".ini", ".conf",
})


# ── Public entry point ────────────────────────────────────────────────


def cmd_watch(db_path: str, path: str, interval: int = POLL_INTERVAL) -> None:
    """Watch a path and incrementally update the graph when files change.

    Args:
        db_path: Path to the Graphician SQLite database.
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

    # Persist build config (extensions / ignored dirs) so hooks can
    # re-read the same set of watched patterns.
    out_dir = root / ".graphician"
    out_dir.mkdir(exist_ok=True)
    write_build_config(
        out_dir,
        extensions={e.lstrip(".") for e in WATCHED_EXTENSIONS},
        ignored_dirs=IGNORED_DIRS,
    )

    # Stamp git HEAD for provenance.
    _stamp_git_head(store, root)

    print(f"watching {root} for graph updates (debounce {WATCH_DEBOUNCE}s)")

    # Initial update to catch up on anything changed while not watching.
    try:
        _run_update(store, pipeline, root, out_dir)
    except Exception as e:  # noqa: BLE001 -- initial update must not kill the watcher
        logger.warning("initial update failed for %s: %s", root, e)

    # Try OS-level file watching first.
    success = _watch_event_driven(store, pipeline, root, out_dir)
    if not success:
        print(
            f"watching {root} for graph updates every {interval}s (polling)",
            file=sys.stderr,
        )
        _watch_polling(store, pipeline, root, out_dir, interval)


# ── Core update logic ────────────────────────────────────────────────


def _run_update(
    store: GraphStore,
    pipeline: ExtractionPipeline,
    root: Path,
    out_dir: Path,
) -> bool:
    """Run an incremental update.

    Returns True if a rebuild actually ran, False if nothing changed.
    """
    # 1. Drain pending changes from hook processes.
    pending = drain_pending_changes(out_dir)
    if pending:
        logger.info("drained %d pending change(s) from hook queue", len(pending))

    # 2. Acquire the rebuild lock. If another rebuild is running,
    # queue our change list and return.
    with rebuild_lock(out_dir) as acquired:
        if not acquired:
            _queue_file_changes(root, out_dir)
            return False

    # 3. Collect changed/deleted files.
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

    # From file hash comparison
    for path_str, new_hash in current_hashes.items():
        old_hash = stored_hashes.get(path_str)
        if old_hash != new_hash:
            changed.append(path_str)
    for path_str in stored_hashes:
        if path_str not in current_hashes:
            deleted.append(path_str)

    # Merge with pending queue paths (they may not have hash changes but
    # were explicitly reported by a hook).
    all_changed: list[str] = list(set(changed))
    for p in pending:
        rel_str = str(p.relative_to(root)) if p.is_relative_to(root) else os.fspath(p)
        # Only add if it looks like a source file we care about.
        if _is_relevant_source(rel_str) and rel_str not in all_changed:
            all_changed.append(rel_str)

    if not all_changed and not deleted:
        return False

    # 4. Run the update.
    apply_resource_limits()
    graph = pipeline.update(root, existing, all_changed, deleted)
    store.save_graph(graph, pipeline._file_hashes or current_hashes)

    # Stamp new HEAD.
    _stamp_git_head(store, root)

    logger.info(
        "updated: %d nodes, %d edges (changed=%d, deleted=%d)",
        graph.node_count(),
        graph.edge_count(),
        len(all_changed),
        len(deleted),
    )
    return True


def _queue_file_changes(root: Path, out_dir: Path) -> None:
    """Queue all source files for the rebuild lock holder to process."""
    with contextlib.suppress(Exception):
        pipeline = ExtractionPipeline(LanguageRegistry())
        files = pipeline.discover_files(root)
        queue_pending_changes(out_dir, files)


def _stamp_git_head(store: GraphStore, root: Path) -> None:
    """Store git HEAD commit hash in the graph metadata."""
    head = get_git_head(root)
    if head:
        with contextlib.suppress(Exception):
            store.set_metadata("git_head", head)


# ── OS-level file watching ───────────────────────────────────────────


def _watch_event_driven(
    store: GraphStore,
    pipeline: ExtractionPipeline,
    root: Path,
    out_dir: Path,
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

        for _change_type, file_path in watchfiles.awatch(
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
                _run_update(store, pipeline, root, out_dir)
            except Exception as e:  # noqa: BLE001 -- one failed update must not kill the watcher
                logger.warning("update failed for %s: %s", file_path, e)

    except ImportError:
        # watchfiles not available
        return False
    except KeyboardInterrupt:
        raise
    except Exception as e:  # noqa: BLE001 -- OS watcher backend raises implementation-specific errors
        logger.warning("OS watcher failed (%s); falling back to polling", e)
        return False

    return True


# ── Polling fallback ─────────────────────────────────────────────────


def _watch_polling(
    store: GraphStore,
    pipeline: ExtractionPipeline,
    root: Path,
    out_dir: Path,
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

            # Detect changed + deleted files
            changed = False
            for rel_str, new in current_hashes.items():
                if _is_relevant_source(rel_str):
                    old = last_hashes.get(rel_str)
                    if old != new:
                        changed = True
                        last_hashes[rel_str] = new

            for rel_str in list(last_hashes):
                if rel_str not in current_hashes:
                    changed = True
                    del last_hashes[rel_str]

            if changed:
                try:
                    _run_update(store, pipeline, root, out_dir)
                except Exception as e:  # noqa: BLE001 -- one failed update must not kill the watcher
                    logger.warning("update failed: %s", e)

    except KeyboardInterrupt:
        pass
    finally:
        store.close()


# ── Helpers ──────────────────────────────────────────────────────────


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
        # Allow dotfiles like .eslintrc, .prettierrc but skip .git, .idea
        if part.startswith(".") and part not in (".venv", ".git") and part in IGNORED_DIRS:
            return False

    return True


def _file_hash(file_path: str) -> str | None:
    """Compute SHA-256 hash of a file, returning None on error."""
    try:
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None
