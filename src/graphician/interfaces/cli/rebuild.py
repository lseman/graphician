"""Rebuild infrastructure: locking, pending queue, config, and git provenance.

Provides:
- Advisory flock-based rebuild lock (prevents concurrent rebuilds)
- Pending changes queue (for hook processes that can't acquire the lock)
- Build config persistence (persisted exclude/ignore patterns)
- Git HEAD stamping (provenance tracking in the graph store)
- Resource limits (nice + memory cap)
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# File names stored beside the graph database
_PENDING_FILENAME = ".pending_changes"
_LOCK_FILENAME = ".rebuild.lock"
_CONFIG_FILENAME = ".graphician_build.json"


def _graph_out_dir(db_path: Path) -> Path:
    """Return the directory containing the graph database file."""
    return db_path.parent


def queue_pending_changes(out_dir: Path, changed_paths: list[Path]) -> None:
    """Append changed paths to the pending changes file.

    Used by post-commit hook processes that cannot acquire the rebuild lock.
    The lock holder drains this file and merges the changes into the rebuild.

    Opened in append mode so concurrent writers don't clobber each other.
    """
    if not changed_paths:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = out_dir / _PENDING_FILENAME
    with open(pending, "a", encoding="utf-8") as fh:
        for p in changed_paths:
            fh.write(f"{os.fspath(p)}\n")


def drain_pending_changes(out_dir: Path) -> list[Path]:
    """Read and remove the pending changes file, returning deduplicated paths.

    Returns an empty list if the file does not exist.
    """
    pending = out_dir / _PENDING_FILENAME
    if not pending.exists():
        return []
    try:
        raw = pending.read_text(encoding="utf-8")
    except OSError:
        return []
    with contextlib.suppress(FileNotFoundError):
        pending.unlink()
    seen: set[str] = set()
    out: list[Path] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(Path(s))
    return out


@contextlib.contextmanager
def rebuild_lock(out_dir: Path, *, blocking: bool = False):
    """Advisory flock-based lock around a rebuild.

    Yields True if acquired, False if another rebuild is already running
    and blocking is False. Uses fcntl.flock so the lock is released
    automatically if the process is killed.

    Falls back to a no-op yield(True) on platforms without fcntl.
    """
    try:
        import fcntl
    except ImportError:
        yield True
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / _LOCK_FILENAME
    with open(lock_path, "a+", encoding="utf-8") as fh:
        acquired = False
        try:
            flags = fcntl.LOCK_EX if blocking else (fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                fcntl.flock(fh.fileno(), flags)
            except BlockingIOError:
                yield False
                return
            acquired = True
            with contextlib.suppress(OSError):
                fh.seek(0)
                fh.truncate()
                fh.write(f"{os.getpid()}\n")
                fh.flush()
            yield True
        finally:
            if acquired:
                with contextlib.suppress(OSError):
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                lock_path.unlink()


def write_build_config(
    out_dir: Path,
    *,
    extensions: Iterable[str] | None = None,
    ignored_dirs: Iterable[str] | None = None,
) -> None:
    """Persist build options under out_dir.

    Best effort and non-clobbering: omitted options retain existing values.
    """
    if extensions is None and ignored_dirs is None:
        return
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / _CONFIG_FILENAME
        try:
            config = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except (OSError, json.JSONDecodeError):
            config = {}
        if not isinstance(config, dict):
            config = {}
        if extensions is not None:
            config["extensions"] = sorted(extensions)
        if ignored_dirs is not None:
            config["ignored_dirs"] = sorted(ignored_dirs)
        path.write_text(json.dumps(config), encoding="utf-8")
    except OSError:
        pass


def read_build_config(out_dir: Path) -> dict[str, Any]:
    """Return persisted build config, or empty dict."""
    try:
        path = out_dir / _CONFIG_FILENAME
        if path.is_file():
            cfg = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(cfg, dict):
                return cfg
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def merge_changed_paths(*sources: list[Path] | None) -> list[Path]:
    """Concatenate path lists, preserving order, dropping duplicates."""
    seen: set[str] = set()
    out: list[Path] = []
    for src in sources:
        if not src:
            continue
        for p in src:
            key = os.fspath(p)
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
    return out


def get_git_head(cwd: Path | str | None = None) -> str | None:
    """Return current git HEAD commit hash, or None outside a repo."""
    import subprocess

    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            cwd=str(cwd) if cwd is not None else None,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def apply_resource_limits() -> None:
    """Best-effort nice + memory cap via env vars.

    Set GRAPHICIAN_REBUILD_MEMORY_LIMIT_MB to cap RSS (e.g. 1024).
    """
    with contextlib.suppress(OSError, AttributeError):
        os.nice(10)
    mb_str = os.environ.get("GRAPHICIAN_REBUILD_MEMORY_LIMIT_MB", "").strip()
    if not mb_str:
        return
    try:
        limit = int(mb_str) * 1024 * 1024
    except ValueError:
        return
    try:
        import resource
        which = resource.RLIMIT_DATA if sys.platform == "darwin" else resource.RLIMIT_AS
        _soft, hard = resource.getrlimit(which)
        new_hard = hard if hard != resource.RLIM_INFINITY and hard < limit else limit
        resource.setrlimit(which, (limit, new_hard))
    except (ImportError, ValueError, OSError):
        pass
