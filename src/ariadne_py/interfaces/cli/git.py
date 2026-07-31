"""Repository snapshot and graph-freshness helpers."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

from ...extraction.languages import LanguageRegistry
from ...extraction.pipeline import ExtractionPipeline
from ...persistence.store import GraphStore


def collect_file_hashes(root: Path) -> dict[str, str]:
    """Hash every source accepted by the extraction pipeline."""
    root = root.resolve()
    pipeline = ExtractionPipeline(LanguageRegistry())
    hashes: dict[str, str] = {}
    for path in pipeline.discover_files(root):
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
        hashes[path.relative_to(root).as_posix()] = digest
    return dict(sorted(hashes.items()))


def git_commit_hash(root: Path, revision: str = "HEAD") -> str | None:
    """Resolve a Git revision without treating non-Git roots as errors."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", revision],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def graph_freshness(store: GraphStore) -> dict[str, Any]:
    """Compare the persisted file snapshot with the indexed repository."""
    root_value = store.get_metadata("repository_root")
    if root_value is None:
        return {
            "state": "unknown",
            "reason": "repository metadata missing; rebuild or update the graph",
        }
    root = Path(root_value)
    if not root.is_dir():
        return {
            "state": "stale",
            "repository_root": root_value,
            "reason": "indexed repository root no longer exists",
        }
    previous = store.get_file_hashes()
    current = collect_file_hashes(root)
    added = sorted(current.keys() - previous.keys())
    deleted = sorted(previous.keys() - current.keys())
    modified = sorted(path for path in current.keys() & previous.keys() if current[path] != previous[path])
    sample_limit = 10
    return {
        "state": "fresh" if not (added or modified or deleted) else "dirty",
        "repository_root": root_value,
        "indexed_commit": store.get_metadata("indexed_commit"),
        "current_commit": git_commit_hash(root),
        "changes": {
            "added": len(added),
            "modified": len(modified),
            "deleted": len(deleted),
            "sample_added": added[:sample_limit],
            "sample_modified": modified[:sample_limit],
            "sample_deleted": deleted[:sample_limit],
        },
    }

