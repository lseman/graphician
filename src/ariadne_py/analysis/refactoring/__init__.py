"""Refactoring operations: rename preview and dead code detection.

Safe, preview-only operations that analyze the graph to
determine the impact of refactoring changes. No source files
are written to — only edit suggestions are produced.
"""

from __future__ import annotations

from .types import RenameEdit, RenamePreview, RenameStats
from .engine import rename_preview, find_dead_code

__all__ = [
    "RenameEdit",
    "RenamePreview",
    "RenameStats",
    "rename_preview",
    "find_dead_code",
]
