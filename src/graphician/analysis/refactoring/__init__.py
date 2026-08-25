"""Refactoring operations: rename preview and dead code detection.

Safe, preview-only operations that analyze the graph to
determine the impact of refactoring changes. No source files
are written to — only edit suggestions are produced.
"""

from __future__ import annotations

from .engine import find_dead_code, rename_preview
from .types import RenameEdit, RenamePreview, RenameStats

__all__ = [
    "RenameEdit",
    "RenamePreview",
    "RenameStats",
    "find_dead_code",
    "rename_preview",
]
