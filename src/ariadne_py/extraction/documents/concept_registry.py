"""Concept (prose + diagram) registry.

Maps file extensions to concept extractors. After AST extraction
fails to match a file, the walker checks this registry for
document/diagram formats.

Each entry is (extension, extractor_function). Extension is
lowercased, without leading dot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ...core.graph import Graph


# Type alias: extractor function
Extractor = Callable[[str, Graph], dict[str, Any]]

# Supported extensions
SUPPORTED_EXTENSIONS: set[str] = {"md", "markdown", "html", "htm", "svg"}


def get_by_path(path: Path) -> Extractor | None:
    """Look up a concept extractor by file path.

    Returns None if no document/diagram extractor matches.
    """
    ext = path.suffix.lstrip(".").lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return None

    # Lazy imports to avoid forward reference issues
    if ext in ("md", "markdown"):
        from .markdown import extract_file
        return extract_file
    elif ext in ("html", "htm"):
        from .html import extract_file
        return extract_file
    elif ext == "svg":
        from .svg import extract_svg
        return extract_svg

    return None  # unreachable, but type checker happy


def is_supported(path: Path) -> bool:
    """True when a concept extractor supports the given path."""
    return get_by_path(path) is not None


def extract_concept(path: str | Path, graph: Graph) -> dict[str, Any]:
    """Extract a concept file using the appropriate extractor.

    Auto-detects format from file extension.
    """
    path = Path(path)
    extractor = get_by_path(path)
    if extractor is None:
        return {"error": f"Unsupported concept format: {path.name}"}
    return extractor(str(path), graph)


def resolve_all_mentions(graph: Graph) -> int:
    """Resolve mentions across all concept extractors.

    Idempotent: running multiple times adds no duplicate edges.
    Currently delegates to markdown's resolver.

    Returns number of mention edges resolved.
    """
    from .markdown import resolve_mentions
    return resolve_mentions(graph)
