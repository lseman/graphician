"""Default directory exclusions for file walking.

Provides ``default_ignored_name()`` which returns True for directory
names that should be skipped during source file discovery. This mirrors
the Rust reference implementation.
"""

from __future__ import annotations


def default_ignored_name(name: str) -> bool:
    """Return True if this directory name should be ignored during walking.

    Covers common build artifacts, version control, and dependency
    directories that should not be indexed as source code.

    Args:
        name: The directory name (not the full path).

    Returns:
        True if the directory should be skipped.
    """
    # Hidden directories (except single-dot which shouldn't occur as a part)
    if name.startswith(".") and len(name) > 1:
        return True

    # Common build and artifact directories
    return name in (
        "target", "node_modules", "__pycache__", ".venv", "venv", "env",
        ".env", "build", "dist", "out", "bin", "obj", ".next", ".nuxt",
        ".svelte-kit", ".cache", ".tox", ".mypy_cache", ".pytest_cache",
        ".ruff_cache", ".eggs", ".egg-info", ".git", ".hg", ".svn",
    )
