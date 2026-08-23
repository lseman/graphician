"""Suppress list for call placeholder filtering.

Determines whether a call:: placeholder node should be suppressed
from the final graph output based on configurable suppression rules.
"""

from __future__ import annotations

import re
from typing import Any

# Default patterns that should be suppressed from call placeholders
DEFAULT_SUPPRESS_PATTERNS: list[str] = [
    r"^call::std::",
    r"^call::core::",
    r"^call::builtins::",
    r"^call::js::",
    r"^call::node::",
    r"^call::java\.lang\.",
    r"^call::rust::",
]


class SuppressList:
    """Configurable suppression list for call placeholder filtering."""

    def __init__(
        self,
        patterns: list[str] | None = None,
        qname_prefixes: list[str] | None = None,
        min_source_length: int = 0,
    ) -> None:
        self._compiled: list[re.Pattern[str]] = []
        for p in (patterns or DEFAULT_SUPPRESS_PATTERNS):
            try:
                self._compiled.append(re.compile(p))
            except re.error:
                pass  # skip invalid patterns
        self._prefixes = qname_prefixes or []
        self._min_source_length = min_source_length

    def should_suppress(self, qname: str, source_uri: str | None = None) -> bool:
        """Return True if this call placeholder should be suppressed.

        A placeholder is suppressed when:
        - Its qname matches any compiled regex pattern, OR
        - Its qname starts with any configured prefix, OR
        - Its source_uri is shorter than min_source_length (if set), OR
        - Its source_uri is None when min_source_length > 0.
        """
        for pattern in self._compiled:
            if pattern.search(qname):
                return True
        for prefix in self._prefixes:
            if qname.startswith(prefix):
                return True
        if self._min_source_length > 0:
            if source_uri is None or len(source_uri) < self._min_source_length:
                return True
        return False

    def add_pattern(self, pattern: str) -> None:
        """Add a regex pattern to the suppression list."""
        try:
            self._compiled.append(re.compile(pattern))
        except re.error:
            pass

    @classmethod
    def default(cls) -> SuppressList:
        """Return a SuppressList with default patterns."""
        return cls()

    def to_dict(self) -> dict[str, Any]:
        """Serialize suppression list configuration."""
        return {
            "patterns": [p.pattern for p in self._compiled],
            "prefixes": list(self._prefixes),
            "min_source_length": self._min_source_length,
        }


def should_suppress_call_placeholder(
    qname: str,
    source_uri: str | None = None,
    patterns: list[str] | None = None,
) -> bool:
    """Convenience function: check if a call placeholder should be suppressed.

    Uses default suppression patterns.
    """
    return SuppressList(patterns=patterns).should_suppress(qname, source_uri)
