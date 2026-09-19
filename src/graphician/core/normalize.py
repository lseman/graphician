"""Identifier normalization for search indexing.

This module is intentionally in core/ so that both the extraction pipeline
(index-time normalization) and the search module (query-time normalization)
can import it without circular dependencies.

Port of Rust ``normalize_identifier`` (fuzzy.rs:4-30).
Splits on camelCase boundaries, acronym transitions, and
digit/letter transitions. E.g.:
  "HTTPRequestParser" → "http request parser"
  "CafeParser"        → "cafe parser"
  "extractDir"        → "extract dir"
"""

from __future__ import annotations

import functools
import re

# Compiled lookaround patterns — order matters: acronym boundary before
# lower→upper so that "FTSResult" splits as "fts result" (not "f tsresult").
#   (?<=[A-Z])(?=[A-Z][a-z])   → acronym→lower  (S|R in FTSResult)
#   (?<=[a-z])(?=[A-Z])        → lower→upper    (t|R in camelCase)
#   (?<=[a-zA-Z])(?=[0-9])     → letter→digit
#   (?<=[0-9])(?=[a-zA-Z])     → digit→letter
_BOUNDARY_RE = re.compile(
    r"(?<=[A-Z])(?=[A-Z][a-z])"
    r"|(?<=[a-z])(?=[A-Z])"
    r"|(?<=[a-zA-Z])(?=[0-9])"
    r"|(?<=[0-9])(?=[a-zA-Z])",
)

# Non-alnum → space (applied after boundary insertion)
_NON_ALNUM_RE = re.compile(r"[^a-zA-Z0-9]")


@functools.lru_cache(maxsize=2048)
def _normalize_identifier(name: str) -> str:
    """Normalize an identifier for fuzzy matching.

    Splits on camelCase boundaries, acronym transitions, and
    digit/letter transitions. Uses lookaround assertions and an LRU cache.
    """
    s = _BOUNDARY_RE.sub(" ", name)
    s = _NON_ALNUM_RE.sub(" ", s)
    return " ".join(s.lower().split())
