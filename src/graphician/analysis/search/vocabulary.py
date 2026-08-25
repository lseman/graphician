"""Tokenization and query vocabulary utilities."""

from __future__ import annotations

import re

SEARCH_STOPWORDS: tuple[str, ...] = (
    "and", "are", "for", "from", "has", "have", "how",
    "the", "what", "when", "where", "who", "why", "with",
)


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase symbol tokens."""
    return re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text.lower())


def _extract_query_identifiers(query: str) -> list[str]:
    """Extract potential symbol names from a query string."""
    tokens = _tokenize(query)
    # Filter stopwords and very short tokens
    return [t for t in tokens if t not in SEARCH_STOPWORDS and len(t) > 1]


def _normalize_identifier(name: str) -> str:
    """Normalize an identifier for fuzzy matching.

    Port of Rust ``normalize_identifier`` (fuzzy.rs:4-30).
    Splits on camelCase boundaries, acronym transitions, and
    digit/letter transitions. E.g.:
      "HTTPRequestParser" → "http request parser"
      "CafeParser"        → "cafe parser"
      "extractDir"        → "extract dir"
    """
    out: list[str] = []
    prev: str | None = None
    chars = list(name)
    for i, c in enumerate(chars):
        next_c = chars[i + 1] if i + 1 < len(chars) else None
        if c.isalnum():
            if prev is not None:
                camel_boundary = prev.islower() and c.isupper()
                acronym_boundary = (
                    prev.isupper()
                    and c.isupper()
                    and next_c is not None
                    and next_c.islower()
                )
                digit_boundary = prev.isalpha() != c.isalpha()
                if camel_boundary or acronym_boundary or digit_boundary:
                    out.append(" ")
            out.append(c.lower())
            prev = c
        else:
            out.append(" ")
            prev = None
    return " ".join("".join(out).split())
