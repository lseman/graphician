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
    """Normalize an identifier for fuzzy matching."""
    # Lowercase, remove common separators
    return name.lower().replace('-', '').replace('_', '').replace(' ', '')
