"""Tokenization and query vocabulary utilities."""

from __future__ import annotations

import re

# Extended stopwords: question/filler words dropped from query terms so content
# words drive candidate selection. Mirrors graphify's _QUERY_STOPWORDS approach.
SEARCH_STOPWORDS: tuple[str, ...] = (
    # Common question/filler words
    "and", "are", "for", "from", "has", "have", "how",
    "the", "what", "when", "where", "who", "why", "with",
    # Additional fillers
    "a", "an", "at", "but", "can", "did", "does",
    "if", "in", "is", "it", "may", "new", "not", "old",
    "on", "or", "out", "per", "see", "set", "via",
    "work", "works", "working",
    "was", "way", "will", "yet", "you",
)


def _search_tokens(text: str) -> list[str]:
    """Split text into word tokens, stripping punctuation and diacritics.

    ``_`` is a separator, exactly like ``-``. ``\\w`` counts underscore as a word
    character but not hyphen, so ``graph_first_guard`` stayed one token while the
    label ``graph-first-guard.py`` split into three -- and the query matched
    nothing. Both the query and the node label pass through here, so splitting
    on ``_`` keeps the two sides consistent and snake_case lookups still resolve
    (their tokens simply match the same way).
    """
    if not text:
        return []
    text = _strip_diacritics(str(text))
    return re.findall(r"[^\W_]+", text.lower())


def _strip_diacritics(text: str) -> str:
    """Strip combining diacritical marks via NFKD decomposition."""
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase symbol tokens (underscore is a word char)."""
    return re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text.lower())


def _extract_query_identifiers(query: str) -> list[str]:
    """Extract potential symbol names from a query string."""
    tokens = _search_tokens(query)
    # Filter stopwords and very short tokens
    return [t for t in tokens if t not in SEARCH_STOPWORDS and len(t) > 1]


def _normalize_identifier(name: str) -> str:
    """Normalize an identifier for fuzzy matching.

    Port of Rust ``normalize_identifier`` (fuzzy.rs:4-30).
    Splits on camelCase boundaries, acronym transitions, and
    digit/letter transitions. E.g.:
      "HTTPRequestParser" -> "http request parser"
      "CafeParser"        -> "cafe parser"
      "extractDir"        -> "extract dir"
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
