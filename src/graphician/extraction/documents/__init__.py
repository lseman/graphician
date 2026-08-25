"""Document extraction: HTML and Markdown parsers.

Parses HTML and Markdown files into graph nodes (Document, Section, Concept)
and edges (Mentions) linking documentation to code symbols.
"""

from .document_utils import (
    mention,
    normalize_for_match,
    resolve_symbol,
    slugify,
    strip_file_suffix,
    tokenize_code,
)
from .html import extract_file as extract_html
from .markdown import extract_file as extract_markdown
from .markdown import resolve_mentions

__all__ = [
    "extract_html",
    "extract_markdown",
    "mention",
    "normalize_for_match",
    "resolve_mentions",
    "resolve_symbol",
    "slugify",
    "strip_file_suffix",
    "tokenize_code",
]
