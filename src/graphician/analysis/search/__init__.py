"""Hybrid search: FTS5 + fuzzy + topology signals + fusion."""

from __future__ import annotations

from .types import SearchIntent, SearchHit
from .vocabulary import _tokenize, _extract_query_identifiers, _normalize_identifier, SEARCH_STOPWORDS
from .fuzzy import _levenshtein, _fuzzy_score
from .fusion import (
    fts_ranked_search,
    apply_source_saturation,
    reciprocal_rank_boost,
)
from .search import (
    ranked_search,
    search_by_name,
    task_aware_search,
    hybrid_search,
    token_overlap_search,
)
from .utils import _graph_summary

__all__ = [
    "SearchIntent",
    "SearchHit",
    "ranked_search",
    "search_by_name",
    "task_aware_search",
    "hybrid_search",
    "fts_ranked_search",
    "token_overlap_search",
    "apply_source_saturation",
    "reciprocal_rank_boost",
    "_tokenize",
    "_extract_query_identifiers",
    "_normalize_identifier",
    "SEARCH_STOPWORDS",
    "_levenshtein",
    "_fuzzy_score",
    "_graph_summary",
]
