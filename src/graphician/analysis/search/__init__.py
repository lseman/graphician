"""Hybrid search: trigram index + tiered scoring + fuzzy + topology signals."""

from __future__ import annotations

from .fusion import (
    apply_source_saturation,
    fts_ranked_search,
    reciprocal_rank_boost,
)
from .fuzzy import _fuzzy_score, _levenshtein
from .search import (
    hybrid_search,
    ranked_search,
    search_by_name,
    task_aware_search,
    token_overlap_search,
)
from .trigram_index import (
    _compute_idf,
    _get_trigram_index,
    _score_tiered,
    _trigram_candidates,
)
from .types import SearchHit, SearchIntent
from .utils import _graph_summary
from .vocabulary import (
    SEARCH_STOPWORDS,
    _extract_query_identifiers,
    _normalize_identifier,
    _search_tokens,
    _tokenize,
)

__all__ = [
    "SEARCH_STOPWORDS",
    "SearchHit",
    "SearchIntent",
    "_compute_idf",
    "_extract_query_identifiers",
    "_fuzzy_score",
    "_get_trigram_index",
    "_graph_summary",
    "_levenshtein",
    "_normalize_identifier",
    "_score_tiered",
    "_search_tokens",
    "_tokenize",
    "_trigram_candidates",
    "apply_source_saturation",
    "fts_ranked_search",
    "hybrid_search",
    "ranked_search",
    "reciprocal_rank_boost",
    "search_by_name",
    "task_aware_search",
    "token_overlap_search",
]
