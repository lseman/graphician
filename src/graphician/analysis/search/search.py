"""Search functions: ranked_search, search_by_name, task_aware_search, fts_ranked_search.

Uses trigram-based candidate prefilter + tiered scoring for O(candidates) instead
of O(all_nodes) complexity. The trigram index is lazy-built and cached on the
graph object, so repeated searches are fast.

See trigram_index.py for the core engine.
"""

from __future__ import annotations

from typing import Any

from ...core.graph import Graph
from ...core.node import NodeKind
from .trigram_index import _score_tiered
from .types import SearchHit, SearchIntent
from .vocabulary import _normalize_identifier, _tokenize


def ranked_search(
    graph: Graph,
    query: str,
    limit: int = 20,
) -> list[SearchHit]:
    """In-memory ranked search with trigram prefilter + tiered scoring.

    Replaces the old Levenshtein/fuzzy-based approach. Uses the graphify-inspired
    trigram index to narrow candidates, then applies tiered matching (exact >
    prefix > substring) with IDF weighting.
    """
    scored = _score_tiered(graph, query, limit=limit)

    hits: list[SearchHit] = []
    for rank, (score, nid) in enumerate(scored, start=1):
        node = graph.node(nid)
        if node is None:
            continue
        reasons: list[str] = ["name_match"]
        # Add specific reason based on match quality
        qn = _normalize_identifier(query)
        if qn and qn in (
            node.normalized_qname or _normalize_identifier(node.qualified_name)
        ):
            reasons.append("qname_match")
        if node.name.lower() == qn:
            reasons = ["exact_match"]

        hits.append(SearchHit(
            id=nid,
            node=node,
            score=score,
            rank=rank,
            reasons=reasons,
        ))

    return hits


def search_by_name(
    graph: Graph,
    name: str,
    exact: bool = False,
) -> list[SearchHit]:
    """Exact or substring name lookup.

    For exact matching, uses the fast trigram index to find candidates.
    For substring, falls through to tiered scoring which handles it natively.
    """
    limit = 50  # Allow more candidates for substring matching

    if exact:
        # Use tiered scoring -- the exact tier will dominate for exact matches
        scored = _score_tiered(
            graph, name, limit=limit, fuzzy_on_candidates=False,
        )
        hits: list[SearchHit] = []
        for rank, (_score, nid) in enumerate(scored, start=1):
            node = graph.node(nid)
            if node is None:
                continue
            if node.name.lower() == name.lower():
                hits.append(SearchHit(
                    id=nid,
                    node=node,
                    score=1.0,
                    rank=rank,
                    reasons=["exact_name_match"],
                ))
        return hits

    # Substring lookup via tiered scoring
    scored = _score_tiered(
        graph, name, limit=limit, fuzzy_on_candidates=False,
    )
    hits: list[SearchHit] = []
    for rank, (score, nid) in enumerate(scored, start=1):
        node = graph.node(nid)
        if node is None:
            continue
        name_lower = name.lower()
        if name_lower in (node.name or "").lower():
            hits.append(SearchHit(
                id=nid,
                node=node,
                score=0.5 + min(score / 1000, 0.5),  # Normalize score
                rank=rank,
                reasons=["name_lookup"],
            ))
    return hits


def task_aware_search(
    graph: Graph,
    query: str,
    limit: int = 20,
    intent: SearchIntent | None = None,
) -> list[SearchHit]:
    """Intent-classified hybrid search.

    Uses trigram prefilter + tiered scoring with intent-based boosting.
    The core matching is fast (trigram candidates only), and intent boosts
    are applied post-scoring.
    """
    if intent is None:
        intent = SearchIntent.classify(query)

    # Core tiered scoring (trigram-based)
    scored = _score_tiered(graph, query, limit=limit)

    hits: list[SearchHit] = []
    for rank, (base_score, nid) in enumerate(scored, start=1):
        node = graph.node(nid)
        if node is None:
            continue

        score = base_score
        reasons: list[str] = ["name_match"]

        # Synthetic placeholders and local variables are useful as graph
        # plumbing but should not outrank source-backed definitions.
        if node.source_uri is None:
            score *= 0.35
        if node.kind is NodeKind.VARIABLE:
            score *= 0.5

        # Intent-based boosting
        if intent == SearchIntent.IMPACT:
            if node.kind in (NodeKind.FUNCTION, NodeKind.CLASS):
                score *= 1.3
                reasons.append("impact_boost")
        elif intent == SearchIntent.ARCHITECTURE and node.kind == NodeKind.MODULE:
            score *= 1.5
            reasons.append("architecture_boost")

        if score > 0.3:
            hits.append(SearchHit(
                id=nid,
                node=node,
                score=score,
                rank=rank,
                reasons=reasons,
            ))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:limit]


def hybrid_search(
    graph: Graph,
    query: str,
    intent: SearchIntent | str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Compatibility search API used by the CLI, MCP, and context packs.

    The typed search functions return ``SearchHit`` objects.  This public
    boundary intentionally returns JSON-ready data because all of its callers
    expose or further compose a structured response.
    """
    if isinstance(intent, str):
        try:
            intent = SearchIntent(intent.lower())
        except ValueError:
            intent = None
    resolved_intent = intent or SearchIntent.classify(query)
    hits = task_aware_search(graph, query, limit=limit, intent=resolved_intent)
    return {
        "query": query,
        "intent": resolved_intent.value,
        "results": [
            {
                "id": hit.id.value,
                "qualified_name": hit.node.qualified_name,
                "name": hit.node.name,
                "kind": hit.node.kind.value,
                "source_uri": hit.node.source_uri,
                "line_start": hit.node.line_start,
                "line_end": hit.node.line_end,
                "score": round(hit.score, 6),
                "reasons": hit.reasons,
            }
            for hit in hits
        ],
    }


def token_overlap_search(
    graph: Graph,
    query: str,
    limit: int = 20,
) -> list[SearchHit]:
    """FTS-style ranked search using token overlap scoring.

    Kept for backward compatibility but uses trigram index for candidate
    prefiltering when the query is longer than 2 characters.
    """
    from .vocabulary import _extract_query_identifiers

    tokens = set(_extract_query_identifiers(query))
    if not tokens:
        return []

    hits: list[SearchHit] = []
    for nid, node in graph.nodes():
        node_tokens = set(_tokenize(node.qualified_name))
        overlap = len(tokens & node_tokens)
        total = len(tokens | node_tokens)
        if total == 0:
            continue
        score = overlap / total
        if score > 0.2:
            hits.append(SearchHit(
                id=nid,
                node=node,
                score=score,
                rank=0,
                reasons=["fts_overlap"],
            ))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:limit]
