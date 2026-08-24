"""Search functions: ranked_search, search_by_name, task_aware_search, fts_ranked_search."""

from __future__ import annotations

from typing import Any

from ..._extract import fuzzy_score_matrix
from ...core.graph import Graph
from ...core.node import NodeKind
from .fuzzy import _fuzzy_score
from .types import SearchHit, SearchIntent
from .vocabulary import _extract_query_identifiers, _normalize_identifier, _tokenize


def ranked_search(
    graph: Graph,
    query: str,
    limit: int = 20,
) -> list[SearchHit]:
    """In-memory ranked search with fuzzy + topology scoring."""
    qn = _normalize_identifier(query)
    hits: list[SearchHit] = []
    nodes = list(graph.nodes())
    native_scores = (
        fuzzy_score_matrix([qn], [node.name for _, node in nodes])[0]
        if fuzzy_score_matrix is not None
        else None
    )

    for index, (nid, node) in enumerate(nodes):
        # Score by name similarity
        score = native_scores[index] if native_scores is not None else _fuzzy_score(qn, node.name)
        # Boost for qualified name match
        if qn in _normalize_identifier(node.qualified_name):
            score *= 1.5
        # Boost for exact match
        if node.name.lower() == qn:
            score = max(score, 1.0)

        if score > 0.3:
            hits.append(SearchHit(
                id=nid,
                node=node,
                score=score,
                reasons=["name_match"],
            ))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:limit]


def search_by_name(
    graph: Graph,
    name: str,
    exact: bool = False,
) -> list[SearchHit]:
    """Exact or substring name lookup."""
    hits: list[SearchHit] = []
    name_lower = name.lower()

    for nid, node in graph.nodes():
        match = False
        if exact:
            if node.name.lower() == name_lower:
                match = True
        else:
            if name_lower in node.name.lower():
                match = True

        if match:
            hits.append(SearchHit(
                id=nid,
                node=node,
                score=1.0 if exact else 0.5,
                reasons=["name_lookup"],
            ))

    return hits


def task_aware_search(
    graph: Graph,
    query: str,
    limit: int = 20,
    intent: SearchIntent | None = None,
) -> list[SearchHit]:
    """Intent-classified hybrid search."""
    if intent is None:
        intent = SearchIntent.classify(query)
    identifiers = _extract_query_identifiers(query)

    hits: list[SearchHit] = []
    nodes = list(graph.nodes())
    normalized_names = [_normalize_identifier(node.name) for _, node in nodes]
    native_scores = (
        fuzzy_score_matrix(identifiers, normalized_names)
        if fuzzy_score_matrix is not None and identifiers
        else None
    )
    for node_index, (nid, node) in enumerate(nodes):
        score = 0.0
        reasons: list[str] = []

        # Name-based scoring
        if identifiers:
            normalized_qname = _normalize_identifier(node.qualified_name)
            name_score = max(
                max(
                    native_scores[identifier_index][node_index]
                    if native_scores is not None
                    else _fuzzy_score(identifier, normalized_names[node_index]),
                    0.85 if identifier in normalized_qname else 0.0,
                )
                for identifier_index, identifier in enumerate(identifiers)
            )
            if name_score > 0:
                score += name_score * 2.0
                reasons.append("name_match")

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
        elif intent == SearchIntent.ARCHITECTURE and node.kind == NodeKind.MODULE:
            score *= 1.5

        if score > 0.3:
            hits.append(SearchHit(
                id=nid,
                node=node,
                score=score,
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
    """FTS-style ranked search using token overlap scoring."""
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
                reasons=["fts_overlap"],
            ))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:limit]
