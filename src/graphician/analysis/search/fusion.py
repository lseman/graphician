"""Hybrid FTS5/semantic search fusion via reciprocal-rank fusion.

Fuses SQLite full-text and embedding candidates with in-memory ranked-
search signals, then applies kind/identifier/noise boosts and source-
saturation decay.

Mirrors the Rust ``fusion.rs`` module.
"""

from __future__ import annotations

from collections import defaultdict

from ...core.graph import Graph
from ...core.id import NodeId
from ...core.node import Node, NodeKind
from .types import SearchHit
from .vocabulary import SEARCH_STOPWORDS, _normalize_identifier

# Reciprocal rank fusion constant
RRF_K: float = 60.0
# Source saturation decay multiplier
SOURCE_SATURATION_DECAY: float = 0.72


def fts_ranked_search(
    graph: Graph,
    query: str,
    limit: int = 20,
    fts_hits: list[tuple[str, float]] | None = None,
    semantic_hits: list[tuple[str, float]] | None = None,
) -> list[SearchHit]:
    """FTS5-boosted hybrid search with reciprocal-rank fusion.

    Runs FTS5 and semantic search, then fuses candidates with the
    in-memory ranked_search results using reciprocal-rank fusion.
    Applies kind, identifier, qualified-name, and source penalties.

    Args:
        graph: The code graph.
        query: The search query.
        limit: Maximum results to return.
        fts_hits: Pre-computed FTS5 hits as [(qname, score), ...].
        semantic_hits: Pre-computed semantic hits as [(qname, score), ...].

    Returns:
        Ranked list of SearchHit objects.
    """
    from .search import ranked_search

    query_identifiers = _extract_query_identifiers(query)
    normalized_query = _normalize_identifier(query)

    # Start from in-memory results
    mem_hits = ranked_search(graph, query, limit * 2)
    merged: dict[int, SearchHit] = {h.id.value: h for h in mem_hits}

    # Fuse FTS5 candidates
    if fts_hits:
        for rank, (qname, _) in enumerate(fts_hits):
            nid = graph.find_by_qname(qname)
            if nid is None:
                continue
            nid_node = graph.node(nid)
            if nid_node is None:
                continue
            fts_boost = reciprocal_rank_boost(rank, 3600.0)
            if nid.value in merged:
                merged[nid.value].score += fts_boost
                if "fts5" not in merged[nid.value].reasons:
                    merged[nid.value].reasons.append("fts5")
            else:
                merged[nid.value] = SearchHit(
                    id=nid,
                    node=nid_node,
                    score=fts_boost,
                    reasons=["fts5"],
                )

    # Fuse semantic candidates
    if semantic_hits:
        for rank, (qname, _) in enumerate(semantic_hits):
            nid = graph.find_by_qname(qname)
            if nid is None:
                continue
            nid_node = graph.node(nid)
            if nid_node is None:
                continue
            semantic_boost = reciprocal_rank_boost(rank, 2700.0)
            if nid.value in merged:
                merged[nid.value].score += semantic_boost
                if "semantic" not in merged[nid.value].reasons:
                    merged[nid.value].reasons.append("semantic")
            else:
                merged[nid.value] = SearchHit(
                    id=nid,
                    node=nid_node,
                    score=semantic_boost,
                    reasons=["semantic"],
                )

    # Apply boosts
    kind_boosts = _query_kind_boosts(query)
    dotted_query = "." in query
    query_tokens = _search_query_tokens(normalized_query)
    symbol_query = _is_symbol_query(query)

    for hit_id, hit in list(merged.items()):
        node = graph.node(NodeId(hit_id))
        if node is None:
            continue

        # Kind boost
        for kind, multiplier in kind_boosts:
            if node.kind == kind:
                hit.score *= multiplier
                if "kind_boost" not in hit.reasons:
                    hit.reasons.append("kind_boost")
                break

        # Qualified name boost
        if dotted_query:
            normalized_qname = _normalize_identifier(node.qualified_name.replace("::", "."))
            if normalized_query in normalized_qname:
                hit.score *= 1.25
                if "qualified_boost" not in hit.reasons:
                    hit.reasons.append("qualified_boost")

        # Identifier boost
        if query_identifiers:
            normalized_qname = _normalize_identifier(
                node.qualified_name.replace("::", " ")
            )
            if any(identifier in normalized_qname for identifier in query_identifiers):
                hit.score *= 1.30
                if "identifier_boost" not in hit.reasons:
                    hit.reasons.append("identifier_boost")

        # Definition boost for symbol queries
        if symbol_query and _is_definition_like_node(node):
            leaf = query.split("::")[-1].split(".")[-1]
            leaf_norm = _normalize_identifier(leaf)
            name_norm = _normalize_identifier(node.name)
            qname_norm = _normalize_identifier(node.qualified_name.replace("::", "."))
            if leaf_norm == name_norm or normalized_query in qname_norm:
                hit.score *= 1.35
                if "definition_boost" not in hit.reasons:
                    hit.reasons.append("definition_boost")

        # Placeholder penalty
        if node.qualified_name.startswith("call::"):
            hit.score *= 0.45
            if "placeholder_penalty" not in hit.reasons:
                hit.reasons.append("placeholder_penalty")

        # Noise penalty
        _apply_noise_penalty(hit, node, normalized_query)

        # File stem boost
        if query_tokens and node.source_uri:
            stem = _source_stem(node.source_uri)
            stem_norm = _normalize_identifier(stem)
            if any(token in stem_norm for token in query_tokens):
                hit.score *= 1.12
                if "file_stem_boost" not in hit.reasons:
                    hit.reasons.append("file_stem_boost")

    # Boost multi-hit sources
    boost_multi_hit_sources(merged, graph)

    # Apply source saturation
    apply_source_saturation(merged, graph)

    # Sort and limit
    hits = sorted(merged.values(), key=lambda h: -h.score)
    return hits[:limit]


def apply_source_saturation(hits: dict[int, SearchHit], graph: Graph) -> None:
    """Apply source saturation decay for repeated sources.

    Each additional hit from the same source file gets multiplied by
    SOURCE_SATURATION_DECAY ^ (count - 1).

    Args:
        hits: Mapping of node id → SearchHit.
        graph: The code graph.
    """
    seen_by_source: dict[str, int] = defaultdict(int)
    for nid, hit in hits.items():
        node = graph.node(NodeId(nid))
        if node is None or node.source_uri is None:
            continue
        source = node.source_uri
        if seen_by_source[source] > 0:
            hit.score *= SOURCE_SATURATION_DECAY ** seen_by_source[source]
            if "source_saturation" not in hit.reasons:
                hit.reasons.append("source_saturation")
        seen_by_source[source] += 1


def reciprocal_rank_boost(rank: int, weight: float) -> float:
    """Compute reciprocal rank fusion boost.

    Args:
        rank: 0-based rank in the source list.
        weight: Maximum boost weight.

    Returns:
        Boost value.
    """
    return weight / (RRF_K + rank + 1.0)


def _is_definition_like_node(node: Node) -> bool:
    """True for nodes that represent real symbols (not placeholders)."""
    if node.qualified_name.startswith("call::"):
        return False
    return node.kind in (
        NodeKind.FUNCTION,
        NodeKind.METHOD,
        NodeKind.CLASS,
        NodeKind.TYPE,
        NodeKind.TRAIT,
        NodeKind.IMPL,
        NodeKind.MODULE,
    )


def _is_symbol_query(query: str) -> bool:
    """True if the query looks like a symbol reference."""
    q = query.strip()
    if not q or " " in q:
        return False
    return any(c in q for c in ("::", ".", "_")) or any(
        c.isupper() for c in q
    )


def _apply_noise_penalty(
    hit: SearchHit,
    node: Node,
    normalized_query: str,
) -> None:
    """Apply noise penalty based on source file type.

    Reduces scores for test files, declaration files, examples, and
    generated/legacy code — unless the query explicitly asks for them.
    """
    if node.source_uri is None:
        return

    # Skip if query explicitly requests noisy content
    if any(token in normalized_query.split() for token in (
        "test", "tests", "testing", "spec", "specs", "example",
        "examples", "sample", "samples", "legacy", "compat",
        "generated", "declaration", "types",
    )):
        return

    source_lower = node.source_uri.replace("\\", "/").lower()
    multiplier = 1.0

    if _is_test_file_path(source_lower):
        multiplier = min(multiplier, 0.72)
    if source_lower.endswith(".d.ts"):
        multiplier = min(multiplier, 0.70)
    if any(x in source_lower for x in ("/examples/", "examples/", "/sample/", "sample/")):
        multiplier = min(multiplier, 0.82)
    if any(x in source_lower for x in ("/legacy/", "legacy/", "/compat/", "compat/", "/generated/", "generated/")):
        multiplier = min(multiplier, 0.78)

    if multiplier < 1.0:
        hit.score *= multiplier
        if "noise_penalty" not in hit.reasons:
            hit.reasons.append("noise_penalty")


def _extract_query_identifiers(query: str) -> list[str]:
    """Extract identifier-like tokens from a query.

    Returns tokens that look like code identifiers (camelCase, snake_case,
    or containing :: or .).
    """
    # Split on whitespace and punctuation
    tokens = query.split()
    identifiers: list[str] = []
    for token in tokens:
        # Remove common punctuation
        cleaned = token.strip(".,;:!?\"'()[]{}")
        if cleaned and any(c.isupper() or c in (":", "_") for c in cleaned):
            identifiers.append(_normalize_identifier(cleaned))
    return identifiers


def _query_kind_boosts(query: str) -> list[tuple[NodeKind, float]]:
    """Return kind multipliers to apply if the query matches a kind."""
    q_lower = query.lower()

    if any(kw in q_lower for kw in ("function", "method", "fn", "func")):
        return [(NodeKind.FUNCTION, 1.3), (NodeKind.METHOD, 1.2)]
    if any(kw in q_lower for kw in ("class", "struct", "type")):
        return [(NodeKind.CLASS, 1.3), (NodeKind.TYPE, 1.2)]
    if any(kw in q_lower for kw in ("module", "package", "lib")):
        return [(NodeKind.MODULE, 1.3)]
    return []


def _search_query_tokens(normalized_query: str) -> list[str]:
    """Extract search tokens (length >= 3, not stopwords)."""
    return [
        token
        for token in normalized_query.split()
        if len(token) >= 3 and token not in SEARCH_STOPWORDS
    ]


def _source_stem(source: str) -> str:
    """Extract the stem (filename without extension) from a source path."""
    from pathlib import Path

    path = Path(source)
    stem = path.stem
    return stem


def _is_test_file_path(source: str) -> bool:
    """Check if a source path looks like a test file."""
    source_lower = source.lower()
    return any(
        x in source_lower
        for x in ("/test", "test_", "_test.", ".test.", "_spec.", "spec_")
    )


def boost_multi_hit_sources(
    hits: dict[int, SearchHit],
    graph: Graph,
) -> None:
    """Boost files that have multiple relevant hits.

    Files with multiple hits get a coherence boost proportional to their
    total score relative to the file with the highest total score.
    """
    # Sum scores by source
    file_sum: dict[str, float] = defaultdict(float)
    best: dict[str, int] = {}

    for nid, hit in hits.items():
        node = graph.node(NodeId(nid))
        if node is None or node.source_uri is None:
            continue
        source = node.source_uri
        file_sum[source] += max(hit.score, 0.0)
        if source not in best or hits[best[source]].score < hit.score:
            best[source] = nid

    if not file_sum:
        return

    max_sum = max(file_sum.values())
    if max_sum <= 0:
        return

    max_score = max((h.score for h in hits.values()), default=0.0)
    boost_unit = max_score * 0.12

    for source, nid in best.items():
        count = sum(1 for h in hits.values() if h.node and h.node.source_uri == source)
        if count < 2:
            continue

        hits[nid].score += boost_unit * file_sum[source] / max_sum
        if "file_coherence" not in hits[nid].reasons:
            hits[nid].reasons.append("file_coherence")
