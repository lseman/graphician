"""Trigram-based candidate prefilter for fast graph search.

Inspired by graphify's serve.py trigram index. Instead of scoring all N nodes
with expensive fuzzy matching, we build a character trigram -> node-id postings
map and intersect posting lists to narrow candidates before any scoring.

Core insight: a query like "CacheManager" contains trigrams "cac", "ach", "che",
"hem", "ema", "mag", "age", "ges", "ges", "est". Only nodes whose searchable
text contains all (or most) of these trigrams can match. For a 15k-node graph,
this typically narrows to <100 candidates.
"""

from __future__ import annotations

import math
from typing import Any

from ...core.graph import Graph
from ...core.node import Node
from .fuzzy import _fuzzy_score
from .vocabulary import (
    _extract_query_identifiers,
    _normalize_identifier,
    _search_tokens,
)

# Scoring constants (graphify-inspired, tuned for codebase search)
_EXACT_MATCH_BONUS = 1000.0
_PREFIX_MATCH_BONUS = 100.0
_SUBSTRING_MATCH_BONUS = 1.0
_SOURCE_MATCH_BONUS = 0.5
_RATIONALE_MATCH_BONUS = 0.75


def _trigrams(text: str) -> set[str]:
    """Character trigrams of `text`; for <3-char text the whole string is the key."""
    if len(text) < 3:
        return {text} if text else set()
    return {text[i:i + 3] for i in range(len(text) - 2)}


def _build_search_text(node: Node, nid: str) -> str:
    """Concatenate every searchable field into one NUL-separated text blob.

    This is indexed once into the trigram postings map, then every search
    predicate (exact, prefix, substring, source, rationale) can check the
    same node without re-reading attributes.

    NUL separators prevent a trigram from spanning two fields (a query
    never contains NUL, so a cross-field trigram can never be a real match).
    """
    fields: list[str] = []

    # Primary label (diacritic-folded via the node's normalized form if available)
    norm_label = node.normalized_name or node.name.lower()
    fields.append(norm_label)

    # Tokenized label -- punctuation-stripped version for token-level matching
    label_tokens = " ".join(_search_tokens(node.name))
    fields.append(label_tokens)

    # Node ID
    fields.append(nid.lower())

    # Source file path
    source = (node.source_uri or "").lower()
    fields.append(source)

    # Tokenized source path
    source_tokens = " ".join(_search_tokens(node.source_uri or ""))
    fields.append(source_tokens)

    # Qualified name (normalized if available)
    if hasattr(node, "normalized_qname") and node.normalized_qname:
        fields.append(node.normalized_qname)
        fields.append(" ".join(_search_tokens(node.normalized_qname)))

    # Properties as searchable text
    props = node.properties or {}
    prop_text_parts: list[str] = []
    prop_token_parts: list[str] = []
    for k, v in props.items():
        k_str = str(k).lower()
        v_str = str(v)
        prop_text_parts.append(f"{k_str} {v_str}")
        prop_token_parts.extend(_search_tokens(k_str))
        prop_token_parts.extend(_search_tokens(v_str))
    if prop_text_parts:
        fields.append(" ".join(prop_text_parts))
        if prop_token_parts:
            fields.append(" ".join(prop_token_parts))

    return "\x00".join(fields)


def _get_trigram_index(graph: Graph) -> dict[str, Any]:
    """Lazily build and cache a trigram -> node-id postings map on the graph.

    Cached on ``graph._trigram_index`` so it auto-invalidates when a fresh graph
    object is constructed (new dict -> no cached index).
    """
    cache_key = "_trigram_index"
    idx = getattr(graph, cache_key, None)
    if idx is not None:
        return idx

    ids = [nid for nid, _ in graph.nodes()]
    n = len(ids)
    if n == 0:
        idx = {"ids": [], "postings": {}, "set_cache": {}}
        setattr(graph, cache_key, idx)
        return idx

    postings: dict[str, list[int]] = {}
    node_texts: list[str] = []

    for i, (_nid, node) in enumerate(graph.nodes()):
        text = _build_search_text(node, str(_nid.value))
        node_texts.append(text)
        for g in _trigrams(text):
            bucket = postings.get(g)
            if bucket is None:
                bucket = []
                postings[g] = bucket
            bucket.append(i)

    idx = {"ids": ids, "node_texts": node_texts, "postings": postings, "set_cache": {}}
    setattr(graph, cache_key, idx)
    return idx


def _trigram_candidates(
    graph: Graph,
    needles: list[str],
    *,
    guard_frac: float = 0.10,
) -> list[str] | None:
    """Node IDs whose searchable text could contain any needle as a substring.

    Uses trigram intersection to find a candidate set that is a superset of
    actual matches -- the caller then re-filters with exact predicates.

    Returns candidates in graph-iteration order, or **None** when the index
    isn't worth it (needle too short, or rarest trigram still too common).
    The caller falls back to full scan in that case, preserving correctness.
    """
    idx = _get_trigram_index(graph)
    ids = idx["ids"]
    postings = idx["postings"]
    set_cache = idx["set_cache"]
    n = len(ids)

    if n == 0:
        return []

    # Lowercase needles for trigram matching (node text is always lowercase)
    needles = [s.lower() for s in needles if s]
    if not needles:
        return None

    thresh = int(n * guard_frac)

    # Check if any needle is too short or has a common trigram
    for s in needles:
        tgs = _trigrams(s)
        if not tgs or any(len(g) < 3 for g in tgs):
            return None  # too short to trigram-filter

        # Check if rarest trigram is still too common
        present_counts = [len(postings[g]) for g in tgs if g in postings]
        if not present_counts:
            continue  # this needle matches nothing -- contributes no candidates
        if min(present_counts) > thresh:
            return None  # rarest trigram still too common -> not worth the index

    # Intersect posting lists (smallest-first)
    cand: set[int] = set()
    for s in needles:
        tgs = _trigrams(s)
        sets: list[set[int]] = []
        for g in tgs:
            bucket = postings.get(g)
            if bucket is None:
                sets = []  # trigram absent everywhere -> needle matches nothing
                break
            cached = set_cache.get(g)
            if cached is None:
                cached = set(bucket)
                set_cache[g] = cached
            sets.append(cached)

        if not sets:
            continue

        sets.sort(key=len)  # intersect smallest-first for efficiency
        hit = sets[0]
        for other in sets[1:]:
            hit = hit & other
            if not hit:
                break
        cand |= hit

    return [ids[i] for i in sorted(cand)]


def _compute_idf(graph: Graph, terms: list[str]) -> dict[str, float]:
    """Compute IDF weights for query terms.

    For each term, counts how many nodes contain it in their label or source
    path, then computes log(1 + N / df). Rare terms get high weight.

    Cached on the graph object alongside the trigram index.
    """
    cache_key = "_idf_cache"
    cache = getattr(graph, cache_key, None)
    if cache is not None:
        return {t: cache.get(t, math.log(1 + graph.node_count())) for t in terms}

    n = graph.node_count()
    # Count document frequency for each term across the whole graph
    term_set = set(terms)
    df: dict[str, int] = {t: 0 for t in term_set}

    for _, node in graph.nodes():
        norm_label = node.normalized_name or node.name.lower()
        source = (node.source_uri or "").lower()
        for t in term_set:
            if t in norm_label or t in source:
                df[t] += 1

    idf = {t: math.log(1 + n / max(df.get(t, 1), 1)) for t in terms}
    # Cache for this graph
    if cache is None:
        setattr(graph, cache_key, idf)
    return idf


def _score_tiered(
    graph: Graph,
    query: str,
    *,
    limit: int = 20,
    fuzzy_on_candidates: bool = True,
    _min_score: float = 0.3,
) -> list[tuple[float, str]]:
    """Tiered scoring with trigram prefilter.

    This is the core search engine: a single-pass scorer that:
    1. Normalizes and tokenizes the query (drops stopwords, splits on boundaries)
    2. Computes IDF weights for query terms
    3. Uses trigram index to get candidate set (falls back to all nodes if index
       isn't selective enough)
    4. Scores each candidate with tiered matching:
       - Exact label match: 1000 x IDF
       - Prefix match: 100 x IDF
       - Substring match: 1 x IDF
       - Source path match: 0.5 x IDF
       - Qualified name match: 1.5x boost
    5. Scales tier bonuses by squared term coverage (for multi-term queries)
    6. Optionally applies fuzzy scoring as refinement on top candidates
    7. Returns top-N scored (score, node_id) pairs
    """
    # Normalize query and extract terms
    qn = _normalize_identifier(query)
    raw_terms = _extract_query_identifiers(query)
    if not raw_terms:
        # Fallback: use the normalized query as a single term
        raw_terms = [t for t in qn.split() if len(t) > 1]

    # Dedupe, order-preserving
    norm_terms = list(dict.fromkeys(raw_terms))
    n_terms = len(norm_terms)
    if n_terms == 0:
        return []

    # IDF weights
    idf = _compute_idf(graph, norm_terms)

    # Full-query string for exact-label matching
    joined = " ".join(norm_terms)
    # Weight the full-query bonus by the rarest constituent term
    joined_w = max((idf.get(t, 1.0) for t in norm_terms), default=1.0)

    # Trigram prefilter
    candidate_ids = _trigram_candidates(graph, norm_terms + ([joined] if joined else []))
    nodes_iter = (
        ((nid, graph.node(nid)) for nid in candidate_ids)
        if candidate_ids is not None
        else graph.nodes()
    )

    scored: list[tuple[float, str]] = []

    for nid, node in nodes_iter:
        score = 0.0
        matched = 0

        # Precompute node fields
        norm_label = node.normalized_name or node.name.lower()
        bare_label = norm_label.rstrip("()")
        label_tokens = " ".join(_search_tokens(node.name))
        source = (node.source_uri or "").lower()
        qname = node.normalized_qname or _normalize_identifier(node.qualified_name)
        nid_lower = str(nid).lower()

        # Full-query tier: multi-word query that equals/prefixes whole label
        if joined:
            if joined in (norm_label, bare_label, label_tokens, nid_lower):
                score += _EXACT_MATCH_BONUS * 10 * joined_w
            elif (
                norm_label.startswith(joined)
                or bare_label.startswith(joined)
                or label_tokens.startswith(joined)
            ):
                score += _PREFIX_MATCH_BONUS * 10 * joined_w

        # Per-term scoring
        tiered = 0.0
        for t in norm_terms:
            w = idf.get(t, 1.0)

            # Tier precedence: exact > prefix > substring (take strongest)
            tier_value = 0.0
            if t in (norm_label, bare_label):
                tier_value = _EXACT_MATCH_BONUS * w
                matched += 1
            elif norm_label.startswith(t) or bare_label.startswith(t):
                tier_value = _PREFIX_MATCH_BONUS * w
                matched += 1
            elif t in norm_label:
                score += _SUBSTRING_MATCH_BONUS * w
                matched += 1

            # Source path match
            if t in source:
                score += _SOURCE_MATCH_BONUS * w

            # Qualified name match
            if t in qname:
                score += _SUBSTRING_MATCH_BONUS * 0.5 * w

            tiered += tier_value

        # Term coverage scaling: squared fraction of terms matched by label
        if tiered:
            score += tiered * (matched / n_terms) ** 2

        if score > _min_score:
            scored.append((score, nid))

    # Optional fuzzy refinement on top candidates
    if fuzzy_on_candidates and scored:
        # Sort by tiered score first, apply fuzzy as tiebreaker to top N
        scored.sort(key=lambda s: -s[0])
        top_n = min(limit * 3, len(scored))
        # Build a lookup: nid -> (score, position)
        nid_to_idx: dict[str, int] = {str(nid): i for i, (_, nid) in enumerate(scored)}
        top_ids = [nid for _, nid in scored[:top_n]]
        for nid in top_ids:
            nid_str = str(nid)
            pos = nid_to_idx.get(nid_str)
            if pos is None:
                continue
            node = graph.node(nid)
            if node is None:
                continue
            current_score, _ = scored[pos]
            fuzzy_score_val = _fuzzy_score(qn, node.name)
            if fuzzy_score_val > 0.7:
                # Apply a small boost (fuzzy confidence adds confidence)
                scored[pos] = (current_score * (1.0 + fuzzy_score_val * 0.05), nid)

    # Final sort: score desc, then shorter label first, then nid asc
    scored.sort(key=lambda s: (
        -s[0],
        len(graph.node(s[1]).name),
        str(s[1]),
    ))

    return scored[:limit]
