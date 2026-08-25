"""Search and related operations.

Mirrors the Rust ``search.rs`` module.
"""

from __future__ import annotations

import re
from typing import Any


def handle_search(graph, params: dict[str, Any]) -> dict[str, Any]:
    """Search the graph by name, qualified name, or fuzzy match.

    Args:
        graph: The code graph.
        params: Search parameters (query, limit, etc.).

    Returns:
        Search results with hits and metadata.
    """
    query = params.get("query", params.get("target", "")).strip().lower()
    limit = int(params.get("limit", 50))

    if not query:
        return {"operation": "search", "hits": [], "total": 0}

    hits: list[dict[str, Any]] = []
    for _, node in graph.nodes():
        score = _score_match(query, node)
        if score > 0:
            hits.append({
                "score": round(score, 4),
                "qualified_name": node.qualified_name,
                "kind": str(node.kind),
                "name": node.name,
                "source_uri": node.source_uri,
                "line_start": node.line_start,
                "line_end": node.line_end,
            })

    hits.sort(key=lambda x: -x["score"])
    return {
        "operation": "search",
        "hits": hits[:limit],
        "total": len(hits),
    }


def handle_context_pack(graph, params: dict[str, Any]) -> dict[str, Any]:
    """Build a context pack around a target symbol.

    Args:
        graph: The code graph.
        params: Parameters (target, max_files, token_budget).

    Returns:
        Context pack with related nodes and source snippets.
    """
    target = params.get("target", "")
    max_files = int(params.get("max_files", 10))
    token_budget = int(params.get("token_budget", 4000))

    if not target:
        return {"operation": "context_pack", "files": []}

    nid = graph.find_by_qname(target) if hasattr(graph, "find_by_qname") else None
    if nid is None:
        for node_id, node in graph.nodes():
            if node.qualified_name == target or node.name == target:
                nid = node_id
                break

    if nid is None:
        return {"operation": "context_pack", "files": [], "error": f"target not found: {target}"}

    # Collect files from neighbors
    files: dict[str, int] = {}
    files[target] = 1
    token_used = 0

    # BFS for neighbors
    from collections import deque

    queue = deque([(nid, 0)])
    seen = {nid}
    while queue and len(files) < max_files:
        current, depth = queue.popleft()
        if depth > 2 or token_used >= token_budget:
            break

        for next_nid in _out_ids(graph, current) + _in_ids(graph, current):
            if next_nid not in seen:
                seen.add(next_nid)
                node = graph.node(next_nid) if hasattr(graph, "node") else None
                if node is None:
                    for _, n in graph.nodes():
                        if _ids_match(n, next_nid):
                            node = n
                            break
                if node:
                    source = node.source_uri or ""
                    if source:
                        files[source] = files.get(source, 0) + 1
                        token_used += max(len(source) // 4, 1)
                    if depth < 1:
                        queue.append((next_nid, depth + 1))

    return {
        "operation": "context_pack",
        "target": target,
        "files": list(files.keys()),
        "token_budget": token_budget,
        "token_used": token_used,
    }


def find_related_json(graph, target: str, line: int | None = None, limit: int = 25) -> dict[str, Any]:
    """Find nodes related to a target by graph proximity.

    Args:
        graph: The code graph.
        target: Qualified name or node ID.
        line: Optional line number for file-scoped matching.
        limit: Max results.

    Returns:
        Related nodes.
    """
    nid = graph.find_by_qname(target) if hasattr(graph, "find_by_qname") else None
    if nid is None:
        for node_id, node in graph.nodes():
            if node.qualified_name == target:
                nid = node_id
                break

    if nid is None:
        return {"operation": "find_related", "related": [], "total": 0}

    related = set()
    for _, src, dst, _ in graph.edges():
        if src == nid:
            related.add(dst)
        if dst == nid:
            related.add(src)

    results = []
    for rid in sorted(related, key=lambda item: item.value)[:limit]:
        node = graph.node(rid) if hasattr(graph, "node") else None
        if node is None:
            for _, n in graph.nodes():
                if _ids_match(n, rid):
                    node = n
                    break
        if node:
            results.append({
                "qualified_name": node.qualified_name,
                "kind": str(node.kind),
                "source_uri": node.source_uri,
                "line_start": node.line_start,
                "line_end": node.line_end,
            })

    return {
        "operation": "find_related",
        "target": target,
        "related": results,
        "total": len(results),
    }


# ── Helpers ────────────────────────────────────────────────────────


def _score_match(query: str, node: Any) -> float:
    """Score a node against a query string."""
    name_lower = (node.name or "").lower()
    qn_lower = (node.qualified_name or "").lower()

    if name_lower == query or qn_lower == query:
        return 1.0

    if name_lower.startswith(query) or qn_lower.startswith(query):
        return 0.9

    if query in name_lower or query in qn_lower:
        return 0.7

    # Fuzzy: check if query chars appear in order
    if _fuzzy_match(query, name_lower) or _fuzzy_match(query, qn_lower):
        return 0.4

    # Token overlap
    query_tokens = set(re.findall(r"\w+", query))
    name_tokens = set(re.findall(r"\w+", name_lower))
    qn_tokens = set(re.findall(r"\w+", qn_lower))

    if query_tokens & name_tokens:
        return 0.3
    if query_tokens & qn_tokens:
        return 0.3

    return 0.0


def _fuzzy_match(pattern: str, text: str) -> bool:
    """Check if pattern chars appear in text in order."""
    it = iter(text)
    return all(c in it for c in pattern)


def _out_ids(graph, node_id: Any) -> list[Any]:
    """Get outgoing neighbor IDs."""
    if hasattr(graph, "out_neighbors"):
        return [n for n, _ in graph.out_neighbors(node_id)]
    return []


def _in_ids(graph, node_id: Any) -> list[Any]:
    """Get incoming neighbor IDs."""
    if hasattr(graph, "in_neighbors"):
        return [n for n, _ in graph.in_neighbors(node_id)]
    return []


def _ids_match(node: Any, node_id: Any) -> bool:
    """Check if a node matches a node ID."""
    if not hasattr(node, "id"):
        return False
    return node.id == node_id
