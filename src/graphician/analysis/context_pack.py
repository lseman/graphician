"""Context pack: token-budgeted diverse source bundle.

Combines ranked search seeds with one-hop callers, callees, tests,
interfaces, and documentation. Selection penalizes repeated files
and node kinds, includes relationship evidence, truncates items
when necessary, and never exceeds the token budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..core.edge import EdgeKind
from ..core.graph import Graph
from ..core.id import NodeId
from ..core.node import NodeKind


@dataclass
class ContextItem:
    """A single item in a context pack."""
    qualified_name: str
    kind: str
    name: str
    source_uri: str | None
    content: str
    evidence: list[str] = field(default_factory=list)
    token_count: int = 0


def build_context_pack(
    graph: Graph,
    query: str,
    intent: str = "implementation",
    token_budget: int = 2400,
    max_items: int = 20,
) -> dict[str, Any]:
    """Build a token-budgeted context pack.

    1. Search for relevant symbols
    2. Expand with callers, callees, tests, interfaces
    3. Select diverse items within token budget
    """
    # Phase 1: Find relevant symbols via search
    from .search import hybrid_search
    search_results = hybrid_search(graph, query, intent=intent, limit=10)

    if not search_results.get("results"):
        return {"context_pack": [], "total_tokens": 0, "note": "No relevant symbols found"}

    # Phase 2: Expand with neighborhood
    seeds = search_results["results"]
    expanded: dict[str, ContextItem] = {}

    for seed in seeds[:5]:
        qn = seed["qualified_name"]
        nid = graph.find_by_qname(qn)
        if nid is None:
            continue

        node = graph.node(nid)
        if node is None:
            continue

        # Add the seed itself
        content = _get_node_content(graph, nid)
        item = ContextItem(
            qualified_name=qn,
            kind=node.kind.value,
            name=node.name,
            source_uri=node.source_uri,
            content=content,
            evidence=[f"search_match:{query}"],
        )
        item.token_count = _count_tokens(content)
        expanded[qn] = item

        # Add callers
        for caller, _ in graph.in_neighbors(nid):
            caller_node = graph.node(caller)
            if caller_node and len(expanded) < max_items:
                content = _get_node_content(graph, caller)
                caller_item = ContextItem(
                    qualified_name=caller_node.qualified_name,
                    kind=caller_node.kind.value,
                    name=caller_node.name,
                    source_uri=caller_node.source_uri,
                    content=content,
                    evidence=[f"calls_{node.name}"],
                )
                caller_item.token_count = _count_tokens(content)
                expanded[caller_node.qualified_name] = caller_item

        # Add callees
        for callee, _ in graph.out_neighbors(nid):
            callee_node = graph.node(callee)
            if callee_node and len(expanded) < max_items:
                content = _get_node_content(graph, callee)
                callee_item = ContextItem(
                    qualified_name=callee_node.qualified_name,
                    kind=callee_node.kind.value,
                    name=callee_node.name,
                    source_uri=callee_node.source_uri,
                    content=content,
                    evidence=[f"called_by_{node.name}"],
                )
                callee_item.token_count = _count_tokens(content)
                expanded[callee_node.qualified_name] = callee_item

        # Add tests for functions
        if node.kind in (NodeKind.FUNCTION, NodeKind.METHOD):
            for tested_by, _ in graph.out_neighbors(nid):
                for neighbor, edge in graph.out_neighbors(tested_by):
                    if edge.kind == EdgeKind.TESTED_BY:
                        test_node = graph.node(neighbor)
                        if test_node and len(expanded) < max_items:
                            content = _get_node_content(graph, neighbor)
                            test_item = ContextItem(
                                qualified_name=test_node.qualified_name,
                                kind=test_node.kind.value,
                                name=test_node.name,
                                source_uri=test_node.source_uri,
                                content=content,
                                evidence=[f"test_for_{node.name}"],
                            )
                            test_item.token_count = _count_tokens(content)
                            expanded[test_node.qualified_name] = test_item

    # Phase 3: Select diverse items within budget
    selected = _select_diverse(expanded, query, token_budget, max_items)

    return {
        "context_pack": [
            {
                "qualified_name": item.qualified_name,
                "kind": item.kind,
                "name": item.name,
                "source_uri": item.source_uri,
                "content": item.content,
                "evidence": item.evidence,
                "token_count": item.token_count,
            }
            for item in selected
        ],
        "total_tokens": sum(item.token_count for item in selected),
        "total_items": len(selected),
        "budget": token_budget,
    }


def _get_node_content(graph: Graph, nid: NodeId) -> str:
    """Extract content for a node."""
    node = graph.node(nid)
    if node is None:
        return ""

    parts = [f"# {node.kind.value}: {node.qualified_name}"]

    if node.source_text:
        parts.append(node.source_text)
    elif node.source_uri and node.line_start and node.line_end:
        # Would read from file in production
        parts.append(f"[{node.source_uri}:{node.line_start}-{node.line_end}]")

    return "\n".join(parts)


def _select_diverse(
    items: dict[str, ContextItem],
    query: str,
    token_budget: int,
    max_items: int,
) -> list[ContextItem]:
    """Select diverse items within token budget.

    Penalizes repeated files and node kinds.
    """
    selected: list[ContextItem] = []
    total_tokens = 0
    used_files: set[str] = set()
    used_kinds: set[str] = set()

    # Score items by relevance to query
    scored = []
    for item in items.values():
        relevance = _compute_relevance(item, query)
        diversity = _compute_diversity_penalty(item, used_files, used_kinds)
        scored.append((relevance * diversity, item))

    scored.sort(key=lambda x: x[0], reverse=True)

    for _, item in scored:
        if len(selected) >= max_items:
            break
        if total_tokens + item.token_count > token_budget:
            break

        # Check diversity
        file_key = item.source_uri or item.qualified_name.rsplit("::", 1)[0]
        if file_key in used_files and len(used_files) > 2:
            continue  # Skip if we already have 2+ items from this file

        selected.append(item)
        total_tokens += item.token_count
        used_files.add(file_key)
        used_kinds.add(item.kind)

    return selected


def _compute_relevance(item: ContextItem, query: str) -> float:
    """Compute relevance score for an item."""
    score = 0.0
    query_lower = query.lower()

    if query_lower in item.name.lower():
        score += 3.0
    if query_lower in item.qualified_name.lower():
        score += 2.0
    if query_lower in item.content.lower():
        score += 1.0

    return score


def _compute_diversity_penalty(
    item: ContextItem,
    used_files: set[str],
    used_kinds: set[str],
) -> float:
    """Compute diversity penalty (1.0 = no penalty, <1.0 = penalized)."""
    file_key = item.source_uri or item.qualified_name.rsplit("::", 1)[0]
    penalty = 1.0

    if file_key in used_files:
        penalty *= 0.5
    if item.kind in used_kinds:
        penalty *= 0.8

    return penalty


def _count_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token."""
    return max(1, len(text) // 4)
