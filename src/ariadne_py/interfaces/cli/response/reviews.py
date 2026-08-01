"""Review operations: context, traverse, counterfactual, suggested questions.

Mirrors the Rust ``reviews.rs`` module.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from ....core.node import NodeKind

logger = logging.getLogger(__name__)


def review_context_json(
    graph,
    base: str,
    max_lines_per_file: int = 200,
    token_budget: int = 1600,
) -> dict[str, Any]:
    """Token-budgeted review context for changed and impacted files.

    Args:
        graph: The code graph.
        base: Git base ref.
        max_lines_per_file: Max lines to include per file.
        token_budget: Max tokens for snippets (~4 chars/token).

    Returns:
        Review context with change analysis and code snippets.
    """
    from .temporal import detect_changes_json

    analysis = detect_changes_json(graph, base, 2)

    # Collect files from changed and impacted
    files: list[str] = []
    seen_files: set[str] = set()
    for f in analysis.get("changed_files", []):
        if f not in seen_files:
            files.append(f)
            seen_files.add(f)

    for item in analysis.get("impacted", []):
        if isinstance(item, dict):
            source = item.get("source_uri")
            if source and source not in seen_files:
                files.append(source)
                seen_files.add(source)

    # Read file snippets within token budget
    snippets: list[dict[str, Any]] = []
    used_tokens = 0

    for file_path in files:
        if used_tokens >= token_budget:
            break
        try:
            snippet_text = file_snippet(file_path, max_lines_per_file)
            tokens = max(len(snippet_text) // 4, 1)
            if used_tokens + tokens > token_budget and snippets:
                continue
            used_tokens += tokens
            snippets.append({
                "path": file_path,
                "tokens": tokens,
                "changed_ranges": [],
                "snippet": snippet_text,
            })
        except OSError:
            continue

    return {
        "operation": "review_context",
        "base": base,
        "token_budget": token_budget,
        "used_tokens": used_tokens,
        "analysis": analysis,
        "snippets": snippets,
    }


def traverse_json(
    graph,
    target: str,
    direction: str = "both",
    max_depth: int = 3,
    token_budget: int = 1200,
) -> dict[str, Any]:
    """Traverse graph relationships from a target with a token budget.

    Args:
        graph: The code graph.
        target: Qualified name or node ID to start from.
        direction: "in", "out", or "both".
        max_depth: Max BFS depth.
        token_budget: Max tokens for traversal output.

    Returns:
        Traversal nodes with depth, degrees, etc.
    """
    # Resolve target to node ID
    seed = _resolve(graph, target)
    if seed is None:
        return {
            "operation": "traverse",
            "direction": direction,
            "used_tokens": 0,
            "error": f"target not found: {target}",
            "nodes": [],
        }

    queue = deque([(seed, 0)])
    seen = {seed}
    nodes: list[dict[str, Any]] = []
    used = 0

    while queue:
        if used >= token_budget:
            break
        node_id, depth = queue.popleft()
        node = graph.node(node_id) if hasattr(graph, "node") else None
        if node is None:
            # Try iterating
            for _, n in graph.nodes():
                if _ids_match(n, node_id):
                    node = n
                    break

        if node:
            item = {
                "depth": depth,
                "qualified_name": node.qualified_name,
                "kind": str(node.kind),
                "source_uri": node.source_uri,
                "in_degree": 0,
                "out_degree": 0,
            }
            # Count degrees
            if hasattr(graph, "in_neighbors"):
                item["in_degree"] = sum(1 for _ in graph.in_neighbors(node_id))
            if hasattr(graph, "out_neighbors"):
                item["out_degree"] = sum(1 for _ in graph.out_neighbors(node_id))

            item_str = str(item)
            used += max(len(item_str) // 4, 1)
            nodes.append(item)

        if depth >= max_depth:
            continue

        # Add neighbors to queue
        neighbors = []
        if direction in ("out", "both"):
            if hasattr(graph, "out_neighbors"):
                neighbors.extend(n for n, _ in graph.out_neighbors(node_id))
        if direction in ("in", "both"):
            if hasattr(graph, "in_neighbors"):
                neighbors.extend(n for n, _ in graph.in_neighbors(node_id))

        for next_id in neighbors:
            if next_id not in seen:
                seen.add(next_id)
                queue.append((next_id, depth + 1))

    return {
        "operation": "traverse",
        "direction": direction,
        "used_tokens": used,
        "nodes": nodes,
    }


def counterfactual_json(
    graph,
    target: str,
    direction: str = "out",
    max_depth: int = 5,
) -> dict[str, Any]:
    """Drop a symbol's edges, rerun BFS, report now-unreachable nodes.

    Args:
        graph: The code graph.
        target: Qualified name or node ID.
        direction: "in", "out", or "both" — which edges to drop.
        max_depth: Max BFS depth for reachability.

    Returns:
        Counterfactual analysis with before/after reachability.
    """
    target_id = _resolve(graph, target)
    if target_id is None:
        return {
            "operation": "counterfactual",
            "target": target,
            "direction": direction,
            "error": f"target not found: {target}",
            "dropped_edges": 0,
            "reachable_before": 0,
            "reachable_after": 0,
            "unreachable_count": 0,
            "now_unreachable": [],
        }

    # Collect edges to drop
    dropped_edge_ids: set = set()
    for edge_id, src, dst, edge in graph.edges():
        drop = False
        if direction == "in" and dst == target_id:
            drop = True
        elif direction == "out" and src == target_id:
            drop = True
        elif direction == "both" and (src == target_id or dst == target_id):
            drop = True
        if drop:
            dropped_edge_ids.add(edge_id)

    # BFS from target with all edges
    def _reach(all_edges: set) -> set:
        seen: set = {target_id}
        queue = deque([(target_id, 0)])
        while queue:
            node_id, depth = queue.popleft()
            if depth >= max_depth:
                continue
            neighbors = []
            if direction == "in":
                for nid, _, dst, _ in all_edges:
                    if dst == node_id:
                        neighbors.append(nid)
            elif direction == "out":
                for nid, src, _, _ in all_edges:
                    if src == node_id:
                        neighbors.append(nid)
            else:  # both
                for nid, src, dst, _ in all_edges:
                    if src == node_id or dst == node_id:
                        neighbors.append(nid)
            for n in neighbors:
                if n not in seen:
                    seen.add(n)
                    queue.append((n, depth + 1))
        return seen

    # All edges
    all_edges = list(graph.edges())
    before = _reach(set(all_edges))

    # Without dropped edges
    remaining = set(e for e in all_edges if e[0] not in dropped_edge_ids)
    after = _reach(set(remaining))

    lost = sorted(before - after, key=lambda x: str(x))

    # Get names for lost nodes
    now_unreachable = []
    for nid in lost[:50]:
        node = graph.node(nid) if hasattr(graph, "node") else None
        if node is None:
            for _, n in graph.nodes():
                if _ids_match(n, nid):
                    node = n
                    break
        if node:
            now_unreachable.append({
                "qualified_name": node.qualified_name,
                "kind": str(node.kind),
                "source_uri": node.source_uri,
            })

    return {
        "operation": "counterfactual",
        "target": target,
        "direction": direction,
        "dropped_edges": len(dropped_edge_ids),
        "reachable_before": len(before),
        "reachable_after": len(after),
        "unreachable_count": len(lost),
        "now_unreachable": now_unreachable,
    }


def suggested_questions_json(
    analysis: dict[str, Any],
    limit: int = 10,
) -> dict[str, Any]:
    """Generate prioritized review questions from graph analysis.

    Args:
        analysis: Change analysis from detect_changes_json.
        limit: Max questions to return.

    Returns:
        Questions dict with list of suggested questions.
    """
    questions: list[str] = []

    for file in analysis.get("changed_files", []):
        if isinstance(file, str):
            questions.append(
                f"What behavior changed in {file} and is it covered by tests?"
            )

    for hit in analysis.get("impacted", []):
        if isinstance(hit, dict):
            name = hit.get("qualified_name", "")
            if name:
                questions.append(f"Does the change alter assumptions relied on by {name}?")

    questions.append(
        "Are any unresolved calls or large functions involved in this change?"
    )

    return {
        "operation": "suggested_questions",
        "questions": questions[:limit],
    }


def file_snippet(path: str, max_lines: int = 200) -> str:
    """Read file and return lines with line numbers.

    Args:
        path: File path to read.
        max_lines: Max lines to return.

    Returns:
        File content with line numbers, or empty string on error.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        out = []
        for i, line in enumerate(lines[:max_lines]):
            out.append(f"{i + 1:4d}: {line.rstrip()}")
        return "\n".join(out)
    except OSError:
        return ""


# ── Internal helpers ───────────────────────────────────────────────


def _resolve(graph, target: str) -> Any:
    """Resolve a target string to a node ID.

    Tries exact qname match, then name match.
    """
    # Try exact qname
    nid = graph.find_by_qname(target) if hasattr(graph, "find_by_qname") else None
    if nid is not None:
        return nid

    # Try as integer node ID
    try:
        return int(target)
    except (ValueError, TypeError):
        pass

    # Try substring match
    for _, node in graph.nodes():
        if node.qualified_name == target or node.name == target:
            return node.id if hasattr(node, "id") else None

    return None


def _ids_match(node: Any, node_id: Any) -> bool:
    """Check if a node matches a node ID."""
    if not hasattr(node, "id"):
        return False
    return node.id == node_id  # type: ignore[return-value]
