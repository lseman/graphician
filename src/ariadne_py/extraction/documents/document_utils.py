"""Shared normalization and tokenization for document extractors.

Provides utilities for:
- Tokenizing code/text into symbol-relevant tokens
- Normalizing names for fuzzy matching (camelCase → snake_case)
- Generating URL-safe slugs from heading text
- Stripping common file suffixes from paths
- Resolving symbols and creating mentions in the graph
"""

from __future__ import annotations

import re
from typing import Any

from ...core.edge import Edge, EdgeKind
from ...core.id import NodeId
from ...core.node import NodeKind


def resolve_symbol(graph, token: str) -> NodeId | None:
    """Resolve a symbol name to a graph node.

    Matches by exact name, qualified_name suffix, or normalized match.
    Only considers code nodes (Function, Class, Method, Type, Trait, Impl).
    """
    if len(token) < 2:
        return None

    for nid, node in graph.nodes():
        if node.kind not in (
            NodeKind.FUNCTION,
            NodeKind.CLASS,
            NodeKind.METHOD,
            NodeKind.TYPE,
            NodeKind.TRAIT,
            NodeKind.IMPL,
        ):
            continue
        if node.name == token:
            return nid
        if node.qualified_name.endswith(f"::{token}"):
            return nid
        if normalize_for_match(node.name) == normalize_for_match(token):
            return nid
    return None


def mention(graph, section_id: NodeId, token: str, confidence: float) -> None:
    """Link a token mentioned in a section to a resolved symbol.

    If the symbol exists, creates a Mentions edge immediately.
    If not, stashes the token as a pending mention for post-pass resolution.
    """
    if len(token) < 2:
        return
    target = resolve_symbol(graph, token)
    if target is not None:
        graph.add_edge(section_id, target, Edge.inferred(EdgeKind.MENTIONS, confidence))
        return
    node = graph.node_mut(section_id)
    if node is None:
        return
    key = "pending_mentions"
    if key not in node.properties:
        node.properties[key] = []
    arr = node.properties[key]
    val = [token, confidence]
    if val not in arr:
        arr.append(val)


def strip_file_suffix(s: str) -> str:
    """Strip common file suffixes from a path component."""
    for suffix in (".html", ".md", ".txt", ".htm", ".php", ".js", ".ts", ".rs", ".py"):
        if s.endswith(suffix):
            return s[: -len(suffix)]
    return s


def tokenize_code(code: str) -> list[str]:
    """Split code/text into tokens on non-identifier characters."""
    tokens: list[str] = []
    current: list[str] = []
    for c in code:
        if c.isalnum() or c in "_:.=+-":
            current.append(c)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def normalize_for_match(s: str) -> str:
    """Normalize a name for matching: lowercase, insert underscores before
    camelCase boundaries."""
    chars = list(s)
    result: list[str] = []
    for i, c in enumerate(chars):
        if c.isupper() and i > 0:
            prev = chars[i - 1]
            nxt = chars[i + 1] if i + 1 < len(chars) else None
            if prev.islower() or prev.isdigit() or (prev.isupper() and nxt and nxt.islower()):
                result.append("_")
        result.append(c.lower())
    return "".join(result)


def slugify(s: str) -> str:
    """Generate a URL-safe slug from heading text."""
    parts = []
    for c in s:
        if c.isalnum():
            parts.append(c.lower())
        else:
            parts.append("-")
    slug = "".join(parts).strip("-")
    return "-".join(p for p in slug.split("-") if p)


def resolve_mentions(graph) -> int:
    """Resolve pending mentions across all sections.

    Runs as a post-pass over the complete graph. Idempotent — never adds
    duplicate Mentions edges. Returns the number of edges added.
    """
    pending: list[tuple[NodeId, list[tuple[str, float]]]] = []
    for nid, node in graph.nodes():
        arr = node.properties.get("pending_mentions")
        if not isinstance(arr, list) or not arr:
            continue
        tokens: list[tuple[str, float]] = []
        for entry in arr:
            if isinstance(entry, list) and len(entry) >= 2:
                token = str(entry[0])
                conf = float(entry[1])
                tokens.append((token, conf))
        if tokens:
            pending.append((nid, tokens))

    # Build set of existing mentions to avoid duplicates
    existing: set[tuple[NodeId, NodeId]] = set()
    for _eid, src, dst, edge in graph.edges():
        if edge.kind == EdgeKind.MENTIONS:
            existing.add((src, dst))

    added = 0
    for section_id, tokens in pending:
        for token, confidence in tokens:
            target = resolve_symbol(graph, token)
            if target is None:
                continue
            if (section_id, target) in existing:
                continue
            graph.add_edge(section_id, target, Edge.inferred(EdgeKind.MENTIONS, confidence))
            existing.add((section_id, target))
            added += 1
        node = graph.node_mut(section_id)
        if node is not None:
            node.properties.pop("pending_mentions", None)
    return added
