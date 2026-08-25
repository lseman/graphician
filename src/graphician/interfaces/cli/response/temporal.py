"""Temporal diff and risk scoring for change analysis.

Mirrors the Rust ``temporal.rs`` and ``differential.rs`` modules.
Provides:
- ``detect_changes_json`` — risk-scored change analysis
- ``risk_json`` — multi-factor risk assessments
- ``temporal_diff`` — node/edge diff by validity windows
- ``graph_diff_json`` — multi-snapshots diff (stub)
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ....core.edge import EdgeKind
from ....core.id import EdgeId, NodeId

logger = logging.getLogger(__name__)


# ── Temporal diff types (mirrors differential.rs) ──────────────────


@dataclass
class ChangedEdge:
    """An edge that was added or removed between two commits."""
    id: EdgeId
    src: NodeId
    dst: NodeId
    edge_kind: EdgeKind
    change: str  # "added" or "removed"


@dataclass
class TemporalDiff:
    """Result of a temporal diff between two commits."""
    added_nodes: list[NodeId] = field(default_factory=list)
    removed_nodes: list[NodeId] = field(default_factory=list)
    added_edges: list[ChangedEdge] = field(default_factory=list)
    removed_edges: list[ChangedEdge] = field(default_factory=list)

    def is_empty(self) -> bool:
        return (
            not self.added_nodes
            and not self.removed_nodes
            and not self.added_edges
            and not self.removed_edges
        )

    def changed_nodes(self) -> list[NodeId]:
        """All nodes involved in any change (added, removed, or edge-touched)."""
        nodes: list[NodeId] = []
        seen: set[int] = set()

        for id_ in self.added_nodes + self.removed_nodes:
            if id_.value not in seen:
                nodes.append(id_)
                seen.add(id_.value)

        for edge in self.added_edges + self.removed_edges:
            for nid in (edge.src, edge.dst):
                if nid.value not in seen:
                    nodes.append(nid)
                    seen.add(nid.value)

        nodes.sort(key=lambda id_: id_.value)
        return nodes


# ── Core temporal diff (mirrors differential.rs) ───────────────────


def is_active_at(
    valid_from: str | None,
    valid_to: str | None,
    commit: str,
    is_ancestor: Callable[[str, str], bool],
) -> bool:
    """Check if an entity is active at a given commit.

    Mirrors the Rust ``is_active_at`` from ``differential.rs``.
    """
    if valid_from is not None and valid_from != commit and not is_ancestor(valid_from, commit):
        return False
    return not (valid_to is not None and valid_to != commit and is_ancestor(valid_to, commit))


def temporal_diff(
    graph,
    base: str,
    head: str,
    is_ancestor: Callable[[str, str], bool],
) -> TemporalDiff:
    """Compute temporal diff between two commits.

    Mirrors the Rust ``temporal_diff`` from ``differential.rs``.
    Every node and edge carries ``valid_from`` and ``valid_to`` SHA columns.
    This classifies graph entities by their validity windows.

    Args:
        graph: The code graph.
        base: Base commit SHA/ref.
        head: Head commit SHA/ref.
        is_ancestor: Callback to check if commit A is an ancestor of B.

    Returns:
        TemporalDiff with added/removed nodes and edges.
    """
    diff = TemporalDiff()

    # Node diff
    for nid, node in graph.nodes():
        active_at_base = is_active_at(
            node.valid_from, node.valid_to, base, is_ancestor
        )
        active_at_head = is_active_at(
            node.valid_from, node.valid_to, head, is_ancestor
        )
        if (not active_at_base) and active_at_head:
            diff.added_nodes.append(nid)
        elif active_at_base and not active_at_head:
            diff.removed_nodes.append(nid)

    # Edge diff
    for eid, src, dst, edge in graph.edges():
        active_at_base = is_active_at(
            edge.valid_from, edge.valid_to, base, is_ancestor
        )
        active_at_head = is_active_at(
            edge.valid_from, edge.valid_to, head, is_ancestor
        )
        change = "added" if active_at_head else "removed"
        changed = ChangedEdge(
            id=eid,
            src=src,
            dst=dst,
            edge_kind=edge.kind,
            change=change,
        )
        if (not active_at_base) and active_at_head:
            diff.added_edges.append(changed)
        elif active_at_base and not active_at_head:
            diff.removed_edges.append(changed)

    # Sort by ID for deterministic output
    diff.added_nodes.sort(key=lambda id_: id_.value)
    diff.removed_nodes.sort(key=lambda id_: id_.value)
    diff.added_edges.sort(key=lambda e: e.id.value)
    diff.removed_edges.sort(key=lambda e: e.id.value)

    return diff


# ── High-level API ─────────────────────────────────────────────────


def detect_changes_json(
    graph,
    base: str,
    max_depth: int = 2,
    top: int = 25,
) -> dict[str, Any]:
    """Risk-scored change analysis from a git diff base.

    Since we may not have git context, this is a simplified version
    that returns a structural analysis.

    Args:
        graph: The code graph.
        base: Git base ref (e.g. "HEAD~1").
        max_depth: Max BFS hops for impact analysis.
        top: Max results to return.

    Returns:
        Analysis dict with changed symbols, impact, risk score.
    """
    # Placeholder: without git context, return structural analysis
    # In production, this would call git diff + temporal diff
    return {
        "operation": "detect_changes",
        "base": base,
        "changed_files": [],
        "changed_ranges": [],
        "changed_symbols": [],
        "changed_nodes": [],
        "temporal": None,
        "impacted": [],
        "test_coverage": {"covered": [], "missing": [], "missing_count": 0},
        "affected_flows": {"hits": [], "total": 0, "truncated": False},
        "risk_score": 0.0,
        "risk": "low",
        "risk_assessments": [],
        "mapping_precision": "none",
        "suggested_next_tools": [
            "review_context",
            "impact",
            "traverse",
            "suggested_questions",
        ],
    }


def risk_json(
    graph,
    base: str,
    top: int = 25,
) -> dict[str, Any]:
    """Multi-factor risk assessments for symbols changed since base.

    Args:
        graph: The code graph.
        base: Git base ref.
        top: Max assessments to return.

    Returns:
        Risk assessment dict.
    """
    analysis = detect_changes_json(graph, base, 2)
    assessments = analysis.get("risk_assessments", [])[:top]

    return {
        "operation": "risk",
        "base": base,
        "hits": assessments,
        "changed_files": analysis.get("changed_files", []),
        "mapping_precision": analysis.get("mapping_precision", "none"),
        "suggested_next_tools": ["review_context", "test_coverage", "affected_flows"],
    }


def graph_diff_json(
    graph,
    base: str,
    head: str = "HEAD",
    top: int = 50,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Graph diff between two commits via temporal validity.

    Checks for temporal data (valid_from/valid_to fields) and runs
    temporal_diff to classify added/removed nodes and edges.

    Args:
        graph: The code graph with temporal metadata.
        base: Base git ref/SHA.
        head: Head git ref/SHA.
        top: Max nodes/edges to return.

    Returns:
        Diff dict with added/removed nodes and edges.
    """
    # Check for temporal data (valid_from/valid_to fields)
    has_temporal = any(
        node.valid_from is not None or node.valid_to is not None
        for _, node in graph.nodes()
    ) or any(
        edge.valid_from is not None or edge.valid_to is not None
        for _, _, _, edge in graph.edges()
    )

    if not has_temporal:
        return {
            "operation": "graph_diff",
            "base": base,
            "head": head,
            "error": (
                "graph has no temporal data; rebuild with git context "
                "(run `build` inside a git repo)"
            ),
        }

    def is_ancestor_fn(ancestor: str, descendant: str) -> bool:
        process = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=repo_root,
            check=False,
            capture_output=True,
        )
        return process.returncode == 0

    diff = temporal_diff(graph, base, head, is_ancestor_fn)

    return {
        "operation": "graph_diff",
        "base": base,
        "head": head,
        "added_nodes": _nodes_json(graph, diff.added_nodes, top),
        "removed_nodes": _nodes_json(graph, diff.removed_nodes, top),
        "added_edges": _edges_json(diff.added_edges, graph, top),
        "removed_edges": _edges_json(diff.removed_edges, graph, top),
    }


# ── Differential (temporal) JSON response ─────────────────────────


def differential_json(
    graph,
    base: str,
    head: str = "HEAD",
    top: int = 50,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Compute a temporal (differential) diff between two commits.

    Mirrors the Rust ``differential.rs`` module.
    Every node and edge carries ``valid_from`` and ``valid_to`` SHA columns.
    This classifies graph entities by their validity windows.

    Args:
        graph: The code graph with temporal metadata.
        base: Base git ref/SHA.
        head: Head git ref/SHA.
        top: Max nodes/edges to return.
        repo_root: Git repository root (default: current working directory).

    Returns:
        Diff dict with added/removed nodes and edges, plus summary stats.
    """
    # Check for temporal data (valid_from/valid_to fields)
    has_temporal = any(
        node.valid_from is not None or node.valid_to is not None
        for _, node in graph.nodes()
    ) or any(
        edge.valid_from is not None or edge.valid_to is not None
        for _, _, _, edge in graph.edges()
    )

    if not has_temporal:
        return {
            "operation": "differential",
            "base": base,
            "head": head,
            "error": (
                "graph has no temporal data; rebuild with git context "
                "(run `build` inside a git repo)"
            ),
            "added_nodes": [],
            "removed_nodes": [],
            "added_edges": [],
            "removed_edges": [],
            "summary": {
                "total_added_nodes": 0,
                "total_removed_nodes": 0,
                "total_added_edges": 0,
                "total_removed_edges": 0,
            },
        }

    def is_ancestor_fn(ancestor: str, descendant: str) -> bool:
        process = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=repo_root,
            check=False,
            capture_output=True,
        )
        return process.returncode == 0

    diff = temporal_diff(graph, base, head, is_ancestor_fn)

    # Compute summary stats
    summary = {
        "total_added_nodes": len(diff.added_nodes),
        "total_removed_nodes": len(diff.removed_nodes),
        "total_added_edges": len(diff.added_edges),
        "total_removed_edges": len(diff.removed_edges),
        "changed_nodes_count": len(diff.changed_nodes()),
    }

    return {
        "operation": "differential",
        "base": base,
        "head": head,
        "added_nodes": _nodes_json(graph, diff.added_nodes, top),
        "removed_nodes": _nodes_json(graph, diff.removed_nodes, top),
        "added_edges": _edges_json(diff.added_edges, graph, top),
        "removed_edges": _edges_json(diff.removed_edges, graph, top),
        "summary": summary,
    }


# ── Helpers ────────────────────────────────────────────────────────


def _nodes_json(graph, ids: list, limit: int = 50) -> list[dict[str, Any]]:
    """Convert node IDs to JSON dicts."""
    out = []
    for node_id in ids[:limit]:
        node = graph.node(node_id) if hasattr(graph, "node") else None
        if node is None:
            # Try to find by iterating
            for _, n in graph.nodes():
                if str(n.__hash__()) == str(node_id) or (
                    hasattr(n, "id") and n.id == node_id  # type: ignore[attr-defined]
                ):
                    node = n
                    break
        if node:
            out.append({
                "qualified_name": node.qualified_name,
                "kind": str(node.kind),
                "source_uri": node.source_uri,
                "line_start": node.line_start,
                "line_end": node.line_end,
            })
    return out


def _edges_json(
    edges: list[ChangedEdge],
    graph,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Convert ChangedEdge list to JSON dicts."""
    out = []
    for edge in edges[:limit]:
        src_node = graph.node(edge.src) if hasattr(graph, "node") else None
        dst_node = graph.node(edge.dst) if hasattr(graph, "node") else None
        out.append({
            "kind": str(edge.edge_kind),
            "change": edge.change,
            "src": src_node.qualified_name if src_node else None,
            "dst": dst_node.qualified_name if dst_node else None,
            "source_uri": (
                src_node.source_uri
                if src_node and src_node.source_uri
                else (dst_node.source_uri if dst_node else None)
            ),
        })
    return out
