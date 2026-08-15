"""Change detection: parse diffs and find affected symbols."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from ...core.edge import EdgeKind
from ...core.graph import Graph
from ...core.id import NodeId
from ...core.node import NodeKind
from .types import Change

logger = logging.getLogger(__name__)


@dataclass
class DiffHunk:
    """A single hunk from a Git diff."""
    file_path: str
    line_start: int
    line_end: int
    change_type: str
    content: str


def _parse_diff(diff_text: str) -> list[DiffHunk]:
    """Parse a Git diff into hunks."""
    hunks: list[DiffHunk] = []
    current_file = None
    current_hunk_start = None
    current_lines: list[str] = []
    change_type = "modified"

    for line in diff_text.split("\n"):
        if line.startswith("diff --git"):
            # Save previous hunk
            if current_file and current_hunk_start and current_lines:
                hunks.append(DiffHunk(
                    file_path=current_file,
                    line_start=current_hunk_start,
                    line_end=current_hunk_start + len(current_lines),
                    change_type=change_type,
                    content="\n".join(current_lines),
                ))
            current_file = None
            current_hunk_start = None
            current_lines = []

            # Extract file path
            match = re.search(r'diff --git a/(.+) b/(.+)', line)
            if match:
                old = match.group(1)
                new = match.group(2)
                if old == new:
                    current_file = old
                else:
                    current_file = new

        elif line.startswith("@@"):
            # Parse hunk header
            match = re.search(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
            if match:
                current_hunk_start = int(match.group(3))
                change_type = "modified"

        elif line.startswith("+") and not line.startswith("+++"):
            current_lines.append(line)
            change_type = "added"
        elif line.startswith("-") and not line.startswith("---"):
            current_lines.append(line)
            change_type = "removed"

    # Save last hunk
    if current_file and current_hunk_start and current_lines:
        hunks.append(DiffHunk(
            file_path=current_file,
            line_start=current_hunk_start,
            line_end=current_hunk_start + len(current_lines),
            change_type=change_type,
            content="\n".join(current_lines),
        ))

    return hunks


def _find_affected_symbols(graph: Graph, hunk: DiffHunk) -> list:
    """Find symbols affected by a diff hunk."""
    affected = []

    # Look for function/class definitions in the changed region
    for nid, node in graph.nodes():
        if node.source_uri == hunk.file_path:
            if node.line_start and node.line_end:
                if node.line_start <= hunk.line_end and node.line_end >= hunk.line_start:
                    affected.append(node)

    return affected


def _compute_change_risk(graph: Graph, hunk: DiffHunk, affected: list) -> float:
    """Compute risk score for a change."""
    if not affected:
        return 0.0

    risk = 0.0
    for node in affected:
        # Functions and classes are more risky
        if node.kind in (NodeKind.FUNCTION, NodeKind.METHOD):
            risk += 1.0
        elif node.kind == NodeKind.CLASS:
            risk += 2.0

        # Check if the symbol has callers
        nid = graph.find_by_qname(node.qualified_name)
        callers = list(graph.in_neighbors(nid)) if nid is not None else []
        if callers:
            risk += 0.5

    return min(risk, 10.0)


def _find_affected_flows(
    graph: Graph,
    changes: list[Change],
) -> list[dict[str, Any]]:
    """Find flows affected by changes."""
    affected = []
    for nid, node in graph.nodes():
        if node.kind == NodeKind.FLOW:
            # Check if any member is affected
            for member, _ in graph.in_neighbors(nid):
                member_node = graph.node(member)
                if member_node:
                    for change in changes:
                        if change.affected_symbols and member_node.qualified_name in change.affected_symbols:
                            affected.append({
                                "flow": node.qualified_name,
                                "entry": node.properties.get("entry", "unknown"),
                                "affected_change": change.file_path,
                            })
                            break
    return affected[:20]


def detect_changes(
    graph: Graph,
    diff_text: str,
    base: str | None = None,
    max_depth: int = 2,
) -> dict[str, Any]:
    """Detect changes from a Git diff text.

    Maps diff hunks to graph symbols and computes risk scores.
    """
    hunks = _parse_diff(diff_text)
    changes: list[Change] = []

    for hunk in hunks:
        # Find affected symbols in the changed region
        affected = _find_affected_symbols(graph, hunk)
        risk = _compute_change_risk(graph, hunk, affected)

        change = Change(
            file_path=hunk.file_path,
            line_start=hunk.line_start,
            line_end=hunk.line_end,
            change_type=hunk.change_type,
            content=hunk.content[:200],  # Truncate for response
            affected_symbols=[s.qualified_name for s in affected],
            risk_score=risk,
        )
        changes.append(change)

    # Find affected flows
    affected_flows = _find_affected_flows(graph, changes)

    return {
        "changes": [
            {
                "file": c.file_path,
                "line_range": f"{c.line_start}-{c.line_end}",
                "type": c.change_type,
                "content": c.content,
                "affected_symbols": c.affected_symbols,
                "risk_score": round(c.risk_score, 4),
            }
            for c in changes
        ],
        "affected_flows": affected_flows,
        "total_changes": len(changes),
        "max_risk": max((c.risk_score for c in changes), default=0.0),
    }
