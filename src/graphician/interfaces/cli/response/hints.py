"""Context-aware hints system for tool responses.

Tracks session state (in-memory) and generates intelligent next-step
suggestions after each tool call. Hints are appended as ``_hints`` to
responses so the LLM can propose follow-up actions without the user
having to discover them.

Pattern: after ``detect_changes`` → suggest ``review_context``,
``affected_flows``, ``blast_radius``. After ``search`` → suggest
``traverse``, ``impact``, ``paths``. Etc.
"""

from __future__ import annotations

from typing import Any

# Workflow adjacency: for each operation, what are useful next steps
_WORKFLOW: dict[str, list[tuple[str, str]]] = {
    "search": [
        ("traverse", "Walk the call graph around a matched symbol"),
        ("impact", "Check the blast radius of a matched symbol"),
        ("paths", "Find call paths between a matched symbol and another"),
        ("flows", "See which execution flows contain the result"),
    ],
    "traverse": [
        ("search", "Semantic search across the graph"),
        ("impact", "Check how much a traversed symbol affects the codebase"),
        ("flows", "See execution flows through traversed nodes"),
    ],
    "paths": [
        ("search", "Find symbols related to a path endpoint"),
        ("traverse", "BFS/DFS between two symbols"),
        ("impact", "Check the impact of nodes along a path"),
    ],
    "flows": [
        ("search", "Search for symbols in these flows"),
        ("affected_flows", "Check which flows are affected by recent changes"),
        ("impact", "Check the impact of nodes in a flow"),
    ],
    "affected_flows": [
        ("detect_changes", "Get risk-scored change analysis"),
        ("review_context", "Build a review context with source snippets"),
        ("blast_radius", "Expand the impact analysis"),
    ],
    "blast_radius": [
        ("detect_changes", "Get risk-scored change analysis"),
        ("review_context", "Build a review context with source snippets"),
        ("test_coverage", "Check which tests cover impacted nodes"),
    ],
    "detect_changes": [
        ("review_context", "Build a review context with source snippets"),
        ("affected_flows", "See which execution flows are affected"),
        ("blast_radius", "Expand the impact analysis"),
        ("test_coverage", "Check test coverage gaps in changed code"),
    ],
    "review_context": [
        ("test_coverage", "Check test coverage gaps"),
        ("affected_flows", "See which flows are affected"),
        ("suggested_questions", "Get AI-generated follow-up questions"),
    ],
    "impact": [
        ("search", "Find symbols related to an impacted node"),
        ("traverse", "Walk the call graph around an impacted node"),
        ("test_coverage", "Check test coverage for impacted nodes"),
        ("gaps", "Find structural weaknesses in impacted areas"),
    ],
    "god_nodes": [
        ("large_functions", "Find other large functions in god nodes"),
        ("bridge_nodes", "Check if god nodes are also bridge nodes"),
        ("gaps", "Find structural weaknesses near god nodes"),
    ],
    "large_functions": [
        ("impact", "Check the impact of large functions"),
        ("gaps", "Find structural weaknesses near large functions"),
        ("test_coverage", "Check test coverage of large functions"),
    ],
    "architecture_overview": [
        ("bridge_nodes", "Find cross-community bridge nodes"),
        ("cycles", "Detect dependency cycles"),
        ("surprises", "Find surprising cross-community connections"),
        ("gaps", "Find structural weaknesses"),
    ],
    "bridge_nodes": [
        ("impact", "Check the impact of bridge nodes"),
        ("architecture_overview", "See the broader architecture"),
        ("surprises", "Find surprising connections through bridges"),
    ],
    "gaps": [
        ("impact", "Check the impact of gap nodes"),
        ("architecture_overview", "See the broader architecture"),
        ("bridge_nodes", "Check for bridge nodes near gaps"),
    ],
    "surprises": [
        ("architecture_overview", "See the broader architecture"),
        ("bridge_nodes", "Check bridge nodes around surprising connections"),
        ("impact", "Check the impact of surprising nodes"),
    ],
    "diagnostics": [
        ("search", "Search for symbols in the graph"),
        ("architecture_overview", "See the architecture"),
        ("flows", "Explore execution flows"),
    ],
    "test_coverage": [
        ("large_functions", "Find uncovered large functions"),
        ("impact", "Check the impact of uncovered nodes"),
        ("detect_changes", "See what recently changed in tested areas"),
    ],
    "report": [
        ("architecture_overview", "See the architecture"),
        ("detect_changes", "Check recent changes"),
        ("diagnostics", "Check graph health"),
    ],
    "cycles": [
        ("impact", "Check the impact of cyclic nodes"),
        ("architecture_overview", "See how cycles affect the architecture"),
        ("bridge_nodes", "Check if cycles involve bridge nodes"),
    ],
    "core": [
        ("bridge_nodes", "Check bridge nodes near core nodes"),
        ("gaps", "Find gaps near core nodes"),
        ("surprises", "Find surprising connections near core nodes"),
    ],
    "articulation": [
        ("impact", "Check the impact of articulation points"),
        ("gaps", "Find gaps near articulation points"),
        ("architecture_overview", "See articulation in architecture"),
    ],
    "suggested_questions": [
        ("detect_changes", "Get more detailed change analysis"),
        ("review_context", "Build a review context"),
        ("affected_flows", "See affected flows"),
    ],
    "graph_diff": [
        ("detect_changes", "Get risk-scored change analysis"),
        ("review_context", "Build a review context"),
        ("suggested_questions", "Get AI-generated follow-up questions"),
    ],
    "counterfactual": [
        ("impact", "Check the impact of removed nodes"),
        ("affected_flows", "See affected flows from counterfactual"),
    ],
    "motifs": [
        ("architecture_overview", "See how motifs fit the architecture"),
        ("surprises", "Find surprising motif instances"),
    ],
    "status": [
        ("diagnostics", "Get detailed graph health"),
        ("architecture_overview", "See the architecture"),
        ("search", "Start querying the graph"),
    ],
}

_MAX_PER_CATEGORY = 3
_MAX_TOOLS_HISTORY = 100


class SessionState:
    """Tracks session history for hint generation."""

    def __init__(self) -> None:
        self.tools_called: list[str] = []
        self.nodes_queried: set[str] = set()
        self.files_touched: set[str] = set()

    def record_tool_call(self, tool_name: str) -> None:
        self.tools_called.append(tool_name)
        if len(self.tools_called) > _MAX_TOOLS_HISTORY:
            self.tools_called.pop(0)

    def record_nodes(self, qnames: list[str]) -> None:
        for qn in qnames:
            if len(self.nodes_queried) < 1000:
                self.nodes_queried.add(qn)

    def record_files(self, files: list[str]) -> None:
        for f in files:
            if len(self.files_touched) < 500:
                self.files_touched.add(f)


def generate_hints(tool_name: str, result: dict[str, Any]) -> dict[str, Any] | None:
    """Generate hints for a tool response.

    Args:
        tool_name: The operation/tool that was called.
        result: The response dict from the tool.

    Returns:
        Hints dict with next_steps, related, warnings, or None if empty.
    """
    state = SessionState()
    state.record_tool_call(tool_name)

    next_steps = _build_next_steps(tool_name, state)
    warnings = _extract_warnings(result)
    related = _build_related(tool_name, result, state)

    _track_result(result, state)

    hints: dict[str, Any] = {}
    if next_steps:
        hints["next_steps"] = next_steps
    if related:
        hints["related"] = related
    if warnings:
        hints["warnings"] = warnings

    return hints if hints else None


def _build_next_steps(tool_name: str, session: SessionState) -> list[dict[str, str]]:
    """Build next-step suggestions from the workflow adjacency table."""
    called = set(session.tools_called)
    candidates = _WORKFLOW.get(tool_name, [])

    out = []
    for next_tool, suggestion in candidates:
        if next_tool not in called:
            out.append({"tool": next_tool, "suggestion": suggestion})
            if len(out) >= _MAX_PER_CATEGORY:
                break
    return out


def _extract_warnings(result: dict[str, Any]) -> list[str]:
    """Extract warnings from the response result."""
    warnings: list[str] = []

    # Test gaps
    test_gaps = result.get("test_gaps")
    if isinstance(test_gaps, list):
        names = []
        for g in test_gaps[:5]:
            if isinstance(g, str):
                names.append(g)
            elif isinstance(g, dict):
                name = g.get("name", "")
                if name:
                    names.append(name)
        if names:
            warnings.append(f"Test coverage gaps: {', '.join(names)}")

    # Risk score
    risk = result.get("risk_score")
    if isinstance(risk, (int, float)) and risk > 0.7:
        warnings.append(f"High risk score ({risk:.2f}) — review carefully")

    # Architecture warnings
    arch_warnings = result.get("warnings")
    if isinstance(arch_warnings, list):
        for w in arch_warnings[:3]:
            if isinstance(w, str):
                warnings.append(w)
            elif isinstance(w, dict):
                msg = w.get("message", "")
                if msg:
                    warnings.append(msg)

    return warnings


def _build_related(
    _tool_name: str, result: dict[str, Any], session: SessionState
) -> list[str]:
    """Suggest files/nodes not yet touched."""
    related: list[str] = []
    seen: set[str] = set()

    impacted = result.get("impacted_files") or result.get("changed_files")
    if isinstance(impacted, list):
        for item in impacted:
            if (
                isinstance(item, str)
                and item not in session.files_touched
                and item not in seen
            ):
                related.append(item)
                seen.add(item)
                if len(related) >= _MAX_PER_CATEGORY:
                    break
    return related


def _track_result(result: dict[str, Any], session: SessionState) -> None:
    """Extract files and nodes from result for session tracking."""
    # Files
    for key in ("changed_files", "impacted_files", "files"):
        items = result.get(key)
        if isinstance(items, list):
            files = [f for f in items if isinstance(f, str)]
            session.record_files(files)

    # Nodes
    qnames: list[str] = []
    for key in ("results", "changed_nodes", "impacted_nodes", "nodes", "nodes_list"):
        items = result.get(key)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    qn = item.get("qualified_name", "")
                    if qn:
                        qnames.append(qn)
    if qnames:
        session.record_nodes(qnames)
