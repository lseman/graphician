"""Jedi-based enrichment for Python call resolution.

After tree-sitter parsing, many method calls on lowercase-receiver
variables are dropped (e.g. ``svc.authenticate()`` where
``svc = factory()``). This module uses the ``jedi`` Python library
to resolve these by tracing return types across files.

Runs as a post-build step: walks Python ASTs to find dropped calls,
uses ``jedi.Script.goto()`` to resolve them, and adds CALLS edges.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from ...core.edge import EdgeKind
from ...core.graph import Graph
from ...core.node import NodeKind
from .parse import parse_jedi_results
from .scan import find_dropped_calls
from .script_gen import build_jedi_script

# Re-export for backward compatibility
__all__ = ["enrich_jedi_calls", "jedi_available"]

logger = logging.getLogger(__name__)


def enrich_jedi_calls(graph: Graph, repo_root: Path) -> int:
    """Resolve untracked Python method calls via Jedi.

    Walks Python files, finds ``receiver.method()`` calls that tree-sitter
    dropped (lowercase receiver, not self/cls), resolves them with Jedi,
    and inserts new CALLS edges into the graph.

    Returns the number of calls resolved (CALLS edges added).

    Args:
        graph: The code graph to enrich.
        repo_root: Project root directory for Jedi project resolution.

    Raises:
        RuntimeError: If Jedi is not installed and subprocess fails.
    """
    # Collect Python file source URIs from the graph.
    py_files: list[str] = [
        node.source_uri
        for _, node in graph.nodes()
        if node.source_uri and node.source_uri.endswith(".py")
    ]

    if not py_files:
        return 0

    # Collect function/method nodes for enclosing-function lookup.
    func_nodes: list[tuple[str, int, int]] = [
        (
            node.qualified_name,
            node.line_start or 0,
            node.line_end or sys.maxsize,
        )
        for _, node in graph.nodes()
        if node.kind in (NodeKind.FUNCTION, NodeKind.METHOD)
    ]

    # Collect existing CALLS edges to avoid duplicates.
    existing_calls: set[tuple[str, int]] = set()
    for _, src_node, dst_node, _ in _iter_edges(graph, EdgeKind.CALLS):
        if src_node and src_node.line_start is not None:
            existing_calls.add((src_node.qualified_name, src_node.line_start))

    # Collect project function names to filter out calls to names
    # that don't exist in the project.
    project_func_names: set[str] = {
        node.name
        for _, node in graph.nodes()
        if node.kind in (NodeKind.FUNCTION, NodeKind.METHOD, NodeKind.CLASS)
        and not node.qualified_name.startswith("call::")
    }

    # Find dropped method calls in each Python file.
    pending_calls: list[tuple[str, int, int, str, str]] = []
    for file_path in py_files:
        try:
            source = Path(file_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        dropped = find_dropped_calls(source, func_nodes)
        for line, col, method_name, enclosing in dropped:
            # Only keep calls whose method name exists in project code.
            if method_name in project_func_names:
                pending_calls.append((file_path, line, col, method_name, enclosing))

    if not pending_calls:
        return 0

    # Build and execute the Jedi resolution script.
    py_source = build_jedi_script(pending_calls, repo_root)

    try:
        output = subprocess.run(
            [sys.executable, "-c", py_source],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        logger.debug("Python interpreter not found, skipping Jedi enrichment")
        return 0
    except subprocess.TimeoutExpired:
        logger.warning("Jedi enrichment timed out after 30s")
        return 0

    if output.returncode != 0:
        stderr = output.stderr or ""
        if "ModuleNotFoundError" in stderr or "jedi" in stderr.lower():
            logger.debug("Jedi not installed, skipping Python enrichment")
            return 0
        logger.warning("Jedi enrichment failed: %s", stderr)
        raise RuntimeError(f"Jedi enrichment failed: {stderr}")

    stdout = output.stdout or ""
    return parse_jedi_results(stdout, graph, existing_calls)


def jedi_available() -> bool:
    """Check whether Jedi enrichment is available (Jedi installed).

    Returns True if the ``jedi`` Python module can be imported.
    """
    try:
        import jedi  # noqa: F401

        return True
    except ImportError:
        return False


# ── Internal helpers ───────────────────────────────────────────────

def _iter_edges(
    graph: Graph,
    kind: EdgeKind,
) -> list[tuple[Any, Any, Any, Any]]:
    """Iterate all edges in the graph, yielding (edge_id, src, dst, edge).

    Wrapper to avoid importing internal Graph details.
    """
    return list(graph.edges()) if hasattr(graph, "edges") else []
