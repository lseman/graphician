"""Token savings quantification.

Computes how many tokens Graphician saves by returning graph-structured
context (qualified names + kind + line refs) instead of raw file content.

Both sides of the comparison are measured directly from the graph and the
files on disk — not estimated as a fixed percentage.

Mirrors the Rust ``token_savings.rs`` module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def approx_tokens(s: str) -> int:
    """Approximate token count for a string.

    Uses the same len / 4 heuristic used elsewhere in the response layer.

    Args:
        s: Input string.

    Returns:
        Approximate token count (minimum 1).
    """
    return max(len(s) // 4, 1)


class FileTokens:
    """Per-file token counts."""

    def __init__(self, raw: int, graph: int) -> None:
        self.raw = raw
        self.graph = graph


def per_file_tokens(graph) -> dict[str, FileTokens]:
    """Compute per-file token counts for the graph's indexed files.

    For each source file referenced by the graph, sums the raw file size
    in tokens and the tokens Graphician would emit describing that file's
    symbols (qualified name + kind + line range per node, no source body).

    Args:
        graph: The code graph.

    Returns:
        Dict mapping file paths to FileTokens objects.
    """
    by_file: dict[str, int] = {}

    for _, node in graph.nodes():
        uri = node.source_uri
        if not uri:
            continue

        # What a symbol-level response actually sends: qualified name, kind
        # tag, and line numbers — not the source body.
        kind_str = str(node.kind)
        line_ref = ""
        if node.line_start is not None and node.line_end is not None:
            line_ref = f"{node.line_start}-{node.line_end}"

        entry_tokens = approx_tokens(node.qualified_name) + approx_tokens(kind_str) + approx_tokens(line_ref)
        by_file[uri] = by_file.get(uri, 0) + entry_tokens

    result: dict[str, FileTokens] = {}
    for uri, graph_tokens in by_file.items():
        raw = 0
        try:
            if os.path.isfile(uri):
                content = Path(uri).read_text(encoding="utf-8", errors="replace")
                raw = approx_tokens(content)
        except OSError:
            pass
        result[uri] = FileTokens(raw=raw, graph=graph_tokens)

    return result


def token_savings_for_graph(
    graph,
    mode: str = "overview",
    include_files: bool = False,
) -> dict[str, Any]:
    """Compute token savings for the graph's indexed files.

    Args:
        graph: The code graph.
        mode: "overview" (default) or "cli" (adds a box-drawing panel).
        include_files: Add per-file breakdown (capped at 20 files).

    Returns:
        Token savings dict with totals and optionally per-file details.
    """
    files = per_file_tokens(graph)

    raw_tokens = sum(f.raw for f in files.values())
    graph_tokens = sum(f.graph for f in files.values())

    savings_percent = (
        (raw_tokens - min(graph_tokens, raw_tokens)) / raw_tokens * 100.0
        if raw_tokens > 0
        else 0.0
    )
    multiplier = raw_tokens / graph_tokens if graph_tokens > 0 else 0.0

    result: dict[str, Any] = {
        "operation": "token_savings",
        "mode": mode,
        "raw_context_tokens": raw_tokens,
        "graph_context_tokens": graph_tokens,
        "savings_percent": f"{savings_percent:.1f}%",
        "savings_multiplier": f"{multiplier:.1f}x",
        "files_measured": len(files),
    }

    if include_files:
        file_details = sorted(files.items(), key=lambda x: x[0])
        result["files"] = [
            {
                "file": uri,
                "raw_tokens": ft.raw,
                "graph_tokens": ft.graph,
            }
            for uri, ft in file_details[:20]
        ]

    if mode == "cli":
        result["panel"] = format_panel(raw_tokens, graph_tokens, savings_percent)

    return result


def format_panel(raw_tokens: int, graph_tokens: int, savings_percent: float) -> str:
    """Format a box-drawing CLI panel for token savings.

    Args:
        raw_tokens: Full context token count.
        graph_tokens: Graph context token count.
        savings_percent: Savings as a percentage.

    Returns:
        Formatted panel string.
    """
    saved = max(raw_tokens - graph_tokens, 0)
    panel = (
        "\u250c" + "\u2500" * 58 + "\u2510\n"
        "\u2502" + " Token Savings".ljust(56) + "\u2502\n"
        "\u251c" + "\u2500" * 58 + "\u2524\n"
        "\u2502" + f" Full context would be: {raw_tokens:>12} tokens".ljust(56) + "\u2502\n"
        "\u2502" + f" Graph context used: {graph_tokens:>12} tokens".ljust(56) + "\u2502\n"
        "\u2502" + f" Saved: {saved:>12} tokens ({savings_percent:.0f}%)".ljust(56) + "\u2502\n"
        "\u2514" + "\u2500" * 58 + "\u2518"
    )
    return panel
