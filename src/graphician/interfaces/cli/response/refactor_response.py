"""Rename preview: shows what would change when renaming a symbol.

Mirrors the Rust ``refactor_response.rs`` module.
"""

from __future__ import annotations

from typing import Any


def rename_preview_json(graph, target: str, new_name: str) -> dict[str, Any] | None:
    """Preview a symbol rename with all edit sites and confidence scores.

    Args:
        graph: The code graph.
        target: Qualified name of the symbol to rename.
        new_name: New name for the symbol.

    Returns:
        Rename preview with edits and stats, or None if target not found.
    """
    from ....analysis.structure import rename_preview as _rename_preview

    result = _rename_preview(graph, target, new_name)
    if not result:
        return None

    edits: list[dict[str, Any]] = []
    for e in result.get("edits", []):
        confidence = e.get("confidence", "medium")
        conf_str = str(confidence).lower()
        edits.append({
            "file": e.get("file", ""),
            "line": e.get("line", 0),
            "old": e.get("old", e.get("old_name", "")),
            "new": e.get("new", new_name),
            "confidence": conf_str,
        })

    stats = result.get("stats", {})
    total = stats.get("total", len(edits))
    return {
        "operation": "rename_preview",
        "target": result.get("target_qname", target),
        "target_name": result.get("target_name", result.get("target_name", "")),
        "new_name": new_name,
        "target_kind": result.get("target_kind", ""),
        "edits": edits,
        "stats": {
            "high": stats.get("high", 0),
            "medium": stats.get("medium", 0),
            "low": stats.get("low", 0),
            "total": total,
        },
    }


def rename_preview_handler(graph, params: dict[str, Any]) -> dict[str, Any]:
    """Handler for rename_preview operation."""
    target = params.get("target", params.get("qname", ""))
    new_name = params.get("new_name", "")

    if not target or not new_name:
        return {
            "operation": "rename_preview",
            "error": "Requires 'target' and 'new_name' parameters",
        }

    result = rename_preview_json(graph, target, new_name)
    if result is None:
        return {
            "operation": "rename_preview",
            "error": f"Node not found: {target}",
        }
    return result
