"""Snapshot-to-snapshot graph diffs.

Compares two independent graph snapshots (e.g. two separate databases)
by qualified name. Reports added/removed/modified nodes and edges, plus
community membership changes.

Mirrors the Rust ``snapshot_diff.rs`` module.
"""

from __future__ import annotations

import logging
from typing import Any

from ....persistence.store import GraphStore

logger = logging.getLogger(__name__)


def snapshot_diff_json(db_path: str, head_db: str, top: int = 50) -> dict[str, Any]:
    """Diff two graph snapshots loaded from separate database files.

    Args:
        db_path: Path to the base database.
        head_db: Path to the head database.
        top: Maximum items per diff category.

    Returns:
        Diff result with added/removed/modified nodes and edges.
    """
    from ....analysis.diff import graph_diff as _diff

    try:
        base_store = GraphStore(db_path)
        head_store = GraphStore(head_db)

        base = base_store.load_graph()
        head = head_store.load_graph()

        result = _diff(base, head)

        # result should have: added_nodes, removed_nodes, modified_nodes,
        # added_edges, removed_edges, community_changes
        added_nodes = result.get("added_nodes", [])[:top]
        removed_nodes = result.get("removed_nodes", [])[:top]
        modified_nodes = result.get("modified_nodes", [])[:top]
        added_edges = result.get("added_edges", [])[:top]
        removed_edges = result.get("removed_edges", [])[:top]
        community_changes = result.get("community_changes", [])[:top]

        return {
            "operation": "snapshot_diff",
            "base_db": str(db_path),
            "head_db": str(head_db),
            "added_nodes": added_nodes,
            "removed_nodes": removed_nodes,
            "modified_nodes": modified_nodes,
            "added_edges": added_edges,
            "removed_edges": removed_edges,
            "community_changes": community_changes,
            "counts": {
                "added_nodes": len(result.get("added_nodes", [])),
                "removed_nodes": len(result.get("removed_nodes", [])),
                "modified_nodes": len(result.get("modified_nodes", [])),
                "added_edges": len(result.get("added_edges", [])),
                "removed_edges": len(result.get("removed_edges", [])),
                "community_changes": len(result.get("community_changes", [])),
            },
        }
    except Exception as e:  # noqa: BLE001 -- snapshot diff must return an error, not crash
        logger.warning("snapshot_diff failed: %s", e)
        return {
            "operation": "snapshot_diff",
            "error": str(e),
            "base_db": str(db_path),
            "head_db": str(head_db),
            "added_nodes": [],
            "removed_nodes": [],
            "modified_nodes": [],
            "added_edges": [],
            "removed_edges": [],
            "community_changes": [],
            "counts": {
                "added_nodes": 0,
                "removed_nodes": 0,
                "modified_nodes": 0,
                "added_edges": 0,
                "removed_edges": 0,
                "community_changes": 0,
            },
        }
