"""Refactoring engine: rename preview and dead code detection."""

from __future__ import annotations

from typing import Any

from ...core.edge import EdgeKind
from ...core.graph import Graph
from ...core.id import NodeId
from ...core.node import Node, NodeKind
from .types import (
    Confidence,
    RenameEdit,
    RenamePreview,
    RenameStats,
)

# ── Rename Preview ────────────────────────────────────────────────────────


def rename_preview(graph: Graph, qname: str, new_name: str) -> RenamePreview | None:
    """Preview rename of a symbol without modifying the graph.

    Analyzes the graph to find:
    1. The definition site (where the symbol is defined)
    2. All call sites (CALLS edges pointing to this node)
    3. All import sites (IMPORTS edges pointing to this node)
    4. Reference sites where the symbol name appears in neighbor names

    Returns None if the node is not found.
    """
    target_id = graph.find_by_qname(qname)
    if target_id is None:
        return None
    target_node = graph.node(target_id)
    if target_node is None:
        return None

    edits: list[RenameEdit] = []
    seen_keys: set[tuple[str | None, int | None]] = set()

    # 1. Definition site
    edits.append(RenameEdit(
        file=target_node.source_uri,
        line=target_node.line_start,
        old=target_node.name,
        new=new_name,
        confidence=Confidence.HIGH,
    ))

    # 2. Call sites — CALLS edges targeting this node
    for _, src, dst, edge in graph.edges():
        if dst == target_id and edge.kind == EdgeKind.CALLS:
            src_node = graph.node(src)
            key = (
                src_node.source_uri if src_node else None,
                src_node.line_start if src_node else None,
            )
            if key not in seen_keys:
                seen_keys.add(key)
                edits.append(RenameEdit(
                    file=key[0],
                    line=key[1],
                    old=target_node.name,
                    new=new_name,
                    confidence=Confidence.HIGH,
                ))

    # 3. Import sites — edges targeting this node
    for _, src, dst, edge in graph.edges():
        if dst == target_id and edge.kind == EdgeKind.IMPORTS:
            src_node = graph.node(src)
            key = (
                src_node.source_uri if src_node else None,
                src_node.line_start if src_node else None,
            )
            if key not in seen_keys:
                seen_keys.add(key)
                edits.append(RenameEdit(
                    file=key[0],
                    line=key[1],
                    old=target_node.name,
                    new=new_name,
                    confidence=Confidence.HIGH,
                ))

    # 4. Where target references others — check for bare-name refs
    for _, src, dst, _edge in graph.edges():
        if src == target_id and dst != target_id:
            dst_node = graph.node(dst)
            if (
                dst_node
                and target_node.name in dst_node.name
                and new_name not in dst_node.qualified_name
            ):
                key = (dst_node.source_uri, dst_node.line_start)
                if key not in seen_keys:
                    seen_keys.add(key)
                    edits.append(RenameEdit(
                        file=key[0],
                        line=key[1],
                        old=target_node.name,
                        new=new_name,
                        confidence=Confidence.MEDIUM,
                    ))

    stats = RenameStats.from_edits(edits)

    return RenamePreview(
        target_qname=target_node.qualified_name,
        target_name=target_node.name,
        new_name=new_name,
        target_kind=target_node.kind.value,
        edits=edits,
        stats=stats,
    )


# ── Dead Code Detection ───────────────────────────────────────────────────

_ENTRY_POINT_PATTERNS = [
    "main", "main_", "test_", "Test", "Handle", "handle_",
    "serve", "run", "start", "entry", "init", "setup", "new",
    "default",
]

_FRAMEWORK_SUFFIXES = [
    "Stack", "Construct", "Resource", "Pipeline", "Model",
    "BaseModel", "BaseSettings", "DeclarativeBase",
    "TableBase", "App",
]

_TEST_FILE_PATTERNS = [
    "__tests__", ".spec.", ".test.", "/test_", "/e2e_test",
    "/test_utils", "/tests/", "/test/", "tests/", "test/",
]


def is_entry_point(node: Node, patterns: list[str] | None = None) -> bool:
    """Check if a node looks like an entry point by name."""
    if patterns is None:
        patterns = _ENTRY_POINT_PATTERNS
    name = node.name
    return any(name == pattern or name.endswith(pattern) for pattern in patterns)


def is_framework_inherited(node: Node, inherited_classes: set[NodeId]) -> bool:
    """Check if a class inherits from framework base classes."""
    # Check if this node is referenced by INHERITS/IMPLEMENTS edges
    # (we check this externally; here we just check name suffixes)
    return any(node.name.endswith(suffix) for suffix in _FRAMEWORK_SUFFIXES)


def is_test_file(node: Node) -> bool:
    """Check if a node is in a test file."""
    uri = node.source_uri
    if not uri:
        return False
    lower = uri.lower()
    return any(p in lower for p in _TEST_FILE_PATTERNS)


def find_dead_code(
    graph: Graph,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Find dead code: functions/classes with no callers, no test refs, no importers.

    Entry points (functions with framework names like `main`, `handle_*`,
    `test_*`, or framework decorators) are excluded.

    Test files are also excluded since test code is not considered dead
    just because it's not called by production code.
    """
    # Build set of nodes that ARE called, imported, or referenced
    called_nodes: set[NodeId] = set()
    imported_nodes: set[NodeId] = set()
    tested_nodes: set[NodeId] = set()
    inherited_classes: set[NodeId] = set()

    for _, src, dst, edge in graph.edges():
        if edge.kind == EdgeKind.CALLS:
            called_nodes.add(src)
            called_nodes.add(dst)
        elif edge.kind == EdgeKind.IMPORTS:
            imported_nodes.add(dst)
        elif edge.kind == EdgeKind.TESTED_BY:
            tested_nodes.add(src)
            tested_nodes.add(dst)
        elif edge.kind in (
            EdgeKind.INHERITS, EdgeKind.IMPLEMENTS,
            EdgeKind.MEMBER_OF, EdgeKind.ENTRY_OF,
        ):
            inherited_classes.add(src)
            inherited_classes.add(dst)
        else:
            inherited_classes.add(src)
            inherited_classes.add(dst)

    # Filter nodes: find candidates with no callers and no references
    dead: list[tuple[NodeId, Node]] = []
    for nid, node in graph.nodes():
        # Only consider functions, methods, and classes
        if node.kind not in (NodeKind.FUNCTION, NodeKind.METHOD, NodeKind.CLASS):
            continue

        # Skip placeholder nodes
        if node.qualified_name.startswith("call::"):
            continue

        # Skip if called
        if nid in called_nodes:
            continue

        # Skip if imported
        if nid in imported_nodes:
            continue

        # Skip if tested
        if nid in tested_nodes:
            continue

        # Skip if it's an entry point
        if is_entry_point(node):
            continue

        # Skip if it inherits from framework bases
        if is_framework_inherited(node, inherited_classes):
            continue

        # Skip if it's in a test file
        if is_test_file(node):
            continue

        dead.append((nid, node))

    # Sort by qualified name for deterministic output
    dead.sort(key=lambda x: x[1].qualified_name)
    dead = dead[:limit]

    return [
        {
            "qualified_name": node.qualified_name,
            "name": node.name,
            "kind": node.kind.value,
            "file": node.source_uri,
            "line_start": node.line_start,
            "line_end": node.line_end,
        }
        for _, node in dead
    ]
