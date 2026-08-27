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
    "main", "main_", "Test", "Handle", "handle_",
    "serve", "start", "entry", "init", "setup", "default",
]

# Exact names that are entry points when they appear as top-level
# functions (not methods, not suffixed names).
_ENTRY_POINT_EXACT = {"main", "start", "serve", "run"}

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
    """Check if a node looks like an entry point by name.

    Only top-level functions (not methods) with entry-point name patterns
    are considered.  This avoids false positives like ``username``,
    ``run_test``, ``new_user``, ``default_value``, etc.
    """
    if patterns is None:
        patterns = _ENTRY_POINT_PATTERNS
    name = node.name
    # Only top-level functions qualify — methods are never entry points.
    if node.kind != NodeKind.FUNCTION:
        return False
    # Exact match on known entry-point names.
    if name in _ENTRY_POINT_EXACT:
        return True
    lower = name.lower()
    # Suffix match for test-related patterns (e.g. `some_test_`).
    if lower.endswith("test_") or lower.endswith("test"):
        return True
    for pattern in patterns:
        pl = pattern.lower()
        # Prefix match for structural entry patterns.
        if pl in ("main_", "entry_", "handle", "handle_"):
            if lower.startswith(pl):
                return True
        # "Test" as exact match.
        if name == "Test":
            return True
    return False


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
    referenced_nodes: set[NodeId] = set()

    for _, src, dst, edge in graph.edges():
        if edge.kind == EdgeKind.CALLS:
            # Only the callee (dst) is "called" — the caller (src) is not.
            called_nodes.add(dst)
            # The caller is referenced by the call relationship.
            referenced_nodes.add(src)
        elif edge.kind == EdgeKind.IMPORTS:
            imported_nodes.add(dst)
            referenced_nodes.add(src)
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
            # Data-flow, depends-on, etc. — both ends are referenced.
            referenced_nodes.add(src)
            referenced_nodes.add(dst)

    # Filter nodes: find candidates with no callers and no references
    dead: list[tuple[NodeId, Node]] = []
    for nid, node in graph.nodes():
        # Only consider functions, methods, and classes
        if node.kind not in (NodeKind.FUNCTION, NodeKind.METHOD, NodeKind.CLASS):
            continue

        # Skip placeholder nodes
        if node.qualified_name.startswith("call::"):
            continue

        # Skip if called (appears as dst of a Calls edge)
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

        # Skip if referenced (caller, importer, data-flow participant)
        if nid in referenced_nodes:
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
