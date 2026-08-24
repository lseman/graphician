"""Library stub resolver — Tier 7 of the call-placeholder resolution chain.

After the 6-tier heuristic resolver exhausts all local-disambiguation
strategies, this module resolves remaining ``call::name`` placeholders
against a pre-built database of library stubs keyed by dependency name
or always-available globals.

Creates stub nodes for known library types (Vec, HashMap, String, etc.)
and connects unresolved ``call::name`` edges to them when a method match
is found.

Language-specific stub definitions are organized into separate modules:
- ``_rust_stubs``: Rust stdlib and common crate stubs
- ``_python_stubs``: Python stdlib, builtins, and popular libraries
- ``_javascript_stubs``: JavaScript/TypeScript built-ins and Node.js
- ``_cpp_stubs``: C++ STL components
- ``_receiver_hints``: Variable name → type mappings for disambiguation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...core.edge import Confidence, Edge, EdgeKind
from ...core.graph import Graph
from ...core.id import EdgeId, NodeId
from ...core.node import Node, NodeKind
from ..call_resolution import should_suppress_call_placeholder
from ._cpp_stubs import _CPP_STUBS
from ._javascript_stubs import _JAVASCRIPT_STUBS
from ._python_stubs import _PYTHON_STUBS
from ._receiver_hints import _RECEIVER_HINTS, get_method_disambiguation
from ._rust_stubs import _RUST_STUBS


# ── Stub lookup builder ──────────────────────────────────────────────────

def _build_stub_lookup() -> dict[str, list[tuple[str, str]]]:
    """Build a reverse lookup: method_name -> list of (type_name, dialect).

    Combines all language-specific stub dictionaries into a single
    lookup table for efficient resolution.
    """
    lookup: dict[str, list[tuple[str, str]]] = {}

    # Rust stdlib (always available)
    for type_name, methods in _RUST_STUBS.items():
        for method in methods:
            lookup.setdefault(method, []).append((type_name, "rust"))

    # JS/TS globals (always available)
    for type_name, methods in _JAVASCRIPT_STUBS.items():
        for method in methods:
            lookup.setdefault(method, []).append((type_name, "javascript"))

    # C++ STL (always available)
    for type_name, methods in _CPP_STUBS.items():
        for method in methods:
            lookup.setdefault(method, []).append((type_name, "cpp"))

    # Python stubs (available globally for Python projects)
    for type_name, methods in _PYTHON_STUBS.items():
        for method in methods:
            lookup.setdefault(method, []).append((type_name, "python"))

    return lookup


# ── Cached stub lookup ──────────────────────────────────────────────────

_STUB_LOOKUP: dict[str, list[tuple[str, str]]] | None = None


def _get_stub_lookup() -> dict[str, list[tuple[str, str]]]:
    global _STUB_LOOKUP
    if _STUB_LOOKUP is None:
        _STUB_LOOKUP = _build_stub_lookup()
    return _STUB_LOOKUP


# ── Stub node creation ──────────────────────────────────────────────────

def _create_stub_node(graph: Graph, type_name: str, dialect: str) -> NodeId | None:
    """Create or return an existing stub node for a library type."""
    stub_name = f"stub::{type_name}"
    existing = graph.find_by_qname(stub_name)
    if existing is not None:
        return existing

    # Find source file for stub node
    source_uri = f"<{dialect}-stdlib:{type_name}>"

    node = Node.new(NodeKind.CLASS, stub_name)
    node = node.with_source(source_uri, 0, 0)
    node = node.with_source_text(f"Library stub for {type_name} ({dialect})")
    node = node.with_property("dialect", dialect)
    node = node.with_property("is_stub", True)

    return graph.add_node(node)


def _source_dialect(source_uri: str | None) -> str | None:
    """Infer the caller dialect without guessing across language families."""
    if not source_uri:
        return None
    suffix = Path(source_uri).suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix == ".rs":
        return "rust"
    if suffix in {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}:
        return "javascript"
    if suffix in {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"}:
        return "cpp"
    return None


def _unresolved_call_edges(graph: Graph) -> int:
    return sum(
        1
        for _, _, target, edge in graph.edges()
        if edge.kind == EdgeKind.CALLS
        and (node := graph.node(target)) is not None
        and node.qualified_name.startswith("call::")
        and not should_suppress_call_placeholder(node.qualified_name[6:])
    )


# ── Resolver ──────────────────────────────────────────────────────────────

def resolve_library_stubs(graph: Graph) -> int:
    """Resolve ``call::name`` placeholders against library stubs (Tier 7).

    Creates stub nodes for known library types (Vec, HashMap, String, etc.)
    and connects unresolved ``call::name`` edges to them when a method
    match is found.

    This is the **7th (final) tier** of call-placeholder resolution.
    It runs after the 6-tier heuristic resolver exhausts all
    local-disambiguation strategies.

    Args:
        graph: The code graph to resolve stubs in.

    Returns:
        Number of new stub edges added.
    """
    lookup = _get_stub_lookup()

    # Gather all stub type names that already have nodes (avoid duplicates)
    stub_type_nodes: dict[str, NodeId] = {}
    for nid, node in graph.nodes():
        if node.qualified_name.startswith("stub::"):
            type_name = node.qualified_name[len("stub::"):]
            stub_type_nodes[type_name] = nid

    # Collect edges to process (avoid mutation during iteration)
    edge_data = []
    for eid, src, dst, edge in graph.edges():
        if edge.kind != EdgeKind.CALLS:
            continue
        dst_node = graph.node(dst)
        if dst_node is None:
            continue
        qn = dst_node.qualified_name
        if not qn.startswith("call::"):
            continue
        if should_suppress_call_placeholder(qn[6:]):
            continue
        edge_data.append((eid, src, dst, edge, qn))

    additions = 0

    stale_edges: list[EdgeId] = []
    for eid, src, _dst, edge, callee_qn in edge_data:
        method_name = callee_qn[6:]  # strip "call::"
        candidates = lookup.get(method_name)

        # If method not found in lookup, check if it's a stub type name itself
        # (e.g., ValueError, SentenceTransformer, MagicMock)
        if candidates is None:
            # Check receiver hints for direct stub type mappings
            if method_name in _RECEIVER_HINTS:
                stub_type = _RECEIVER_HINTS[method_name]
                if stub_type in _PYTHON_STUBS:
                    candidates = [(stub_type, "python")]
                elif stub_type in _RUST_STUBS:
                    candidates = [(stub_type, "rust")]
                elif stub_type in _JAVASCRIPT_STUBS:
                    candidates = [(stub_type, "javascript")]
                elif stub_type in _CPP_STUBS:
                    candidates = [(stub_type, "cpp")]

            if candidates is None:
                continue

        source_node = graph.node(src)
        source_dialect = _source_dialect(
            source_node.source_uri if source_node is not None else None
        )
        if source_dialect is not None:
            candidates = [item for item in candidates if item[1] == source_dialect]
            if not candidates:
                continue

        # Pick the best candidate using disambiguation hints
        best_type = None
        best_dialect = None

        # 1. Check for exact method name match with a stub type
        for type_name, dialect in candidates:
            if type_name == method_name:
                best_type = type_name
                best_dialect = dialect
                break

        # 2. Check method-level disambiguation hints
        if best_type is None:
            preferred_type = get_method_disambiguation(method_name)
            if preferred_type is not None:
                for type_name, dialect in candidates:
                    if type_name == preferred_type:
                        best_type = type_name
                        best_dialect = dialect
                        break

        # 3. Fallback: prefer Python, then Rust, then JS, then C++
        if best_type is None:
            dialect_order = ["python", "rust", "javascript", "cpp"]
            for preferred_dialect in dialect_order:
                for type_name, dialect in candidates:
                    if dialect == preferred_dialect:
                        best_type = type_name
                        best_dialect = dialect
                        break
                if best_type is not None:
                    break

        # 4. Final fallback: pick first candidate
        if best_type is None and candidates:
            best_type, best_dialect = candidates[0]

        # Get or create the stub node
        stub_id = stub_type_nodes.get(best_type)
        if stub_id is None:
            stub_id = _create_stub_node(graph, best_type, best_dialect)
            if stub_id is not None:
                stub_type_nodes[best_type] = stub_id

        if stub_id is not None:
            already_resolved = any(
                target == stub_id and existing.kind == edge.kind
                for target, existing in graph.out_neighbors(src)
            )
            if not already_resolved:
                resolved_edge = Edge(
                    kind=edge.kind,
                    confidence=Confidence.INFERRED,
                    valid_from=edge.valid_from,
                    valid_to=edge.valid_to,
                )
                resolved_edge.properties["resolved_from"] = "library_stub"
                graph.add_edge(src, stub_id, resolved_edge)
            stale_edges.append(eid)
            additions += 1

    if stale_edges:
        graph.remove_edges_by_id(stale_edges)
        orphaned = [
            node_id
            for node_id, node in graph.nodes()
            if node.qualified_name.startswith("call::")
            and not any(graph.in_neighbors(node_id))
            and not any(graph.out_neighbors(node_id))
        ]
        for node_id in orphaned:
            graph.remove_node(node_id)

    return additions


# ── Batch resolution ─────────────────────────────────────────────────────

def resolve_library_stubs_batch(graph: Graph) -> dict[str, Any]:
    """Full batch resolution with statistics.

    Args:
        graph: The code graph.

    Returns:
        Statistics dict with total placeholders, resolved, and unresolved.
    """
    total = _unresolved_call_edges(graph)

    # Resolve
    added = resolve_library_stubs(graph)
    remaining = _unresolved_call_edges(graph)

    return {
        "operation": "library_stubs",
        "total_unresolved": total,
        "resolved": added,
        "unresolved_remaining": remaining,
        "resolution_rate": round(added / total, 3) if total > 0 else 0.0,
    }
