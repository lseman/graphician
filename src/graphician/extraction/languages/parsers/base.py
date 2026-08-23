"""Shared utilities for language parsers."""

from __future__ import annotations

from pathlib import Path

import tree_sitter as ts

from graphician.core.edge import Edge, EdgeKind
from graphician.core.graph import Graph
from graphician.core.node import Node, NodeKind

# Suppress these call placeholders — they're noise from boilerplate.
_SUPPRESS_CALLS = frozenset([
    "std::panic", "std::result", "std::option", "std::vec",
    "std::vec::Vec", "std::option::Option", "std::result::Result",
    "std::boxed::Box", "std::sync", "std::rc", "std::cell",
    "std::mem", "std::convert", "std::ops", "std::fmt",
])


def _text(node: ts.Node, source: bytes) -> str:
    """Get text content of a tree-sitter node."""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _children(node: ts.Node) -> list[ts.Node]:
    """Return direct children as a list."""
    return list(node.children)


def _child_by_name(node: ts.Node, name: str) -> ts.Node | None:
    """Get first child with given type or field name."""
    for i, c in enumerate(node.children):
        if c.type == name or node.field_name_for_child(i) == name:
            return c
    return None


def _walk_descendants(node: ts.Node):
    """Walk all descendants (not just direct children)."""
    stack = list(_children(node))
    while stack:
        child = stack.pop()
        yield child
        stack.extend(_children(child))


def _is_test_name(name: str) -> bool:
    """Check if a name looks like a test."""
    return (name.startswith("test_") or name.endswith("_test")
            or name.startswith("Test") or name.lower().startswith("test"))


def _is_test_attribute(raw: str) -> bool:
    """Check if an attribute marks something as a test."""
    inner = raw.strip()
    if inner.startswith("#["):
        inner = inner[2:]
    if inner.endswith("]"):
        inner = inner[:-1]
    inner = inner.strip()
    head = inner.split("(")[0].strip()
    if head in ("test", "rstest", "pytest.mark"):
        return True
    if head.endswith("::test") or head in ("test_case", "pytest.mark"):
        return True
    return False


def _extract_source_text(source: bytes, start_line: int, end_line: int) -> str:
    """Extract source text for a line range (1-based)."""
    lines = source.split(b"\n")
    # Clamp
    start = max(0, start_line - 1)
    end = min(len(lines), end_line)
    return "\n".join(line.decode("utf-8", errors="replace") for line in lines[start:end])


def _add_node(
    graph: Graph,
    kind: NodeKind,
    qn: str,
    path: Path,
    line_start: int,
    line_end: int,
    source: bytes | None = None,
    props: dict | None = None,
) -> int:
    """Add a node and return its index. Returns -1 if already exists."""
    existing = graph.find_by_qname(qn)
    if existing is not None:
        return existing.value
    node = Node.new(kind, qn)
    uri = str(path)
    node = node.with_source(uri, line_start, line_end)
    if source is not None:
        node = node.with_source_text(source.decode("utf-8", errors="replace"))
    if props:
        for k, v in props.items():
            node = node.with_property(k, v)
    graph.add_node(node)
    return graph.find_by_qname(qn).value


def _scoped_qname(file_qn: str, scope: list[str], name: str) -> str:
    """Build a qualified name: file_qn::scope::name."""
    if scope:
        return f"{file_qn}::{'::'.join(scope)}::{name}"
    return f"{file_qn}::{name}"


def _file_qn(path: Path) -> str:
    """Build file qualified name from path."""
    return path.stem


def _clean_use_path(s: str) -> str:
    """Clean a use/import path."""
    return s.strip().rstrip(";").strip()


def _should_suppress_call(name: str) -> bool:
    """Check if a call name should be suppressed."""
    return name in _SUPPRESS_CALLS


def _extract_decorators(node: ts.Node, source: bytes) -> list[str]:
    """Extract decorator names from a definition node."""
    decorators = []
    for child in node.children:
        if child.type in ("decorator", "decorators"):
            for c in child.children:
                if c.type != "@":
                    text = _text(c, source).strip()
                    if text:
                        decorators.append(text)
        # Also check for Python-style decorator nodes
        if child.type == "decorated":
            for c in child.children:
                if c.type == "decorator":
                    for cc in c.children:
                        if cc.type != "@":
                            text = _text(cc, source).strip()
                            if text:
                                decorators.append(text)
    return decorators if decorators else []


def _emit_calls(
    node: ts.Node,
    source: bytes,
    graph: Graph,
    caller_id: int,
    suppress_kinds: tuple[str, ...] | None = None,
) -> None:
    """Walk node tree and emit call edges for call expressions."""
    if suppress_kinds is None:
        suppress_kinds = ("function_definition", "class_definition", "function_item", "struct_item", "enum_item", "trait_item")

    stack = list(_children(node))
    while stack:
        child = stack.pop()
        # Don't descend into nested function/class bodies
        if child.type in suppress_kinds:
            continue
        if child.type == "call":
            _emit_single_call(child, source, graph, caller_id)
        elif child.type == "call_expression":
            _emit_single_call(child, source, graph, caller_id)
        stack.extend(_children(child))


def _emit_single_call(
    call_node: ts.Node,
    source: bytes,
    graph: Graph,
    caller_id: int,
) -> None:
    """Emit a call edge from a single call node."""
    func_node = None
    for c in call_node.children:
        if c.child_by_field_name("function") is not None:
            func_node = c
            break

    if func_node is None:
        # Try first child as function
        func_node = call_node.children[0] if call_node.children else None

    if func_node is None:
        return

    name = None
    scope = None

    if func_node.type == "identifier":
        name = _text(func_node, source)
    elif func_node.type in ("property_identifier", "field_identifier"):
        name = _text(func_node, source)
    elif func_node.type in ("scoped_identifier", "scoped_type_identifier"):
        text = _text(func_node, source)
        parts = text.split("::")
        name = parts[-1] if parts else None
        scope = "::".join(parts[:-1]) if len(parts) > 1 else None
    elif func_node.type == "member_expression":
        prop = _child_by_name(func_node, "property")
        if prop:
            name = _text(prop, source)
    elif func_node.type == "field_expression":
        field = _child_by_name(func_node, "field")
        if field:
            name = _text(field, source)

    if name and not _should_suppress_call(name):
        callee_qn = f"call::{name}"
        callee_idx = _add_node(graph, NodeKind.FUNCTION, callee_qn, Path(""), 0, 0)
        if callee_idx >= 0:
            edge = Edge.ambiguous(EdgeKind.CALLS)
            if scope:
                edge.properties["call_scope"] = scope
            graph.add_edge(
                type("Id", (), {"value": caller_id})(),
                type("Id", (), {"value": callee_idx})(),
                edge,
            )


def _parse_source(
    path: Path,
    language: ts.Language,
) -> tuple[ts.Tree, str, str, str]:
    """Parse a source file and return (tree, source_text, file_uri, file_qn)."""
    with open(path, "rb") as f:
        source = f.read()
    parser = ts.Parser(language)
    tree = parser.parse(source)
    file_uri = str(path)
    file_qn = _file_qn(path)
    return tree, source.decode("utf-8", errors="replace"), file_uri, file_qn
