"""C++ source extraction.

Emits nodes for:
- `Function` (function declarations and definitions)
- `Method` (functions inside class bodies)
- `Class` (class/struct declarations)
- `Module` (namespace declarations, include targets)

Emits edges for:
- `Defines` from parent → child
- `Inherits` from class → base classes
- `Imports` from file → include target
- `Calls` from function → callee
"""

from __future__ import annotations

from pathlib import Path

import tree_sitter as ts
import tree_sitter_cpp

from ariadne_py.core.edge import Edge, EdgeKind
from ariadne_py.core.graph import Graph
from ariadne_py.core.id import NodeId
from ariadne_py.core.node import Node, NodeKind


def _text(node: ts.Node) -> str:
    return node.text.decode("utf-8", errors="replace")


def _children(node: ts.Node) -> list[ts.Node]:
    
    return list(node.children)


def _add_node(
    graph: Graph,
    kind: NodeKind,
    qn: str,
    path: Path,
    line_start: int,
    line_end: int,
    source: str | None = None,
    props: dict | None = None,
) -> NodeId:
    existing = graph.find_by_qname(qn)
    if existing is not None:
        return existing
    node = Node.new(kind, qn)
    node = node.with_source(str(path), line_start + 1, line_end + 1)
    if source is not None:
        node = node.with_source_text(source)
    default_props = {"dialect": "cpp"}
    if props:
        default_props.update(props)
    if default_props:
        for k, v in default_props.items():
            node = node.with_property(k, v)
    graph.add_node(node)
    return graph.find_by_qname(qn)


def _emit_calls(
    node: ts.Node,
    source: str,
    graph: Graph,
    caller_id: NodeId,
) -> None:
    """Walk node tree and emit call edges."""
    stack = _children(node)
    while stack:
        child = stack.pop()
        if child.type == "call_expression":
            _emit_call_expr(child, source, graph, caller_id)
        elif child.type == "new_expression":
            _emit_call_expr(child, source, graph, caller_id)
        stack.extend(_children(child))


def _emit_call_expr(
    node: ts.Node,
    source: str,
    graph: Graph,
    caller_id: NodeId,
) -> None:
    """Emit a call edge from a call_expression node."""
    func_node = node.child_by_field_name("function")
    if func_node is None:
        func_node = node.children[0] if node.children else None
    if func_node is None:
        return

    name = None
    receiver = None
    if func_node.type == "identifier":
        name = _text(func_node)
    elif func_node.type == "field_expression":
        field = func_node.child_by_field_name("field")
        if field:
            name = _text(field)
        arg = func_node.child_by_field_name("argument")
        if arg:
            receiver = _text(arg)
    elif func_node.type == "qualified_identifier":
        # namespace::name — take the last segment
        parts = _text(func_node).split("::")
        name = parts[-1] if parts else None

    if name and not name.startswith("_"):
        callee_qn = f"call::{name}"
        callee_id = _add_node(graph, NodeKind.FUNCTION, callee_qn, Path(""), 0, 0)
        edge = Edge.ambiguous(EdgeKind.CALLS)
        if receiver:
            edge = edge.with_property("call_receiver", receiver)
        graph.add_edge(caller_id, callee_id, edge)


def extract_file(
    path: Path, graph: Graph, *, file_qn: str | None = None, source_path: Path | None = None
) -> None:
    """Parse a C++ source file and emit nodes/edges into the graph."""
    with open(path, "rb") as f:
        raw = f.read()
    source = raw.decode("utf-8", errors="replace")

    language = ts.Language(tree_sitter_cpp.language())
    parser = ts.Parser(language)
    tree = parser.parse(raw)
    root = tree.root_node

    record_path = source_path if source_path is not None else path
    file_qn = file_qn or path.stem
    file_id = _add_node(graph, NodeKind.FILE, file_qn, record_path, 0, 0, source)

    # Emit includes at top level
    for child in _children(root):
        if child.type in {"include", "preproc_include"}:
            _handle_include(child, source, graph, file_id)

    # Walk top-level definitions
    _walk_body(root, source, graph, file_qn, record_path, file_id, [])


def _walk_body(
    node: ts.Node,
    source: str,
    graph: Graph,
    file_qn: str,
    path: Path,
    parent_id: NodeId,
    scope: list[str],
) -> None:
    """Walk a body and emit definitions."""
    for child in _children(node):
        if child.type in {"include", "preproc_include"}:
            _handle_include(child, source, graph, parent_id)
        elif child.type == "namespace_definition":
            _handle_namespace(child, source, graph, file_qn, path, parent_id, scope)
        elif child.type == "class_specifier":
            _handle_class(child, source, graph, file_qn, path, parent_id, scope)
        elif child.type == "struct_specifier":
            _handle_struct(child, source, graph, file_qn, path, parent_id, scope)
        elif child.type == "function_definition":
            _handle_function(child, source, graph, file_qn, path, parent_id, scope, False)
        elif child.type == "declaration":
            _handle_declaration(child, source, graph, file_qn, path, parent_id, scope)


def _handle_include(node: ts.Node, source: str, graph: Graph, file_id: NodeId) -> None:
    """Handle #include directives."""
    for child in _children(node):
        if child.type in {"string_literal", "system_lib_string"}:
            include_path = _text(child).strip('"<>')
            mod_qn = f"include::{include_path}"
            mod_id = _add_node(graph, NodeKind.MODULE, mod_qn, Path(""), 0, 0)
            graph.add_edge(file_id, mod_id, Edge.extracted(EdgeKind.IMPORTS))


def _handle_namespace(
    node: ts.Node,
    source: str,
    graph: Graph,
    file_qn: str,
    path: Path,
    parent_id: NodeId,
    scope: list[str],
) -> None:
    """Handle namespace definitions."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        # Anonymous namespace
        name = "__anon"
    else:
        name = _text(name_node)

    child_scope = scope + [name]
    qn = f"{file_qn}::{'::'.join(child_scope)}"
    mod_qn = f"module::{qn}"

    mod_id = _add_node(graph, NodeKind.MODULE, mod_qn, path,
                       node.start_point.row, node.end_point.row,
                       _text(node))
    graph.add_edge(parent_id, mod_id, Edge.extracted(EdgeKind.DEFINES))

    # Walk namespace body
    body = node.child_by_field_name("body")
    if body:
        _walk_body(body, source, graph, file_qn, path, mod_id, child_scope)


def _handle_class(
    node: ts.Node,
    source: str,
    graph: Graph,
    file_qn: str,
    path: Path,
    parent_id: NodeId,
    scope: list[str],
) -> None:
    """Handle a class declaration."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return

    name = _text(name_node)
    child_scope = scope + [name]
    qn = f"{file_qn}::{'::'.join(child_scope)}"

    class_id = _add_node(graph, NodeKind.CLASS, qn, path,
                         node.start_point.row, node.end_point.row,
                         _text(node))
    graph.add_edge(parent_id, class_id, Edge.extracted(EdgeKind.DEFINES))

    # Handle inheritance (base classes)
    _handle_bases(node, source, graph, class_id)

    # Walk class body
    body = node.child_by_field_name("body")
    if body:
        _walk_class_body(body, source, graph, file_qn, path, class_id, child_scope)


def _handle_struct(
    node: ts.Node,
    source: str,
    graph: Graph,
    file_qn: str,
    path: Path,
    parent_id: NodeId,
    scope: list[str],
) -> None:
    """Handle a struct declaration (same as class for our purposes)."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return

    name = _text(name_node)
    child_scope = scope + [name]
    qn = f"{file_qn}::{'::'.join(child_scope)}"

    struct_id = _add_node(graph, NodeKind.CLASS, qn, path,
                          node.start_point.row, node.end_point.row,
                          _text(node))
    graph.add_edge(parent_id, struct_id, Edge.extracted(EdgeKind.DEFINES))

    # Handle inheritance
    _handle_bases(node, source, graph, struct_id)

    # Walk struct body
    body = node.child_by_field_name("body")
    if body:
        _walk_class_body(body, source, graph, file_qn, path, struct_id, child_scope)


def _handle_bases(
    node: ts.Node,
    source: str,
    graph: Graph,
    class_id: NodeId,
) -> None:
    """Handle base class inheritance."""
    for child in _children(node):
        if child.type == "base_class_clause":
            for base in _children(child):
                if base.type == "base_specifier":
                    type_node = base.child_by_field_name("type")
                    if type_node:
                        base_name = _text(type_node)
                        base_qn = f"type::{base_name}"
                        base_id = _add_node(graph, NodeKind.CLASS, base_qn, Path(""), 0, 0)
                        graph.add_edge(class_id, base_id, Edge.extracted(EdgeKind.INHERITS))
                    # Handle access specifier (public/protected/private)
                    for c in _children(base):
                        if c.type in ("public", "protected", "private"):
                            pass  # Access specifier — just note it


def _walk_class_body(
    node: ts.Node,
    source: str,
    graph: Graph,
    file_qn: str,
    path: Path,
    parent_id: NodeId,
    scope: list[str],
) -> None:
    """Walk a class body and emit methods and fields."""
    for child in _children(node):
        if child.type == "function_definition":
            _handle_function(child, source, graph, file_qn, path, parent_id, scope, True)
        elif child.type == "declaration":
            _handle_declaration(child, source, graph, file_qn, path, parent_id, scope)


def _handle_function(
    node: ts.Node,
    source: str,
    graph: Graph,
    file_qn: str,
    path: Path,
    parent_id: NodeId,
    scope: list[str],
    is_method: bool,
) -> None:
    """Handle a function definition."""
    name_node = None
    for child in _children(node):
        if child.type in ("declarator", "function_declarator"):
            for c in _children(child):
                if c.type in ("identifier", "field_identifier"):
                    name_node = c
                    break
            if name_node:
                break
        elif child.type == "field_identifier":
            name_node = child
            break
    if not name_node:
        return

    name = _text(name_node)
    child_scope = scope + [name]
    qn = f"{file_qn}::{'::'.join(child_scope)}"

    kind = NodeKind.METHOD if is_method else NodeKind.FUNCTION
    fn_id = _add_node(graph, kind, qn, path,
                      node.start_point.row, node.end_point.row,
                      _text(node))
    graph.add_edge(parent_id, fn_id, Edge.extracted(EdgeKind.DEFINES))

    # Walk function body for calls
    body = node.child_by_field_name("body")
    if body:
        _emit_calls(body, source, graph, fn_id)
        _walk_body(body, source, graph, file_qn, path, fn_id, child_scope)


def _handle_declaration(
    node: ts.Node,
    source: str,
    graph: Graph,
    file_qn: str,
    path: Path,
    parent_id: NodeId,
    scope: list[str],
) -> None:
    """Handle a declaration — could be a variable, type alias, or function decl."""
    for child in _children(node):
        if child.type == "init_declarator":
            name_node = child.child_by_field_name("name")
            if name_node:
                name = _text(name_node)
                child_scope = scope + [name]
                qn = f"{file_qn}::{'::'.join(child_scope)}"
                _add_node(graph, NodeKind.VARIABLE, qn, path,
                          node.start_point.row, node.end_point.row)
                graph.add_edge(parent_id, graph.find_by_qname(qn),
                              Edge.extracted(EdgeKind.DEFINES))
        elif child.type == "function_definition":
            _handle_function(child, source, graph, file_qn, path, parent_id, scope, False)
