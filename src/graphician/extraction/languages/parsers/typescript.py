"""TypeScript source extraction.

Emits nodes for:
- `Function` (function declarations, arrow functions, async functions)
- `Method` (method definitions in classes)
- `Class` (class declarations)
- `Trait` (interface declarations)
- `Module` (import targets)
- `Type` (type aliases, enums)

Emits edges for:
- `Defines` from parent → child
- `Inherits` from class → superclass
- `Implements` from class → interfaces
- `Imports` from file → module
- `Calls` from function → callee
"""

from __future__ import annotations

from pathlib import Path

import tree_sitter as ts
import tree_sitter_typescript

from graphician.core.edge import Edge, EdgeKind
from graphician.core.graph import Graph
from graphician.core.id import NodeId
from graphician.core.node import Node, NodeKind


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
    default_props = {"dialect": "typescript"}
    if props:
        default_props.update(props)
    if default_props:
        for k, v in default_props.items():
            node = node.with_property(k, v)
    graph.add_node(node)
    return graph.find_by_qname(qn)


def _extract_decorators(node: ts.Node, source: str) -> list[str]:
    """Extract decorator/annotation names from a node."""
    decorators = []
    for child in _children(node):
        if child.type == "decorator":
            for c in _children(child):
                if c.type != "@":
                    decorators.append(_text(c).strip())
        elif child.type == "decorated":
            for c in _children(child):
                if c.type == "decorator":
                    for cc in _children(c):
                        if cc.type != "@":
                            decorators.append(_text(cc).strip())
    return [d for d in decorators if d]


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
    elif func_node.type == "property_identifier":
        name = _text(func_node)
    elif func_node.type == "member_expression":
        obj = func_node.child_by_field_name("object")
        if obj:
            receiver = _text(obj)
        prop = func_node.child_by_field_name("property")
        if prop:
            name = _text(prop)
    elif func_node.type == "parenthesized_expression":
        for c in _children(func_node):
            if c.type == "call_expression":
                _emit_call_expr(c, source, graph, caller_id)
                return

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
    """Parse a TypeScript source file and emit nodes/edges into the graph."""
    with open(path, "rb") as f:
        raw = f.read()
    source = raw.decode("utf-8", errors="replace")

    lang_ptr = tree_sitter_typescript.language_typescript()
    language = ts.Language(lang_ptr)
    parser = ts.Parser(language)
    tree = parser.parse(raw)
    root = tree.root_node

    record_path = source_path if source_path is not None else path
    file_qn = file_qn or path.stem
    file_id = _add_node(graph, NodeKind.FILE, file_qn, record_path, 0, 0, source)

    # Walk top-level definitions (includes imports/exports)
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
        if child.type == "import_statement":
            _handle_import(child, source, graph, parent_id)
        elif child.type == "export_statement":
            _handle_export(child, source, graph, parent_id)
            _walk_body(child, source, graph, file_qn, path, parent_id, scope)
        elif child.type == "function_declaration":
            _handle_function(child, source, graph, file_qn, path, parent_id, scope, False)
        elif child.type == "class_declaration":
            _handle_class(child, source, graph, file_qn, path, parent_id, scope)
        elif child.type == "interface_declaration":
            _handle_interface(child, source, graph, file_qn, path, parent_id, scope)
        elif child.type == "type_alias_declaration":
            _handle_type_alias(child, source, graph, file_qn, path, parent_id, scope)
        elif child.type == "enum_declaration":
            _handle_enum(child, source, graph, file_qn, path, parent_id, scope)
        elif child.type == "lexical_declaration":
            _handle_lexical_decl(child, source, graph, file_qn, path, parent_id, scope)


def _handle_import(node: ts.Node, source: str, graph: Graph, file_id: NodeId) -> None:
    """Handle import statements."""
    for child in _children(node):
        if child.type == "string":
            import_path = _text(child).strip('"')
            mod_qn = f"module::{import_path}"
            mod_id = _add_node(graph, NodeKind.MODULE, mod_qn, Path(""), 0, 0)
            graph.add_edge(file_id, mod_id, Edge.extracted(EdgeKind.IMPORTS))
            return


def _handle_export(node: ts.Node, source: str, graph: Graph, file_id: NodeId) -> None:
    """Handle export statements."""
    for child in _children(node):
        if child.type == "string":
            import_path = _text(child).strip('"')
            mod_qn = f"module::{import_path}"
            mod_id = _add_node(graph, NodeKind.MODULE, mod_qn, Path(""), 0, 0)
            graph.add_edge(file_id, mod_id, Edge.extracted(EdgeKind.IMPORTS))
            return


def _handle_function(
    node: ts.Node,
    source: str,
    graph: Graph,
    file_qn: str,
    path: Path,
    parent_id: NodeId,
    scope: list[str],
    parent_is_class: bool,
) -> None:
    """Handle a function declaration."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return

    name = _text(name_node)
    child_scope = scope + [name]
    qn = f"{file_qn}::{'::'.join(child_scope)}"

    decorators = _extract_decorators(node, source)
    props: dict = {}
    if decorators:
        props["decorators"] = decorators

    kind = NodeKind.METHOD if parent_is_class else NodeKind.FUNCTION
    fn_id = _add_node(graph, kind, qn, path,
                      node.start_point.row, node.end_point.row,
                      _text(node), props)
    graph.add_edge(parent_id, fn_id, Edge.extracted(EdgeKind.DEFINES))

    # Walk function body
    body = node.child_by_field_name("body")
    if body:
        _emit_calls(body, source, graph, fn_id)
        _walk_body(body, source, graph, file_qn, path, fn_id, child_scope)


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

    decorators = _extract_decorators(node, source)
    props: dict = {}
    if decorators:
        props["decorators"] = decorators

    class_id = _add_node(graph, NodeKind.CLASS, qn, path,
                         node.start_point.row, node.end_point.row,
                         _text(node), props)
    graph.add_edge(parent_id, class_id, Edge.extracted(EdgeKind.DEFINES))

    # Handle extends
    superclass = node.child_by_field_name("superclass")
    if superclass:
        super_name = _text(superclass)
        super_qn = f"type::{super_name}"
        super_id = _add_node(graph, NodeKind.CLASS, super_qn, Path(""), 0, 0)
        graph.add_edge(class_id, super_id, Edge.extracted(EdgeKind.INHERITS))

    # Handle implements
    interfaces = node.child_by_field_name("interfaces")
    if interfaces:
        for iface in _children(interfaces):
            if iface.type == "identifier":
                iface_name = _text(iface)
                iface_qn = f"type::{iface_name}"
                iface_id = _add_node(graph, NodeKind.TRAIT, iface_qn, Path(""), 0, 0)
                graph.add_edge(class_id, iface_id, Edge.extracted(EdgeKind.IMPLEMENTS))

    # Walk class body
    body = node.child_by_field_name("body")
    if body:
        _walk_class_body(body, source, graph, file_qn, path, class_id, child_scope)


def _walk_class_body(
    node: ts.Node,
    source: str,
    graph: Graph,
    file_qn: str,
    path: Path,
    parent_id: NodeId,
    scope: list[str],
) -> None:
    """Walk a class body and emit methods."""
    for child in _children(node):
        if child.type == "method_definition":
            _handle_method(child, source, graph, file_qn, path, parent_id, scope)
        elif child.type == "class_declaration":
            _handle_class(child, source, graph, file_qn, path, parent_id, scope)


def _handle_method(
    node: ts.Node,
    source: str,
    graph: Graph,
    file_qn: str,
    path: Path,
    parent_id: NodeId,
    scope: list[str],
) -> None:
    """Handle a method definition."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return

    name = _text(name_node)
    child_scope = scope + [name]
    qn = f"{file_qn}::{'::'.join(child_scope)}"

    decorators = _extract_decorators(node, source)
    props: dict = {}
    if decorators:
        props["decorators"] = decorators

    method_id = _add_node(graph, NodeKind.METHOD, qn, path,
                          node.start_point.row, node.end_point.row,
                          _text(node), props)
    graph.add_edge(parent_id, method_id, Edge.extracted(EdgeKind.DEFINES))

    # Walk method body
    body = node.child_by_field_name("body")
    if body:
        _emit_calls(body, source, graph, method_id)


def _handle_interface(
    node: ts.Node,
    source: str,
    graph: Graph,
    file_qn: str,
    path: Path,
    parent_id: NodeId,
    scope: list[str],
) -> None:
    """Handle an interface declaration."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return

    name = _text(name_node)
    child_scope = scope + [name]
    qn = f"{file_qn}::{'::'.join(child_scope)}"

    iface_id = _add_node(graph, NodeKind.TRAIT, qn, path,
                         node.start_point.row, node.end_point.row,
                         _text(node))
    graph.add_edge(parent_id, iface_id, Edge.extracted(EdgeKind.DEFINES))

    # Handle extends (parent interfaces)
    interfaces = node.child_by_field_name("interfaces")
    if interfaces:
        for iface in _children(interfaces):
            if iface.type == "identifier":
                iface_name = _text(iface)
                iface_qn = f"type::{iface_name}"
                iface_id = _add_node(graph, NodeKind.TRAIT, iface_qn, Path(""), 0, 0)
                graph.add_edge(iface_id, iface_id, Edge.extracted(EdgeKind.INHERITS))

    # Walk interface body
    body = node.child_by_field_name("body")
    if body:
        _walk_body(body, source, graph, file_qn, path, iface_id, child_scope)


def _handle_type_alias(
    node: ts.Node,
    source: str,
    graph: Graph,
    file_qn: str,
    path: Path,
    parent_id: NodeId,
    scope: list[str],
) -> None:
    """Handle a type alias declaration."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return

    name = _text(name_node)
    child_scope = scope + [name]
    qn = f"{file_qn}::{'::'.join(child_scope)}"

    _add_node(graph, NodeKind.TYPE, qn, path,
              node.start_point.row, node.end_point.row,
              _text(node))
    graph.add_edge(parent_id, graph.find_by_qname(qn),
                  Edge.extracted(EdgeKind.DEFINES))


def _handle_enum(
    node: ts.Node,
    source: str,
    graph: Graph,
    file_qn: str,
    path: Path,
    parent_id: NodeId,
    scope: list[str],
) -> None:
    """Handle an enum declaration."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return

    name = _text(name_node)
    child_scope = scope + [name]
    qn = f"{file_qn}::{'::'.join(child_scope)}"

    enum_id = _add_node(graph, NodeKind.TYPE, qn, path,
                        node.start_point.row, node.end_point.row,
                        _text(node))
    graph.add_edge(parent_id, enum_id, Edge.extracted(EdgeKind.DEFINES))

    # Emit enum members as Variable nodes
    body = node.child_by_field_name("body")
    if body:
        for member in _children(body):
            if member.type == "enum_member":
                member_name = member.child_by_field_name("name")
                if member_name:
                    member_qn = f"{qn}::{_text(member_name)}"
                    _add_node(graph, NodeKind.VARIABLE, member_qn, path,
                              member.start_point.row, member.end_point.row)
                    graph.add_edge(enum_id, graph.find_by_qname(member_qn),
                                   Edge.extracted(EdgeKind.DEFINES))


def _handle_lexical_decl(
    node: ts.Node,
    source: str,
    graph: Graph,
    file_qn: str,
    path: Path,
    parent_id: NodeId,
    scope: list[str],
) -> None:
    """Handle var/let/const declarations — check for arrow functions."""
    for child in _children(node):
        if child.type == "variable_declarator":
            name_node = child.child_by_field_name("name")
            init = child.child_by_field_name("value")
            if not name_node:
                continue
            name = _text(name_node)
            child_scope = scope + [name]
            qn = f"{file_qn}::{'::'.join(child_scope)}"

            if init and init.type == "arrow_function":
                fn_id = _add_node(graph, NodeKind.FUNCTION, qn, path,
                                  init.start_point.row, init.end_point.row,
                                  _text(init))
                graph.add_edge(parent_id, fn_id, Edge.extracted(EdgeKind.DEFINES))

                body = init.child_by_field_name("body")
                if body:
                    _emit_calls(body, source, graph, fn_id)
