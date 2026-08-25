"""Java source extraction.

Emits nodes for:
- `Class` (class declarations)
- `Trait` (interface declarations)
- `Method` (method declarations)
- `Variable` (fields, parameters)
- `Module` (package declarations)

Emits edges for:
- `Defines` from parent → child
- `Inherits` from class → super class
- `Implements` from class → interfaces
- `Imports` from file → import target
- `Calls` from method → callee
"""

from __future__ import annotations

from pathlib import Path

import tree_sitter as ts
import tree_sitter_java

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
    default_props = {"dialect": "java"}
    if props:
        default_props.update(props)
    if default_props:
        for k, v in default_props.items():
            node = node.with_property(k, v)
    graph.add_node(node)
    return graph.find_by_qname(qn)


def _extract_annotations(node: ts.Node, source: str) -> list[str]:
    """Extract annotation names from a node."""
    annotations = []
    for child in _children(node):
        if child.type == "annotation":
            name_node = child.child_by_field_name("name")
            if name_node:
                annotations.append(_text(name_node))
            # Also check for argument
            for c in _children(child):
                if c.type == "argument_list":
                    for gc in _children(c):
                        if gc.type == "equals":
                            pass  # annotation with args
    return annotations


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
        if child.type == "method_invocation":
            name_node = child.child_by_field_name("name")
            if name_node:
                name = _text(name_node)
                callee_qn = f"call::{name}"
                callee_id = _add_node(graph, NodeKind.FUNCTION, callee_qn, Path(""), 0, 0)
                edge = Edge.ambiguous(EdgeKind.CALLS)
                object_node = child.child_by_field_name("object")
                if object_node:
                    edge = edge.with_property("call_receiver", _text(object_node))
                graph.add_edge(caller_id, callee_id, edge)
        stack.extend(_children(child))


def extract_file(
    path: Path, graph: Graph, *, file_qn: str | None = None, source_path: Path | None = None
) -> None:
    """Parse a Java source file and emit nodes/edges into the graph."""
    with open(path, "rb") as f:
        raw = f.read()
    source = raw.decode("utf-8", errors="replace")

    tree_sitter_java.language()
    tree_sitter_java.language()
    parser = ts.Parser(ts.Language(tree_sitter_java.language()))
    tree = parser.parse(raw)
    root = tree.root_node

    record_path = source_path if source_path is not None else path
    file_qn = file_qn or path.stem
    file_id = _add_node(graph, NodeKind.FILE, file_qn, record_path, 0, 0, source)

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
        if child.type == "package_declaration":
            name_node = child.child_by_field_name("name")
            if name_node:
                mod_qn = f"module::{_text(name_node)}"
                mod_id = _add_node(graph, NodeKind.MODULE, mod_qn, Path(""), 0, 0)
                graph.add_edge(parent_id, mod_id, Edge.extracted(EdgeKind.IMPORTS))
        elif child.type == "import_declaration":
            name_node = child.child_by_field_name("name") or child.child_by_field_name("path")
            if name_node is None:
                name_node = next(iter(child.named_children), None)
            if name_node:
                import_path = _text(name_node)
                # Handle static imports
                if import_path.endswith(".*"):
                    import_path = import_path[:-2]
                mod_qn = f"module::{import_path}"
                mod_id = _add_node(graph, NodeKind.MODULE, mod_qn, Path(""), 0, 0)
                graph.add_edge(parent_id, mod_id, Edge.extracted(EdgeKind.IMPORTS))
        elif child.type == "class_declaration":
            _handle_class(child, source, graph, file_qn, path, parent_id, scope)
        elif child.type == "interface_declaration":
            _handle_interface(child, source, graph, file_qn, path, parent_id, scope)
        elif child.type == "method_declaration":
            _handle_method(child, source, graph, file_qn, path, parent_id, scope, False)


def _handle_class(
    node: ts.Node,
    source: str,
    graph: Graph,
    file_qn: str,
    path: Path,
    parent_id: NodeId,
    scope: list[str],
) -> None:
    name_node = node.child_by_field_name("name")
    if not name_node:
        return

    name = _text(name_node)
    child_scope = [*scope, name]
    qn = f"{file_qn}::{'::'.join(child_scope)}"

    annotations = _extract_annotations(node, source)
    props: dict = {}
    if annotations:
        props["annotations"] = annotations

    class_id = _add_node(graph, NodeKind.CLASS, qn, path,
                         node.start_point.row, node.end_point.row,
                         _text(node), props)
    graph.add_edge(parent_id, class_id, Edge.extracted(EdgeKind.DEFINES))

    # Handle extends (superclass)
    superclass = node.child_by_field_name("superclass")
    if superclass:
        type_node = superclass.child_by_field_name("type")
        super_name = _text(type_node) if type_node else _text(superclass)
        super_qn = f"type::{super_name}"
        super_id = _add_node(graph, NodeKind.CLASS, super_qn, Path(""), 0, 0)
        graph.add_edge(class_id, super_id, Edge.extracted(EdgeKind.INHERITS))

    # Handle implements (interfaces)
    super_interfaces = node.child_by_field_name("super_interfaces")
    if super_interfaces:
        for iface in _children(super_interfaces):
            if iface.type == "identifier":
                iface_name = _text(iface)
                iface_qn = f"type::{iface_name}"
                iface_id = _add_node(graph, NodeKind.TRAIT, iface_qn, Path(""), 0, 0)
                graph.add_edge(class_id, iface_id, Edge.extracted(EdgeKind.IMPLEMENTS))

    # Walk class body
    body = node.child_by_field_name("body")
    if body:
        _walk_body(body, source, graph, file_qn, path, class_id, child_scope)


def _handle_interface(
    node: ts.Node,
    source: str,
    graph: Graph,
    file_qn: str,
    path: Path,
    parent_id: NodeId,
    scope: list[str],
) -> None:
    name_node = node.child_by_field_name("name")
    if not name_node:
        return

    name = _text(name_node)
    child_scope = [*scope, name]
    qn = f"{file_qn}::{'::'.join(child_scope)}"

    annotations = _extract_annotations(node, source)
    props: dict = {}
    if annotations:
        props["annotations"] = annotations

    iface_id = _add_node(graph, NodeKind.TRAIT, qn, path,
                         node.start_point.row, node.end_point.row,
                         _text(node), props)
    graph.add_edge(parent_id, iface_id, Edge.extracted(EdgeKind.DEFINES))

    # Handle extends (parent interfaces)
    super_interfaces = node.child_by_field_name("super_interfaces")
    if super_interfaces:
        for iface in _children(super_interfaces):
            if iface.type == "identifier":
                iface_name = _text(iface)
                iface_qn = f"type::{iface_name}"
                iface_id = _add_node(graph, NodeKind.TRAIT, iface_qn, Path(""), 0, 0)
                graph.add_edge(iface_id, iface_id, Edge.extracted(EdgeKind.INHERITS))

    # Walk interface body
    body = node.child_by_field_name("body")
    if body:
        _walk_body(body, source, graph, file_qn, path, iface_id, child_scope)


def _handle_method(
    node: ts.Node,
    source: str,
    graph: Graph,
    file_qn: str,
    path: Path,
    parent_id: NodeId,
    scope: list[str],
    is_nested: bool,
) -> None:
    name_node = node.child_by_field_name("name")
    if not name_node:
        return

    name = _text(name_node)
    child_scope = [*scope, name]
    qn = f"{file_qn}::{'::'.join(child_scope)}"

    annotations = _extract_annotations(node, source)
    props: dict = {}
    if annotations:
        props["annotations"] = annotations

    method_id = _add_node(graph, NodeKind.METHOD, qn, path,
                          node.start_point.row, node.end_point.row,
                          _text(node), props)
    graph.add_edge(parent_id, method_id, Edge.extracted(EdgeKind.DEFINES))

    # Handle parameters as Variable nodes
    params = node.child_by_field_name("parameters")
    if params:
        for param in _children(params):
            if param.type == "formal_parameter":
                param_name = param.child_by_field_name("name")
                if param_name:
                    param_qn = f"{qn}::{_text(param_name)}"
                    _add_node(graph, NodeKind.VARIABLE, param_qn, path,
                              param.start_point.row, param.end_point.row)

    # Walk method body for calls
    body = node.child_by_field_name("body")
    if body:
        _emit_calls(body, source, graph, method_id)
        # Also walk nested definitions
        _walk_body(body, source, graph, file_qn, path, method_id, child_scope)
