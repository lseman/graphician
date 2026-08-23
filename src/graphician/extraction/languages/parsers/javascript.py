"""JavaScript source extraction.

Emits: Function, Method, Class, Module, Type, File
Edges: Defines, Inherits, Imports, Calls
"""

from __future__ import annotations

from pathlib import Path

import tree_sitter as ts
import tree_sitter_javascript

from graphician.core.edge import Edge, EdgeKind
from graphician.core.graph import Graph
from graphician.core.id import NodeId
from graphician.core.node import Node, NodeKind


def _text(node: ts.Node, source: str) -> str:
    return source[node.start_byte:node.end_byte]


def _children(node: ts.Node) -> list[ts.Node]:
    return list(node.children)


def _child_by_field(node: ts.Node, field: str) -> ts.Node | None:
    for i, c in enumerate(node.children):
        if node.field_name_for_child(i) == field:
            return c
    return None


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
    default_props = {"dialect": "javascript"}
    if props:
        default_props.update(props)
    if default_props:
        for k, v in default_props.items():
            node = node.with_property(k, v)
    graph.add_node(node)
    return graph.find_by_qname(qn)


def _emit_call(
    graph: Graph,
    caller_id: NodeId,
    name: str,
    scope: str | None = None,
    receiver: str | None = None,
) -> None:
    callee_qn = f"call::{name}"
    callee_id = _add_node(graph, NodeKind.FUNCTION, callee_qn, Path(""), 0, 0)
    edge = Edge.ambiguous(EdgeKind.CALLS)
    if scope:
        edge.properties["call_scope"] = scope
    if receiver:
        edge.properties["call_receiver"] = receiver
    graph.add_edge(caller_id, callee_id, edge)


def _extract_decorators(node: ts.Node, source: str) -> list[str]:
    decorators = []
    for child in node.children:
        if child.type == "decorated":
            for c in child.children:
                if c.type == "decorator":
                    decorators.append(_text(c, source))
    return decorators if decorators else []


def _call_target_name(node: ts.Node, source: str) -> tuple[str, str | None] | None:
    if node.type == "identifier":
        return (_text(node, source), None)
    if node.type in ("property_identifier", "field_identifier"):
        return (_text(node, source), None)
    if node.type in ("scoped_identifier", "scoped_type_identifier"):
        text = _text(node, source)
        parts = text.split("::")
        return (parts[-1], "::".join(parts[:-1]) if len(parts) > 1 else None)
    if node.type == "member_expression":
        prop = _child_by_field(node, "property")
        if prop:
            return (_text(prop, source), None)
    if node.type == "field_expression":
        field = _child_by_field(node, "field")
        if field:
            return (_text(field, source), None)
    return None


def _extract_receiver(func_node: ts.Node, source: str) -> str | None:
    if func_node.type == "member_expression":
        obj = _child_by_field(func_node, "object")
        if obj:
            return _text(obj, source)
    if func_node.type in ("scoped_identifier", "scoped_type_identifier"):
        text = _text(func_node, source)
        return text.split("::")[0] if "::" in text else None
    return None


def _emit_calls_in_tree(node: ts.Node, graph: Graph, caller_id: NodeId, source: str) -> None:
    skip = frozenset(["function_declaration", "arrow_function", "method_definition",
                      "class_declaration", "function_expression"])
    stack = _children(node)
    while stack:
        child = stack.pop()
        if child.type in skip:
            continue
        if child.type in ("call", "call_expression"):
            func = None
            for i, c in enumerate(child.children):
                if child.field_name_for_child(i) == "function":
                    func = c
                    break
            if func is None:
                func = child.children[0] if child.children else None
            if func:
                result = _call_target_name(func, source)
                if result:
                    name, scope = result
                    receiver = _extract_receiver(func, source)
                    _emit_call(graph, caller_id, name, scope, receiver)
        stack.extend(_children(child))


def _walk(node: ts.Node, source: str, graph: Graph, file_qn: str, path: Path, parent_id: NodeId, parent_is_class: bool, parent_is_method: bool) -> None:
    for child in _children(node):
        if child.type == "export_statement":
            _walk(child, source, graph, file_qn, path, parent_id, parent_is_class, parent_is_method)
        elif child.type == "function_declaration":
            name_node = _child_by_field(child, "name")
            if not name_node:
                continue
            name = _text(name_node, source)
            qn = f"{file_qn}::{name}"
            kind = NodeKind.METHOD if parent_is_class else NodeKind.FUNCTION
            decorators = _extract_decorators(child, source)
            props = {}
            if decorators:
                props["decorators"] = decorators
            fn_id = _add_node(graph, kind, qn, path, child.start_point.row, child.end_point.row, source, props)
            graph.add_edge(parent_id, fn_id, Edge.extracted(EdgeKind.DEFINES))
            body = _child_by_field(child, "body")
            if body:
                _emit_calls_in_tree(body, graph, fn_id, source)
                _walk(body, source, graph, file_qn, path, fn_id, False, False)

        elif child.type == "arrow_function":
            qn = f"{file_qn}::arrow_{child.start_point.row}"
            kind = NodeKind.METHOD if parent_is_class else NodeKind.FUNCTION
            fn_id = _add_node(graph, kind, qn, path, child.start_point.row, child.end_point.row, source)
            graph.add_edge(parent_id, fn_id, Edge.extracted(EdgeKind.DEFINES))
            body = _child_by_field(child, "body")
            if body:
                _emit_calls_in_tree(body, graph, fn_id, source)
                _walk(body, source, graph, file_qn, path, fn_id, False, False)

        elif child.type == "function_expression":
            name_node = _child_by_field(child, "name")
            name = _text(name_node, source) if name_node else f"anonymous_{child.start_point.row}"
            qn = f"{file_qn}::{name}"
            kind = NodeKind.METHOD if parent_is_class else NodeKind.FUNCTION
            fn_id = _add_node(graph, kind, qn, path, child.start_point.row, child.end_point.row, source)
            graph.add_edge(parent_id, fn_id, Edge.extracted(EdgeKind.DEFINES))
            body = _child_by_field(child, "body")
            if body:
                _emit_calls_in_tree(body, graph, fn_id, source)
                _walk(body, source, graph, file_qn, path, fn_id, False, False)

        elif child.type == "method_definition":
            name_node = _child_by_field(child, "name")
            if not name_node:
                continue
            name = _text(name_node, source)
            qn = f"{file_qn}::{name}"
            decorators = _extract_decorators(child, source)
            props = {}
            if decorators:
                props["decorators"] = decorators
            fn_id = _add_node(graph, NodeKind.METHOD, qn, path, child.start_point.row, child.end_point.row, source, props)
            graph.add_edge(parent_id, fn_id, Edge.extracted(EdgeKind.DEFINES))
            body = _child_by_field(child, "body")
            if body:
                _emit_calls_in_tree(body, graph, fn_id, source)
                _walk(body, source, graph, file_qn, path, fn_id, True, False)

        elif child.type == "class_declaration":
            name_node = _child_by_field(child, "name")
            if not name_node:
                continue
            name = _text(name_node, source)
            qn = f"{file_qn}::{name}"
            decorators = _extract_decorators(child, source)
            props = {}
            if decorators:
                props["decorators"] = decorators

            class_id = _add_node(graph, NodeKind.CLASS, qn, path, child.start_point.row, child.end_point.row, source, props)
            graph.add_edge(parent_id, class_id, Edge.extracted(EdgeKind.DEFINES))

            # Extends
            superclass = _child_by_field(child, "superclass")
            if superclass is None:
                heritage = next(
                    (item for item in _children(child) if item.type == "class_heritage"),
                    None,
                )
                if heritage is not None:
                    named = _children(heritage)
                    superclass = named[-1] if named else None
            if superclass:
                super_name = _text(superclass, source)
                super_qn = f"type::{super_name}"
                _add_node(graph, NodeKind.CLASS, super_qn, Path(""), 0, 0)
                graph.add_edge(class_id, graph.find_by_qname(super_qn), Edge.extracted(EdgeKind.INHERITS))

            body = _child_by_field(child, "body")
            if body:
                _walk(body, source, graph, file_qn, path, class_id, True, False)

        elif child.type == "class_expression":
            name_node = _child_by_field(child, "name")
            name = _text(name_node, source) if name_node else f"anon_class_{child.start_point.row}"
            qn = f"{file_qn}::{name}"
            class_id = _add_node(graph, NodeKind.CLASS, qn, path, child.start_point.row, child.end_point.row, source)
            graph.add_edge(parent_id, class_id, Edge.extracted(EdgeKind.DEFINES))
            superclass = _child_by_field(child, "superclass")
            if superclass is None:
                heritage = next(
                    (item for item in _children(child) if item.type == "class_heritage"),
                    None,
                )
                if heritage is not None:
                    named = _children(heritage)
                    superclass = named[-1] if named else None
            if superclass:
                super_qn = f"type::{_text(superclass, source)}"
                _add_node(graph, NodeKind.CLASS, super_qn, Path(""), 0, 0)
                graph.add_edge(class_id, graph.find_by_qname(super_qn), Edge.extracted(EdgeKind.INHERITS))
            body = _child_by_field(child, "body")
            if body:
                _walk(body, source, graph, file_qn, path, class_id, True, False)

        elif child.type == "lexical_declaration":
            for declaration in _children(child):
                if declaration.type != "variable_declarator":
                    continue
                name_node = _child_by_field(declaration, "name")
                value_node = _child_by_field(declaration, "value")
                if (
                    name_node is None
                    or value_node is None
                    or value_node.type not in {"arrow_function", "function_expression"}
                ):
                    continue
                name = _text(name_node, source)
                fn_id = _add_node(
                    graph,
                    NodeKind.METHOD if parent_is_class else NodeKind.FUNCTION,
                    f"{file_qn}::{name}",
                    path,
                    value_node.start_point.row,
                    value_node.end_point.row,
                    _text(value_node, source),
                )
                graph.add_edge(parent_id, fn_id, Edge.extracted(EdgeKind.DEFINES))
                body = _child_by_field(value_node, "body")
                if body:
                    _emit_calls_in_tree(body, graph, fn_id, source)

        elif child.type == "import_statement":
            source_node = _child_by_field(child, "source")
            if source_node is None:
                source_node = next((c for c in _children(child) if c.type == "string"), None)
            if source_node:
                src_text = _text(source_node, source).strip().strip('"').strip("'")
                mod_qn = f"module::{src_text}"
                mod_id = _add_node(graph, NodeKind.MODULE, mod_qn, Path(""), 0, 0)
                graph.add_edge(parent_id, mod_id, Edge.extracted(EdgeKind.IMPORTS))

        elif child.type == "export_statement":
            source_node = _child_by_field(child, "source")
            if source_node:
                src_text = _text(source_node, source).strip().strip('"').strip("'")
                mod_qn = f"module::{src_text}"
                mod_id = _add_node(graph, NodeKind.MODULE, mod_qn, Path(""), 0, 0)
                graph.add_edge(parent_id, mod_id, Edge.extracted(EdgeKind.IMPORTS))


def extract_file(
    path: Path, graph: Graph, *, file_qn: str | None = None, source_path: Path | None = None
) -> None:
    with open(path, "rb") as f:
        raw = f.read()
    source = raw.decode("utf-8", errors="replace")

    lang_ptr = tree_sitter_javascript.language()
    language = ts.Language(lang_ptr)
    parser = ts.Parser(language)
    tree = parser.parse(raw)
    root = tree.root_node

    record_path = source_path if source_path is not None else path
    file_qn = file_qn or path.stem
    file_id = _add_node(graph, NodeKind.FILE, file_qn, record_path, 0, 0, source)
    _walk(root, source, graph, file_qn, record_path, file_id, False, False)
