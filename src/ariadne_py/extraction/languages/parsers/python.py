"""Python source extraction using tree-sitter 0.26."""

from __future__ import annotations

from pathlib import Path

import tree_sitter as ts
import tree_sitter_python

from ariadne_py.core.edge import Edge, EdgeKind
from ariadne_py.core.graph import Graph
from ariadne_py.core.id import NodeId
from ariadne_py.core.node import Node, NodeKind


def _extract_decorators(node: ts.Node) -> list[str]:
    decorators = []
    for child in node.children:
        if child.type == "decorated":
            for c in child.children:
                if c.type == "decorator":
                    for cc in c.children:
                        if cc.type != "@":
                            text = cc.text.decode("utf-8", errors="replace").strip()
                            if text:
                                decorators.append(text)
    return decorators if decorators else []


def _is_test_name(name: str) -> bool:
    return name.startswith("test_") or name.endswith("_test")


# Protocol/ABC base classes that make a class a trait
_PROTOCOL_BASES = frozenset(
    ["Protocol", "ABC", "typing.Protocol", "typing_extensions.Protocol"]
)


def _is_protocol_class(node: ts.Node) -> bool:
    """Check if a class inherits from Protocol or ABC (makes it a TRAIT)."""
    bases = node.child_by_field_name("superclasses")
    if bases is None:
        # Check for base_list or argument_list children
        for c in node.children:
            if c.type in ("arguments", "argument_list", "base_list"):
                bases = c
                break
    if bases is None:
        return False
    for base in bases.children:
        base_name = None
        if base.type == "identifier":
            base_name = base.text.decode("utf-8", errors="replace")
        elif base.type == "attribute":
            attr = base.child_by_field_name("attribute")
            if attr:
                base_name = attr.text.decode("utf-8", errors="replace")
        elif base.type == "dotted_name":
            base_name = base.text.decode("utf-8", errors="replace")
        if base_name and base_name in _PROTOCOL_BASES:
            return True
    return False


def _is_type_alias(node: ts.Node) -> bool:
    """Check if a function definition is actually a type alias assignment."""
    # TypeAlias: Name = ... or Name: Type = ...
    # TypedDict: class Name(TypedDict): ...
    # TypeVar: Name = TypeVar("Name")
    assign = node.child_by_field_name("initializer")
    if assign is not None:
        # Check if RHS looks like a TypeVar, NewType, or type alias pattern
        body = node.child_by_field_name("body")
        if body is None:
            rhs_text = assign.text.decode("utf-8", errors="replace").strip()
            if any(p in rhs_text for p in ["TypeVar(", "NewType(", "ParamSpec(", "TypeAlias"]):
                return True
    return False


def _is_typeddict_class(node: ts.Node) -> bool:
    """Check if a class inherits from TypedDict."""
    bases = node.child_by_field_name("superclasses")
    if bases is None:
        for c in node.children:
            if c.type in ("arguments", "argument_list", "base_list"):
                bases = c
                break
    if bases is None:
        return False
    for base in bases.children:
        base_name = None
        if base.type == "identifier":
            base_name = base.text.decode("utf-8", errors="replace")
        elif base.type == "attribute":
            attr = base.child_by_field_name("attribute")
            if attr:
                base_name = attr.text.decode("utf-8", errors="replace")
        elif base.type == "dotted_name":
            base_name = base.text.decode("utf-8", errors="replace")
        if base_name and "TypedDict" in base_name:
            return True
    return False


def _is_typevar_definition(node: ts.Node) -> bool:
    """Check if a function is actually a TypeVar() call."""
    body = node.child_by_field_name("body")
    if body is None:
        return False
    # TypeVar is typically a simple assignment or a function returning TypeVar
    for child in body.children:
        if child.type == "expression_statement":
            expr = child.children[0] if child.children else None
            if expr and expr.type == "call":
                func = None
                for i, c in enumerate(expr.children):
                    if expr.field_name_for_child(i) == "function":
                        func = c
                        break
                if func:
                    name = func.text.decode("utf-8", errors="replace")
                    if name in ("TypeVar", "ParamSpec", "TypeVarTuple", "GenericAlias"):
                        return True
    return False


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
    if props:
        for k, v in props.items():
            node = node.with_property(k, v)
    graph.add_node(node)
    return graph.find_by_qname(qn)


def _emit_calls(
    node: ts.Node,
    graph: Graph,
    caller_id: NodeId,
    suppress_types: tuple[str, ...] | None = None,
) -> None:
    if suppress_types is None:
        suppress_types = ("function_definition", "class_definition")
    stack = list(node.children)
    while stack:
        child = stack.pop()
        if child.type in suppress_types:
            continue
        if child.type == "call":
            func_node = None
            for i, c in enumerate(child.children):
                if child.field_name_for_child(i) == "function":
                    func_node = c
                    break
            if func_node is None:
                func_node = child.children[0] if child.children else None
            if func_node is None:
                continue
            name = None
            receiver = None
            if func_node.type == "identifier":
                name = func_node.text.decode("utf-8", errors="replace")
            elif func_node.type == "attribute":
                attr = func_node.child_by_field_name("attribute")
                if attr:
                    name = attr.text.decode("utf-8", errors="replace")
                obj = func_node.child_by_field_name("object")
                if obj:
                    receiver = obj.text.decode("utf-8", errors="replace")
            if name and not name.startswith("_"):
                callee_qn = f"call::{name}"
                callee_id = _add_node(graph, NodeKind.FUNCTION, callee_qn, Path(""), 0, 0,
                                      props={"dialect": "python", "role": "call_placeholder"})
                edge = Edge.ambiguous(EdgeKind.CALLS)
                if receiver:
                    edge = edge.with_property("call_receiver", receiver)
                graph.add_edge(caller_id, callee_id, edge)
        stack.extend(child.children)


def _walk_scope(
    node: ts.Node,
    graph: Graph,
    file_qn: str,
    path: Path,
    parent_id: NodeId,
    scope: list[str],
    parent_is_class: bool,
    file_is_test: bool,
) -> None:
    for child in node.children:
        if child.type == "decorated_definition":
            _walk_scope(
                child, graph, file_qn, path, parent_id, scope, parent_is_class, file_is_test
            )
        elif child.type == "class_definition":
            _handle_class(child, graph, file_qn, path, parent_id, scope, file_is_test)
        elif child.type == "function_definition":
            _handle_function(
                child, graph, file_qn, path, parent_id, scope, parent_is_class, file_is_test
            )


def _handle_class(
    node: ts.Node,
    graph: Graph,
    file_qn: str,
    path: Path,
    parent_id: NodeId,
    scope: list[str],
    file_is_test: bool,
) -> None:
    name_node = node.child_by_field_name("name")
    if not name_node:
        return
    name = name_node.text.decode("utf-8", errors="replace")
    child_scope = scope + [name]
    qn = f"{file_qn}::{'::'.join(child_scope)}"

    decorators = _extract_decorators(node)
    props: dict = {}
    if decorators:
        props["decorators"] = decorators

    # Determine kind: TRAIT for Protocol/ABC, TYPE for TypedDict, CLASS otherwise
    if _is_protocol_class(node):
        kind = NodeKind.TRAIT
        props["role"] = "protocol"
    elif _is_typeddict_class(node):
        kind = NodeKind.TYPE
        props["role"] = "typeddict"
    else:
        kind = NodeKind.CLASS

    # Add dialect property
    props["dialect"] = "python"

    class_id = _add_node(graph, kind, qn, path,
                         node.start_point.row, node.end_point.row,
                         node.text.decode("utf-8", errors="replace"), props)
    graph.add_edge(parent_id, class_id, Edge.extracted(EdgeKind.DEFINES))

    # Handle inheritance
    bases = node.child_by_field_name("superclasses")
    for c in node.children:
        if bases is None and c.type in ("arguments", "argument_list", "base_list"):
            bases = c
            break
    if bases:
        for base in bases.children:
            base_name = None
            if base.type == "identifier":
                base_name = base.text.decode("utf-8", errors="replace")
            elif base.type == "attribute":
                attr = base.child_by_field_name("attribute")
                if attr:
                    base_name = attr.text.decode("utf-8", errors="replace")
            elif base.type == "dotted_name":
                base_name = base.text.decode("utf-8", errors="replace").split(".")[-1]
            if base_name:
                base_qn = f"type::{base_name}"
                base_id = _add_node(graph, NodeKind.CLASS, base_qn, Path(""), 0, 0,
                                    props={"dialect": "python", "role": "base_class"})
                graph.add_edge(class_id, base_id, Edge.extracted(EdgeKind.INHERITS))

    # Walk class body
    body = node.child_by_field_name("body")
    if body:
        _walk_scope(body, graph, file_qn, path, class_id, child_scope, True, file_is_test)


def _handle_function(
    node: ts.Node,
    graph: Graph,
    file_qn: str,
    path: Path,
    parent_id: NodeId,
    scope: list[str],
    parent_is_class: bool,
    file_is_test: bool,
) -> None:
    name_node = node.child_by_field_name("name")
    if not name_node:
        return
    name = name_node.text.decode("utf-8", errors="replace")
    is_test = file_is_test or _is_test_name(name)

    child_scope = scope + [name]
    qn = f"{file_qn}::{'::'.join(child_scope)}"

    decorators = _extract_decorators(node)
    props: dict = {}
    if decorators:
        props["decorators"] = decorators
    if is_test:
        props["is_test"] = True

    # Determine kind: TYPE for TypeVar(), METHOD for class methods, FUNCTION otherwise
    if parent_is_class:
        kind = NodeKind.METHOD
    elif _is_typevar_definition(node):
        kind = NodeKind.TYPE
        props["role"] = "typevar"
    else:
        kind = NodeKind.FUNCTION
        # Add dialect property for language identification
        props["dialect"] = "python"

    fn_id = _add_node(graph, kind, qn, path,
                      node.start_point.row, node.end_point.row,
                      node.text.decode("utf-8", errors="replace"), props)
    graph.add_edge(parent_id, fn_id, Edge.extracted(EdgeKind.DEFINES))

    # Emit calls in function body
    body = node.child_by_field_name("body")
    if body:
        _emit_calls(body, graph, fn_id)
        _walk_scope(body, graph, file_qn, path, fn_id, child_scope, False, file_is_test)


def _emit_imports(
    node: ts.Node,
    graph: Graph,
    file_id: NodeId,
) -> None:
    for child in node.children:
        if child.type == "import_statement":
            for c in child.children:
                if c.type == "dotted_name":
                    path_text = c.text.decode("utf-8", errors="replace")
                    mod_qn = f"module::{path_text}"
                    mod_id = _add_node(graph, NodeKind.MODULE, mod_qn, Path(""), 0, 0,
                                       props={"dialect": "python"})
                    graph.add_edge(file_id, mod_id, Edge.extracted(EdgeKind.IMPORTS))
        elif child.type == "import_from_statement":
            module_name = child.child_by_field_name("module_name")
            if module_name:
                mod_qn = f"module::{module_name.text.decode('utf-8', errors='replace')}"
                mod_id = _add_node(graph, NodeKind.MODULE, mod_qn, Path(""), 0, 0,
                                   props={"dialect": "python"})
                graph.add_edge(file_id, mod_id, Edge.extracted(EdgeKind.IMPORTS))


def extract_file(
    path: Path, graph: Graph, *, file_qn: str | None = None, source_path: Path | None = None
) -> None:
    """Parse a Python source file and emit nodes/edges into the graph."""
    with open(path, "rb") as f:
        raw = f.read()

    lang = tree_sitter_python.language()
    parser = ts.Parser(ts.Language(tree_sitter_python.language()))
    tree = parser.parse(raw)
    root = tree.root_node

    record_path = source_path if source_path is not None else path
    file_qn = file_qn or path.stem
    file_id = _add_node(graph, NodeKind.FILE, file_qn, record_path, 0, 0,
                        raw.decode("utf-8", errors="replace"),
                        props={"dialect": "python"})

    _emit_imports(root, graph, file_id)
    _walk_scope(root, graph, file_qn, record_path, file_id, [], False,
                _is_test_name(path.stem))
