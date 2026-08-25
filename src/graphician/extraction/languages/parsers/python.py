"""Python source extraction using tree-sitter 0.26."""

from __future__ import annotations

from pathlib import Path

import tree_sitter as ts
import tree_sitter_python

from graphician.core.edge import Edge, EdgeKind
from graphician.core.graph import Graph
from graphician.core.id import NodeId
from graphician.core.node import Node, NodeKind


def _text(node: ts.Node | None, source: bytes) -> str:
    """Decode a node's source text via byte-slicing rather than node.text.

    node.text has triggered intermittent use-after-free segfaults in the
    tree-sitter 0.26 Python binding on complex trees; slicing the source
    buffer we already hold is equivalent and safe.
    """
    if node is None:
        return ""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _extract_decorators(node: ts.Node, source: bytes) -> list[str]:
    decorators = []
    for child in node.children:
        if child.type == "decorated":
            for c in child.children:
                if c.type == "decorator":
                    for cc in c.children:
                        if cc.type != "@":
                            text = _text(cc, source).strip()
                            if text:
                                decorators.append(text)
    return decorators if decorators else []


def _is_test_name(name: str) -> bool:
    return name.startswith("test_") or name.endswith("_test")


# Protocol/ABC base classes that make a class a trait
_PROTOCOL_BASES = frozenset(
    ["Protocol", "ABC", "typing.Protocol", "typing_extensions.Protocol"]
)


def _is_protocol_class(bases: ts.Node | None, source: bytes) -> bool:
    """Check if a class's base-list node includes Protocol or ABC (makes it a TRAIT)."""
    if bases is None:
        return False
    for base in bases.children:
        base_name = None
        if base.type == "identifier":
            base_name = _text(base, source)
        elif base.type == "attribute":
            attr = base.child_by_field_name("attribute")
            if attr:
                base_name = _text(attr, source)
        elif base.type == "dotted_name":
            base_name = _text(base, source)
        if base_name and base_name in _PROTOCOL_BASES:
            return True
    return False


def _is_type_alias(node: ts.Node, source: bytes) -> bool:
    """Check if a function definition is actually a type alias assignment."""
    # TypeAlias: Name = ... or Name: Type = ...
    # TypedDict: class Name(TypedDict): ...
    # TypeVar: Name = TypeVar("Name")
    assign = node.child_by_field_name("initializer")
    if assign is not None:
        # Check if RHS looks like a TypeVar, NewType, or type alias pattern
        body = node.child_by_field_name("body")
        if body is None:
            rhs_text = _text(assign, source).strip()
            if any(p in rhs_text for p in ["TypeVar(", "NewType(", "ParamSpec(", "TypeAlias"]):
                return True
    return False


def _is_typeddict_class(bases: ts.Node | None, source: bytes) -> bool:
    """Check if a class's base-list node includes TypedDict."""
    if bases is None:
        return False
    for base in bases.children:
        base_name = None
        if base.type == "identifier":
            base_name = _text(base, source)
        elif base.type == "attribute":
            attr = base.child_by_field_name("attribute")
            if attr:
                base_name = _text(attr, source)
        elif base.type == "dotted_name":
            base_name = _text(base, source)
        if base_name and "TypedDict" in base_name:
            return True
    return False


def _is_typevar_definition(body: ts.Node | None, source: bytes) -> bool:
    """Check if a function body is actually a TypeVar() call.

    Takes the already-fetched body node rather than re-fetching it via
    child_by_field_name: creating a second independent Node wrapper for the
    same underlying tree-sitter node has triggered use-after-free segfaults
    in the tree-sitter Python binding once one wrapper is garbage collected.
    """
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
                    name = _text(func, source)
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
    source: bytes,
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
                name = _text(func_node, source)
            elif func_node.type == "attribute":
                attr = func_node.child_by_field_name("attribute")
                if attr:
                    name = _text(attr, source)
                obj = func_node.child_by_field_name("object")
                if obj:
                    receiver = _text(obj, source)
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
    source: bytes,
) -> None:
    for child in node.children:
        if child.type == "decorated_definition":
            _walk_scope(
                child, graph, file_qn, path, parent_id, scope, parent_is_class, file_is_test,
                source,
            )
        elif child.type == "class_definition":
            _handle_class(child, graph, file_qn, path, parent_id, scope, file_is_test, source)
        elif child.type == "function_definition":
            _handle_function(
                child, graph, file_qn, path, parent_id, scope, parent_is_class, file_is_test,
                source,
            )


def _handle_class(
    node: ts.Node,
    graph: Graph,
    file_qn: str,
    path: Path,
    parent_id: NodeId,
    scope: list[str],
    file_is_test: bool,
    source: bytes,
) -> None:
    name_node = node.child_by_field_name("name")
    if not name_node:
        return
    name = _text(name_node, source)
    child_scope = [*scope, name]
    qn = f"{file_qn}::{'::'.join(child_scope)}"

    decorators = _extract_decorators(node, source)
    props: dict = {}
    if decorators:
        props["decorators"] = decorators

    # Fetch bases/body once and reuse below: re-fetching the same field via
    # child_by_field_name creates a second independent Node wrapper, which
    # has triggered use-after-free segfaults in the tree-sitter binding.
    bases = node.child_by_field_name("superclasses")
    for c in node.children:
        if bases is None and c.type in ("arguments", "argument_list", "base_list"):
            bases = c
            break
    body = node.child_by_field_name("body")

    # Determine kind: TRAIT for Protocol/ABC, TYPE for TypedDict, CLASS otherwise
    if _is_protocol_class(bases, source):
        kind = NodeKind.TRAIT
        props["role"] = "protocol"
    elif _is_typeddict_class(bases, source):
        kind = NodeKind.TYPE
        props["role"] = "typeddict"
    else:
        kind = NodeKind.CLASS

    # Add dialect property
    props["dialect"] = "python"

    class_id = _add_node(graph, kind, qn, path,
                         node.start_point.row, node.end_point.row,
                         _text(node, source), props)
    graph.add_edge(parent_id, class_id, Edge.extracted(EdgeKind.DEFINES))

    # Handle inheritance
    if bases:
        for base in bases.children:
            base_name = None
            if base.type == "identifier":
                base_name = _text(base, source)
            elif base.type == "attribute":
                attr = base.child_by_field_name("attribute")
                if attr:
                    base_name = _text(attr, source)
            elif base.type == "dotted_name":
                base_name = _text(base, source).split(".")[-1]
            if base_name:
                base_qn = f"type::{base_name}"
                base_id = _add_node(graph, NodeKind.CLASS, base_qn, Path(""), 0, 0,
                                    props={"dialect": "python", "role": "base_class"})
                graph.add_edge(class_id, base_id, Edge.extracted(EdgeKind.INHERITS))

    # Walk class body
    if body:
        _walk_scope(body, graph, file_qn, path, class_id, child_scope, True, file_is_test, source)


def _handle_function(
    node: ts.Node,
    graph: Graph,
    file_qn: str,
    path: Path,
    parent_id: NodeId,
    scope: list[str],
    parent_is_class: bool,
    file_is_test: bool,
    source: bytes,
) -> None:
    name_node = node.child_by_field_name("name")
    if not name_node:
        return
    name = _text(name_node, source)
    is_test = file_is_test or _is_test_name(name)

    child_scope = [*scope, name]
    qn = f"{file_qn}::{'::'.join(child_scope)}"

    decorators = _extract_decorators(node, source)
    props: dict = {}
    if decorators:
        props["decorators"] = decorators
    if is_test:
        props["is_test"] = True

    # Fetch the body once and reuse it below: re-fetching the same field via
    # child_by_field_name creates a second independent Node wrapper, which
    # has triggered use-after-free segfaults in the tree-sitter binding.
    body = node.child_by_field_name("body")

    # Determine kind: TYPE for TypeVar(), METHOD for class methods, FUNCTION otherwise
    if parent_is_class:
        kind = NodeKind.METHOD
    elif _is_typevar_definition(body, source):
        kind = NodeKind.TYPE
        props["role"] = "typevar"
    else:
        kind = NodeKind.FUNCTION
        # Add dialect property for language identification
        props["dialect"] = "python"

    fn_id = _add_node(graph, kind, qn, path,
                      node.start_point.row, node.end_point.row,
                      _text(node, source), props)
    graph.add_edge(parent_id, fn_id, Edge.extracted(EdgeKind.DEFINES))

    # Emit calls in function body
    if body:
        _emit_calls(body, graph, fn_id, source)
        _walk_scope(body, graph, file_qn, path, fn_id, child_scope, False, file_is_test, source)


def _emit_imports(
    node: ts.Node,
    graph: Graph,
    file_id: NodeId,
    source: bytes,
) -> None:
    for child in node.children:
        if child.type == "import_statement":
            for c in child.children:
                if c.type == "dotted_name":
                    path_text = _text(c, source)
                    mod_qn = f"module::{path_text}"
                    mod_id = _add_node(graph, NodeKind.MODULE, mod_qn, Path(""), 0, 0,
                                       props={"dialect": "python"})
                    graph.add_edge(file_id, mod_id, Edge.extracted(EdgeKind.IMPORTS))
        elif child.type == "import_from_statement":
            module_name = child.child_by_field_name("module_name")
            if module_name:
                module_path = _text(module_name, source)
                imported_symbols: dict[str, str] = {}
                seen_import = False
                for imported in child.children:
                    if imported.type == "import":
                        seen_import = True
                        continue
                    if not seen_import:
                        continue
                    if imported.type == "dotted_name":
                        original = _text(imported, source).rsplit(".", 1)[-1]
                        imported_symbols[original] = original
                    elif imported.type == "aliased_import":
                        alias = imported.child_by_field_name("alias")
                        name = imported.child_by_field_name("name")
                        if name is not None:
                            original = _text(name, source).rsplit(".", 1)[-1]
                            local_name = _text(alias, source) if alias is not None else original
                            imported_symbols[local_name] = original
                mod_qn = f"module::{module_path}"
                mod_id = _add_node(graph, NodeKind.MODULE, mod_qn, Path(""), 0, 0,
                                   props={"dialect": "python"})
                import_edge = Edge.extracted(EdgeKind.IMPORTS)
                import_edge.properties["module_path"] = module_path
                import_edge.properties["imported_symbols"] = imported_symbols
                graph.add_edge(file_id, mod_id, import_edge)


def extract_file(
    path: Path, graph: Graph, *, file_qn: str | None = None, source_path: Path | None = None
) -> None:
    """Parse a Python source file and emit nodes/edges into the graph."""
    with open(path, "rb") as f:
        raw = f.read()

    parser = ts.Parser(ts.Language(tree_sitter_python.language()))
    tree = parser.parse(raw)
    root = tree.root_node

    record_path = source_path if source_path is not None else path
    file_qn = file_qn or path.stem
    file_id = _add_node(graph, NodeKind.FILE, file_qn, record_path, 0, 0,
                        raw.decode("utf-8", errors="replace"),
                        props={"dialect": "python"})

    _emit_imports(root, graph, file_id, raw)
    _walk_scope(root, graph, file_qn, record_path, file_id, [], False,
                _is_test_name(path.stem), raw)
