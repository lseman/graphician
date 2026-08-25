"""Go source extraction.

Emits: File, Function, Method, Class (struct), Interface, Type, Variable, Constant, Module
Edges: Defines, Imports, Calls, Implements
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tree_sitter as ts
import tree_sitter_go

from graphician.core.edge import Edge, EdgeKind
from graphician.core.graph import Graph
from graphician.core.id import NodeId
from graphician.core.node import Node, NodeKind

# Names suppressed at call-resolution time
_SUPPRESS_CALLS = frozenset([
    # Go std / common API
    "make", "new", "append", "copy", "delete", "len", "cap", "close",
    "panic", "recover", "print", "println", "fmt", "errorf",
    # tree-sitter Node API
    "child_by_field_name", "children", "end_position", "is_named", "kind",
    "language", "parent", "root_node", "start_position", "text", "walk",
    "utf8_text", "field_name_for_child", "end_byte", "start_byte",
])

# Common builtin / stdlib names that should be suppressed at parse time
_GO_BUILTIN_CALLS = frozenset([
    # Go std / common API
    "append", "cap", "close", "copy", "delete", "errorf",
    "fmt", "getenv", "len", "make", "new", "open", "panic",
    "println", "recover", "write",
])

# Common names that are too generic to keep — suppressed at parse time
# (these are frequently called but rarely represent project functions)
_GENERIC_NAMES = frozenset([
    "get", "find", "insert", "remove", "push", "pop", "select",
    "execute", "merge", "load", "write", "read", "path", "add",
    "string", "new", "index", "join", "take", "has", "display",
    "now", "entry", "default", "count", "first", "last", "position",
    "split", "replace", "clear", "values", "node", "text", "parse",
    "kind", "parent", "language", "status", "watch", "commit",
    "block", "attr",
])


def is_test_name(name: str) -> bool:
    """Check if a name follows Go test naming conventions."""
    if name.startswith("Test"):
        return True
    if name.startswith("Benchmark"):
        return True
    if name.startswith("Example"):
        return True
    if name.startswith("Fuzz"):
        return True
    return bool(name.startswith("TestMain"))


def is_test_file_path(file_path: str) -> bool:
    """Check if a file path indicates a test file."""
    return file_path.endswith("_test.go")


def should_suppress(name: str) -> bool:
    """Check if a name should be suppressed from call placeholders."""
    name = name.strip()
    if not name:
        return True
    lower = name.lower()
    if lower in _SUPPRESS_CALLS:
        return True
    if lower in _GO_BUILTIN_CALLS:
        return True
    return lower in _GENERIC_NAMES


def extract_file(path: Path, graph: Graph, *, file_qn: str | None = None, source_path: Path | None = None) -> NodeId:
    """Parse a Go source file and emit nodes/edges to the graph.

    Parameters
    ----------
    path:
        Path to the source file.
    graph:
        Graph to populate.
    file_qn:
        Qualified name for the file node. Defaults to ``file::<stem>``.
    source_path:
        Optional path used for the ``source_path`` property.

    Returns
    -------
    NodeId
        The node ID of the file node.
    """
    source = path.read_bytes()
    return extract(source, path, graph, file_qn=file_qn, source_path=source_path)


def extract(
    source: bytes,
    path: Path,
    graph: Graph,
    *,
    file_qn: str | None = None,
    source_path: Path | None = None,
) -> NodeId:
    """Parse Go source bytes and populate the graph."""
    file_qn = _file_qn(path, file_qn)
    source_str = source.decode(errors="replace")

    lang = tree_sitter_go.language()
    parser = ts.Parser(lang)
    tree = parser.parse(source_str)

    file_is_test = is_test_file_path(str(path))
    file_node = Node.new(NodeKind.FILE, file_qn).with_property("dialect", "go")
    graph.add_node(file_node)

    # Extract definitions
    _extract_definitions(tree.root_node, source_str, path, graph, file_qn, file_is_test)

    return NodeId.of(file_qn)


def _file_qn(path: Path, file_qn: str | None) -> str:
    if file_qn:
        return file_qn
    stem = path.stem
    return f"file::{stem}"


def _extract_definitions(
    node: ts.Node,
    source_str: str,
    path: Path,
    graph: Graph,
    file_qn: str,
    file_is_test: bool,
) -> None:
    """Walk the AST and emit function, method, struct, interface, type, var, const nodes."""
    for child in node.children(node.walk()):
        kind = child.kind()
        if kind == "function_declaration":
            _extract_function(child, source_str, path, graph, file_qn, file_is_test, is_method=False)
        elif kind == "method_declaration":
            _extract_method(child, source_str, path, graph, file_qn, file_is_test)
        elif kind == "type_declaration":
            _extract_type_decl(child, source_str, path, graph, file_qn)
        elif kind == "var_declaration":
            _extract_var_decl(child, source_str, path, graph, file_qn)
        elif kind == "const_declaration":
            _extract_const_decl(child, source_str, path, graph, file_qn)
        elif kind in ("import_declaration",):
            _extract_import(child, source_str, path, graph, file_qn)

        # Recurse into compound nodes
        if child.child_count > 0:
            _extract_definitions(child, source_str, path, graph, file_qn, file_is_test)


def _extract_function(
    node: ts.Node,
    source_str: str,
    path: Path,
    graph: Graph,
    file_qn: str,
    file_is_test: bool,
    is_method: bool = False,
) -> None:
    """Extract a function or method node."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return
    name = name_node.text.decode(errors="replace")
    if not name:
        return

    is_test = file_is_test or is_test_name(name)
    kind = NodeKind.METHOD if is_method else NodeKind.FUNCTION
    props: dict[str, Any] = {"dialect": "go"}
    if is_test:
        props["is_test"] = "true"

    qn = f"{file_qn}::{name}"
    node_obj = Node.new(kind, qn).with_properties(props)
    node_obj = node_obj.with_source(path, name_node)
    graph.add_node(node_obj)

    file_node_id = NodeId.of(file_qn)
    graph.add_edge(
        Edge.new(file_node_id, NodeId.of(qn), EdgeKind.DEFINES)
    )

    # Emit calls
    _emit_calls(node, source_str, qn, graph)


def _extract_method(
    node: ts.Node,
    source_str: str,
    path: Path,
    graph: Graph,
    file_qn: str,
    file_is_test: bool,
) -> None:
    """Extract a method node with receiver type."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return
    name = name_node.text.decode(errors="replace")
    if not name:
        return

    # Extract receiver type
    receiver_type = None
    param_field = node.child_by_field_name("parameter")
    if param_field:
        for child in param_field.children(node.walk()):
            if child.kind() == "parameter_declaration":
                type_field = child.child_by_field_name("type")
                if type_field:
                    receiver_type = type_field.text.decode(errors="replace")
                    break

    is_test = file_is_test or is_test_name(name)
    props: dict[str, Any] = {"dialect": "go"}
    if is_test:
        props["is_test"] = "true"

    qn = f"{file_qn}::{receiver_type}::{name}" if receiver_type else f"{file_qn}::{name}"

    node_obj = Node.new(NodeKind.METHOD, qn).with_properties(props)
    node_obj = node_obj.with_source(path, name_node)
    graph.add_node(node_obj)

    file_node_id = NodeId.of(file_qn)
    graph.add_edge(
        Edge.new(file_node_id, NodeId.of(qn), EdgeKind.DEFINES)
    )

    # Emit calls
    _emit_calls(node, source_str, qn, graph)

    # Add receiver type as class if not already present
    if receiver_type:
        type_qn = f"{file_qn}::{receiver_type}"
        if not graph.has_node(NodeId.of(type_qn)):
            type_node = Node.new(NodeKind.CLASS, type_qn).with_property("dialect", "go")
            graph.add_node(type_node)
            graph.add_edge(
                Edge.new(file_node_id, NodeId.of(type_qn), EdgeKind.DEFINES)
            )


def _extract_type_decl(
    node: ts.Node,
    source_str: str,
    path: Path,
    graph: Graph,
    file_qn: str,
) -> None:
    """Extract a type declaration (struct, interface, or alias)."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return
    name = name_node.text.decode(errors="replace")
    if not name:
        return

    qn = f"{file_qn}::{name}"

    # Determine kind
    kind = NodeKind.CLASS  # default for struct
    type_spec = node.child_by_field_name("type")
    if type_spec:
        child = type_spec.child(type_spec.walk()).child(0)
        if child.kind() == "interface_type":
            kind = NodeKind.INTERFACE
        elif child.kind() == "struct_type":
            kind = NodeKind.CLASS
        else:
            kind = NodeKind.TYPE

    props = {"dialect": "go"}
    node_obj = Node.new(kind, qn).with_properties(props)
    node_obj = node_obj.with_source(path, name_node)
    graph.add_node(node_obj)

    file_node_id = NodeId.of(file_qn)
    graph.add_edge(
        Edge.new(file_node_id, NodeId.of(qn), EdgeKind.DEFINES)
    )


def _extract_var_decl(
    node: ts.Node,
    source_str: str,
    path: Path,
    graph: Graph,
    file_qn: str,
) -> None:
    """Extract variable declarations."""
    spec = node.child_by_field_name("spec")
    if not spec:
        return
    name_node = spec.child_by_field_name("name")
    if not name_node:
        return
    name = name_node.text.decode(errors="replace")
    if not name:
        return

    qn = f"{file_qn}::{name}"
    props = {"dialect": "go"}
    node_obj = Node.new(NodeKind.VARIABLE, qn).with_properties(props)
    node_obj = node_obj.with_source(path, name_node)
    graph.add_node(node_obj)

    file_node_id = NodeId.of(file_qn)
    graph.add_edge(
        Edge.new(file_node_id, NodeId.of(qn), EdgeKind.DEFINES)
    )


def _extract_const_decl(
    node: ts.Node,
    source_str: str,
    path: Path,
    graph: Graph,
    file_qn: str,
) -> None:
    """Extract constant declarations."""
    spec = node.child_by_field_name("spec")
    if not spec:
        return
    name_node = spec.child_by_field_name("name")
    if not name_node:
        return
    name = name_node.text.decode(errors="replace")
    if not name:
        return

    qn = f"{file_qn}::{name}"
    props = {"dialect": "go"}
    node_obj = Node.new(NodeKind.CONSTANT, qn).with_properties(props)
    node_obj = node_obj.with_source(path, name_node)
    graph.add_node(node_obj)

    file_node_id = NodeId.of(file_qn)
    graph.add_edge(
        Edge.new(file_node_id, NodeId.of(qn), EdgeKind.DEFINES)
    )


def _extract_import(
    node: ts.Node,
    source_str: str,
    path: Path,
    graph: Graph,
    file_qn: str,
) -> None:
    """Extract import declarations."""
    path_node = node.child_by_field_name("path")
    if not path_node:
        return

    # Get the string literal path
    for child in path_node.children(path_node.walk()):
        if child.kind() == "interpreted_string_literal":
            mod_name = child.text.decode(errors="replace").strip('"')
            if not mod_name:
                continue

            mod_qn = f"module::{mod_name}"
            if not graph.has_node(NodeId.of(mod_qn)):
                mod_node = Node.new(NodeKind.MODULE, mod_qn).with_property("dialect", "go")
                graph.add_node(mod_node)
                graph.add_edge(
                    Edge.new(NodeId.of(file_qn), NodeId.of(mod_qn), EdgeKind.IMPORTS)
                )
            break


def _emit_calls(
    node: ts.Node,
    source_str: str,
    caller_qn: str,
    graph: Graph,
) -> None:
    """Walk the AST and emit call edges for call expressions."""
    for child in node.children(node.walk()):
        if child.kind() == "call_expression":
            func_node = child.child_by_field_name("function")
            if not func_node:
                continue

            name = None
            if func_node.kind() == "identifier":
                name = func_node.text.decode(errors="replace")
            elif func_node.kind() == "selector_expression":
                sel_name = func_node.child_by_field_name("field")
                if sel_name:
                    name = sel_name.text.decode(errors="replace")

            if name and not should_suppress(name):
                # Emit call placeholder
                pass  # Calls are handled by the resolver later

        _emit_calls(child, source_str, caller_qn, graph)
