"""Go source extraction.

Emits: File, Function, Method, Class (struct), Type (interface/alias), Variable
(includes const), Module (imports)
Edges: Defines, Imports, Calls
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


def _text(node: ts.Node) -> str:
    raw = node.text
    if raw is None:
        return ""
    return raw.decode("utf-8", errors="replace")


def _add_node(
    graph: Graph,
    kind: NodeKind,
    qn: str,
    path: Path,
    line_start: int,
    line_end: int,
    props: dict[str, Any] | None = None,
) -> NodeId:
    """Create (or fetch) a node for ``qn``, attaching source location and properties."""
    existing = graph.find_by_qname(qn)
    if existing is not None:
        return existing
    node = Node.new(kind, qn)
    node = node.with_source(str(path), line_start + 1, line_end + 1)
    default_props: dict[str, Any] = {"dialect": "go"}
    if props:
        default_props.update(props)
    for k, v in default_props.items():
        node = node.with_property(k, v)
    return graph.add_node(node)


def _emit_call(graph: Graph, caller_id: NodeId, name: str) -> None:
    if should_suppress(name):
        return
    callee_qn = f"call::{name}"
    callee_id = _add_node(graph, NodeKind.FUNCTION, callee_qn, Path(""), 0, 0)
    edge = Edge.ambiguous(EdgeKind.CALLS)
    graph.add_edge(caller_id, callee_id, edge)


def extract_file(
    path: Path,
    graph: Graph,
    *,
    file_qn: str | None = None,
    source_path: Path | None = None,
) -> None:
    """Parse a Go source file and emit nodes/edges to the graph."""
    with open(path, "rb") as f:
        source = f.read()
    extract(source, path, graph, file_qn=file_qn, source_path=source_path)


def extract(
    source: bytes,
    path: Path,
    graph: Graph,
    *,
    file_qn: str | None = None,
    source_path: Path | None = None,
) -> None:
    """Parse Go source bytes and populate the graph."""
    file_qn = _file_qn(path, file_qn)
    record_path = source_path if source_path is not None else path

    language = ts.Language(tree_sitter_go.language())
    parser = ts.Parser(language)
    tree = parser.parse(source)

    file_is_test = is_test_file_path(str(path))
    file_id = _add_node(graph, NodeKind.FILE, file_qn, record_path, 0, 0)

    _extract_definitions(tree.root_node, record_path, graph, file_qn, file_id, file_is_test)


def _file_qn(path: Path, file_qn: str | None) -> str:
    if file_qn:
        return file_qn
    return f"file::{path.stem}"


def _extract_definitions(
    node: ts.Node,
    path: Path,
    graph: Graph,
    file_qn: str,
    file_id: NodeId,
    file_is_test: bool,
) -> None:
    """Walk the AST and emit function, method, struct, interface, type, var, const nodes."""
    for child in node.children:
        kind = child.type
        if kind == "function_declaration":
            _extract_function(child, path, graph, file_qn, file_id, file_is_test)
        elif kind == "method_declaration":
            _extract_method(child, path, graph, file_qn, file_id, file_is_test)
        elif kind == "type_declaration":
            _extract_type_decl(child, path, graph, file_qn, file_id)
        elif kind in ("var_declaration", "const_declaration"):
            _extract_var_or_const_decl(child, path, graph, file_qn, file_id, NodeKind.VARIABLE)
        elif kind == "import_declaration":
            _extract_import(child, graph, file_qn, file_id)
        else:
            # Recurse into compound nodes (e.g. top-level blocks) looking for
            # further declarations; function/method bodies are handled by
            # their own extractors and are not recursed into here.
            if child.child_count > 0:
                _extract_definitions(child, path, graph, file_qn, file_id, file_is_test)


def _extract_function(
    node: ts.Node,
    path: Path,
    graph: Graph,
    file_qn: str,
    file_id: NodeId,
    file_is_test: bool,
) -> None:
    """Extract a top-level function declaration."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return
    name = _text(name_node)
    if not name:
        return

    is_test = file_is_test or is_test_name(name)
    props: dict[str, Any] = {"is_test": True} if is_test else {}

    qn = f"{file_qn}::{name}"
    fn_id = _add_node(
        graph, NodeKind.FUNCTION, qn, path,
        node.start_point.row, node.end_point.row, props,
    )
    graph.add_edge(file_id, fn_id, Edge.extracted(EdgeKind.DEFINES))

    body = node.child_by_field_name("body")
    if body is not None:
        _emit_calls(body, fn_id, graph)


def _extract_method(
    node: ts.Node,
    path: Path,
    graph: Graph,
    file_qn: str,
    file_id: NodeId,
    file_is_test: bool,
) -> None:
    """Extract a method declaration, keyed by its receiver type."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return
    name = _text(name_node)
    if not name:
        return

    # Receiver type, e.g. `func (r Rect) Area() ...` -> "Rect".
    receiver_type = None
    receiver = node.child_by_field_name("receiver")
    if receiver is not None:
        for child in receiver.children:
            if child.type == "parameter_declaration":
                type_field = child.child_by_field_name("type")
                if type_field is not None:
                    # Receivers may be pointer types (`*Rect`); strip the `*`.
                    receiver_type = _text(type_field).lstrip("*")
                    break

    is_test = file_is_test or is_test_name(name)
    props: dict[str, Any] = {"is_test": True} if is_test else {}

    qn = f"{file_qn}::{receiver_type}::{name}" if receiver_type else f"{file_qn}::{name}"
    method_id = _add_node(
        graph, NodeKind.METHOD, qn, path,
        node.start_point.row, node.end_point.row, props,
    )
    graph.add_edge(file_id, method_id, Edge.extracted(EdgeKind.DEFINES))

    body = node.child_by_field_name("body")
    if body is not None:
        _emit_calls(body, method_id, graph)

    # Ensure the receiver type has a node, even if its own type_declaration
    # wasn't visited yet (e.g. defined in another file).
    if receiver_type:
        type_qn = f"{file_qn}::{receiver_type}"
        if graph.find_by_qname(type_qn) is None:
            type_id = _add_node(graph, NodeKind.CLASS, type_qn, path, 0, 0)
            graph.add_edge(file_id, type_id, Edge.extracted(EdgeKind.DEFINES))


def _extract_type_decl(
    node: ts.Node,
    path: Path,
    graph: Graph,
    file_qn: str,
    file_id: NodeId,
) -> None:
    """Extract type declarations: struct, interface, alias, or named type."""
    for spec in node.children:
        if spec.type == "type_spec":
            _extract_type_spec(spec, path, graph, file_qn, file_id)
        elif spec.type == "type_alias":
            name_node = spec.child(0)
            if name_node is None:
                continue
            name = _text(name_node)
            if not name:
                continue
            qn = f"{file_qn}::{name}"
            type_id = _add_node(
                graph, NodeKind.TYPE, qn, path,
                spec.start_point.row, spec.end_point.row,
            )
            graph.add_edge(file_id, type_id, Edge.extracted(EdgeKind.DEFINES))


def _extract_type_spec(
    node: ts.Node,
    path: Path,
    graph: Graph,
    file_qn: str,
    file_id: NodeId,
) -> None:
    name_node = node.child_by_field_name("name")
    if not name_node:
        return
    name = _text(name_node)
    if not name:
        return

    kind = NodeKind.TYPE
    type_spec_kind = node.child_by_field_name("type")
    if type_spec_kind is not None:
        if type_spec_kind.type == "interface_type":
            kind = NodeKind.TYPE
        elif type_spec_kind.type == "struct_type":
            kind = NodeKind.CLASS

    qn = f"{file_qn}::{name}"
    type_id = _add_node(
        graph, kind, qn, path,
        node.start_point.row, node.end_point.row,
    )
    graph.add_edge(file_id, type_id, Edge.extracted(EdgeKind.DEFINES))


def _extract_var_or_const_decl(
    node: ts.Node,
    path: Path,
    graph: Graph,
    file_qn: str,
    file_id: NodeId,
    kind: NodeKind,
) -> None:
    """Extract var/const declarations, including grouped `var ( ... )` blocks."""
    spec_type = "var_spec" if node.type == "var_declaration" else "const_spec"
    for child in node.children:
        if child.type == spec_type:
            _extract_spec(child, path, graph, file_qn, file_id, kind)
        elif child.type in ("var_spec_list", "const_spec_list"):
            for spec in child.children:
                if spec.type == spec_type:
                    _extract_spec(spec, path, graph, file_qn, file_id, kind)


def _extract_spec(
    spec: ts.Node,
    path: Path,
    graph: Graph,
    file_qn: str,
    file_id: NodeId,
    kind: NodeKind,
) -> None:
    name_node = spec.child_by_field_name("name")
    if not name_node:
        return
    name = _text(name_node)
    if not name:
        return

    qn = f"{file_qn}::{name}"
    node_id = _add_node(
        graph, kind, qn, path,
        spec.start_point.row, spec.end_point.row,
    )
    graph.add_edge(file_id, node_id, Edge.extracted(EdgeKind.DEFINES))


def _extract_import(
    node: ts.Node,
    graph: Graph,
    file_qn: str,
    file_id: NodeId,
) -> None:
    """Extract import declarations, including grouped `import ( ... )` blocks."""
    for spec in _iter_import_specs(node):
        path_node = spec.child_by_field_name("path")
        if path_node is None:
            continue
        raw = _text(path_node).strip('"')
        if not raw:
            continue

        mod_qn = f"module::{raw}"
        mod_id = graph.find_by_qname(mod_qn)
        if mod_id is None:
            mod_node = Node.new(NodeKind.MODULE, mod_qn).with_property("dialect", "go")
            mod_id = graph.add_node(mod_node)
        graph.add_edge(file_id, mod_id, Edge.extracted(EdgeKind.IMPORTS))


def _iter_import_specs(node: ts.Node) -> list[ts.Node]:
    specs: list[ts.Node] = []
    for child in node.children:
        if child.type == "import_spec":
            specs.append(child)
        elif child.type == "import_spec_list":
            specs.extend(c for c in child.children if c.type == "import_spec")
    return specs


def _emit_calls(node: ts.Node, caller_id: NodeId, graph: Graph) -> None:
    """Walk a function/method body and emit call placeholder edges."""
    if node.type == "call_expression":
        func_node = node.child_by_field_name("function")
        if func_node is not None:
            name = None
            if func_node.type == "identifier":
                name = _text(func_node)
            elif func_node.type == "selector_expression":
                field = func_node.child_by_field_name("field")
                if field is not None:
                    name = _text(field)
            if name:
                _emit_call(graph, caller_id, name)

    for child in node.children:
        _emit_calls(child, caller_id, graph)
