"""Rust source extraction.

Emits: Function, Method, Class (struct), Trait, Type (enum), Module (mod/use), File
Edges: Defines, Imports, Calls
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tree_sitter as ts
import tree_sitter_rust

from ariadne_py.core.edge import Edge, EdgeKind
from ariadne_py.core.graph import Graph
from ariadne_py.core.id import NodeId
from ariadne_py.core.node import Node, NodeKind


_SUPPRESS_CALLS = frozenset(["std::panic", "std::result", "std::option", "std::vec",
    "std::vec::Vec", "std::option::Option", "std::result::Result",
    "std::boxed::Box", "std::sync", "std::rc", "std::cell",
    "std::mem", "std::convert", "std::ops", "std::fmt"])


def _text(node: ts.Node) -> str:
    raw = node.text
    if raw is None:
        return ""
    return raw.decode("utf-8", errors="replace")


def _child_by_field(node: ts.Node, field: str) -> ts.Node | None:
    return node.child_by_field_name(field)


def _add_node(
    graph: Graph,
    kind: NodeKind,
    qn: str,
    path: Path,
    line_start: int,
    line_end: int,
    source: str | None = None,
    props: dict[str, Any] | None = None,
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
    result = graph.find_by_qname(qn)
    assert result is not None, f"Node not found after add: {qn}"
    return result


def _emit_call(
    graph: Graph,
    caller_id: NodeId,
    name: str,
    scope: str | None = None,
    receiver: str | None = None,
) -> None:
    if name in _SUPPRESS_CALLS:
        return
    callee_qn = f"call::{name}"
    callee_id = _add_node(graph, NodeKind.FUNCTION, callee_qn, Path(""), 0, 0)
    if callee_id is None:
        return
    edge = Edge.ambiguous(EdgeKind.CALLS)
    if scope:
        edge.properties["call_scope"] = scope
    if receiver:
        edge.properties["call_receiver"] = receiver
    graph.add_edge(caller_id, callee_id, edge)


def _has_test_attr(node: ts.Node) -> bool:
    """Check if node is preceded by attribute_item siblings marking it as a test."""
    parent = node.parent
    if not parent:
        return False
    cursor = parent.walk()
    siblings = list(parent.children)
    try:
        idx = next(i for i, s in enumerate(siblings) if s.id == node.id)
    except StopIteration:
        return False
    # Walk backwards from the node, collecting preceding attribute_items
    for sib in reversed(siblings[:idx]):
        if sib.type == "attribute_item":
            if _attribute_marks_test(sib):
                return True
        elif sib.is_named:
            break
    return False


def _attribute_marks_test(attr_node: ts.Node) -> bool:
    """Check if an attribute_item marks something as a test."""
    raw = _text(attr_node)
    inner = raw.strip()
    if inner.startswith("#["):
        inner = inner[2:]
    if inner.endswith("]"):
        inner = inner[:-1]
    inner = inner.strip()
    head = inner.split("(")[0].strip()
    if head in ("test", "rstest"):
        return True
    if head.endswith("::test") or head == "test_case":
        return True
    return False


def _find_cfg_test_mod_ranges(root: ts.Node, source: bytes) -> list[tuple[int, int]]:
    """Find byte ranges of #[cfg(test)] mod blocks for test detection."""
    ranges: list[tuple[int, int]] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "mod_item" and _preceding_marks_cfg_test(node, source):
            ranges.append((node.start_byte, node.end_byte))
        stack.extend(list(node.children))
    return ranges


def _preceding_marks_cfg_test(node: ts.Node, source: bytes) -> bool:
    """True iff node is preceded by #[cfg(test)] attribute."""
    parent = node.parent
    if not parent:
        return False
    cursor = parent.walk()
    siblings = list(parent.children)
    try:
        idx = next(i for i, s in enumerate(siblings) if s.id == node.id)
    except StopIteration:
        return False
    for sib in reversed(siblings[:idx]):
        if sib.type == "attribute_item":
            raw = _text(sib)
            if "cfg(test)" in raw or "cfg ( test )" in raw:
                return True
        elif sib.is_named:
            break
    return False


def _in_any_range(byte_range: tuple[int, int], ranges: list[tuple[int, int]]) -> bool:
    """True if a node's byte range falls inside any of the given ranges."""
    s, e = byte_range
    return any(rs <= s and e <= re for rs, re in ranges)


def _has_method_parent(node: ts.Node) -> bool:
    """True iff node is directly inside an impl/trait block."""
    cur = node.parent
    while cur:
        if cur.type in ("impl_item", "trait_item"):
            return True
        if cur.type == "declaration_list":
            cur = cur.parent
            continue
        return False
    return False


def _rust_scope(node: ts.Node) -> list[str]:
    scope = []
    original_id = node.id
    cur = node.parent
    while cur:
        if cur.type == "impl_item":
            impl_name = _impl_type_name(cur)
            if impl_name:
                scope.append(impl_name)
        elif cur.type == "trait_item":
            for c in cur.children:
                if c.type in ("type_identifier", "scoped_type_identifier"):
                    scope.append(_text(c))
                    break
        elif cur.type == "mod_item":
            for c in cur.children:
                if c.type == "identifier":
                    scope.append(_text(c))
                    break
        elif cur.type == "function_item" and cur.id != original_id:
            for c in cur.children:
                if c.type == "identifier":
                    scope.append(_text(c))
                    break
        cur = cur.parent
    scope.reverse()
    return scope


def _call_target_name(node: ts.Node) -> tuple[str, str | None] | None:
    if node.type == "identifier":
        return (_text(node), None)
    if node.type in ("property_identifier", "field_identifier"):
        return (_text(node), None)
    if node.type in ("scoped_identifier", "scoped_type_identifier"):
        text = _text(node)
        parts = text.split("::")
        return (parts[-1], "::".join(parts[:-1]) if len(parts) > 1 else None)
    if node.type == "field_expression":
        field = _child_by_field(node, "field")
        if field:
            return (_text(field), None)
    return None


def _impl_type_name(node: ts.Node) -> str | None:
    """Extract the type being impl'd, handling generic types like Foo<T>."""
    for c in node.children:
        if c.type in ("type_identifier", "scoped_type_identifier", "generic_type"):
            text = _text(c)
            return text.split("<")[0].strip()
    return None


def _extract_receiver(func_node: ts.Node) -> str | None:
    if func_node.type == "field_expression":
        obj = _child_by_field(func_node, "object")
        if obj:
            return _text(obj)
    if func_node.type in ("scoped_identifier", "scoped_type_identifier"):
        text = _text(func_node)
        return text.split("::")[0] if "::" in text else None
    return None


def _emit_calls_in_body(
    graph: Graph,
    caller_id: NodeId,
    body: ts.Node,
) -> None:
    skip = frozenset(["function_item", "closure_expression"])
    stack = list(body.children)
    while stack:
        node = stack.pop()
        if node.type in skip:
            continue
        if node.type == "call_expression":
            func = None
            for i, c in enumerate(node.children):
                if node.field_name_for_child(i) == "function":
                    func = c
                    break
            if func:
                result = _call_target_name(func)
                if result:
                    name, scope = result
                    receiver = _extract_receiver(func)
                    _emit_call(graph, caller_id, name, scope, receiver)
        elif node.type == "macro_invocation":
            _emit_macro_calls(graph, caller_id, node)
        stack.extend(list(node.children))


def _emit_macro_calls(
    graph: Graph,
    caller_id: NodeId,
    macro_node: ts.Node,
) -> None:
    stack = [macro_node]
    while stack:
        node = stack.pop()
        if node.type == "token_tree":
            kids = list(node.children)
            for i in range(len(kids) - 1):
                if kids[i].type != "identifier":
                    continue
                nxt = kids[i + 1]
                if nxt.type != "token_tree":
                    continue
                inner = list(nxt.children)
                if not inner or inner[0].type != "(":
                    continue
                name = _text(kids[i])
                if name in ("return", "if", "else", "let", "match",
                            "for", "while", "loop", "in", "mut",
                            "ref", "as", "move"):
                    continue
                _emit_call(graph, caller_id, name)
        stack.extend(list(node.children))


def _walk_enum_body(
    graph: Graph,
    enum_id: NodeId,
    enum_qn: str,
    body: ts.Node,
) -> None:
    """Walk an enum body and emit Variant Variable nodes."""
    for child in body.children:
        if child.type == "enum_variant":
            name_node = child.child_by_field_name("name")
            if not name_node:
                continue
            name = _text(name_node)
            var_qn = f"{enum_qn}::{name}"
            var_id = _add_node(graph, NodeKind.VARIABLE, var_qn, Path(""),
                               child.start_point.row, child.end_point.row)
            graph.add_edge(enum_id, var_id, Edge.extracted(EdgeKind.DEFINES))
            # Walk variant fields (named fields like Red { value: i32 })
            fields = child.child_by_field_name("fields")
            if fields:
                _walk_enum_fields(graph, var_id, var_qn, fields)


def _walk_enum_fields(
    graph: Graph,
    var_id: NodeId,
    var_qn: str,
    fields: ts.Node,
) -> None:
    """Walk enum variant fields (named or positional)."""
    for child in fields.children:
        if child.type in ("field_declaration", "field_identifier"):
            name_node = child.child_by_field_name("name") if child.type == "field_declaration" else child
            if name_node:
                name = _text(name_node)
                field_qn = f"{var_qn}::{name}"
                field_id = _add_node(graph, NodeKind.VARIABLE, field_qn, Path(""),
                                     child.start_point.row, child.end_point.row)
                if field_id is not None:
                    graph.add_edge(var_id, field_id, Edge.extracted(EdgeKind.DEFINES))
        elif child.type == "primitive_type":
            # Positional field: Blue(i32)
            name = _text(child)
            field_qn = f"{var_qn}::{name}"
            field_id = _add_node(graph, NodeKind.VARIABLE, field_qn, Path(""),
                                 child.start_point.row, child.end_point.row)
            if field_id is not None:
                graph.add_edge(var_id, field_id, Edge.extracted(EdgeKind.DEFINES))


def _walk_impl_body(
    graph: Graph,
    file_qn: str,
    impl_id: NodeId,
    body: ts.Node,
    impl_node: ts.Node,
) -> None:
    """Walk an impl body for methods and call edges."""
    # Determine impl context for QN scoping
    impl_type = None
    trait_name = None
    for i, c in enumerate(impl_node.children):
        fn = impl_node.field_name_for_child(i)
        if fn == "type" and c.type in ("type_identifier", "scoped_type_identifier"):
            impl_type = _text(c).split("<")[0].strip()
        elif fn == "trait":
            trait_name = _text(c)

    if trait_name:
        impl_scope = f"{impl_type}::{trait_name}::impl"
    else:
        impl_scope = f"{impl_type}::impl"

    for child in body.children:
        if child.type == "function_item":
            name_node = child.child_by_field_name("name")
            if not name_node:
                continue
            name = _text(name_node)
            qn = f"{file_qn}::{impl_scope}::{name}"
            is_test = _has_test_attr(child)
            fn_props: dict[str, Any] = {"is_test": is_test} if is_test else {}
            fn_text = child.text.decode("utf-8", errors="replace") if child.text else ""
            fn_id = _add_node(graph, NodeKind.METHOD, qn, Path(""),
                              child.start_point.row, child.end_point.row,
                              fn_text, fn_props)
            if fn_id is None:
                continue
            graph.add_edge(impl_id, fn_id, Edge.extracted(EdgeKind.DEFINES))

            # Walk method body for calls
            fn_body = child.child_by_field_name("body")
            if fn_body:
                _emit_calls_in_body(graph, fn_id, fn_body)


def extract_file(path: Path, graph: Graph) -> None:
    with open(path, "rb") as f:
        raw = f.read()
    source = raw.decode("utf-8", errors="replace")

    lang_ptr = tree_sitter_rust.language()
    language = ts.Language(lang_ptr)
    parser = ts.Parser(language)
    tree = parser.parse(raw)
    root = tree.root_node

    file_qn = path.stem
    file_id = _add_node(graph, NodeKind.FILE, file_qn, path, 0, 0, source)

    # Pre-compute #[cfg(test)] mod ranges for test detection.
    test_mod_ranges = _find_cfg_test_mod_ranges(root, raw)

    # Functions
    fn_q = ts.Query(language, r"(function_item name: (identifier) @name body: (block) @body) @def")
    cursor = ts.QueryCursor(fn_q)
    for _, caps in cursor.matches(root):
        name_node = None
        def_node = None
        body_node = None
        for name, nodes in caps.items():
            for cap in nodes:
                if name == "name" and name_node is None:
                    name_node = cap
                elif name == "def" and def_node is None:
                    def_node = cap
                elif name == "body" and body_node is None:
                    body_node = cap
        if not name_node or not def_node:
            continue
        name = _text(name_node)
        scope = _rust_scope(def_node)
        qn = f"{file_qn}::{'::'.join(scope)}::{name}" if scope else f"{file_qn}::{name}"
        is_method = _has_method_parent(def_node)
        kind = NodeKind.METHOD if is_method else NodeKind.FUNCTION
        fn_byte_range = (def_node.start_byte, def_node.end_byte)
        is_test = (name.startswith("test_") or name.endswith("_test")
                   or _has_test_attr(def_node)
                   or _in_any_range(fn_byte_range, test_mod_ranges))
        props = {"is_test": True} if is_test else {}
        fn_id = _add_node(graph, kind, qn, path, def_node.start_point.row, def_node.end_point.row, source, props)
        graph.add_edge(file_id, fn_id, Edge.extracted(EdgeKind.DEFINES))
        if body_node:
            _emit_calls_in_body(graph, fn_id, body_node)

    # Traits
    trait_q = ts.Query(language, r"(trait_item name: (type_identifier) @name) @def")
    cursor = ts.QueryCursor(trait_q)
    for _, caps in cursor.matches(root):
        name_node = None
        def_node = None
        for name, nodes in caps.items():
            for cap in nodes:
                if name == "name" and name_node is None:
                    name_node = cap
                elif name == "def" and def_node is None:
                    def_node = cap
        if not name_node or not def_node:
            continue
        name = _text(name_node)
        qn = f"{file_qn}::{name}"
        tid = _add_node(graph, NodeKind.TRAIT, qn, path, def_node.start_point.row, def_node.end_point.row, source)
        graph.add_edge(file_id, tid, Edge.extracted(EdgeKind.DEFINES))

    # Structs
    struct_q = ts.Query(language, r"(struct_item name: (type_identifier) @name) @def")
    cursor = ts.QueryCursor(struct_q)
    for _, caps in cursor.matches(root):
        name_node = None
        def_node = None
        for name, nodes in caps.items():
            for cap in nodes:
                if name == "name" and name_node is None:
                    name_node = cap
                elif name == "def" and def_node is None:
                    def_node = cap
        if not name_node or not def_node:
            continue
        name = _text(name_node)
        qn = f"{file_qn}::{name}"
        sid = _add_node(graph, NodeKind.CLASS, qn, path, def_node.start_point.row, def_node.end_point.row, source)
        graph.add_edge(file_id, sid, Edge.extracted(EdgeKind.DEFINES))

    # Enums
    enum_q = ts.Query(language, r"(enum_item name: (type_identifier) @name) @def")
    cursor = ts.QueryCursor(enum_q)
    for _, caps in cursor.matches(root):
        name_node = None
        def_node = None
        for name, nodes in caps.items():
            for cap in nodes:
                if name == "name" and name_node is None:
                    name_node = cap
                elif name == "def" and def_node is None:
                    def_node = cap
        if not name_node or not def_node:
            continue
        name = _text(name_node)
        qn = f"{file_qn}::{name}"
        eid = _add_node(graph, NodeKind.TYPE, qn, path, def_node.start_point.row, def_node.end_point.row, source)
        graph.add_edge(file_id, eid, Edge.extracted(EdgeKind.DEFINES))

        # Walk enum body to extract variant names as Variable nodes
        body = def_node.child_by_field_name("body")
        if body:
            _walk_enum_body(graph, eid, qn, body)

    # Impl blocks
    impl_q = ts.Query(language, r"(impl_item) @def")
    cursor = ts.QueryCursor(impl_q)
    for _, caps in cursor.matches(root):
        def_node = None
        for name, nodes in caps.items():
            if name == "def" and def_node is None:
                def_node = nodes[0]
                break
        if not def_node:
            continue
        # Determine impl'd type and whether this is a trait impl.
        # For "impl Baz for Foo": type_identifier(fn=trait)=Baz, type_identifier(fn=type)=Foo
        # For "impl Foo": type_identifier(fn=type)=Foo
        impl_type = None
        is_trait_impl = False
        trait_name = None
        for i, c in enumerate(def_node.children):
            fn = def_node.field_name_for_child(i)
            if fn == "type" and c.type in ("type_identifier", "scoped_type_identifier"):
                impl_type = _text(c).split("<")[0].strip()
            elif fn == "trait":
                is_trait_impl = True
                trait_name = _text(c)

        impl_type = impl_type or _impl_type_name(def_node) or "anon"
        # Make impl QNs unique: include trait if trait impl
        if is_trait_impl and trait_name:
            qn = f"{file_qn}::{impl_type}::{trait_name}::impl"
        else:
            qn = f"{file_qn}::{impl_type}::impl"
        impl_props: dict[str, Any] = {"type": impl_type}
        if is_trait_impl and trait_name:
            impl_props["trait"] = trait_name

        impl_id = _add_node(graph, NodeKind.IMPL, qn, path,
                            def_node.start_point.row, def_node.end_point.row, source, impl_props)
        if impl_id is None:
            continue
        graph.add_edge(file_id, impl_id, Edge.extracted(EdgeKind.DEFINES))

        if is_trait_impl and trait_name:
            trait_qn = f"type::{trait_name}"
            trait_id = graph.find_by_qname(trait_qn)
            if trait_id is not None:
                graph.add_edge(impl_id, trait_id, Edge.extracted(EdgeKind.IMPLEMENTS))

        # Walk impl body for methods and calls
        body = def_node.child_by_field_name("body")
        if body:
            _walk_impl_body(graph, file_qn, impl_id, body, def_node)

    # Use declarations (imports)
    use_q = ts.Query(language, r"(use_declaration argument: (_) @path) @use")
    cursor = ts.QueryCursor(use_q)
    for _, caps in cursor.matches(root):
        path_node = None
        for name, nodes in caps.items():
            for cap in nodes:
                if name == "path":
                    path_node = cap
                    break
        if path_node:
            break
        if not path_node:
            continue
        path_text = _text(path_node).strip().rstrip(";").strip()
        if not path_text:
            continue
        mod_qn = f"module::{path_text}"
        mod_id = _add_node(graph, NodeKind.MODULE, mod_qn, Path(""), 0, 0)
        graph.add_edge(file_id, mod_id, Edge.extracted(EdgeKind.IMPORTS))

    # Mod declarations
    mod_q = ts.Query(language, r"(mod_item name: (identifier) @name) @def")
    cursor = ts.QueryCursor(mod_q)
    for _, caps in cursor.matches(root):
        name_node = None
        for name, nodes in caps.items():
            for cap in nodes:
                if name == "name":
                    name_node = cap
                    break
        if name_node:
            break
        if not name_node:
            continue
        name = _text(name_node)
        qn = f"{file_qn}::{name}"
        mid = _add_node(graph, NodeKind.MODULE, qn, path, 0, 0, None)
        graph.add_edge(file_id, mid, Edge.extracted(EdgeKind.DEFINES))
