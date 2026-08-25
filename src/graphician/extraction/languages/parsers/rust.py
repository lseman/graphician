"""Rust source extraction.

Emits: Function, Method, Class (struct), Trait, Type (enum), Module (mod/use), File
Edges: Defines, Imports, Calls
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tree_sitter as ts
import tree_sitter_rust

from graphician.core.edge import Edge, EdgeKind
from graphician.core.graph import Graph
from graphician.core.id import NodeId
from graphician.core.node import Node, NodeKind

_SUPPRESS_CALLS = frozenset(["std::panic", "std::result", "std::option", "std::vec",
    "std::vec::Vec", "std::option::Option", "std::result::Result",
    "std::boxed::Box", "std::sync", "std::rc", "std::cell",
    "std::mem", "std::convert", "std::ops", "std::fmt",
    # tree-sitter Node API — the AST extractors walk these methods
    # and emit call placeholders; they're not project functions.
    "child_by_field_name", "children", "end_position", "is_named", "kind",
    "language", "parent", "root_node", "start_position", "text", "walk",
    "utf8_text", "field_name_for_child", "end_byte", "start_byte",
])

# Common builtin / stdlib names that should be suppressed at parse time
# (same set as call_resolution._SUPPRESS_CALLS, but only the ones that
# Rust code commonly calls — e.g. Vec::push, HashMap::get, etc.).
# Names that are likely to collide with real project symbols ("get",
# "find", "execute", "select") are NOT listed here; they are handled
# by the 6-tier resolver after the full graph is built.
_RUST_BUILTIN_CALLS = frozenset([
    # Rust std / common fluent API
    "and_then", "as_bytes", "as_deref", "as_ref", "as_str", "chars",
    "clone", "cloned", "clamp", "collect", "contains", "copied", "count",
    "default", "ends_with", "entry", "err", "expect", "extend",
    "filter_map", "first", "flat_map", "fold", "from",
    "from_str", "get_mut", "index", "insert", "into", "into_iter",
    "is_empty", "is_none", "is_some_and", "iter_mut", "join", "last",
    "lines", "map_err", "none", "ok", "ok_or", "ok_or_else",
    "or_default", "position", "push_str", "rsplit", "some",
    "split", "splitn", "starts_with", "take",
    "to_string_lossy", "trim", "unwrap", "unwrap_or", "unwrap_or_default",
    "unwrap_or_else", "with_capacity", "sort_by", "sort_by_key",
    "sort_unstable", "truncate", "reserve", "clear", "contains_key",
    "values", "concat", "to_vec", "read", "read_to_string",
    "read_to_end", "remove_dir_all", "remove_file", "create_dir_all",
    "exists", "write", "write_all", "flush", "load", "display",
    "execute", "fg", "temp_dir", "args", "strip_prefix",
    "to_ascii_lowercase", "trim_matches", "is_some", "or_else",
    "values_mut", "borrow", "has", "pop_front", "push_back", "remove",
    "to_lowercase", "to_uppercase", "split_whitespace",
    "saturating_sub", "replace", "to_string_pretty",
    "as_array", "as_u64", "string", "path", "file_name",
    "file_stem", "wrapping_add", "current_dir", "as_bool", "as_f64",
    "as_object", "render_widget", "highlight_style", "attr", "block",
    "border_style", "borders", "checkAvailable", "percentage",
    "strip_suffix", "trim_end_matches", "chunks", "or_insert",
    "or_insert_with",
    # SQLite rusqlite bindings.
    "query_map", "prepare", "commit", "transaction", "select", "selected",
    "query_row", "add_modifier",
    # std::time methods.
    "duration_since", "now", "as_nanos", "saturating_add", "wrapping_mul",
    # std::path.
    "extension",
    # C/C++ and libc-style calls.
    "malloc", "free", "printf", "fprintf", "memcpy", "memset", "strlen",
    "strcmp", "std",
    # Graphician internal methods.
    "resolve_mentions", "original_nodes", "edges_mut", "qname_index",
])


def _should_suppress_rust_call(name: str) -> bool:
    """Return True if this Rust call name should be suppressed at parse time.

    Only suppress names that can *never* plausibly be a project-defined
    symbol — language/runtime/tree-sitter API surface.  Names that are
    likely to collide with real project symbols must NOT be suppressed
    here; the 6-tier resolver handles them after the full graph is built.
    """
    if not name:
        return True
    lower = name.lower().strip()
    if lower in _RUST_BUILTIN_CALLS:
        return True
    # Also check the tree-sitter / std names
    return lower in _SUPPRESS_CALLS


def _text(node: ts.Node) -> str:
    raw = node.text
    if raw is None:
        return ""
    return raw.decode("utf-8", errors="replace")


def _child_by_field(node: ts.Node, field: str) -> ts.Node | None:
    return node.child_by_field_name(field)


def _extract_source_text(lines: list[str], line_start: int, line_end: int) -> str:
    """Extract source text by line range, matching ariadne-rust's extract_source_text.

    The Rust code assumes tree-sitter rows are 1-indexed and subtracts 1.
    In practice tree-sitter uses 0-indexed rows, so this is a known mismatch.
    We match the Rust behavior for parity.
    """
    if line_start == 0 or line_end == 0 or line_end < line_start:
        return ""
    # Match Rust: subtract 1 from each row (assumes 1-indexed)
    s = max(0, line_start - 1) if line_start > 0 else 0
    e = min(len(lines), line_end - 1) if line_end > 0 else 0
    if s >= e or s >= len(lines):
        return ""
    return "\n".join(lines[s:e])


def _add_node(
    graph: Graph,
    kind: NodeKind,
    qn: str,
    path: Path,
    line_start: int,
    line_end: int,
    source_lines: list[str] | str | None = None,
    props: dict[str, Any] | None = None,
) -> NodeId:
    existing = graph.find_by_qname(qn)
    if existing is not None:
        return existing
    node = Node.new(kind, qn)
    node = node.with_source(str(path), line_start + 1, line_end + 1)
    # Extract source text by line range (matching ariadne-rust's
    # extract_source_text behaviour) rather than passing the raw bytes.
    # This avoids data_flow from seeing the entire file for every node.
    source_text = _extract_source_text(source_lines, line_start, line_end)
    if source_text:
        node = node.with_source_text(source_text)
    # Extract source text: if source_lines is a list, use line-range extraction;
    # if it's a string, use it directly (for FILE nodes).
    if isinstance(source_lines, list):
        source_text = _extract_source_text(source_lines, line_start, line_end)
    elif source_lines is not None:
        source_text = source_lines
    else:
        source_text = ""
    if source_text:
        node = node.with_source_text(source_text)
    # Default dialect for Rust
    default_props: dict[str, Any] = {"dialect": "rust"}
    if props:
        default_props.update(props)
    if default_props:
        for k, v in default_props.items():
            node = node.with_property(k, v)
    graph.add_node(node)
    result = graph.find_by_qname(qn)
    assert result is not None, f"Node not found after add: {qn}"
    return result
    # Default dialect for Rust
    default_props = {"dialect": "rust"}
    if props:
        default_props.update(props)
    if default_props:
        for k, v in default_props.items():
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
    if _should_suppress_rust_call(name):
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
    parent.walk()
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
    return bool(head.endswith("::test") or head == "test_case")


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
    parent.walk()
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
        elif cur.type == "mod_item" or (cur.type == "function_item" and cur.id != original_id):
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
        # tree-sitter-rust 0.24 exposes the receiver positionally rather
        # than through an ``object`` field.
        obj = func_node.child(0)
        if obj and obj.type == "identifier":
            return _text(obj)
    if func_node.type in ("scoped_identifier", "scoped_type_identifier"):
        text = _text(func_node)
        return text.split("::")[0] if "::" in text else None
    return None


def _expand_use_paths(node: ts.Node, prefix: str = "") -> list[str]:
    """Expand a Rust use-tree AST into concrete import paths.

    ``use std::{fmt, collections::{HashMap, HashSet}}`` becomes three
    independently resolvable module targets. Aliases retain the source path;
    the local alias is not a module identity.
    """
    if node.type == "scoped_use_list":
        path = node.child_by_field_name("path")
        use_list = node.child_by_field_name("list")
        base = _join_use_path(prefix, _text(path)) if path else prefix
        return _expand_use_paths(use_list, base) if use_list else [base]

    if node.type == "use_list":
        paths: list[str] = []
        for child in node.named_children:
            paths.extend(_expand_use_paths(child, prefix))
        return paths

    if node.type == "use_as_clause":
        path = node.child_by_field_name("path")
        return _expand_use_paths(path, prefix) if path else []

    text = _text(node).strip().rstrip(";")
    if not text:
        return []
    if text == "self":
        return [prefix] if prefix else [text]
    return [_join_use_path(prefix, text)]


def _join_use_path(prefix: str, path: str) -> str:
    path = path.strip()
    if not prefix:
        return path
    if not path or path == "self":
        return prefix
    return f"{prefix}::{path}"


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


def extract_file(
    path: Path, graph: Graph, *, file_qn: str | None = None, source_path: Path | None = None
) -> None:
    with open(path, "rb") as f:
        raw = f.read()
    source = raw.decode("utf-8", errors="replace")
    source_lines = source.splitlines()

    lang_ptr = tree_sitter_rust.language()
    language = ts.Language(lang_ptr)
    parser = ts.Parser(language)
    tree = parser.parse(raw)
    root = tree.root_node

    file_qn = file_qn or path.stem
    record_path = source_path if source_path is not None else path
    path = record_path
    # File node stores the full source text (no line-range extraction).
    # File node stores the full source text directly (no line-range extraction).
    file_id = _add_node(graph, NodeKind.FILE, file_qn, path, 0, 0, source_lines)

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
        fn_id = _add_node(graph, kind, qn, path, def_node.start_point.row, def_node.end_point.row,
                          source_lines, props)
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
        tid = _add_node(graph, NodeKind.TRAIT, qn, path, def_node.start_point.row, def_node.end_point.row,
                        source_lines)
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
        sid = _add_node(graph, NodeKind.CLASS, qn, path, def_node.start_point.row, def_node.end_point.row,
                        source_lines)
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
        eid = _add_node(graph, NodeKind.TYPE, qn, path, def_node.start_point.row, def_node.end_point.row,
                        source_lines)
        graph.add_edge(file_id, eid, Edge.extracted(EdgeKind.DEFINES))

        # Walk enum body to extract variant names as Variable nodes
        body = def_node.child_by_field_name("body")
        if body:
            _walk_enum_body(graph, eid, qn, body)

    # Impl blocks — skipped. Methods are already extracted by the
    # function_item query with their correct scope.  Creating a
    # separate IMPL node would duplicate relationships that the
    # qualified-name scope already captures.
    pass

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
        if not path_node:
            continue
        for path_text in _expand_use_paths(path_node):
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
        if not name_node:
            continue
        name = _text(name_node)
        qn = f"{file_qn}::{name}"
        mid = _add_node(graph, NodeKind.MODULE, qn, path, 0, 0, None)
        graph.add_edge(file_id, mid, Edge.extracted(EdgeKind.DEFINES))
