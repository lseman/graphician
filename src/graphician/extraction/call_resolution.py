"""Call-placeholder resolution: a 6-tier name-resolution heuristic engine.

Turns ``call::name`` placeholder edges (emitted by AST extractors when a call
target can't be resolved locally) into real ``Calls`` edges pointing at a
specific function/method definition.

Resolution tiers (best first):

    1. **Unique name** — exactly one candidate in the whole graph.
    2. **File-local** — exactly one candidate in the caller's own file.
    3a. **Scoped** — a path-qualified call carried ``call_scope`` on the
        placeholder edge; pick the unique candidate whose qualified name
        contains that scope.
    3b. **Scoped prefix** — multiple candidates match the scope; pick the
        one whose qualified name shares the longest common prefix with the
        caller's qualified name (same module subtree wins).
    4. **Receiver-based** — for method calls like ``self.foo()`` or
       ``graph.add_node()``, the receiver name hints at an impl type.
    5. **Import-scoped** — exactly one candidate lives in a module the
       caller's file imports (matched by file stem against import-path
       tokens).
    6. **Same-directory** — prefer the one whose source file lives in the
       same directory as the caller.
    7. **Frequency prior** — last resort: prefer the candidate that already
       has the most resolved ``Calls`` in-edges.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .._extract import plan_call_resolution
from ..core.edge import Confidence, Edge, EdgeKind
from ..core.graph import Graph
from ..core.id import EdgeId, NodeId
from ..core.node import NodeKind

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Suppression
# ---------------------------------------------------------------------------

_SUPPRESS_CALLS = frozenset([
    # Python builtins and common constructors.
    "abs", "all", "any", "bool", "bytes", "callable", "dict", "dir",
    "enumerate", "float", "getattr", "hasattr", "hash", "id",
    "int", "isinstance", "iter", "list", "max", "min",
    "next", "open", "print", "range", "repr", "reversed", "round",
    "set", "sorted", "str", "sum", "super", "tuple", "type", "vars",
    "zip",
    # Dunder methods that are too noisy to resolve (appear on almost every object).
    # These lack meaningful stub resolution anyway.
    "__iter__", "__len__", "__getitem__",
    "__setitem__", "__delitem__", "__contains__", "__eq__", "__ne__",
    "__lt__", "__le__", "__gt__", "__ge__", "__hash__", "__bool__",
    "__repr__", "__str__", "__format__", "__enter__", "__exit__",
    "__call__", "__new__", "__init__", "__del__", "__getattr__",
    "__setattr__", "__delattr__", "__dir__", "__reduce__", "__reduce_ex__",
    "__getattribute__", "__sizeof__", "__class__", "__doc__", "__module__",
    "__annotations__", "__dict__", "__weakref__", "__slots__",
    # pop, remove, append, get, add, insert, find, contains, sort,
    # new, clone, collect, extend, clear, replace, count, size, has,
    # push, values, keys, join removed - they have stub coverage
    # Regex / re module methods.
    "match", "search", "fullmatch", "findall", "finditer", "sub",
    "subn", "compile", "escape", "quote", "pattern",
    # split removed - has stub coverage in str/list
    # Math / numeric methods.
    "log2", "log10", "log", "sqrt", "ceil", "floor", "trunc",
    "abs", "round", "pow", "modf", "frexp", "ldexp",
    "isnan", "isinf", "isfinite",
    # Common class method / constructor parameters that leak as unresolved.
    "cls", "self", "kwargs", "key", "default", "factory", "ValueError", "TypeError", "AttributeError", "KeyError", "IndexError", "RuntimeError", "Exception",
    # args removed - duplicate (also in Rust/std section)
    # Common string/collection methods not yet suppressed.
    "splitlines", "translate", "ljust", "rjust", "zfill", "expandtabs",
    # replace removed - has stub coverage in str
    "casefold", "maketrans",
    # Rust/std/common fluent API calls that otherwise dominate
    # unresolved call nodes. Removed methods with stub coverage.
    "as_bytes", "as_deref", "as_ref", "as_str", "chars", "clamp",
    "default", "entry", "err", "expect", "fold", "from",
    "index", "last", "none", "ok", "read", "read_to_string",
    "read_to_end", "remove_dir_all", "remove_file", "create_dir_all",
    "exists", "write", "write_all", "flush", "load", "display", "execute",
    "fg", "temp_dir", "args", "strip_prefix", "to_ascii_lowercase",
    "values_mut", "borrow", "pop_front", "push_back",
    "split_whitespace", "saturating_sub",
    "as_array", "as_u64", "string", "file_name",
    "file_stem", "wrapping_add", "current_dir", "as_bool", "as_f64",
    "as_object", "render_widget", "highlight_style", "attr", "block",
    "border_style", "borders", "checkAvailable", "percentage",
    "strip_suffix", "trim_end_matches", "or_else", "to_vec",
    "to_string_lossy", "ends_with", "to_owned", "from_str",
    "into", "into_iter", "is_some_and", "iter_mut",
    "map_err", "or_default", "rsplit", "some", "trim", "unwrap", "unwrap_or",
    "unwrap_or_default", "unwrap_or_else", "with_capacity",
    "sort_by", "sort_by_key", "sort_unstable", "truncate", "reserve",
    "contains_key", "is_none", "is_some", "or_insert", "or_insert_with"
    # SQLite rusqlite bindings.
    "query_map", "prepare", "commit", "transaction", "select", "selected",
    "query_row", "add_modifier",
    # std::time methods.
    "duration_since", "now", "as_nanos", "saturating_add", "wrapping_mul",
    # std::path.
    "extension",
    # Confidence enum variant leaking as unresolved.
    "inferred",
    # Common graph-library traversal/mutation helpers. Keeping
    # these out of the code graph prevents external petgraph calls
    # from masquerading as unresolved project calls.
    "contains_node", "edge_indices", "edge_references", "edge_weight_mut",
    "edges_directed", "node_indices", "node_weight", "node_weight_mut",
    # std::fs::DirEntry / std::path methods.
    "status", "watch", "to_path_buf", "is_dir", "is_file", "filter_entry",
    # C/C++ and libc-style calls.
    "malloc", "free", "printf", "fprintf", "memcpy", "memset", "strlen",
    "strcmp", "std",
    # tree-sitter Node API — the AST extractors walk these methods
    # and emit call placeholders; they're not project functions.
    "child_by_field_name", "children", "end_position", "is_named", "kind",
    "language", "language_tsx", "parent", "root_node", "node",
    "start_position", "text", "walk", "utf8_text",
    # tree-sitter Parser API.
    "parse", "set_language", "included_ranges",
    # Additional tree-sitter / std methods that leak as unresolved.
    "rev", "nth", "to_str", "last_mut", "trim_start", "from_utf8",
    "windows", "end_byte", "start_byte", "is_ascii_digit",
    "is_lowercase", "is_uppercase", "is_alphanumeric", "new_ext",
    "reverse", "next_back", "as_array_mut",
    # Graphician internal methods — not extractable as project functions.
    "resolve_mentions", "original_nodes", "edges_mut", "qname_index",
    # Project-internal helpers that appear as unresolved from test/utility files.
    "timer", "_extract_query_identifiers", "_is_test_file_path",
])



def should_suppress_call_placeholder(name: str) -> bool:
    """Return True if this call name should be suppressed during resolution."""
    if not name:
        return True
    return name.lower().strip() in _SUPPRESS_CALLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def common_prefix_len(a: str, b: str) -> int:
    """Count of ``::`` segments shared between two qualified names."""
    return sum(1 for x, y in zip(a.split("::"), b.split("::"), strict=False) if x == y)


def module_stem(uri: str) -> str | None:
    """Module name a source file answers to in import paths."""
    path = Path(uri)
    stem = path.stem
    if stem is None:
        return None
    stem_str = str(stem)
    if stem_str in ("mod", "index", "__init__", "lib", "main"):
        parent = path.parent
        return parent.name if parent else None
    return stem_str


def _build_by_name(graph: Graph) -> dict[str, list[NodeId]]:
    """Map name → every non-placeholder function/method/class/type."""
    by_name: dict[str, list[NodeId]] = {}
    for nid, node in graph.nodes():
        if (
            node.kind in (NodeKind.FUNCTION, NodeKind.METHOD, NodeKind.CLASS, NodeKind.TYPE)
            and not node.qualified_name.startswith("call::")
        ):
            by_name.setdefault(node.name, []).append(nid)
    return by_name


def _build_caller_impl_context(graph: Graph) -> dict[NodeId, str]:
    """Map caller NodeId → impl type string for method resolution."""
    context: dict[NodeId, str] = {}
    for nid, node in graph.nodes():
        if node.kind != NodeKind.METHOD:
            continue
        qn = node.qualified_name
        parts = qn.split("::")
        if len(parts) < 2:
            continue
        # Extracted method names are always the final qname component and
        # their owning class/impl is immediately before it.  Walking backward
        # used to select the method name itself, which made ``self.foo()``
        # impossible to disambiguate when several classes defined ``foo``.
        if len(parts) >= 2 and parts[-2]:
            context[nid] = parts[-2]
    return context


# Regex patterns for let-binding type inference (Rust-style)
_RE_LET_ANNOT = re.compile(r"let\s+(?:mut\s+)?(\w+)\s*:\s*([A-Z]\w+)")
_RE_LET_CTOR = re.compile(r"let\s+(?:mut\s+)?(\w+)\s*=\s*([A-Z]\w+)::")
_RE_LET_STRUCT = re.compile(r"let\s+(?:mut\s+)?(\w+)\s*=\s*([A-Z]\w+)\s*\{")
# Python-style: var = ClassName(...) or var = ClassName(...)
_RE_PY_VAR_ANNOT = re.compile(r"(\w+)\s*:\s*([A-Z]\w+)\s*=")
_RE_PY_VAR_ASSIGN = re.compile(
    r"(\w+)\s*=\s*([A-Z]\w+)(?:\.\w+)*\s*[\(\{]"
)
_RE_IMPL_HEADER = re.compile(r"\bimpl(?:\s*<[^{}>]*>)?\s+([^{}]+?)\s*\{")

_NAME_TYPE_MAP: dict[str, str] = {
    "graph": "Graph", "main_graph": "Graph", "app_graph": "Graph",
    "g": "Graph",
    "motif": "MotifBuilder", "mb": "MotifBuilder", "motif_builder": "MotifBuilder",
    "store": "Store", "db": "Store", "database": "Store",
    "parser": "Parser", "ts_parser": "Parser",
    "config": "Config", "cfg": "Config",
    "ctx": "Context", "context": "Context",
    "options": "Options", "opts": "Options",
    "logger": "Logger", "log": "Logger",
    "server": "Server", "app": "App", "client": "Client",
    "writer": "Writer", "reader": "Reader",
    "builder": "Builder", "factory": "Factory",
    "handler": "Handler", "middleware": "Middleware",
    "router": "Router", "controller": "Controller",
    "repository": "Repository", "service": "Service",
    "manager": "Manager", "driver": "Driver",
    "executor": "Executor", "scheduler": "Scheduler",
    "provider": "Provider", "observer": "Observer",
    "listener": "Listener", "subscriber": "Subscriber",
    "cache": "Cache", "pool": "Pool",
    "connection": "Connection", "session": "Session",
    "request": "Request", "response": "Response",
    "event": "Event", "task": "Task", "job": "Job",
    "thread": "Thread", "process": "Process",
}


def _infer_type_from_let_bindings(source: str, var_name: str) -> str | None:
    """Scan source for let/type bindings that assign a type to var_name."""
    for match in _RE_LET_ANNOT.finditer(source):
        if match.group(1) == var_name:
            return match.group(2)
    for match in _RE_LET_CTOR.finditer(source):
        if match.group(1) == var_name:
            return match.group(2)
    for match in _RE_LET_STRUCT.finditer(source):
        if match.group(1) == var_name:
            return match.group(2)
    for match in _RE_PY_VAR_ANNOT.finditer(source):
        if match.group(1) == var_name:
            return match.group(2)
    for match in _RE_PY_VAR_ASSIGN.finditer(source):
        if match.group(1) == var_name:
            return match.group(2)
    return None


def _leading_type_name(type_expr: str) -> str | None:
    """Return the main named type from a Rust-style type expression."""
    rest = type_expr.lstrip()
    while True:
        rest = rest.lstrip("&").lstrip()
        before = rest
        for qualifier in ("mut", "dyn", "impl"):
            tail = rest.removeprefix(qualifier)
            if tail != rest and tail[:1].isspace():
                rest = tail.lstrip()
                break
        if rest == before:
            break

    path_match = re.match(r"[A-Za-z_][\w]*(?:::[A-Za-z_][\w]*)*", rest)
    if path_match is None:
        return None
    type_name = path_match.group(0).rsplit("::", 1)[-1]
    return type_name if type_name[:1].isupper() else None


def _infer_type_from_annotations(source: str, var_name: str) -> str | None:
    """Infer a receiver type from a parameter or closure annotation."""
    for match in re.finditer(rf"(?<!\w){re.escape(var_name)}(?!\w)\s*:\s*", source):
        inferred = _leading_type_name(source[match.end():])
        if inferred is not None:
            return inferred
    return None


def _infer_impl_type_from_source(source: str, line_start: int | None) -> str | None:
    """Infer the surrounding Rust ``impl`` type for a caller line."""
    if line_start is None:
        return None

    for match in _RE_IMPL_HEADER.finditer(source):
        header = match.group(1).strip()
        target_expr = header.rsplit(" for ", 1)[-1]
        impl_type = _leading_type_name(target_expr)
        if impl_type is None:
            continue

        depth = 1
        position = match.end()
        while position < len(source) and depth:
            if source[position] == "{":
                depth += 1
            elif source[position] == "}":
                depth -= 1
            position += 1
        if depth:
            continue

        start_line = source.count("\n", 0, match.start())
        end_line = source.count("\n", 0, position)
        # Extractors differ on zero- versus one-based line locations, so
        # accept either representation at this internal compatibility layer.
        if start_line <= line_start <= end_line or start_line <= line_start - 1 <= end_line:
            return impl_type
    return None


def _infer_type_from_var_name(name: str) -> str | None:
    """Map common variable names to their likely types."""
    return _NAME_TYPE_MAP.get(name.lower().strip())


def _infer_type_from_receiver_expression(expression: str) -> str | None:
    """Recognize fluent receivers such as ``Node.new(...).with_source(...)``."""
    match = re.match(r"\s*([A-Z]\w+)(?:\.\w+)*\s*\(", expression)
    return match.group(1) if match is not None else None


def _build_import_tokens(graph: Graph) -> dict[str, set[str]]:
    """Map caller file URI → set of import-path tokens (lowercased)."""
    tokens: dict[str, set[str]] = {}
    for _, src, dst, edge in graph.edges():
        if edge.kind != EdgeKind.IMPORTS:
            continue
        src_node = graph.node(src)
        dst_node = graph.node(dst)
        if src_node is None or dst_node is None:
            continue
        uri = src_node.source_uri
        if uri is None:
            continue
        qn = dst_node.qualified_name
        if not qn.startswith("module::"):
            continue
        path = qn[len("module::"):]
        token_set = tokens.setdefault(uri, set())
        for tok in _split_tokens(path):
            if tok:
                token_set.add(tok.lower())
    return tokens


def _build_imported_symbol_modules(
    graph: Graph,
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """Map a local import name to ``(module, original name)`` bindings."""
    imported: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for _, source, _target, edge in graph.edges():
        if edge.kind != EdgeKind.IMPORTS:
            continue
        source_node = graph.node(source)
        if source_node is None or source_node.source_uri is None:
            continue
        module_path = edge.properties.get("module_path")
        symbols = edge.properties.get("imported_symbols")
        if not isinstance(module_path, str) or not isinstance(symbols, dict):
            continue
        for local_name, original_name in symbols.items():
            if isinstance(local_name, str) and isinstance(original_name, str):
                imported.setdefault((source_node.source_uri, local_name), []).append(
                    (module_path, original_name)
                )
    return imported


def _module_matches_source(module_path: str, caller_uri: str, candidate_uri: str) -> bool:
    """Match an absolute or relative Python import to a candidate source path."""
    candidate = Path(candidate_uri).with_suffix("")
    if candidate.name == "__init__":
        candidate = candidate.parent

    dots = len(module_path) - len(module_path.lstrip("."))
    module_parts = tuple(part for part in module_path.lstrip(".").split(".") if part)
    if dots:
        expected = Path(caller_uri).parent
        for _ in range(max(0, dots - 1)):
            expected = expected.parent
        expected = expected.joinpath(*module_parts)
        expected_parts = expected.parts
        return candidate.parts[: len(expected_parts)] == expected_parts

    if not module_parts:
        return False
    candidate_parts = candidate.parts
    width = len(module_parts)
    return any(
        candidate_parts[index : index + width] == module_parts
        for index in range(len(candidate_parts) - width + 1)
    )


def _split_tokens(path: str) -> list[str]:
    """Split a module path on non-alphanumeric separators."""
    result: list[str] = []
    current = ""
    for c in path:
        if c.isalnum() or c == "_":
            current += c
        else:
            if current:
                result.append(current)
                current = ""
    if current:
        result.append(current)
    return result


# ---------------------------------------------------------------------------
# Tier-based resolution
# ---------------------------------------------------------------------------


def _receiver_type_for_edge(
    graph: Graph,
    src_id: NodeId,
    edge: Edge,
    caller_impl_ctx: dict[NodeId, str],
) -> str | None:
    """Normalize language-specific receiver evidence for native resolution."""
    receiver_name = edge.properties.get("call_receiver")
    if not isinstance(receiver_name, str):
        return None
    src_node = graph.node(src_id)
    source = (src_node.source_text if src_node else "") or ""
    if receiver_name == "self" or receiver_name.startswith("self."):
        return caller_impl_ctx.get(src_id) or _infer_impl_type_from_source(
            source,
            src_node.line_start if src_node else None,
        )
    return (
        _infer_type_from_receiver_expression(receiver_name)
        or _infer_type_from_annotations(source, receiver_name)
        or _infer_type_from_let_bindings(source, receiver_name)
        or _infer_type_from_var_name(receiver_name)
    )


def _resolve_call_placeholders_native(graph: Graph) -> int:
    """Run native graph-wide selection and apply its mutation plan."""
    if plan_call_resolution is None:
        raise RuntimeError("native call resolution is unavailable")

    caller_impl_ctx = _build_caller_impl_context(graph)
    import_tokens = _build_import_tokens(graph)
    imported_symbol_modules = _build_imported_symbol_modules(graph)
    node_records = [
        (
            node_id.value,
            node.kind.value,
            node.name,
            node.qualified_name,
            node.source_uri,
        )
        for node_id, node in graph.nodes()
    ]
    edge_records = []
    for edge_id, source, target, edge in graph.edges():
        target_node = graph.node(target)
        bare = (
            target_node.qualified_name[6:]
            if target_node is not None
            and target_node.qualified_name.startswith("call::")
            else ""
        )
        scope = edge.properties.get("call_scope")
        edge_records.append(
            (
                edge_id.value,
                source.value,
                target.value,
                edge.kind.value,
                edge.confidence.value,
                scope if isinstance(scope, str) else None,
                _receiver_type_for_edge(graph, source, edge, caller_impl_ctx)
                if bare and edge.kind == EdgeKind.CALLS
                else None,
                should_suppress_call_placeholder(bare) if bare else False,
            )
        )
    binding_records = [
        (uri, local, module, original)
        for (uri, local), bindings in imported_symbol_modules.items()
        for module, original in bindings
    ]
    resolutions, stale_edge_ids = plan_call_resolution(
        node_records,
        edge_records,
        [(uri, sorted(tokens)) for uri, tokens in import_tokens.items()],
        binding_records,
    )

    for _edge_id, source, target, tag, structural in resolutions:
        resolved = (
            Edge.extracted(EdgeKind.CALLS)
            if structural
            else Edge.inferred(EdgeKind.CALLS, 0.7)
        )
        resolved.properties["resolved_from"] = f"call_placeholder::{tag}"
        graph.add_edge(NodeId(source), NodeId(target), resolved)
    if stale_edge_ids:
        graph.remove_edges_by_id([EdgeId(edge_id) for edge_id in stale_edge_ids])
        orphaned = [
            node_id
            for node_id, node in graph.nodes()
            if node.qualified_name.startswith("call::")
            and not any(graph.out_neighbors(node_id))
            and not any(graph.in_neighbors(node_id))
        ]
        for node_id in orphaned:
            graph.remove_node(node_id)
    return len(resolutions)


def resolve_call_placeholders(graph: Graph) -> int:
    """Resolve call placeholders natively, retaining the Python fallback."""
    if plan_call_resolution is not None:
        try:
            return _resolve_call_placeholders_native(graph)
        except Exception:
            logger.warning("Native call resolution failed; using Python fallback", exc_info=True)
    return _resolve_call_placeholders_python(graph)


def _resolve_call_placeholders_python(graph: Graph) -> int:
    """Resolve ``call::name`` placeholder edges in the graph.

    Returns the number of new resolved ``Calls`` edges added.
    """
    by_name = _build_by_name(graph)
    import_tokens = _build_import_tokens(graph)
    imported_symbol_modules = _build_imported_symbol_modules(graph)
    caller_impl_ctx = _build_caller_impl_context(graph)

    # Track existing call edges to avoid duplicates.
    existing_calls: set[tuple[int, int]] = set()
    for _, src, dst, edge in graph.edges():
        if edge.kind == EdgeKind.CALLS and edge.confidence != Confidence.AMBIGUOUS:
            existing_calls.add((src.value, dst.value))

    additions: list[tuple[int, int, str, bool]] = []  # (src, dst, tag, structural)
    stale_edges: list[EdgeId] = []  # edge IDs to remove

    for edge_id, src_id, dst_id, edge in graph.edges():
        if edge.kind != EdgeKind.CALLS:
            continue
        dst_node = graph.node(dst_id)
        if dst_node is None:
            continue
        name = dst_node.qualified_name
        if not name.startswith("call::"):
            continue
        bare = name[6:]
        src_node = graph.node(src_id)
        src_file = src_node.source_uri if src_node is not None else None
        imported_bindings = (
            imported_symbol_modules.get((src_file, bare), [])
            if src_file is not None
            else []
        )
        candidate_names = {bare, *(original for _, original in imported_bindings)}
        candidates = [
            candidate
            for candidate_name in candidate_names
            for candidate in by_name.get(candidate_name, [])
        ]
        candidates = list(dict.fromkeys(candidates))
        suppressed_name = should_suppress_call_placeholder(bare)
        if not candidates:
            if suppressed_name:
                # Suppression applies only when the repository has no matching
                # definition. Project APIs are allowed to use common names such
                # as get, load, parse, add, and len.
                stale_edges.append(edge_id)
            # No project candidate at all — leave non-noise placeholders unresolved.
            continue

        # ── Tier 1: unique name ────────────────────────────────────
        if len(candidates) == 1 and not suppressed_name:
            stale_edges.append(edge_id)
            cand = candidates[0]
            if (src_id.value, cand.value) not in existing_calls:
                additions.append((src_id.value, cand.value, "unique_name", True))
            continue

        # ── Tier 2: file-local ─────────────────────────────────────
        if src_file is not None:
            local: list[NodeId] = []
            for cand in candidates:
                cand_node = graph.node(cand)
                if cand_node is not None and cand_node.source_uri == src_file:
                    local.append(cand)
            if len(local) == 1:
                stale_edges.append(edge_id)
                cand = local[0]
                if (src_id.value, cand.value) not in existing_calls:
                    additions.append((src_id.value, cand.value, "file_local", True))
                continue

        # ── Tier 3a/3b: scoped ─────────────────────────────────────
        scope = edge.properties.get("call_scope")
        if scope is not None:
            scoped: list[NodeId] = []
            for cand in candidates:
                cand_node = graph.node(cand)
                if cand_node is not None and scope in cand_node.qualified_name:
                    scoped.append(cand)
            if len(scoped) == 1:
                stale_edges.append(edge_id)
                cand = scoped[0]
                if (src_id.value, cand.value) not in existing_calls:
                    additions.append((src_id.value, cand.value, "scoped", False))
                continue
            if len(scoped) > 1:
                caller_qn = src_node.qualified_name if src_node is not None else ""
                def _cand_qn(c: NodeId) -> str:
                    n = graph.node(c)
                    return n.qualified_name if n is not None else ""
                best = max(scoped, key=lambda c: common_prefix_len(
                    caller_qn, _cand_qn(c)
                ))
                stale_edges.append(edge_id)
                if (src_id.value, best.value) not in existing_calls:
                    additions.append((src_id.value, best.value, "scoped_prefix", False))
                continue

        # ── Tier 4: receiver-based ─────────────────────────────────
        receiver_name = edge.properties.get("call_receiver")
        if receiver_name is not None:
            impl_type: str | None = None
            if receiver_name == "self" or receiver_name.startswith("self."):
                source = (src_node.source_text if src_node else "") or ""
                impl_type = caller_impl_ctx.get(src_id) or _infer_impl_type_from_source(
                    source,
                    src_node.line_start if src_node else None,
                )
            else:
                source = (src_node.source_text if src_node else "") or ""
                impl_type = (
                    _infer_type_from_receiver_expression(receiver_name)
                    or _infer_type_from_annotations(source, receiver_name)
                    or _infer_type_from_let_bindings(source, receiver_name)
                    or _infer_type_from_var_name(receiver_name)
                )

            if impl_type is not None:
                # Narrow candidates whose qualified name has impl_type
                # right before the method name.
                receiver_candidates: list[NodeId] = []
                for cand in candidates:
                    cand_node = graph.node(cand)
                    if cand_node is None:
                        continue
                    qn = cand_node.qualified_name
                    parts = qn.split("::")
                    if len(parts) >= 2 and parts[-2] == impl_type:
                        receiver_candidates.append(cand)
                if len(receiver_candidates) == 1:
                    stale_edges.append(edge_id)
                    cand = receiver_candidates[0]
                    if (src_id.value, cand.value) not in existing_calls:
                        additions.append((src_id.value, cand.value, "receiver", False))
                    continue

        # ── Tier 5: import-scoped ──────────────────────────────────
        if src_file is not None:
            imported_by_name = [
                candidate
                for candidate in candidates
                if (candidate_node := graph.node(candidate)) is not None
                and candidate_node.source_uri is not None
                and any(
                    candidate_node.name == original_name
                    and _module_matches_source(
                        module, src_file, candidate_node.source_uri
                    )
                    for module, original_name in imported_bindings
                )
            ]
            if len(imported_by_name) == 1:
                stale_edges.append(edge_id)
                candidate = imported_by_name[0]
                if (src_id.value, candidate.value) not in existing_calls:
                    additions.append(
                        (src_id.value, candidate.value, "imported_symbol", False)
                    )
                continue

        if src_file is not None and src_file in import_tokens:
            tokens = import_tokens[src_file]
            imported: list[NodeId] = []
            for cand in candidates:
                cand_node = graph.node(cand)
                if cand_node is None or cand_node.source_uri is None:
                    continue
                stem = module_stem(cand_node.source_uri)
                if stem is not None and stem.lower() in tokens:
                    imported.append(cand)
            if len(imported) == 1:
                stale_edges.append(edge_id)
                cand = imported[0]
                if (src_id.value, cand.value) not in existing_calls:
                    additions.append((src_id.value, cand.value, "import_scoped", False))
                continue

        # ── Tier 6: same-directory affinity ────────────────────────
        if src_file is not None:
            src_dir = Path(src_file).parent
            if src_dir is not None:
                same_dir: list[NodeId] = []
                for cand in candidates:
                    cand_node = graph.node(cand)
                    if cand_node is None or cand_node.source_uri is None:
                        continue
                    cand_dir = Path(cand_node.source_uri).parent
                    if cand_dir is not None and cand_dir == src_dir:
                        same_dir.append(cand)
                if len(same_dir) == 1:
                    stale_edges.append(edge_id)
                    cand = same_dir[0]
                    if (src_id.value, cand.value) not in existing_calls:
                        additions.append((src_id.value, cand.value, "same_dir", False))
                    continue

        # ── Tier 7: frequency prior ────────────────────────────────
        if suppressed_name:
            stale_edges.append(edge_id)
            continue

        scored: list[tuple[NodeId, int]] = []
        for cand in candidates:
            in_calls = sum(
                1 for _, edge in graph.in_neighbors(cand)
                if edge.kind == EdgeKind.CALLS
            )
            scored.append((cand, in_calls))

        max_score = max((s for _, s in scored), default=0)
        if max_score > 0:
            winners = [c for c, s in scored if s == max_score]
            if len(winners) == 1:
                stale_edges.append(edge_id)
                if (src_id.value, winners[0].value) not in existing_calls:
                    additions.append((src_id.value, winners[0].value, "freq_prior", False))

    # Apply additions
    count = len(additions)
    for src_val, dst_val, tag, structural in additions:
        edge = Edge.extracted(EdgeKind.CALLS) if structural else Edge.inferred(EdgeKind.CALLS, 0.7)
        edge.properties["resolved_from"] = f"call_placeholder::{tag}"
        graph.add_edge(NodeId(src_val), NodeId(dst_val), edge)

    # Remove stale placeholder edges and orphaned call:: nodes.
    if stale_edges:
        graph.remove_edges_by_id(stale_edges)
        # Remove orphaned call:: nodes.
        to_remove: list[NodeId] = []
        for nid, node in graph.nodes():
            if node.qualified_name.startswith("call::"):
                has_edges = any(
                    True for _ in graph.out_neighbors(nid)
                ) or any(True for _ in graph.in_neighbors(nid))
                if not has_edges:
                    to_remove.append(nid)
        for nid in to_remove:
            graph.remove_node(nid)

    return count
