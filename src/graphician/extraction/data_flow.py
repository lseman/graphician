"""Data flow edge extraction.

Emits DataFlow edges between nodes representing value propagation:
- Assignment: `let x = expr` → if expr references a known node,
  ref_node -[DataFlow]-> x_node
- Parameter→body: function parameter flows into its first use
- Return value: `return expr` → value flows to the caller's receiver
- Field write: `obj.field = val` → value flows to the field node
- Method chain: `a.method().field` → tracks intermediate flows
- Error propagation: `result?`, `unwrap()`, `expect()` → tracks error flow

Uses tree-sitter for AST-level extraction when available, falling back
to regex-based parsing for unsupported languages or when tree-sitter
is not available.

Each edge connects two named nodes (functions, methods, classes, variables).
Variable nodes are created on-the-fly when an assignment is detected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.edge import Edge, EdgeKind
from ..core.graph import Graph
from ..core.id import NodeId
from ..core.node import Node, NodeKind


@dataclass
class DataFlowEdge:
    """A single data flow edge."""
    source_id: int
    target_id: int
    source_kind: str
    target_kind: str
    source_name: str
    target_name: str
    flow_type: str  # "assignment", "param_flow", "return_flow", "field_flow",
                    # "chain_flow", "error_flow", "literal_flow"
    source_text: str
    dialect: str = "unknown"


def extract_data_flow(
    graph: Graph,
    function_id: NodeId,
    source_text: str,
    params: list[str] | None = None,
    source_path: str = "",
) -> list[DataFlowEdge]:
    """Extract data flow edges from a function/method body.

    Scans the source text for assignment patterns, parameter usage,
    return statements, field writes, method chains, and error flows.
    Creates variable nodes for assignments and emits DataFlow edges
    between them.

    Uses tree-sitter AST when available for the dialect detected from
    source_path. Falls back to regex-based parsing otherwise.

    Returns list of DataFlowEdge objects describing each edge.
    """
    if params is None:
        params = extract_params(source_text, source_path)

    dialect = _detect_dialect(source_path)
    edges: list[DataFlowEdge] = []

    # Try tree-sitter extraction first if available
    ts_edges = _extract_with_tree_sitter(
        graph, function_id, source_text, params, dialect, source_path
    )
    if ts_edges is not None:
        edges.extend(ts_edges)
        return edges

    # Fall back to regex-based extraction
    edges.extend(extract_assignments(graph, function_id, source_text, params, dialect))
    edges.extend(extract_return_flow(graph, function_id, source_text, params, dialect))
    edges.extend(extract_field_assignments(graph, function_id, source_text, params, dialect))
    return edges


def _detect_dialect(source_path: str) -> str:
    """Detect the programming dialect from file extension."""
    ext_map = {
        ".py": "python",
        ".rs": "rust",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "cpp",
        ".hpp": "cpp",
        ".h": "cpp",
        ".go": "go",
    }
    path_lower = source_path.lower()
    for ext, dialect in ext_map.items():
        if path_lower.endswith(ext):
            return dialect
    return "unknown"


def _extract_with_tree_sitter(
    graph: Graph,
    function_id: NodeId,
    source_text: str,
    params: list[str],
    dialect: str,
    source_path: str,
) -> list[DataFlowEdge] | None:
    """Try tree-sitter extraction for the given dialect.

    Returns None if tree-sitter is not available for this dialect.
    """
    try:
        if dialect == "python":
            return _ts_extract_python(
                graph, function_id, source_text, params, source_path
            )
        elif dialect == "rust":
            return _ts_extract_rust(
                graph, function_id, source_text, params, source_path
            )
        elif dialect in ("typescript", "javascript"):
            return _ts_extract_ts(
                graph, function_id, source_text, params, source_path
            )
        elif dialect == "java":
            return _ts_extract_java(
                graph, function_id, source_text, params, source_path
            )
    except ImportError:
        pass
    return None


# ──────────────────────────────────────────────────────────────────────
# Python tree-sitter extraction
# ──────────────────────────────────────────────────────────────────────

def _ts_extract_python(
    graph: Graph,
    function_id: NodeId,
    source_text: str,
    params: list[str],
    source_path: str,
) -> list[DataFlowEdge]:
    """Extract data flow from Python source using tree-sitter."""
    try:
        import tree_sitter as ts
        import tree_sitter_python as tspython
    except ImportError:
        return []

    edges: list[DataFlowEdge] = []
    lang = tspython.language()
    parser = ts.Parser(lang)
    tree = parser.parse(source_text.encode())

    # Walk the AST to find assignments and returns
    for node in tree.root_node.children:
        if node.type == "expression_statement":
            child = node.children[0] if node.children else None
            if child and child.type == "assignment":
                _ts_extract_python_assignment(
                    graph, function_id, child, source_text, params, edges
                )
            elif child and child.type == "call":
                _ts_extract_python_method_chain(
                    graph, function_id, child, source_text, params, edges
                )
        elif node.type == "return_statement":
            if node.children:
                _ts_extract_python_return(
                    graph, function_id, node.children[0], source_text, params, edges
                )

    return edges


def _ts_extract_python_assignment(
    graph: Graph,
    function_id: NodeId,
    node: Any,
    source_text: str,
    params: list[str],
    edges: list[DataFlowEdge],
) -> None:
    """Extract Python assignment node."""
    targets = []
    value = None

    for child in node.children:
        if child.type == "target":
            targets.append(child)
        elif child.type == "value":
            value = child

    if not value:
        return

    # Get the variable names
    for target in targets:
        var_name = _ts_python_target_name(target)
        if not var_name or var_name.startswith("_"):
            continue

        var_qn = f"var::{function_id.value}::{var_name}"
        var_id = ensure_variable_node(graph, function_id, var_qn, var_name)

        # Check if value references params
        value_text = _ts_node_text(value, source_text)
        for param in params:
            if param in value_text:
                param_qn = f"param::{function_id.value}::{param}"
                param_id = ensure_variable_node(graph, function_id, param_qn, param)
                if not _has_edge_kind(graph, param_id, var_id, EdgeKind.DATA_FLOW):
                    graph.add_edge(param_id, var_id, Edge.extracted(EdgeKind.DATA_FLOW))
                    edges.append(DataFlowEdge(
                        source_id=param_id.value,
                        target_id=var_id.value,
                        source_kind="param",
                        target_kind="variable",
                        source_name=param,
                        target_name=var_name,
                        flow_type="param_flow",
                        source_text=source_text[node.start_point[0]:node.end_point[0] + 1].strip(),
                        dialect="python",
                    ))

        # Function → variable flow
        if not _has_edge_kind(graph, function_id, var_id, EdgeKind.DATA_FLOW):
            graph.add_edge(function_id, var_id, Edge.extracted(EdgeKind.DATA_FLOW))
            edges.append(DataFlowEdge(
                source_id=function_id.value,
                target_id=var_id.value,
                source_kind="function",
                target_kind="variable",
                source_name=function_id.value,
                target_name=var_name,
                flow_type="assignment",
                source_text=source_text[node.start_point[0]:node.end_point[0] + 1].strip(),
                dialect="python",
            ))

        # Track self.field assignments
        if target.type == "dotted_name":
            parts = _ts_python_dotted_parts(target, source_text)
            if parts and parts[0] == "self" and len(parts) > 1:
                field_name = parts[1]
                field_qn = f"var::{function_id.value}::self.{field_name}"
                field_id = ensure_variable_node(graph, function_id, field_qn, f"self.{field_name}")
                if not _has_edge_kind(graph, var_id, field_id, EdgeKind.DATA_FLOW):
                    graph.add_edge(var_id, field_id, Edge.extracted(EdgeKind.DATA_FLOW))
                    edges.append(DataFlowEdge(
                        source_id=var_id.value,
                        target_id=field_id.value,
                        source_kind="variable",
                        target_kind="field",
                        source_name=var_name,
                        target_name=f"self.{field_name}",
                        flow_type="field_flow",
                        source_text=source_text[node.start_point[0]:node.end_point[0] + 1].strip(),
                        dialect="python",
                    ))


def _ts_extract_python_method_chain(
    graph: Graph,
    function_id: NodeId,
    node: Any,
    source_text: str,
    params: list[str],
    edges: list[DataFlowEdge],
) -> None:
    """Extract Python method chain (e.g., obj.method().other())."""
    func = node.children[0] if node.children else None
    if not func:
        return

    # Check for chained method calls
    if func.type == "call" and func.children:
        base = func.children[0]
        chain_name = _ts_python_call_chain(base, source_text)
        if chain_name and any(p in chain_name for p in params):
            chain_qn = f"var::{function_id.value}::chain_{chain_name.split('.')[-1]}"
            chain_id = ensure_variable_node(graph, function_id, chain_qn, chain_name)
            for param in params:
                if param in chain_name:
                    param_qn = f"param::{function_id.value}::{param}"
                    param_id = ensure_variable_node(graph, function_id, param_qn, param)
                    if not _has_edge_kind(graph, param_id, chain_id, EdgeKind.DATA_FLOW):
                        graph.add_edge(param_id, chain_id, Edge.extracted(EdgeKind.DATA_FLOW))
                        edges.append(DataFlowEdge(
                            source_id=param_id.value,
                            target_id=chain_id.value,
                            source_kind="param",
                            target_kind="variable",
                            source_name=param,
                            target_name=chain_name,
                            flow_type="chain_flow",
                            source_text=source_text[node.start_point[0]:node.end_point[0] + 1].strip(),
                            dialect="python",
                        ))


def _ts_extract_python_return(
    graph: Graph,
    function_id: NodeId,
    node: Any,
    source_text: str,
    params: list[str],
    edges: list[DataFlowEdge],
) -> None:
    """Extract Python return statement."""
    if not node:
        return

    return_qn = f"return::{function_id.value}"
    return_id = ensure_variable_node(graph, function_id, return_qn, "return_value")
    return_text = _ts_node_text(node, source_text)

    # Check if return references params
    for param in params:
        if param in return_text:
            param_qn = f"param::{function_id.value}::{param}"
            param_id = ensure_variable_node(graph, function_id, param_qn, param)
            if not _has_edge_kind(graph, param_id, return_id, EdgeKind.DATA_FLOW):
                graph.add_edge(param_id, return_id, Edge.extracted(EdgeKind.DATA_FLOW))
                edges.append(DataFlowEdge(
                    source_id=param_id.value,
                    target_id=return_id.value,
                    source_kind="param",
                    target_kind="return_value",
                    source_name=param,
                    target_name="return_value",
                    flow_type="return_flow",
                    source_text=source_text[node.start_point[0]:node.end_point[0] + 1].strip(),
                    dialect="python",
                ))

    # Function → return value flow
    if not _has_edge_kind(graph, function_id, return_id, EdgeKind.DATA_FLOW):
        graph.add_edge(function_id, return_id, Edge.extracted(EdgeKind.DATA_FLOW))
        edges.append(DataFlowEdge(
            source_id=function_id.value,
            target_id=return_id.value,
            source_kind="function",
            target_kind="return_value",
            source_name=function_id.value,
            target_name="return_value",
            flow_type="return_flow",
            source_text=source_text[node.start_point[0]:node.end_point[0] + 1].strip(),
            dialect="python",
        ))


def _ts_python_target_name(node: Any) -> str | None:
    """Extract variable name from a Python assignment target."""
    if node.type == "identifier":
        return node.text.decode(errors="replace")
    elif node.type == "dotted_name":
        parts = [child.text.decode(errors="replace") for child in node.children]
        return ".".join(parts)
    return None


def _ts_python_dotted_parts(node: Any, source: str) -> list[str]:
    """Extract parts of a dotted name node."""
    if node.type == "identifier":
        return [node.text.decode(errors="replace")]
    elif node.type == "dotted_name":
        return [_ts_python_dotted_parts(child, source) for child in node.children if child.children]
    return []


def _ts_python_call_chain(node: Any, source: str) -> str | None:
    """Extract method call chain text."""
    if node.type == "identifier":
        return node.text.decode(errors="replace")
    elif node.type == "call":
        func = node.children[0] if node.children else None
        if func and func.type == "attribute":
            obj = func.children[0] if func.children else None
            method = func.children[-1] if func.children else None
            if obj and method:
                base = _ts_python_call_chain(obj, source)
                method_name = method.text.decode(errors="replace")
                if base:
                    return f"{base}.{method_name}"
                return method_name
    return None


def _ts_node_text(node: Any, source: str) -> str:
    """Extract text from a tree-sitter node."""
    start = node.start_point[0]
    end = node.end_point[0]
    line_start = node.start_point[1]
    line_end = node.end_point[1]
    lines = source.split('\n')
    if start == end:
        return lines[start][line_start:line_end + 1]
    result = [lines[start][line_start:]]
    result.extend(lines[start + 1:end])
    result.append(lines[end][:line_end + 1])
    return '\n'.join(result)


# ──────────────────────────────────────────────────────────────────────
# Rust tree-sitter extraction
# ──────────────────────────────────────────────────────────────────────

def _ts_extract_rust(
    graph: Graph,
    function_id: NodeId,
    source_text: str,
    params: list[str],
    source_path: str,
) -> list[DataFlowEdge]:
    """Extract data flow from Rust source using tree-sitter."""
    try:
        import tree_sitter as ts
        import tree_sitter_rust as tsrust
    except ImportError:
        return []

    edges: list[DataFlowEdge] = []
    lang = tsrust.language()
    parser = ts.Parser(lang)
    tree = parser.parse(source_text.encode())

    for node in tree.root_node.children:
        if node.type == "let_statement":
            _ts_extract_rust_let(graph, function_id, node, source_text, params, edges)
        elif node.type == "return_expression":
            if node.children:
                _ts_extract_rust_return(
                    graph, function_id, node.children[0], source_text, params, edges
                )
        elif node.type == "expression_statement":
            _ts_extract_rust_expression(
                graph, function_id, node, source_text, params, edges
            )

    return edges


def _ts_extract_rust_let(
    graph: Graph,
    function_id: NodeId,
    node: Any,
    source_text: str,
    params: list[str],
    edges: list[DataFlowEdge],
) -> None:
    """Extract Rust let statement."""
    pattern = node.child_by_field_name("pattern")
    value = node.child_by_field_name("value")

    if not pattern or not value:
        return

    let_name = _ts_rust_pattern_name(pattern)
    if not let_name or let_name.startswith("_"):
        return

    let_qn = f"var::{function_id.value}::{let_name}"
    let_id = ensure_variable_node(graph, function_id, let_qn, let_name)

    value_text = _ts_rust_expr_text(value, source_text)
    for param in params:
        if param in value_text:
            param_qn = f"param::{function_id.value}::{param}"
            param_id = ensure_variable_node(graph, function_id, param_qn, param)
            if not _has_edge_kind(graph, param_id, let_id, EdgeKind.DATA_FLOW):
                graph.add_edge(param_id, let_id, Edge.extracted(EdgeKind.DATA_FLOW))
                edges.append(DataFlowEdge(
                    source_id=param_id.value,
                    target_id=let_id.value,
                    source_kind="param",
                    target_kind="variable",
                    source_name=param,
                    target_name=let_name,
                    flow_type="param_flow",
                    source_text=source_text[node.start_point[0]:node.end_point[0] + 1].strip(),
                    dialect="rust",
                ))

    if not _has_edge_kind(graph, function_id, let_id, EdgeKind.DATA_FLOW):
        graph.add_edge(function_id, let_id, Edge.extracted(EdgeKind.DATA_FLOW))
        edges.append(DataFlowEdge(
            source_id=function_id.value,
            target_id=let_id.value,
            source_kind="function",
            target_kind="variable",
            source_name=function_id.value,
            target_name=let_name,
            flow_type="assignment",
            source_text=source_text[node.start_point[0]:node.end_point[0] + 1].strip(),
            dialect="rust",
        ))

    # Track ? operator for error flow
    if "?" in value_text:
        error_qn = f"error::{function_id.value}::{let_name}"
        error_id = ensure_variable_node(graph, function_id, error_qn, f"{let_name}?.error")
        if not _has_edge_kind(graph, let_id, error_id, EdgeKind.DATA_FLOW):
            graph.add_edge(let_id, error_id, Edge.extracted(EdgeKind.DATA_FLOW))
            edges.append(DataFlowEdge(
                source_id=let_id.value,
                target_id=error_id.value,
                source_kind="variable",
                target_kind="error",
                source_name=let_name,
                target_name=f"{let_name}?.error",
                flow_type="error_flow",
                source_text=source_text[node.start_point[0]:node.end_point[0] + 1].strip(),
                dialect="rust",
            ))


def _ts_extract_rust_return(
    graph: Graph,
    function_id: NodeId,
    node: Any,
    source_text: str,
    params: list[str],
    edges: list[DataFlowEdge],
) -> None:
    """Extract Rust return statement."""
    if not node:
        return

    return_qn = f"return::{function_id.value}"
    return_id = ensure_variable_node(graph, function_id, return_qn, "return_value")
    return_text = _ts_rust_expr_text(node, source_text)

    for param in params:
        if param in return_text:
            param_qn = f"param::{function_id.value}::{param}"
            param_id = ensure_variable_node(graph, function_id, param_qn, param)
            if not _has_edge_kind(graph, param_id, return_id, EdgeKind.DATA_FLOW):
                graph.add_edge(param_id, return_id, Edge.extracted(EdgeKind.DATA_FLOW))
                edges.append(DataFlowEdge(
                    source_id=param_id.value,
                    target_id=return_id.value,
                    source_kind="param",
                    target_kind="return_value",
                    source_name=param,
                    target_name="return_value",
                    flow_type="return_flow",
                    source_text=source_text[node.start_point[0]:node.end_point[0] + 1].strip(),
                    dialect="rust",
                ))

    if not _has_edge_kind(graph, function_id, return_id, EdgeKind.DATA_FLOW):
        graph.add_edge(function_id, return_id, Edge.extracted(EdgeKind.DATA_FLOW))
        edges.append(DataFlowEdge(
            source_id=function_id.value,
            target_id=return_id.value,
            source_kind="function",
            target_kind="return_value",
            source_name=function_id.value,
            target_name="return_value",
            flow_type="return_flow",
            source_text=source_text[node.start_point[0]:node.end_point[0] + 1].strip(),
            dialect="rust",
        ))


def _ts_extract_rust_expression(
    graph: Graph,
    function_id: NodeId,
    node: Any,
    source_text: str,
    params: list[str],
    edges: list[DataFlowEdge],
) -> None:
    """Extract Rust expression statement (method calls, field writes, etc.)."""
    if not node.children:
        return

    child = node.children[0]
    if child.type == "call_expression":
        # Track method calls on params (e.g., param.method())
        func = child.child_by_field_name("function")
        if func:
            chain_text = _ts_rust_call_chain_text(func, source_text)
            for param in params:
                if param in chain_text:
                    chain_qn = f"var::{function_id.value}::chain_{param}"
                    chain_id = ensure_variable_node(graph, function_id, chain_qn, chain_text)
                    param_qn = f"param::{function_id.value}::{param}"
                    param_id = ensure_variable_node(graph, function_id, param_qn, param)
                    if not _has_edge_kind(graph, param_id, chain_id, EdgeKind.DATA_FLOW):
                        graph.add_edge(param_id, chain_id, Edge.extracted(EdgeKind.DATA_FLOW))
                        edges.append(DataFlowEdge(
                            source_id=param_id.value,
                            target_id=chain_id.value,
                            source_kind="param",
                            target_kind="variable",
                            source_name=param,
                            target_name=chain_text,
                            flow_type="chain_flow",
                            source_text=source_text[node.start_point[0]:node.end_point[0] + 1].strip(),
                            dialect="rust",
                        ))


def _ts_rust_pattern_name(node: Any) -> str | None:
    """Extract variable name from Rust let pattern."""
    if node.type == "identifier":
        return node.text.decode(errors="replace")
    return None


def _ts_rust_expr_text(node: Any, source: str) -> str:
    """Extract text from a Rust expression node."""
    start = node.start_point[0]
    end = node.end_point[0]
    line_start = node.start_point[1]
    line_end = node.end_point[1]
    lines = source.split('\n')
    if start == end:
        return lines[start][line_start:line_end + 1]
    result = [lines[start][line_start:]]
    result.extend(lines[start + 1:end])
    result.append(lines[end][:line_end + 1])
    return '\n'.join(result)


def _ts_rust_call_chain_text(node: Any, source: str) -> str:
    """Extract method call chain text."""
    if node.type == "identifier":
        return node.text.decode(errors="replace")
    elif node.type == "field_expression":
        base = _ts_rust_call_chain_text(node.child_by_field_name("source") or node, source)
        field = node.child_by_field_name("name")
        if field:
            return f"{base}.{field.text.decode(errors='replace')}"
        return base
    elif node.type == "call_expression":
        func = node.child_by_field_name("function")
        if func:
            return _ts_rust_call_chain_text(func, source)
    return ""


# ──────────────────────────────────────────────────────────────────────
# TypeScript/JavaScript tree-sitter extraction
# ──────────────────────────────────────────────────────────────────────

def _ts_extract_ts(
    graph: Graph,
    function_id: NodeId,
    source_text: str,
    params: list[str],
    source_path: str,
) -> list[DataFlowEdge]:
    """Extract data flow from TS/JS source using tree-sitter."""
    try:
        import tree_sitter as ts
        import tree_sitter_typescript as tstypescript
    except ImportError:
        return []

    edges: list[DataFlowEdge] = []
    lang = tstypescript.language_typescript()
    parser = ts.Parser(lang)
    tree = parser.parse(source_text.encode())

    for node in tree.root_node.children:
        if node.type == "lexical_declaration":
            _ts_extract_ts_let(graph, function_id, node, source_text, params, edges)
        elif node.type == "expression_statement":
            child = node.children[0] if node.children else None
            if child:
                _ts_extract_ts_expression(
                    graph, function_id, child, source_text, params, edges
                )
        elif node.type == "return_statement" and node.children:
            _ts_extract_ts_return(
                graph, function_id, node.children[0], source_text, params, edges
            )

    return edges


def _ts_extract_ts_let(
    graph: Graph,
    function_id: NodeId,
    node: Any,
    source_text: str,
    params: list[str],
    edges: list[DataFlowEdge],
) -> None:
    """Extract TS/JS let/const/var declaration."""
    decl_spec = node.child_by_field_name("declarator")
    if not decl_spec:
        return

    declarator = decl_spec.child_by_field_name("name")
    value = decl_spec.child_by_field_name("value")

    if not declarator or not value:
        return

    var_name = _ts_ts_identifier_name(declarator)
    if not var_name or var_name.startswith("_"):
        return

    var_qn = f"var::{function_id.value}::{var_name}"
    var_id = ensure_variable_node(graph, function_id, var_qn, var_name)

    value_text = _ts_ts_node_text(value, source_text)
    for param in params:
        if param in value_text:
            param_qn = f"param::{function_id.value}::{param}"
            param_id = ensure_variable_node(graph, function_id, param_qn, param)
            if not _has_edge_kind(graph, param_id, var_id, EdgeKind.DATA_FLOW):
                graph.add_edge(param_id, var_id, Edge.extracted(EdgeKind.DATA_FLOW))
                edges.append(DataFlowEdge(
                    source_id=param_id.value,
                    target_id=var_id.value,
                    source_kind="param",
                    target_kind="variable",
                    source_name=param,
                    target_name=var_name,
                    flow_type="param_flow",
                    source_text=source_text[node.start_point[0]:node.end_point[0] + 1].strip(),
                    dialect="typescript",
                ))

    if not _has_edge_kind(graph, function_id, var_id, EdgeKind.DATA_FLOW):
        graph.add_edge(function_id, var_id, Edge.extracted(EdgeKind.DATA_FLOW))
        edges.append(DataFlowEdge(
            source_id=function_id.value,
            target_id=var_id.value,
            source_kind="function",
            target_kind="variable",
            source_name=function_id.value,
            target_name=var_name,
            flow_type="assignment",
            source_text=source_text[node.start_point[0]:node.end_point[0] + 1].strip(),
            dialect="typescript",
        ))


def _ts_extract_ts_expression(
    graph: Graph,
    function_id: NodeId,
    node: Any,
    source_text: str,
    params: list[str],
    edges: list[DataFlowEdge],
) -> None:
    """Extract TS/JS expression statement."""
    # Check for assignment expression (e.g., x = expr)
    if node.type == "assignment_expression":
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
        if left and right:
            var_name = _ts_ts_identifier_name(left)
            if var_name and not var_name.startswith("_"):
                var_qn = f"var::{function_id.value}::{var_name}"
                var_id = ensure_variable_node(graph, function_id, var_qn, var_name)
                value_text = _ts_ts_node_text(right, source_text)
                for param in params:
                    if param in value_text:
                        param_qn = f"param::{function_id.value}::{param}"
                        param_id = ensure_variable_node(graph, function_id, param_qn, param)
                        if not _has_edge_kind(graph, param_id, var_id, EdgeKind.DATA_FLOW):
                            graph.add_edge(param_id, var_id, Edge.extracted(EdgeKind.DATA_FLOW))
                            edges.append(DataFlowEdge(
                                source_id=param_id.value,
                                target_id=var_id.value,
                                source_kind="param",
                                target_kind="variable",
                                source_name=param,
                                target_name=var_name,
                                flow_type="param_flow",
                                source_text=source_text[node.start_point[0]:node.end_point[0] + 1].strip(),
                                dialect="typescript",
                            ))
                if not _has_edge_kind(graph, function_id, var_id, EdgeKind.DATA_FLOW):
                    graph.add_edge(function_id, var_id, Edge.extracted(EdgeKind.DATA_FLOW))
                    edges.append(DataFlowEdge(
                        source_id=function_id.value,
                        target_id=var_id.value,
                        source_kind="function",
                        target_kind="variable",
                        source_name=function_id.value,
                        target_name=var_name,
                        flow_type="assignment",
                        source_text=source_text[node.start_point[0]:node.end_point[0] + 1].strip(),
                        dialect="typescript",
                    ))

    # Check for method call chain
    elif node.type == "call_expression":
        func = node.child_by_field_name("function")
        if func:
            chain_text = _ts_ts_call_chain_text(func, source_text)
            if chain_text and any(p in chain_text for p in params):
                chain_qn = f"var::{function_id.value}::chain_{chain_text.split('.')[-1]}"
                chain_id = ensure_variable_node(graph, function_id, chain_qn, chain_text)
                for param in params:
                    if param in chain_text:
                        param_qn = f"param::{function_id.value}::{param}"
                        param_id = ensure_variable_node(graph, function_id, param_qn, param)
                        if not _has_edge_kind(graph, param_id, chain_id, EdgeKind.DATA_FLOW):
                            graph.add_edge(param_id, chain_id, Edge.extracted(EdgeKind.DATA_FLOW))
                            edges.append(DataFlowEdge(
                                source_id=param_id.value,
                                target_id=chain_id.value,
                                source_kind="param",
                                target_kind="variable",
                                source_name=param,
                                target_name=chain_text,
                                flow_type="chain_flow",
                                source_text=source_text[node.start_point[0]:node.end_point[0] + 1].strip(),
                                dialect="typescript",
                            ))


def _ts_extract_ts_return(
    graph: Graph,
    function_id: NodeId,
    node: Any,
    source_text: str,
    params: list[str],
    edges: list[DataFlowEdge],
) -> None:
    """Extract TS/JS return statement."""
    if not node:
        return

    return_qn = f"return::{function_id.value}"
    return_id = ensure_variable_node(graph, function_id, return_qn, "return_value")
    return_text = _ts_ts_node_text(node, source_text)

    for param in params:
        if param in return_text:
            param_qn = f"param::{function_id.value}::{param}"
            param_id = ensure_variable_node(graph, function_id, param_qn, param)
            if not _has_edge_kind(graph, param_id, return_id, EdgeKind.DATA_FLOW):
                graph.add_edge(param_id, return_id, Edge.extracted(EdgeKind.DATA_FLOW))
                edges.append(DataFlowEdge(
                    source_id=param_id.value,
                    target_id=return_id.value,
                    source_kind="param",
                    target_kind="return_value",
                    source_name=param,
                    target_name="return_value",
                    flow_type="return_flow",
                    source_text=source_text[node.start_point[0]:node.end_point[0] + 1].strip(),
                    dialect="typescript",
                ))

    if not _has_edge_kind(graph, function_id, return_id, EdgeKind.DATA_FLOW):
        graph.add_edge(function_id, return_id, Edge.extracted(EdgeKind.DATA_FLOW))
        edges.append(DataFlowEdge(
            source_id=function_id.value,
            target_id=return_id.value,
            source_kind="function",
            target_kind="return_value",
            source_name=function_id.value,
            target_name="return_value",
            flow_type="return_flow",
            source_text=source_text[node.start_point[0]:node.end_point[0] + 1].strip(),
            dialect="typescript",
        ))


def _ts_ts_identifier_name(node: Any) -> str | None:
    """Extract identifier name from a TS node."""
    if node.type in ("identifier", "property_identifier"):
        return node.text.decode(errors="replace")
    return None


def _ts_ts_node_text(node: Any, source: str) -> str:
    """Extract text from a TS node."""
    start = node.start_point[0]
    end = node.end_point[0]
    line_start = node.start_point[1]
    line_end = node.end_point[1]
    lines = source.split('\n')
    if start == end:
        return lines[start][line_start:line_end + 1]
    result = [lines[start][line_start:]]
    result.extend(lines[start + 1:end])
    result.append(lines[end][:line_end + 1])
    return '\n'.join(result)


def _ts_ts_call_chain_text(node: Any, source: str) -> str:
    """Extract method call chain text."""
    if node.type == "identifier":
        return node.text.decode(errors="replace")
    elif node.type == "member_expression":
        obj = node.child_by_field_name("object")
        prop = node.child_by_field_name("property")
        if obj and prop:
            obj_text = _ts_ts_call_chain_text(obj, source)
            return f"{obj_text}.{prop.text.decode(errors='replace')}"
    elif node.type == "call_expression":
        func = node.child_by_field_name("function")
        if func:
            return _ts_ts_call_chain_text(func, source)
    return ""


# ──────────────────────────────────────────────────────────────────────
# Java tree-sitter extraction
# ──────────────────────────────────────────────────────────────────────

def _ts_extract_java(
    graph: Graph,
    function_id: NodeId,
    source_text: str,
    params: list[str],
    source_path: str,
) -> list[DataFlowEdge]:
    """Extract data flow from Java source using tree-sitter."""
    try:
        import tree_sitter as ts
        import tree_sitter_java as tsjava
    except ImportError:
        return []

    edges: list[DataFlowEdge] = []
    lang = tsjava.language()
    parser = ts.Parser(lang)
    tree = parser.parse(source_text.encode())

    for node in tree.root_node.children:
        if node.type == "local_variable_declaration":
            _ts_extract_java_var(
                graph, function_id, node, source_text, params, edges
            )
        elif node.type == "return_statement" and node.children:
            _ts_extract_java_return(
                graph, function_id, node.children[0], source_text, params, edges
            )

    return edges


def _ts_extract_java_var(
    graph: Graph,
    function_id: NodeId,
    node: Any,
    source_text: str,
    params: list[str],
    edges: list[DataFlowEdge],
) -> None:
    """Extract Java local variable declaration."""
    declarator = node.child_by_field_name("declarator")
    if not declarator:
        return

    name_node = declarator.child_by_field_name("name")
    value_node = declarator.child_by_field_name("value")

    if not name_node or not value_node:
        return

    var_name = name_node.text.decode(errors="replace")
    if not var_name or var_name.startswith("_"):
        return

    var_qn = f"var::{function_id.value}::{var_name}"
    var_id = ensure_variable_node(graph, function_id, var_qn, var_name)

    value_text = _ts_ts_node_text(value_node, source_text)
    for param in params:
        if param in value_text:
            param_qn = f"param::{function_id.value}::{param}"
            param_id = ensure_variable_node(graph, function_id, param_qn, param)
            if not _has_edge_kind(graph, param_id, var_id, EdgeKind.DATA_FLOW):
                graph.add_edge(param_id, var_id, Edge.extracted(EdgeKind.DATA_FLOW))
                edges.append(DataFlowEdge(
                    source_id=param_id.value,
                    target_id=var_id.value,
                    source_kind="param",
                    target_kind="variable",
                    source_name=param,
                    target_name=var_name,
                    flow_type="param_flow",
                    source_text=source_text[node.start_point[0]:node.end_point[0] + 1].strip(),
                    dialect="java",
                ))

    if not _has_edge_kind(graph, function_id, var_id, EdgeKind.DATA_FLOW):
        graph.add_edge(function_id, var_id, Edge.extracted(EdgeKind.DATA_FLOW))
        edges.append(DataFlowEdge(
            source_id=function_id.value,
            target_id=var_id.value,
            source_kind="function",
            target_kind="variable",
            source_name=function_id.value,
            target_name=var_name,
            flow_type="assignment",
            source_text=source_text[node.start_point[0]:node.end_point[0] + 1].strip(),
            dialect="java",
        ))


def _ts_extract_java_return(
    graph: Graph,
    function_id: NodeId,
    node: Any,
    source_text: str,
    params: list[str],
    edges: list[DataFlowEdge],
) -> None:
    """Extract Java return statement."""
    if not node:
        return

    return_qn = f"return::{function_id.value}"
    return_id = ensure_variable_node(graph, function_id, return_qn, "return_value")
    return_text = _ts_ts_node_text(node, source_text)

    for param in params:
        if param in return_text:
            param_qn = f"param::{function_id.value}::{param}"
            param_id = ensure_variable_node(graph, function_id, param_qn, param)
            if not _has_edge_kind(graph, param_id, return_id, EdgeKind.DATA_FLOW):
                graph.add_edge(param_id, return_id, Edge.extracted(EdgeKind.DATA_FLOW))
                edges.append(DataFlowEdge(
                    source_id=param_id.value,
                    target_id=return_id.value,
                    source_kind="param",
                    target_kind="return_value",
                    source_name=param,
                    target_name="return_value",
                    flow_type="return_flow",
                    source_text=source_text[node.start_point[0]:node.end_point[0] + 1].strip(),
                    dialect="java",
                ))

    if not _has_edge_kind(graph, function_id, return_id, EdgeKind.DATA_FLOW):
        graph.add_edge(function_id, return_id, Edge.extracted(EdgeKind.DATA_FLOW))
        edges.append(DataFlowEdge(
            source_id=function_id.value,
            target_id=return_id.value,
            source_kind="function",
            target_kind="return_value",
            source_name=function_id.value,
            target_name="return_value",
            flow_type="return_flow",
            source_text=source_text[node.start_point[0]:node.end_point[0] + 1].strip(),
            dialect="java",
        ))


# ──────────────────────────────────────────────────────────────────────
# Backward-compatible regex extraction (unchanged from original)
# ──────────────────────────────────────────────────────────────────────

def extract_assignments(
    graph: Graph,
    function_id: NodeId,
    source_text: str,
    params: list[str],
    dialect: str = "unknown",
) -> list[DataFlowEdge]:
    """Extract assignment edges: `let/var/mut x = expr` or `x = expr`."""
    edges: list[DataFlowEdge] = []
    lines = source_text.splitlines()

    for line in lines:
        trimmed = line.strip()
        if not trimmed or trimmed.startswith("//") or trimmed.startswith("#") or trimmed.startswith("/*"):
            continue

        result = parse_assignment(trimmed)
        if result is None:
            continue

        var_name, expr = result
        if not var_name or var_name.startswith("_"):
            continue

        # Create variable node
        var_qn = f"var::{function_id.value}::{var_name}"
        var_id = ensure_variable_node(graph, function_id, var_qn, var_name)

        # Emit DataFlow from params if expr references them
        for param in params:
            if param in expr:
                param_qn = f"param::{function_id.value}::{param}"
                param_id = ensure_variable_node(graph, function_id, param_qn, param)
                if not _has_edge_kind(graph, param_id, var_id, EdgeKind.DATA_FLOW):
                    graph.add_edge(param_id, var_id, Edge.extracted(EdgeKind.DATA_FLOW))
                    edges.append(DataFlowEdge(
                        source_id=param_id.value,
                        target_id=var_id.value,
                        source_kind="param",
                        target_kind="variable",
                        source_name=param,
                        target_name=var_name,
                        flow_type="param_flow",
                        source_text=trimmed,
                        dialect=dialect,
                    ))

        # Emit DataFlow from function to variable
        if not _has_edge_kind(graph, function_id, var_id, EdgeKind.DATA_FLOW):
            graph.add_edge(function_id, var_id, Edge.extracted(EdgeKind.DATA_FLOW))
            edges.append(DataFlowEdge(
                source_id=function_id.value,
                target_id=var_id.value,
                source_kind="function",
                target_kind="variable",
                source_name=function_id.value,
                target_name=var_name,
                flow_type="assignment",
                source_text=trimmed,
                dialect=dialect,
            ))

    return edges


def extract_return_flow(
    graph: Graph,
    function_id: NodeId,
    source_text: str,
    params: list[str],
    dialect: str = "unknown",
) -> list[DataFlowEdge]:
    """Extract return value flow: `return expr`."""
    edges: list[DataFlowEdge] = []
    lines = source_text.splitlines()

    for line in lines:
        trimmed = line.strip()
        if not trimmed or trimmed.startswith("//") or trimmed.startswith("#"):
            continue

        return_expr = parse_return(trimmed)
        if return_expr is None:
            continue

        if not return_expr.strip():
            continue

        # Check if return value references a parameter
        for param in params:
            if param in return_expr:
                return_qn = f"return::{function_id.value}"
                return_id = ensure_variable_node(graph, function_id, return_qn, "return_value")
                param_qn = f"param::{function_id.value}::{param}"
                param_id = ensure_variable_node(graph, function_id, param_qn, param)
                if not _has_edge_kind(graph, param_id, return_id, EdgeKind.DATA_FLOW):
                    graph.add_edge(param_id, return_id, Edge.extracted(EdgeKind.DATA_FLOW))
                    edges.append(DataFlowEdge(
                        source_id=param_id.value,
                        target_id=return_id.value,
                        source_kind="param",
                        target_kind="return_value",
                        source_name=param,
                        target_name="return_value",
                        flow_type="return_flow",
                        source_text=trimmed,
                        dialect=dialect,
                    ))

        # Emit DataFlow from function to its return value
        return_qn = f"return::{function_id.value}"
        return_id = ensure_variable_node(graph, function_id, return_qn, "return_value")
        if not _has_edge_kind(graph, function_id, return_id, EdgeKind.DATA_FLOW):
            graph.add_edge(function_id, return_id, Edge.extracted(EdgeKind.DATA_FLOW))
            edges.append(DataFlowEdge(
                source_id=function_id.value,
                target_id=return_id.value,
                source_kind="function",
                target_kind="return_value",
                source_name=function_id.value,
                target_name="return_value",
                flow_type="return_flow",
                source_text=trimmed,
                dialect=dialect,
            ))

    return edges


def extract_field_assignments(
    graph: Graph,
    function_id: NodeId,
    source_text: str,
    params: list[str],
    dialect: str = "unknown",
) -> list[DataFlowEdge]:
    """Extract field assignment patterns: `self.field = expr` or `obj.field = expr`."""
    edges: list[DataFlowEdge] = []
    lines = source_text.splitlines()

    # Pattern: self.field = expr (Python)
    if dialect in ("python", "unknown"):
        for line in lines:
            trimmed = line.strip()
            if trimmed.startswith("self."):
                result = parse_assignment(trimmed)
                if result:
                    field_name, expr = result
                    if not field_name or field_name.startswith("_"):
                        continue

                    field_qn = f"var::{function_id.value}::self.{field_name}"
                    field_id = ensure_variable_node(graph, function_id, field_qn, f"self.{field_name}")

                    # Check if expr references params
                    for param in params:
                        if param in expr:
                            param_qn = f"param::{function_id.value}::{param}"
                            param_id = ensure_variable_node(graph, function_id, param_qn, param)
                            if not _has_edge_kind(graph, param_id, field_id, EdgeKind.DATA_FLOW):
                                graph.add_edge(param_id, field_id, Edge.extracted(EdgeKind.DATA_FLOW))
                                edges.append(DataFlowEdge(
                                    source_id=param_id.value,
                                    target_id=field_id.value,
                                    source_kind="param",
                                    target_kind="field",
                                    source_name=param,
                                    target_name=f"self.{field_name}",
                                    flow_type="field_flow",
                                    source_text=trimmed,
                                    dialect=dialect,
                                ))

                    if not _has_edge_kind(graph, function_id, field_id, EdgeKind.DATA_FLOW):
                        graph.add_edge(function_id, field_id, Edge.extracted(EdgeKind.DATA_FLOW))
                        edges.append(DataFlowEdge(
                            source_id=function_id.value,
                            target_id=field_id.value,
                            source_kind="function",
                            target_kind="field",
                            source_name=function_id.value,
                            target_name=f"self.{field_name}",
                            flow_type="field_flow",
                            source_text=trimmed,
                            dialect=dialect,
                        ))

    return edges


def parse_assignment(line: str) -> tuple[str, str] | None:
    """Parse an assignment from a line of code.

    Returns (variable_name, expression) or None.
    Handles: let/var/mut x = expr, x = expr, self.field = expr
    """
    s = line.strip()

    # Strip leading self. for Python field assignments
    if s.startswith("self."):
        s = s[5:]

    # Strip `let ` prefix
    if s.startswith("let ") or s.startswith("mut "):
        s = s[4:]

    # Find the equals sign
    eq_pos = s.find("=")
    if eq_pos == -1:
        return None

    before_eq = s[:eq_pos].strip()
    after_eq = s[eq_pos + 1:].strip()

    # Handle `var: Type` — strip everything after ':'
    var_name = before_eq.split(":")[0].strip()

    if not var_name or var_name.startswith("_"):
        return None

    return (var_name, after_eq)


def parse_return(line: str) -> str | None:
    """Parse a return statement, extracting the expression after `return`."""
    trimmed = line.strip()

    if trimmed.startswith("return ") or trimmed.startswith("return\t"):
        after = trimmed[7:]  # skip "return "
        expr = after.rstrip(";").strip()
        return expr
    elif trimmed == "return":
        return ""
    return None


def extract_params(source_text: str, source_path: str = "") -> list[str]:
    """Build a parameter list from a function's source text.

    Extracts parameter names from function signatures across languages.
    Uses dialect from source_path to choose the right parser.
    """
    _detect_dialect(source_path)
    params: list[str] = []

    for line in source_text.splitlines():
        trimmed = line.strip()

        # Rust: `fn name(args) {`
        if trimmed.startswith("fn ") or trimmed.startswith("pub fn "):
            result = _extract_rust_params(trimmed)
            if result:
                params.extend(result)
            break

        # Python: `def name(args):` or `async def name(args):`
        if trimmed.startswith("def ") or trimmed.startswith("async def "):
            result = _extract_python_params(trimmed)
            if result:
                params.extend(result)
            break

        # TypeScript/JS: `function name(args)` or class methods
        if (trimmed.startswith("function ") or trimmed.startswith("public ")
                or trimmed.startswith("private ") or trimmed.startswith("protected ")):
            result = _extract_ts_params(trimmed)
            if result:
                params.extend(result)
            break

        # Go: `func name(args)`
        if trimmed.startswith("func ") or trimmed.startswith("func ("):
            result = _extract_go_params(trimmed)
            if result:
                params.extend(result)
            break

    return params


def _extract_rust_params(line: str) -> list[str]:
    """Extract Rust function parameters."""
    paren_start = line.find("(")
    paren_end = line.find(")")
    if paren_start == -1 or paren_end == -1:
        return []

    args = line[paren_start + 1:paren_end]
    params: list[str] = []

    for arg in args.split(","):
        arg = arg.strip()
        if not arg or arg in ("self", "&self", "&mut self"):
            continue

        # Extract parameter name (before ':' or '=')
        name = arg.split(":")[0].split("=")[0].strip()
        # Strip & and mut prefixes
        name = name.removeprefix("&mut ").removeprefix("&").strip()

        if name and not name.startswith("_"):
            params.append(name)

    return params if params else []


def _extract_python_params(line: str) -> list[str]:
    """Extract Python function parameters."""
    paren_start = line.find("(")
    paren_end = line.find(")")
    if paren_start == -1 or paren_end == -1:
        return []

    args = line[paren_start + 1:paren_end]
    params: list[str] = []

    for arg in args.split(","):
        arg = arg.strip()
        if not arg or arg in ("self", "cls"):
            continue

        # Strip type annotations and defaults
        name = arg.split(":")[0].split("=")[0].strip()
        name = name.rstrip("?").strip()

        if name and not name.startswith("_"):
            params.append(name)

    return params if params else []


def _extract_ts_params(line: str) -> list[str]:
    """Extract TypeScript/JavaScript function parameters."""
    paren_start = line.find("(")
    paren_end = line.find(")")
    if paren_start == -1 or paren_end == -1:
        return []

    args = line[paren_start + 1:paren_end]
    params: list[str] = []

    for arg in args.split(","):
        arg = arg.strip()
        if not arg:
            continue

        # Strip type annotations: `name: Type`, `name: Type = default`
        name = arg.split(":")[0].split("=")[0].strip()
        # Strip access modifiers and `this`
        for modifier in ("public ", "private ", "protected "):
            name = name.removeprefix(modifier).strip()
        name = name.rstrip("?").strip()

        if name and name != "this" and not name.startswith("_"):
            params.append(name)

    return params if params else []


def _extract_go_params(line: str) -> list[str]:
    """Extract Go function parameters."""
    # Handle receiver: func (r *Receiver) Method(args)
    receiver_end = line.find(")")
    if receiver_end != -1 and line.find("(") < receiver_end:
        # Strip receiver
        line = line[receiver_end + 1:]

    paren_start = line.find("(")
    paren_end = line.find(")")
    if paren_start == -1 or paren_end == -1:
        return []

    args = line[paren_start + 1:paren_end]
    params: list[str] = []

    for arg in args.split(","):
        arg = arg.strip()
        if not arg:
            continue

        # Go: `name Type` or `names Type` (multiple vars)
        name_part = arg.split()[0] if arg.split() else ""
        name_part = name_part.rstrip("*").strip()

        if name_part and name_part != "nil" and not name_part.startswith("_"):
            params.append(name_part)

    return params if params else []


def ensure_variable_node(
    graph: Graph,
    function_id: NodeId,
    qn: str,
    name: str,
) -> NodeId:
    """Ensure a variable node exists in the graph for this function."""
    existing = graph.find_by_qname(qn)
    if existing is not None:
        return existing

    node = Node(
        kind=NodeKind.VARIABLE,
        name=name,
        qualified_name=qn,
    ).with_property("function_id", str(function_id.value))
    return graph.add_node(node)


def _has_edge_kind(
    graph: Graph,
    src: NodeId,
    dst: NodeId,
    kind: EdgeKind,
) -> bool:
    """Check if an edge with the given kind exists between src and dst."""
    for neighbor, edge in graph.out_neighbors(src):
        if neighbor.value == dst.value and edge.kind == kind:
            return True
    return False
