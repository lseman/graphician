"""Data flow edge extraction.

Emits DataFlow edges between nodes representing value propagation:
- Assignment: `let x = expr` → if expr references a known node,
  ref_node -[DataFlow]-> x_node
- Parameter→body: function parameter flows into its first use
- Return value: `return expr` → value flows to the caller's receiver
- Field write: `obj.field = val` → value flows to the field node

Each edge connects two named nodes (functions, methods, classes, variables).
Variable nodes are created on-the-fly when an assignment is detected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
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
    flow_type: str  # "assignment", "param_flow", "return_flow"
    source_text: str


def extract_data_flow(
    graph: Graph,
    function_id: NodeId,
    source_text: str,
    params: list[str] | None = None,
) -> list[DataFlowEdge]:
    """Extract data flow edges from a function/method body.

    Scans the source text for assignment patterns, parameter usage,
    return statements, and field writes. Creates variable nodes for
    assignments and emits DataFlow edges between them.

    Returns list of DataFlowEdge objects describing each edge.
    """
    if params is None:
        params = extract_params(source_text)

    edges: list[DataFlowEdge] = []
    edges.extend(extract_assignments(graph, function_id, source_text, params))
    edges.extend(extract_return_flow(graph, function_id, source_text, params))
    return edges


def extract_assignments(
    graph: Graph,
    function_id: NodeId,
    source_text: str,
    params: list[str],
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
            ))

    return edges


def extract_return_flow(
    graph: Graph,
    function_id: NodeId,
    source_text: str,
    params: list[str],
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
    if s.startswith("let "):
        s = s[4:]
    # Strip `mut ` prefix (Rust)
    elif s.startswith("mut "):
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


def extract_params(source_text: str) -> list[str]:
    """Build a parameter list from a function's source text.

    Extracts parameter names from function signatures across languages.
    """
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
