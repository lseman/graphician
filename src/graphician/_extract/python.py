"""Python wrapper for Rust-accelerated extraction.

This module provides a drop-in replacement for the Python AST walker
that uses the Rust tree-sitter bindings for significantly faster parsing.

Usage:
    from graphician._extract.python import extract_python_file
    # Same signature as graphician.extraction.languages.parsers.python.extract_file
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from graphician.core.edge import Edge, EdgeKind
from graphician.core.graph import Graph
from graphician.core.id import NodeId
from graphician.core.node import Node, NodeKind

# Import the Rust extension
from . import HAS_RUST

if HAS_RUST:
    from . import extract_data_flow as _rust_extract_data_flow
    from . import extract_python_file as _rust_extract_single
    from . import extract_python_files as _rust_extract_parallel


def _add_node(
    graph: Graph,
    kind: NodeKind,
    qn: str,
    path: Path,
    line_start: int,
    line_end: int,
    source_text: str | None = None,
    props: dict[str, Any] | None = None,
) -> NodeId:
    """Add a node to the graph, deduplicating by qualified_name."""
    existing = graph.find_by_qname(qn)
    if existing is not None:
        return existing
    node = Node.new(kind, qn)
    node = node.with_source(str(path), line_start + 1, line_end + 1)
    if source_text is not None:
        node = node.with_source_text(source_text)
    if props:
        for k, v in props.items():
            node = node.with_property(k, v)
    graph.add_node(node)
    return graph.find_by_qname(qn)


def _parse_properties(props_str: str) -> dict[str, Any]:
    """Parse a JSON string of properties into a dict."""
    import json
    try:
        return json.loads(props_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def extract_python_file(
    path: Path,
    graph: Graph,
    *,
    file_qn: str | None = None,
    source_path: Path | None = None,
) -> None:
    """Parse a Python source file and emit nodes/edges into the graph.

    Uses the Rust-accelerated extraction if available, falling back to
    the Python implementation otherwise.

    Args:
        path: Path to the Python source file.
        graph: The graph to add nodes and edges to.
        file_qn: Optional qualified name for the file node.
        source_path: Optional path to use for source_uri.
    """
    if not HAS_RUST:
        # Fallback to Python implementation
        from graphician.extraction.languages.parsers.python import (
            extract_file as _python_extract,
        )
        _python_extract(path, graph, file_qn=file_qn, source_path=source_path)
        return

    with open(path, "rb") as f:
        raw = f.read()

    source = raw.decode("utf-8", errors="replace")
    record_path = source_path if source_path is not None else path

    # Call the Rust extraction
    result = _rust_extract_single(raw, file_path=str(record_path), file_qn=file_qn or "")

    # Create file node
    file_name = record_path.stem
    file_qn_full = file_qn or f"file::{file_name}"
    _add_node(
        graph,
        NodeKind.FILE,
        file_qn_full,
        record_path,
        0,
        0,
        source_text=source,
        props={"dialect": "python"},
    )

    # Add nodes
    for node_data in result["nodes"]:
        if node_data["kind"] == "file":
            continue  # Already created
        kind_map = {
            "function": NodeKind.FUNCTION,
            "class": NodeKind.CLASS,
            "method": NodeKind.METHOD,
            "trait": NodeKind.TRAIT,
            "type": NodeKind.TYPE,
            "module": NodeKind.MODULE,
        }
        kind = kind_map.get(node_data["kind"], NodeKind.FUNCTION)
        props = (
            _parse_properties(node_data.get("properties", "{}"))
            if isinstance(node_data.get("properties"), str)
            else node_data.get("properties", {})
        )
        _add_node(
            graph,
            kind,
            node_data["qualified_name"],
            record_path,
            max(0, node_data["line_start"] - 1),
            max(0, node_data["line_end"] - 1),
            source_text=node_data.get("source_text", ""),
            props=props,
        )

    # Add edges
    for edge_data in result["edges"]:
        src_qn = edge_data["src_qn"]
        dst_qn = edge_data["dst_qn"]
        src_id = graph.find_by_qname(src_qn)
        dst_id = graph.find_by_qname(dst_qn)
        if src_id is None or dst_id is None:
            continue

        kind = edge_data["kind"]
        if kind == "defines":
            graph.add_edge(src_id, dst_id, Edge.extracted(EdgeKind.DEFINES))
        elif kind == "imports":
            graph.add_edge(src_id, dst_id, Edge.extracted(EdgeKind.IMPORTS))
        elif kind == "inherits":
            graph.add_edge(src_id, dst_id, Edge.extracted(EdgeKind.INHERITS))
        elif kind == "member_of":
            graph.add_edge(src_id, dst_id, Edge.extracted(EdgeKind.MEMBER_OF))

    # Add call placeholders
    for call_data in result["calls"]:
        caller_id = graph.find_by_qname(call_data["caller_qn"])
        if caller_id is None:
            continue
        callee_qn = call_data["callee_qn"]
        receiver = call_data.get("receiver")

        callee_id = _add_node(
            graph,
            NodeKind.FUNCTION,
            callee_qn,
            Path(""),
            0,
            0,
            props={"dialect": "python", "role": "call_placeholder"},
        )
        edge = Edge.ambiguous(EdgeKind.CALLS)
        if receiver:
            edge = edge.with_property("call_receiver", receiver)
        graph.add_edge(caller_id, callee_id, edge)


def extract_python_files(
    paths: list[Path],
    graph: Graph,
    *,
    file_qns: dict[Path, str] | None = None,
    source_paths: dict[Path, Path] | None = None,
) -> None:
    """Parse multiple Python source files in parallel and emit nodes/edges into the graph.

    Uses the Rust-accelerated parallel extraction if available, falling back to
    sequential Python extraction otherwise.

    Args:
        paths: List of paths to Python source files.
        graph: The graph to add nodes and edges to.
        file_qns: Optional mapping of path to qualified name for each file.
        source_paths: Optional mapping of path to source path for source_uri.
    """
    if not HAS_RUST or len(paths) < 2:
        # Fallback: use single-file extraction for 0-1 files or no Rust
        for path in paths:
            extract_python_file(
                path,
                graph,
                file_qn=file_qns.get(path) if file_qns else None,
                source_path=source_paths.get(path) if source_paths else None,
            )
        return

    # Prepare data for parallel extraction
    file_data = []
    for path in paths:
        with open(path, "rb") as f:
            raw = f.read()
        file_data.append({
            "path": str(path),
            "source": raw,
        })

    # Call the Rust parallel extraction
    result = _rust_extract_parallel(file_data)

    # Create file nodes and extract data
    file_nodes = {}
    for node_data in result["nodes"]:
        if node_data["kind"] == "file":
            path = Path(node_data["source_uri"])
            file_qn = node_data["qualified_name"]
            file_nodes[path] = file_qn
            _add_node(
                graph,
                NodeKind.FILE,
                file_qn,
                path,
                0,
                0,
                source_text=node_data.get("source_text", ""),
                props={"dialect": "python"},
            )
        else:
            kind_map = {
                "function": NodeKind.FUNCTION,
                "class": NodeKind.CLASS,
                "method": NodeKind.METHOD,
                "trait": NodeKind.TRAIT,
                "type": NodeKind.TYPE,
                "module": NodeKind.MODULE,
            }
            kind = kind_map.get(node_data["kind"], NodeKind.FUNCTION)
            props = node_data.get("properties", {})
            path = Path(node_data["source_uri"])
            _add_node(
                graph,
                kind,
                node_data["qualified_name"],
                path,
                max(0, node_data["line_start"] - 1),
                max(0, node_data["line_end"] - 1),
                source_text=node_data.get("source_text", ""),
                props=props,
            )

    # Add edges
    for edge_data in result["edges"]:
        src_qn = edge_data["src_qn"]
        dst_qn = edge_data["dst_qn"]
        src_id = graph.find_by_qname(src_qn)
        dst_id = graph.find_by_qname(dst_qn)
        if src_id is None or dst_id is None:
            continue

        kind = edge_data["kind"]
        if kind == "defines":
            graph.add_edge(src_id, dst_id, Edge.extracted(EdgeKind.DEFINES))
        elif kind == "imports":
            graph.add_edge(src_id, dst_id, Edge.extracted(EdgeKind.IMPORTS))
        elif kind == "inherits":
            graph.add_edge(src_id, dst_id, Edge.extracted(EdgeKind.INHERITS))
        elif kind == "member_of":
            graph.add_edge(src_id, dst_id, Edge.extracted(EdgeKind.MEMBER_OF))

    # Add call placeholders
    for call_data in result["calls"]:
        caller_id = graph.find_by_qname(call_data["caller_qn"])
        if caller_id is None:
            continue
        callee_qn = call_data["callee_qn"]
        receiver = call_data.get("receiver")

        callee_id = _add_node(
            graph,
            NodeKind.FUNCTION,
            callee_qn,
            Path(""),
            0,
            0,
            props={"dialect": "python", "role": "call_placeholder"},
        )
        edge = Edge.ambiguous(EdgeKind.CALLS)
        if receiver:
            edge = edge.with_property("call_receiver", receiver)
        graph.add_edge(caller_id, callee_id, edge)


def extract_data_flow(
    graph: Graph,
    function_id: NodeId,
    source_text: str,
    params: list[str] | None = None,
    source_path: str = "",
) -> int:
    """Extract data flow edges from a function/method body using Rust.

    Uses the Rust-accelerated data flow extraction if available,
    falling back to the Python implementation otherwise.

    Args:
        graph: The graph to add data flow edges to.
        function_id: The ID of the function/method node.
        source_text: The source code of the function body.
        params: Optional list of parameter names. If None, auto-extracted.

    Returns:
        Number of data flow edges added.
    """
    if not HAS_RUST:
        # Fallback to Python implementation
        from graphician.extraction.data_flow import (
            extract_data_flow as _python_extract,
        )
        return _python_extract(graph, function_id, source_text, params, source_path=source_path)

    # Call the Rust data flow extraction
    result = _rust_extract_data_flow(
        source_text=source_text,
        function_id=function_id.value,
        params=params,
    )

    # Create variable nodes and emit DataFlow edges
    count = 0
    for edge_data in result.get("edges", []):
        flow_type = edge_data.get("flow_type", "")
        source_kind = edge_data.get("source_kind", "")
        source_name = edge_data.get("source_name", "")
        target_name = edge_data.get("target_name", "")
        source_text_line = edge_data.get("source_text", "")

        # Determine source and target QNs
        if flow_type in ("param_flow", "return_flow"):
            # Return edges may originate directly from the function when the
            # returned expression is not one of its parameters.
            source_id = (
                function_id
                if source_kind == "function"
                else _ensure_param_node(graph, function_id, source_name, source_text_line)
            )
            if flow_type == "return_flow":
                target_id = _ensure_return_node(graph, function_id, source_text_line)
            else:
                target_id = _ensure_var_node(
                    graph, function_id, target_name, source_text_line
                )
        elif flow_type == "assignment":
            # For assignment flow, function -> var
            source_id = function_id
            target_id = _ensure_var_node(
                graph, function_id, target_name, source_text_line
            )
        else:
            continue

        if (
            source_id is not None
            and target_id is not None
            and not any(
                neighbor == target_id and edge.kind == EdgeKind.DATA_FLOW
                for neighbor, edge in graph.out_neighbors(source_id)
            )
        ):
            graph.add_edge(source_id, target_id, Edge.extracted(EdgeKind.DATA_FLOW))
            count += 1

    return count


def _ensure_param_node(
    graph: Graph,
    function_id: NodeId,
    param_name: str,
    source_line: str,
) -> NodeId | None:
    """Ensure a parameter variable node exists."""
    qn = f"param::{function_id.value}::{param_name}"
    existing = graph.find_by_qname(qn)
    if existing is not None:
        return existing
    node = Node(
        kind=NodeKind.VARIABLE,
        name=param_name,
        qualified_name=qn,
    ).with_property("function_id", str(function_id.value))
    if source_line:
        node = node.with_source_text(source_line)
    graph.add_node(node)
    return graph.find_by_qname(qn)


def _ensure_return_node(
    graph: Graph,
    function_id: NodeId,
    source_line: str,
) -> NodeId | None:
    """Ensure a return value node exists."""
    qn = f"return::{function_id.value}"
    existing = graph.find_by_qname(qn)
    if existing is not None:
        return existing
    node = Node(
        kind=NodeKind.VARIABLE,
        name="return_value",
        qualified_name=qn,
    ).with_property("function_id", str(function_id.value))
    if source_line:
        node = node.with_source_text(source_line)
    graph.add_node(node)
    return graph.find_by_qname(qn)


def _ensure_var_node(
    graph: Graph,
    function_id: NodeId,
    var_name: str,
    source_line: str,
) -> NodeId | None:
    """Ensure a variable node exists."""
    qn = f"var::{function_id.value}::{var_name}"
    existing = graph.find_by_qname(qn)
    if existing is not None:
        return existing
    node = Node(
        kind=NodeKind.VARIABLE,
        name=var_name,
        qualified_name=qn,
    ).with_property("function_id", str(function_id.value))
    if source_line:
        node = node.with_source_text(source_line)
    graph.add_node(node)
    return graph.find_by_qname(qn)
