"""Language-specific extraction wrappers.

Wraps the Rust-accelerated extractors for Rust, TypeScript, JavaScript, Java, and C++.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from graphician.core.edge import Edge, EdgeKind
from graphician.core.graph import Graph
from graphician.core.id import NodeId
from graphician.core.node import Node, NodeKind

from . import HAS_RUST

if HAS_RUST:
    from . import extract_cpp_file as _rust_extract_cpp
    from . import extract_go_file as _rust_extract_go
    from . import extract_java_file as _rust_extract_java
    from . import extract_javascript_file as _rust_extract_js
    from . import extract_rust_file as _rust_extract_rust
    from . import extract_typescript_file as _rust_extract_ts


def _add_node(
    graph: Graph,
    kind: NodeKind,
    qn: str,
    path: Path,
    line_start: int,
    line_end: int,
    source_text: str | None = None,
    props: dict[str, Any] | None = None,
) -> NodeId | None:
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


def _parse_properties(props: Any) -> dict[str, Any]:
    """Parse properties from various formats into a dict."""
    if isinstance(props, dict):
        return props
    if isinstance(props, str):
        try:
            import json
            return json.loads(props)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _extract_file(
    path: Path,
    graph: Graph,
    dialect: str,
    rust_fn,
    *,
    file_qn: str | None = None,
    source_path: Path | None = None,
) -> None:
    """Common extraction logic for all language extractors."""
    if rust_fn is None:
        return

    with open(path, "rb") as f:
        raw = f.read()

    source = raw.decode("utf-8", errors="replace")
    record_path = source_path if source_path is not None else path
    file_name = record_path.stem
    file_qn_full = file_qn or f"file::{file_name}"

    # Call the Rust extraction
    result = rust_fn(raw, file_path=str(record_path), file_qn=file_qn or "")

    # Create file node
    _add_node(
        graph,
        NodeKind.FILE,
        file_qn_full,
        record_path,
        0,
        0,
        source_text=source,
        props={"dialect": dialect},
    )

    # Add nodes
    for node_data in result["nodes"]:
        if node_data["kind"] == "file":
            continue
        kind_map = {
            "function": NodeKind.FUNCTION,
            "class": NodeKind.CLASS,
            "method": NodeKind.METHOD,
            "trait": NodeKind.TRAIT,
            "type": NodeKind.TYPE,
            "variable": NodeKind.VARIABLE,
            "module": NodeKind.MODULE,
        }
        kind = kind_map.get(node_data["kind"], NodeKind.FUNCTION)
        props = _parse_properties(node_data.get("properties", {}))
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

        edge_kind = None
        if edge_data["kind"] == "defines":
            edge_kind = EdgeKind.DEFINES
        elif edge_data["kind"] == "imports":
            edge_kind = EdgeKind.IMPORTS
        elif edge_data["kind"] == "inherits":
            edge_kind = EdgeKind.INHERITS
        elif edge_data["kind"] == "implements":
            edge_kind = EdgeKind.IMPLEMENTS
        elif edge_data["kind"] == "member_of":
            edge_kind = EdgeKind.MEMBER_OF
        elif edge_data["kind"] == "data_flow":
            edge_kind = EdgeKind.DATA_FLOW

        if edge_kind:
            graph.add_edge(src_id, dst_id, Edge.extracted(edge_kind))

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
            props={"dialect": dialect},
        )
        edge = Edge.ambiguous(EdgeKind.CALLS)
        if receiver:
            edge = edge.with_property("call_receiver", receiver)
        graph.add_edge(caller_id, callee_id, edge)


def extract_rust_file(
    path: Path,
    graph: Graph,
    *,
    file_qn: str | None = None,
    source_path: Path | None = None,
) -> None:
    """Parse a Rust source file using the Rust-accelerated extractor."""
    _extract_file(path, graph, "rust", _rust_extract_rust, file_qn=file_qn, source_path=source_path)


def extract_typescript_file(
    path: Path,
    graph: Graph,
    *,
    file_qn: str | None = None,
    source_path: Path | None = None,
) -> None:
    """Parse a TypeScript source file using the Rust-accelerated extractor."""
    _extract_file(
        path, graph, "typescript", _rust_extract_ts, file_qn=file_qn, source_path=source_path
    )


def extract_javascript_file(
    path: Path,
    graph: Graph,
    *,
    file_qn: str | None = None,
    source_path: Path | None = None,
) -> None:
    """Parse a JavaScript source file using the Rust-accelerated extractor."""
    _extract_file(
        path, graph, "javascript", _rust_extract_js, file_qn=file_qn, source_path=source_path
    )


def extract_java_file(
    path: Path,
    graph: Graph,
    *,
    file_qn: str | None = None,
    source_path: Path | None = None,
) -> None:
    """Parse a Java source file using the Rust-accelerated extractor."""
    _extract_file(path, graph, "java", _rust_extract_java, file_qn=file_qn, source_path=source_path)


def extract_cpp_file(
    path: Path,
    graph: Graph,
    *,
    file_qn: str | None = None,
    source_path: Path | None = None,
) -> None:
    """Parse a C/C++ source file using the Rust-accelerated extractor."""
    _extract_file(path, graph, "cpp", _rust_extract_cpp, file_qn=file_qn, source_path=source_path)


def extract_go_file(
    path: Path,
    graph: Graph,
    *,
    file_qn: str | None = None,
    source_path: Path | None = None,
) -> None:
    """Parse a Go source file using the Rust-accelerated extractor."""
    _extract_file(path, graph, "go", _rust_extract_go, file_qn=file_qn, source_path=source_path)
