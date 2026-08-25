"""SVG diagram extraction.

Registers SVG files as Diagram nodes and emits Concept nodes for
non-empty <text> elements. Concept → symbol cross-linking is
delegated to the markdown mention resolver for consistency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...core.edge import Edge, EdgeKind
from ...core.graph import Graph
from ...core.node import Node, NodeKind


def extract_svg(path: str | Path, graph: Graph) -> dict[str, Any]:
    """Extract an SVG file into diagram/concept nodes.

    Creates:
    - One Diagram node for the SVG file
    - One Concept node per non-empty <text> element
    - Infered Illustrates edges from Diagram → Concept

    Returns summary of extraction results.
    """
    path = Path(path)
    try:
        source = path.read_text()
    except (OSError, UnicodeDecodeError) as e:
        return {"error": str(e), "diagrams": 0, "concepts": 0}

    file_uri = str(path)
    qn = f"diagram::{file_uri}"
    diag_id = graph.add_node(
        Node(kind=NodeKind.DIAGRAM, name=path.name, qualified_name=qn)
        .with_source(file_uri, 0, 0)
    )

    labels = extract_text_labels(source)
    concepts = 0

    for label in labels:
        concept_qn = f"concept::{label}"
        concept_id = graph.add_node(
            Node(kind=NodeKind.CONCEPT, name=label, qualified_name=concept_qn)
        )
        graph.add_edge(
            diag_id, concept_id,
            Edge.inferred(EdgeKind.ILLUSTRATES, 0.7),
        )
        concepts += 1

    return {"diagrams": 1, "concepts": concepts, "file": file_uri}


def extract_text_labels(svg: str) -> list[str]:
    """Extract text labels from SVG content.

    Parses <text>...</text> elements, strips nested HTML-like tags,
    and returns non-empty cleaned labels.
    """
    out: list[str] = []
    bytes_data = svg.encode("utf-8")
    i = 0

    while i < len(bytes_data):
        # Find next "<text"
        start = _find_subslice(bytes_data[i:], b"<text")
        if start is None:
            break

        open_start = i + start
        # Find end of opening tag
        after_open = _find_subslice(bytes_data[open_start:], b">")
        if after_open is None:
            break
        after_open = open_start + after_open + 1

        # Find closing </text>
        close = _find_subslice(bytes_data[after_open:], b"</text>")
        if close is None:
            break

        try:
            text = bytes_data[after_open:after_open + close].decode("utf-8")
        except UnicodeDecodeError:
            i = after_open + close + len(b"</text>")
            continue

        clean = _strip_inner_tags(text).strip()
        if clean:
            out.append(clean)

        i = after_open + close + len(b"</text>")

    return out


def _find_subslice(haystack: bytes, needle: bytes) -> int | None:
    """Find needle in haystack, return offset or None."""
    if not needle:
        return None
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return None


def _strip_inner_tags(s: str) -> str:
    """Strip nested <tags> from SVG text content."""
    out: list[str] = []
    in_tag = False

    for ch in s:
        if ch == "<":
            in_tag = True
        elif ch == ">":
            in_tag = False
        elif not in_tag:
            out.append(ch)

    return "".join(out)
