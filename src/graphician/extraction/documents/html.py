"""HTML extraction using ``html5lib``.

Parses HTML documents and emits:
- ``Document`` nodes for each HTML file
- ``Section`` nodes for semantic headings (h1-h6)
- ``Concept`` nodes for meaningful text (link text, heading text, table cells)
- ``Mentions`` edges from sections/concepts to code symbols

Supports:
- Semantic HTML structure (header, nav, main, article, section, aside, footer)
- Headings (h1-h6)
- Links (a) — extract symbol references from href and text
- Code (code, pre) — extract symbol references from content
- Tables (table, th, td) — extract cell content
- Lists (ul, ol, li)
- Inline elements: strong, em, abbr, cite, samp
- Meta tags for page description and keywords
- Script tags with inline code extraction

HTML blocks without semantic structure fall back to paragraph-level
Concept extraction.
"""

from __future__ import annotations

from html.parser import HTMLParser as StdlibHTMLParser
from pathlib import Path
from typing import Any

try:
    import html5lib
    from html5lib import HTMLParser
except ImportError:
    html5lib = None
    HTMLParser = None

from ...core.edge import Edge, EdgeKind
from ...core.id import NodeId
from ...core.node import Node, NodeKind
from .document_utils import (
    mention,
    slugify,
    strip_file_suffix,
    tokenize_code,
)


def extract_file(path: str | Path, graph) -> None:
    """Extract an HTML file into the graph.

    Creates a Document node, walks the DOM tree to build Sections from
    headings, extracts symbol mentions from text/code/links, and records
    page metadata.
    """
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    file_uri = str(path)
    file_qn = f"doc::{file_uri}"

    file_id = graph.add_node(
        Node.new(NodeKind.DOCUMENT, file_qn)
        .with_source(file_uri, 0, len(source.splitlines()))
    )

    if html5lib is None:
        fallback = _FallbackHTMLExtractor(graph, file_qn, file_id)
        fallback.feed(source)
        return

    parser = HTMLParser()
    dom = parser.parse(source)

    heading_counter = 0
    section_stack: list[tuple[NodeId, int]] = []
    current_section_id: NodeId = file_id

    _extract_dom_tree(
        dom, file_qn, file_id,
        heading_counter, section_stack, current_section_id, graph,
    )

    # Extract page metadata
    _extract_meta(dom, graph, file_id, file_qn)


class _FallbackHTMLExtractor(StdlibHTMLParser):
    """Extract headings and code text when html5lib is unavailable."""

    def __init__(self, graph, file_qn: str, file_id: NodeId) -> None:
        super().__init__()
        self.graph = graph
        self.file_qn = file_qn
        self.file_id = file_id
        self.current = file_id
        self.capture: str | None = None
        self.buffer: list[str] = []
        self.counter = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6", "code", "pre"}:
            self.capture = tag
            self.buffer.clear()

    def handle_data(self, data: str) -> None:
        if self.capture:
            self.buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != self.capture:
            return
        text = " ".join(self.buffer).strip()
        if tag.startswith("h") and text:
            self.current = self.graph.add_node(
                Node.new(NodeKind.SECTION, f"{self.file_qn}::{slugify(text) or self.counter}")
                .with_property("heading", text)
            )
            self.graph.add_edge(self.file_id, self.current, Edge.extracted(EdgeKind.DEFINES))
            self.counter += 1
        elif text:
            for symbol in tokenize_code(text):
                mention(self.graph, self.current, symbol, 0.85)
        self.capture = None
        self.buffer.clear()


def _extract_dom_tree(
    handle: Any,
    file_qn: str,
    parent_id: NodeId,
    heading_counter: int,
    section_stack: list[tuple[NodeId, int]],
    current_section_id: NodeId,
    graph,
) -> None:
    """Walk the DOM tree, building sections from headings."""
    if handle is None:
        return

    children = getattr(handle, "children", [])
    if children is None:
        children = []

    for child in children:
        node_data = getattr(child, "data", None)
        if node_data is None:
            continue

        node_type = type(node_data).__name__

        # ── Text nodes ──────────────────────────────────────────────
        if node_type == "Text":
            contents = getattr(node_data, "value", "")
            if not contents or not contents.strip() or len(contents) <= 1:
                continue
            # Extract symbols from inline code spans (backticks).
            tokens = contents.split("`")
            for i, chunk in enumerate(tokens):
                if i % 2 == 1:
                    # Inside backticks — treat as code.
                    for token in tokenize_code(chunk):
                        mention(graph, current_section_id, token, 0.85)
                elif chunk.strip():
                    # Plain text.
                    for token in tokenize_code(chunk):
                        if len(token) >= 2:
                            mention(graph, current_section_id, token, 0.75)

        # ── Element nodes ───────────────────────────────────────────
        elif node_type == "Element":
            tag_name = getattr(node_data, "name", "").lower()
            attrs = _get_attrs(child)

            # Headings create Section nodes.
            if tag_name.startswith("h") and len(tag_name) == 2:
                try:
                    level = int(tag_name[1])
                except ValueError:
                    level = 0
                if 1 <= level <= 6:
                    # Pop section stack entries that are deeper or equal.
                    while section_stack and section_stack[-1][1] >= level:
                        section_stack.pop()
                    parent = section_stack[-1][0] if section_stack else parent_id

                    # Collect heading text from children.
                    heading_text = _collect_text(child, only_inline=True)
                    if not heading_text:
                        continue

                    slug = slugify(heading_text)
                    qn = (
                        f"{file_qn}::{slug}" if slug
                        else f"{file_qn}::section-{heading_counter}"
                    )
                    heading_counter += 1

                    section_id = graph.add_node(
                        Node.new(NodeKind.SECTION, qn)
                        .with_source(file_qn, 0, 0)
                    )
                    graph.add_edge(parent, section_id, Edge.extracted(EdgeKind.DEFINES))
                    section_stack.append((section_id, level))
                    mention(graph, section_id, heading_text, 0.9)

                    # Recurse into children of this section.
                    _extract_dom_tree(
                        child, file_qn, parent_id,
                        heading_counter, section_stack, section_id, graph,
                    )
                    continue

            # Extract code from <code> and <pre> blocks.
            elif tag_name in ("code", "pre"):
                text = _collect_text(child, only_inline=False)
                for token in tokenize_code(text):
                    mention(graph, current_section_id, token, 0.80)

            # Extract text from semantic inline elements.
            elif tag_name in ("abbr", "cite", "samp", "strong", "em"):
                text = _collect_text(child, only_inline=True)
                if text and len(text) > 1:
                    mention(graph, current_section_id, text, 0.8)

            # Links: extract symbol from link text and href.
            elif tag_name == "a":
                link_text = _collect_text(child, only_inline=True)
                if link_text:
                    mention(graph, current_section_id, link_text, 0.7)
                href = attrs.get("href", "")
                if href:
                    _extract_symbol_from_url(href, graph, current_section_id, file_qn)

            # Table cells: extract cell text as concepts.
            elif tag_name in ("td", "th"):
                text = _collect_text(child, only_inline=True)
                for token in tokenize_code(text):
                    mention(graph, current_section_id, token, 0.65)

            # Script: extract inline JS code.
            elif tag_name == "script":
                script_type = attrs.get("type", "")
                if "json" not in script_type:
                    text = _collect_text(child, only_inline=False)
                    for token in tokenize_code(text):
                        mention(graph, current_section_id, token, 0.7)

            # Recurse into other elements.
            else:
                _extract_dom_tree(
                    child, file_qn, parent_id,
                    heading_counter, section_stack, current_section_id, graph,
                )


def _collect_text(handle: Any, only_inline: bool) -> str:
    """Collect text content from a node's children."""
    parts: list[str] = []
    children = getattr(handle, "children", []) or []
    for child in children:
        node_data = getattr(child, "data", None)
        if node_data is None:
            continue
        node_type = type(node_data).__name__

        if node_type == "Text":
            text = getattr(node_data, "value", "")
            if text.strip():
                parts.append(text)
        elif node_type == "Element":
            tag_name = getattr(node_data, "name", "").lower()
            if only_inline:
                # Skip block elements when only inline is requested.
                block_tags = {
                    "div", "section", "article", "header", "footer", "nav",
                    "main", "aside", "ul", "ol", "li", "table", "tr",
                    "thead", "tbody", "form", "fieldset", "details", "summary",
                }
                if tag_name in block_tags:
                    continue
            parts.append(_collect_text(child, only_inline))
    return "".join(parts)


def _get_attrs(handle: Any) -> dict[str, str]:
    """Extract attributes from an element node."""
    attrs: dict[str, str] = {}
    attr_map = getattr(handle, "attributes", None)
    if attr_map is None:
        return attrs
    for key, value in attr_map.items():
        attrs[key] = value or ""
    return attrs


def _extract_symbol_from_url(url: str, graph, section_id: NodeId, _file_qn: str) -> None:
    """Extract symbol references from a URL."""
    candidate = url.lstrip("#") if url.startswith("#") else url
    candidate = candidate.rstrip("/")
    candidate = candidate.split("/")[-1] if "/" in candidate else candidate
    candidate = strip_file_suffix(candidate)
    if candidate and len(candidate) >= 2:
        mention(graph, section_id, candidate, 0.70)


def _extract_meta(handle: Any, graph, file_id: NodeId, _file_qn: str) -> None:
    """Extract meta tags for page metadata (description, keywords, author)."""
    stack: list[Any] = [handle]
    while stack:
        current = stack.pop()
        children = getattr(current, "children", []) or []
        for child in children:
            node_data = getattr(child, "data", None)
            if node_data is None:
                continue
            if type(node_data).__name__ != "Element":
                stack.append(child)
                continue
            tag_name = getattr(node_data, "name", "").lower()
            if tag_name != "meta":
                stack.append(child)
                continue

            attrs = _get_attrs(child)
            content = attrs.get("content", "")
            name = attrs.get("name", attrs.get("property", ""))

            if not content:
                continue

            name_lower = name.lower()
            if name_lower == "description" and content:
                node = graph.node_mut(file_id)
                if node is not None:
                    node.properties["description"] = content
            elif name_lower == "keywords":
                for kw in content.split(","):
                    kw = kw.strip()
                    if kw and len(kw) >= 2:
                        mention(graph, file_id, kw, 0.6)
            elif name_lower == "author" and content:
                mention(graph, file_id, content, 0.7)
