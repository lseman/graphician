"""Markdown extraction using ``pulldown-cmark``.

Parses markdown into a proper AST and emits:
- ``Document`` nodes for each markdown file
- ``Section`` nodes for headings (including nested heading levels)
- ``Concept`` nodes for meaningful inline elements
- ``Mentions`` edges from sections/concepts to code symbols

Supports: ATX/setext headings, reference/inline links, fenced code blocks,
tables, lists, bold/italic, footnotes, blockquotes, images.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    import pulldown_cmark as pc
    from pulldown_cmark import Event, Options, Parser, Tag, TagEnd
except ImportError:
    pc = None
    Options = None
    Parser = None
    Event = None
    Tag = None
    TagEnd = None

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
    """Extract a markdown file into the graph.

    Creates Document and Section nodes. Extracts symbol mentions from
    code blocks, inline code, links, and footnotes.
    """
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    file_uri = str(path)
    file_qn = f"doc::{file_uri}"
    file_id = graph.add_node(
        Node.new(NodeKind.DOCUMENT, file_qn)
        .with_source(file_uri, 0, len(source.splitlines()))
    )

    refs = _collect_reference_definitions(source)

    if pc is None:
        _extract_markdown_fallback(source, file_qn, file_id, graph)
        return

    opts = Options.ALL
    parser = Parser.new_ext(source, opts)

    section_stack: list[tuple[NodeId, int]] = []
    in_code_block = False
    current_section_id: NodeId = file_id
    heading_text_buffer: list[str] = []
    in_heading = False
    heading_counter = 0

    for event in parser:
        if not isinstance(event, Event):
            continue

        tag = getattr(event, "tag", None)
        if tag is None:
            continue

        # ── Start events ────────────────────────────────────────────
        if isinstance(tag, Tag):
            if isinstance(tag, Tag.Heading):
                level = _level_to_int(tag.level)
                in_heading = True
                heading_text_buffer.clear()
                while section_stack and section_stack[-1][1] >= level:
                    section_stack.pop()
                parent = section_stack[-1][0] if section_stack else file_id
                qn = f"{file_qn}::section-{heading_counter}"
                section_id = graph.add_node(
                    Node.new(NodeKind.SECTION, qn)
                    .with_source(str(file_qn), 0, 0)
                )
                heading_counter += 1
                graph.add_edge(parent, section_id, Edge.extracted(EdgeKind.DEFINES))
                section_stack.append((section_id, level))
                current_section_id = section_id

            elif isinstance(tag, Tag.CodeBlock):
                in_code_block = True

            elif isinstance(tag, Tag.Link):
                link_type = getattr(tag, "link_type", None)
                dest_url = getattr(tag, "dest_url", "")
                if link_type in (
                    pc.LinkType.REFERENCE,
                    pc.LinkType.COLLABSED,
                    pc.LinkType.SHORTCUT,
                ):
                    ref_id = getattr(tag, "id", "")
                    target = refs.get(ref_id, (dest_url, None))[0]
                    _extract_symbol_from_url(target, graph, current_section_id, file_qn)
                elif link_type in (pc.LinkType.INLINE, pc.LinkType.WIKILINK):
                    _extract_symbol_from_url(dest_url, graph, current_section_id, file_qn)
                elif link_type == pc.LinkType.EMAIL:
                    local = dest_url.replace("mailto:", "")
                    _extract_symbol_from_url(local, graph, current_section_id, file_qn)

            elif isinstance(tag, Tag.FootnoteDefinition):
                name = getattr(tag, "name", "")
                mention(graph, current_section_id, name, 0.85)

        # ── End events ──────────────────────────────────────────────
        elif isinstance(tag, TagEnd):
            if isinstance(tag, TagEnd.Heading):
                if heading_text_buffer:
                    heading_text = "".join(heading_text_buffer)
                    sl = slugify(heading_text)
                    new_qn = f"{file_qn}::{sl}" if sl else f"{file_qn}::section"
                    eff_id = graph.rename_node(current_section_id, new_qn, sl)
                    if eff_id != current_section_id:
                        if section_stack:
                            top = section_stack[-1]
                            if top[0] == current_section_id:
                                section_stack[-1] = (eff_id, top[1])
                        current_section_id = eff_id
                in_heading = False

            elif isinstance(tag, TagEnd.CodeBlock):
                in_code_block = False

        # ── Text / code events ──────────────────────────────────────
        elif isinstance(event, Event.Text):
            text = str(event)
            if in_heading:
                heading_text_buffer.append(text)
            elif in_code_block:
                for token in tokenize_code(text):
                    mention(graph, current_section_id, token, 0.80)
            else:
                chunks = text.split("`")
                for i, chunk in enumerate(chunks):
                    if i % 2 == 1:
                        mention(graph, current_section_id, chunk, 0.85)

        elif isinstance(event, Event.Code):
            mention(graph, current_section_id, str(event), 0.85)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _level_to_int(level: Any) -> int:
    """Convert a HeadingLevel to int."""
    if isinstance(level, int):
        return level
    mapping = {"H1": 1, "H2": 2, "H3": 3, "H4": 4, "H5": 5, "H6": 6}
    return mapping.get(str(level), 1)


def _collect_reference_definitions(source: str) -> dict[str, tuple[str, str | None]]:
    """Collect reference-style link definitions from markdown source."""
    refs: dict[str, tuple[str, str | None]] = {}
    for line in source.splitlines():
        line = line.lstrip()
        if not line.startswith("["):
            continue
        rest = line[1:]
        idx = rest.find("]: ")
        if idx < 0:
            continue
        ref_id = rest[:idx]
        after = rest[idx + 3:]
        url, _ = _parse_url_title(after)
        if url:
            refs[ref_id] = (url, None)
    return refs


def _parse_url_title(s: str) -> tuple[str, str | None]:
    """Parse URL with optional title from a link definition."""
    s = s.lstrip()
    if s.startswith("<"):
        end = s.find(">")
        if end < 0:
            return ("", None)
        return (s[1:end], None)
    for i, c in enumerate(s):
        if c.isspace():
            return (s[:i], None)
    return (s, None)


def _extract_symbol_from_url(url: str, graph, section_id: NodeId, _file_qn: str) -> None:
    """Extract symbol references from a URL."""
    candidate = url.lstrip("#") if url.startswith("#") else url
    candidate = candidate.rstrip("/")
    candidate = candidate.split("/")[-1] if "/" in candidate else candidate
    candidate = strip_file_suffix(candidate)
    if candidate and len(candidate) >= 2:
        mention(graph, section_id, candidate, 0.70)


def _extract_markdown_fallback(source: str, file_qn: str, file_id: NodeId, graph) -> None:
    """Standard-library fallback for headings and code-symbol mentions."""
    current = file_id
    section_counter = 0
    for line_number, line in enumerate(source.splitlines(), 1):
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*$", line)
        if heading:
            title = heading.group(2).strip()
            current = graph.add_node(
                Node.new(NodeKind.SECTION, f"{file_qn}::{slugify(title) or section_counter}")
                .with_source(file_qn, line_number, line_number)
                .with_property("heading", title)
            )
            graph.add_edge(file_id, current, Edge.extracted(EdgeKind.DEFINES))
            section_counter += 1
        for token in re.findall(r"`([^`\n]+)`", line):
            for symbol in tokenize_code(token):
                mention(graph, current, symbol, 0.85)


def resolve_mentions(graph) -> int:
    """Resolve pending mentions across all sections.

    Idempotent post-pass. Returns number of edges added.
    """
    from .document_utils import resolve_mentions as _resolve_mentions
    return _resolve_mentions(graph)
