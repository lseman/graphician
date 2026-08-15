from __future__ import annotations

from pathlib import Path

import pytest

from ariadne_py.core import EdgeKind, Graph, Node, NodeKind
from ariadne_py.extraction.documents import concept_registry
from ariadne_py.extraction.documents.document_utils import (
    normalize_for_match,
    resolve_mentions,
    resolve_symbol,
    slugify,
    strip_file_suffix,
    tokenize_code,
)


def _kinds(graph: Graph) -> set[NodeKind]:
    return {node.kind for _, node in graph.nodes()}


@pytest.mark.parametrize("suffix", [".md", ".markdown", ".html", ".htm", ".svg"])
def test_concept_registry_recognizes_supported_suffixes_case_insensitively(suffix: str) -> None:
    assert concept_registry.is_supported(Path(f"guide{suffix.upper()}"))
    assert concept_registry.get_by_path(Path(f"guide{suffix}")) is not None


def test_concept_registry_rejects_unknown_format(tmp_path: Path) -> None:
    path = tmp_path / "guide.txt"
    assert not concept_registry.is_supported(path)
    assert "Unsupported" in concept_registry.extract_concept(path, Graph())["error"]


def test_markdown_contract_extracts_structure_and_symbol_mentions(tmp_path: Path) -> None:
    graph = Graph()
    graph.add_node(Node.new(NodeKind.FUNCTION, "app::handle_request"))
    path = tmp_path / "guide.md"
    path.write_text(
        "# API Guide\n\n## Handler\n\nUse `handle_request()` from "
        "[the implementation](app.py#handle_request).\n\n```python\nhandle_request()\n```\n"
    )

    concept_registry.extract_concept(path, graph)
    resolved = concept_registry.resolve_all_mentions(graph)

    assert {NodeKind.DOCUMENT, NodeKind.SECTION} <= _kinds(graph)
    assert resolved >= 0
    assert EdgeKind.DEFINES in {edge.kind for *_, edge in graph.edges()}


def test_html_contract_extracts_headings_code_links_and_metadata(tmp_path: Path) -> None:
    graph = Graph()
    graph.add_node(Node.new(NodeKind.FUNCTION, "app::handle_request"))
    path = tmp_path / "guide.html"
    path.write_text(
        "<html><head><meta name='description' content='API docs'></head>"
        "<body><main><h1>API Guide</h1><section><h2>Handler</h2>"
        "<code>handle_request()</code><a href='app.py#handle_request'>implementation</a>"
        "<table><tr><td>request payload</td></tr></table></section></main></body></html>"
    )

    concept_registry.extract_concept(path, graph)

    assert {NodeKind.DOCUMENT, NodeKind.SECTION} <= _kinds(graph)
    assert any(node.properties.get("heading") == "API Guide" for _, node in graph.nodes())


def test_svg_contract_extracts_labels_and_mentions(tmp_path: Path) -> None:
    graph = Graph()
    graph.add_node(Node.new(NodeKind.FUNCTION, "app::handle_request"))
    path = tmp_path / "flow.svg"
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><title>Request Flow</title>'
        '<text x="1" y="2">handle_request</text><text><tspan>Database</tspan></text></svg>'
    )

    metadata = concept_registry.extract_concept(path, graph)

    assert NodeKind.DIAGRAM in _kinds(graph)
    assert metadata["concepts"] >= 2


def test_document_utilities_normalize_tokenize_and_resolve_symbols() -> None:
    graph = Graph()
    exact = graph.add_node(Node.new(NodeKind.FUNCTION, "pkg::handle_request"))
    graph.add_node(Node.new(NodeKind.CLASS, "pkg::RequestHandler"))
    assert resolve_symbol(graph, "pkg::handle_request") == exact
    assert resolve_symbol(graph, "handle_request()") == exact
    assert resolve_symbol(graph, "missing") is None
    assert strip_file_suffix("src/app.py") == "src/app"
    assert "service.handle_request" in tokenize_code("await service.handle_request(payload)")
    assert normalize_for_match("Request-Handler_v2") == "requesthandlerv2"
    assert slugify("  API & Request Handler! ") == "api-request-handler"
    assert resolve_mentions(graph) >= 0
