from __future__ import annotations

from pathlib import Path

import pytest

from graphician.core import EdgeKind, Graph, NodeKind
from graphician.extraction.languages import custom_lang
from graphician.extraction.languages.language_registry import (
    LanguageDef,
    LanguageRegistry,
    _merge_entry,
    get_language,
    get_language_by_path,
    load_builtins,
    normalize_extension,
    normalize_extensions,
)


def test_bundled_registry_normalizes_extensions_and_matches_paths() -> None:
    builtins = load_builtins()
    assert {"python", "rust", "typescript", "javascript", "java", "cpp", "tsx"} <= set(builtins)
    assert builtins["python"].matches_ext(Path("model.PY"))
    assert builtins["tsx"].matches_ext(Path("view.jsx"))
    assert not builtins["rust"].matches_ext(Path("README"))
    assert normalize_extension(".PY") == "py"
    assert normalize_extensions([".PY", "py", ".pyi", ""]) == ["py", "pyi"]


def test_registry_loads_user_overlay_and_valid_custom_language(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / ".ariadne"
    config_dir.mkdir()
    (config_dir / "languages.toml").write_text(
        """[languages.python]
extensions = [".py", ".pyw"]

[languages.demo]
grammar = "python"
extensions = [".demo"]
function_node_types = ["function_definition"]
comment = "demo"
"""
    )
    monkeypatch.chdir(tmp_path)
    LanguageRegistry._instance = None

    loaded = LanguageRegistry.load()
    assert loaded.get("PYTHON").matches_ext(Path("app.pyw"))
    assert loaded.get_by_path(Path("app.demo")).name == "demo"
    assert "demo" in loaded.names()
    assert len(loaded.all()) == len(loaded.names())
    assert get_language("demo").name == "demo"
    assert get_language_by_path(Path("x.demo")).name == "demo"


def test_invalid_custom_registry_entries_are_skipped() -> None:
    languages = load_builtins()
    before = set(languages)
    _merge_entry(languages, "no_grammar", {"extensions": ["x"], "function_node_types": ["f"]})
    _merge_entry(languages, "no_ext", {"grammar": "python", "function_node_types": ["f"]})
    _merge_entry(languages, "no_types", {"grammar": "python", "extensions": ["x"]})
    assert set(languages) == before


@pytest.mark.parametrize(
    ("filename", "extractor", "expected_kind"),
    [
        ("demo.py", "python", NodeKind.FUNCTION),
        ("demo.js", "javascript", NodeKind.FUNCTION),
        ("demo.ts", "typescript", NodeKind.FUNCTION),
        ("demo.rs", "rust", NodeKind.FUNCTION),
        ("demo.cpp", "cpp", NodeKind.FUNCTION),
        ("Demo.java", "java", NodeKind.CLASS),
    ],
)
def test_custom_language_dispatches_to_specialized_extractors(
    tmp_path: Path, filename: str, extractor: str, expected_kind: NodeKind
) -> None:
    sources = {
        "python": "def run():\n    return 1\n",
        "javascript": "function run() { return 1; }",
        "typescript": "function run(): number { return 1; }",
        "rust": "fn run() -> i32 { 1 }",
        "cpp": "int run() { return 1; }",
        "java": "class Demo { int run() { return 1; } }",
    }
    path = tmp_path / filename
    path.write_text(sources[extractor])
    graph = Graph()
    definition = LanguageDef("demo", grammar=extractor, extractor=extractor, extensions=[path.suffix])

    custom_lang.extract_file(path, graph, definition)

    assert expected_kind in {node.kind for _, node in graph.nodes()}


def test_custom_language_helpers_cover_unknown_grammar_and_test_paths() -> None:
    assert custom_lang._resolve_language("unknown") is None
    assert custom_lang._resolve_language("python") is not None
    assert custom_lang._is_test_file_path(Path("feature_spec.demo"))
    assert not custom_lang._is_test_file_path(Path("feature.demo"))


def test_generic_custom_language_extracts_complete_graph_contract(tmp_path: Path) -> None:
    path = tmp_path / "feature_test.demo"
    path.write_text(
        "import os\n"
        "class Service:\n"
        "    def run(self):\n"
        "        helper()\n"
        "def helper():\n"
        "    pass\n"
    )
    definition = LanguageDef(
        "demo",
        grammar="python",
        extractor="generic",
        extensions=["demo"],
        function_node_types=["function_definition"],
        class_node_types=["class_definition"],
        import_node_types=["import_statement"],
        call_node_types=["call"],
    )
    graph = Graph()

    custom_lang.extract_file(path, graph, definition)

    file_qn = f"file::{path}"
    expected = {
        file_qn: NodeKind.FILE,
        f"{file_qn}::Service": NodeKind.CLASS,
        f"{file_qn}::Service::run": NodeKind.FUNCTION,
        f"{file_qn}::helper": NodeKind.FUNCTION,
        "module::os": NodeKind.MODULE,
        "call::helper": NodeKind.FUNCTION,
    }
    for qname, kind in expected.items():
        node_id = graph.find_by_qname(qname)
        assert node_id is not None
        assert graph.node(node_id).kind is kind

    definitions = [
        node for _, node in graph.nodes() if node.kind in {NodeKind.CLASS, NodeKind.FUNCTION}
        and not node.qualified_name.startswith("call::")
    ]
    assert all(node.properties == {"language": "demo", "is_test": True} for node in definitions)
    edge_contract = {
        (graph.node(src).qualified_name, edge.kind, graph.node(dst).qualified_name)
        for _, src, dst, edge in graph.edges()
    }
    assert (file_qn, EdgeKind.IMPORTS, "module::os") in edge_contract
    assert (file_qn, EdgeKind.DEFINES, f"{file_qn}::Service") in edge_contract
    assert (
        f"{file_qn}::Service",
        EdgeKind.DEFINES,
        f"{file_qn}::Service::run",
    ) in edge_contract
    assert (f"{file_qn}::Service::run", EdgeKind.CALLS, "call::helper") in edge_contract
