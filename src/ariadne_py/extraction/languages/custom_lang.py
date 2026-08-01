"""Custom language support via TOML-based language definitions.

Delegates to the central ``language_registry`` for language resolution
and file extraction. Custom languages defined in ``.ariadne/languages.toml``
get a lightweight generic tree-sitter walker.

Mirrors the Rust ``custom_lang.rs`` module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .language_registry import get_language, get_language_by_path, registry, LanguageDef


def extract_file(
    path: Path,
    graph,
    lang_def: LanguageDef,
) -> None:
    """Extract a single file for a known language definition.

    Specialized extractors are selected by the registry's ``extractor``
    field. Languages without one use the generic walker.

    Args:
        path: File path to extract.
        graph: Mutable graph to add nodes/edges to.
        lang_def: Language definition from the registry.
    """
    extractor = lang_def.extractor

    if extractor == "python":
        _extract_python(path, graph, lang_def)
    elif extractor == "typescript" or extractor == "tsx":
        _extract_typescript(path, graph, lang_def)
    elif extractor == "javascript":
        _extract_javascript(path, graph, lang_def)
    elif extractor == "rust":
        _extract_rust(path, graph, lang_def)
    elif extractor == "cpp":
        _extract_cpp(path, graph, lang_def)
    elif extractor == "java":
        _extract_java(path, graph, lang_def)
    else:
        _extract_custom(path, graph, lang_def)


def _extract_custom(path: Path, graph, lang_def: LanguageDef) -> None:
    """Extract a file using the generic custom-language walker.

    Uses tree-sitter queries from the language definition to extract
    functions, classes, imports, and call placeholders.

    Args:
        path: File path to extract.
        graph: Mutable graph.
        lang_def: Language definition with node type queries.
    """
    try:
        from tree_sitter import Parser, Query, QueryCursor

        source = path.read_text(encoding="utf-8", errors="replace")
        parser = Parser()

        ts_lang = _resolve_language(lang_def.grammar)
        if ts_lang is None:
            return

        parser.set_language(ts_lang)
        tree = parser.parse(source.encode("utf-8"))

        file_uri = str(path)
        file_qn = f"file::{file_uri}"
        file_is_test = _is_test_file_path(path)
        file_id = graph.add_node(
            type("Node", (), {
                "kind": "File",
                "name": "file",
                "qualified_name": file_qn,
                "source_uri": file_uri,
                "line_start": 0,
                "line_end": len(source.splitlines()),
                "properties": {},
                "source_text": "",
            })()
        )

        cursor = QueryCursor()

        # Extract functions
        for node_type in lang_def.function_node_types:
            query_str = f"({node_type} name: (identifier) @name) @def"
            try:
                query = Query(ts_lang, query_str)
                matches = cursor.matches(query, tree.root_node, source.encode("utf-8"))
                for match in matches:
                    name = None
                    start = 0
                    end = 0
                    for cap_name, cap_node in match.captures:
                        if cap_name == "name":
                            name = cap_node.text.decode("utf-8")
                        elif cap_name == "def":
                            start = cap_node.start_point[0]
                            end = cap_node.end_point[0]
                    if name:
                        qn = f"{file_qn}::{name}"
                        node = type("Node", (), {
                            "kind": "Function",
                            "name": name,
                            "qualified_name": qn,
                            "source_uri": file_uri,
                            "line_start": start,
                            "line_end": end,
                            "properties": {"is_test": file_is_test} if file_is_test else {},
                        })()
                        nid = graph.add_node(node)
                        graph.add_edge(file_id, nid, "extracted", "Defines")
            except Exception:
                continue

        # Extract classes
        for node_type in lang_def.class_node_types:
            query_str = f"({node_type} name: (type_identifier) @name) @def"
            try:
                query = Query(ts_lang, query_str)
                matches = cursor.matches(query, tree.root_node, source.encode("utf-8"))
                for match in matches:
                    name = None
                    start = 0
                    end = 0
                    for cap_name, cap_node in match.captures:
                        if cap_name == "name":
                            name = cap_node.text.decode("utf-8")
                        elif cap_name == "def":
                            start = cap_node.start_point[0]
                            end = cap_node.end_point[0]
                    if name:
                        qn = f"{file_qn}::{name}"
                        node = type("Node", (), {
                            "kind": "Class",
                            "name": name,
                            "qualified_name": qn,
                            "source_uri": file_uri,
                            "line_start": start,
                            "line_end": end,
                            "properties": {},
                        })()
                        nid = graph.add_node(node)
                        graph.add_edge(file_id, nid, "extracted", "Defines")
            except Exception:
                continue

    except ImportError:
        # tree_sitter not available, skip
        pass


def _resolve_language(name: str) -> Any:
    """Resolve a grammar name to a tree-sitter language.

    Args:
        name: Grammar name (e.g. 'python', 'rust', 'typescript').

    Returns:
        tree-sitter Language object, or None.
    """
    try:
        if name == "python":
            from tree_sitter_python import language
            return language()
        elif name == "rust":
            from tree_sitter_rust import language
            return language()
        elif name == "typescript" or name == "tsx":
            from tree_sitter_typescript import language_tsx
            return language_tsx()
        elif name == "javascript":
            from tree_sitter_typescript import language_typescript
            return language_typescript()
        elif name == "cpp" or name == "c" or name == "c++":
            from tree_sitter_cpp import language
            return language()
        elif name == "java":
            from tree_sitter_java import language
            return language()
    except ImportError:
        pass
    return None


def _is_test_file_path(path: Path) -> bool:
    """Check if a path looks like a test file."""
    name = path.name.lower()
    return any(
        x in name
        for x in ("test", "spec", "mock", "fixture", "_test", "_spec")
    )


def _extract_python(path: Path, graph, lang_def: LanguageDef) -> None:
    """Python-specific extraction."""
    from .parsers.python import extract_file as python_extract
    python_extract(path, graph)


def _extract_typescript(path: Path, graph, lang_def: LanguageDef) -> None:
    """TypeScript/TSX extraction."""
    from .parsers.typescript import extract_file as ts_extract
    ts_extract(path, graph)


def _extract_javascript(path: Path, graph, lang_def: LanguageDef) -> None:
    """JavaScript extraction."""
    from .parsers.javascript import extract_file as js_extract
    js_extract(path, graph)


def _extract_rust(path: Path, graph, lang_def: LanguageDef) -> None:
    """Rust extraction."""
    from .parsers.rust import extract_file as rust_extract
    rust_extract(path, graph)


def _extract_cpp(path: Path, graph, lang_def: LanguageDef) -> None:
    """C++ extraction."""
    from .parsers.cpp import extract_file as cpp_extract
    cpp_extract(path, graph)


def _extract_java(path: Path, graph, lang_def: LanguageDef) -> None:
    """Java extraction."""
    from .parsers.java import extract_file as java_extract
    java_extract(path, graph)
