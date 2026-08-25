"""Custom language support via TOML-based language definitions.

Delegates to the central ``language_registry`` for language resolution
and file extraction. Custom languages defined in ``.graphician/languages.toml``
get a lightweight generic tree-sitter walker.

Mirrors the Rust ``custom_lang.rs`` module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...core.edge import Edge, EdgeKind
from ...core.node import Node, NodeKind
from .language_registry import LanguageDef


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
        from tree_sitter import Language, Parser
    except ImportError:
        return

    source = path.read_text(encoding="utf-8", errors="replace")
    raw_language = _resolve_language(lang_def.grammar)
    if raw_language is None:
        return
    language = raw_language if isinstance(raw_language, Language) else Language(raw_language)
    parser = Parser(language)
    tree = parser.parse(source.encode("utf-8"))

    file_uri = str(path)
    file_qn = f"file::{file_uri}"
    file_is_test = _is_test_file_path(path)
    file_id = graph.add_node(
        Node.new(NodeKind.FILE, file_qn)
        .with_source(file_uri, 1, max(len(source.splitlines()), 1))
        .with_source_text(source)
        .with_property("language", lang_def.name)
    )

    function_types = set(lang_def.function_node_types)
    class_types = set(lang_def.class_node_types)
    import_types = set(lang_def.import_node_types)
    call_types = set(lang_def.call_node_types)

    def walk(ast_node: Any, parent_id: Any, scope_qn: str) -> None:
        for child in ast_node.named_children:
            next_parent = parent_id
            next_scope = scope_qn
            if child.type in function_types or child.type in class_types:
                name = _definition_name(child, source)
                if name:
                    kind = NodeKind.FUNCTION if child.type in function_types else NodeKind.CLASS
                    qname = f"{scope_qn}::{name}"
                    properties = {"language": lang_def.name}
                    if file_is_test:
                        properties["is_test"] = True
                    definition = Node(
                        kind=kind,
                        name=name,
                        qualified_name=qname,
                        source_uri=file_uri,
                        line_start=child.start_point.row + 1,
                        line_end=child.end_point.row + 1,
                        properties=properties,
                        source_text=_node_text(child, source),
                    )
                    next_parent = graph.add_node(definition)
                    graph.add_edge(parent_id, next_parent, Edge.extracted(EdgeKind.DEFINES))
                    next_scope = qname
            elif child.type in import_types:
                import_name = _import_name(child, source)
                if import_name:
                    module_id = graph.add_node(Node.new(NodeKind.MODULE, f"module::{import_name}"))
                    graph.add_edge(parent_id, module_id, Edge.extracted(EdgeKind.IMPORTS))
            elif child.type in call_types:
                call_name = _call_name(child, source)
                if call_name:
                    target = graph.add_node(Node.new(NodeKind.FUNCTION, f"call::{call_name}"))
                    graph.add_edge(parent_id, target, Edge.extracted(EdgeKind.CALLS))
            walk(child, next_parent, next_scope)

    walk(tree.root_node, file_id, file_qn)


def _node_text(node: Any, source: str) -> str:
    return source.encode("utf-8")[node.start_byte : node.end_byte].decode(
        "utf-8", errors="replace"
    )


def _definition_name(node: Any, source: str) -> str | None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        name_node = next(
            (child for child in node.named_children if child.type in {"identifier", "type_identifier"}),
            None,
        )
    return _node_text(name_node, source) if name_node is not None else None


def _import_name(node: Any, source: str) -> str | None:
    candidate = (
        node.child_by_field_name("source")
        or node.child_by_field_name("path")
        or node.child_by_field_name("module_name")
    )
    text = _node_text(candidate or node, source).strip()
    if text.startswith(("import ", "use ")):
        text = text.split(maxsplit=1)[1]
    return text.strip(" ;'\"") or None


def _call_name(node: Any, source: str) -> str | None:
    function = node.child_by_field_name("function")
    if function is None and node.named_children:
        function = node.named_children[0]
    if function is None:
        return None
    return _node_text(function, source).strip()


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
            from tree_sitter_typescript import language_tsx, language_typescript
            return language_tsx() if name == "tsx" else language_typescript()
        elif name == "javascript":
            from tree_sitter_javascript import language
            return language()
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
