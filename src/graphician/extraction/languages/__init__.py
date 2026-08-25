"""Language registry and spec.

Maps file extensions to tree-sitter parsers and extraction configs.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tree_sitter


class Language(enum.StrEnum):
    """Supported programming languages."""
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    RUST = "rust"
    JAVA = "java"
    CPP = "cpp"
    GO = "go"


@dataclass
class LanguageSpec:
    """Configuration for extracting a single language."""
    name: Language
    extensions: list[str]
    parser_factory: Callable[[], tree_sitter.Parser]
    # Patterns to extract from AST
    extract_file: bool = True
    extract_functions: bool = True
    extract_classes: bool = True
    extract_imports: bool = True
    extract_calls: bool = True
    extract_inheritance: bool = True


class LanguageRegistry:
    """Registry mapping extensions → language specs."""

    def __init__(self) -> None:
        self._specs: dict[str, LanguageSpec] = {}
        self._lang_to_spec: dict[Language, LanguageSpec] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register all built-in language specs."""
        import tree_sitter_cpp as tscpp
        import tree_sitter_go as tsgo
        import tree_sitter_java as tsjava
        import tree_sitter_javascript as tsjavascript
        import tree_sitter_python as tspython
        import tree_sitter_rust as tsrust
        import tree_sitter_typescript as tstypescript

        specs = [
            LanguageSpec(
                name=Language.PYTHON,
                extensions=[".py"],
                parser_factory=lambda: self._make_parser(tspython.language()),
                extract_file=True,
                extract_functions=True,
                extract_classes=True,
                extract_imports=True,
                extract_calls=True,
                extract_inheritance=True,
            ),
            LanguageSpec(
                name=Language.TYPESCRIPT,
                extensions=[".ts", ".tsx"],
                parser_factory=lambda: self._make_parser(tstypescript.language_typescript()),
                extract_file=True,
                extract_functions=True,
                extract_classes=True,
                extract_imports=True,
                extract_calls=True,
                extract_inheritance=True,
            ),
            LanguageSpec(
                name=Language.JAVASCRIPT,
                extensions=[".js", ".mjs", ".cjs"],
                parser_factory=lambda: self._make_parser(tsjavascript.language()),
                extract_file=True,
                extract_functions=True,
                extract_classes=True,
                extract_imports=True,
                extract_calls=True,
                extract_inheritance=True,
            ),
            LanguageSpec(
                name=Language.RUST,
                extensions=[".rs"],
                parser_factory=lambda: self._make_parser(tsrust.language()),
                extract_file=True,
                extract_functions=True,
                extract_classes=True,
                extract_imports=True,
                extract_calls=True,
                extract_inheritance=True,
            ),
            LanguageSpec(
                name=Language.JAVA,
                extensions=[".java"],
                parser_factory=lambda: self._make_parser(tsjava.language()),
                extract_file=True,
                extract_functions=True,
                extract_classes=True,
                extract_imports=True,
                extract_calls=True,
                extract_inheritance=True,
            ),
            LanguageSpec(
                name=Language.CPP,
                extensions=[".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"],
                parser_factory=lambda: self._make_parser(tscpp.language()),
                extract_file=True,
                extract_functions=True,
                extract_classes=True,
                extract_imports=True,
                extract_calls=True,
                extract_inheritance=True,
            ),
            LanguageSpec(
                name=Language.GO,
                extensions=[".go"],
                parser_factory=lambda: self._make_parser(tsgo.language()),
                extract_file=True,
                extract_functions=True,
                extract_classes=True,
                extract_imports=True,
                extract_calls=True,
                extract_inheritance=True,
            ),
        ]

        for spec in specs:
            self._specs[spec.name] = spec
            for ext in spec.extensions:
                self._specs[ext] = spec
            self._lang_to_spec[spec.name] = spec

    @staticmethod
    def _make_parser(lang_ptr: object) -> tree_sitter.Parser:
        """Create a tree-sitter Parser from a language pointer."""
        from tree_sitter import Language as TreeSitterLanguage
        from tree_sitter import Parser

        language = (
            lang_ptr
            if isinstance(lang_ptr, TreeSitterLanguage)
            else TreeSitterLanguage(lang_ptr)
        )
        return Parser(language)

    def get_spec(self, path: Path) -> LanguageSpec | None:
        """Get language spec for a file path, or None if unsupported."""
        return self._specs.get(path.suffix)

    def get_spec_by_name(self, name: Language) -> LanguageSpec:
        return self._lang_to_spec[name]

    def supported_extensions(self) -> set[str]:
        return set(self._specs.keys()) - {lang for lang in Language}
