"""Central language registry — bundled defaults + user TOML overlay.

Built-in language definitions are embedded as a constant. On first access
the registry loads the bundled defaults, then merges any user overlay
found at ``.graphician/languages.toml`` relative to the current working
directory.

TOML schema:
.. code-block:: toml

   [languages.rust]
   grammar = "rust"
   extractor = "rust"
   extensions = [".rs"]
   function_node_types = ["function_item", "closure_expression"]
   class_node_types = ["struct_item", "enum_item"]
   import_node_types = ["use_declaration"]
   call_node_types = ["call_expression"]
   comment = "Rust"

Usage::

   from graphician.extraction.languages.language_registry import (
       registry,
       get_language,
       get_language_by_path,
   )

   lang = get_language("python")
   is_python = get_language_by_path(Path("src/main.py")) is not None
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_RELATIVE_PATH = ".graphician/languages.toml"
MAX_CUSTOM_LANGUAGES = 20

# ── Bundled defaults ────────────────────────────────────────────────

BUNDLED_LANGUAGES_TOML = """
[languages.rust]
grammar = "rust"
extractor = "rust"
extensions = [".rs"]
function_node_types = ["function_item", "closure_expression"]
class_node_types = ["struct_item", "enum_item", "trait_item"]
import_node_types = ["use_declaration"]
call_node_types = ["call_expression"]
comment = "Rust"

[languages.python]
grammar = "python"
extractor = "python"
extensions = [".py", ".pyi"]
function_node_types = ["function_definition", "async_function_definition"]
class_node_types = ["class_definition"]
import_node_types = ["import_statement", "import_from_statement"]
call_node_types = ["call"]
comment = "#"

[languages.cpp]
grammar = "cpp"
extractor = "cpp"
extensions = [".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"]
function_node_types = ["function_definition"]
class_node_types = ["class_specifier", "struct_specifier"]
import_node_types = ["include"]
call_node_types = ["call_expression"]
comment = "C++"

[languages.java]
grammar = "java"
extractor = "java"
extensions = [".java"]
function_node_types = ["method_declaration"]
class_node_types = ["class_declaration", "interface_declaration", "enum_declaration"]
import_node_types = ["import_declaration"]
call_node_types = ["method_invocation"]
comment = "Java"

[languages.typescript]
grammar = "typescript"
extractor = "typescript"
extensions = [".ts"]
function_node_types = ["function_declaration", "method_declaration", "arrow_function"]
class_node_types = ["class_declaration"]
import_node_types = ["import_statement"]
call_node_types = ["call_expression"]
comment = "TypeScript"

[languages.javascript]
grammar = "javascript"
extractor = "javascript"
extensions = [".js", ".mjs", ".cjs"]
function_node_types = ["function_declaration", "method_definition", "arrow_function"]
class_node_types = ["class"]
import_node_types = ["import_statement"]
call_node_types = ["call_expression"]
comment = "JavaScript"

[languages.tsx]
grammar = "typescript"
extractor = "typescript"
extensions = [".tsx", ".jsx"]
function_node_types = ["function_declaration", "method_declaration", "arrow_function"]
class_node_types = ["class_declaration"]
import_node_types = ["import_statement"]
call_node_types = ["call_expression", "jsx_self_closing_element", "jsx_element"]
comment = "TSX"
"""


# ── Types ───────────────────────────────────────────────────────────


class LanguageDef:
    """Language definition with node types and extensions."""

    __slots__ = (
        "call_node_types",
        "class_node_types",
        "comment",
        "extensions",
        "extractor",
        "function_node_types",
        "grammar",
        "import_node_types",
        "name",
    )

    def __init__(
        self,
        name: str,
        grammar: str = "",
        extractor: str = "generic",
        extensions: list[str] | None = None,
        function_node_types: list[str] | None = None,
        class_node_types: list[str] | None = None,
        import_node_types: list[str] | None = None,
        call_node_types: list[str] | None = None,
        comment: str = "",
    ) -> None:
        self.name = name
        self.grammar = grammar or name
        self.extractor = extractor
        self.extensions = sorted(set(normalize_extensions(extensions or [])))
        self.function_node_types = function_node_types or []
        self.class_node_types = class_node_types or []
        self.import_node_types = import_node_types or []
        self.call_node_types = call_node_types or []
        self.comment = comment

    def matches_ext(self, path: Path) -> bool:
        """Check if a file path matches this language's extensions."""
        ext = path.suffix.lstrip(".")
        if not ext:
            return False
        ext = ext.lower()
        # Special case: .jsx is also tsx, .tsx is tsx
        if ext == "jsx":
            return "jsx" in self.extensions or "tsx" in self.extensions
        if ext == "tsx":
            return "tsx" in self.extensions
        return ext in self.extensions


class LanguageRegistry:
    """Thread-safe language registry loaded once on first access."""

    _instance = None  # type: LanguageRegistry | None

    def __init__(self) -> None:
        self.languages: dict[str, LanguageDef] = {}

    @classmethod
    def _get_global(cls) -> LanguageRegistry:
        """Get the global registry singleton (lazy init)."""
        if cls._instance is None:
            cls._instance = cls.load()
        return cls._instance

    @classmethod
    def load(cls) -> LanguageRegistry:
        """Load registry from bundled defaults + user overlay."""
        registry = LanguageRegistry()
        all_langs = load_builtins()

        # Merge user overlay
        repo_root = Path.cwd()
        config_path = repo_root / CONFIG_RELATIVE_PATH

        if config_path.exists():
            try:
                content = config_path.read_text(encoding="utf-8")
                config = tomllib.loads(content)
                user_langs = config.get("languages", {})
                if len(user_langs) > MAX_CUSTOM_LANGUAGES:
                    logger.warning(
                        "config has %d entries, using top %d",
                        len(user_langs),
                        MAX_CUSTOM_LANGUAGES,
                    )
                sorted_entries = sorted(user_langs.items())
                for name, entry in sorted_entries[:MAX_CUSTOM_LANGUAGES]:
                    _merge_entry(all_langs, name, entry)
            except Exception as e:  # noqa: BLE001 -- user-supplied TOML must not crash the registry
                logger.warning("failed to parse %s: %s", config_path, e)

        registry.languages = all_langs
        return registry

    def get(self, name: str) -> LanguageDef | None:
        """Look up a language by name (case-insensitive)."""
        return self.languages.get(name) or self.languages.get(name.lower())

    def get_by_path(self, path: Path) -> LanguageDef | None:
        """Look up a language by file path (matches extension)."""
        for lang in self.languages.values():
            if lang.matches_ext(path):
                return lang
        return None

    def names(self) -> list[str]:
        """Get all registered language names."""
        return list(self.languages.keys())

    def all(self) -> list[LanguageDef]:
        """Get all language definitions."""
        return list(self.languages.values())


# ── Module-level helpers ────────────────────────────────────────────


def registry() -> LanguageRegistry:
    """Get the global registry."""
    return LanguageRegistry._get_global()


def get_language(name: str) -> LanguageDef | None:
    """Look up a language by name."""
    return registry().get(name)


def get_language_by_path(path: Path) -> LanguageDef | None:
    """Look up a language by file path."""
    return registry().get_by_path(path)


# ── Internal ────────────────────────────────────────────────────────


def load_builtins() -> dict[str, LanguageDef]:
    """Parse bundled TOML into a dict of LanguageDef."""
    config = tomllib.loads(BUNDLED_LANGUAGES_TOML)
    result: dict[str, LanguageDef] = {}
    for name, entry in config.get("languages", {}).items():
        name_lower = name.lower()
        result[name_lower] = LanguageDef(
            name=name_lower,
            grammar=entry.get("grammar", name_lower),
            extractor=entry.get("extractor", "generic"),
            extensions=entry.get("extensions"),
            function_node_types=entry.get("function_node_types"),
            class_node_types=entry.get("class_node_types"),
            import_node_types=entry.get("import_node_types"),
            call_node_types=entry.get("call_node_types"),
            comment=entry.get("comment", ""),
        )
    return result


def _merge_entry(
    all_langs: dict[str, LanguageDef],
    name: str,
    entry: dict[str, Any],
) -> None:
    """Merge a user config entry into the registry."""
    name_lower = name.lower()

    if name_lower in all_langs:
        # Update existing language
        lang_def = all_langs[name_lower]
        if entry.get("extensions"):
            lang_def.extensions = sorted(set(normalize_extensions(entry["extensions"])))
        if entry.get("grammar"):
            lang_def.grammar = entry["grammar"]
        if entry.get("extractor"):
            lang_def.extractor = entry["extractor"]
        if entry.get("function_node_types"):
            lang_def.function_node_types = entry["function_node_types"]
        if entry.get("class_node_types"):
            lang_def.class_node_types = entry["class_node_types"]
        if entry.get("import_node_types"):
            lang_def.import_node_types = entry["import_node_types"]
        if entry.get("call_node_types"):
            lang_def.call_node_types = entry["call_node_types"]
    else:
        # New custom language — validate
        grammar = entry.get("grammar", "")
        extensions = entry.get("extensions")
        has_types = any(
            entry.get(k)
            for k in (
                "function_node_types",
                "class_node_types",
                "import_node_types",
                "call_node_types",
            )
        )

        if not grammar:
            logger.warning("custom language '%s' has empty grammar, skipping", name)
            return
        if not extensions or not extensions[0]:
            logger.warning("custom language '%s' has no extensions, skipping", name)
            return
        if not has_types:
            logger.warning("custom language '%s' has no node types, skipping", name)
            return

        all_langs[name_lower] = LanguageDef(
            name=name_lower,
            grammar=grammar,
            extractor=entry.get("extractor", "generic"),
            extensions=extensions,
            function_node_types=entry.get("function_node_types", []),
            class_node_types=entry.get("class_node_types", []),
            import_node_types=entry.get("import_node_types", []),
            call_node_types=entry.get("call_node_types", []),
            comment=entry.get("comment", ""),
        )


def normalize_extension(ext: str) -> str:
    """Normalize a file extension (strip dot, lowercase)."""
    return ext.strip(".").lower()


def normalize_extensions(extensions: list[str]) -> list[str]:
    """Normalize a list of extensions (dedup, sort)."""
    normalized = sorted(set(normalize_extension(e) for e in extensions if e.strip(".")))
    return normalized
