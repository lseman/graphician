"""Extraction pipeline.

Orchestrates file discovery, tree-sitter parsing, symbol extraction,
and edge building to populate the graph.

Respects .gitignore, .ariadneignore, and common generated-directory
conventions. File discovery uses walkdir-like traversal with ignore support.
"""

from __future__ import annotations

import hashlib
import fnmatch
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

from ..core.edge import Edge, EdgeKind
from ..core.node import NodeKind
from ..core.graph import Graph
from ..core.id import EdgeId, NodeId
from ..core.node import Node
from .languages import Language, LanguageRegistry
from .data_flow import extract_data_flow
from .call_resolution import resolve_call_placeholders
from .type_resolution import resolve_type_placeholders
from .patterns.framework_patterns import detect_patterns
from .flows import compute_flows, FlowOptions
from .documents import extract_html, extract_markdown, resolve_mentions
from .documents.svg import extract_svg
from .languages.tsconfig_resolver import resolve_ts_path_aliases
from .manifests import extract_manifest

logger = logging.getLogger(__name__)


def _is_ignored(path: Path, patterns: list[str]) -> bool:
    """Apply common gitignore-style patterns, including later negation."""
    value = path.as_posix()
    ignored = False
    for raw_pattern in patterns:
        negated = raw_pattern.startswith("!")
        pattern = raw_pattern[1:] if negated else raw_pattern
        pattern = pattern.lstrip("/").rstrip("/")
        if not pattern:
            continue
        matched = fnmatch.fnmatch(value, pattern) or fnmatch.fnmatch(value, f"{pattern}/**")
        if "/" not in pattern:
            matched = matched or any(fnmatch.fnmatch(part, pattern) for part in path.parts)
        if matched:
            ignored = not negated
    return ignored


def _is_test_symbol(path: Path, name: str) -> bool:
    normalized = path.as_posix().lower()
    stem = path.stem.lower()
    return (
        name.lower().startswith("test")
        or stem.startswith("test_")
        or stem.endswith("_test")
        or "/tests/" in f"/{normalized}"
        or "/test/" in f"/{normalized}"
        or ".spec." in normalized
        or ".test." in normalized
    )

# Directories to skip by default
DEFAULT_EXCLUDES = {
    "__pycache__", ".git", ".hg", ".svn", "node_modules", ".venv", "venv",
    "env", ".env", "build", "dist", "target", "out", "bin", "obj",
    ".next", ".nuxt", ".svelte-kit", ".cache", ".tox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".eggs", "*.egg-info",
}
DOCUMENT_SUFFIXES = {".md", ".markdown", ".html", ".htm", ".svg"}
MANIFEST_NAMES = {"package.json", "pyproject.toml", "cargo.toml", "setup.py", "setup.cfg"}


class ExtractionPipeline:
    """Main extraction pipeline.

    Walks a source tree, parses files with tree-sitter, extracts symbols
    and relationships, and populates a Graph.
    """

    def __init__(
        self,
        registry: LanguageRegistry,
        *,
        strict: bool = False,
        workers: int | None = None,
    ) -> None:
        self.registry = registry
        self.strict = strict
        # Tree-sitter releases enough native work for a small thread-level
        # gain, but larger pools lose time to fragment merging and scheduling.
        self.workers = 2 if workers is None else max(1, workers)
        self.graph = Graph()
        self._file_hashes: dict[str, str] = {}  # path → sha256

    # ── File discovery ───────────────────────────────────────────────

    def discover_files(
        self,
        root: Path,
        exclude: set[str] | None = None,
    ) -> list[Path]:
        """Discover source files under root.

        Respects .gitignore and .ariadneignore. Skips generated directories.
        """
        exclude = (exclude or set()) | DEFAULT_EXCLUDES
        root = root.resolve()
        files: list[Path] = []
        ignore_patterns: list[str] = []

        # Load .ariadneignore
        ariadneignore = root / ".ariadneignore"
        if ariadneignore.exists():
            for line in ariadneignore.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    ignore_patterns.append(line)

        gitignore = root / ".gitignore"
        if gitignore.exists():
            for line in gitignore.read_text(errors="replace").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    ignore_patterns.append(line)

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if any(
                any(fnmatch.fnmatch(part, pattern) for pattern in exclude)
                for part in relative.parts[:-1]
            ):
                continue
            if _is_ignored(relative, ignore_patterns):
                continue
            if self.is_supported(path):
                files.append(path)

        return sorted(files)

    def is_supported(self, path: Path) -> bool:
        return (
            path.suffix.lower() in self.registry.supported_extensions() | DOCUMENT_SUFFIXES
            or path.name.lower() in MANIFEST_NAMES
        )

    # ── Main extraction ──────────────────────────────────────────────

    def build(self, root: Path) -> Graph:
        """Full graph build from a project root.

        1. Discover files
        2. Parse each file with tree-sitter
        3. Extract nodes (files, functions, classes, etc.)
        4. Extract edges (defines, calls, imports, etc.)
        5. Resolve calls
        6. Build flows
        """
        files = self.discover_files(root)
        logger.info("Discovered %d source files in %s", len(files), root)

        self._process_files(files, root)

        # Post-extraction: resolve calls, resolve types, detect patterns, build flows
        self._resolve_calls()
        self._resolve_type_placeholders()
        self._detect_patterns()
        self._derive_tested_by_edges()
        self._enrich_data_flow()
        resolve_ts_path_aliases(self.graph, root)
        resolve_mentions(self.graph)
        self._build_flows()

        logger.info(
            "Graph built: %d nodes, %d edges",
            self.graph.node_count(),
            self.graph.edge_count(),
        )
        return self.graph

    def _process_files(self, files: list[Path], root: Path) -> None:
        """Extract independent file fragments and merge them deterministically."""
        if self.workers == 1 or len(files) < 2:
            for file_path in files:
                self._process_file(file_path, root)
            return

        extract = partial(self._extract_fragment, root=root)
        with ThreadPoolExecutor(max_workers=min(self.workers, len(files))) as executor:
            # executor.map preserves discovery order, keeping node IDs and
            # serialized output deterministic while parsing happens in parallel.
            for fragment, file_hashes in executor.map(extract, files):
                self.graph.merge(fragment)
                self._file_hashes.update(file_hashes)

    def _extract_fragment(self, file_path: Path, *, root: Path) -> tuple[Graph, dict[str, str]]:
        fragment = ExtractionPipeline(self.registry, strict=self.strict, workers=1)
        fragment._process_file(file_path, root)
        return fragment.graph, fragment._file_hashes

    def update(
        self,
        root: Path,
        existing: Graph,
        changed: list[str],
        deleted: list[str],
    ) -> Graph:
        """Incrementally replace changed code files in an existing graph.

        Documents and manifests can create shared semantic/package nodes, so
        changes to those inputs deliberately fall back to a full rebuild.
        """
        root = root.resolve()
        touched = [Path(path) for path in changed + deleted]
        if any(
            path.suffix.lower() in DOCUMENT_SUFFIXES
            or path.name.lower() in MANIFEST_NAMES
            for path in touched
        ):
            return self.build(root)

        self.graph = existing
        self._file_hashes = {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.discover_files(root)
        }
        touched_values = {path.as_posix() for path in touched}
        remove_ids = []
        for node_id, node in self.graph.nodes():
            source = (node.source_uri or "").replace("\\", "/")
            qname = node.qualified_name
            if node.kind == NodeKind.FLOW or any(
                source == value
                or source.endswith(f"/{value}")
                or qname.startswith(f"file::{value}")
                for value in touched_values
            ):
                remove_ids.append(node_id)
        for node_id in remove_ids:
            self.graph.remove_node(node_id)

        for relative in changed:
            path = root / relative
            if path.is_file() and self.is_supported(path):
                self._process_file(path, root)
        self._resolve_calls()
        self._resolve_type_placeholders()
        self._detect_patterns()
        self._derive_tested_by_edges()
        self._enrich_data_flow()
        resolve_ts_path_aliases(self.graph, root)
        resolve_mentions(self.graph)
        self._build_flows()
        return self.graph

    def _process_file(self, file_path: Path, root: Path) -> None:
        """Parse a single file and extract symbols."""
        rel_path = file_path.relative_to(root)
        try:
            raw = file_path.read_bytes()
        except (OSError, UnicodeError):
            if self.strict:
                raise
            return
        self._file_hashes[rel_path.as_posix()] = hashlib.sha256(raw).hexdigest()

        suffix = file_path.suffix.lower()
        if file_path.name.lower() in MANIFEST_NAMES:
            extract_manifest(file_path, self.graph)
            return
        if suffix in DOCUMENT_SUFFIXES:
            try:
                if suffix in {".md", ".markdown"}:
                    extract_markdown(file_path, self.graph)
                elif suffix in {".html", ".htm"}:
                    extract_html(file_path, self.graph)
                else:
                    extract_svg(file_path, self.graph)
            except (ImportError, OSError, UnicodeError) as exc:
                if self.strict:
                    raise
                logger.warning("Could not extract document %s: %s", file_path, exc)
            return

        spec = self.registry.get_spec(file_path)
        if spec is None:
            return

        try:
            source = file_path.read_text(
                encoding="utf-8", errors="strict" if self.strict else "replace"
            )
        except OSError:
            if self.strict:
                raise
            return

        # Compute hash for incremental updates
        file_hash = hashlib.sha256(source.encode()).hexdigest()
        file_key = f"file::{rel_path}"

        # Rust needs its specialized extractor: the generic walker does not
        # understand function_item, trait_item, impl_item, use trees, or raw
        # macro token trees. Keep the pipeline's path-qualified file key so
        # equal stems in different directories cannot collide.
        if spec.name is Language.RUST:
            from .languages.parsers.rust import extract_file as extract_rust

            extract_rust(file_path, self.graph, file_qn=file_key)
            return

        # Create file node
        file_node = Node.new(NodeKind.FILE, file_key)
        file_node = file_node.with_source(
            str(rel_path), 0, source.count("\n") + 1
        )
        file_node = file_node.with_source_text(source)
        file_node = file_node.with_property("hash", file_hash)
        file_node = file_node.with_property("language", spec.name)
        self.graph.add_node(file_node)

        # Parse with tree-sitter
        parser = spec.parser_factory()
        tree = parser.parse(source.encode("utf-8"))

        # Extract symbols based on language spec
        self._extract_symbols(
            tree, source, file_path, rel_path, spec, file_key
        )

        # Extract imports
        if spec.extract_imports:
            self._extract_imports(tree, source, file_key, spec)

        # Extract calls
        if spec.extract_calls:
            self._extract_calls(tree, source, file_key, spec)

        # Extract inheritance
        if spec.extract_inheritance:
            self._extract_inheritance(tree, source, file_key, spec)

    # ── Symbol extraction ────────────────────────────────────────────

    def _extract_symbols(
        self,
        tree: Any,
        source: str,
        file_path: Path,
        rel_path: Path,
        spec: Any,
        file_key: str,
    ) -> None:
        """Extract function, class, and type symbols from AST."""
        import tree_sitter

        root = tree.root_node
        self._walk_symbols(root, source, rel_path, spec, file_key, depth=0)

    def _walk_symbols(
        self,
        node: Any,
        source: str,
        rel_path: Path,
        spec: Any,
        file_key: str,
        depth: int,
        parent_qname: str | None = None,
    ) -> None:
        """Recursively walk AST nodes to extract symbols."""
        node_type = node.type
        source_lines = source.split("\n")

        # Determine module path from qualified name
        module_path = self._module_path(rel_path)

        method_types = (
            "method_definition",
            "method_declaration",
            "constructor_declaration",
        )
        if node_type in ("function_definition", "function_declaration", *method_types):
            name_node = None
            for child in node.children:
                if child.type in ("identifier", "name", "property_identifier"):
                    name_node = child
                    break
            if name_node is None and hasattr(node, "field_dict"):
                name_node = node.field_dict.get("name")

            if name_node is not None:
                prefix = parent_qname or f"{file_key}::{module_path}"
                qname = f"{prefix}::{name_node.text.decode()}"
                kind = (
                    NodeKind.METHOD
                    if parent_qname or node_type in method_types
                    else NodeKind.FUNCTION
                )
                func_node = Node.new(kind, qname)
                func_node = func_node.with_source(
                    str(rel_path),
                    node.start_point[0] + 1,
                    node.end_point[0] + 1,
                )
                func_node = func_node.with_source_text(
                    self._extract_body(source, node)
                )
                func_node = func_node.with_property(
                    "line_count", node.end_point[0] - node.start_point[0]
                )
                if _is_test_symbol(rel_path, func_node.name):
                    func_node = func_node.with_property("is_test", True)
                self.graph.add_node(func_node)

                # Add defines edge from file
                self.graph.add_edge(
                    self.graph.find_by_qname(file_key) or NodeId(-1),
                    self.graph.find_by_qname(qname) or NodeId(-1),
                    Edge.extracted(EdgeKind.DEFINES),
                )

                # Add member_of edge if parent is a class
                if parent_qname:
                    self.graph.add_edge(
                        self.graph.find_by_qname(qname) or NodeId(-1),
                        self.graph.find_by_qname(parent_qname) or NodeId(-1),
                        Edge.extracted(EdgeKind.MEMBER_OF),
                    )

        elif node_type in ("class_definition", "class_declaration"):
            name_node = None
            for child in node.children:
                if child.type in ("identifier", "name"):
                    name_node = child
                    break
            if name_node is None and hasattr(node, "field_dict"):
                name_node = node.field_dict.get("name")

            if name_node is not None:
                prefix = parent_qname or f"{file_key}::{module_path}"
                qname = f"{prefix}::{name_node.text.decode()}"
                class_node = Node.new(NodeKind.CLASS, qname)
                class_node = class_node.with_source(
                    str(rel_path),
                    node.start_point[0] + 1,
                    node.end_point[0] + 1,
                )
                class_node = class_node.with_source_text(
                    self._extract_body(source, node)
                )
                self.graph.add_node(class_node)

                self.graph.add_edge(
                    self.graph.find_by_qname(file_key) or NodeId(-1),
                    self.graph.find_by_qname(qname) or NodeId(-1),
                    Edge.extracted(EdgeKind.DEFINES),
                )

                # Walk children for methods
                for child in node.children:
                    if child.type in (
                        "body",
                        "block",
                        "member_declarations",
                        "class_body",
                        "fields",
                    ):
                        self._walk_symbols(
                            child, source, rel_path, spec, file_key,
                            depth + 1, parent_qname=qname,
                        )
                        break

        elif node_type in ("module", "program", "source_file"):
            # Top-level node — walk children
            for child in node.children:
                self._walk_symbols(
                    child, source, rel_path, spec, file_key, depth,
                    parent_qname=parent_qname,
                )
        else:
            for child in node.children:
                if child.is_named:
                    self._walk_symbols(
                        child,
                        source,
                        rel_path,
                        spec,
                        file_key,
                        depth,
                        parent_qname=parent_qname,
                    )

    def _module_path(self, rel_path: Path) -> str:
        """Convert file path to module path.

        Python: file.py → file
        TypeScript: src/foo/bar.ts → src.foo.bar
        """
        parts = list(rel_path.parts)
        # Remove extension from last part
        if parts:
            parts[-1] = parts[-1].rsplit(".", 1)[0]

        # Determine separator based on directory structure
        if parts[0] in ("src", "lib"):
            return ".".join(parts[1:]) if len(parts) > 1 else parts[0]
        return ".".join(parts)

    def _extract_body(self, source: str, node: Any) -> str:
        """Extract the body text of an AST node."""
        start = node.start_byte
        end = node.end_byte
        if start >= end:
            return ""
        return source[start:end]

    # ── Import extraction ────────────────────────────────────────────

    def _extract_imports(
        self,
        tree: Any,
        source: str,
        file_key: str,
        spec: Any,
    ) -> None:
        """Extract import statements and create Import edges."""
        self._walk_imports(tree.root_node, source, file_key, spec)

    def _walk_imports(self, node: Any, source: str, file_key: str, spec: Any) -> None:
        """Recursively find import nodes."""
        node_type = node.type

        # Python: import X, from X import Y
        if node_type in ("import_statement", "import_from_statement"):
            self._parse_import_node(node, file_key)

        # TypeScript/JS: import X from 'Y', require('Y')
        elif node_type in ("import_declaration", "call_expression"):
            if node_type == "call_expression":
                # Check for require()
                func = node.child_by_field_name("function")
                if func and func.text.decode() == b"require":
                    self._parse_require_node(node, file_key)
            else:
                self._parse_ts_import_node(node, file_key)

        for child in node.children:
            self._walk_imports(child, source, file_key, spec)

    def _parse_import_node(self, node: Any, file_key: str) -> None:
        """Parse a Python-style import node."""
        import tree_sitter

        # from X import Y
        if node.type == "import_from_statement":
            module = node.child_by_field_name("module")
            if module:
                module_name = module.text.decode()
                import_qname = f"import::{module_name}"
                import_node = Node.new(NodeKind.MODULE, import_qname)
                self.graph.add_node(import_node)
                self.graph.add_edge(
                    self.graph.find_by_qname(file_key) or NodeId(-1),
                    self.graph.find_by_qname(import_qname) or NodeId(-1),
                    Edge.extracted(EdgeKind.IMPORTS),
                )

        # import X
        elif node.type == "import_statement":
            name_node = node.child_by_field_name("name")
            if name_node:
                module_name = name_node.text.decode()
                import_qname = f"import::{module_name}"
                import_node = Node.new(NodeKind.MODULE, import_qname)
                self.graph.add_node(import_node)
                self.graph.add_edge(
                    self.graph.find_by_qname(file_key) or NodeId(-1),
                    self.graph.find_by_qname(import_qname) or NodeId(-1),
                    Edge.extracted(EdgeKind.IMPORTS),
                )

    def _parse_ts_import_node(self, node: Any, file_key: str) -> None:
        """Parse a TypeScript/JS import declaration."""
        source = node.text.decode()
        # Extract module specifier
        for child in node.children:
            if child.type == "string" or child.type == "module_specifier":
                module = child.text.decode().strip("'\"")
                import_qname = f"import::{module}"
                import_node = Node.new(NodeKind.MODULE, import_qname)
                self.graph.add_node(import_node)
                self.graph.add_edge(
                    self.graph.find_by_qname(file_key) or NodeId(-1),
                    self.graph.find_by_qname(import_qname) or NodeId(-1),
                    Edge.extracted(EdgeKind.IMPORTS),
                )
                break

    def _parse_require_node(self, node: Any, file_key: str) -> None:
        """Parse a require() call."""
        args = node.child_by_field_name("arguments")
        if args:
            for child in args.children:
                if child.type == "string":
                    module = child.text.decode().strip("'\"")
                    import_qname = f"import::{module}"
                    import_node = Node.new(NodeKind.MODULE, import_qname)
                    self.graph.add_node(import_node)
                    self.graph.add_edge(
                        self.graph.find_by_qname(file_key) or NodeId(-1),
                        self.graph.find_by_qname(import_qname) or NodeId(-1),
                        Edge.extracted(EdgeKind.IMPORTS),
                    )
                    break

    # ── Call extraction ──────────────────────────────────────────────

    def _extract_calls(
        self,
        tree: Any,
        source: str,
        file_key: str,
        spec: Any,
    ) -> None:
        """Extract function calls from AST."""
        file_id = self.graph.find_by_qname(file_key)
        if file_id is not None:
            self._walk_calls(tree.root_node, source, file_key, file_id)

    def _walk_calls(
        self,
        node: Any,
        source: str,
        file_key: str,
        caller_id: NodeId,
    ) -> None:
        """Recursively find call expressions."""
        node_type = node.type

        if node_type in (
            "function_definition",
            "function_declaration",
            "method_definition",
            "method_declaration",
            "constructor_declaration",
        ):
            definition_id = self._definition_id(node, file_key)
            if definition_id is not None:
                caller_id = definition_id

        if node_type in ("call_expression", "call"):
            self._parse_call_node(node, caller_id)
        for child in node.children:
            self._walk_calls(child, source, file_key, caller_id)

    def _definition_id(self, node: Any, file_key: str) -> NodeId | None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return None
        name = name_node.text.decode("utf-8", errors="replace")
        line = node.start_point[0] + 1
        for node_id, candidate in self.graph.nodes():
            if (
                candidate.qualified_name.startswith(f"{file_key}::")
                and candidate.name == name
                and candidate.line_start == line
                and candidate.kind in (NodeKind.FUNCTION, NodeKind.METHOD)
            ):
                return node_id
        return None

    def _parse_call_node(self, node: Any, caller_id: NodeId) -> None:
        """Parse a call expression and record the call."""
        func = node.child_by_field_name("function")
        if func is None:
            return

        call_text = func.text.decode("utf-8", errors="replace")
        name = call_text.rsplit(".", 1)[-1]
        from .call_resolution import should_suppress_call_placeholder

        if not name or should_suppress_call_placeholder(name):
            return

        target_qname = f"call::{name}"
        call_node = Node.new(NodeKind.FUNCTION, target_qname)
        target_id = self.graph.add_node(call_node)
        edge = Edge.ambiguous(EdgeKind.CALLS)
        if "." in call_text:
            edge = edge.with_property("call_receiver", call_text.rsplit(".", 1)[0])
        self.graph.add_edge(caller_id, target_id, edge)

    # ── Inheritance extraction ───────────────────────────────────────

    def _extract_inheritance(
        self,
        tree: Any,
        source: str,
        file_key: str,
        spec: Any,
    ) -> None:
        """Extract class inheritance relationships."""
        self._walk_inheritance(tree.root_node, source, file_key)

    def _walk_inheritance(self, node: Any, source: str, file_key: str) -> None:
        """Find inheritance declarations."""
        node_type = node.type

        # Python: class Foo(Bar, Baz)
        if node_type == "class_definition":
            for child in node.children:
                if child.type == "parameters":
                    for base in child.children:
                        if base.type == "identifier" or base.type == "dotted_name":
                            base_name = base.text.decode()
                            base_qname = f"{file_key}::{base_name}"
                            base_node = Node.new(NodeKind.CLASS, base_qname)
                            self.graph.add_node(base_node)
                            self.graph.add_edge(
                                self.graph.find_by_qname(file_key) or NodeId(-1),
                                self.graph.find_by_qname(base_qname) or NodeId(-1),
                                Edge.extracted(EdgeKind.INHERITS),
                            )
                    break

        # TypeScript: class Foo extends Bar
        elif node_type == "class_declaration":
            for child in node.children:
                if child.type == "heritage_clause":
                    for base in child.children:
                        if base.type == "identifier":
                            base_name = base.text.decode()
                            base_qname = f"{file_key}::{base_name}"
                            base_node = Node.new(NodeKind.CLASS, base_qname)
                            self.graph.add_node(base_node)
                            self.graph.add_edge(
                                self.graph.find_by_qname(file_key) or NodeId(-1),
                                self.graph.find_by_qname(base_qname) or NodeId(-1),
                                Edge.extracted(EdgeKind.INHERITS),
                            )

        for child in node.children:
            self._walk_inheritance(child, source, file_key)

    # ── Post-extraction ──────────────────────────────────────────────

    def _resolve_calls(self) -> None:
        """Resolve call:: placeholders via the 6-tier resolver."""
        resolved = resolve_call_placeholders(self.graph)
        logger.info("Call resolution: %d resolved", resolved)

    def _resolve_type_placeholders(self) -> None:
        """Resolve type:: placeholders left by supertype extraction."""
        resolved = resolve_type_placeholders(self.graph)
        if resolved:
            logger.info("Type resolution: %d edges rewired", resolved)

    def _detect_patterns(self) -> None:
        """Detect framework patterns and annotate matched nodes."""
        matches = detect_patterns(self.graph)
        for m in matches:
            pkg_qn = f"pattern::{m.pattern_id}"
            existing = self.graph.find_by_qname(pkg_qn)
            pkg_id = existing if existing is not None else self.graph.add_node(
                Node.new(NodeKind.PACKAGE, pkg_qn)
                    .with_property("pattern", m.pattern_id)
                    .with_property("display_name", m.display_name)
                    .with_property("framework", m.framework)
                    .with_property("category", m.category)
                    .with_property("confidence", m.confidence)
                    .with_property("matched_nodes", m.matched_node_names)
                    .with_property("source_uris", m.source_uris),
            )
            # Link matched nodes to the pattern package.
            for nid_str in m.matched_node_ids:
                try:
                    nid = NodeId(int(nid_str))
                    self.graph.add_edge(
                        nid,
                        pkg_id,
                        Edge.extracted(EdgeKind.DEPENDS_ON),
                    )
                except (ValueError, TypeError):
                    pass
        if matches:
            logger.info("Pattern detection: %d patterns found", len(matches))

    def _enrich_data_flow(self) -> None:
        functions = [
            (node_id, node.source_text)
            for node_id, node in self.graph.nodes()
            if node.kind in (NodeKind.FUNCTION, NodeKind.METHOD) and node.source_text
        ]
        for node_id, source_text in functions:
            extract_data_flow(self.graph, node_id, source_text or "")

    def _derive_tested_by_edges(self) -> None:
        """Derive production-to-test relationships from test-file calls."""
        tests_by_file: dict[str, list[NodeId]] = {}
        for node_id, node in self.graph.nodes():
            if node.kind in (NodeKind.FUNCTION, NodeKind.METHOD) and node.properties.get("is_test"):
                tests_by_file.setdefault(node.source_uri or "", []).append(node_id)

        additions: list[tuple[NodeId, NodeId]] = []
        for _, source_id, target_id, edge in self.graph.edges():
            if edge.kind != EdgeKind.CALLS:
                continue
            source = self.graph.node(source_id)
            target = self.graph.node(target_id)
            if source is None or target is None or target.properties.get("is_test"):
                continue
            production_id = target_id
            if target.source_uri is None:
                candidates = [
                    node_id
                    for node_id, node in self.graph.nodes()
                    if node_id != target_id
                    and node.name == target.name
                    and node.kind in (NodeKind.FUNCTION, NodeKind.METHOD)
                    and not node.properties.get("is_test")
                    and node.source_uri is not None
                ]
                if len(candidates) != 1:
                    continue
                production_id = candidates[0]
            additions.extend(
                (production_id, test_id)
                for test_id in tests_by_file.get(source.source_uri or "", [])
                if production_id != test_id
            )
        for production, test in additions:
            self.graph.add_edge(production, test, Edge.extracted(EdgeKind.TESTED_BY))

    def _build_flows(self) -> None:
        """Build execution flows via the dedicated flow detection engine."""
        count = compute_flows(
            self.graph,
            FlowOptions(
                max_depth=6,
                max_nodes_per_flow=200,
                min_flow_size=3,
            ),
        )
        if count:
            logger.info("Flow detection: %d flows built", count)
