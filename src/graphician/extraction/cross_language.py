"""Cross-language call resolution for polyglot codebases.

Detects FFI boundaries between languages and resolves call placeholders
that cross language boundaries — e.g. Python calling Rust via PyO3,
TypeScript importing JavaScript, Python calling C via pybind11.

Resolution strategy:
1. FFI boundary detection via manifest parsing and source-level FFI patterns
2. Cross-language type registry mapping types/functions across languages
3. Tier 8 call resolution: when all 7 previous tiers fail, look up FFI targets
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..core.edge import Edge, EdgeKind
from ..core.graph import Graph
from ..core.id import EdgeId, NodeId
from ..core.node import Node, NodeKind
from .call_resolution import should_suppress_call_placeholder

# ── Data structures ──────────────────────────────────────────────────────


@dataclass
class FFIBoundary:
    """A boundary where one language calls into another."""

    caller_language: str  # e.g. "python", "rust", "typescript"
    callee_language: str  # e.g. "rust", "python", "javascript"
    binding_name: str  # e.g. "pyo3", "pybind11", "ctypes", "esm"
    source_uri: str  # The file declaring the boundary
    module_name: str | None = None  # The importable module name


@dataclass
class CrossLangType:
    """A type/function available across language boundaries."""

    qualified_name: str
    language: str
    kind: NodeKind
    source_dialect: str  # Where this type lives
    exportable: bool = True  # Can it be imported from another language?


# ── FFI Detection ────────────────────────────────────────────────────────

# PyO3 decorator patterns that expose Python-callable Rust functions
_PYTHON_CALLABLE_RUST_PATTERNS = [
    # #[pyfunction]
    re.compile(r"^\s*#\s*\[pyfunction\]"),
    # #[pyo3::pyfunction]
    re.compile(r"^\s*#\s*\[pyo3::pyfunction\]"),
    # #[pymodule]
    re.compile(r"^\s*#\s*\[pymodule\]"),
    # #[pyclass]
    re.compile(r"^\s*#\s*\[pyclass\]"),
]

# pybind11 patterns
_PYBIND11_PATTERNS = [
    re.compile(r"PYBIND11_MODULE\s*\("),
    re.compile(r"PYBIND11_MODULE\("),
    re.compile(r"\.def\s*\(\s*\""),
    re.compile(r"\.def_\w+\s*\(\s*\""),
]

# ctypes/cffi import patterns in Python
_CTYPE_IMPORT_PATTERNS = [
    re.compile(r"^\s*from\s+ctypes\s+import"),
    re.compile(r"^\s*import\s+ctypes"),
    re.compile(r"^\s*from\s+cffi\s+import"),
    re.compile(r"^\s*import\s+cffi"),
    re.compile(r"^\s*from\s+cython\s+import"),
    re.compile(r"^\s*from\s+numpy.*cimport\s"),
]

# Rust PyO3 import patterns
_RUST_PYO3_IMPORTS = [
    re.compile(r"^\s*use\s+pyo3::prelude::"),
    re.compile(r"^\s*use\s+pyo3::"),
]

# TypeScript ESM → CommonJS interop
_TS_JS_IMPORT_PATTERNS = [
    re.compile(r"^\s*import\s+.*\s+from\s+['\"]\.[^'\"]*\.(js|cjs|mjs)['\"]"),
    re.compile(r"^\s*require\s*\(\s*['\"]\.[^'\"]*\.(js|cjs|mjs)['\"]"),
]


def _detect_ffi_boundaries(graph: Graph) -> list[FFIBoundary]:
    """Detect FFI boundaries in the graph by analyzing source patterns.

    Scans all files for FFI-related patterns and returns detected boundaries.
    """
    boundaries: list[FFIBoundary] = []
    seen: set[tuple[str, str, str]] = set()  # (caller_lang, callee_lang, module)

    for _nid, node in graph.nodes():
        if not node.source_text or not node.source_uri:
            continue

        source = node.source_text
        uri = node.source_uri
        ext = Path(uri).suffix.lower()

        # Determine caller language
        if ext == ".py":
            caller_lang = "python"
        elif ext == ".rs":
            caller_lang = "rust"
        elif ext in (".ts", ".tsx"):
            caller_lang = "typescript"
        elif ext in (".js", ".jsx"):
            caller_lang = "javascript"
        elif ext in (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"):
            caller_lang = "cpp"
        else:
            continue

        # Detect: Python calls Rust via PyO3 bindings
        if caller_lang == "rust":
            for pattern in _PYTHON_CALLABLE_RUST_PATTERNS:
                if pattern.search(source):
                    # Find the module name from #[pymodule] or derive from file
                    module_match = re.search(
                        r"#\[pymodule\]\s*fn\s+(\w+)", source
                    )
                    module_name = module_match.group(1) if module_match else None
                    # Also check if function has #[pyfunction]
                    if re.search(r"#\[pyfunction\]", source):
                        key = ("rust", "python", node.qualified_name)
                        if key not in seen:
                            seen.add(key)
                            boundaries.append(FFIBoundary(
                                caller_language="rust",
                                callee_language="python",
                                binding_name="pyo3",
                                source_uri=uri,
                                module_name=module_name,
                            ))
                    break

        # Detect: Rust uses PyO3 imports (confirms PyO3 project)
        if caller_lang == "rust":
            for pattern in _RUST_PYO3_IMPORTS:
                if pattern.search(source):
                    key = ("rust", "python", "pyo3_project")
                    if key not in seen:
                        seen.add(key)
                        # Find the nearest Cargo.toml for project-level detection
                        cargo_path = Path(uri).parent / "Cargo.toml"
                        boundaries.append(FFIBoundary(
                            caller_language="rust",
                            callee_language="python",
                            binding_name="pyo3",
                            source_uri=str(cargo_path) if cargo_path.exists() else uri,
                        ))
                    break

        # Detect: Python uses ctypes/cffi/Cython → calls C
        if caller_lang == "python":
            for pattern in _CTYPE_IMPORT_PATTERNS:
                if pattern.search(source):
                    key = ("python", "cpp", "ffi")
                    if key not in seen:
                        seen.add(key)
                        boundaries.append(FFIBoundary(
                            caller_language="python",
                            callee_language="cpp",
                            binding_name="ctypes",
                            source_uri=uri,
                        ))
                    break

        # Detect: TypeScript imports JavaScript files
        if caller_lang == "typescript":
            for pattern in _TS_JS_IMPORT_PATTERNS:
                if pattern.search(source):
                    module_match = re.search(
                        r"['\"]\.[^'\"]*\.(js|cjs|mjs)['\"]", source
                    )
                    if module_match:
                        module_name = module_match.group(0).strip("'\"")
                        key = ("typescript", "javascript", module_name)
                        if key not in seen:
                            seen.add(key)
                            boundaries.append(FFIBoundary(
                                caller_language="typescript",
                                callee_language="javascript",
                                binding_name="esm",
                                source_uri=uri,
                                module_name=module_name,
                            ))
                    break

    return boundaries


# ── Cross-Language Type Registry ─────────────────────────────────────────


class CrossLanguageRegistry:
    """Registry of types and functions available across language boundaries.

    Built from FFI boundaries + library stubs, this registry maps
    cross-language type references to their actual definitions.
    """

    def __init__(self) -> None:
        self._types: dict[str, CrossLangType] = {}
        self._boundaries: list[FFIBoundary] = []
        self._callee_index: dict[str, list[str]] = {}  # name -> [qnames]

    def add_boundary(self, boundary: FFIBoundary) -> None:
        """Register an FFI boundary."""
        self._boundaries.append(boundary)

    def register_type(self, qname: str, language: str, kind: NodeKind,
                      source_dialect: str, exportable: bool = True) -> None:
        """Register a cross-language type."""
        self._types[qname] = CrossLangType(
            qualified_name=qname,
            language=language,
            kind=kind,
            source_dialect=source_dialect,
            exportable=exportable,
        )
        self._callee_index.setdefault(qname.rsplit("::", 1)[-1], []).append(qname)

    def lookup_by_name(self, name: str) -> list[CrossLangType]:
        """Look up types by method/function name."""
        return self._types.values()

    def get_boundaries(self) -> list[FFIBoundary]:
        return self._boundaries

    def has_boundary(self, caller_lang: str, callee_lang: str) -> bool:
        """Check if there's an FFI boundary between two languages."""
        return any(
            b.caller_language == caller_lang and b.callee_language == callee_lang
            for b in self._boundaries
        )


# ── Build Cross-Language Registry ───────────────────────────────────────


def build_cross_language_registry(
    graph: Graph,
    boundaries: list[FFIBoundary] | None = None,
) -> CrossLanguageRegistry:
    """Build a cross-language registry from FFI boundaries + stubs.

    This connects stub types to their cross-language availability:
    - Rust PyO3 types → callable from Python
    - Python stubs with FFI → callable from Rust
    - JS stubs → callable from TypeScript
    """
    registry = CrossLanguageRegistry()

    # If no boundaries detected, create a minimal one for TS→JS
    if boundaries is None:
        boundaries = _detect_ffi_boundaries(graph)

    for b in boundaries:
        registry.add_boundary(b)

    # Register cross-language stub types from known stubs
    from .library_stubs._cpp_stubs import _CPP_STUBS
    from .library_stubs._javascript_stubs import _JAVASCRIPT_STUBS
    from .library_stubs._python_stubs import _PYTHON_STUBS
    from .library_stubs._rust_stubs import _RUST_STUBS

    # Rust types available in Python via PyO3
    rust_to_python = "pyo3" in [b.binding_name for b in boundaries]
    for type_name, _methods in _RUST_STUBS.items():
        qname = f"stub::{type_name}"
        registry.register_type(qname, "rust", NodeKind.CLASS, "rust")
    for _type_name, _methods in _RUST_STUBS.items():
        if rust_to_python:
            registry.register_type(qname, "python", NodeKind.CLASS, "rust", exportable=True)

    # Python stubs available from other languages (where FFI exists)
    python_to_rust = any(
        b.caller_language == "python" and b.callee_language == "rust"
        for b in boundaries
    )
    for type_name, _methods in _PYTHON_STUBS.items():
        qname = f"stub::{type_name}"
        registry.register_type(qname, "python", NodeKind.CLASS, "python")
        if python_to_rust:
            registry.register_type(qname, "rust", NodeKind.CLASS, "python", exportable=True)

    # JS stubs (always cross-compatible with TypeScript)
    for type_name, _methods in _JAVASCRIPT_STUBS.items():
        qname = f"stub::{type_name}"
        registry.register_type(qname, "javascript", NodeKind.CLASS, "javascript")

    # C++ stubs
    for type_name, _methods in _CPP_STUBS.items():
        qname = f"stub::{type_name}"
        registry.register_type(qname, "cpp", NodeKind.CLASS, "cpp")

    # Index by method name for lookup
    for type_name, _methods in _RUST_STUBS.items():
        for method in _methods:
            qname = f"stub::{type_name}"
            registry._callee_index.setdefault(method, []).append(qname)

    for type_name, _methods in _JAVASCRIPT_STUBS.items():
        for method in _methods:
            qname = f"stub::{type_name}"
            registry._callee_index.setdefault(method, []).append(qname)

    for type_name, _methods in _CPP_STUBS.items():
        for method in _methods:
            qname = f"stub::{type_name}"
            registry._callee_index.setdefault(method, []).append(qname)

    for type_name, _methods in _PYTHON_STUBS.items():
        for method in _methods:
            qname = f"stub::{type_name}"
            registry._callee_index.setdefault(method, []).append(qname)

    return registry


# ── Tier 8: Cross-Language Call Resolution ─────────────────────────────


def resolve_cross_language_calls(
    graph: Graph,
    registry: CrossLanguageRegistry | None = None,
) -> int:
    """Tier 8: resolve remaining call:: placeholders via cross-language resolution.

    After all 7 previous tiers fail, this resolver checks if the unresolved
    call target is available via an FFI boundary.

    Resolution rules:
    1. If caller is Python and target method exists in Rust stubs + PyO3 detected:
       resolve to the Rust stub node
    2. If caller is TypeScript and target exists in JS stubs: resolve to JS stub
    3. If caller is Rust and target method exists in Python stubs + PyO3 detected:
       resolve to Python stub node

    Returns:
        Number of new cross-language edges added.
    """
    if registry is None:
        boundaries = _detect_ffi_boundaries(graph)
        registry = build_cross_language_registry(graph, boundaries)

    # Gather unresolved call edges
    edge_data = []
    for eid, src, dst, edge in graph.edges():
        if edge.kind != EdgeKind.CALLS:
            continue
        dst_node = graph.node(dst)
        if dst_node is None:
            continue
        qn = dst_node.qualified_name
        if not qn.startswith("call::"):
            continue
        if should_suppress_call_placeholder(qn[6:]):
            continue
        edge_data.append((eid, src, dst, edge, qn))

    if not edge_data:
        return 0

    # Build source dialect map
    source_dialects: dict[int, str] = {}
    for _eid, src, _dst, _edge, _qn in edge_data:
        src_node = graph.node(src)
        if src_node is not None:
            suffix = Path(src_node.source_uri).suffix.lower() if src_node.source_uri else ""
            if suffix == ".py":
                source_dialects[src.value] = "python"
            elif suffix == ".rs":
                source_dialects[src.value] = "rust"
            elif suffix in (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"):
                source_dialects[src.value] = "javascript"
            elif suffix in (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"):
                source_dialects[src.value] = "cpp"

    additions: list[tuple[int, int, str, bool]] = []  # (src, dst, tag, structural)
    stale_edges: list[EdgeId] = []

    for _eid, src, _dst, _edge, callee_qn in edge_data:
        method_name = callee_qn[6:]
        src_dialect = source_dialects.get(src.value)
        if src_dialect is None:
            continue

        # Check if this method exists in cross-language targets
        candidates = registry._callee_index.get(method_name, [])
        if not candidates:
            continue

        # Resolve based on caller's language context
        resolved_qname = _resolve_cross_language_candidate(
            method_name, candidates, src_dialect, registry
        )

        if resolved_qname is None:
            continue

        # Find the target node
        target_node = graph.find_by_qname(resolved_qname)
        if target_node is None:
            # Create stub node
            target_node = _create_stub_node(graph, resolved_qname)
            if target_node is None:
                continue

        target_id = target_node
        additions.append((src.value, target_id.value, "cross_language", False))
        stale_edges.append(eid)

    # Apply additions
    for src_val, dst_val, tag, _structural in additions:
        edge = Edge.inferred(EdgeKind.CALLS, 0.5)
        edge.properties["resolved_from"] = f"call_placeholder::{tag}"
        edge.properties["cross_language"] = True
        graph.add_edge(NodeId(src_val), NodeId(dst_val), edge)

    # Remove stale placeholder edges and orphaned call:: nodes
    if stale_edges:
        graph.remove_edges_by_id(stale_edges)
        to_remove: list[NodeId] = []
        for nid, node in graph.nodes():
            if node.qualified_name.startswith("call::"):
                has_edges = any(
                    True for _ in graph.out_neighbors(nid)
                ) or any(True for _ in graph.in_neighbors(nid))
                if not has_edges:
                    to_remove.append(nid)
        for nid in to_remove:
            graph.remove_node(nid)

    return len(additions)


def _resolve_cross_language_candidate(
    method_name: str,
    candidates: list[str],
    caller_dialect: str,
    registry: CrossLanguageRegistry,
) -> str | None:
    """Pick the best cross-language candidate for a call target.

    Resolution priority:
    1. Same dialect (already handled by earlier tiers, but fallback here)
    2. Cross-language via detected FFI boundary
    3. Default dialect order: python > rust > javascript > cpp
    """
    # Python caller → look for Rust/C++ stubs
    # A "rust→python" PyO3 boundary means Rust modules are callable from Python
    if caller_dialect == "python":
        # PyO3: Rust functions callable from Python
        if registry.has_boundary("rust", "python"):
            rust_candidates = [c for c in candidates if registry._types.get(c, CrossLangType("", "", NodeKind.TYPE, "")).language == "rust"]
            if rust_candidates:
                return rust_candidates[0]
        # pybind11: C++ functions callable from Python
        if registry.has_boundary("cpp", "python"):
            cpp_candidates = [c for c in candidates if registry._types.get(c, CrossLangType("", "", NodeKind.TYPE, "")).language == "cpp"]
            if cpp_candidates:
                return cpp_candidates[0]
        # Default: prefer Rust stubs (PyO3 is the most common cross-lang pattern)
        rust_candidates = [c for c in candidates if registry._types.get(c, CrossLangType("", "", NodeKind.TYPE, "")).language == "rust"]
        if rust_candidates:
            return rust_candidates[0]
        # Then C++
        cpp_candidates = [c for c in candidates if registry._types.get(c, CrossLangType("", "", NodeKind.TYPE, "")).language == "cpp"]
        if cpp_candidates:
            return cpp_candidates[0]

    # Rust caller → look for Python stubs (pybind11)
    elif caller_dialect == "rust":
        python_candidates = [c for c in candidates if registry._types.get(c, CrossLangType("", "", NodeKind.TYPE, "")).language == "python"]
        if python_candidates:
            return python_candidates[0]

    # TypeScript/JavaScript caller → cross between JS dialects
    elif caller_dialect in ("typescript", "javascript"):
        js_candidates = [c for c in candidates if registry._types.get(c, CrossLangType("", "", NodeKind.TYPE, "")).language in ("javascript", "typescript")]
        if js_candidates:
            return js_candidates[0]

    # C++ caller → Python stubs (pybind11)
    elif caller_dialect == "cpp":
        python_candidates = [c for c in candidates if registry._types.get(c, CrossLangType("", "", NodeKind.TYPE, "")).language == "python"]
        if python_candidates:
            return python_candidates[0]

    return None


def _create_stub_node(graph: Graph, qname: str) -> Node | None:
    """Create or return a stub node for a cross-language type."""
    type_name = qname[len("stub::"):] if qname.startswith("stub::") else qname

    existing = graph.find_by_qname(qname)
    if existing is not None:
        return existing

    # Infer dialect from the type name
    dialect = _infer_stub_dialect(type_name)
    source_uri = f"<{dialect}-stdlib:{type_name}>"

    node = Node.new(NodeKind.CLASS, qname)
    node = node.with_source(source_uri, 0, 0)
    node = node.with_source_text(f"Cross-language stub for {type_name} ({dialect})")
    node = node.with_property("dialect", dialect)
    node = node.with_property("is_stub", True)
    node = node.with_property("cross_language", True)

    return graph.add_node(node)


def _infer_stub_dialect(type_name: str) -> str:
    """Infer the dialect of a stub type from its name."""
    # Rust stdlib types often start with uppercase and have Rust naming
    rust_indicators = {"Vec", "HashMap", "HashSet", "String", "Option", "Result",
                       "Box", "Arc", "Mutex", "RwLock", "Path", "PathBuf", "Cow"}
    if type_name in rust_indicators:
        return "rust"

    # C++ STL types
    cpp_indicators = {"vector", "string", "map", "unordered_map", "set",
                      "unordered_set", "shared_ptr", "unique_ptr", "iostream"}
    if type_name.lower() in cpp_indicators:
        return "cpp"

    # JavaScript/Node types
    js_indicators = {"Promise", "Map", "Set", "ArrayBuffer", "Error",
                     "process", "console", "Buffer", "URL", "URLSearchParams"}
    if type_name in js_indicators:
        return "javascript"

    # Default to Python for common types
    return "python"


# ── Integration helpers ──────────────────────────────────────────────────


def enrich_with_cross_language_data(
    graph: Graph,
) -> dict[str, int]:
    """Run full cross-language enrichment: detect FFI + build registry + resolve.

    Returns statistics about what was found and resolved.
    """
    boundaries = _detect_ffi_boundaries(graph)
    registry = build_cross_language_registry(graph, boundaries)

    unresolved_before = sum(
        1
        for _, _, target, edge in graph.edges()
        if edge.kind == EdgeKind.CALLS
        and (node := graph.node(target)) is not None
        and node.qualified_name.startswith("call::")
        and not should_suppress_call_placeholder(node.qualified_name[6:])
    )

    resolved = resolve_cross_language_calls(graph, registry)

    return {
        "boundaries_detected": len(boundaries),
        "registry_types": len(registry._types),
        "resolved": resolved,
        "unresolved_remaining": unresolved_before - resolved,
    }
