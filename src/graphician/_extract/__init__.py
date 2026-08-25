"""High-performance code extraction using tree-sitter Rust bindings.

This module provides a Rust-accelerated alternative to the Python AST walker
for significantly faster parsing and symbol extraction.

Usage:
    from graphician._extract import extract_python_file
    result = extract_python_file(source_bytes, file_path="test.py")
"""

from __future__ import annotations

from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

# Try to load the compiled Rust extension
_module_dir = Path(__file__).parent
_lib_path = next(
    (
        _module_dir / f"graphician_native{suffix}"
        for suffix in EXTENSION_SUFFIXES
        if (_module_dir / f"graphician_native{suffix}").exists()
    ),
    None,
)

if _lib_path is not None:
    import importlib.util

    _spec = importlib.util.spec_from_file_location("graphician_native", _lib_path)
    assert _spec is not None and _spec.loader is not None, (
        f"failed to build a module spec for {_lib_path}"
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    extract_python_file = _mod.extract_python_file
    extract_python_files = _mod.extract_python_files
    extract_data_flow = _mod.extract_data_flow
    available = _mod.available
    version = _mod.version
    # Language-specific extractors
    extract_rust_file = _mod.extract_rust_file
    extract_typescript_file = _mod.extract_typescript_file
    extract_javascript_file = _mod.extract_javascript_file
    extract_java_file = _mod.extract_java_file
    extract_cpp_file = _mod.extract_cpp_file
    extract_go_file = _mod.extract_go_file
    CommunityOptions = _mod.CommunityOptions
    community_detection_louvain = _mod.community_detection_louvain
    community_detection_leiden = _mod.community_detection_leiden
    community_detection_infomap = _mod.community_detection_infomap
    dedup_candidate_pairs = _mod.dedup_candidate_pairs
    fuzzy_score_matrix = _mod.fuzzy_score_matrix
    plan_type_resolution = getattr(_mod, "plan_type_resolution", None)
    plan_call_resolution = getattr(_mod, "plan_call_resolution", None)
    save_graph_sqlite = getattr(_mod, "save_graph_sqlite", None)
    save_graph_incremental_sqlite = getattr(_mod, "save_graph_incremental_sqlite", None)
    load_graph_sqlite = getattr(_mod, "load_graph_sqlite", None)
    NativeGraph = _mod.NativeGraph
    HAS_RUST = True
else:
    HAS_RUST = False
    extract_python_file = None
    extract_python_files = None
    extract_data_flow = None
    extract_rust_file = None
    extract_typescript_file = None
    extract_javascript_file = None
    extract_java_file = None
    extract_cpp_file = None
    extract_go_file = None
    CommunityOptions = None
    community_detection_louvain = None
    community_detection_leiden = None
    community_detection_infomap = None
    dedup_candidate_pairs = None
    fuzzy_score_matrix = None
    plan_type_resolution = None
    plan_call_resolution = None
    save_graph_sqlite = None
    save_graph_incremental_sqlite = None
    load_graph_sqlite = None
    NativeGraph = None

    def available() -> bool:
        return False

    def version() -> str:
        return "0.0.0"


__all__ = [
    "HAS_RUST",
    "CommunityOptions",
    "NativeGraph",
    "available",
    "community_detection_infomap",
    "community_detection_leiden",
    "community_detection_louvain",
    "dedup_candidate_pairs",
    "extract_cpp_file",
    "extract_data_flow",
    "extract_go_file",
    "extract_java_file",
    "extract_javascript_file",
    "extract_python_file",
    "extract_python_files",
    "extract_rust_file",
    "extract_typescript_file",
    "fuzzy_score_matrix",
    "load_graph_sqlite",
    "plan_call_resolution",
    "plan_type_resolution",
    "save_graph_incremental_sqlite",
    "save_graph_sqlite",
    "version",
]
