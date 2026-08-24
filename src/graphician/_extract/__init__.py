"""High-performance code extraction using tree-sitter Rust bindings.

This module provides a Rust-accelerated alternative to the Python AST walker
for significantly faster parsing and symbol extraction.

Usage:
    from graphician._extract import extract_python_file
    result = extract_python_file(source_bytes, file_path="test.py")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Try to load the compiled Rust extension
_lib_path = Path(__file__).parent / "graphician_extract.cpython-314-x86_64-linux-gnu.so"

if _lib_path.exists():
    import importlib.util
    _spec = importlib.util.spec_from_file_location("graphician_extract", _lib_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    extract_python_file = _mod.extract_python_file
    extract_python_files = _mod.extract_python_files
    available = _mod.available
    version = _mod.version
    HAS_RUST = True
else:
    HAS_RUST = False
    extract_python_file = None
    extract_python_files = None
    available = lambda: False
    version = lambda: "0.0.0"

__all__ = ["extract_python_file", "extract_python_files", "available", "version", "HAS_RUST"]
