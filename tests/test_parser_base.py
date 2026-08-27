"""Tests for base parser utilities and multi-language parser extraction."""

from __future__ import annotations

import tempfile
from pathlib import Path

from graphician.core.graph import Graph
from graphician.core.node import NodeKind
from graphician.extraction.languages.parsers.base import (
    _extract_decorators,
    _is_test_attribute,
    _is_test_name,
    _should_suppress_call,
)
from graphician.extraction.languages.parsers.cpp import extract_file as extract_cpp
from graphician.extraction.languages.parsers.java import extract_file as extract_java
from graphician.extraction.languages.parsers.javascript import extract_file as extract_javascript
from graphician.extraction.languages.parsers.python import extract_file as extract_python
from graphician.extraction.languages.parsers.typescript import extract_file as extract_typescript


class _MockNode:
    """Mock node for tests."""

    type = "statement"
    children = []  # noqa: RUF012


# ---------------------------------------------------------------------------
# Base parser utilities
# ---------------------------------------------------------------------------

class TestExtractDecorators:
    """Tests for _extract_decorators."""

    def test_no_decorators(self) -> None:
        class MockChild:
            type = "statement"
            children = []  # noqa: RUF012

        assert _extract_decorators(MockChild(), b"") == []

    def test_decorator_node(self) -> None:
        # Create a mock that mimics tree-sitter node structure
        class MockChild:
            def __init__(self, node_type: str, source: bytes, children=None):
                self.type = node_type
                self.children = children or []

        class MockGrandChild:
            def __init__(self, node_type: str, start: int, end: int, source: bytes):
                self.type = node_type
                self.start_byte = start
                self.end_byte = end
                self._source = source

            def text(self) -> bytes:
                return self._source[self.start_byte:self.end_byte]

        source = b"@decorator\ndef foo():\n    pass\n"
        grandchild = MockGrandChild("identifier", 1, 11, source)
        child = MockChild("decorator", source, [grandchild])
        parent = _MockNode()
        parent.children = [child]
        result = _extract_decorators(parent, source)
        assert len(result) == 1
        assert result[0] == "decorator"


class TestIsTestName:
    """Tests for _is_test_name."""

    def test_test_prefix(self) -> None:
        assert _is_test_name("test_foo")

    def test_test_suffix(self) -> None:
        assert _is_test_name("foo_test")

    def test_dunder_test(self) -> None:
        assert not _is_test_name("__test__")

    def test_non_test(self) -> None:
        assert not _is_test_name("process_data")

    def test_test_prefix_lowercase(self) -> None:
        assert _is_test_name("testify")


class TestIsTestAttribute:
    """Tests for _is_test_attribute."""

    def test_rust_test_attribute(self) -> None:
        assert _is_test_attribute("#[test]")

    def test_non_test(self) -> None:
        assert not _is_test_attribute("method")
        assert not _is_test_attribute("test_method")
        assert not _is_test_attribute("_test_foo")


class TestShouldSuppressCall:
    """Tests for _should_suppress_call."""

    def test_suppressed_name(self) -> None:
        assert _should_suppress_call("std::mem")
        assert _should_suppress_call("std::vec::Vec")

    def test_not_suppressed(self) -> None:
        assert not _should_suppress_call("process_data")
        assert not _should_suppress_call("calculate_total")
        assert not _should_suppress_call("len")
        assert not _should_suppress_call("range")
        assert not _should_suppress_call("print")


# ---------------------------------------------------------------------------
# Python parser integration tests
# ---------------------------------------------------------------------------

class TestPythonParser:
    """Test Python file extraction."""

    def _make_graph(self, code: bytes) -> Graph:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            graph = Graph()
            extract_python(Path(f.name), graph)
        return graph

    def test_extract_simple_function(self) -> None:
        code = b'def hello():\n    """Say hello."""\n    pass\n'
        graph = self._make_graph(code)
        nodes = list(graph.nodes())
        assert len(nodes) >= 1

    def test_extract_function_with_docstring(self) -> None:
        code = b'def foo():\n    """Foo docs."""\n    pass\n'
        graph = self._make_graph(code)
        found = False
        for _, node in graph.nodes():
            if node.name == "foo":
                assert node.source_text and "Foo docs." in node.source_text
                found = True
                break
        assert found

    def test_extract_class(self) -> None:
        code = b'class MyClass:\n    """My docs."""\n    pass\n'
        graph = self._make_graph(code)
        found = False
        for _, node in graph.nodes():
            if node.name == "MyClass" and node.kind == NodeKind.CLASS:
                assert node.source_text and "My docs." in node.source_text
                found = True
                break
        assert found

    def test_extract_function_call(self) -> None:
        code = b'def caller():\n    callee()\n\ndef callee():\n    pass\n'
        graph = self._make_graph(code)
        assert graph.node_count() >= 2


# ---------------------------------------------------------------------------
# JavaScript parser integration tests
# ---------------------------------------------------------------------------

class TestJavaScriptParser:
    """Test JavaScript file extraction."""

    def _make_graph(self, code: bytes) -> Graph:
        with tempfile.NamedTemporaryFile(suffix=".js", delete=False) as f:
            f.write(code)
            f.flush()
            graph = Graph()
            extract_javascript(Path(f.name), graph)
        return graph

    def test_extract_function(self) -> None:
        code = b'function hello() { console.log("hi"); }'
        graph = self._make_graph(code)
        assert graph.node_count() >= 1

    def test_extract_arrow_function(self) -> None:
        code = b'const foo = () => { return 42; }'
        graph = self._make_graph(code)
        assert graph.node_count() >= 1


# ---------------------------------------------------------------------------
# TypeScript parser integration tests
# ---------------------------------------------------------------------------

class TestTypeScriptParser:
    """Test TypeScript file extraction."""

    def _make_graph(self, code: bytes) -> Graph:
        with tempfile.NamedTemporaryFile(suffix=".ts", delete=False) as f:
            f.write(code)
            f.flush()
            graph = Graph()
            extract_typescript(Path(f.name), graph)
        return graph

    def test_extract_function_with_types(self) -> None:
        code = b'function foo(x: number): string { return String(x); }'
        graph = self._make_graph(code)
        assert graph.node_count() >= 1

    def test_extract_class(self) -> None:
        code = b'class Greeter { greet(): string { return "hi"; } }'
        graph = self._make_graph(code)
        assert graph.node_count() >= 1


# ---------------------------------------------------------------------------
# Java parser integration tests
# ---------------------------------------------------------------------------

class TestJavaParser:
    """Test Java file extraction."""

    def _make_graph(self, code: bytes) -> Graph:
        with tempfile.NamedTemporaryFile(suffix=".java", delete=False) as f:
            f.write(code)
            f.flush()
            graph = Graph()
            extract_java(Path(f.name), graph)
        return graph

    def test_extract_method(self) -> None:
        code = b'public class Foo { public void bar() {} }'
        graph = self._make_graph(code)
        assert graph.node_count() >= 1

    def test_extract_class_declaration(self) -> None:
        code = b'public class MyClass { int x; }'
        graph = self._make_graph(code)
        assert graph.node_count() >= 1


# ---------------------------------------------------------------------------
# C++ parser integration tests
# ---------------------------------------------------------------------------

class TestCppParser:
    """Test C++ file extraction."""

    def _make_graph(self, code: bytes) -> Graph:
        with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False) as f:
            f.write(code)
            f.flush()
            graph = Graph()
            extract_cpp(Path(f.name), graph)
        return graph

    def test_extract_function(self) -> None:
        code = b'void foo() { int x = 1; }'
        graph = self._make_graph(code)
        assert graph.node_count() >= 1

    def test_extract_class(self) -> None:
        code = b'class Foo { public: void bar(); };'
        graph = self._make_graph(code)
        assert graph.node_count() >= 1
