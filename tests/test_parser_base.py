"""Tests for shared parser utilities in extraction/languages/parsers/base.py.

Covers text extraction, tree walking, QName building, source text
extraction, test detection, decorator extraction, and call suppression.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from graphician.core.edge import Edge, EdgeKind
from graphician.core.graph import Graph
from graphician.core.id import NodeId
from graphician.core.node import Node, NodeKind


# ── text / children helpers ──────────────────────────────────────────


class TestTextHelpers:
    """Tests for _text, _children, _child_by_name, _walk_descendants."""

    def test_text_returns_decoded_bytes(self) -> None:
        from graphician.extraction.languages.parsers.base import _text

        fake_node = MagicMock()
        fake_node.start_byte = 0
        fake_node.end_byte = 5
        source = b"hello"
        assert _text(fake_node, source) == "hello"

    def test_text_handles_invalid_utf8(self) -> None:
        from graphician.extraction.languages.parsers.base import _text

        fake_node = MagicMock()
        fake_node.start_byte = 0
        fake_node.end_byte = 3
        source = b"\xff\xfe\x00\x01\x02"
        result = _text(fake_node, source)
        assert "�" in result

    def _make_node(self, child_types: list[str], field_names: list[str | None] | None = None) -> MagicMock:
        """Create a fake tree-sitter node for testing."""
        children: list[MagicMock] = []
        for i, ctype in enumerate(child_types):
            child = MagicMock()
            child.type = ctype
            child.start_byte = i * 10
            child.end_byte = i * 10 + len(ctype) * 5
            child.children = []
            children.append(child)
        node = MagicMock()
        node.children = children
        node.start_byte = 0
        node.end_byte = sum(c.end_byte for c in children) or 100
        node.field_name_for_child.return_value = None
        return node

    def test_children_returns_list(self) -> None:
        from graphician.extraction.languages.parsers.base import _children

        node = self._make_node(["a", "b", "c"])
        children = _children(node)
        assert len(children) == 3
        assert all(c.type in ("a", "b", "c") for c in children)

    def test_child_by_name_type_match(self) -> None:
        from graphician.extraction.languages.parsers.base import _child_by_name

        node = self._make_node(["foo", "bar", "baz"])
        child = _child_by_name(node, "bar")
        assert child is not None
        assert child.type == "bar"

    def test_child_by_name_field_match(self) -> None:
        from graphician.extraction.languages.parsers.base import _child_by_name

        children: list[MagicMock] = []
        for i, ctype in enumerate(["foo", "bar"]):
            c = MagicMock()
            c.type = ctype
            c.children = []
            children.append(c)
        node = MagicMock()
        node.children = children
        node.field_name_for_child.side_effect = [None, "target_field"]

        child = _child_by_name(node, "target_field")
        assert child is not None
        assert child.type == "bar"

    def test_child_by_name_returns_none(self) -> None:
        from graphician.extraction.languages.parsers.base import _child_by_name

        node = self._make_node(["foo", "bar"])
        child = _child_by_name(node, "missing")
        assert child is None

    def test_walk_descendants_flattens(self) -> None:
        from graphician.extraction.languages.parsers.base import _walk_descendants

        leaf = MagicMock()
        leaf.type = "leaf"
        leaf.children = []

        inner = MagicMock()
        inner.type = "inner"
        inner.children = [leaf]

        root = MagicMock()
        root.type = "root"
        root.children = [inner]

        nodes = list(_walk_descendants(root))
        types = {n.type for n in nodes}
        assert "inner" in types
        assert "leaf" in types


# ── test detection ───────────────────────────────────────────────────


class TestTestDetection:
    """Tests for _is_test_name and _is_test_attribute."""

    def test_is_test_name_prefixed(self) -> None:
        from graphician.extraction.languages.parsers.base import _is_test_name

        assert _is_test_name("test_foo") is True
        assert _is_test_name("test_bar_baz") is True

    def test_is_test_name_suffixed(self) -> None:
        from graphician.extraction.languages.parsers.base import _is_test_name

        assert _is_test_name("foo_test") is True
        assert _is_test_name("bar_test") is True

    def test_is_test_name_capitalized(self) -> None:
        from graphician.extraction.languages.parsers.base import _is_test_name

        assert _is_test_name("TestFoo") is True
        assert _is_test_name("Test") is True

    def test_is_test_name_lowercase(self) -> None:
        from graphician.extraction.languages.parsers.base import _is_test_name

        assert _is_test_name("test_something") is True
        assert _is_test_name("test") is True

    def test_is_test_name_non_test(self) -> None:
        from graphician.extraction.languages.parsers.base import _is_test_name

        assert _is_test_name("main") is False
        assert _is_test_name("foo") is False

    def test_is_test_attribute_rust_test(self) -> None:
        from graphician.extraction.languages.parsers.base import _is_test_attribute

        assert _is_test_attribute("#[test]") is True
        assert _is_test_attribute("#[test_case]") is True
        assert _is_test_attribute("#[rstest]") is True
        assert _is_test_attribute("#[pytest.mark]") is True

    def test_is_test_attribute_module_style(self) -> None:
        from graphician.extraction.languages.parsers.base import _is_test_attribute

        assert _is_test_attribute("module::test") is True
        assert _is_test_attribute("foo::bar::test") is True

    def test_is_test_attribute_not_test(self) -> None:
        from graphician.extraction.languages.parsers.base import _is_test_attribute

        assert _is_test_attribute("#[derive]") is False
        assert _is_test_attribute("#[serde]") is False
        assert _is_test_attribute("some_macro") is False


# ── source text extraction ───────────────────────────────────────────


class TestSourceExtraction:
    """Tests for _extract_source_text."""

    def test_extract_source_text_basic(self) -> None:
        from graphician.extraction.languages.parsers.base import _extract_source_text

        source = b"line1\nline2\nline3\nline4"
        result = _extract_source_text(source, 2, 3)
        assert result == "line2\nline3"

    def test_extract_source_text_clamp_start(self) -> None:
        from graphician.extraction.languages.parsers.base import _extract_source_text

        source = b"a\nb\nc"
        result = _extract_source_text(source, 0, 2)
        assert result == "a\nb"

    def test_extract_source_text_clamp_end(self) -> None:
        from graphician.extraction.languages.parsers.base import _extract_source_text

        source = b"a\nb\nc"
        result = _extract_source_text(source, 3, 100)
        assert result == "c"

    def test_extract_source_text_single_line(self) -> None:
        from graphician.extraction.languages.parsers.base import _extract_source_text

        source = b"only_line"
        result = _extract_source_text(source, 1, 1)
        assert result == "only_line"


# ── graph node helpers ───────────────────────────────────────────────


class TestGraphNodeHelpers:
    """Tests for _add_node."""

    def test_add_node_creates_new(self, tmp_path: Path) -> None:
        from graphician.extraction.languages.parsers.base import _add_node

        graph = Graph()
        path = tmp_path / "app.py"
        path.write_text("")
        idx = _add_node(graph, NodeKind.FUNCTION, "app::foo", path, 1, 1)
        assert idx >= 0
        node = graph.node(NodeId(idx))
        assert node.qualified_name == "app::foo"
        assert node.kind == NodeKind.FUNCTION

    def test_add_node_skips_existing(self, tmp_path: Path) -> None:
        from graphician.extraction.languages.parsers.base import _add_node

        graph = Graph()
        path = tmp_path / "app.py"
        path.write_text("")
        idx1 = _add_node(graph, NodeKind.FUNCTION, "app::foo", path, 1, 1)
        idx2 = _add_node(graph, NodeKind.FUNCTION, "app::foo", path, 2, 2)
        assert idx1 == idx2

    def test_add_node_with_source_text(self, tmp_path: Path) -> None:
        from graphician.extraction.languages.parsers.base import _add_node

        graph = Graph()
        path = tmp_path / "app.py"
        path.write_text("")
        idx = _add_node(
            graph, NodeKind.FUNCTION, "app::foo", path, 1, 1,
            source=b"def foo(): pass\n",
        )
        node = graph.node(NodeId(idx))
        assert "def foo(): pass" in node.source_text

    def test_add_node_with_properties(self, tmp_path: Path) -> None:
        from graphician.extraction.languages.parsers.base import _add_node

        graph = Graph()
        path = tmp_path / "app.py"
        path.write_text("")
        idx = _add_node(
            graph, NodeKind.FUNCTION, "app::foo", path, 1, 1,
            props={"visibility": "public"},
        )
        node = graph.node(NodeId(idx))
        assert node.properties["visibility"] == "public"


# ── QName helpers ────────────────────────────────────────────────────


class TestQNameHelpers:
    """Tests for _scoped_qname, _file_qn, _clean_use_path."""

    def test_scoped_qname_with_scope(self) -> None:
        from graphician.extraction.languages.parsers.base import _scoped_qname

        result = _scoped_qname("app", ["Service"], "method")
        assert result == "app::Service::method"

    def test_scoped_qname_multi_scope(self) -> None:
        from graphician.extraction.languages.parsers.base import _scoped_qname

        result = _scoped_qname("app", ["A", "B"], "x")
        assert result == "app::A::B::x"

    def test_scoped_qname_no_scope(self) -> None:
        from graphician.extraction.languages.parsers.base import _scoped_qname

        result = _scoped_qname("app", [], "method")
        assert result == "app::method"

    def test_file_qn_from_stem(self) -> None:
        from graphician.extraction.languages.parsers.base import _file_qn

        assert _file_qn(Path("app.py")) == "app"
        assert _file_qn(Path("utils.py")) == "utils"
        assert _file_qn(Path("/deep/path/mod.rs")) == "mod"

    def test_clean_use_path_strips_semicolon(self) -> None:
        from graphician.extraction.languages.parsers.base import _clean_use_path

        assert _clean_use_path("std::io::Write;") == "std::io::Write"
        assert _clean_use_path("  foo::bar  ;  ") == "foo::bar"
        assert _clean_use_path("no_semicolon") == "no_semicolon"


# ── call suppression ─────────────────────────────────────────────────


class TestCallSuppression:
    """Tests for _should_suppress_call."""

    def test_suppresses_rust_std_panic(self) -> None:
        from graphician.extraction.languages.parsers.base import _should_suppress_call

        assert _should_suppress_call("std::panic") is True
        assert _should_suppress_call("std::result") is True
        assert _should_suppress_call("std::option::Option") is True

    def test_suppresses_rust_std_containers(self) -> None:
        from graphician.extraction.languages.parsers.base import _should_suppress_call

        assert _should_suppress_call("std::vec::Vec") is True
        assert _should_suppress_call("std::boxed::Box") is True

    def test_suppresses_rust_std_traits(self) -> None:
        from graphician.extraction.languages.parsers.base import _should_suppress_call

        assert _should_suppress_call("std::fmt") is True
        assert _should_suppress_call("std::ops") is True

    def test_does_not_suppress_unknown(self) -> None:
        from graphician.extraction.languages.parsers.base import _should_suppress_call

        assert _should_suppress_call("my_function") is False
        assert _should_suppress_call("foo::bar") is False


# ── decorator extraction ─────────────────────────────────────────────


class TestDecoratorExtraction:
    """Tests for _extract_decorators."""

    def test_extract_decorators_single(self) -> None:
        from graphician.extraction.languages.parsers.base import _extract_decorators

        # @mydec\n  ->  @=0, m=1, y=2, d=3, e=4, c=5, \n=6
        source = b"@mydec\ndef foo(): pass"
        dec_ident = MagicMock()
        dec_ident.type = "identifier"
        dec_ident.start_byte = 1
        dec_ident.end_byte = 6

        dec_node = MagicMock()
        dec_node.type = "decorator"
        dec_node.children = [dec_ident]

        node = MagicMock()
        node.children = [dec_node]
        node.field_name_for_child.return_value = None

        result = _extract_decorators(node, source)
        assert result == ["mydec"]

    def test_extract_decorators_empty(self) -> None:
        from graphician.extraction.languages.parsers.base import _extract_decorators

        node = MagicMock()
        node.children = []
        node.field_name_for_child.return_value = None

        source = b"def foo(): pass"
        result = _extract_decorators(node, source)
        assert result == []

    def test_extract_decorators_multiple(self) -> None:
        from graphician.extraction.languages.parsers.base import _extract_decorators

        source = b"@dec1\n@dec2\ndef foo(): pass"
        # dec1 bytes: 1-4 -> b'dec' -> WRONG, let me check:
        # @ = 0, d=1, e=2, c=3, 1=4, \n=5
        # dec1 is 1-5, @ = 0
        # @ = 0, d=1, e=2, c=3, 1=4, \n=5, @=6, d=7, e=8, c=9, 2=10
        # dec1 = 1-5, dec2 = 7-11
        dec1_ident = MagicMock()
        dec1_ident.type = "identifier"
        dec1_ident.start_byte = 1
        dec1_ident.end_byte = 5

        dec1_node = MagicMock()
        dec1_node.type = "decorator"
        dec1_node.children = [dec1_ident]

        dec2_ident = MagicMock()
        dec2_ident.type = "identifier"
        dec2_ident.start_byte = 7
        dec2_ident.end_byte = 11

        dec2_node = MagicMock()
        dec2_node.type = "decorator"
        dec2_node.children = [dec2_ident]

        node = MagicMock()
        node.children = [dec1_node, dec2_node]
        node.field_name_for_child.return_value = None

        result = _extract_decorators(node, source)
        assert len(result) == 2
        assert "dec1" in result
        assert "dec2" in result

    def test_extract_decorators_skips_at_symbol(self) -> None:
        from graphician.extraction.languages.parsers.base import _extract_decorators

        # @mydec  ->  @=0, m=1, y=2, d=3, e=4, c=5
        source = b"@mydec"
        at_sym = MagicMock()
        at_sym.type = "@"
        at_sym.start_byte = 0
        at_sym.end_byte = 1

        ident = MagicMock()
        ident.type = "identifier"
        ident.start_byte = 1
        ident.end_byte = 6

        dec_node = MagicMock()
        dec_node.type = "decorator"
        dec_node.children = [at_sym, ident]

        node = MagicMock()
        node.children = [dec_node]
        node.field_name_for_child.return_value = None

        result = _extract_decorators(node, source)
        assert result == ["mydec"]


# ── scoped qname helper ──────────────────────────────────────────────


class TestScopedQNameHelper:
    """Tests for _scoped_qname edge cases."""

    def test_empty_scope(self) -> None:
        from graphician.extraction.languages.parsers.base import _scoped_qname

        result = _scoped_qname("app", [], "func")
        assert result == "app::func"

    def test_single_scope(self) -> None:
        from graphician.extraction.languages.parsers.base import _scoped_qname

        result = _scoped_qname("app", ["MyClass"], "method")
        assert result == "app::MyClass::method"

    def test_deep_scope(self) -> None:
        from graphician.extraction.languages.parsers.base import _scoped_qname

        result = _scoped_qname("app", ["A", "B", "C", "D"], "x")
        assert result == "app::A::B::C::D::x"
