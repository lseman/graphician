"""Tests for refactoring module: rename_preview and find_dead_code."""

from __future__ import annotations

import pytest

from graphician.core.edge import Edge, EdgeKind
from graphician.core.graph import Graph
from graphician.core.node import Node, NodeKind
from graphician.analysis.refactoring.types import RenameEdit, RenamePreview, RenameStats, Confidence
from graphician.analysis.refactoring.engine import (
    rename_preview,
    find_dead_code,
    is_entry_point,
    is_framework_inherited,
    is_test_file,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _add_fn(graph: Graph, qname: str, file: str | None = None, line: int = 0) -> int:
    node = Node.new(NodeKind.FUNCTION, qname).with_source(file or "file", line, line + 10)
    return graph.add_node(node)


def _add_class(graph: Graph, qname: str, file: str | None = None, line: int = 0) -> int:
    node = Node.new(NodeKind.CLASS, qname).with_source(file or "file", line, line + 10)
    return graph.add_node(node)


# ── RenamePreview Types ──────────────────────────────────────────────────


class TestRenameEdit:
    def test_defaults(self):
        e = RenameEdit()
        assert e.file is None
        assert e.line is None
        assert e.old == ""
        assert e.new == ""
        assert e.confidence == Confidence.HIGH

    def test_from_str(self):
        assert Confidence.from_str("high") == Confidence.HIGH
        assert Confidence.from_str("medium") == Confidence.MEDIUM
        assert Confidence.from_str("low") == Confidence.LOW
        assert Confidence.from_str("unknown") == Confidence.LOW


class TestRenameStats:
    def test_empty(self):
        stats = RenameStats.from_edits([])
        assert stats.total == 0
        assert stats.high == 0

    def test_from_edits(self):
        edits = [
            RenameEdit(old="a", new="b", confidence=Confidence.HIGH),
            RenameEdit(old="a", new="b", confidence=Confidence.HIGH),
            RenameEdit(old="a", new="b", confidence=Confidence.MEDIUM),
            RenameEdit(old="a", new="b", confidence=Confidence.LOW),
        ]
        stats = RenameStats.from_edits(edits)
        assert stats.high == 2
        assert stats.medium == 1
        assert stats.low == 1
        assert stats.total == 4


class TestRenamePreviewSerialization:
    def test_to_dict(self):
        preview = RenamePreview(
            target_qname="pkg::foo",
            target_name="foo",
            new_name="bar",
            target_kind="function",
            edits=[
                RenameEdit(file="main.rs", line=10, old="foo", new="bar", confidence="high"),
            ],
            stats=RenameStats(high=1, medium=0, low=0, total=1),
        )
        d = preview.to_dict()
        assert d["target_qname"] == "pkg::foo"
        assert d["new_name"] == "bar"
        assert len(d["edits"]) == 1
        assert d["stats"]["high"] == 1


# ── Rename Preview ──────────────────────────────────────────────────────


class TestRenamePreview:
    """Tests for rename_preview function."""

    def test_finds_call_sites(self):
        g = Graph()
        foo = _add_fn(g, "pkg::foo", "src/lib.rs", 5)
        bar = _add_fn(g, "pkg::bar", "src/main.rs", 10)
        g.add_edge(bar, foo, Edge.extracted(EdgeKind.CALLS))

        preview = rename_preview(g, "pkg::foo", "baz")
        assert preview is not None
        assert preview.target_name == "foo"
        assert preview.new_name == "baz"
        assert len(preview.edits) >= 2  # definition + at least one call site

        # Find the call site edit
        call_edit = next(
            (e for e in preview.edits
             if e.file == "src/main.rs" and e.confidence == Confidence.HIGH),
            None,
        )
        assert call_edit is not None, "should find call site in main.rs"

    def test_no_calls_returns_definition_only(self):
        g = Graph()
        unused = _add_fn(g, "pkg::unused_fn", "src/lib.rs", 50)

        preview = rename_preview(g, "pkg::unused_fn", "renamed")
        assert preview is not None
        assert len(preview.edits) == 1  # only the definition

    def test_finds_import_sites(self):
        g = Graph()
        util = _add_fn(g, "pkg::util", "src/util.rs", 1)
        main = _add_fn(g, "pkg::main", "src/main.rs", 1)
        g.add_edge(main, util, Edge.extracted(EdgeKind.IMPORTS))

        preview = rename_preview(g, "pkg::util", "helper")
        assert preview is not None
        import_edit = next(
            (e for e in preview.edits if e.file == "src/main.rs"),
            None,
        )
        assert import_edit is not None

    def test_not_found_returns_none(self):
        g = Graph()
        assert rename_preview(g, "pkg::nonexistent", "x") is None

    def test_stats_computed(self):
        g = Graph()
        foo = _add_fn(g, "pkg::foo", "src/lib.rs", 5)
        bar = _add_fn(g, "pkg::bar", "src/main.rs", 10)
        g.add_edge(bar, foo, Edge.extracted(EdgeKind.CALLS))

        preview = rename_preview(g, "pkg::foo", "baz")
        assert preview.stats.high >= 1
        assert preview.stats.total == len(preview.edits)


# ── Dead Code Detection ──────────────────────────────────────────────────


class TestIsEntryPoint:
    def test_main(self):
        n = Node.new(NodeKind.FUNCTION, "main")
        assert is_entry_point(n)
        assert is_entry_point(Node.new(NodeKind.FUNCTION, "main_"))
        assert is_entry_point(Node.new(NodeKind.FUNCTION, "run_main"))

    def test_suffix_patterns(self):
        # The Rust reference checks suffix patterns, not prefix
        assert is_entry_point(Node.new(NodeKind.FUNCTION, "some_test_"))
        assert is_entry_point(Node.new(NodeKind.FUNCTION, "Test"))
        assert not is_entry_point(Node.new(NodeKind.FUNCTION, "test_foo"))  # prefix, not suffix

    def test_not_entry(self):
        assert not is_entry_point(Node.new(NodeKind.FUNCTION, "helper"))
        assert not is_entry_point(Node.new(NodeKind.FUNCTION, "compute"))


class TestIsTestFile:
    def test_unit_test(self):
        n = Node.new(NodeKind.FUNCTION, "x").with_source("tests/unit.rs", 1, 10)
        assert is_test_file(n)

    def test_spec_file(self):
        n = Node.new(NodeKind.FUNCTION, "x").with_source("src/foo.spec.ts", 1, 10)
        assert is_test_file(n)

    def test_test_module(self):
        n = Node.new(NodeKind.FUNCTION, "x").with_source("src/test_foo.py", 1, 10)
        assert is_test_file(n)

    def test_regular_file(self):
        n = Node.new(NodeKind.FUNCTION, "x").with_source("src/main.rs", 1, 10)
        assert not is_test_file(n)


class TestIsFrameworkInherited:
    def test_stack_suffix(self):
        n = Node.new(NodeKind.CLASS, "MyStack")
        assert is_framework_inherited(n, set())

    def test_resource_suffix(self):
        n = Node.new(NodeKind.CLASS, "DbResource")
        assert is_framework_inherited(n, set())

    def test_model_suffix(self):
        n = Node.new(NodeKind.CLASS, "UserModel")
        assert is_framework_inherited(n, set())

    def test_regular_class(self):
        n = Node.new(NodeKind.CLASS, "Helper")
        assert not is_framework_inherited(n, set())


class TestFindDeadCode:
    def test_excludes_called_functions(self):
        g = Graph()
        caller = _add_fn(g, "pkg::caller", "src/lib.rs", 5)
        alive = _add_fn(g, "pkg::alive", "src/lib.rs", 10)
        dead = _add_fn(g, "pkg::dead_fn", "src/lib.rs", 50)
        # caller calls alive, so alive is referenced
        g.add_edge(caller, alive, Edge.extracted(EdgeKind.CALLS))

        result = find_dead_code(g)
        dead_names = [d["name"] for d in result]
        assert "dead_fn" in dead_names
        assert "alive" not in dead_names

    def test_excludes_entry_points(self):
        g = Graph()
        main_fn = _add_fn(g, "pkg::main", "src/lib.rs", 1)

        result = find_dead_code(g)
        dead_names = [d["name"] for d in result]
        assert "main" not in dead_names

    def test_excludes_test_files(self):
        g = Graph()
        _add_fn(g, "pkg::test_helper", "tests/unit.rs", 10)

        result = find_dead_code(g)
        dead_names = [d["name"] for d in result]
        assert "test_helper" not in dead_names

    def test_excludes_imported_nodes(self):
        g = Graph()
        util = _add_fn(g, "pkg::util_helper", "src/util.rs", 1)
        main = _add_fn(g, "pkg::main", "src/main.rs", 1)
        g.add_edge(main, util, Edge.extracted(EdgeKind.IMPORTS))

        result = find_dead_code(g)
        dead_names = [d["name"] for d in result]
        assert "util_helper" not in dead_names

    def test_returns_empty_for_fully_connected_graph(self):
        g = Graph()
        a = _add_fn(g, "pkg::a", "src/a.rs", 1)
        b = _add_fn(g, "pkg::b", "src/b.rs", 1)
        g.add_edge(a, b, Edge.extracted(EdgeKind.CALLS))
        g.add_edge(b, a, Edge.extracted(EdgeKind.CALLS))

        result = find_dead_code(g)
        assert result == []

    def test_limit_truncates(self):
        g = Graph()
        for i in range(10):
            _add_fn(g, f"pkg::dead_{i}", "src/lib.rs", 50 + i)

        result = find_dead_code(g, limit=5)
        assert len(result) <= 5

    def test_dead_code_has_full_info(self):
        g = Graph()
        dead = _add_fn(g, "pkg::dead_fn", "src/lib.rs", 50)

        result = find_dead_code(g)
        assert len(result) == 1
        d = result[0]
        assert d["qualified_name"] == "pkg::dead_fn"
        assert d["kind"] == "function"
        assert d["file"] == "src/lib.rs"
        assert d["line_start"] == 50
        assert d["line_end"] == 60
