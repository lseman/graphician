"""Contract tests for response-layer modules that lack dedicated test files.

Covers refactor_response, snapshot_diff, token_savings, cache, and
temporal modules to raise coverage on the response/ transport layer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from graphician.core import Edge, EdgeKind, Graph, Node, NodeKind

# ── refactor_response ────────────────────────────────────────────────


class TestRenamePreviewResponse:
    """Tests for rename_preview_json and rename_preview_handler."""

    def test_rename_preview_json_returns_none_for_missing_target(self) -> None:
        from graphician.interfaces.cli.response.refactor_response import (
            rename_preview_json,
        )

        graph = Graph()
        result = rename_preview_json(graph, "nonexistent::func", "new_name")
        assert result is None

    def test_rename_preview_handler_missing_params(self) -> None:
        from graphician.interfaces.cli.response.refactor_response import (
            rename_preview_handler,
        )

        result = rename_preview_handler(Graph(), {})
        assert result["operation"] == "rename_preview"
        assert "error" in result

    def test_rename_preview_handler_missing_new_name(self) -> None:
        from graphician.interfaces.cli.response.refactor_response import (
            rename_preview_handler,
        )

        result = rename_preview_handler(Graph(), {"target": "app::foo"})
        assert result["operation"] == "rename_preview"
        assert "error" in result

    def test_rename_preview_handler_not_found(self) -> None:
        from graphician.interfaces.cli.response.refactor_response import (
            rename_preview_handler,
        )

        result = rename_preview_handler(Graph(), {"target": "missing::func", "new_name": "bar"})
        assert result["operation"] == "rename_preview"
        assert "error" in result
        assert "not found" in result["error"]


# ── token_savings ────────────────────────────────────────────────────


class TestTokenSavings:
    """Tests for approx_tokens and per_file_tokens."""

    def test_approx_tokens_empty_string(self) -> None:
        from graphician.interfaces.cli.response.token_savings import approx_tokens

        assert approx_tokens("") == 1

    def test_approx_tokens_short_string(self) -> None:
        from graphician.interfaces.cli.response.token_savings import approx_tokens

        assert approx_tokens("hello") == 1  # 5 // 4 = 1

    def test_approx_tokens_long_string(self) -> None:
        from graphician.interfaces.cli.response.token_savings import approx_tokens

        s = "a" * 200
        assert approx_tokens(s) == 50  # 200 // 4 = 50

    def test_file_tokens_class(self) -> None:
        from graphician.interfaces.cli.response.token_savings import FileTokens

        ft = FileTokens(1000, 200)
        assert ft.raw == 1000
        assert ft.graph == 200

    def test_per_file_tokens_empty_graph(self) -> None:
        from graphician.interfaces.cli.response.token_savings import per_file_tokens

        graph = Graph()
        result = per_file_tokens(graph)
        assert result == {}

    def test_per_file_tokens_with_nodes(self, tmp_path: Path) -> None:
        from graphician.interfaces.cli.response.token_savings import per_file_tokens

        graph = Graph()
        src = tmp_path / "app.py"
        src.write_text("def foo(): pass\n")
        file_node = graph.add_node(
            Node.new(NodeKind.FILE, f"file://{src}").with_source(str(src), 1, 1)
        )
        func_node = graph.add_node(
            Node.new(NodeKind.FUNCTION, "app::foo").with_source(str(src), 1, 1)
        )
        graph.add_edge(file_node, func_node, Edge.extracted(EdgeKind.IMPORTS))
        result = per_file_tokens(graph)
        assert str(src) in result
        tokens = result[str(src)]
        assert tokens.graph > 0
        assert tokens.raw > 0


# ── cache ────────────────────────────────────────────────────────────


class TestCache:
    """Tests for the cache module."""

    def test_cache_stats_empty(self) -> None:
        from graphician.interfaces.transport.cache import cache_stats

        stats = cache_stats()
        assert "cache_size" in stats

    def test_cache_clear(self) -> None:
        from graphician.interfaces.transport.cache import clear_cache

        clear_cache()  # Should not raise

    def test_fingerprint_equality(self) -> None:
        from graphician.interfaces.transport.cache import _DbFingerprint

        fp1 = _DbFingerprint(main_mtime=1.0, main_len=1000)
        fp2 = _DbFingerprint(main_mtime=1.0, main_len=1000)
        fp3 = _DbFingerprint(main_mtime=2.0, main_len=2000)

        assert fp1 == fp2
        assert fp1 != fp3


# ── temporal ─────────────────────────────────────────────────────────


class TestTemporal:
    """Tests for temporal diff and time-travel queries."""

    def test_differential_json_basic(self) -> None:
        from graphician.interfaces.cli.response.temporal import differential_json

        graph = Graph()
        result = differential_json(graph, "HEAD~1", "HEAD")
        assert result["operation"] == "differential"

    def test_differential_json_no_temporal_data(self) -> None:
        from graphician.interfaces.cli.response.temporal import differential_json

        graph = Graph()
        result = differential_json(graph, "base", "head")
        assert result["operation"] == "differential"
        assert result["added_nodes"] == []


# ── snapshot_diff ───────────────────────────────────────────────────


class TestSnapshotDiff:
    """Tests for snapshot-to-snapshot graph diffs."""

    def test_snapshot_diff_json_invalid_paths(self) -> None:
        from graphician.interfaces.cli.response.snapshot_diff import snapshot_diff_json

        result = snapshot_diff_json("/nonexistent/a.db", "/nonexistent/b.db")
        assert result["operation"] == "snapshot_diff"
        assert "error" in result


# ── transport http ──────────────────────────────────────────────────


class TestHttpTransport:
    """Tests for HTTP transport layer."""

    def test_http_handler_class_exists(self) -> None:
        from graphician.interfaces.transport.http import GraphicianHTTPHandler

        assert hasattr(GraphicianHTTPHandler, "do_GET")
        assert GraphicianHTTPHandler.graph_db is None


# ── analysis response helpers ───────────────────────────────────────


class TestAnalysisResponseHelpers:
    """Tests for analysis response utility functions."""

    def test_diagnostics_json_raises_on_missing_db(self) -> None:
        import sqlite3

        from graphician.interfaces.cli.response.analysis import diagnostics_json

        with pytest.raises(sqlite3.OperationalError):
            diagnostics_json("/nonexistent/db.sqlite")

    def test_gaps_json_empty_graph(self) -> None:
        from graphician.interfaces.cli.response.analysis import gaps_json

        graph = Graph()
        result = gaps_json(graph, 10)
        assert result["operation"] == "gaps"
        assert result["total"] == 0

    def test_bridge_nodes_empty_graph(self) -> None:
        from graphician.interfaces.cli.response.analysis import bridge_nodes_json

        graph = Graph()
        result = bridge_nodes_json(graph, 10)
        assert result["operation"] == "bridge_nodes"

    def test_cycles_json_empty_graph(self) -> None:
        from graphician.interfaces.cli.response.analysis import cycles_json

        graph = Graph()
        result = cycles_json(graph, 10)
        assert result["operation"] == "cycles"
        assert result["hits"] == []

    def test_core_json_empty_graph(self) -> None:
        from graphician.interfaces.cli.response.analysis import core_json

        graph = Graph()
        result = core_json(graph, 10)
        assert result["operation"] == "core"

    def test_articulation_json_empty_graph(self) -> None:
        from graphician.interfaces.cli.response.analysis import articulation_json

        graph = Graph()
        result = articulation_json(graph, 10)
        assert result["operation"] == "articulation_points"

    def test_surprises_json_empty_graph(self) -> None:
        from graphician.interfaces.cli.response.analysis import surprises_json

        graph = Graph()
        result = surprises_json(graph, 10)
        assert result["operation"] == "surprises"

    def test_large_functions_json_empty_graph(self) -> None:
        from graphician.interfaces.cli.response.analysis import large_functions_json

        graph = Graph()
        result = large_functions_json(graph, 50, 10)
        assert result["operation"] == "large_functions"
