"""Tests for the persistent JSONL worker (``graphician worker``).

Covers the worker contract that the runtime's JsonlWorker relies on:
in-process query serving, background (non-blocking) index refresh with
dedupe, build-state reporting, fingerprint-driven graph reload after a
refresh, and protocol error handling.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from graphician.core import Edge, EdgeKind, Graph, Node, NodeKind
from graphician.persistence.store import GraphStore

_ids = itertools.count()


def _fixture_db(path: Path) -> Path:
    """Persist a small hand-rolled graph to ``path``."""
    graph = Graph()
    alpha = graph.add_node(Node.new(NodeKind.FUNCTION, "app::alpha"))
    beta = graph.add_node(Node.new(NodeKind.FUNCTION, "app::beta"))
    graph.add_edge(alpha, beta, Edge.extracted(EdgeKind.CALLS))
    with GraphStore(path) as store:
        store.save_graph(graph, {"src/app.py": "hash"})
    return path


def _tiny_project(root: Path, files: int) -> None:
    """Create a throwaway Python project large enough for a multi-second build."""
    src = root / "src"
    src.mkdir(parents=True)
    for i in range(files):
        body = "\n".join(f"def f{i}_{j}():\n    return {j}" for j in range(20))
        (src / f"mod{i:03d}.py").write_text(body, encoding="utf-8")


class _Worker:
    """Thin JSONL client over a spawned ``graphician worker`` process."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "graphician", "worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def request(self, method: str, **fields: Any) -> dict[str, Any]:
        line = json.dumps({"id": f"t{next(_ids)}", "method": method, **fields})
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        assert self.proc.stdout is not None
        return json.loads(self.proc.stdout.readline())

    def wait_idle(self, timeout: float = 300.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            build = self.request("build_status")["build"]
            if build["state"] == "idle":
                return build
            time.sleep(0.5)
        raise AssertionError(f"refresh did not finish within {timeout}s")

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def test_worker_serves_queries_in_process(tmp_path: Path) -> None:
    db = _fixture_db(tmp_path / "graph.db")
    worker = _Worker()
    try:
        assert worker.request("ping")["ok"] is True
        response = worker.request("query", db=str(db), operation="status", params={})
        assert response["ok"] is True
        assert response["result"]["nodes"] == 2
        assert response["build"]["state"] == "idle"
    finally:
        worker.close()


def test_refresh_never_blocks_queries_and_reloads_graph(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _tiny_project(project, 600)
    db = tmp_path / "graph.db"
    worker = _Worker()
    try:
        response = worker.request(
            "refresh", db=str(db), root=str(project)
        )
        assert response["ok"] is True
        assert response["started"] is True

        # A query must answer while the refresh is still running.
        query = worker.request(
            "query", db=str(db), operation="status", params={}
        )
        assert query["ok"] is True
        assert query["build"]["state"] == "running"

        build = worker.wait_idle()
        assert build["last_exit"] == 0

        # The fingerprint cache must have reloaded the refreshed graph.
        after = worker.request("query", db=str(db), operation="status", params={})
        assert after["result"]["nodes"] > 0
    finally:
        worker.close()


def test_refresh_dedupes_while_running(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _tiny_project(project, 600)
    db = tmp_path / "graph.db"
    worker = _Worker()
    try:
        first = worker.request("refresh", db=str(db), root=str(project))
        assert first["started"] is True
        second = worker.request("refresh", db=str(db), root=str(project))
        assert second["ok"] is True
        assert second["started"] is False
        assert second["build"]["state"] == "running"
        worker.wait_idle()
    finally:
        worker.close()


def test_protocol_errors_do_not_kill_worker(tmp_path: Path) -> None:
    db = _fixture_db(tmp_path / "graph.db")
    worker = _Worker()
    try:
        unknown = worker.request("nope")
        assert unknown["ok"] is False
        assert "unknown method" in unknown["error"]
        bad = worker.request("query", db="", operation="status", params={})
        assert bad["ok"] is False
        # The worker survives protocol errors.
        assert worker.request("ping")["ok"] is True
        response = worker.request(
            "query", db=str(db), operation="status", params={}
        )
        assert response["ok"] is True
        assert response["result"]["nodes"] == 2
    finally:
        worker.close()
