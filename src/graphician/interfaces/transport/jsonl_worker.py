"""Persistent JSONL worker transport for Graphician.

Serves ``query`` requests in-process (holding a process-lifetime graph
cache keyed by DB fingerprint) and runs index refreshes in a child
process so they never block query handling — the worker answers a query
against the current graph immediately while a refresh runs in the
background, and the next query after the refresh completes picks up the
fresh graph automatically (fingerprint cache miss).

Protocol: one JSON object per line on stdin/stdout. Requests carry
``id`` + ``method``; responses carry ``id`` + ``ok`` — the same contract
as Legroom's ``sdk_worker``, consumed by the runtime's ``JsonlWorker``.

Methods:
- ``ping``         → liveness check.
- ``query``        → ``{db, operation, params}`` → ``{result, build}``.
- ``refresh``      → ``{db, root}`` → starts at most one background
  ``graphician build`` child process → ``{started, build}``.
- ``build_status`` → ``{build}`` (state of the background refresh).
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Any, TextIO

from ..cli.response import tool_response_cached

logger = logging.getLogger(__name__)

_DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
_MAX_BUILD_SECONDS = 3600
_MAX_BUILD_OUTPUT = 4096


class RequestTimeoutError(Exception):
    """A single request exceeded the worker's wall-clock budget."""


def _default_request_timeout() -> float:
    try:
        value = float(
            os.environ.get(
                "GRAPHICIAN_WORKER_TIMEOUT", _DEFAULT_REQUEST_TIMEOUT_SECONDS
            )
        )
    except ValueError:
        return _DEFAULT_REQUEST_TIMEOUT_SECONDS
    return value if value >= 0 else _DEFAULT_REQUEST_TIMEOUT_SECONDS


def _on_main_thread() -> bool:
    return threading.current_thread() is threading.main_thread()


def _alarm_handler(signum: int, frame: Any) -> None:
    raise RequestTimeoutError()


# ── Background refresh state ───────────────────────────────────────────────


class _BuildState:
    """State of the index refresh child process (at most one at a time)."""

    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        self.state = "idle"
        self.root: str | None = None
        self.started_at: float | None = None
        self.last_exit: int | None = None
        self.last_finished_at: float | None = None
        self.last_output: str | None = None

    @property
    def running(self) -> bool:
        return (
            self.state == "running"
            and self.process is not None
            and self.process.poll() is None
        )

    def snapshot(self) -> dict[str, Any]:
        finished = self.last_finished_at is not None
        return {
            "state": self.state,
            "root": self.root,
            "last_exit": self.last_exit if finished else None,
            "last_output": self.last_output if finished else None,
        }


_build = _BuildState()


def _reap(proc: subprocess.Popen[str]) -> None:
    """Wait for a refresh child and record its outcome."""
    try:
        output, _ = proc.communicate(timeout=_MAX_BUILD_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
        output, _ = proc.communicate()
    _build.state = "idle"
    _build.last_exit = proc.returncode
    _build.last_finished_at = time.monotonic()
    _build.last_output = (output or "").strip()[-_MAX_BUILD_OUTPUT:]
    _build.process = None
    logger.info(
        "Index refresh for %s exited with %s", _build.root, proc.returncode
    )


def _spawn_refresh(db: str, root: str) -> bool:
    """Start a background ``graphician build``; False if one is running.

    ``build`` is smart: full build on an empty DB, incremental when file
    hashes already exist — one command covers both cases.
    """
    if _build.running:
        return False
    command = [sys.executable, "-m", "graphician", "--db", db, "build", root]
    proc = subprocess.Popen(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _build.process = proc
    _build.state = "running"
    _build.root = root
    _build.started_at = time.monotonic()
    threading.Thread(target=_reap, args=(proc,), daemon=True).start()
    return True


# ── Request handlers ───────────────────────────────────────────────────────


def _handle_ping(request: dict[str, Any]) -> dict[str, Any]:
    return {"id": request.get("id"), "ok": True, "pong": True}


def _handle_query(request: dict[str, Any]) -> dict[str, Any]:
    db = request.get("db")
    operation = request.get("operation")
    params = request.get("params") or {}
    if not isinstance(db, str) or not db:
        raise ValueError("query.db must be a non-empty string")
    if not isinstance(operation, str) or not operation:
        raise ValueError("query.operation must be a non-empty string")
    if not isinstance(params, dict):
        raise ValueError("query.params must be an object")
    result = tool_response_cached(db, operation.replace("-", "_"), params)
    return {
        "id": request.get("id"),
        "ok": True,
        "result": result,
        "build": _build.snapshot(),
    }


def _handle_refresh(request: dict[str, Any]) -> dict[str, Any]:
    db = request.get("db")
    root = request.get("root") or "."
    if not isinstance(db, str) or not db:
        raise ValueError("refresh.db must be a non-empty string")
    if not isinstance(root, str) or not root:
        raise ValueError("refresh.root must be a non-empty string")
    try:
        started = _spawn_refresh(db, root)
    except OSError as error:
        raise ValueError(f"failed to start index refresh: {error}") from error
    return {
        "id": request.get("id"),
        "ok": True,
        "started": started,
        "build": _build.snapshot(),
    }


def _handle_build_status(request: dict[str, Any]) -> dict[str, Any]:
    return {"id": request.get("id"), "ok": True, "build": _build.snapshot()}


# ── Dispatch table ─────────────────────────────────────────────────────────

_METHOD_HANDLERS: dict[str, Any] = {
    "ping": _handle_ping,
    "query": _handle_query,
    "refresh": _handle_refresh,
    "build_status": _handle_build_status,
}


def _response(request: object) -> dict[str, Any]:
    """Parse and dispatch a single JSONL request."""
    if not isinstance(request, dict):
        raise TypeError("request must be a JSON object")

    request_id = request.get("id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request.id must be a non-empty string")

    method = request.get("method")
    if not isinstance(method, str) or not method:
        raise ValueError("request.method must be a non-empty string")

    handler = _METHOD_HANDLERS.get(method)
    if handler is None:
        raise ValueError(
            f"unknown method: {method!r}. "
            f"Supported: {', '.join(sorted(_METHOD_HANDLERS))}"
        )

    return handler(request)


# ── Serve loop ─────────────────────────────────────────────────────────────


def serve(
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
    request_timeout: float | None = None,
) -> None:
    """Serve requests until stdin reaches EOF.

    Each request is bounded by ``request_timeout`` wall-clock seconds
    (default: 30, override with ``GRAPHICIAN_WORKER_TIMEOUT``). A request
    that exceeds the budget gets an error response instead of wedging the
    worker. Refresh requests return immediately (the build runs in a
    child process) and are never the ones that hit the budget.
    """
    if request_timeout is None:
        request_timeout = _default_request_timeout()
    timeout_active = False
    if request_timeout > 0 and _on_main_thread():
        try:
            signal.signal(signal.SIGALRM, _alarm_handler)
            timeout_active = True
        except (OSError, ValueError):
            timeout_active = False

    try:
        for line in input_stream:
            if not line.strip():
                continue
            request_id: object = None
            started = time.monotonic()
            if timeout_active:
                signal.setitimer(signal.ITIMER_REAL, request_timeout)
            try:
                request = json.loads(line)
                if isinstance(request, dict):
                    request_id = request.get("id")
                response = _response(request)
            except RequestTimeoutError:
                elapsed = time.monotonic() - started
                logger.error(
                    "Request %s timed out after %.1fs (budget %.0fs); "
                    "returning error and staying alive",
                    request_id,
                    elapsed,
                    request_timeout,
                )
                response = {
                    "id": request_id,
                    "ok": False,
                    "error": f"request timed out after {request_timeout:.0f}s",
                }
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                response = {"id": request_id, "ok": False, "error": str(error)}
            except Exception as error:  # noqa: BLE001
                response = {"id": request_id, "ok": False, "error": f"worker error: {error}"}
            finally:
                if timeout_active:
                    try:
                        previous = signal.signal(signal.SIGALRM, signal.SIG_IGN)
                        signal.setitimer(signal.ITIMER_REAL, 0)
                    finally:
                        signal.signal(signal.SIGALRM, previous)
            output_stream.write(json.dumps(response, separators=(",", ":")) + "\n")
            output_stream.flush()
    finally:
        # Do not orphan a running refresh when the worker goes away.
        if _build.running and _build.process is not None:
            _build.process.terminate()
