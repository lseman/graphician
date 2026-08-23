"""rust-analyzer adapter using the standard LSP call-hierarchy API."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, cast
from urllib.parse import unquote, urlparse

from ..core.graph import Graph
from ..core.id import NodeId
from ..core.node import Node, NodeKind
from .compiler import (
    COMPILER_EVIDENCE_VERSION,
    CompilerEdgeEvidence,
    CompilerEnrichmentReport,
    CompilerEvidenceFile,
    apply_compiler_evidence,
)
from ..core.edge import EdgeKind


@dataclass(frozen=True)
class RustAnalyzerOptions:
    binary: Path = Path("rust-analyzer")
    max_symbols: int = 2_000


@dataclass
class RustAnalyzerReport:
    symbols_queried: int = 0
    compiler_edges: int = 0
    adapter_unresolved: int = 0
    enrichment: CompilerEnrichmentReport = field(default_factory=CompilerEnrichmentReport)


@dataclass(frozen=True)
class _RustSymbol:
    qualified_name: str
    path: Path
    line: int
    character: int


def enrich_with_rust_analyzer(
    graph: Graph,
    root: Path,
    options: RustAnalyzerOptions | None = None,
) -> RustAnalyzerReport:
    """Query outgoing calls from rust-analyzer and apply resolved edges."""
    options = options or RustAnalyzerOptions()
    if options.max_symbols < 0:
        raise ValueError("max_symbols must be non-negative")
    root = root.resolve(strict=True)
    symbols = _rust_symbols(graph, root, options.max_symbols)
    evidence_edges: list[CompilerEdgeEvidence] = []
    unresolved = 0
    seen: set[tuple[str, str]] = set()
    with _LspClient.start(options.binary, root) as client:
        for symbol in symbols:
            prepared = client.request(
                "textDocument/prepareCallHierarchy",
                {
                    "textDocument": {"uri": symbol.path.as_uri()},
                    "position": {"line": symbol.line, "character": symbol.character},
                },
            )
            if not isinstance(prepared, list) or not prepared:
                continue
            outgoing = client.request("callHierarchy/outgoingCalls", {"item": prepared[0]})
            if not isinstance(outgoing, list):
                continue
            for call in outgoing:
                target = _resolve_call_item(graph, root, call.get("to", {}))
                target_node = graph.node(target) if target is not None else None
                if target_node is None:
                    unresolved += 1
                    continue
                key = (symbol.qualified_name, target_node.qualified_name)
                if key in seen:
                    continue
                seen.add(key)
                evidence_edges.append(
                    CompilerEdgeEvidence(
                        key[0], key[1], EdgeKind.CALLS, "LSP callHierarchy/outgoingCalls"
                    )
                )
    evidence = CompilerEvidenceFile(
        COMPILER_EVIDENCE_VERSION, "rust-analyzer-lsp", tuple(evidence_edges)
    )
    enrichment = apply_compiler_evidence(graph, evidence)
    return RustAnalyzerReport(
        symbols_queried=len(symbols),
        compiler_edges=len(evidence_edges),
        adapter_unresolved=unresolved,
        enrichment=enrichment,
    )


def _rust_symbols(graph: Graph, root: Path, limit: int) -> list[_RustSymbol]:
    symbols: list[_RustSymbol] = []
    for _, node in graph.nodes():
        if node.kind not in (NodeKind.FUNCTION, NodeKind.METHOD):
            continue
        if node.source_uri is None or not node.source_uri.endswith(".rs"):
            continue
        path = Path(node.source_uri)
        path = path if path.is_absolute() else root / path
        line = max((node.line_start or 1) - 1, 0)
        symbols.append(
            _RustSymbol(node.qualified_name, path, line, _symbol_character(path, line, node.name))
        )
        if len(symbols) >= limit:
            break
    return symbols


def _symbol_character(path: Path, line_number: int, name: str) -> int:
    try:
        line = path.read_text(encoding="utf-8").splitlines()[line_number]
    except (OSError, IndexError, UnicodeError):
        return 0
    column = line.find(name)
    return max(column, 0)


def _resolve_call_item(graph: Graph, root: Path, item: Any) -> NodeId | None:
    if not isinstance(item, dict):
        return None
    name = item.get("name")
    uri = item.get("uri")
    location_range = item.get("selectionRange") or item.get("range") or {}
    if not isinstance(location_range, dict):
        return None
    start = location_range.get("start", {})
    if not isinstance(name, str) or not isinstance(uri, str) or not isinstance(start, dict):
        return None
    line_value = start.get("line")
    if not isinstance(line_value, int):
        return None
    path = _file_uri_to_path(uri)
    if path is None:
        return None
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    line = line_value + 1
    candidates: list[tuple[int, NodeId]] = []
    for node_id, node in graph.nodes():
        if node.name != name or node.source_uri is None:
            continue
        if not _source_matches(node.source_uri, relative, path):
            continue
        if node.line_start is not None and node.line_start > line:
            continue
        if node.line_end is not None and node.line_end < line:
            continue
        candidates.append((line - (node.line_start or line), node_id))
    return min(candidates, default=(0, None), key=lambda item: item[0])[1]


def _source_matches(source: str, relative: Path, absolute: Path) -> bool:
    normalized = source.replace("\\", "/")
    relative_value = relative.as_posix()
    absolute_value = absolute.as_posix()
    return normalized in (relative_value, absolute_value) or absolute_value.endswith(f"/{normalized}")


def _file_uri_to_path(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return None
    path = unquote(parsed.path)
    if parsed.netloc:
        path = f"//{parsed.netloc}{path}"
    return Path(path)


class _LspClient:
    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("rust-analyzer pipes unavailable")
        self.process = process
        self.stdin = cast(BinaryIO, process.stdin)
        self.stdout = cast(BinaryIO, process.stdout)
        self.next_id = 1
        self._closed = False

    @classmethod
    def start(cls, binary: Path, root: Path) -> _LspClient:
        try:
            process = subprocess.Popen(
                [str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE
            )
        except OSError as exc:
            raise RuntimeError(
                f"failed to start {binary}; install rust-analyzer or pass --binary"
            ) from exc
        client = cls(process)
        client.request(
            "initialize",
            {
                "processId": None,
                "rootUri": root.as_uri(),
                "capabilities": {"textDocument": {"callHierarchy": {}}},
                "workspaceFolders": [{"uri": root.as_uri(), "name": root.name or "workspace"}],
            },
        )
        client.notify("initialized", {})
        return client

    def request(self, method: str, params: Any) -> Any:
        request_id = self.next_id
        self.next_id += 1
        self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while True:
            message = _read_message(self.stdout)
            if message.get("id") == request_id:
                if "error" in message:
                    raise RuntimeError(f"rust-analyzer {method} failed: {message['error']}")
                return message.get("result")
            if "method" in message and "id" in message:
                self._write({"jsonrpc": "2.0", "id": message["id"], "result": None})

    def notify(self, method: str, params: Any) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def _write(self, message: dict[str, Any]) -> None:
        body = json.dumps(message, separators=(",", ":")).encode()
        self.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
        self.stdin.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.request("shutdown", None)
            self.notify("exit", None)
            self.process.wait(timeout=10)
        except (RuntimeError, OSError, subprocess.TimeoutExpired):
            self.process.kill()
            self.process.wait()

    def __enter__(self) -> _LspClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _read_message(stream: BinaryIO) -> dict[str, Any]:
    content_length: int | None = None
    while True:
        line = stream.readline()
        if not line:
            raise RuntimeError("rust-analyzer closed the LSP stream")
        if line in (b"\r\n", b"\n"):
            break
        name, separator, value = line.partition(b":")
        if separator and name.lower() == b"content-length":
            try:
                content_length = int(value.strip())
            except ValueError as exc:
                raise RuntimeError("invalid LSP Content-Length") from exc
    if content_length is None:
        raise RuntimeError("LSP response omitted Content-Length")
    body = stream.read(content_length)
    if len(body) != content_length:
        raise RuntimeError("truncated LSP response")
    value = json.loads(body)
    if not isinstance(value, dict):
        raise RuntimeError("LSP response must be an object")
    return value
