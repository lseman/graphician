"""HTTP server for the interactive graph explorer.

Serves the D3-based graph visualization with API endpoints:
- GET / — index.html
- GET /app.js — application JavaScript
- GET /style.css — stylesheet
- GET /api/graph — paginated graph data (nodes + edges)
- GET /api/search — search results

Based on the Rust reference implementation.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Static assets ──────────────────────────────────────────────────

STATIC_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "static"


def _read_static(name: str) -> str:
    """Read a static file from the static directory."""
    path = STATIC_DIR / name
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


INDEX_HTML = _read_static("index.html")
APP_JS = _read_static("app.js")
STYLE_CSS = _read_static("style.css")


# ── HTTP handling ──────────────────────────────────────────────────

CONTENT_TYPES: dict[str, str] = {
    "/": "text/html; charset=utf-8",
    "/app.js": "application/javascript; charset=utf-8",
    "/style.css": "text/css; charset=utf-8",
    "/api/graph": "application/json",
    "/api/search": "application/json",
}


def _write_response(stream: Any, content_type: str, body: str) -> None:
    """Write an HTTP response to a socket."""
    body_bytes = body.encode("utf-8")
    response = (
        f"HTTP/1.1 200 OK\r\n"
        f"Content-Type: {content_type}\r\n"
        f"Content-Length: {len(body_bytes)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    )
    stream.sendall(response.encode("utf-8"))
    stream.sendall(body_bytes)


def _write_not_found(stream: Any) -> None:
    """Write a 404 response."""
    body = "not found"
    body_bytes = body.encode("utf-8")
    response = (
        f"HTTP/1.1 404 Not Found\r\n"
        f"Content-Type: text/plain\r\n"
        f"Content-Length: {len(body_bytes)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    )
    stream.sendall(response.encode("utf-8"))
    stream.sendall(body_bytes)


def _parse_request(stream: Any) -> tuple[str, str]:
    """Parse an HTTP request. Returns (method, path)."""
    buf = stream.recv(4096)
    if not buf:
        return "", ""
    request = buf.decode("utf-8", errors="replace")
    first_line = request.split("\r\n")[0]
    parts = first_line.split(" ")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", "/"


def _query_param(path: str, name: str) -> str | None:
    """Extract a query parameter from the request path."""
    query_start = path.find("?")
    if query_start == -1:
        return None
    query = path[query_start + 1:]
    for pair in query.split("&"):
        if "=" in pair:
            key, value = pair.split("=", 1)
            if key == name:
                # URL decode
                value = value.replace("+", " ")
                parts = value.split("%")
                result = parts[0]
                for part in parts[1:]:
                    if len(part) >= 2:
                        try:
                            result += chr(int(part[:2], 16))
                            result += part[2:]
                        except ValueError:
                            result += "%" + part
                    else:
                        result += "%" + part
                return result
    return None


def _query_usize(path: str, name: str) -> int:
    """Parse a usize query parameter with default."""
    val = _query_param(path, name)
    if val is None:
        return 0
    try:
        return int(val)
    except ValueError:
        return 0


def _handle_request(
    stream: Any,
    db_path: str,
    algorithm: str,
) -> None:
    """Handle a single HTTP request."""
    method, path = _parse_request(stream)
    if not method:
        return

    if path == "/":
        _write_response(stream, CONTENT_TYPES["/"], INDEX_HTML)
    elif path == "/app.js":
        _write_response(stream, CONTENT_TYPES["/app.js"], APP_JS)
    elif path == "/style.css":
        _write_response(stream, CONTENT_TYPES["/style.css"], STYLE_CSS)
    elif path.startswith("/api/graph"):
        body = _graph_json(db_path, algorithm, path)
        _write_response(stream, CONTENT_TYPES["/api/graph"], body)
    elif path.startswith("/api/search"):
        q = _query_param(path, "q") or ""
        body = _search_json(db_path, q, path)
        _write_response(stream, CONTENT_TYPES["/api/search"], body)
    else:
        _write_not_found(stream)


def _graph_json(db_path: str, algorithm: str, request_path: str) -> str:
    """Generate graph data JSON for the explorer."""
    from ...analysis.communities import detect_communities
    from ...persistence.store import GraphStore

    store = GraphStore(db_path)
    graph = store.load_graph()

    node_offset = _query_usize(request_path, "offset")
    node_limit = max(1, min(5000, _query_usize(request_path, "limit") or 1000))
    edge_offset = _query_usize(request_path, "edge_offset")
    edge_limit = max(1, min(10000, (_query_usize(request_path, "edge_limit") or node_limit * 2)))

    # Community detection
    communities = detect_communities(graph, algorithm)
    community_map: dict[int, int] = {}
    for comm in communities.get("communities", []):
        for node_id in comm.get("node_ids", []):
            community_map[node_id] = comm.get("id", 0)

    all_nodes = []
    for nid, node in graph.nodes():
        in_deg = sum(1 for _ in graph.in_neighbors(nid))
        out_deg = sum(1 for _ in graph.out_neighbors(nid))
        all_nodes.append({
            "id": nid.value,
            "label": node.name,
            "qname": node.qualified_name,
            "kind": node.kind.value,
            "source": node.source_uri,
            "degree": in_deg + out_deg,
            "community": community_map.get(nid.value, 0),
        })

    all_edges = []
    for _, src, dst, edge in graph.edges():
        all_edges.append({
            "source": src.value,
            "target": dst.value,
            "kind": edge.kind.value,
            "confidence": edge.confidence.score(),
        })

    total_nodes = len(all_nodes)
    total_edges = len(all_edges)
    nodes = all_nodes[node_offset:node_offset + node_limit]
    edges = all_edges[edge_offset:edge_offset + edge_limit]

    result = {
        "nodes": nodes,
        "links": edges,
        "graph_summary": {
            "node_count": total_nodes,
            "edge_count": total_edges,
        },
        "guardrails": {
            "nodes": {
                "offset": node_offset,
                "limit": node_limit,
                "returned": len(nodes),
                "total": total_nodes,
                "has_more": node_offset + len(nodes) < total_nodes,
            },
            "links": {
                "offset": edge_offset,
                "limit": edge_limit,
                "returned": len(edges),
                "total": total_edges,
                "has_more": edge_offset + len(edges) < total_edges,
            },
        },
    }

    store.close()
    return json.dumps(result, indent=2)


def _search_json(db_path: str, query: str, request_path: str) -> str:
    """Generate search results JSON for the explorer."""
    from ...analysis.search import ranked_search
    from ...persistence.store import GraphStore

    store = GraphStore(db_path)
    graph = store.load_graph()

    offset = _query_usize(request_path, "offset")
    limit = max(1, min(100, _query_usize(request_path, "limit") or 20))

    hits = ranked_search(graph, query, offset + limit)
    results = []
    for hit in hits:
        node = graph.node(hit.id)
        if node is None:
            continue
        results.append({
            "id": hit.id.value,
            "score": hit.score,
            "label": node.name,
            "qname": node.qualified_name,
            "kind": node.kind.value,
            "signals": hit.signals,
        })

    total = len(results)
    page = results[offset:offset + limit]

    result = {
        "hits": page,
        "graph_summary": {
            "node_count": graph.node_count(),
            "edge_count": graph.edge_count(),
        },
        "guardrails": {
            "hits": {
                "offset": offset,
                "limit": limit,
                "returned": len(page),
                "total": total,
                "has_more": offset + len(page) < total,
            },
        },
    }

    store.close()
    return json.dumps(result, indent=2)


# ── CLI command ────────────────────────────────────────────────────


def cmd_serve(db_path: str, bind: str = "127.0.0.1:8080", algorithm: str = "louvain") -> None:
    """Serve the interactive graph explorer.

    Args:
        db_path: Path to the Graphician SQLite database.
        bind: Address:port to bind to (default: 127.0.0.1:8080).
        algorithm: Community detection algorithm (louvain, leiden, infomap).
    """
    import socket

    db = Path(db_path).resolve()
    if not db.exists():
        print(f"Error: database {db} does not exist", file=sys.stderr)
        sys.exit(1)

    # Parse bind address
    if ":" in bind:
        host, port = bind.rsplit(":", 1)
        port = int(port)
    else:
        host = "127.0.0.1"
        port = int(bind)

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen(5)

    print(f"Graphician graph explorer listening on http://{host}:{port}")

    try:
        while True:
            try:
                stream, _ = listener.accept()
                _handle_request(stream, str(db), algorithm)
                stream.close()
            except Exception as e:  # noqa: BLE001 -- one failed request must not kill the server loop
                logger.warning("request failed: %s", e)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        listener.close()
