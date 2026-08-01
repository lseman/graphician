"""HTTP transport for Ariadne graph API.

Provides a lightweight HTTP server that serves the static dashboard
and exposes JSON APIs for graph data and search.

Mirrors the Rust ``http.rs`` transport module.
"""

from __future__ import annotations

import json
import os
import re
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any

# Embed static assets from the static directory
_STATIC_DIR = Path(__file__).parent.parent.parent.parent / "static"


class AriadneHTTPHandler(SimpleHTTPRequestHandler):
    """HTTP handler for the Ariadne graph API."""

    graph_db: str | None = None
    algorithm: str = "leiden"  # leiden | louvain | infomap

    def do_GET(self) -> None:
        """Handle GET requests."""
        path = self.path.split("?")[0]

        if path == "/" or path == "":
            self._serve_static("index.html", "text/html; charset=utf-8")
        elif path == "/app.js":
            self._serve_static("app.js", "application/javascript; charset=utf-8")
        elif path == "/style.css":
            self._serve_static("style.css", "text/css; charset=utf-8")
        elif path.startswith("/api/graph"):
            self._serve_graph(path)
        elif path.startswith("/api/search"):
            self._serve_search(path)
        else:
            self._send_error(404, "not found")

    def _serve_static(self, filename: str, content_type: str) -> None:
        """Serve a static file from the static directory."""
        filepath = _STATIC_DIR / filename
        if not filepath.exists():
            self._send_error(404, f"static file not found: {filename}")
            return

        try:
            content = filepath.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(content)
        except OSError as e:
            self._send_error(500, str(e))

    def _serve_graph(self, request_path: str) -> None:
        """Serve graph data with optional community algorithm."""
        if not self.graph_db:
            self._send_error(503, "no database configured")
            return

        # Parse query parameters
        query_string = request_path.split("?", 1)[1] if "?" in request_path else ""
        params = dict(pair.split("=", 1) for pair in query_string.split("&") if "=" in pair)

        node_offset = int(params.get("offset", "0"))
        node_limit = min(int(params.get("limit", "1000")), 5000)
        edge_offset = int(params.get("edge_offset", "0"))
        edge_limit = min(int(params.get("edge_limit", str(node_limit * 2))), 10000)

        try:
            # Import and load graph
            from ...persistence.store import GraphStore
            from ..cli.response import architecture_overview_json

            store = GraphStore(self.graph_db)
            graph = store.load_graph()

            # Get community assignments
            communities = self._get_communities(graph)

            # Build node list
            all_nodes: list[dict[str, Any]] = []
            all_edges: list[dict[str, Any]] = []

            for nid, node in graph.nodes():
                degree = sum(1 for _ in graph.in_neighbors(nid)) + sum(
                    1 for _ in graph.out_neighbors(nid)
                )
                community = communities.get(nid.value, 0)
                all_nodes.append({
                    "id": nid.value,
                    "label": node.name,
                    "qname": node.qualified_name,
                    "kind": node.kind.value,
                    "source": node.source_uri,
                    "degree": degree,
                    "community": community,
                })

            for _, src, dst, edge in graph.edges():
                all_edges.append({
                    "source": src.value,
                    "target": dst.value,
                    "kind": edge.kind.value,
                    "confidence": edge.confidence.score(),
                })

            # Pagination
            nodes = all_nodes[node_offset : node_offset + node_limit]
            edges = all_edges[edge_offset : edge_offset + edge_limit]

            # Summary
            summary = architecture_overview_json(graph)

            response = {
                "nodes": nodes,
                "links": edges,
                "graph_summary": summary,
                "guardrails": {
                    "nodes": {
                        "offset": node_offset,
                        "limit": node_limit,
                        "returned": len(nodes),
                        "total": len(all_nodes),
                        "has_more": node_offset + len(nodes) < len(all_nodes),
                    },
                    "links": {
                        "offset": edge_offset,
                        "limit": edge_limit,
                        "returned": len(edges),
                        "total": len(all_edges),
                        "has_more": edge_offset + len(edges) < len(all_edges),
                    },
                },
            }

            self._send_json(200, response)

        except Exception as e:
            self._send_error(500, str(e))

    def _serve_search(self, request_path: str) -> None:
        """Serve search results."""
        if not self.graph_db:
            self._send_error(503, "no database configured")
            return

        # Parse query parameters
        query_string = request_path.split("?", 1)[1] if "?" in request_path else ""
        params = dict(pair.split("=", 1) for pair in query_string.split("&") if "=" in pair)

        query = params.get("q", "")
        offset = int(params.get("offset", "0"))
        limit = min(int(params.get("limit", "20")), 100)

        try:
            from ...persistence.store import GraphStore
            from ...analysis.search import ranked_search

            store = GraphStore(self.graph_db)
            graph = store.load_graph()

            # Search
            hits = ranked_search(graph, query, offset + limit)

            # Build response
            results: list[dict[str, Any]] = []
            for hit in hits:
                node = graph.node(hit.id)
                if node:
                    results.append({
                        "id": hit.id.value,
                        "score": hit.score,
                        "label": node.name,
                        "qname": node.qualified_name,
                        "kind": node.kind.value,
                        "signals": hit.reasons,
                    })

            page = results[offset : offset + limit]

            response = {
                "hits": page,
                "guardrails": {
                    "hits": {
                        "offset": offset,
                        "limit": limit,
                        "returned": len(page),
                        "total": len(results),
                        "has_more": offset + len(page) < len(results),
                    },
                },
            }

            self._send_json(200, response)

        except Exception as e:
            self._send_error(500, str(e))

    def _get_communities(self, graph: Graph) -> dict[int, int]:
        """Get community assignments for all nodes."""
        try:
            from ...analysis.communities import detect_communities

            result = detect_communities(graph, self.algorithm)
            communities: dict[int, int] = {}
            for comm in result.get("communities", []):
                for node_info in comm.get("nodes", []):
                    nid = graph.find_by_qname(node_info["qualified_name"])
                    if nid:
                        communities[nid.value] = comm["id"]
            return communities
        except Exception:
            return {}

    def _send_json(self, status: int, data: dict[str, Any]) -> None:
        """Send a JSON response."""
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str) -> None:
        """Send an error response."""
        body = f"{status} {message}".encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


def serve_http(host: str = "127.0.0.1", port: int = 8080, db: str | None = None) -> None:
    """Start the HTTP server.

    Args:
        host: Bind address.
        port: Bind port.
        db: Path to the graph database.
    """
    server = HTTPServer((host, port), AriadneHTTPHandler)
    server.graph_db = db
    server.algorithm = "leiden"
    print(f"Ariadne HTTP server running at http://{host}:{port}")
    print(f"  Graph database: {db or 'not configured'}")
    print("  Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()
