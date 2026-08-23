"""MCP (Model Context Protocol) stdio server.

Exposes the Graphician code graph as an MCP tool with:
- One tool: graphician (operation + params)
- Five prompt templates for structured agent workflows
- JSON-RPC 2.0 over stdio

Wired to the structured response system (``response/``) for:
- Hint generation (workflow suggestions)
- Response guardrails (pagination, hard limits)
- Graph summary insertion
- Caching (long-lived server mode)
"""

from __future__ import annotations

import json
import logging
import sys
from collections import deque
from pathlib import Path
from typing import Any

from ...core.graph import Graph
from ...core.edge import EdgeKind
from ...core.id import NodeId
from ...core.node import NodeKind
from ...persistence.store import GraphStore
from ...analysis.search import hybrid_search
from ...analysis.impact import compute_impact
from ...analysis.paths import find_paths
from ...analysis.communities import (
    detect_communities,
    find_bridge_nodes,
    find_hub_nodes,
    compute_centrality,
)
from ...analysis.structure import (
    find_cycles,
    find_articulation_points,
    find_god_nodes,
    find_large_functions,
    find_dead_code,
    find_counterfactual,
    find_motifs,
    compute_surprise_scoring,
    rename_preview,
    export_graphml,
)
from ...analysis.diff import graph_diff
from ...analysis.changes import detect_changes, compute_risk, compute_test_coverage
from ...analysis.context_pack import build_context_pack
from ...analysis.semsearch import EmbeddingIndex
from ...analysis.dedup import deduplicate_nodes, DedupOptions, DedupResult
from ...analysis.patterns import detect_patterns
from ..cli.response import tool_response, tool_response_cached

logger = logging.getLogger(__name__)


class GraphicianMCP:
    """MCP server for Graphician.

    Implements the MCP stdio protocol. Exposes one tool and five prompts.

    Uses process-lifetime caching for long-lived servers — the graph is
    loaded once per DB fingerprint and reused across tool calls.
    """

    def __init__(self, db_path: str = "graphician.db") -> None:
        self.db_path = db_path
        self.graph: Graph | None = None
        self.store: GraphStore | None = None
        self.embedding_index = EmbeddingIndex()
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy-load the graph using the process-lifetime cache."""
        if self._initialized:
            return
        if self.graph is not None:
            return
        from .cache import load_cached
        if self.store is None:
            self.store = GraphStore(self.db_path)
        self.graph = load_cached(self.db_path, self.store)
        self._initialized = True

    def run(self) -> None:
        """Run the MCP stdio server."""
        logger.info("Starting Graphician MCP server")
        self._initialized = False  # Reset for first connection

        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue

                request = json.loads(line)
                response = self._handle_request(request)
                self._send_response(response)

            except json.JSONDecodeError as e:
                self._send_error(-32700, f"Parse error: {e}")
            except Exception as e:
                logger.exception("MCP server error")
                self._send_error(-32603, f"Internal error: {e}")

    def _handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle a JSON-RPC request."""
        method = request.get("method", "")
        params = request.get("params", {})
        msg_id = request.get("id")

        if method == "initialize":
            return self._handle_initialize(msg_id)
        elif method == "tools/list":
            return self._handle_tools_list(msg_id)
        elif method == "tools/call":
            return self._handle_tool_call(params, msg_id)
        elif method == "prompts/list":
            return self._handle_prompts_list(msg_id)
        elif method == "prompts/get":
            return self._handle_prompt_get(params, msg_id)
        elif method == "notifications/initialized":
            return {"jsonrpc": "2.0", "result": {}}
        else:
            return self._send_error(-32601, f"Method not found: {method}", msg_id)

    def _handle_initialize(self, msg_id: Any) -> dict[str, Any]:
        """Handle initialization."""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": "graphician",
                    "version": "0.1.0",
                },
                "capabilities": {
                    "tools": {"listChanged": False},
                    "prompts": {"listChanged": False},
                },
            },
        }

    def _handle_tools_list(self, msg_id: Any) -> dict[str, Any]:
        """List available tools."""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "tools": [
                    {
                        "name": "graphician",
                        "description": (
                            "Query the Graphician codebase graph. "
                            "Operations: minimal_context, search, impact, traverse, paths, "
                            "callers_of, callees_of, flows, affected_flows, communities, "
                            "bridge_nodes, hub_nodes, god_nodes, cycles, articulation, "
                            "core, gaps, surprises, diagnostics, counterfactual, motifs, "
                            "suggested_questions, dedup, patterns, report, wiki, "
                            "rename_preview, find_related, export_graphml, health, "
                            "diagnostics, status, core, graph_diff, snapshot_diff, "
                            "differential, detect_changes, risk, review_context, "
                            "context_pack, token_savings, token_benchmark, large_functions, "
                            "community_split, dead_code, knowledge_gaps, blast_radius, "
                            "test_coverage, architecture, architecture_overview."
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "operation": {
                                    "type": "string",
                                    "description": "Operation name (e.g., minimal_context, search, impact)",
                                },
                                "params": {
                                    "type": "object",
                                    "description": "Operation-specific parameters",
                                },
                            },
                            "required": ["operation"],
                        },
                    }
                ],
            },
        }

    def _handle_tool_call(
        self,
        params: dict[str, Any],
        msg_id: Any,
    ) -> dict[str, Any]:
        """Handle a tool call.

        Uses the structured response system for hint generation,
        guardrails, and caching (in server mode).
        """
        self._ensure_initialized()
        if self.graph is None:
            return self._send_error(-32603, "Graph not loaded", msg_id)

        operation = params.get("operation", "")
        tool_params = params.get("params", {})

        try:
            # Use response system for operations it covers
            response_ops = {
                "status",
                "search",
                "context_pack",
                "impact",
                "detect_changes",
                "risk",
                "review_context",
                "traverse",
                "large_functions",
                "bridge_nodes",
                "cycles",
                "core",
                "articulation",
                "gaps",
                "surprises",
                "diagnostics",
                "counterfactual",
                "suggested_questions",
                "architecture_overview",
                "architecture",
                "god_nodes",
                "flows",
                "affected_flows",
                "blast_radius",
                "test_coverage",
                "hub_nodes",
                "community_split",
                "dead_code",
                "find_related",
                "minimal_context",
                "context",
                "export_graphml",
                "paths",
                "rename_preview",
                "snapshot_diff",
                "differential",
                "token_savings",
                "token_benchmark",
            }

            if operation in response_ops:
                # Use structured response system (with hints, guardrails, caching)
                use_cache = True  # Server mode: enable caching
                result = tool_response_cached(self.db_path, operation, tool_params)
            else:
                # Fallback to legacy operation handlers
                result = self._execute_operation(operation, tool_params)

            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2),
                        }
                    ],
                },
            }
        except Exception as e:
            logger.exception("Tool execution error")
            return self._send_error(-32603, str(e), msg_id)

    def _execute_operation(self, operation: str, params: dict[str, Any]) -> Any:
        """Execute a graph operation."""
        graph = self.graph
        if graph is None:
            raise RuntimeError("Graph not loaded")

        match operation:
            case "minimal_context" | "context":
                return self._op_minimal_context(graph, params)
            case "status":
                return self.store.status() if self.store else {}
            case "search":
                return hybrid_search(
                    graph,
                    params.get("query", ""),
                    limit=params.get("limit", 20),
                )
            case "impact":
                return compute_impact(
                    graph,
                    params.get("target", ""),
                    max_hops=params.get("max_hops", 4),
                    limit=params.get("limit", 25),
                )
            case "blast_radius":
                impact = compute_impact(
                    graph,
                    params.get("target", ""),
                    max_hops=params.get("max_hops", 4),
                    limit=params.get("limit", 25),
                )
                return {"target": params.get("target", ""), "blast_radius": impact}
            case "paths":
                return find_paths(
                    graph,
                    params.get("source", ""),
                    params.get("target", ""),
                    max_hops=params.get("max_hops", 6),
                )
            case "traverse":
                return self._op_traverse(graph, params)
            case "callers_of":
                return self._op_callers(graph, params)
            case "callees_of":
                return self._op_callees(graph, params)
            case "flows":
                return self._op_flows(graph, params)
            case "affected_flows":
                return self._op_affected_flows(graph, params)
            case "communities":
                return detect_communities(graph, params.get("algorithm", "louvain"))
            case "bridge_nodes" | "bridge-nodes":
                return find_bridge_nodes(graph)
            case "hub_nodes" | "hub-nodes":
                return find_hub_nodes(graph)
            case "god_nodes" | "god-nodes":
                return find_god_nodes(graph)
            case "cycles":
                return find_cycles(graph)
            case "articulation_points" | "articulation":
                return find_articulation_points(graph)
            case "core":
                return self._op_core(graph, params)
            case "gaps":
                return self._op_gaps(graph)
            case "diagnostics":
                return self._op_health(graph)
            case "surprise_scoring" | "surprises":
                return compute_surprise_scoring(graph)
            case "knowledge_gaps":
                return self._op_knowledge_gaps(graph)
            case "dead_code":
                return find_dead_code(graph)
            case "test_coverage":
                return compute_test_coverage(graph)
            case "suggested_questions":
                return self._op_suggested_questions(graph)
            case "architecture":
                return self._op_architecture(graph)
            case "large_functions":
                return find_large_functions(graph, params.get("min_lines", 30))
            case "counterfactual":
                return find_counterfactual(graph, params.get("target", ""))
            case "motifs":
                return find_motifs(graph, params.get("pattern", "diamond"))
            case "graph_diff":
                return self._op_graph_diff(graph, params)
            case "differential":
                return self._op_differential(graph, params)
            case "dedup":
                return self._op_dedup(graph)
            case "patterns":
                return {
                    "patterns": [
                        {
                            "pattern_id": match.pattern_id,
                            "display_name": match.display_name,
                            "framework": match.framework,
                            "category": match.category,
                            "confidence": match.confidence,
                            "matched_nodes": match.matched_nodes,
                        }
                        for match in detect_patterns(graph)
                    ]
                }
            case "wiki":
                return self._op_wiki(graph)
            case "report":
                return self._op_report(graph)
            case "rename_preview":
                return self._op_rename_preview(graph, params)
            case "find_related":
                return self._op_find_related(graph, params)
            case "export_graphml":
                return self._op_export_graphml(graph)
            case "health":
                return self._op_health(graph)
            case "detect_changes" | "detect-changes":
                diff = params.get("diff", "")
                return detect_changes(graph, diff, base=params.get("base"))
            case "risk":
                return compute_risk(graph, base=params.get("base"))
            case "review_context":
                return self._op_review_context(graph, params)
            case "context_pack":
                return build_context_pack(
                    graph,
                    params.get("query", ""),
                    intent=params.get("intent", "implementation"),
                    token_budget=params.get("token_budget", 2400),
                    max_items=params.get("max_items", 20),
                )
            case "token_savings" | "token-savings":
                return self._op_token_savings(graph)
            case "token_benchmark" | "token-benchmark":
                return self._op_token_benchmark(graph, params)
            case _:
                return {"error": f"Unknown operation: {operation}"}

    def _op_minimal_context(
        self,
        graph: Graph,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve a target and return bounded graph neighborhood."""
        target = params.get("target", "")
        mode = params.get("mode", "review")
        max_hops = params.get("max_hops", 2)
        include_graph_summary = params.get("include_graph_summary", False)

        nid = graph.find_by_qname(target)
        if nid is None:
            # Try fuzzy match
            results = hybrid_search(graph, target, limit=5)
            if results.get("results"):
                nid = graph.find_by_qname(results["results"][0]["qualified_name"])
            if nid is None:
                return {"error": f"Symbol not found: {target}"}

        node = graph.node(nid)
        if node is None:
            return {"error": f"Node not found: {target}"}

        # Build neighborhood via BFS
        neighborhood: list[dict[str, Any]] = []
        visited: set[int] = {nid.value}
        queue = deque([(nid.value, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth >= max_hops:
                continue

            for neighbor, edge in graph.out_neighbors(type("X", (), {"value": current})()):
                if neighbor.value not in visited:
                    visited.add(neighbor.value)
                    neighbor_node = graph.node(neighbor)
                    if neighbor_node:
                        neighborhood.append({
                            "qualified_name": neighbor_node.qualified_name,
                            "kind": neighbor_node.kind.value,
                            "name": neighbor_node.name,
                            "edge_kind": edge.kind.value,
                            "depth": depth + 1,
                        })
                        queue.append((neighbor.value, depth + 1))

            for neighbor, edge in graph.in_neighbors(type("X", (), {"value": current})()):
                if neighbor.value not in visited:
                    visited.add(neighbor.value)
                    neighbor_node = graph.node(neighbor)
                    if neighbor_node:
                        neighborhood.append({
                            "qualified_name": neighbor_node.qualified_name,
                            "kind": neighbor_node.kind.value,
                            "name": neighbor_node.name,
                            "edge_kind": edge.kind.value,
                            "depth": depth + 1,
                        })
                        queue.append((neighbor.value, depth + 1))

        result: dict[str, Any] = {
            "target": node.qualified_name,
            "mode": mode,
            "node": {
                "qualified_name": node.qualified_name,
                "kind": node.kind.value,
                "name": node.name,
                "source_uri": node.source_uri,
                "line_start": node.line_start,
                "line_end": node.line_end,
            },
            "neighborhood": neighborhood,
        }

        if include_graph_summary:
            result["graph_summary"] = {
                "node_count": graph.node_count(),
                "edge_count": graph.edge_count(),
            }

        return result

    def _op_traverse(self, graph: Graph, params: dict[str, Any]) -> dict[str, Any]:
        target = params.get("target", "")
        start = graph.find_by_qname(target)
        if start is None:
            return {"error": f"Symbol not found: {target}"}
        max_hops = max(0, int(params.get("max_hops", 3)))
        token_budget = max(0, int(params.get("token_budget", 1600)))
        queue = deque([(start, 0)])
        visited: set[int] = set()
        items: list[dict[str, Any]] = []
        used_tokens = 0
        while queue:
            node_id, hops = queue.popleft()
            if node_id.value in visited or hops > max_hops:
                continue
            visited.add(node_id.value)
            node = graph.node(node_id)
            if node is None:
                continue
            tokens = max(1, len((node.source_text or node.qualified_name).split()))
            if used_tokens + tokens > token_budget:
                break
            used_tokens += tokens
            items.append({
                "qualified_name": node.qualified_name,
                "kind": node.kind.value,
                "source_uri": node.source_uri,
                "hops": hops,
            })
            if hops < max_hops:
                queue.extend((neighbor, hops + 1) for neighbor, _ in graph.out_neighbors(node_id))
                queue.extend((neighbor, hops + 1) for neighbor, _ in graph.in_neighbors(node_id))
        return {
            "target": target,
            "items": items,
            "used_tokens": used_tokens,
            "token_budget": token_budget,
        }

    def _op_callers(self, graph: Graph, params: dict[str, Any]) -> dict[str, Any]:
        target = params.get("target", "")
        nid = graph.find_by_qname(target)
        if nid is None:
            return {"error": f"Not found: {target}"}

        callers = []
        for src, edge in graph.in_neighbors(nid):
            node = graph.node(src)
            if node:
                callers.append({
                    "qualified_name": node.qualified_name,
                    "kind": node.kind.value,
                    "edge_kind": edge.kind.value,
                })

        return {"callers": callers, "total": len(callers)}

    def _op_callees(self, graph: Graph, params: dict[str, Any]) -> dict[str, Any]:
        target = params.get("target", "")
        nid = graph.find_by_qname(target)
        if nid is None:
            return {"error": f"Not found: {target}"}

        callees = []
        for dst, edge in graph.out_neighbors(nid):
            node = graph.node(dst)
            if node:
                callees.append({
                    "qualified_name": node.qualified_name,
                    "kind": node.kind.value,
                    "edge_kind": edge.kind.value,
                })

        return {"callees": callees, "total": len(callees)}

    def _op_flows(self, graph: Graph, params: dict[str, Any]) -> dict[str, Any]:
        flows = []
        for nid, node in graph.nodes():
            if node.kind.value == "flow":
                flows.append({
                    "qualified_name": node.qualified_name,
                    "entry": node.properties.get("entry", "unknown"),
                    "size": node.properties.get("size", 0),
                })
        flows.sort(key=lambda f: f["size"], reverse=True)
        return {"flows": flows[:params.get("top", 20)], "total": len(flows)}

    def _op_affected_flows(self, graph: Graph, params: dict[str, Any]) -> dict[str, Any]:
        targets = params.get("targets", [])
        if isinstance(targets, str):
            targets = [targets]
        affected: dict[str, dict[str, Any]] = {}
        for target in targets:
            node_id = graph.find_by_qname(target)
            if node_id is None:
                continue
            for flow_id, edge in graph.out_neighbors(node_id):
                flow = graph.node(flow_id)
                if (
                    flow is not None
                    and flow.kind == NodeKind.FLOW
                    and edge.kind in (EdgeKind.ENTRY_OF, EdgeKind.MEMBER_OF)
                ):
                    affected[flow.qualified_name] = {
                        "qualified_name": flow.qualified_name,
                        "entry": flow.properties.get("entry"),
                        "size": flow.properties.get("size", 0),
                    }
        return {"affected_flows": list(affected.values()), "total": len(affected)}

    def _op_gaps(self, graph: Graph) -> dict[str, Any]:
        """Find structural weaknesses and review blind spots."""
        gaps: list[dict[str, Any]] = []

        # Isolated nodes (no edges)
        for nid, node in graph.nodes():
            out = sum(1 for _ in graph.out_neighbors(nid))
            in_ = sum(1 for _ in graph.in_neighbors(nid))
            if out == 0 and in_ == 0 and node.kind in (
                NodeKind.FUNCTION, NodeKind.METHOD, NodeKind.CLASS
            ):
                gaps.append({
                    "type": "isolated",
                    "qualified_name": node.qualified_name,
                    "kind": node.kind.value,
                })

        # Functions without tests
        tested: set[str] = set()
        for _, src, dst, edge in graph.edges():
            if edge.kind.value == "tested_by":
                src_node = graph.node(src)
                if src_node:
                    tested.add(src_node.qualified_name)

        for nid, node in graph.nodes():
            if node.kind in (NodeKind.FUNCTION, NodeKind.METHOD):
                if node.qualified_name not in tested:
                    gaps.append({
                        "type": "no_test",
                        "qualified_name": node.qualified_name,
                        "kind": node.kind.value,
                    })

        return {"gaps": gaps[:50], "total": len(gaps)}

    def _op_core(self, graph: Graph, params: dict[str, Any]) -> dict[str, Any]:
        adjacency: dict[int, set[int]] = {node_id.value: set() for node_id, _ in graph.nodes()}
        for _, source, target, _ in graph.edges():
            adjacency[source.value].add(target.value)
            adjacency[target.value].add(source.value)
        remaining = set(adjacency)
        coreness = {node_id: 0 for node_id in remaining}
        k = 0
        while remaining:
            removed = True
            while removed:
                removed = False
                for node_id in list(remaining):
                    degree = len(adjacency[node_id] & remaining)
                    if degree <= k:
                        coreness[node_id] = k
                        remaining.remove(node_id)
                        removed = True
            k += 1
        ranked = []
        for node_id, core in coreness.items():
            node = graph.node(NodeId(node_id))
            if node:
                ranked.append({
                    "qualified_name": node.qualified_name,
                    "kind": node.kind.value,
                    "core": core,
                })
        ranked.sort(key=lambda item: (-item["core"], item["qualified_name"]))
        return {"core_nodes": ranked[: int(params.get("limit", 25))]}

    def _op_wiki(self, graph: Graph) -> dict[str, Any]:
        architecture = self._op_architecture(graph)
        lines = [
            "# Graphician Graph Wiki",
            "",
            f"- Nodes: {graph.node_count()}",
            f"- Edges: {graph.edge_count()}",
            f"- Communities: {architecture['communities']}",
            "",
            "## Key components",
        ]
        for node in architecture["god_nodes"][:10]:
            lines.append(f"- `{node.get('qualified_name', node.get('name', 'unknown'))}`")
        return {"wiki": "\n".join(lines)}

    def _op_report(self, graph: Graph) -> dict[str, Any]:
        return {
            "report": {
                "health": self._op_health(graph),
                "architecture": self._op_architecture(graph),
                "gaps": self._op_gaps(graph),
                "surprises": compute_surprise_scoring(graph),
            }
        }

    def _op_knowledge_gaps(self, graph: Graph) -> dict[str, Any]:
        """Find undocumented or isolated code areas."""
        gaps: list[dict[str, Any]] = []
        for nid, node in graph.nodes():
            if node.kind in (NodeKind.FUNCTION, NodeKind.CLASS):
                has_docs = (
                    node.source_text and
                    any(kw in node.source_text.lower() for kw in ["doc", "comment", """"\""""""])
                )
                if not has_docs and node.kind == NodeKind.CLASS:
                    gaps.append({
                        "type": "undocumented",
                        "qualified_name": node.qualified_name,
                        "kind": node.kind.value,
                    })
        return {"knowledge_gaps": gaps[:50], "total": len(gaps)}

    def _op_suggested_questions(self, graph: Graph) -> list[dict[str, Any]]:
        """Generate prioritized review questions."""
        questions: list[dict[str, Any]] = []

        # Questions based on structural analysis
        for nid, node in graph.nodes():
            if node.kind in (NodeKind.FUNCTION, NodeKind.METHOD):
                in_deg = sum(1 for _ in graph.in_neighbors(nid))
                if in_deg == 0 and node.kind == NodeKind.FUNCTION:
                    questions.append({
                        "priority": "medium",
                        "question": f"Is {node.qualified_name} reachable? No callers found.",
                    })

        return questions[:20]

    def _op_architecture(self, graph: Graph) -> dict[str, Any]:
        comm = detect_communities(graph)
        hubs = find_hub_nodes(graph)
        gods = find_god_nodes(graph)
        bridges = find_bridge_nodes(graph)

        return {
            "communities": comm["community_count"],
            "modularity": round(comm["quality"], 4),
            "bridge_nodes": bridges["bridge_nodes"][:10],
            "hub_nodes": hubs["hub_nodes"][:10],
            "god_nodes": gods["god_nodes"][:10],
        }

    def _op_review_context(
        self,
        graph: Graph,
        params: dict[str, Any],
    ) -> Any:
        """Token-budgeted context for code review."""
        return build_context_pack(
            graph,
            query="review changes",
            intent="review",
            token_budget=params.get("token_budget", 1600),
        )

    def _op_graph_diff(self, graph: Graph, params: dict[str, Any]) -> dict[str, Any]:
        """Compare two graph snapshots. Expects 'base_graph' and 'head_graph' in params."""
        base_qname = params.get("base_qname", "")
        head_qname = params.get("head_qname", "")
        if not base_qname or not head_qname:
            return {"error": "Requires base_qname and head_qname"}
        # Look up base/head graphs from store (params carry snapshot identifiers)
        base = params.get("_base_graph", graph)
        head = params.get("_head_graph", graph)
        if base is head:
            return {"graph_diff": {}, "message": "base and head must be different graphs"}
        result = graph_diff(base, head)
        return {"graph_diff": result}

    def _op_differential(self, graph: Graph, params: dict[str, Any]) -> dict[str, Any]:
        """Compute temporal (differential) diff between two commits."""
        from ..cli.response.temporal import graph_diff_json

        base = params.get("base", "HEAD~1")
        head = params.get("head", "HEAD")
        top = params.get("top", 50)
        return graph_diff_json(graph, base, head, top)

    def _op_dedup(self, graph: Graph) -> dict[str, Any]:
        """Deduplicate nodes using MinHash/LSH pipeline."""
        result: DedupResult = deduplicate_nodes(graph)
        return {
            "dedup": {
                "candidates_examined": result.candidates_examined,
                "merges": result.merges,
                "nodes_removed": result.nodes_removed,
                "edges_rewired": result.edges_rewired,
            }
        }

    def _op_rename_preview(self, graph: Graph, params: dict[str, Any]) -> dict[str, Any]:
        """Preview rename of a symbol without modifying the graph."""
        result = rename_preview(graph, params.get("qname", ""), params.get("new_name", ""))
        return {"rename_preview": result or {"error": "Node not found"}}

    def _op_find_related(self, graph: Graph, params: dict[str, Any]) -> dict[str, Any]:
        """Find structurally similar nodes by name normalization and edge pattern matching."""
        qname = params.get("qname", "")
        if not qname:
            return {"error": "Requires qname"}
        target_id = graph.find_by_qname(qname)
        if target_id is None:
            return {"error": "Node not found"}
        target_node = graph.node(target_id)
        if target_node is None:
            return {"error": "Node not found"}
        target_name = target_node.name
        from ...extraction.documents.document_utils import normalize_for_match
        target_norm = normalize_for_match(target_name)
        related: list[dict[str, Any]] = []
        for nid, node in graph.nodes():
            if nid == target_id:
                continue
            if node.kind != target_node.kind:
                continue
            node_norm = normalize_for_match(node.name)
            if node_norm == target_norm:
                related.append({
                    "id": nid.value,
                    "qualified_name": node.qualified_name,
                    "name": node.name,
                    "kind": node.kind.value,
                    "similarity": 1.0,
                })
            elif node_norm in target_norm or target_norm in node_norm:
                related.append({
                    "id": nid.value,
                    "qualified_name": node.qualified_name,
                    "name": node.name,
                    "kind": node.kind.value,
                    "similarity": 0.7,
                })
        related.sort(key=lambda r: -r["similarity"])
        return {"related": related[:20]}

    def _op_export_graphml(self, graph: Graph) -> dict[str, Any]:
        """Export the graph as GraphML XML."""
        xml = export_graphml(graph)
        return {"export": xml}

    def _op_health(self, graph: Graph) -> dict[str, Any]:
        """Graph health diagnostics."""
        total_nodes = graph.node_count()
        total_edges = graph.edge_count()

        # Check for orphaned nodes
        nodes_with_edges: set[int] = set()
        for _, src, dst, _ in graph.edges():
            nodes_with_edges.add(src.value)
            nodes_with_edges.add(dst.value)

        orphans = 0
        for nid, _ in graph.nodes():
            if nid.value not in nodes_with_edges:
                orphans += 1

        return {
            "node_count": total_nodes,
            "edge_count": total_edges,
            "orphan_nodes": orphans,
            "avg_degree": (total_edges * 2 / total_nodes) if total_nodes else 0,
            "call_resolution_rate": "N/A",  # Would need call resolution tracking
        }

    def _op_token_savings(self, graph: Graph) -> dict[str, Any]:
        """Estimate token savings from using Graphician graph vs full repo."""
        total_nodes = graph.node_count()
        total_edges = graph.edge_count()
        # Rough estimate: each node ~50 tokens avg, each edge ~20 tokens
        graph_tokens = total_nodes * 50 + total_edges * 20
        # CLI context (full graph dump) ~ 2x graph tokens
        cli_context = graph_tokens * 2
        # MCP context (targeted queries) ~ 0.3x graph tokens
        mcp_context = graph_tokens * 0.3
        cli_savings = max(0, cli_context - graph_tokens)
        mcp_savings = max(0, mcp_context - graph_tokens)
        return {
            "token_savings": {
                "cli": round(cli_savings),
                "mcp": round(mcp_savings),
                "graph_tokens": graph_tokens,
                "cli_context": cli_context,
                "mcp_context": mcp_context,
            }
        }

    def _op_token_benchmark(self, graph: Graph, params: dict[str, Any]) -> dict[str, Any]:
        """Token benchmark for different query strategies."""
        total_nodes = graph.node_count()
        total_edges = graph.edge_count()
        graph_tokens = total_nodes * 50 + total_edges * 20
        # Strategy costs
        strategies = {
            "full_repo": graph_tokens * 10,
            "full_graph": graph_tokens,
            "cli_context": graph_tokens * 2,
            "mcp_targeted": graph_tokens * 0.3,
            "minimal_context": graph_tokens * 0.1,
        }
        benchmark = []
        for name, cost in strategies.items():
            benchmark.append({
                "strategy": name,
                "token_estimate": cost,
            })
        benchmark.sort(key=lambda b: b["token_estimate"])
        return {"benchmark": benchmark}

    # ── Prompt templates ─────────────────────────────────────────────

    # Token-efficiency rules appended to every prompt
    _TOKEN_EFFICIENCY_RULES = (
        "## Rules for Token-Efficient Graph Usage\n"
        "1. ALWAYS call `minimal_context(target=\"<task description>\")` first.\n"
        "2. Use `detail_level=\"minimal\"` on all tool calls unless the minimal output\n"
        "   is insufficient.\n"
        "3. Only escalate to `detail_level=\"standard\"` or `\"full\"` for the specific\n"
        "   entities that need deeper inspection.\n"
        "4. Never request more than 3 tool calls per turn unless absolutely necessary.\n"
        "5. Prefer targeted queries (search with a specific symbol) over broad scans\n"
        "   (list_communities with full members).\n"
        "6. When reviewing changes: detect_changes(detail_level=\"minimal\") → only\n"
        "   expand on high-risk items.\n"
    )

    def _handle_prompts_list(self, msg_id: Any) -> dict[str, Any]:
        """List available prompt templates."""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "prompts": [
                    {
                        "name": "review_changes",
                        "description": "Pre-commit code review workflow. Analyzes changes against a base ref and recommends improvements.",
                        "arguments": [
                            {
                                "name": "base",
                                "description": "Git ref to diff against (e.g. HEAD~1, main, abc1234).",
                                "required": False,
                            }
                        ],
                    },
                    {
                        "name": "architecture_map",
                        "description": "Generate architecture documentation from community structure and execution flows.",
                    },
                    {
                        "name": "debug_issue",
                        "description": "Guided debugging workflow. Systematically traces issues through the call graph.",
                        "arguments": [
                            {
                                "name": "description",
                                "description": "Brief description of the issue (e.g. 'NullPointer in UserService.login').",
                                "required": False,
                            }
                        ],
                    },
                    {
                        "name": "onboard_developer",
                        "description": "New developer orientation. Provides technology overview, architecture, and critical flows.",
                    },
                    {
                        "name": "pre_merge_check",
                        "description": "PR readiness check. Validates risk, test coverage, dead code, and flow impact before merge.",
                        "arguments": [
                            {
                                "name": "base",
                                "description": "Git ref to diff against (e.g. HEAD~1, main, develop).",
                                "required": False,
                            }
                        ],
                    },
                ],
            },
        }

    def _handle_prompt_get(
        self,
        params: dict[str, Any],
        msg_id: Any,
    ) -> dict[str, Any]:
        """Get a prompt template."""
        name = params.get("name", "")
        args = params.get("arguments", {})

        prompt_map = {
            "review_changes": self._prompt_review_changes,
            "architecture_map": self._prompt_architecture_map,
            "debug_issue": self._prompt_debug_issue,
            "onboard_developer": self._prompt_onboard_developer,
            "pre_merge_check": self._prompt_pre_merge_check,
        }

        handler = prompt_map.get(name)
        if handler:
            messages = handler(args)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"messages": messages},
            }

        return self._send_error(-32602, f"Unknown prompt: {name}", msg_id)

    def _prompt_review_changes(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        base = args.get("base", "HEAD~1")
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": (
                        f"{self._TOKEN_EFFICIENCY_RULES}\n"
                        f"## Review Workflow\n\n"
                        f"You are reviewing code changes against `{base}`. Follow these steps:\n\n"
                        f"1. Call `minimal_context(target=\"review changes against {base}\")` to get a risk overview.\n"
                        f"2. If risk is 'low':\n"
                        f"   - Call `detect_changes(base='{base}', detail_level=\"minimal\")` → report summary + any test gaps.\n"
                        f"   - Skip to step 5.\n"
                        f"3. If risk is 'medium' or 'high':\n"
                        f"   a. Call `detect_changes(base='{base}', detail_level=\"standard\")` for full change list.\n"
                        f"   b. For each high-risk function (risk score > 0.5), call:\n"
                        f"      `callers_of(target=\"<func_qname>\")`\n"
                        f"   c. Call `affected_flows(base='{base}')` only if >3 changed functions.\n"
                        f"4. Identify test gaps: untested functions with high caller counts.\n"
                        f"5. Summarize:\n"
                        f"   - Risk level (LOW/MEDIUM/HIGH/CRITICAL)\n"
                        f"   - What changed (files + functions)\n"
                        f"   - Test gaps found\n"
                        f"   - Specific improvements needed\n\n"
                        f"IMPORTANT: Do NOT call `review_context` unless you need source code snippets\n"
                        f"for a specific function. Prefer graph queries over raw context.\n\n"
                        f"Do NOT escalate to `detail_level=\"full\"` unless explicitly asked for line-level diffs."
                    ),
                },
            }
        ]

    def _prompt_architecture_map(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": (
                        f"{self._TOKEN_EFFICIENCY_RULES}\n"
                        "## Architecture Mapping Workflow\n\n"
                        "Produce a concise architecture map for this codebase. Follow these steps:\n\n"
                        "1. Call `minimal_context(target=\"map architecture\")` to set context.\n"
                        "2. Call `architecture_overview(detail_level=\"minimal\")` for community coupling summary.\n"
                        "3. Call `flows` for critical flow names + criticality scores.\n"
                        "4. Call `communities` to list all communities with sizes.\n"
                        "5. Only call `community_split` for the 1-2 communities the user\n"
                        "   is most interested in.\n"
                        "6. Produce a concise text diagram showing:\n"
                        "   - Communities as boxes\n"
                        "   - Key flows as arrows between them\n"
                        "   - Hub nodes (high centrality) within each community\n\n"
                        "FORMAT: Use Mermaid syntax for the diagram, or plain text with indentation\n"
                        "if Mermaid is not supported.\n\n"
                        "IMPORTANT: Stop after step 6 unless the user asks for deeper analysis of a\n"
                        "specific community or flow."
                    ),
                },
            }
        ]

    def _prompt_debug_issue(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        description = args.get("description", "<issue description>")
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": (
                        f"{self._TOKEN_EFFICIENCY_RULES}\n"
                        f"## Debug Workflow\n\n"
                        f"Debug the following issue: **{description}**\n\n"
                        f"Follow these steps systematically:\n\n"
                        f"1. Call `minimal_context(target=\"debug: {description}\")` to get baseline context.\n"
                        f"2. Call `search(query=\"<keywords from description>\")` to find relevant symbols.\n"
                        f"3. For the top 1-2 results:\n"
                        f"   - Call `impact(target=\"<symbol_qname>\")` to check blast radius\n"
                        f"   - Call `callers_of(target=\"<symbol_qname>\")` to find callers\n"
                        f"4. If the issue involves execution flow:\n"
                        f"   - Call `flows` to find relevant flows\n"
                        f"   - Call `affected_flows` if the issue is in changed code\n"
                        f"5. Check for dead code: `dead_code` on affected files.\n"
                        f"6. Only call `review_context` if you need to trace the blast radius of a\n"
                        f"   specific change or see raw source snippets.\n\n"
                        f"DEBUGGING STRATEGY:\n"
                        f"- Start broad (search), narrow to specific symbols, trace flows\n"
                        f"- Use `minimal` detail until you confirm the affected scope\n"
                        f"- Track the call chain: caller → callee → deeper callers\n"
                        f"- Report: root cause, affected functions, suggested fix location\n\n"
                        f"IMPORTANT: Do NOT call `review_context` or `impact` on unrelated code.\n"
                        f"Stay focused on the call chain leading to the issue."
                    ),
                },
            }
        ]

    def _prompt_onboard_developer(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": (
                        f"{self._TOKEN_EFFICIENCY_RULES}\n"
                        "## Onboarding Workflow\n\n"
                        "Welcome a new developer to this codebase. Follow these steps:\n\n"
                        "1. Call `minimal_context(target=\"onboard developer\")` for high-level context.\n"
                        "2. Call `status` for technology overview (languages, file counts, node types).\n"
                        "3. Call `architecture_overview(detail_level=\"minimal\")` for the 30-second\n"
                        "   mental model of the codebase structure.\n"
                        "4. Call `communities` and present as a table:\n"
                        "   | Community | Size | Dominant Language |\n"
                        "   |-----------|------|-------------------|\n"
                        "   | auth      | 42   | TypeScript        |\n"
                        "   | core      | 28   | Rust              |\n"
                        "5. Call `flows` and highlight the top 3 critical flows:\n"
                        "   - Flow name, criticality score, member count\n"
                        "   - Brief description of what each flow does\n"
                        "6. Call `knowledge_gaps` to identify undocumented or isolated code areas.\n"
                        "7. Only drill into a specific community or flow if the developer asks.\n\n"
                        "OUTPUT FORMAT:\n"
                        "```\n"
                        "## Codebase Overview\n"
                        "- Languages: TypeScript (60%), Rust (30%), Python (10%)\n"
                        "- Files: 142 source files, 28 modules\n"
                        "- Communities: 5 major communities\n\n"
                        "## Architecture Summary\n"
                        "- auth-service: handles authentication, 42 nodes\n"
                        "- core-lib: shared utilities, 28 nodes\n"
                        "- ...\n\n"
                        "## Critical Flows\n"
                        "1. [login-flow] criticality: 0.85, depth: 4\n"
                        "   Entry: `AuthService.login()` → cascades to 12 functions\n"
                        "2. ...\n\n"
                        "## Known Gaps\n"
                        "- `module-x` has low cohesion (0.3) - consider splitting\n"
                        "- 3 orphan functions with no callers\n"
                        "```\n\n"
                        "IMPORTANT: This is a high-level orientation. Do NOT drill into implementation\n"
                        "details unless specifically asked. The developer can request deeper analysis\n"
                        "after this overview."
                    ),
                },
            }
        ]

    def _prompt_pre_merge_check(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        base = args.get("base", "HEAD~1")
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": (
                        f"{self._TOKEN_EFFICIENCY_RULES}\n"
                        f"## Pre-Merge Check Workflow\n\n"
                        f"Perform a pre-merge check for changes against `{base}`. Follow these steps:\n\n"
                        f"1. Call `minimal_context(target=\"pre-merge check\")` to set baseline.\n"
                        f"2. Call `detect_changes(detail_level=\"minimal\")` for risk score and test gaps.\n"
                        f"3. If risk > 0.4:\n"
                        f"   - Call `affected_flows` to see which flows are impacted\n"
                        f"4. If test_gap_count > 0:\n"
                        f"   - Call `search(query=\"<untested function names>\")` to verify they're real\n"
                        f"   - Report: function name, risk score, caller count, recommendation\n"
                        f"5. Call `dead_code` to check for newly dead code.\n"
                        f"6. Call `large_functions(min_lines=30)` if any changed functions are large.\n"
                        f"7. Only call `impact` or `review_context` if risk > 0.7.\n\n"
                        f"OUTPUT FORMAT:\n"
                        f"```\n"
                        f"## Pre-Merge Check Results\n"
                        f"- Risk level: HIGH (0.65)\n"
                        f"- Changed functions: 5\n"
                        f"- Test gaps: 2 (UserService.login, PaymentProcessor.process)\n"
                        f"- Dead code found: 0\n"
                        f"- Large functions: 1 (AuthService.authenticate, 45 lines)\n\n"
                        f"## GO/NO-GO Recommendation\n"
                        f"GO (conditional)\n"
                        f"- Fix test gaps before merge\n"
                        f"- Review flow impact: [flow-list]\n"
                        f"- Follow-ups: [list]\n\n"
                        f"## Required Follow-ups\n"
                        f"1. Add tests for UserService.login (risk: 0.45)\n"
                        f"2. Add tests for PaymentProcessor.process (risk: 0.52)\n"
                        f"3. Review affected flows: [login-flow, payment-flow]\n"
                        f"```\n\n"
                        f"IMPORTANT: The recommendation must be GO/NO-GO with 1-sentence justification.\n"
                        f"Be conservative: if uncertain, recommend NO-GO with specific conditions."
                    ),
                },
            }
        ]

    # ── Response helpers ─────────────────────────────────────────────

    def _send_response(self, response: dict[str, Any]) -> None:
        """Send a JSON-RPC response."""
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()

    def _send_error(
        self,
        code: int,
        message: str,
        msg_id: Any = None,
    ) -> dict[str, Any]:
        """Create an error response."""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": code,
                "message": message,
            },
        }
