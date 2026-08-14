"""MCP (Model Context Protocol) stdio server.

Exposes Ariadne as an MCP tool with:
- One tool: ariadne (operation + params)
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


class AriadneMCP:
    """MCP server for Ariadne.

    Implements the MCP stdio protocol. Exposes one tool and five prompts.
    """

    def __init__(self, db_path: str = "ariadne.db") -> None:
        self.db_path = db_path
        self.graph: Graph | None = None
        self.store: GraphStore | None = None
        self.embedding_index = EmbeddingIndex()
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy-load the graph."""
        if self._initialized:
            return
        if self.store is None:
            self.store = GraphStore(self.db_path)
        if self.graph is None:
            self.graph = self.store.load_graph()
        self._initialized = True

    def run(self) -> None:
        """Run the MCP stdio server."""
        logger.info("Starting Ariadne MCP server")
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
                    "name": "ariadne",
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
                        "name": "ariadne",
                        "description": (
                            "Query the Ariadne codebase graph. "
                            "Search, review context, impact analysis, dependency walking, "
                            "architecture analysis, and more."
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
        queue: list[tuple[int, int]] = [(nid.value, 0)]

        while queue:
            current, depth = queue.pop(0)
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
        queue: list[tuple[NodeId, int]] = [(start, 0)]
        visited: set[int] = set()
        items: list[dict[str, Any]] = []
        used_tokens = 0
        while queue:
            node_id, hops = queue.pop(0)
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
            "# Ariadne Graph Wiki",
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
        """Estimate token savings from using Ariadne graph vs full repo."""
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

    def _handle_prompts_list(self, msg_id: Any) -> dict[str, Any]:
        """List available prompt templates."""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "prompts": [
                    {
                        "name": "review_changes",
                        "description": "Pre-commit code review workflow",
                        "arguments": [
                            {
                                "name": "base",
                                "description": "Git base reference (default: HEAD~1)",
                                "required": False,
                            }
                        ],
                    },
                    {
                        "name": "architecture_map",
                        "description": "Architecture documentation from communities and flows",
                    },
                    {
                        "name": "debug_issue",
                        "description": "Guided debugging through call graph tracing",
                        "arguments": [
                            {
                                "name": "description",
                                "description": "Issue description",
                                "required": True,
                            }
                        ],
                    },
                    {
                        "name": "onboard_developer",
                        "description": "New developer orientation overview",
                    },
                    {
                        "name": "pre_merge_check",
                        "description": "PR readiness with risk, test gaps, and dead code",
                        "arguments": [
                            {
                                "name": "base",
                                "description": "Git base reference (default: HEAD~1)",
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
                        f"Review changes from {base}. "
                        f"Use: detect_changes(base='{base}'), risk(base='{base}'), "
                        f"test_coverage(base='{base}'), and review_context."
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
                        "Map the architecture. Use: architecture(), communities(), "
                        "flows(), bridge_nodes(), god_nodes()."
                    ),
                },
            }
        ]

    def _prompt_debug_issue(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        description = args.get("description", "")
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": (
                        f"Debug: {description}. "
                        f"Use: search('{description}'), traverse(target), "
                        f"impact(target), paths(source, target)."
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
                        "Onboard a new developer. Use: architecture(), flows(), "
                        "large_functions(), status()."
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
                        f"Pre-merge check for {base}. "
                        f"Use: risk(base='{base}'), test_coverage(base='{base}'), "
                        f"dead_code(), cycles(), detect_changes(base='{base}')."
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
