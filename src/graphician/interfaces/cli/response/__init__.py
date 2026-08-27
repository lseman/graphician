"""Structured response system for MCP tool operations.

Orchestrates:
- ``tool_response`` / ``tool_response_cached`` — main entry points
- ``DetailLevel`` — response compactness control
- Response guardrails (pagination, hard limits)
- Hint generation (workflow adjacency)
- Graph summary insertion

Mirrors the Rust ``mod.rs`` as the response dispatcher.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Any, ClassVar

from ....analysis.communities import detect_communities, knowledge_gaps
from ....analysis.structure import find_dead_code
from ....core.edge import EdgeKind
from ....core.graph import Graph
from ....core.id import NodeId
from ....persistence.store import GraphStore

# Sub-modules
from .analysis import (
    articulation_json,
    bridge_nodes_json,
    core_json,
    cycles_json,
    diagnostics_json,
    gaps_json,
    large_functions_json,
    surprises_json,
)
from .architecture import architecture_overview_json, community_split_json
from .flows import handle_affected_flows, handle_blast_radius, handle_flows, handle_test_coverage
from .hints import SessionState, generate_hints
from .impact import handle_god_nodes, handle_impact, hub_nodes_json
from .paths import handle_paths
from .refactor_response import rename_preview_json
from .reports import export_graphml, generate_report_markdown
from .reviews import (
    counterfactual_json,
    review_context_json,
    suggested_questions_json,
    traverse_json,
)
from .search import find_related_json, handle_context_pack, handle_search
from .snapshot_diff import snapshot_diff_json
from .temporal import detect_changes_json, differential_json, graph_diff_json, risk_json
from .token_benchmark import benchmark_json, run_token_benchmark
from .token_savings import token_savings_for_graph

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------


class DetailLevel:
    """Response compactness control."""

    Minimal = "minimal"
    Standard = "standard"
    Full = "full"

    _MAP: ClassVar[dict[str, str]] = {"minimal": "minimal", "full": "full"}

    @classmethod
    def parse(cls, value: str) -> str:
        return cls._MAP.get(value, "standard")

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> str:
        return cls.parse(params.get("detail_level", "standard"))

    def limit(self, standard: int) -> int:
        if self == "minimal":
            return min(standard, 5)
        elif self == "full":
            return standard * 4
        return standard


def _limit_param(params: dict[str, Any], default: int = 50) -> int:
    return int(params.get("limit", default))


def _required_str(params: dict[str, Any], key: str) -> str:
    val = params.get(key)
    if not isinstance(val, str):
        raise ValueError(f"missing string param '{key}'")
    return val


def _base_param(params: dict[str, Any]) -> str:
    return params.get("base", "HEAD~1")


# -----------------------------------------------------------------------
# Cache (process-lifetime)
# -----------------------------------------------------------------------

_cache_data: dict[str, tuple[float, Graph]] = {}
_cache_lock: Any = None

try:
    import threading

    _cache_lock = threading.Lock()
except ImportError:
    _cache_lock = None


def _cache_key(db_path: str) -> str:
    return str(Path(db_path).resolve())


def _load_cached(db_path: str) -> Graph | None:
    """Load graph from process-lifetime cache if DB unchanged."""
    key = _cache_key(db_path)
    with _cache_lock:
        if key in _cache_data:
            mtime, graph = _cache_data[key]
        else:
            return None

    # Check if DB file changed
    try:
        new_mtime = Path(db_path).stat().st_mtime
        if new_mtime == mtime:
            return graph
    except OSError:
        return None

    return None


def _cache_put(db_path: str, graph: Graph) -> None:
    with _cache_lock:
        _cache_data[_cache_key(db_path)] = (
            Path(db_path).stat().st_mtime,
            graph,
        )


# -----------------------------------------------------------------------
# Response guardrails
# -----------------------------------------------------------------------

_PAGEABLE_KEYS = [
    "hits",
    "nodes",
    "paths",
    "impacted",
    "changed_files",
    "changed_ranges",
    "changed_symbols",
    "changed_nodes",
    "snippets",
    "communities",
    "cross_community_coupling",
    "bridge_nodes",
    "cycles",
    "core_nodes",
    "articulation_points",
    "warnings",
    "questions",
]

HARD_LIMIT = 500


def _apply_guardrails(
    value: dict[str, Any],
    graph: Graph,
    params: dict[str, Any],
    detail: str,
) -> dict[str, Any]:
    """Apply pagination and graph_summary to response."""
    default_limit = {"minimal": 10, "standard": 50, "full": 200}.get(detail, 50)
    offset = int(params.get("offset", 0))
    limit = min(
        max(int(params.get("response_limit", params.get("page_limit", default_limit))), 1),
        HARD_LIMIT,
    )
    include_summary = params.get("include_graph_summary", False) is True

    pagination = {}
    for key in _PAGEABLE_KEYS:
        if key in value and isinstance(value[key], list):
            total = len(value[key])
            start = min(offset, total)
            end = min(start + limit, total)
            page = value[key][start:end]
            value[key] = page
            pagination[key] = {
                "offset": offset,
                "limit": limit,
                "returned": end - start,
                "total": total,
                "has_more": end < total,
            }

    if include_summary and "graph_summary" not in value:
        value["graph_summary"] = _graph_summary_json(graph)

    guardrails: dict[str, Any] = {
        "response_limit": limit,
        "offset": offset,
        "hard_limit": HARD_LIMIT,
    }
    if pagination:
        guardrails["pagination"] = pagination

    value["guardrails"] = guardrails
    return value


def _graph_summary_json(graph: Graph) -> dict[str, Any]:
    """Generate a compact graph summary."""
    kind_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for _, node in graph.nodes():
        kind = str(node.kind)
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        if node.source_uri:
            source_counts[node.source_uri] = source_counts.get(node.source_uri, 0) + 1

    kinds = sorted(kind_counts.items(), key=lambda x: -x[1])
    sources = sorted(source_counts.items(), key=lambda x: -x[1])[:5]

    return {
        "node_count": graph.node_count(),
        "edge_count": graph.edge_count(),
        "kind_counts": [{"kind": k, "count": c} for k, c in kinds],
        "top_sources": [{"source": s, "nodes": c} for s, c in sources],
    }


def _compact_for_detail(value: dict[str, Any], detail: str) -> dict[str, Any]:
    """Remove snippet fields for minimal detail."""
    if detail == "minimal" and "snippets" in value:
        for item in value["snippets"]:
            if isinstance(item, dict):
                item.pop("snippet", None)
    return value


# -----------------------------------------------------------------------
# Main entry points
# -----------------------------------------------------------------------

def tool_response(db_path: str, operation: str, params: dict[str, Any]) -> dict[str, Any]:
    """One-shot JSON interface for agents (no cache)."""
    return _tool_response(db_path, operation, params, False)


def tool_response_cached(
    db_path: str, operation: str, params: dict[str, Any]
) -> dict[str, Any]:
    """Cached JSON interface for long-lived processes."""
    return _tool_response(db_path, operation, params, True)


def _tool_response(
    db_path: str,
    operation: str,
    params: dict[str, Any],
    use_cache: bool,
) -> dict[str, Any]:
    """Internal: build response."""
    store = GraphStore(db_path)
    try:
        cached = None
        if use_cache:
            cached = _load_cached(db_path)
        if cached:
            graph = cached
        else:
            graph = store.load_graph()
            if use_cache:
                _cache_put(db_path, graph)

        detail = DetailLevel.from_params(params)
        response = _dispatch(operation, graph, params, detail, db_path)
        response = _apply_guardrails(response, graph, params, detail)

        # Attach hints
        no_hints = params.get("no_hints", False) is True
        if not no_hints:
            hints = generate_hints(operation, response)
            if hints and (
                hints.get("next_steps")
                or hints.get("related")
                or hints.get("warnings")
            ):
                response["_hints"] = hints

        return response
    finally:
        store.close()


def _dispatch(
    operation: str,
    graph: Graph,
    params: dict[str, Any],
    detail: str,
    db_path: str,
) -> dict[str, Any]:
    """Route operation to handler."""
    operation = {
        "k_core": "core",
        "articulation_points": "articulation",
        "surprise_scoring": "surprises",
        "health": "diagnostics",
        "impact_radius": "blast_radius",
        "differential": "differential",  # alias to itself for clarity
    }.get(operation, operation)
    def limit_fn(d=_limit_param, p=params):
        return d(p)
    def base_fn(p=params):
        return _base_param(p)

    handlers = {
        "status": lambda: _status(graph),
        "freshness": lambda: _freshness(db_path),
        "search": lambda: _compact_for_detail(handle_search(graph, params), detail),
        "context_pack": lambda: _compact_for_detail(
            handle_context_pack(graph, params), detail
        ),
        "impact": lambda: _compact_for_detail(handle_impact(graph, params), detail),
        "detect_changes": lambda: _compact_for_detail(
            detect_changes_json(graph, base_fn(), _limit_param(params, 2), 2), detail
        ),
        "risk": lambda: _compact_for_detail(
            risk_json(graph, base_fn(), limit_fn()), detail
        ),
        "review_context": lambda: _compact_for_detail(
            review_context_json(graph, base_fn(), _limit_param(params, 200), _limit_param(params, 1600)),
            detail,
        ),
        "traverse": lambda: _compact_for_detail(
            traverse_json(
                graph,
                _required_str(params, "target"),
                params.get("direction", "both"),
                _limit_param(params, 3),
                _limit_param(params, 1200),
            ),
            detail,
        ),
        "large_functions": lambda: _compact_for_detail(
            large_functions_json(
                graph,
                _limit_param(params, 80),
                limit_fn(),
            ),
            detail,
        ),
        "bridge_nodes": lambda: _compact_for_detail(bridge_nodes_json(graph, limit_fn()), detail),
        "cycles": lambda: _compact_for_detail(cycles_json(graph, limit_fn()), detail),
        "core": lambda: _compact_for_detail(core_json(graph, limit_fn()), detail),
        "articulation": lambda: _compact_for_detail(
            articulation_json(graph, limit_fn()), detail
        ),
        "gaps": lambda: _compact_for_detail(gaps_json(graph, limit_fn()), detail),
        "surprises": lambda: _compact_for_detail(surprises_json(graph, limit_fn()), detail),
        "diagnostics": lambda: _compact_for_detail(
            diagnostics_json(db_path, limit_fn()), detail
        ),
        "graph_diff": lambda: _compact_for_detail(
            _graph_diff(graph, db_path, params), detail
        ),
        "differential": lambda: _compact_for_detail(
            differential_json(
                graph,
                base_fn(),
                _required_str(params, "head") if "head" in params else "HEAD",
                limit_fn(),
            ),
            detail,
        ),
        "counterfactual": lambda: _compact_for_detail(
            counterfactual_json(
                graph,
                _required_str(params, "target"),
                params.get("direction", "out"),
                _limit_param(params, 5),
            ),
            detail,
        ),
        "motifs": lambda: _compact_for_detail(_motifs(graph, params), detail),
        "dedup": lambda: _compact_for_detail(_dedup(graph), detail),
        "patterns": lambda: _compact_for_detail(_patterns(graph), detail),
        "wiki": lambda: _compact_for_detail(_wiki(graph, params), detail),
        "suggested_questions": lambda: _compact_for_detail(
            suggested_questions_json(
                detect_changes_json(graph, base_fn(), 2),
                limit_fn(),
            ),
            detail,
        ),
        "architecture_overview": lambda: architecture_overview_json(graph, detail),
        "architecture": lambda: architecture_overview_json(graph, detail),
        "god_nodes": lambda: _compact_for_detail(handle_god_nodes(graph, params), detail),
        "flows": lambda: _compact_for_detail(handle_flows(graph, params), detail),
        "affected_flows": lambda: _compact_for_detail(
            handle_affected_flows(graph, db_path, params), detail
        ),
        "blast_radius": lambda: _compact_for_detail(
            handle_blast_radius(graph, db_path, params), detail
        ),
        "test_coverage": lambda: _compact_for_detail(
            handle_test_coverage(graph, db_path, params), detail
        ),
        "report": lambda: _write_report(db_path, _required_str(params, "output")),
        "hub_nodes": lambda: _compact_for_detail(hub_nodes_json(graph, limit_fn()), detail),
        "community_split": lambda: _compact_for_detail(
            community_split_json(
                graph,
                float(params.get("threshold", 0.25)),
                _limit_param(params, 10),
            ),
            detail,
        ),
        "dead_code": lambda: _compact_for_detail(_dead_code(graph, limit_fn()), detail),
        "find_related": lambda: _compact_for_detail(
            find_related_json(graph, _required_str(params, "target"), params.get("line")),
            detail,
        ),
        "minimal_context": lambda: _compact_for_detail(
            _minimal_context(graph, params), detail
        ),
        "context": lambda: _compact_for_detail(_minimal_context(graph, params), detail),
        "communities": lambda: _compact_for_detail(
            detect_communities(graph, params.get("algorithm", "leiden")), detail
        ),
        "knowledge_gaps": lambda: _compact_for_detail(knowledge_gaps(graph), detail),
        "callers_of": lambda: _compact_for_detail(
            _call_neighbors(graph, params, incoming=True), detail
        ),
        "callees_of": lambda: _compact_for_detail(
            _call_neighbors(graph, params, incoming=False), detail
        ),
        "export_graphml": lambda: _compact_for_detail(
            export_graphml(graph, _required_str(params, "output")),
            detail,
        ),
        "paths": lambda: _compact_for_detail(handle_paths(graph, params), detail),
        "rename_preview": lambda: _compact_for_detail(
            rename_preview_json(
                graph,
                _required_str(params, "target"),
                _required_str(params, "new_name"),
            ) or {"error": "Node not found"},
            detail,
        ),
        "snapshot_diff": lambda: _compact_for_detail(
            snapshot_diff_json(db_path, _required_str(params, "head_db"), limit_fn()),
            detail,
        ),
        "token_savings": lambda: _compact_for_detail(
            token_savings_for_graph(
                graph,
                params.get("mode", "overview"),
                params.get("include_files", False),
            ),
            detail,
        ),
        "token_benchmark": lambda: _compact_for_detail(
            benchmark_json(run_token_benchmark(
                db_path,
                graph,
                str(Path(db_path).parent),
                params.get("questions", []),
            )),
            detail,
        ),
    }

    handler = handlers.get(operation)
    if handler:
        try:
            return handler()
        except Exception as e:  # noqa: BLE001 -- tool dispatch must return an error, not crash
            return {"operation": operation, "error": str(e)}

    return {"operation": operation, "error": f"unknown tool operation {operation}"}


def _resolve_node(graph: Graph, target: str) -> NodeId | None:
    """Resolve an exact qname, integer ID, or unambiguous symbol name."""
    exact = graph.find_by_qname(target)
    if exact is not None:
        return exact
    try:
        candidate = NodeId(int(target))
    except (TypeError, ValueError):
        candidate = None
    if candidate is not None and graph.node(candidate) is not None:
        return candidate

    matches = [node_id for node_id, node in graph.nodes() if node.name == target]
    return matches[0] if len(matches) == 1 else None


def _minimal_context(graph: Graph, params: dict[str, Any]) -> dict[str, Any]:
    """Return a bounded bidirectional neighborhood around one symbol."""
    target = _required_str(params, "target")
    start = _resolve_node(graph, target)
    if start is None:
        return {"operation": "minimal_context", "error": f"target not found: {target}"}

    max_hops = max(0, min(int(params.get("max_hops", 2)), 8))
    limit = max(1, min(_limit_param(params, 50), 500))
    queue = deque([(start, 0)])
    visited = {start}
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    while queue and len(nodes) < limit:
        node_id, depth = queue.popleft()
        node = graph.node(node_id)
        if node is None:
            continue
        nodes.append({
            "qualified_name": node.qualified_name,
            "kind": node.kind.value,
            "name": node.name,
            "source_uri": node.source_uri,
            "line_start": node.line_start,
            "line_end": node.line_end,
            "depth": depth,
        })
        if depth >= max_hops:
            continue
        neighbors = [
            (neighbor, edge, "out") for neighbor, edge in graph.out_neighbors(node_id)
        ]
        neighbors.extend(
            (neighbor, edge, "in") for neighbor, edge in graph.in_neighbors(node_id)
        )
        for neighbor, edge, direction in neighbors:
            neighbor_node = graph.node(neighbor)
            if neighbor_node is None:
                continue
            edges.append({
                "source": node.qualified_name if direction == "out" else neighbor_node.qualified_name,
                "target": neighbor_node.qualified_name if direction == "out" else node.qualified_name,
                "kind": edge.kind.value,
            })
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    start_node = graph.node(start)
    return {
        "operation": "minimal_context",
        "target": start_node.qualified_name if start_node is not None else target,
        "mode": params.get("mode", "review"),
        "nodes": nodes,
        "edges": edges,
    }


def _call_neighbors(
    graph: Graph, params: dict[str, Any], *, incoming: bool
) -> dict[str, Any]:
    target = _required_str(params, "target")
    node_id = _resolve_node(graph, target)
    operation = "callers_of" if incoming else "callees_of"
    result_key = "callers" if incoming else "callees"
    if node_id is None:
        return {"operation": operation, result_key: [], "error": f"target not found: {target}"}
    iterator = graph.in_neighbors(node_id) if incoming else graph.out_neighbors(node_id)
    hits: list[dict[str, str | None]] = []
    for neighbor_id, edge in iterator:
        if edge.kind is not EdgeKind.CALLS:
            continue
        if edge.confidence.score() < 0.5:
            continue
        node = graph.node(neighbor_id)
        if node is not None:
            hits.append({
                "qualified_name": node.qualified_name,
                "kind": node.kind.value,
                "source_uri": node.source_uri,
                "edge_kind": edge.kind.value,
            })
    hits.sort(key=lambda item: item["qualified_name"] or "")
    hits = hits[: _limit_param(params, 50)]
    return {"operation": operation, result_key: hits, "total": len(hits)}


def _dead_code(graph: Graph, limit: int) -> dict[str, Any]:
    result = find_dead_code(graph, limit)
    return {
        "operation": "dead_code",
        "dead_nodes": result["dead_code"],
        "total_dead": result["total"],
    }


def _freshness(db_path: str) -> dict[str, Any]:
    from ..git import graph_freshness

    with GraphStore(db_path) as store:
        return {"operation": "freshness", **graph_freshness(store)}


def _write_report(db_path: str, output: str) -> dict[str, Any]:
    markdown = generate_report_markdown(db_path)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return {"operation": "report", "output": str(path), "written": True}


def _graph_diff(
    graph: Graph, db_path: str, params: dict[str, Any]
) -> dict[str, Any]:
    with GraphStore(db_path) as store:
        repo_root = store.get_metadata("repository_root")
    return graph_diff_json(
        graph,
        _base_param(params),
        str(params.get("head", "HEAD")),
        _limit_param(params, 50),
        repo_root,
    )


def _motifs(graph: Graph, params: dict[str, Any]) -> dict[str, Any]:
    from ....analysis.motifs import (
        diamond_inheritance_motif,
        doc_function_triangle,
        find_motifs,
        security_audit_motif,
    )

    built_in = str(params.get("built_in", "security_audit"))
    factories = {
        "security_audit": security_audit_motif,
        "diamond": diamond_inheritance_motif,
        "doc_triangle": doc_function_triangle,
    }
    factory = factories.get(built_in)
    if factory is None:
        return {
            "operation": "motifs",
            "error": (
                f"unknown built-in motif {built_in}; expected "
                "security_audit, diamond, or doc_triangle"
            ),
        }
    matches = find_motifs(graph, factory(), _limit_param(params, 50))
    return {
        "operation": "motifs",
        "built_in": built_in,
        "match_count": len(matches),
        "matches": [match.to_dict() for match in matches],
    }


def _dedup(graph: Graph) -> dict[str, Any]:
    from ....analysis.dedup import deduplicate_nodes

    return {"operation": "dedup", **deduplicate_nodes(graph.clone())}


def _patterns(graph: Graph) -> dict[str, Any]:
    from ....analysis.patterns import detect_patterns

    matches = detect_patterns(graph)
    return {
        "operation": "patterns",
        "patterns": [
            {
                "pattern_id": match.pattern_id,
                "display_name": match.display_name,
                "framework": match.framework,
                "category": match.category,
                "confidence": match.confidence,
                "matched_nodes": match.matched_nodes,
                "matched_edges": match.matched_edges,
            }
            for match in matches
        ],
        "total": len(matches),
    }


def _wiki(graph: Graph, params: dict[str, Any]) -> dict[str, Any]:
    from ..wiki import _generate_wiki

    output = _required_str(params, "output")
    result = _generate_wiki(graph, output, bool(params.get("force", False)))
    return {"operation": "wiki", "output_dir": output, **result}


def _status(graph: Graph) -> dict[str, Any]:
    """Status response with graph stats."""
    from ....analysis.structure import call_resolution_stats

    calls = call_resolution_stats(graph)

    return {
        "operation": "status",
        "nodes": graph.node_count(),
        "edges": graph.edge_count(),
        "call_resolution": {
            "resolved": calls["resolved"],
            "unresolved": calls["unresolved"],
            "rate": calls["rate"],
        },
    }


def _session() -> SessionState:
    """Process-lifetime session state (singleton)."""
    return SessionState()
