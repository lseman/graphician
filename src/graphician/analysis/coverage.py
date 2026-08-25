"""Coverage and health metrics for a built code graph."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..core.edge import EdgeKind
from ..core.graph import Graph
from ..core.node import Node, NodeKind
from .structure import call_resolution_stats

_SOURCE_KINDS = {
    NodeKind.MODULE,
    NodeKind.CLASS,
    NodeKind.FUNCTION,
    NodeKind.METHOD,
    NodeKind.TRAIT,
    NodeKind.IMPL,
    NodeKind.TYPE,
}
_CALLABLE_KINDS = {NodeKind.FUNCTION, NodeKind.METHOD}


def _rate(covered: int, total: int) -> float:
    return round(covered / total, 4) if total else 0.0


def _is_test(node: Node) -> bool:
    value = node.properties.get("is_test", False)
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def _language(source_uri: str | None) -> str:
    if not source_uri or source_uri.startswith("<"):
        return "unknown"
    suffix = Path(source_uri).suffix.lower().lstrip(".")
    return suffix or "unknown"


def graph_coverage(
    graph: Graph,
    *,
    expected_files: Iterable[str] | None = None,
    example_limit: int = 20,
) -> dict[str, Any]:
    """Measure extraction and relationship coverage for an in-memory graph.

    Rates are in ``[0, 1]``. Test coverage means static ``tested_by`` links,
    not runtime line or branch coverage.
    """
    if example_limit < 0:
        raise ValueError("example_limit must be non-negative")

    nodes = list(graph.nodes())
    edges = list(graph.edges())
    node_kinds = Counter(node.kind.value for _, node in nodes)
    edge_kinds = Counter(edge.kind.value for _, _, _, edge in edges)

    connected_ids: set[Any] = set()
    callers = set()
    callees = set()
    tested = set()
    for _, source, target, edge in edges:
        connected_ids.update((source, target))
        if edge.kind == EdgeKind.CALLS:
            callers.add(target)
            callees.add(source)
        elif edge.kind == EdgeKind.TESTED_BY:
            tested.add(source)

    source_symbols = [
        (node_id, node)
        for node_id, node in nodes
        if node.kind in _SOURCE_KINDS and not node.qualified_name.startswith("call::")
    ]
    located_symbols = [item for item in source_symbols if item[1].source_uri]
    missing_source = [
        node.qualified_name for _, node in source_symbols if not node.source_uri
    ][:example_limit]

    callables = [
        (node_id, node)
        for node_id, node in nodes
        if node.kind in _CALLABLE_KINDS
        and not node.qualified_name.startswith("call::")
    ]
    production_callables = [item for item in callables if not _is_test(item[1])]
    tested_production = sum(1 for node_id, _ in production_callables if node_id in tested)
    with_callers = sum(1 for node_id, _ in callables if node_id in callers)
    with_callees = sum(1 for node_id, _ in callables if node_id in callees)

    isolated = [
        node.qualified_name
        for node_id, node in nodes
        if node_id not in connected_ids
    ]
    source_files = {
        node.source_uri.replace("\\", "/").removeprefix("./")
        for _, node in nodes
        if node.source_uri and not node.source_uri.startswith("<")
    }
    extraction_roots = {NodeKind.FILE, NodeKind.DOCUMENT, NodeKind.DIAGRAM}
    files = {
        node.source_uri.replace("\\", "/").removeprefix("./")
        for _, node in nodes
        if node.kind in extraction_roots and node.source_uri
    }
    expected_file_set = {
        str(path).replace("\\", "/").removeprefix("./")
        for path in (expected_files or [])
    }
    covered_files = {
        expected
        for expected in expected_file_set
        if expected in files or any(source.endswith(f"/{expected}") for source in files)
    }
    missing_files = sorted(expected_file_set - covered_files)

    language_counts: dict[str, dict[str, int]] = {}
    for _, node in nodes:
        language = _language(node.source_uri)
        bucket = language_counts.setdefault(
            language, {"nodes": 0, "symbols": 0, "callables": 0}
        )
        bucket["nodes"] += 1
        if node.kind in _SOURCE_KINDS and not node.qualified_name.startswith("call::"):
            bucket["symbols"] += 1
        if node.kind in _CALLABLE_KINDS and not node.qualified_name.startswith("call::"):
            bucket["callables"] += 1

    call_resolution = call_resolution_stats(graph)
    if call_resolution["total"] == 0:
        call_resolution["rate"] = 0.0
    source_rate = _rate(len(located_symbols), len(source_symbols))
    connectivity_rate = _rate(len(connected_ids), len(nodes))
    test_link_rate = _rate(tested_production, len(production_callables))
    health_factors = [
        source_rate,
        connectivity_rate,
        call_resolution["rate"],
        test_link_rate,
    ]
    if expected_file_set:
        health_factors.append(_rate(len(covered_files), len(expected_file_set)))
    health_score = round(sum(health_factors) / len(health_factors), 4)

    return {
        "operation": "coverage",
        "summary": {
            "health_score": health_score,
            "nodes": len(nodes),
            "edges": len(edges),
            "source_files": len(source_files),
        },
        "source_location": {
            "covered": len(located_symbols),
            "total": len(source_symbols),
            "rate": source_rate,
            "missing_examples": missing_source,
        },
        "file_extraction": {
            "covered": len(covered_files) if expected_file_set else len(files),
            "total": len(expected_file_set) if expected_file_set else len(files),
            "rate": (
                _rate(len(covered_files), len(expected_file_set))
                if expected_file_set
                else (1.0 if files else 0.0)
            ),
            "missing_examples": missing_files[:example_limit],
            "manifest_available": bool(expected_file_set),
        },
        "call_resolution": call_resolution,
        "function_connectivity": {
            "total": len(callables),
            "with_callers": with_callers,
            "caller_rate": _rate(with_callers, len(callables)),
            "with_callees": with_callees,
            "callee_rate": _rate(with_callees, len(callables)),
        },
        "test_links": {
            "covered": tested_production,
            "total": len(production_callables),
            "rate": test_link_rate,
            "definition": "production functions with a static tested_by edge",
        },
        "connectivity": {
            "connected": len(connected_ids),
            "total": len(nodes),
            "rate": connectivity_rate,
            "isolated": len(nodes) - len(connected_ids),
            "isolated_examples": isolated[:example_limit],
        },
        "node_kinds": dict(sorted(node_kinds.items())),
        "edge_kinds": dict(sorted(edge_kinds.items())),
        "by_language": [
            {"language": language, **counts}
            for language, counts in sorted(language_counts.items())
        ],
    }
