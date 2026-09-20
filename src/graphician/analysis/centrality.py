"""Centrality metrics for the code graph.

``pagerank`` runs a weighted random-walk-with-damping iteration on the
directed graph. Edge kind and confidence shape transition probability, and
``personalized_pagerank`` biases the teleport distribution around supplied
seed nodes.

Edges with :attr:`Confidence.AMBIGUOUS` are skipped: those are the
unresolved call-site placeholders pointing at ``call::<name>`` synthetic
nodes, and including them distorts rank toward common function names like
``new``, ``len``, ``clone``.

:attr:`is_rank_noise` identifies nodes that inflate god-node rankings
without representing a real symbol: file containers, synthetic flow
nodes, and unresolved call placeholders.

Optimized versions use numpy/numba for performance-critical paths.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.edge import Confidence, EdgeKind
from ..core.graph import Graph
from ..core.id import NodeId
from ..core.node import Node, NodeKind
from .adjacency import AdjacencyConfig, build_adjacency_matrix

# ── Edge kind weights ────────────────────────────────────────────────

def _edge_weight(kind: EdgeKind) -> float:
    """Transition weight for an edge kind."""
    weights: dict[EdgeKind, float] = {
        EdgeKind.DEFINES: 0.7,
        EdgeKind.CALLS: 1.0,
        EdgeKind.IMPORTS: 0.55,
        EdgeKind.DEPENDS_ON: 0.55,
        EdgeKind.INHERITS: 1.15,
        EdgeKind.IMPLEMENTS: 1.15,
        EdgeKind.DATA_FLOW: 0.8,
        EdgeKind.READS_WRITES: 0.9,
        EdgeKind.MENTIONS: 0.75,
        EdgeKind.DESCRIBES: 0.75,
        EdgeKind.DOCUMENTED_BY: 0.75,
        EdgeKind.SIMILAR_TO: 0.6,
        EdgeKind.RATIONALE_FOR: 0.6,
        EdgeKind.ILLUSTRATES: 0.6,
        # Production→test edge: low weight so tests don't pull rank away.
        EdgeKind.TESTED_BY: 0.3,
        # Flow bookkeeping — overlay-only; don't let it skew rank.
        EdgeKind.MEMBER_OF: 0.05,
        EdgeKind.ENTRY_OF: 0.05,
    }
    return weights.get(kind, 0.5)


# ── PageRank ─────────────────────────────────────────────────────────

def pagerank(
    graph: Graph,
    damping: float = 0.85,
    iterations: int = 30,
) -> dict[NodeId, float]:
    """Run PageRank on *graph* and return a ``{node_id: rank}`` mapping.

    Uses edge-kind-dependent weights and skips ambiguous (unresolved)
    placeholder edges.
    Uses numba-accelerated algorithm when available.
    """
    return _weighted_pagerank(graph, damping, iterations, {})


def personalized_pagerank(
    graph: Graph,
    seeds: list[tuple[NodeId, float]],
    damping: float = 0.85,
    iterations: int = 30,
) -> dict[NodeId, float]:
    """Run personalized PageRank biased toward *seeds*."""
    total = sum(w for _, w in seeds)
    if total > 0.0:
        personalization: dict[NodeId, float] = {
            nid: max(w, 0.0) / total for nid, w in seeds
        }
    else:
        personalization = {}
    return _weighted_pagerank(graph, damping, iterations, personalization)


def _weighted_pagerank(
    graph: Graph,
    damping: float,
    iterations: int,
    personalization: dict[NodeId, float],
) -> dict[NodeId, float]:
    from .native import native_graph

    snapshot = native_graph(graph)
    if snapshot is not None:
        seeds = [(nid.value, weight) for nid, weight in personalization.items()]
        ranks = snapshot.pagerank(damping, iterations, seeds or None)
        return {NodeId(node_id): float(rank) for node_id, rank in ranks.items()}

    nodes = [nid for nid, _ in graph.nodes()]
    n = len(nodes)
    if n == 0:
        return {}

    # Build adjacency matrix with weighted edges
    config = AdjacencyConfig(
        weights={
            EdgeKind.DEFINES: 0.7,
            EdgeKind.CALLS: 1.0,
            EdgeKind.IMPORTS: 0.55,
            EdgeKind.DEPENDS_ON: 0.55,
            EdgeKind.INHERITS: 1.15,
            EdgeKind.IMPLEMENTS: 1.15,
            EdgeKind.DATA_FLOW: 0.8,
            EdgeKind.READS_WRITES: 0.9,
            EdgeKind.MENTIONS: 0.75,
            EdgeKind.DESCRIBES: 0.75,
            EdgeKind.DOCUMENTED_BY: 0.75,
            EdgeKind.SIMILAR_TO: 0.6,
            EdgeKind.RATIONALE_FOR: 0.6,
            EdgeKind.ILLUSTRATES: 0.6,
            EdgeKind.TESTED_BY: 0.3,
            EdgeKind.MEMBER_OF: 0.05,
            EdgeKind.ENTRY_OF: 0.05,
        },
        min_confidence=0.0,
        exclude_ambiguous=True,
    )

    row_ptr, col_idx, edge_weight, node_list, node_to_idx = build_adjacency_matrix(graph, config)

    # Compute out-degree
    out_degree = np.zeros(n, dtype=np.float64)
    for u in range(n):
        start = row_ptr[u]
        end = row_ptr[u + 1]
        out_degree[u] = edge_weight[start:end].sum()

    # Build personalization vector if needed
    personalization_vec = None
    has_personalization = bool(personalization)
    if has_personalization:
        personalization_vec = np.zeros(n, dtype=np.float64)
        for nid, weight in personalization.items():
            idx = node_to_idx.get(nid)
            if idx is not None:
                personalization_vec[idx] = max(weight, 0.0)
        # Normalize
        total = personalization_vec.sum()
        if total > 0:
            personalization_vec /= total
        else:
            personalization_vec = np.ones(n, dtype=np.float64) / n

    # Use numba-accelerated PageRank if available
    from .numba_pagerank import pagerank_csr as _pagerank_csr
    try:
        ranks_arr = _pagerank_csr(
            row_ptr, col_idx, edge_weight, out_degree,
            damping, iterations, personalization_vec,
        )
        return {node_list[i]: float(ranks_arr[i]) for i in range(n)}
    except Exception:  # noqa: BLE001 -- native/numba backend raises implementation-specific errors
        # Fallback to pure Python
        pass

    # Pure Python fallback (original implementation)
    node_index = {nid: idx for idx, nid in enumerate(nodes)}
    init = 1.0 / n
    ranks = [init] * n

    transitions = _weighted_transitions(graph, nodes, node_index)

    fallback_personalization = (
        [personalization.get(nodes[i], 0.0) for i in range(n)]
        if has_personalization else None
    )

    for _ in range(iterations):
        if fallback_personalization is not None:
            next_ranks = [
                (1.0 - damping) * p for p in fallback_personalization
            ]
        else:
            uniform = 1.0 / n
            next_ranks = [(1.0 - damping) * uniform] * n

        dangling_mass = 0.0
        for idx, out_edges in enumerate(transitions):
            if not out_edges["edges"]:
                dangling_mass += ranks[idx]
                continue
            for neighbor_idx, weight in out_edges["edges"]:
                next_ranks[neighbor_idx] += (
                    damping * ranks[idx] * weight / out_edges["total"]
                )

        for idx in range(n):
            p = personalization.get(nodes[idx], 0.0) if has_personalization else 1.0 / n
            next_ranks[idx] += damping * dangling_mass * p

        ranks = next_ranks

    return dict(zip(nodes, ranks, strict=True))


# ── Weighted transitions ─────────────────────────────────────────────

def _weighted_transitions(
    graph: Graph,
    nodes: list[NodeId],
    node_index: dict[NodeId, int],
) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    for nid in nodes:
        edges: list[tuple[int, float]] = []
        for neighbor, edge in graph.out_neighbors(nid):
            if edge.confidence == Confidence.AMBIGUOUS:
                continue
            if neighbor not in node_index:
                continue
            idx = node_index[neighbor]
            w = _edge_weight(edge.kind) * max(edge.confidence.score(), 0.05)
            edges.append((idx, w))
        total = sum(w for _, w in edges)
        transitions.append({"edges": edges, "total": total})
    return transitions


# ── Noise detection ──────────────────────────────────────────────────

def is_rank_noise(node: Node) -> bool:
    """Return ``True`` for nodes that inflate god-node rankings without
    representing a real symbol.

    Filters file containers, synthetic flow nodes, hyperedges, unresolved
    call placeholders (``call::<name>``), concept nodes, builtin type names,
    method stubs, and JSON-key identifiers that carry no engineering signal.
    """
    if node.kind in (NodeKind.FILE, NodeKind.FLOW, NodeKind.HYPEREDGE):
        return True
    if node.qualified_name.startswith("call::"):
        return True
    if node.kind == NodeKind.CONCEPT:
        return True
    # Single-char or all-numeric names carry no engineering signal
    if len(node.name) == 1 or (len(node.name) > 0 and node.name.isdigit()):
        return True
    # Method stubs: synthetic AST nodes like ".method_name()" or "function_name()"
    if node.name.startswith(".") or (node.name.endswith("()") and node.kind in (NodeKind.FUNCTION, NodeKind.METHOD)):
        return True
    # File-level hub: node name matches the source filename (no engineering signal)
    if node.source_uri:
        basename = node.source_uri.rsplit("/", 1)[-1]
        if basename and node.name == basename.rsplit(".", 1)[0]:
            return True
    # JSON key nodes: common keys in config/manifest files carry no signal
    if (
        node.source_uri
        and node.source_uri.lower().endswith(".json")
        and node.name in _JSON_KEY_NOISE
    ):
        return True
    return node.name in _BUILTIN_NOISE


# Python builtins, typing generics, and common framework/mock symbols that
# frequently inflate god-node rankings without representing project abstractions.
_BUILTIN_NOISE: frozenset[str] = frozenset({
    # Python builtins
    "str", "int", "float", "bool", "bytes", "bytearray", "complex",
    "object", "list", "dict", "set", "tuple", "frozenset",
    "range", "slice", "enumerate", "zip", "map", "filter",
    "len", "type", "isinstance", "issubclass", "hasattr", "getattr", "setattr",
    "super", "property", "staticmethod", "classmethod",
    "None", "True", "False",
    # Python typing generics
    "Any", "Optional", "Union", "Literal",
    "List", "Dict", "Set", "Tuple", "Callable",
    "Type", "ClassVar", "Final", "Protocol",
    "Counter", "defaultdict", "OrderedDict", "Deque",
    "Path", "Pattern", "Match",
    "Iterator", "Iterable", "Generator", "Sequence", "Mapping",
    "Enum", "ABCMeta", "ABC",
    "datetime", "timedelta", "date", "time",
    # Python stdlib modules commonly imported as names
    "os", "sys", "re", "json", "io", "abc", "typing",
    "copy", "dataclasses", "functools", "itertools", "operator",
    "pathlib", "collections", "math", "random",
    # Mock / testing frameworks
    "MagicMock", "Mock", "AsyncMock",
    "NonCallableMock", "NonCallableMagicMock", "PropertyMock",
    "patch", "sentinel",
    # Common framework / platform types (Swift, etc.)
    "Foundation", "SwiftUI", "UIKit", "AppKit", "Combine",
    "View", "Color", "Font", "DispatchQueue",
    "String", "Int", "Double", "Float", "Bool", "Data", "URL", "Date", "UUID",
    "Sendable", "Codable", "Decodable", "Encodable",
    "Equatable", "Hashable", "Identifiable", "Comparable",
    "AnyObject", "Error", "LocalizedError",
    "NSObject", "NSString", "NSError", "NSLock",
})

# JSON keys that appear in config/manifest files but don't represent project abstractions.
_JSON_KEY_NOISE: frozenset[str] = frozenset({
    "start", "end", "name", "id", "type", "properties",
    "value", "key", "data", "items", "title", "description", "version",
    "dependencies", "devdependencies", "peerdependencies",
    "optionaldependencies", "bundleddependencies", "bundledependencies",
    "scripts", "main", "exports", "import", "module",
    "keywords", "license", "author", "contributors",
    "engines", "os", "cpu", "platform",
    "config", "settings", "options", "args",
})
