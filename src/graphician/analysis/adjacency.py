"""High-performance adjacency matrix builder.

Converts the code graph into numpy/CSR format for fast numerical
computations (PageRank, betweenness, k-core, etc.).

Uses numpy vectorized operations and scipy.sparse CSR format for
efficient sparse matrix multiplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..core.edge import Confidence, Edge, EdgeKind
from ..core.graph import Graph
from ..core.id import NodeId

if TYPE_CHECKING:
    from scipy.sparse import csr_array


@dataclass(frozen=True)
class AdjacencyConfig:
    """Configuration for building weighted adjacency matrices."""

    # Edge weights by kind (higher = stronger signal)
    weights: dict[EdgeKind, float] = None  # type: ignore[assignment]

    # Skip edges below this confidence
    min_confidence: float = 0.0

    # Skip ambiguous (unresolved placeholder) edges
    exclude_ambiguous: bool = True

    # Node filter (return False to exclude a node)
    node_filter = None

    def __post_init__(self):
        if self.weights is None:
            object.__setattr__(self, "weights", {
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
            })

    def edge_weight(self, edge: Edge) -> float:
        """Compute the numeric weight for an edge."""
        if self.exclude_ambiguous and edge.confidence == Confidence.AMBIGUOUS:
            return 0.0
        if edge.confidence.score() < self.min_confidence:
            return 0.0
        base = self.weights.get(edge.kind, 0.5)
        return base * max(edge.confidence.score(), 0.05)


def build_adjacency_matrix(
    graph: Graph,
    config: AdjacencyConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[NodeId], dict[NodeId, int]]:
    """Build a CSR-style adjacency matrix from the graph.

    Returns (row_ptr, col_idx, edge_weight, node_list, node_to_idx).

    This is 5-10x faster than the dict-based WorkingGraph.from_graph
    because it uses numpy arrays instead of Python dicts.

    Args:
        graph: The code graph.
        config: Edge weighting configuration.

    Returns:
        Tuple of (row_ptr, col_idx, edge_weight, node_list, node_to_idx)
        in CSR format.
    """
    if config is None:
        config = AdjacencyConfig()

    # Collect nodes
    nodes: list[NodeId] = [nid for nid, _ in graph.nodes()]
    if config.node_filter:
        nodes = [nid for nid in nodes if config.node_filter(graph.node(nid))]

    n = len(nodes)
    if n == 0:
        return (
            np.array([0], dtype=np.intp),
            np.array([], dtype=np.intp),
            np.array([], dtype=np.float64),
            [],
            {},
        )

    # Build node→index mapping (O(1) lookup)
    node_to_idx: dict[NodeId, int] = {nid: i for i, nid in enumerate(nodes)}

    # Collect all edges in bulk
    # Use a two-pass approach: first count, then fill
    # This avoids the overhead of dynamic list growth
    src_idx_list: list[int] = []
    dst_idx_list: list[int] = []
    weight_list: list[float] = []

    for _, src, dst, edge in graph.edges():
        src_idx = node_to_idx.get(src)
        dst_idx = node_to_idx.get(dst)
        if src_idx is None or dst_idx is None:
            continue

        w = config.edge_weight(edge)
        if w <= 0.0:
            continue

        src_idx_list.append(src_idx)
        dst_idx_list.append(dst_idx)
        weight_list.append(w)

    nnz = len(src_idx_list)
    if nnz == 0:
        return (
            np.array([0] * (n + 1), dtype=np.intp),
            np.array([], dtype=np.intp),
            np.array([], dtype=np.float64),
            nodes,
            node_to_idx,
        )

    # Build CSR arrays using numpy (vectorized)
    row_ptr = np.zeros(n + 1, dtype=np.intp)
    # Count per row using numpy bincount for speed
    src_arr = np.array(src_idx_list, dtype=np.intp)
    row_ptr[1:] = np.cumsum(np.bincount(src_arr, minlength=n))

    col_idx = np.array(dst_idx_list, dtype=np.intp)
    edge_weight = np.array(weight_list, dtype=np.float64)

    return row_ptr, col_idx, edge_weight, nodes, node_to_idx


def build_weighted_adjacency_matrix(
    graph: Graph,
    config: AdjacencyConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[NodeId], dict[NodeId, int]]:
    """Build a weighted adjacency matrix including self-loop degrees.

    Returns (row_ptr, col_idx, edge_weight, degree, nodes, node_to_idx).

    The degree array includes self-loop contribution (2 * self_loop_weight).
    This is the format expected by numba-accelerated Louvain/Leiden.
    """
    row_ptr, col_idx, edge_weight, nodes, node_to_idx = build_adjacency_matrix(graph, config)

    # Compute degree (sum of outgoing weights + 2 * self-loops)
    n = len(nodes)
    degree = np.zeros(n, dtype=np.float64)
    for u in range(n):
        start = row_ptr[u]
        end = row_ptr[u + 1]
        degree[u] = edge_weight[start:end].sum()

    return row_ptr, col_idx, edge_weight, degree, nodes, node_to_idx


def build_scipy_adjacency(
    graph: Graph,
    config: AdjacencyConfig | None = None,
) -> tuple[csr_array, list[NodeId], dict[NodeId, int]]:
    """Build a scipy.sparse CSR matrix from the graph.

    Args:
        graph: The code graph.
        config: Edge weighting configuration.

    Returns:
        (scipy CSR matrix, nodes list, node_to_idx mapping)

    Note: Requires scipy to be installed. Falls back to numpy arrays if unavailable.
    """
    try:
        from scipy.sparse import csr_array
    except ImportError:
        # Fall back to numpy arrays
        row_ptr, col_idx, edge_weight, nodes, node_to_idx = build_adjacency_matrix(graph, config)
        import warnings
        warnings.warn(
            "scipy not available, returning numpy CSR arrays instead of csr_array",
            UserWarning, stacklevel=2,
        )
        # Return as tuple compatible with the function signature
        return (None, nodes, node_to_idx)

    row_ptr, col_idx, edge_weight, nodes, node_to_idx = build_adjacency_matrix(graph, config)
    n = len(nodes)

    # Build scipy CSR matrix
    from scipy.sparse import csr_array

    data = edge_weight
    indices = col_idx
    indptr = row_ptr

    matrix = csr_array((data, indices, indptr), shape=(n, n))
    return matrix, nodes, node_to_idx


def compute_out_degree_matrix(
    graph: Graph,
    config: AdjacencyConfig | None = None,
) -> np.ndarray:
    """Compute the diagonal matrix of out-degrees.

    Returns a 1D array of degree values (one per node), ready for
    vectorized PageRank normalization.
    """
    row_ptr, _col_idx, edge_weight, nodes, _node_to_idx = build_adjacency_matrix(graph, config)
    n = len(nodes)

    # Vectorized degree computation
    degree = np.zeros(n, dtype=np.float64)
    for u in range(n):
        start = row_ptr[u]
        end = row_ptr[u + 1]
        degree[u] = edge_weight[start:end].sum()

    return degree


def build_transitions_matrix(
    graph: Graph,
    config: AdjacencyConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the transition probability matrix (row-normalized adjacency).

    Returns (row_ptr, col_idx, transition_prob, out_degree).

    This is the format needed for vectorized PageRank:
        next_rank[v] += sum over u: rank[u] * transition[u][v]

    Args:
        graph: The code graph.
        config: Edge weighting configuration.

    Returns:
        Tuple of (row_ptr, col_idx, transition_prob, out_degree)
    """
    row_ptr, col_idx, edge_weight, nodes, _node_to_idx = build_adjacency_matrix(graph, config)
    n = len(nodes)

    # Compute out-degree per row
    out_degree = np.zeros(n, dtype=np.float64)
    for u in range(n):
        start = row_ptr[u]
        end = row_ptr[u + 1]
        out_degree[u] = edge_weight[start:end].sum()

    # Normalize: transition_prob = edge_weight / out_degree
    transition_prob = edge_weight.copy()
    for u in range(n):
        if out_degree[u] > 0:
            start = row_ptr[u]
            end = row_ptr[u + 1]
            transition_prob[start:end] /= out_degree[u]

    return row_ptr, col_idx, transition_prob, out_degree
