"""Numba-accelerated PageRank implementation.

Provides JIT-compiled versions of the PageRank algorithm using
numpy CSR-format adjacency matrices. Significantly faster than
the pure Python version for large graphs.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        """Return the original function when Numba is unavailable."""
        del args, kwargs

        def decorator(function):
            return function

        return decorator

    prange = range


@njit(cache=True, parallel=False)
def _pagerank_csr(
    n: int,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    edge_weight: np.ndarray,
    out_degree: np.ndarray,
    damping: float,
    iterations: int,
    personalization: np.ndarray,
    has_personalization: bool,
) -> np.ndarray:
    """CSR-format PageRank, numba-accelerated.

    Computes the PageRank vector using the power iteration method.

    Args:
        n: Number of nodes.
        row_ptr: CSR row pointers (n+1 elements).
        col_idx: Column indices (nnz elements).
        edge_weight: Edge weights (nnz elements).
        out_degree: Out-degree of each node (sum of outgoing weights).
        damping: Damping factor (default 0.85).
        iterations: Number of power iterations.
        personalization: Personalization vector (n elements).
        has_personalization: Whether to use personalization.

    Returns:
        PageRank vector of shape (n,).
    """
    init = 1.0 / n
    ranks = np.full(n, init, dtype=np.float64)
    uniform = 1.0 / n

    for _ in range(iterations):
        next_ranks = np.zeros(n, dtype=np.float64)

        # Compute the personalization term
        if has_personalization:
            for i in range(n):
                next_ranks[i] = (1.0 - damping) * personalization[i]
        else:
            for i in range(n):
                next_ranks[i] = (1.0 - damping) * uniform

        # Dangling node mass
        dangling_mass = 0.0
        for idx in range(n):
            if out_degree[idx] == 0:
                dangling_mass += ranks[idx]
                continue
            # Distribute rank to neighbors
            for e in range(row_ptr[idx], row_ptr[idx + 1]):
                v = col_idx[e]
                w = edge_weight[e] / out_degree[idx]
                next_ranks[v] += damping * ranks[idx] * w

        # Add dangling node contribution
        for i in range(n):
            if has_personalization:
                next_ranks[i] += damping * dangling_mass * personalization[i]
            else:
                next_ranks[i] += damping * dangling_mass * uniform

        ranks = next_ranks

    return ranks


@njit(cache=True, parallel=False)
def _weighted_pagerank_csr(
    n: int,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    edge_weight: np.ndarray,
    out_degree: np.ndarray,
    damping: float,
    iterations: int,
) -> np.ndarray:
    """Unpersonalized weighted PageRank with dangling node handling.

    This is a simplified version that handles dangling nodes by
    redistributing their mass uniformly.
    """
    ranks = np.full(n, 1.0 / n, dtype=np.float64)
    uniform = 1.0 / n

    for _ in range(iterations):
        next_ranks = np.full(n, (1.0 - damping) * uniform, dtype=np.float64)

        # Process each node
        for idx in range(n):
            if out_degree[idx] == 0:
                # Dangling node — add its rank to the uniform pool
                next_ranks[:] += damping * ranks[idx] * uniform
                continue
            # Distribute rank to neighbors
            rank_share = damping * ranks[idx] / out_degree[idx]
            for e in range(row_ptr[idx], row_ptr[idx + 1]):
                v = col_idx[e]
                next_ranks[v] += edge_weight[e] * rank_share

        ranks = next_ranks

    return ranks


def pagerank_csr(
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    edge_weight: np.ndarray,
    out_degree: np.ndarray,
    damping: float = 0.85,
    iterations: int = 30,
    personalization: np.ndarray | None = None,
) -> np.ndarray:
    """Run PageRank on CSR-format adjacency matrix.

    Args:
        row_ptr: CSR row pointers.
        col_idx: Column indices.
        edge_weight: Edge weights.
        out_degree: Out-degree of each node.
        damping: Damping factor.
        iterations: Number of iterations.
        personalization: Optional personalization vector.

    Returns:
        PageRank scores as numpy array.
    """
    n = len(out_degree)
    has_personalization = personalization is not None

    if personalization is not None and len(personalization) != n:
        raise ValueError("Personalization vector length must match n")

    personalization_arr = (
        personalization if personalization is not None
        else np.zeros(n, dtype=np.float64)
    )

    return _pagerank_csr(
        n, row_ptr, col_idx, edge_weight, out_degree,
        damping, iterations, personalization_arr, has_personalization,
    )


def has_numba() -> bool:
    """Check if numba acceleration is available."""
    return _HAS_NUMBA
