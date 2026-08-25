"""Numba-accelerated core loops for community detection algorithms.

Provides JIT-compiled versions of the tightest algorithm loops.
Uses CSR (Compressed Sparse Row) format for adjacency.
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


# ── CSR builder ────────────────────────────────────────────────────


def build_csr_from_working(working) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a WorkingGraph to CSR arrays for numba.

    Args:
        working: WorkingGraph instance.

    Returns:
        (row_ptr, col_idx, edge_weight) CSR arrays.
    """
    n = working.len()
    row_ptr = np.zeros(n + 1, dtype=np.intp)
    for u in range(n):
        row_ptr[u + 1] = row_ptr[u] + len(working.adj[u])

    nnz = row_ptr[n]
    col_idx = np.zeros(nnz, dtype=np.intp)
    edge_weight = np.zeros(nnz, dtype=np.float64)

    for u in range(n):
        start = row_ptr[u]
        for i, (v, w) in enumerate(working.adj[u]):
            col_idx[start + i] = v
            edge_weight[start + i] = float(w)

    return row_ptr, col_idx, edge_weight


# ── Louvain/Leiden local-move ──────────────────────────────────────


@njit(cache=True, parallel=False)
def _local_move_csr(
    n: np.intp,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    edge_weight: np.ndarray,
    degree: np.ndarray,
    resolution: float,
    max_passes: int,
    min_gain: float,
    rng_seed: int,
) -> np.ndarray:
    """CSR-format Louvain local-move, numba-accelerated.

    O(E) per pass with minimal Python overhead.
    Uses deterministic random permutation for node ordering.

    Args:
        n: Number of nodes.
        row_ptr: CSR row pointers (n+1 elements).
        col_idx: Column indices (nnz elements).
        edge_weight: Edge weights (nnz elements).
        degree: Node degrees.
        resolution: Modularity resolution parameter.
        max_passes: Maximum passes per level.
        min_gain: Minimum gain to move.
        rng_seed: Random seed for node ordering.

    Returns:
        Community labels array of shape (n,).
    """
    comm = np.arange(n, dtype=np.int32)
    comm_degree = degree.copy()

    two_m = 0.0
    for u in range(n):
        two_m += degree[u]

    if two_m <= 0.0:
        return comm

    for _pass in range(max_passes):
        moved = False
        # Deterministic permutation (xorshift)
        order = np.arange(n, dtype=np.int64)
        seed = np.int64(rng_seed + _pass * 31)
        for i in range(n - 1, 0, -1):
            seed = (seed * np.int64(1103515245) + np.int64(12345)) & np.int64(0x7fffffff)
            j = seed % i
            order[i], order[j] = order[j], order[i]

        for k in range(n):
            u = order[k]
            current = comm[u]
            nd = degree[u]
            if nd == 0.0:
                continue

            # Remove u from its community
            comm_degree[current] -= nd

            # Compute weight to current community
            stay_w = 0.0
            for e in range(row_ptr[u], row_ptr[u + 1]):
                v = col_idx[e]
                if v == u:
                    continue
                if comm[v] == current:
                    stay_w += edge_weight[e]

            # Best gain: stay
            best = current
            best_gain = stay_w - resolution * nd * comm_degree[current] / two_m

            # Try each candidate community
            for e in range(row_ptr[u], row_ptr[u + 1]):
                v = col_idx[e]
                if v == u:
                    continue
                c_v = comm[v]
                if c_v == current:
                    continue
                gain = edge_weight[e] - resolution * nd * comm_degree[c_v] / two_m
                if gain > best_gain:
                    best_gain = gain
                    best = c_v

            comm[u] = best
            comm_degree[best] += nd
            if best != current:
                moved = True

        if not moved:
            break

    return comm


@njit(cache=True, parallel=False)
def _refinement_local_move_csr(
    n: np.intp,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    edge_weight: np.ndarray,
    degree: np.ndarray,
    resolution: float,
    max_passes: int,
    min_gain: float,
    rng_seed: int,
    total_weight: np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """Local-move with node_mass tracking (used in Leiden/Infomap refinement).

    Like _local_move_csr but tracks comm_size for well-connectedness checks.

    Returns:
        (comm, comm_size) where comm_size is node count per community.
    """
    comm = np.arange(n, dtype=np.int32)
    comm_degree = degree.copy()
    comm_size = np.ones(n, dtype=np.float64)

    two_m = 2.0 * total_weight
    if two_m <= 0.0:
        return comm, comm_size

    for _pass in range(max_passes):
        moved = False
        order = np.arange(n, dtype=np.int64)
        seed = np.int64(rng_seed + _pass * 31)
        for i in range(n - 1, 0, -1):
            seed = (seed * np.int64(1103515245) + np.int64(12345)) & np.int64(0x7fffffff)
            j = seed % i
            order[i], order[j] = order[j], order[i]

        for k in range(n):
            u = order[k]
            current = comm[u]
            nd = degree[u]
            comm_size[current]  # node mass (size of community u was in)
            if nd == 0.0:
                continue

            comm_degree[current] -= nd
            comm_size[current] -= 1.0

            stay_w = 0.0
            for e in range(row_ptr[u], row_ptr[u + 1]):
                v = col_idx[e]
                if v == u:
                    continue
                if comm[v] == current:
                    stay_w += edge_weight[e]

            best = current
            best_gain = stay_w - resolution * nd * comm_degree[current] / two_m

            for e in range(row_ptr[u], row_ptr[u + 1]):
                v = col_idx[e]
                if v == u:
                    continue
                c_v = comm[v]
                if c_v == current:
                    continue
                gain = edge_weight[e] - resolution * nd * comm_degree[c_v] / two_m
                if gain > best_gain:
                    best_gain = gain
                    best = c_v

            comm[u] = best
            comm_degree[best] += nd
            comm_size[best] += 1.0
            if best != current:
                moved = True

        if not moved:
            break

    return comm, comm_size


# ── Infomap LMDL ───────────────────────────────────────────────────


@njit(cache=True, parallel=False)
def _compute_community_flow_csr(
    n: np.intp,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    edge_weight: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Compute community outflow for each community.

    Returns flows[c] = total edge weight leaving/within community c.

    For random-walk model: outflow of community c = sum of all edge
    weights where at least one endpoint is in c.
    """
    flows = np.zeros(n, dtype=np.float64)

    for u in range(n):
        c_u = labels[u]
        for e in range(row_ptr[u], row_ptr[u + 1]):
            v = col_idx[e]
            c_v = labels[v]
            w = edge_weight[e]
            flows[c_u] += w
            if c_u != c_v:
                flows[c_v] += w

    return flows


@njit(cache=True, parallel=False)
def _compute_lmdl_csr(
    n: np.intp,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    edge_weight: np.ndarray,
    labels: np.ndarray,
    two_m: float,
) -> float:
    """Compute LMDL (Level-modified Description Length) for Infomap.

    Mirrors Rust infomap.rs LMDL computation.

    LMDL = -[ sum_c( m_c * log(m_c / 2m) + sum_{i->c, j->c} w_ij * log(1 / q_c) ) ]

    Where:
        m_c = intra-flow of community c
        q_c = outflow of community c
        2m = total edge weight
    """
    if two_m <= 0.0:
        return 0.0

    # Compute flows
    flows = _compute_community_flow_csr(n, row_ptr, col_idx, edge_weight, labels)

    # Compute intra-flow per community
    intra = np.zeros(n, dtype=np.float64)
    for u in range(n):
        c_u = labels[u]
        for e in range(row_ptr[u], row_ptr[u + 1]):
            v = col_idx[e]
            c_v = labels[v]
            if c_u == c_v:
                intra[c_u] += edge_weight[e]

    # LMDL: -sum over communities of description length
    lmdl = 0.0
    visited = np.zeros(n, dtype=np.int32)

    for u in range(n):
        c = labels[u]
        if visited[c] == 1:
            continue
        visited[c] = 1

        q_c = flows[c] / two_m
        if q_c <= 0.0:
            continue

        # Node entropy: q_c * log(q_c)
        lmdl -= q_c * np.log(q_c)

        # Edge entropy within community
        for e in range(row_ptr[u], row_ptr[u + 1]):
            v = col_idx[e]
            if labels[v] != c:
                continue
            w = edge_weight[e]
            # Skip if already counted (both endpoints in same community)
            # Actually we need to count each edge once
        # This is tricky without double-counting; let's use a different approach

    # Simpler: just compute the edge part separately
    # Count each intra-community edge once by only processing u where u <= v
    edge_part = 0.0
    for u in range(n):
        c_u = labels[u]
        for e in range(row_ptr[u], row_ptr[u + 1]):
            v = col_idx[e]
            if v < u:
                continue  # skip to avoid double counting
            c_v = labels[v]
            if c_u != c_v:
                continue
            w = edge_weight[e]
            q_c = flows[c_u] / two_m
            if q_c > 0.0:
                edge_part += (w / two_m) * np.log(two_m / flows[c_u])

    return lmdl + edge_part


# ── Infomap random walk init ───────────────────────────────────────


@njit(cache=True, parallel=False)
def _random_walk_init_csr(
    n: int,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    edge_weight: np.ndarray,
    degree: np.ndarray,
    self_loop: np.ndarray,
    walk_steps: int,
    walk_count: int,
    rng_seed: int,
) -> np.ndarray:
    """Numba-accelerated random-walk initialization.

    Runs random walks and assigns each node the label of its
    most-visited neighbor.

    Uses a deterministic LCG RNG (same as Rust's LcgRng).
    """
    # Compute degree including self-loops
    total_degree = np.zeros(n, dtype=np.float64)
    for u in range(n):
        total_degree[u] = degree[u] + 2.0 * self_loop[u]

    # Visits count
    visits = np.zeros(n, dtype=np.int64)

    # LCG state (same as Rust's LcgRng)
    state = np.uint64(rng_seed) if rng_seed > 0 else np.uint64(0x5DEECE66D)
    LCG_A = np.uint64(6364136223846793005)  # noqa: N806
    LCG_C = np.uint64(1)  # noqa: N806
    MASK = np.uint64(0xFFFFFFFFFFFFFFFF)  # noqa: N806
    SCALE = np.uint64(0x1FFFFF)  # noqa: N806 -- 21 bits = 2097151
    SCALE_F = 2097152.0  # noqa: N806 -- 2^21 as float

    # Run random walks
    for _walk in range(walk_count):
        # Start at random node using LCG
        state = (state * LCG_A + LCG_C) & MASK
        node = int((state >> 33) % n)

        for _step in range(walk_steps):
            visits[node] += 1
            total = total_degree[node]
            if total <= 0.0:
                break

            # Generate uniform random number in [0, total) using LCG
            state = (state * LCG_A + LCG_C) & MASK
            # Extract 21 bits and scale to [0, 1)
            frac = float((state >> 11) & SCALE) / SCALE_F
            r = frac * total

            # Walk to neighbor
            next_node = node
            for e in range(row_ptr[node], row_ptr[node + 1]):
                r -= edge_weight[e]
                if r <= 0.0:
                    next_node = col_idx[e]
                    break
            node = next_node

    # Assign label: neighbor with highest visit count
    labels = np.zeros(n, dtype=np.int32)
    for u in range(n):
        best_neighbor = u
        best_visits = visits[u]
        for e in range(row_ptr[u], row_ptr[u + 1]):
            v = col_idx[e]
            if visits[v] > best_visits:
                best_visits = visits[v]
                best_neighbor = v
        labels[u] = best_neighbor

    return labels


# ── Utility ────────────────────────────────────────────────────────


def has_numba() -> bool:
    """Check if numba acceleration is available."""
    return _HAS_NUMBA
