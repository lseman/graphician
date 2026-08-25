"""Numba-accelerated graph algorithms.

Provides JIT-compiled versions of:
- Betweenness centrality (BFS-based, approximated)
- K-core decomposition
- Articulation points (iterative Tarjan's)

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


# ── Betweenness Centrality ─────────────────────────────────────────


@njit(cache=True, parallel=False)
def _betweenness_bfs_csr(
    source: int,
    n: int,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BFS from a single source, returning data needed for betweenness.

    Returns (dist, pred, order) where:
    - dist: shortest distance from source to each node
    - pred: predecessors list (encoded as start/length in parallel arrays)
    - order: BFS order (for reverse dependency accumulation)

    Note: pred is stored as flat arrays with offsets.
    """
    dist = np.full(n, -1, dtype=np.int32)
    dist[source] = 0

    queue = np.zeros(n, dtype=np.int32)
    head = 0
    tail = 0

    queue[tail] = source
    tail += 1

    # Order of nodes as they are first visited
    order = np.zeros(n, dtype=np.int32)
    order_count = 0

    # Store predecessors using a flat array + offsets approach
    # Pre-allocate max possible (each node can have at most n predecessors)
    pred_starts = np.zeros(n, dtype=np.int32)
    pred_counts = np.zeros(n, dtype=np.int32)

    while head < tail:
        node = queue[head]
        head += 1
        order[order_count] = node
        order_count += 1

        next_dist = dist[node] + 1

        # Iterate neighbors
        for e in range(row_ptr[node], row_ptr[node + 1]):
            neighbor = col_idx[e]
            if dist[neighbor] == -1:
                # First time visiting this node
                dist[neighbor] = next_dist
                pred_starts[neighbor] = 0
                pred_counts[neighbor] = 0
                queue[tail] = neighbor
                tail += 1
            elif dist[neighbor] == next_dist:
                # Another shortest path to this node
                pass

    # Second pass: collect predecessors (simplified - just return counts)
    # For full betweenness, we need the predecessor lists
    # This simplified version returns enough for dependency accumulation

    return dist, pred_starts, order[:order_count]


@njit(cache=True, parallel=False)
def _betweenness_accumulate(
    n: int,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    sources: np.ndarray,
    max_sources: int,
) -> np.ndarray:
    """Compute approximate betweenness centrality using BFS from multiple sources.

    Uses the Brandes algorithm approximation with a fixed number of sources.

    Args:
        n: Number of nodes.
        row_ptr: CSR row pointers.
        col_idx: Column indices.
        sources: Array of source node indices.
        max_sources: Number of sources to sample.

    Returns:
        Betweenness centrality scores as numpy array.
    """
    scores = np.zeros(n, dtype=np.float64)

    actual_sources = min(max_sources, len(sources))

    for s_idx in range(actual_sources):
        source = sources[s_idx]

        # BFS from source
        dist = np.full(n, -1, dtype=np.int32)
        dist[source] = 0

        queue = np.zeros(n, dtype=np.int32)
        head = 0
        tail = 0
        queue[tail] = source
        tail += 1

        # Collect BFS order
        order = np.zeros(n, dtype=np.int32)
        order_count = 0

        path_count = np.zeros(n, dtype=np.float64)
        path_count[source] = 1.0

        while head < tail:
            node = queue[head]
            head += 1
            order[order_count] = node
            order_count += 1

            next_dist = dist[node] + 1

            for e in range(row_ptr[node], row_ptr[node + 1]):
                neighbor = col_idx[e]
                if dist[neighbor] == -1:
                    dist[neighbor] = next_dist
                    queue[tail] = neighbor
                    tail += 1
                if dist[neighbor] == next_dist:
                    path_count[neighbor] += path_count[node]

        # Dependency accumulation
        dependency = np.zeros(n, dtype=np.float64)
        for i in range(order_count - 1, -1, -1):
            node = order[i]
            if path_count[node] > 0.0:
                coeff = (1.0 + dependency[node]) / path_count[node]
                for e in range(row_ptr[node], row_ptr[node + 1]):
                    predecessor = col_idx[e]
                    if dist[predecessor] == dist[node] - 1:
                        dependency[predecessor] += path_count[predecessor] * coeff
            if node != source:
                scores[node] += dependency[node]

    return scores


# ── K-core Decomposition ───────────────────────────────────────────


@njit(cache=True, parallel=False)
def _kcore_csr(
    n: int,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
) -> np.ndarray:
    """K-core decomposition using the peeling algorithm.

    Iteratively removes nodes with degree < k until no more nodes
    can be removed. The k-value of each node is its core number.

    Uses an efficient bucket-based approach:
    - Nodes are stored in buckets by current degree
    - When a node is removed, neighbors' degrees are decremented

    Args:
        n: Number of nodes.
        row_ptr: CSR row pointers.
        col_idx: Column indices.

    Returns:
        Core number for each node as numpy array.
    """
    # Compute initial degrees
    degree = np.zeros(n, dtype=np.int32)
    for u in range(n):
        degree[u] = row_ptr[u + 1] - row_ptr[u]

    # Find maximum degree
    max_deg = 0
    for d in degree:
        if d > max_deg:
            max_deg = d

    # Bucket sort: group nodes by degree
    # bucket[k] = linked list of nodes with degree k
    # We use a simple array with next pointers
    bucket_head = np.zeros(max_deg + 2, dtype=np.int32)  # -1 means empty
    node_bucket = np.full(n, -1, dtype=np.int32)  # which bucket slot
    bucket_next = np.full(n, -1, dtype=np.int32)  # next node in same bucket

    # Initialize buckets
    for i in range(n):
        d = degree[i]
        if d <= max_deg:
            bucket_next[i] = bucket_head[d]
            bucket_head[d] = i
            node_bucket[i] = d

    # K-core decomposition (peeling algorithm)
    core = np.zeros(n, dtype=np.int32)
    k = 0

    while k <= max_deg:
        # Find the next non-empty bucket at or above k
        while bucket_head[k] == -1 and k <= max_deg:
            k += 1

        if k > max_deg:
            break

        # Process all nodes in bucket k
        while bucket_head[k] != -1:
            u = bucket_head[k]
            bucket_head[k] = bucket_next[u]  # Remove from bucket

            core[u] = k

            # Decrease degree of neighbors
            for e in range(row_ptr[u], row_ptr[u + 1]):
                v = col_idx[e]
                if core[v] == 0 and degree[v] > k:
                    # Remove v from its current bucket
                    old_bucket = degree[v] - 1  # degree is 1-based in bucket
                    if node_bucket[v] >= 0:
                        # Find and remove v from old bucket
                        prev = -1
                        curr = bucket_head[old_bucket]
                        while curr != -1:
                            if curr == v:
                                break
                            prev = curr
                            curr = bucket_next[curr]
                        if prev == -1:
                            bucket_head[old_bucket] = bucket_next[curr]
                        else:
                            bucket_next[prev] = bucket_next[curr]

                    # Decrement degree and add to new bucket
                    degree[v] -= 1
                    new_deg = degree[v]
                    if new_deg > k:
                        # Add to new bucket (new_deg - 1, since bucket is 0-indexed by degree-1)
                        bucket_next[v] = bucket_head[new_deg]
                        bucket_head[new_deg] = v
                        node_bucket[v] = new_deg

        k += 1

    return core


# ── Articulation Points ────────────────────────────────────────────


@njit(cache=True, parallel=False)
def _articulation_points_csr(
    n: int,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
) -> np.ndarray:
    """Find articulation points using iterative Tarjan's algorithm.

    Returns a boolean array where True indicates an articulation point.

    Uses an iterative DFS with explicit stack to avoid recursion limits.

    Args:
        n: Number of nodes.
        row_ptr: CSR row pointers.
        col_idx: Column indices.

    Returns:
        Boolean array of articulation points.
    """
    is_articulation = np.zeros(n, dtype=np.int32)
    visited = np.zeros(n, dtype=np.int32)
    discovery = np.zeros(n, dtype=np.int32)
    low = np.zeros(n, dtype=np.int32)

    time_counter = np.array([0], dtype=np.int32)

    # Iterative DFS using explicit stack
    # Stack entries: (node, parent, neighbor_index)
    stack_node = np.zeros(n * 4, dtype=np.int32)
    stack_parent = np.zeros(n * 4, dtype=np.int32)
    stack_nidx = np.zeros(n * 4, dtype=np.int32)
    stack_top = 0

    for start in range(n):
        if visited[start] == 1:
            continue

        # Initialize root
        visited[start] = 1
        discovery[start] = time_counter[0]
        low[start] = time_counter[0]
        time_counter[0] += 1

        # Push root onto stack
        stack_node[stack_top] = start
        stack_parent[stack_top] = -1
        stack_nidx[stack_top] = 0
        stack_top += 1

        child_count = np.zeros(n, dtype=np.int32)

        while stack_top > 0:
            u = stack_node[stack_top - 1]
            parent = stack_parent[stack_top - 1]
            nidx = stack_nidx[stack_top - 1]

            if nidx < row_ptr[u + 1]:
                # Process next neighbor
                v = col_idx[row_ptr[u] + nidx]
                stack_nidx[stack_top - 1] = nidx + 1

                if visited[v] == 0:
                    # Tree edge
                    visited[v] = 1
                    discovery[v] = time_counter[0]
                    low[v] = time_counter[0]
                    time_counter[0] += 1

                    # Push v onto stack
                    stack_node[stack_top] = v
                    stack_parent[stack_top] = u
                    stack_nidx[stack_top] = 0
                    stack_top += 1
                elif v != parent:
                    # Back edge
                    low[u] = min(low[u], discovery[v])
            else:
                # All neighbors processed, pop
                stack_top -= 1

                if parent >= 0:
                    # Update parent's low value
                    low[parent] = min(low[parent], low[u])

                    # Check articulation point condition
                    if parent != -1 and low[u] >= discovery[parent]:
                        child_count[parent] += 1
                        if child_count[parent] > 1:
                            is_articulation[parent] = 1
                elif parent == -1:
                    # Root is articulation if it has > 1 children
                    if child_count[u] > 1:
                        is_articulation[u] = 1

    return is_articulation


# ── Public API ─────────────────────────────────────────────────────


def has_numba() -> bool:
    """Check if numba acceleration is available."""
    return _HAS_NUMBA


def betweenness_approx_csr(
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    sources: np.ndarray,
    max_sources: int = 64,
) -> np.ndarray:
    """Compute approximate betweenness centrality using CSR format.

    Args:
        row_ptr: CSR row pointers.
        col_idx: Column indices.
        sources: Source node indices.
        max_sources: Maximum sources to sample.

    Returns:
        Betweenness centrality scores.
    """
    n = len(row_ptr) - 1
    return _betweenness_accumulate(n, row_ptr, col_idx, sources, max_sources)


def kcore_csr(
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
) -> np.ndarray:
    """Compute k-core decomposition using CSR format.

    Args:
        row_ptr: CSR row pointers.
        col_idx: Column indices.

    Returns:
        Core number for each node.
    """
    n = len(row_ptr) - 1
    return _kcore_csr(n, row_ptr, col_idx)


def articulation_points_csr(
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
) -> np.ndarray:
    """Find articulation points using CSR format.

    Args:
        row_ptr: CSR row pointers.
        col_idx: Column indices.

    Returns:
        Boolean array of articulation points.
    """
    n = len(row_ptr) - 1
    return _articulation_points_csr(n, row_ptr, col_idx)
