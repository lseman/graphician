"""Bridge Graphician's graph model to the optional Rust community engine.

Uses NativeGraph for zero-string-conversion community detection.
"""

from __future__ import annotations

from collections.abc import Callable

from ..._extract import (
    HAS_RUST,
    community_detection_infomap_from_native,
    community_detection_leiden_from_native,
    community_detection_louvain_from_native,
)
from ..._extract import (
    CommunityOptions as RustCommunityOptions,
)
from ..._extract import (
    NativeGraph as RustNativeGraph,
)
from ...core.graph import Graph
from ...core.id import NodeId
from .core import CommunityOptions


def detect_native(
    graph: Graph,
    options: CommunityOptions,
    algorithm: str,
) -> dict[NodeId, int] | None:
    """Run a native algorithm, or return ``None`` when the extension is absent."""
    if not HAS_RUST or RustCommunityOptions is None:
        return None

    # Collect node IDs and edges from the Python Graph
    node_ids: list[int] = []
    node_id_map: dict[int, int] = {}  # source node_id value -> NativeGraph index
    for i, (node_id, _node) in enumerate(graph.nodes()):
        node_ids.append(node_id.value)
        node_id_map[node_id.value] = i

    edges: list[tuple[int, int, str, str]] = []
    for _edge_id, src, dst, edge in graph.edges():
        edges.append(
            (src.value, dst.value, edge.kind.value, edge.confidence.value)
        )

    if not node_ids:
        return {}

    # Build the native graph -- O(V+E) with indexed adjacency, no string conversion
    native_graph = RustNativeGraph(node_ids, edges)

    native_options = RustCommunityOptions(
        resolution=options.resolution,
        max_passes=options.max_passes,
        max_levels=options.max_levels,
        well_connectedness=options.well_connectedness,
        min_modularity_gain=options.min_modularity_gain,
    )

    # Dispatch to the native-from-native-graph function
    functions: dict[str, Callable[..., dict[int, int]]] = {
        "louvain": community_detection_louvain_from_native,
        "leiden": community_detection_leiden_from_native,
        "infomap": community_detection_infomap_from_native,
    }
    function = functions.get(algorithm)
    if function is None:
        return None

    labels = function(native_graph, native_options)
    return {NodeId(int(node_id)): int(label) for node_id, label in labels.items()}
