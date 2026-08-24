"""Bridge Graphician's graph model to the optional Rust community engine."""

from __future__ import annotations

from collections.abc import Callable

from ..._extract import (
    HAS_RUST,
    community_detection_infomap,
    community_detection_leiden,
    community_detection_louvain,
)
from ..._extract import (
    CommunityOptions as RustCommunityOptions,
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

    functions: dict[str, Callable[..., dict[str, int]] | None] = {
        "louvain": community_detection_louvain,
        "leiden": community_detection_leiden,
        "infomap": community_detection_infomap,
    }
    function = functions.get(algorithm)
    if function is None:
        return None

    nodes = [str(node_id.value) for node_id, _ in graph.nodes()]
    edges = [
        (str(src.value), str(dst.value), edge.kind.value, edge.confidence.value)
        for _, src, dst, edge in graph.edges()
    ]
    native_options = RustCommunityOptions(
        resolution=options.resolution,
        max_passes=options.max_passes,
        max_levels=options.max_levels,
        well_connectedness=options.well_connectedness,
        min_modularity_gain=options.min_modularity_gain,
    )
    labels = function(nodes, edges, native_options)
    return {NodeId(int(node)): int(label) for node, label in labels.items()}
