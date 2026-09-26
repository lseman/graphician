"""Knowledge gaps — structural weaknesses in the codebase graph."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ...core.edge import EdgeKind
from ...core.graph import Graph
from ...core.node import NodeKind
from .core import CommunityOptions, WorkingGraph
from .leiden import _local_move, _refinement_phase


def knowledge_gaps(graph: Graph) -> dict[str, Any]:
    """Knowledge gaps — structural weaknesses in the codebase graph.

    Identifies:
    - isolated_nodes: nodes with degree <= 1 (excluding File nodes)
    - thin_communities: communities with fewer than 3 members
    - untested_hotspots: high-degree (>= 5) nodes with no TestedBy edge
    - single_file_communities: communities of size >= 3 all in one file
    """
    # Build node lookup: index -> Node
    node_by_idx: dict[int, Any] = {}
    for nid, node in graph.nodes():
        node_by_idx[nid.value] = node

    # Compute degrees and tested nodes
    degree: dict[int, int] = defaultdict(int)
    tested_nodes: set[int] = set()
    for _, src, dst, edge in graph.edges():
        degree[src.value] += 1
        degree[dst.value] += 1
        if edge.kind == EdgeKind.TESTED_BY:
            tested_nodes.add(src.value)

    # Isolated nodes: degree <= 1, excluding File nodes
    isolated: list[dict[str, Any]] = []
    for idx, node in node_by_idx.items():
        if node.kind == NodeKind.FILE:
            continue
        d = degree.get(idx, 0)
        if d <= 1:
            isolated.append({
                "qualified_name": node.qualified_name,
                "name": node.name,
                "kind": node.kind.value,
                "file": node.source_uri,
                "degree": d,
            })

    # Community detection: run Leiden directly on WorkingGraph,
    # then map results back to node indices in a single pass.
    working = WorkingGraph.from_graph(graph)
    if working.total_weight <= 0.0:
        # No edges: every node is its own community
        comm_sizes: dict[int, int] = {i: 1 for i in range(working.len())}
        comm_files: dict[int, set[str]] = {}
        for i in range(working.len()):
            node = node_by_idx[i]
            if node.source_uri:
                comm_files[i] = {node.source_uri}
            else:
                comm_files[i] = set()
    else:
        options = CommunityOptions()
        partition = _local_move(working, options)
        partition = _refinement_phase(working, partition, options)

        # _refinement_phase already calls enforce_connected + densify internally

        # Build comm_sizes and comm_files in a single pass over nodes
        comm_sizes: dict[int, int] = defaultdict(int)
        comm_files: dict[int, set[str]] = defaultdict(set)
        for u in range(working.len()):
            cid = partition[u]
            comm_sizes[cid] += 1
            for nid in working.members[u]:
                node = node_by_idx.get(nid.value)
                if node and node.source_uri:
                    comm_files[cid].add(node.source_uri)

    # Thin communities: size < 3
    thin: list[dict[str, Any]] = [
        {"community_id": cid, "size": size}
        for cid, size in comm_sizes.items()
        if size < 3
    ]

    # Untested hotspots: degree >= 5, not tested, not File
    untested: list[dict[str, Any]] = []
    for idx, node in node_by_idx.items():
        if node.properties.get("is_test"):
            continue
        d = degree.get(idx, 0)
        if d >= 5 and idx not in tested_nodes:
            untested.append({
                "qualified_name": node.qualified_name,
                "name": node.name,
                "kind": node.kind.value,
                "file": node.source_uri,
                "degree": d,
            })

    # Single-file communities: size >= 3, all in one file
    single_file: list[dict[str, Any]] = []
    for cid, size in comm_sizes.items():
        files = comm_files.get(cid)
        if size >= 3 and files and len(files) == 1:
            single_file.append({
                "community_id": cid,
                "size": size,
                "file": next(iter(files)),
            })

    total_gaps = len(isolated) + len(thin) + len(untested) + len(single_file)

    return {
        "operation": "knowledge_gaps",
        "total_gaps": total_gaps,
        "isolated_nodes": isolated,
        "thin_communities": thin,
        "untested_hotspots": untested,
        "single_file_communities": single_file,
    }
