"""Knowledge gaps — structural weaknesses in the codebase graph."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ...core.edge import EdgeKind
from ...core.graph import Graph
from ...core.node import NodeKind
from .louvain import detect_communities


def knowledge_gaps(graph: Graph) -> dict[str, Any]:
    """Knowledge gaps — structural weaknesses in the codebase graph.

    Identifies:
    - isolated_nodes: nodes with degree <= 1 (excluding File nodes)
    - thin_communities: communities with fewer than 3 members
    - untested_hotspots: high-degree (>= 5) nodes with no TestedBy edge
    - single_file_communities: communities of size >= 3 all in one file
    """
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
    for nid, node in graph.nodes():
        if node.kind == NodeKind.FILE:
            continue
        d = degree.get(nid.value, 0)
        if d <= 1:
            isolated.append({
                "qualified_name": node.qualified_name,
                "name": node.name,
                "kind": node.kind.value,
                "file": node.source_uri,
                "degree": d,
            })

    # Community stats
    comm_result = detect_communities(graph, algorithm="leiden")
    comm_sizes: dict[int, int] = {}
    comm_files: dict[int, set[str]] = defaultdict(set)
    for comm in comm_result.get("communities", []):
        cid = comm["id"]
        comm_sizes[cid] = comm["size"]

    # Build file membership from nodes list (limited to 20 per community)
    for comm in comm_result.get("communities", []):
        cid = comm["id"]
        for node_info in comm.get("nodes", []):
            qn = node_info.get("qualified_name", "")
            nid = graph.find_by_qname(qn)
            if nid:
                node = graph.node(nid)
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
    for nid, node in graph.nodes():
        if node.properties.get("is_test"):
            continue
        d = degree.get(nid.value, 0)
        if d >= 5 and nid.value not in tested_nodes:
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
        if size >= 3 and len(comm_files.get(cid, set())) == 1:
            files = comm_files.get(cid, set())
            single_file.append({
                "community_id": cid,
                "size": size,
                "file": next(iter(files)) if files else None,
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
