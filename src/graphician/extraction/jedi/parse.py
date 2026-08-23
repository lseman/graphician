"""Parsing of Jedi subprocess output into graph CALLS edges.

Reads JSON output from the Jedi resolution script and adds
inferred ``CALLS`` edges to the graph.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ...core.edge import Edge, EdgeKind
from ...core.graph import Graph
from ...core.id import NodeId

logger = logging.getLogger(__name__)


def parse_jedi_results(
    stdout: str,
    graph: Graph,
    existing_calls: set[tuple[str, int]],
) -> int:
    """Parse Jedi results and add CALLS edges to the graph.

    Args:
        stdout: JSON output from the Jedi resolution script.
            Format: list of [file_path, jedi_line, enclosing_qname, target_qname]
        graph: The graph to add edges to.
        existing_calls: Set of (qname, line) pairs that already have CALLS edges.

    Returns:
        Number of edges added.
    """
    try:
        results: list[list[Any]] = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return 0

    count = 0
    for result in results:
        if not isinstance(result, list) or len(result) < 4:
            continue

        file_path = str(result[0])
        jedi_line = int(result[1])
        enclosing = str(result[2])
        target = str(result[3])

        # Skip if we already have a CALLS edge from this source at this line.
        if (enclosing, jedi_line) in existing_calls:
            continue

        # Find the source and target nodes.
        src_id = graph.find_by_qname(enclosing)
        dst_id = graph.find_by_qname(target)

        if src_id is None or dst_id is None:
            continue

        # Check for duplicate edge.
        already_exists = any(
            dst == dst_id and e.kind == EdgeKind.CALLS
            for dst, e in graph.out_neighbors(src_id)
        )
        if already_exists:
            continue

        edge = Edge.inferred(EdgeKind.CALLS, 0.7)
        edge.properties["resolved_from"] = "jedi_enrichment"
        graph.add_edge(src_id, dst_id, edge)
        count += 1

    return count
