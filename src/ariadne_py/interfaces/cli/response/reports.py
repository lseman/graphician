"""Report generation and graph export.

Mirrors the Rust ``reports.rs`` module.
"""

from __future__ import annotations

from typing import Any

from .analysis import bridge_nodes_json, diagnostics_json, gaps_json, surprises_json
from .architecture import architecture_overview_json


def generate_report_markdown(db_path: str, top: int = 25) -> str:
    """Generate a Markdown report from the graph and diagnostics.

    Args:
        db_path: Path to the database.
        top: Max items per section.

    Returns:
        Markdown report string.
    """
    from ....persistence.store import GraphStore

    store = GraphStore(db_path)
    try:
        graph = store.load_graph()
        diag = diagnostics_json(db_path, top)
        arch = architecture_overview_json(graph, "standard")

        # PageRank-like ranking (simple degree-based)
        degrees: dict[int, int] = {}
        for _, src, dst, _ in graph.edges():
            degrees[src] = degrees.get(src, 0) + 1
            degrees[dst] = degrees.get(dst, 0) + 1
        sorted_degrees = sorted(degrees.items(), key=lambda x: -x[1])
        top_nodes = []
        for nid, _ in sorted_degrees[:top]:
            node = graph.node(nid) if hasattr(graph, "node") else None
            if node is None:
                for _, n in graph.nodes():
                    if _ids_match(n, nid):
                        node = n
                        break
            if node:
                top_nodes.append(node.qualified_name)

        # Bridges
        bridges = bridge_nodes_json(graph, top)
        top_bridges = []
        for b in bridges.get("hits", [])[:10]:
            top_bridges.append(f"{b.get('qualified_name', '?')} (score: {b.get('score', 0):.4f})")

        # Gaps
        gap_data = gaps_json(graph, top)
        top_gaps = [
            h.get("qualified_name", "?")
            for h in gap_data.get("hits", [])
        ]

        # Surprises
        surprise_data = surprises_json(graph, top)
        top_surprises = [
            f"{h.get('src', '?')} ↔ {h.get('dst', '?')} ({h.get('score', 0):.2f})"
            for h in surprise_data.get("hits", [])
        ]

        # Communities
        comm_lines = []
        for comm in arch.get("communities", [])[:10]:
            members = comm.get("kind_counts", [])
            comm_lines.append(
                f"- **Community {comm.get('community', '?')}** "
                f"({comm.get('size', 0)} members): "
                f"{', '.join(m.get('kind', '?') for m in members[:3])}"
            )

        # Build markdown
        md = []
        md.append("# Ariadne Graph Report\n")
        md.append(f"Generated from: `{db_path}`\n")

        # Health
        md.append("## Health\n")
        md.append(f"- **Status**: {diag.get('health', 'unknown')}\n")
        for w in diag.get("warnings", []):
            kind = w.get("kind", "unknown")
            msg = w.get("message", "")
            md.append(f"- ⚠️ **{kind}**: {msg}\n")

        # Confidence
        md.append("## Confidence Mix\n")
        cm = diag.get("confidence_mix", {})
        for key in ("extracted", "inferred", "ambiguous"):
            val = cm.get(key, 0)
            if val:
                md.append(f"- **{key}**: {val}\n")

        # Call resolution
        md.append("## Call Resolution\n")
        cr = diag.get("call_resolution", {})
        resolved = cr.get("resolved", 0)
        unresolved = cr.get("unresolved", 0)
        rate = cr.get("rate", 0.0)
        md.append(f"- Resolved: {resolved}\n")
        md.append(f"- Unresolved: {unresolved}\n")
        md.append(f"- Rate: {rate:.1f}%\n")

        # Architecture
        md.append("## Architecture\n")
        overview = arch.get("summary", "")
        if overview:
            md.append(f"{overview}\n")
        comm_count = arch.get("community_count", 0)
        md.append(f"Communities: {comm_count}\n")

        # Top communities
        md.append("## Top Communities\n")
        if comm_lines:
            md.extend(comm_lines)
        else:
            md.append("_No communities detected._\n")

        # God nodes
        md.append("## Top Nodes (PageRank)\n")
        if top_nodes:
            for i, name in enumerate(top_nodes[:15], 1):
                md.append(f"{i}. {name}\n")
        else:
            md.append("_No nodes ranked._\n")

        # Bridges
        md.append("## Bridge Nodes\n")
        if top_bridges:
            md.extend(top_bridges)
        else:
            md.append("_No significant bridges._\n")

        # Gaps
        md.append("## Gaps\n")
        if top_gaps:
            md.extend(top_gaps)
        else:
            md.append("_No gaps detected._\n")

        # Surprises
        md.append("## Surprises\n")
        if top_surprises:
            md.extend(top_surprises)
        else:
            md.append("_No surprises._\n")

        # Summary
        md.append("## Summary\n")
        md.append(f"- **Nodes**: {graph.node_count()}\n")
        md.append(f"- **Edges**: {graph.edge_count()}\n")

        return "\n".join(md)
    finally:
        store.close()


def export_graphml(graph, output: str) -> dict[str, Any]:
    """Export the graph as GraphML XML.

    Args:
        graph: The code graph.
        output: Output file path.

    Returns:
        Export result dict.
    """
    from ....analysis.communities import detect_communities

    communities = detect_communities(graph)
    comm_map: dict[int, int] = {}
    for comm in communities.get("communities", []):
        for item in comm.get("nodes", []):
            node_id = graph.find_by_qname(item.get("qualified_name", ""))
            if node_id is not None:
                comm_map[node_id] = comm.get("id", 0)

    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">')
    lines.append('  <key id="qname" for="node" attr.name="qualified_name" attr.type="string"/>')
    lines.append('  <key id="kind" for="node" attr.name="kind" attr.type="string"/>')
    lines.append('  <key id="source" for="node" attr.name="source_uri" attr.type="string"/>')
    lines.append('  <key id="comm" for="node" attr.name="community" attr.type="int"/>')
    lines.append('  <key id="edge_kind" for="edge" attr.name="kind" attr.type="string"/>')
    lines.append('  <graph edgedefault="directed">')

    node_map: dict[Any, str] = {}
    idx = 0
    for nid, node in graph.nodes():
        graph_id = f"n{idx}"
        node_map[nid] = graph_id
        comm = comm_map.get(nid, -1)
        lines.append(
            f'    <node id="{graph_id}">'
            f'<data key="qname">{_xml_escape(node.qualified_name)}</data>'
            f'<data key="kind">{_xml_escape(str(node.kind))}</data>'
            f'<data key="source">{_xml_escape(node.source_uri or "")}</data>'
            f'<data key="comm">{comm}</data>'
            f'</node>'
        )
        idx += 1

    for _, src, dst, edge in graph.edges():
        src_id = node_map.get(src, "")
        dst_id = node_map.get(dst, "")
        if src_id and dst_id:
            lines.append(
                f'    <edge source="{src_id}" target="{dst_id}">'
                f'<data key="edge_kind">{_xml_escape(str(edge.kind))}</data>'
                f'</edge>'
            )

    lines.append("  </graph>")
    lines.append("</graphml>")

    xml = "\n".join(lines)
    try:
        with open(output, "w", encoding="utf-8") as f:
            f.write(xml)
        written = True
        size = len(xml)
    except OSError as e:
        written = False
        size = 0

    return {
        "operation": "export_graphml",
        "output": output,
        "format": "graphml",
        "written": written,
        "size": size,
    }


# ── Helpers ────────────────────────────────────────────────────────


def _xml_escape(text: str) -> str:
    """Escape XML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _ids_match(node: Any, node_id: Any) -> bool:
    """Check if a node matches a node ID."""
    if not hasattr(node, "id"):
        return False
    return node.id == node_id
