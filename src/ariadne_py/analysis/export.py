"""Graph export formats.

Exports the Ariadne graph as GraphML XML — the de facto standard for graph
interchange, supported by Gephi, yEd, Cytoscape, and many other tools.
"""

from __future__ import annotations

from typing import Any


def export_graphml(
    graph,
    community_map: dict[int, int] | None = None,
) -> str:
    """Export the graph as GraphML XML.

    Returns the XML as a string. Nodes carry kind, qualified_name, name,
    file, and community_id attributes. Edges carry kind, confidence, and
    score attributes.

    Args:
        graph: The graph to export.
        community_map: Optional node_id -> community_id mapping.

    Returns:
        GraphML XML string.
    """
    community_map = community_map or {}
    parts: list[str] = []

    parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    parts.append(
        '<graphml xmlns="http://graphml.graphstruct.org/graphml"'
        ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
        ' xsi:schemaLocation="http://graphml.graphstruct.org/graphml">'
    )

    # Node keys
    for attr, typ in [
        ("kind", "string"),
        ("qualified_name", "string"),
        ("name", "string"),
        ("file", "string"),
        ("kind_raw", "string"),
        ("community_id", "int"),
    ]:
        parts.append(
            f'  <key id="{attr}" for="node" attr.name="{attr}" attr.type="{typ}"/>'
        )

    # Edge keys
    for attr, typ in [
        ("edge_kind", "string"),
        ("confidence", "string"),
        ("score", "double"),
        ("source_file", "string"),
        ("target_file", "string"),
    ]:
        parts.append(
            f'  <key id="{attr}" for="edge" attr.name="{attr}" attr.type="{typ}"/>'
        )

    parts.append('  <graph id="ariadne" edgedefault="directed">')

    # Nodes
    for nid, node in graph.nodes():
        idx = nid.value
        qn = _xml_escape(node.qualified_name)
        name = _xml_escape(node.name)
        kind = _xml_escape(node.kind)
        file_ = _xml_escape(node.source_uri or "")
        comm_id = community_map.get(idx)

        parts.append(f'    <node id="n{idx}">')
        parts.append(f'      <data key="qualified_name">{qn}</data>')
        parts.append(f'      <data key="name">{name}</data>')
        parts.append(f'      <data key="kind">{kind}</data>')
        parts.append(f'      <data key="file">{file_}</data>')
        parts.append(f'      <data key="kind_raw">{kind}</data>')
        if comm_id is not None:
            parts.append(f'      <data key="community_id">{comm_id}</data>')
        parts.append('    </node>')

    # Edges
    for eid, src, dst, edge in graph.edges():
        src_kind = _xml_escape(edge.kind)
        if edge.confidence == "extracted":
            conf_str = "extracted"
        elif edge.confidence == "inferred":
            score = edge.properties.get("score", 0.0)
            conf_str = f"inferred:{score:.3f}"
        else:
            conf_str = "ambiguous"

        score = _confidence_score(edge.confidence)
        src_node = graph.node(src)
        dst_node = graph.node(dst)
        src_file = _xml_escape((src_node.source_uri if src_node else None) or "")
        dst_file = _xml_escape((dst_node.source_uri if dst_node else None) or "")

        parts.append(
            f'    <edge id="e{eid.value}" source="n{src.value}" target="n{dst.value}">'
        )
        parts.append(f'      <data key="edge_kind">{src_kind}</data>')
        parts.append(f'      <data key="confidence">{conf_str}</data>')
        parts.append(f'      <data key="score">{score:.3f}</data>')
        parts.append(f'      <data key="source_file">{src_file}</data>')
        parts.append(f'      <data key="target_file">{dst_file}</data>')
        parts.append('    </edge>')

    parts.append("  </graph>")
    parts.append("</graphml>")

    return "\n".join(parts)


def _xml_escape(s: str) -> str:
    """Minimal XML escaping for attribute-safe strings."""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _confidence_score(confidence: str) -> float:
    """Convert confidence string to numeric score."""
    if confidence == "extracted":
        return 1.0
    return 0.0
