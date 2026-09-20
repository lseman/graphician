"""Graph export formats.

Exports the code graph as GraphML XML — the de facto standard for graph
interchange, supported by Gephi, yEd, Cytoscape, and many other tools.
"""

from __future__ import annotations

from typing import Any

from ..core.id import NodeId


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

    parts.append('  <graph id="graphician" edgedefault="directed">')

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

def export_to_cypher(
    graph,
    community_map: dict[int, int] | None = None,
) -> str:
    """Export the graph as Cypher statements for Neo4j / FalkorDB.

    Generates ``MERGE`` statements for nodes (with labels from kind)
    and ``MERGE`` statements for edges.  Community membership is set
    as a ``community_id`` property.

    Args:
        graph: The graph to export.
        community_map: Optional node_id -> community_id mapping.

    Returns:
        Cypher script as a string, suitable for ``neo4j-admin`` import
        or ``cypher-shell -f`` execution.
    """
    community_map = community_map or {}
    lines: list[str] = []
    lines.append("// Graphician graph — Neo4j / FalkorDB import")
    lines.append(f"// Nodes: {sum(1 for _ in graph.nodes())}  Edges: {sum(1 for _ in graph.edges())}")
    lines.append("")

    # --- Node statements ---
    for nid, node in graph.nodes():
        idx = nid.value
        qn = _cypher_escape(node.qualified_name)
        name = _cypher_escape(node.name)
        kind = node.kind.value
        source_uri = _cypher_escape(node.source_uri or "")
        comm_id = community_map.get(idx)
        props = []
        props.append(f'qualified_name: "{qn}"')
        props.append(f'name: "{name}"')
        if source_uri:
            props.append(f'source_uri: "{source_uri}"')
        if comm_id is not None:
            props.append(f'community_id: {comm_id}')
        props_str = "{" + ", ".join(props) + "}"
        lines.append(f'MERGE (n:`{kind}` {{id: {idx}}}) SET n += {props_str}')

    lines.append("")

    # --- Edge statements ---
    for _eid, src, dst, edge in graph.edges():
        src_qn = graph.node(src).qualified_name if graph.node(src) else ""
        dst_qn = graph.node(dst).qualified_name if graph.node(dst) else ""
        src_qn_esc = _cypher_escape(src_qn)
        dst_qn_esc = _cypher_escape(dst_qn)
        edge_kind = _cypher_escape(edge.kind.value)
        confidence = edge.confidence
        props = [f'edge_kind: "{edge_kind}"']
        props.append(f'confidence: "{confidence}"')
        score = edge.properties.get("score", 0.0)
        props.append(f'score: {score:.3f}')
        props_str = "{" + ", ".join(props) + "}"
        lines.append(
            f'MERGE (a {{qualified_name: "{src_qn_esc}"}})'
            f'MERGE (b {{qualified_name: "{dst_qn_esc}"}})'
            f'MERGE (a)-[r:`{edge_kind}`]->(b) SET r += {props_str}'
        )

    lines.append("")
    return "\n".join(lines)


def _cypher_escape(s: str) -> str:
    """Escape a string for use inside a Cypher string literal."""
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )

def export_wiki(
    graph,
    communities: dict[int, set[int]],
    hub_nodes: dict[int, int] | None = None,
    cohesion: dict[int, float] | None = None,
    bridge_nodes: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Generate wiki-style markdown articles for each community.

    Produces one ``.md`` file per community (named ``community_N.md``)
    plus an ``index.md`` that links all communities together.
    Each article contains:
      - Community name (from highest-degree hub node)
      - Cohesion score
      - Hub and bridge nodes
      - Member list grouped by kind
      - Key intra-community edges
      - Cross-community bridges

    Args:
        graph: The graph to export.
        communities: Mapping of community_id -> set of node_ids.
        hub_nodes: Optional mapping of community_id -> hub node_id.
        cohesion: Optional community_id -> cohesion score.
        bridge_nodes: Optional list of bridge dicts from analysis.

    Returns:
        Dict mapping filenames to markdown content.
    """
    hub_nodes = hub_nodes or {}
    cohesion = cohesion or {}
    bridge_nodes = bridge_nodes or []
    output: dict[str, str] = {}

    # Sort communities by size descending for the index
    sorted_communities = sorted(
        communities.items(),
        key=lambda kv: len(kv[1]),
        reverse=True,
    )

    # --- Index ---
    index_parts = [
        "# Community Index",
        "",
        f"**{len(sorted_communities)} communities** detected in the codebase.",
        "",
        "| # | Community | Size | Hub | Cohesion |",
        "|---|-----------|------|-----|----------|",
    ]

    for idx, (cid, members) in enumerate(sorted_communities, 1):
        hub_id = hub_nodes.get(cid)
        hub_name = ""
        if hub_id is not None:
            hub_node = graph.node(NodeId(hub_id))
            if hub_node:
                hub_name = f" [{hub_node.qualified_name}]"
        coh = cohesion.get(cid, 0.0)
        index_parts.append(
            f"| {idx} | [community_{cid}.md](community_{cid}.md) | {len(members)} | {hub_name} | {coh:.3f} |"
        )

    index_parts.append("")
    output["index.md"] = "\n".join(index_parts)

    # --- Per-community articles ---
    for cid, members in sorted_communities:
        parts = [f"# Community {cid}", ""]

        # Hub node
        hub_id = hub_nodes.get(cid)
        hub_node = None
        if hub_id is not None:
            hub_node = graph.node(NodeId(hub_id))
        if hub_node:
            parts.append(f"**Hub:** [{hub_node.qualified_name}](#{_slug(hub_node.name)})")
            parts.append(f"**Kind:** {hub_node.kind.value}")
            if hub_node.source_uri:
                parts.append(f"**Source:** {hub_node.source_uri}")
            if hub_node.line_start:
                parts.append(f"**Line:** {hub_node.line_start}")
            parts.append("")

        # Cohesion
        parts.append(f"**Cohesion:** {cohesion.get(cid, 0.0):.3f}")
        parts.append("")

        # Members by kind
        members_by_kind: dict[str, list] = {}
        for mid in members:
            node = graph.node(NodeId(mid))
            if node:
                members_by_kind.setdefault(node.kind.value, []).append(node)

        parts.append("## Members")
        parts.append("")
        for kind in sorted(members_by_kind):
            parts.append(f"### {kind}")
            parts.append("")
            for node in sorted(members_by_kind[kind], key=lambda n: n.qualified_name):
                uri = f" [{node.source_uri}]" if node.source_uri else ""
                parts.append(f"- `{node.qualified_name}`{uri}")
            parts.append("")

        # Intra-community edges
        parts.append("## Key Connections")
        parts.append("")
        member_set = set(members)
        edge_count = 0
        for _, src, dst, edge in graph.edges():
            if src.value in member_set and dst.value in member_set:
                src_node = graph.node(src)
                dst_node = graph.node(dst)
                if src_node and dst_node:
                    parts.append(
                        f"- {src_node.qualified_name} --[{edge.kind.value}]--> {dst_node.qualified_name}"
                    )
                    edge_count += 1
                    if edge_count >= 30:
                        parts.append(f"- ... and {len(members) * (len(members) - 1) - edge_count} more")
                        break
        if edge_count == 0:
            parts.append("- No intra-community edges detected")
        parts.append("")

        # Bridges
        community_bridges = [
            b for b in bridge_nodes
            if b.get("community_id") == cid
        ]
        if community_bridges:
            parts.append("## Bridge Nodes")
            parts.append("")
            for b in community_bridges[:10]:
                parts.append(f"- `{b['qualified_name']}` (betweenness: {b['betweenness']:.4f})")
            parts.append("")

        output[f"community_{cid}.md"] = "\n".join(parts)

    return output


def _slug(name: str) -> str:
    """Simple slug generator for markdown anchors."""
    return name.lower().replace(" ", "-").replace("_", "-")
