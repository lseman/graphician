"""Generate a markdown wiki from community structure.

For each community, generates a markdown page with:
- Overview (size, cohesion, dominant language)
- Members table (top 50 non-file nodes)
- Execution flows through the community
- Cross-community dependencies (incoming/outgoing)

Also generates an ``index.md`` linking all community pages.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ...core.edge import EdgeKind
from ...core.node import NodeKind

logger = logging.getLogger(__name__)


def cmd_wiki(db_path: str, output: str, force: bool = False) -> dict[str, Any]:
    """Generate a markdown wiki from the community structure.

    Args:
        db_path: Path to the Graphician SQLite database.
        output: Directory to write wiki files to.
        force: Overwrite existing files even if unchanged.

    Returns:
        Summary dict with pages_generated, pages_updated, pages_unchanged.
    """
    from ...persistence.store import GraphStore

    store = GraphStore(db_path)
    graph = store.load_graph()

    try:
        result = _generate_wiki(graph, output, force)
        print(
            json.dumps({
                "pages_generated": result["pages_generated"],
                "pages_updated": result["pages_updated"],
                "pages_unchanged": result["pages_unchanged"],
                "output_dir": output,
            }, indent=2))
        return result
    finally:
        store.close()


def _generate_wiki(
    graph,
    wiki_dir: str,
    force: bool = False,
) -> dict[str, Any]:
    """Generate wiki markdown for all communities.

    Args:
        graph: The code graph.
        wiki_dir: Directory to write wiki files to.
        force: Overwrite existing files even if unchanged.

    Returns:
        WikiResult dict.
    """
    communities = _compute_communities(graph)
    wiki_path = Path(wiki_dir)
    wiki_path.mkdir(parents=True, exist_ok=True)

    pages_generated = 0
    pages_updated = 0
    pages_unchanged = 0
    page_entries: list[tuple[str, str, int]] = []

    # Track slugs to avoid collisions
    used_slugs: set[str] = set()

    for comm in communities:
        base_slug = _slugify(comm["name"])
        slug = base_slug
        suffix = 2
        while slug in used_slugs:
            slug = f"{base_slug}-{suffix}"
            suffix += 1
        used_slugs.add(slug)

        content = _generate_community_page(graph, comm)
        filepath = wiki_path / f"{slug}.md"
        page_existed = filepath.exists()

        if not force and page_existed:
            try:
                existing = filepath.read_text(encoding="utf-8")
                if existing == content:
                    pages_unchanged += 1
                    page_entries.append((slug, comm["name"], comm["size"]))
                    continue
            except OSError:
                pass

        filepath.write_text(content, encoding="utf-8")
        if page_existed:
            pages_updated += 1
        else:
            pages_generated += 1
        page_entries.append((slug, comm["name"], comm["size"]))

    # Generate index.md
    index_content = _generate_index(communities, page_entries)
    index_path = wiki_path / "index.md"
    index_existed = index_path.exists()

    if not force and index_existed:
        try:
            existing = index_path.read_text(encoding="utf-8")
            if existing == index_content:
                pages_unchanged += 1
            else:
                index_path.write_text(index_content, encoding="utf-8")
                pages_updated += 1
        except OSError:
            index_path.write_text(index_content, encoding="utf-8")
            pages_updated += 1
    else:
        index_path.write_text(index_content, encoding="utf-8")
        if index_existed:
            pages_updated += 1
        else:
            pages_generated += 1

    return {
        "pages_generated": pages_generated,
        "pages_updated": pages_updated,
        "pages_unchanged": pages_unchanged,
    }


def _compute_communities(graph) -> list[dict[str, Any]]:
    """Compute communities using Leiden and retain every member."""
    from collections import defaultdict

    from ...analysis.communities.core import CommunityOptions
    from ...analysis.communities.leiden import leiden_with_options

    labels = leiden_with_options(graph, CommunityOptions())
    communities: dict[int, list] = defaultdict(list)
    for node_id, community_id in labels.items():
        communities[community_id].append(node_id)

    output: list[dict[str, Any]] = []
    for community_id, members in communities.items():
        size = len(members)

        # Determine dominant language from member files
        lang_counts: dict[str, int] = {}
        member_qns: list[str] = []
        for node_id in members:
            node = graph.node(node_id)
            if node is None:
                continue
            if node.kind == NodeKind.FILE:
                continue
            member_qns.append(node.qualified_name)
            if node.source_uri:
                ext = Path(node.source_uri).suffix.lstrip(".")
                if ext:
                    lang_counts[ext] = lang_counts.get(ext, 0) + 1

        dominant_lang = None
        if lang_counts:
            dominant_lang = max(lang_counts, key=lambda ext: lang_counts[ext])

        output.append({
            "name": f"Community {community_id}",
            "size": size,
            "cohesion": 0.0,
            "dominant_language": dominant_lang,
            "members": member_qns,
        })

    output.sort(key=lambda c: -c["size"])
    return output


def _generate_community_page(graph, comm: dict[str, Any]) -> str:
    """Generate markdown for a single community page.

    Args:
        graph: The code graph.
        comm: Community info dict.

    Returns:
        Markdown content string.
    """
    lines: list[str] = []
    lines.append(f"# {comm['name']}")
    lines.append("")

    # Overview
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- **Size**: {comm['size']} nodes")
    lines.append(f"- **Cohesion**: {comm['cohesion']:.4f}")
    if comm.get("dominant_language"):
        lines.append(f"- **Dominant Language**: {comm['dominant_language']}")
    lines.append("")

    # Members table (top 50 non-file nodes)
    lines.append("## Members")
    lines.append("")

    member_nodes: list[tuple[int, Any]] = []
    member_set = set(comm["members"])
    for qn in comm["members"]:
        nid = graph.find_by_qname(qn)
        if nid is None:
            continue
        node = graph.node(nid)
        if node is None or node.kind == NodeKind.FILE:
            continue
        member_nodes.append((nid.value, node))
        if len(member_nodes) >= 50:
            break

    if member_nodes:
        lines.append("| Name | Kind | File | Lines |")
        lines.append("|------|------|------|-------|")
        for _nid_val, node in member_nodes:
            name = _sanitize_name(node.name)
            kind_str = node.kind.value
            file = node.source_uri or "-"
            lines_str = f"{node.line_start}-{node.line_end}" if (
                node.line_start is not None and node.line_end is not None
            ) else "-"
            lines.append(f"| {name} | {kind_str} | {file} | {lines_str} |")

        if comm["size"] > 50:
            lines.append("")
            lines.append(f"*... and {comm['size'] - 50} more members.*")
    else:
        lines.append("No non-file members found.")
    lines.append("")

    # Execution flows through community
    lines.append("## Execution Flows")
    lines.append("")

    community_flows: list[tuple[Any, Any, Any, Any]] = []
    for nid, node in graph.nodes():
        if node.kind != NodeKind.FLOW:
            continue
        flow_qn = node.qualified_name
        if flow_qn in member_set:
            crit = node.properties.get("criticality", 0.0)
            if isinstance(crit, str):
                try:
                    crit = float(crit)
                except (ValueError, TypeError):
                    crit = 0.0
            depth = node.properties.get("depth", 0)
            community_flows.append((nid, crit, flow_qn, depth))

    community_flows.sort(key=lambda x: (-x[1], x[2]))

    flow_count = 0
    for _nid, crit, flow_qn, depth in community_flows:
        if flow_count >= 10:
            break
        lines.append(
            f"- **{_sanitize_name(flow_qn)}** (criticality: {crit:.2f}, depth: {depth})"
        )
        flow_count += 1

    if flow_count == 0:
        lines.append("No execution flows pass through this community.")
    lines.append("")

    # Dependencies (cross-community edges)
    lines.append("## Dependencies")
    lines.append("")

    outgoing: dict[str, int] = {}
    incoming: dict[str, int] = {}

    member_ids: list[Any] = []
    for qn in comm["members"]:
        nid = graph.find_by_qname(qn)
        if nid is not None:
            member_ids.append(nid)

    for nid in member_ids:
        # Outgoing edges
        for dst, edge in graph.out_neighbors(nid):
            if edge.kind in (EdgeKind.CALLS, EdgeKind.DATA_FLOW):
                dst_node = graph.node(dst)
                if dst_node is None:
                    continue
                dst_qn = dst_node.qualified_name
                if dst_qn not in member_set:
                    outgoing[dst_qn] = outgoing.get(dst_qn, 0) + 1

        # Incoming edges
        for src, edge in graph.in_neighbors(nid):
            if edge.kind in (EdgeKind.CALLS, EdgeKind.DATA_FLOW):
                src_node = graph.node(src)
                if src_node is None:
                    continue
                src_qn = src_node.qualified_name
                if src_qn not in member_set:
                    incoming[src_qn] = incoming.get(src_qn, 0) + 1

    outgoing_sorted = sorted(outgoing.items(), key=lambda x: -x[1])
    incoming_sorted = sorted(incoming.items(), key=lambda x: -x[1])

    if outgoing_sorted:
        lines.append("### Outgoing")
        lines.append("")
        for target, count in outgoing_sorted[:15]:
            lines.append(f"- `{_sanitize_name(target)}` ({count} edge(s))")
        lines.append("")

    if incoming_sorted:
        lines.append("### Incoming")
        lines.append("")
        for source, count in incoming_sorted[:15]:
            lines.append(f"- `{_sanitize_name(source)}` ({count} edge(s))")
        lines.append("")

    if not outgoing_sorted and not incoming_sorted:
        lines.append("No cross-community dependencies detected.")
        lines.append("")

    return "\n".join(lines)


def _generate_index(
    communities: list[dict[str, Any]],
    page_entries: list[tuple[str, str, int]],
) -> str:
    """Generate the index.md page.

    Args:
        communities: List of community info dicts.
        page_entries: List of (slug, name, size) tuples.

    Returns:
        Markdown content string.
    """
    lines: list[str] = []
    lines.append("# Code Wiki")
    lines.append("")
    lines.append(
        "Auto-generated documentation from the code knowledge graph "
        "community structure."
    )
    lines.append("")
    lines.append(f"**Total communities**: {len(communities)}")
    lines.append("")
    lines.append("## Communities")
    lines.append("")
    lines.append("| Community | Size | Link |")
    lines.append("|-----------|------|------|")

    sorted_entries = sorted(page_entries, key=lambda x: x[1])
    for slug, name, size in sorted_entries:
        lines.append(f"| {name} | {size} | [{slug}.md]({slug}.md) |")
    lines.append("")

    return "\n".join(lines)


def _slugify(name: str) -> str:
    """Convert a name to a safe filename slug."""
    slug = "".join(
        c.lower() if c.isalnum() else "-" for c in name
    ).strip("-")
    if not slug:
        return "unnamed"
    if len(slug) > 80:
        slug = slug[:80]
    return "-".join(p for p in slug.split("-") if p)


def _sanitize_name(name: str) -> str:
    """Sanitize a name for display in markdown tables."""
    return name.replace("|", "\\|").replace("\n", " ").replace("\r", "")
