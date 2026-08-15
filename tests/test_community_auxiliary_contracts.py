from __future__ import annotations

import xml.etree.ElementTree as ET

from ariadne_py.analysis.communities import gaps
from ariadne_py.analysis.communities.nodes import (
    compute_centrality,
    find_bridge_nodes,
    find_god_nodes,
    find_hub_nodes,
    is_rank_noise,
)
from ariadne_py.analysis.communities.split import _build_subgraph
from ariadne_py.analysis.export import export_graphml
from ariadne_py.core import Edge, EdgeKind, Graph, Node, NodeKind


def _fixture_graph():
    graph = Graph()
    ids = []
    for index in range(6):
        node = Node.new(NodeKind.FUNCTION, f"app::n{index}").with_source("app.py", index + 1, index + 1)
        ids.append(graph.add_node(node))
    for target in ids[1:]:
        graph.add_edge(ids[0], target, Edge.extracted(EdgeKind.CALLS))
    graph.add_edge(ids[1], ids[2], Edge.extracted(EdgeKind.CALLS))
    return graph, ids


def test_graphml_export_is_valid_escaped_and_declares_community_data() -> None:
    graph = Graph()
    source = graph.add_node(Node.new(NodeKind.FUNCTION, "app::<source&>").with_source("a&b.py", 1, 2))
    target = graph.add_node(Node.new(NodeKind.FUNCTION, "app::target"))
    graph.add_edge(source, target, Edge.inferred(EdgeKind.CALLS, 0.7))

    xml = export_graphml(graph, {source.value: 3})
    root = ET.fromstring(xml)
    assert root.tag.endswith("graphml")
    assert "app::&lt;source&amp;&gt;" in xml
    assert 'id="community_id"' in xml
    assert '<data key="community_id">3</data>' in xml


def test_community_node_rankings_work_without_scipy() -> None:
    graph, ids = _fixture_graph()
    assert find_bridge_nodes(graph)["total"] >= 1
    assert find_hub_nodes(graph, top=1)["hub_nodes"][0]["qualified_name"] == "app::n0"
    assert find_god_nodes(graph, top=2)["total"] == 2
    centrality = compute_centrality(graph)
    assert set(centrality) == {"degree_centrality", "betweenness_centrality", "pagerank"}
    assert centrality["pagerank"]
    assert not is_rank_noise(graph.node(ids[0]))
    assert is_rank_noise(Node.new(NodeKind.FILE, "file::app.py"))
    assert is_rank_noise(Node.new(NodeKind.FUNCTION, "call::missing"))


def test_knowledge_gaps_reports_isolates_hotspots_and_single_file_communities(monkeypatch) -> None:
    graph, ids = _fixture_graph()
    isolated = graph.add_node(Node.new(NodeKind.FUNCTION, "app::isolated"))
    community_result = {
        "communities": [
            {"id": 0, "size": 6, "nodes": [{"qualified_name": f"app::n{i}"} for i in range(6)]},
            {"id": 1, "size": 1, "nodes": [{"qualified_name": "app::isolated"}]},
        ]
    }
    monkeypatch.setattr(gaps, "detect_communities", lambda graph, algorithm: community_result)

    result = gaps.knowledge_gaps(graph)

    assert any(item["qualified_name"] == "app::isolated" for item in result["isolated_nodes"])
    assert result["thin_communities"] == [{"community_id": 1, "size": 1}]
    assert result["untested_hotspots"][0]["qualified_name"] == "app::n0"
    assert result["single_file_communities"][0]["file"] == "app.py"


def test_subgraph_builder_preserves_only_selected_nodes_and_internal_edges() -> None:
    graph, ids = _fixture_graph()
    subgraph = _build_subgraph(graph, ids[:3])
    assert subgraph.node_count() == 3
    assert subgraph.edge_count() == 3
