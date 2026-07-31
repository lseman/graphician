"""Tests for the execution flow detection engine."""

from ariadne_py.core.edge import Edge, EdgeKind
from ariadne_py.core.graph import Graph
from ariadne_py.core.node import Node, NodeKind
from ariadne_py.extraction.flows import (
    FlowOptions,
    all_flows,
    compute_flows,
    flows_through,
    affected_flows,
)


def _add_fn(graph: Graph, qname: str) -> Node:
    return Node.new(NodeKind.FUNCTION, qname)


def _add_test_fn(graph: Graph, qname: str) -> Node:
    return (
        Node.new(NodeKind.FUNCTION, qname)
            .with_property("is_test", True)
    )


class TestComputeFlows:
    def test_main_is_entry(self):
        graph = Graph()
        main_id = graph.add_node(_add_fn(graph, "file::main.rs::main"))
        helper_id = graph.add_node(_add_fn(graph, "file::main.rs::helper"))
        graph.add_edge(main_id, helper_id, Edge.extracted(EdgeKind.CALLS))
        count = compute_flows(graph, FlowOptions(min_flow_size=1))
        assert count == 1
        flows = all_flows(graph)
        assert len(flows) == 1
        flow_node = graph.node(flows[0])
        assert flow_node is not None
        assert flow_node.kind == NodeKind.FLOW

    def test_orphan_is_entry_but_called_helper_is_not(self):
        graph = Graph()
        public_id = graph.add_node(_add_fn(graph, "file::lib.rs::public_api"))
        helper_id = graph.add_node(_add_fn(graph, "file::lib.rs::helper"))
        graph.add_edge(public_id, helper_id, Edge.extracted(EdgeKind.CALLS))
        compute_flows(graph, FlowOptions(min_flow_size=1))
        flows = all_flows(graph)
        assert len(flows) == 1
        entry_qn = graph.node(flows[0])
        assert entry_qn is not None
        assert "public_api" in entry_qn.qualified_name

    def test_test_flow_has_lower_criticality(self):
        graph = Graph()
        prod_main = graph.add_node(_add_fn(graph, "file::main.rs::main"))
        prod_helper = graph.add_node(_add_fn(graph, "file::main.rs::helper"))
        graph.add_edge(prod_main, prod_helper, Edge.extracted(EdgeKind.CALLS))
        test_main = graph.add_node(_add_test_fn(graph, "file::tests.rs::test_main"))
        test_helper = graph.add_node(_add_fn(graph, "file::tests.rs::helper"))
        graph.add_edge(test_main, test_helper, Edge.extracted(EdgeKind.CALLS))
        compute_flows(graph, FlowOptions(min_flow_size=1))
        flows = all_flows(graph)
        assert len(flows) == 2
        prod_flow = next(n for n in flows if "main.rs::main" in graph.node(n).qualified_name)
        test_flow = next(n for n in flows if graph.node(n).properties.get("is_test_flow"))
        prod_crit = graph.node(prod_flow).properties.get("criticality", 0)
        test_crit = graph.node(test_flow).properties.get("criticality", 0)
        assert prod_crit > test_crit, f"{prod_crit} > {test_crit}"

    def test_bfs_respects_max_depth(self):
        graph = Graph()
        main = graph.add_node(_add_fn(graph, "file::main.rs::main"))
        level1 = graph.add_node(_add_fn(graph, "file::main.rs::level1"))
        level2 = graph.add_node(_add_fn(graph, "file::main.rs::level2"))
        graph.add_edge(main, level1, Edge.extracted(EdgeKind.CALLS))
        graph.add_edge(level1, level2, Edge.extracted(EdgeKind.CALLS))
        compute_flows(graph, FlowOptions(max_depth=1, min_flow_size=2))
        flow_id = all_flows(graph)[0]
        members = [
            nid for nid, edge in graph.in_neighbors(flow_id)
            if edge.kind in (EdgeKind.MEMBER_OF, EdgeKind.ENTRY_OF)
        ]
        assert len(members) == 2

    def test_ambiguous_placeholder_skipped(self):
        from ariadne_py.core.edge import Confidence
        graph = Graph()
        main = graph.add_node(_add_fn(graph, "file::main.rs::main"))
        real = graph.add_node(_add_fn(graph, "file::main.rs::real"))
        placeholder = graph.add_node(_add_fn(graph, "call::external"))
        graph.add_edge(main, real, Edge.extracted(EdgeKind.CALLS))
        graph.add_edge(main, placeholder, Edge(EdgeKind.CALLS, Confidence.AMBIGUOUS))
        compute_flows(graph, FlowOptions(min_flow_size=1))
        flow_id = all_flows(graph)[0]
        members = {
            nid for nid, edge in graph.in_neighbors(flow_id)
            if edge.kind in (EdgeKind.MEMBER_OF, EdgeKind.ENTRY_OF)
        }
        # Real target is reached; placeholder is skipped.
        assert main in members
        assert real in members
        assert not any(
            graph.node(n).qualified_name.startswith("call::")
            for n in members
            if graph.node(n) is not None
        )

    def test_idempotent_on_rerun(self):
        graph = Graph()
        main = graph.add_node(_add_fn(graph, "file::main.rs::main"))
        helper = graph.add_node(_add_fn(graph, "file::main.rs::helper"))
        graph.add_edge(main, helper, Edge.extracted(EdgeKind.CALLS))
        compute_flows(graph, FlowOptions(min_flow_size=1))
        flow_count_before = len(all_flows(graph))
        membership_before = sum(
            1 for _, _, _, e in graph.edges()
            if e.kind in (EdgeKind.MEMBER_OF, EdgeKind.ENTRY_OF)
        )
        compute_flows(graph, FlowOptions(min_flow_size=1))
        assert len(all_flows(graph)) == flow_count_before
        membership_after = sum(
            1 for _, _, _, e in graph.edges()
            if e.kind in (EdgeKind.MEMBER_OF, EdgeKind.ENTRY_OF)
        )
        assert membership_after == membership_before

    def test_framework_decorator_entry(self):
        graph = Graph()
        decorated = graph.add_node(
            Node.new(NodeKind.FUNCTION, "app::views::route_users")
                .with_property("decorators", ["@api.route"])
        )
        fetch = graph.add_node(_add_fn(graph, "app::db::fetch_users"))
        graph.add_edge(decorated, fetch, Edge.extracted(EdgeKind.CALLS))
        count = compute_flows(graph, FlowOptions(min_flow_size=1))
        assert count >= 1

    def test_empty_graph(self):
        graph = Graph()
        assert compute_flows(graph) == 0

    def test_min_flow_size_filter(self):
        graph = Graph()
        main = graph.add_node(_add_fn(graph, "file::main.rs::main"))
        helper = graph.add_node(Node.new(NodeKind.VARIABLE, "file::main.rs::helper"))
        graph.add_edge(main, helper, Edge.extracted(EdgeKind.CALLS))
        # Helper is VARIABLE, not FUNCTION, so not an entry point.
        # Flow has 2 members (main + helper) but min_flow_size=2 should still pass.
        count = compute_flows(graph, FlowOptions(min_flow_size=3))
        assert count == 0


def _add_node(graph: Graph, qname: str, kind: NodeKind) -> Node:
    return Node.new(kind, qname)


class TestFlowsThrough:
    def test_member_belongs_to_flow(self):
        graph = Graph()
        main = graph.add_node(_add_fn(graph, "file::main.rs::main"))
        helper = graph.add_node(_add_fn(graph, "file::main.rs::helper"))
        graph.add_edge(main, helper, Edge.extracted(EdgeKind.CALLS))
        compute_flows(graph, FlowOptions(min_flow_size=2))
        flows = flows_through(graph, helper)
        assert len(flows) == 1

    def test_unrelated_node_not_in_flow(self):
        graph = Graph()
        main = graph.add_node(_add_fn(graph, "file::main.rs::main"))
        helper = graph.add_node(_add_fn(graph, "file::main.rs::helper"))
        unrelated = graph.add_node(_add_fn(graph, "file::other.rs::unrelated"))
        graph.add_edge(main, helper, Edge.extracted(EdgeKind.CALLS))
        compute_flows(graph, FlowOptions(min_flow_size=2))
        flows = flows_through(graph, unrelated)
        assert len(flows) == 0


class TestAffectedFlows:
    def test_two_changed_nodes_two_flows(self):
        graph = Graph()
        main = graph.add_node(_add_fn(graph, "file::main.rs::main"))
        helper = graph.add_node(_add_fn(graph, "file::main.rs::helper"))
        unrelated = graph.add_node(_add_fn(graph, "file::other.rs::unrelated"))
        graph.add_edge(main, helper, Edge.extracted(EdgeKind.CALLS))
        compute_flows(graph, FlowOptions(min_flow_size=1))
        affected = affected_flows(graph, [helper])
        assert len(affected) == 1
        both = affected_flows(graph, [helper, unrelated])
        assert len(both) == 2
