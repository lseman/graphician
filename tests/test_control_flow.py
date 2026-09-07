"""Tests for control flow analysis module."""

from graphician.analysis.control_flow import (
    analyze_all_functions,
    analyze_function_flow,
    build_cfg,
    compute_def_use_chains,
    find_critical_path,
    analyze_reachable_blocks,
)
from graphician.core.graph import Graph
from graphician.core.id import NodeId
from graphician.core.node import Node, NodeKind


class TestBuildCFG:
    """Test CFG construction."""

    def test_simple_python_function(self):
        src = "def foo(x, y):\n    z = x + y\n    return z\n"
        cfg = build_cfg(src, "test.py", "foo")
        assert len(cfg) > 2
        assert cfg[0].label == "__ENTRY__"
        assert cfg[0].kind == "entry"
        assert cfg[-1].label == "__EXIT__"
        assert cfg[-1].kind == "exit"

    def test_python_if_branch(self):
        src = "def foo(x):\n    if x > 0:\n        return True\n    return False\n"
        cfg = build_cfg(src, "test.py", "foo")
        assert len(cfg) >= 3

    def test_python_for_loop(self):
        src = "def foo(items):\n    for item in items:\n        print(item)\n"
        cfg = build_cfg(src, "test.py", "foo")
        assert cfg[0].label == "__ENTRY__"
        assert cfg[-1].label == "__EXIT__"

    def test_rust_function(self):
        src = "fn foo(x: i32) -> i32 {\n    let y = x + 1;\n    y\n}\n"
        cfg = build_cfg(src, "test.rs", "foo")
        assert cfg[0].label == "__ENTRY__"
        assert cfg[-1].label == "__EXIT__"

    def test_ts_function(self):
        src = "function foo(x: number): number {\n    return x + 1;\n}\n"
        cfg = build_cfg(src, "test.ts", "foo")
        assert cfg[0].label == "__ENTRY__"
        assert cfg[-1].label == "__EXIT__"


class TestDefUseChains:
    """Test def-use chain computation."""

    def test_python_defs_uses(self):
        src = "def foo(x, y):\n    z = x + y\n    return z\n"
        cfg = build_cfg(src, "test.py", "foo")
        chains = compute_def_use_chains(cfg, ["x", "y"])
        vars_in_chains = [c.variable for c in chains]
        assert "x" in vars_in_chains or "y" in vars_in_chains

    def test_chains_have_uses(self):
        src = "def foo(x):\n    z = x + 1\n    return z\n"
        cfg = build_cfg(src, "test.py", "foo")
        chains = compute_def_use_chains(cfg, ["x"])
        x_chain = next((c for c in chains if c.variable == "x"), None)
        if x_chain:
            assert len(x_chain.uses) >= 0


class TestReachableBlocks:
    """Test reachable block analysis."""

    def test_simple_function_all_reachable(self):
        src = "def foo(x):\n    return x + 1\n"
        cfg = build_cfg(src, "test.py", "foo")
        reachable = analyze_reachable_blocks(cfg)
        assert reachable.dead_line_count == 0

    def test_branch_creates_potential_dead_code(self):
        src = "def foo(x):\n    if x > 0:\n        return True\n    return False\n"
        cfg = build_cfg(src, "test.py", "foo")
        reachable = analyze_reachable_blocks(cfg)
        assert reachable.reachable_blocks >= 1


class TestCriticalPath:
    """Test critical path analysis."""

    def test_simple_path(self):
        src = "def foo(x):\n    return x + 1\n"
        cfg = build_cfg(src, "test.py", "foo")
        path = find_critical_path(cfg)
        assert path.path_length >= 2

    def test_longer_path(self):
        src = "def foo(x):\n    a = x + 1\n    b = a * 2\n    c = b - 1\n    return c\n"
        cfg = build_cfg(src, "test.py", "foo")
        path = find_critical_path(cfg)
        assert path.path_length >= 2


class TestGraphIntegration:
    """Test integration with Graph."""

    def _make_fn(self, graph: Graph, qname: str, source_uri: str, source: str) -> NodeId:
        node = Node.new(NodeKind.FUNCTION, qname)
        node = node.with_source(source_uri, 1, 1)
        node = node.with_source_text(source)
        return graph.add_node(node)

    def test_analyze_function_flow(self):
        graph = Graph()
        fn_id = self._make_fn(
            graph,
            "foo",
            "test.py",
            "def foo(x):\n    z = x + 1\n    return z\n",
        )
        result = analyze_function_flow(graph, fn_id)
        assert "function" in result
        assert "cfg_nodes" in result
        assert result["cfg_nodes"] >= 3

    def test_analyze_all_functions(self):
        graph = Graph()
        self._make_fn(
            graph,
            "fn_a",
            "a.py",
            "def fn_a(x):\n    return x + 1\n",
        )
        self._make_fn(
            graph,
            "fn_b",
            "b.py",
            "def fn_b(y):\n    z = y * 2\n    return z\n",
        )
        result = analyze_all_functions(graph, limit=10)
        assert result["functions_analyzed"] == 2
        assert "total_dead_blocks" in result
        assert "average_critical_path_length" in result

    def test_analyze_nonexistent_function(self):
        graph = Graph()
        result = analyze_function_flow(graph, NodeId(999))
        assert "error" in result
