"""Benchmark cold construction versus reuse of Graphician's native snapshot.

Run from the repository root after building the native extension:

    PYTHONPATH=src python benchmarks/native_snapshot_lifetime.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

from graphician.analysis.native import native_graph
from graphician.core.edge import Edge, EdgeKind
from graphician.core.graph import Graph
from graphician.core.node import Node, NodeKind


def build_graph(node_count: int, edge_count: int) -> Graph:
    graph = Graph()
    nodes = [
        graph.add_node(Node.new(NodeKind.FUNCTION, f"benchmark::node_{index}"))
        for index in range(node_count)
    ]
    for index in range(edge_count):
        source = nodes[index % node_count]
        target = nodes[(index * 17 + 1) % node_count]
        graph.add_edge(source, target, Edge.extracted(EdgeKind.CALLS))
    return graph


def measure(graph: Graph, iterations: int, *, cold: bool) -> list[float]:
    timings = []
    for _ in range(iterations):
        if cold:
            graph._native_snapshot = None
            graph._native_snapshot_key = None
        started = time.perf_counter()
        snapshot = native_graph(graph)
        timings.append(time.perf_counter() - started)
        if snapshot is None:
            raise RuntimeError("graphician-native is not installed")
    return timings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=10_000)
    parser.add_argument("--edges", type=int, default=30_000)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.nodes < 1 or args.edges < 0 or args.iterations < 1:
        parser.error("nodes and iterations must be positive; edges must be non-negative")

    graph = build_graph(args.nodes, args.edges)
    cold = measure(graph, args.iterations, cold=True)
    warm = measure(graph, args.iterations, cold=False)
    cold_median = statistics.median(cold)
    warm_median = statistics.median(warm)
    print(
        json.dumps(
            {
                "nodes": args.nodes,
                "edges": args.edges,
                "iterations": args.iterations,
                "cold_median_ms": round(cold_median * 1_000, 3),
                "warm_median_ms": round(warm_median * 1_000, 3),
                "reuse_speedup": round(cold_median / warm_median, 3),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
