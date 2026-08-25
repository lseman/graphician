"""End-to-end benchmark for native and Python call-placeholder resolution."""

from __future__ import annotations

import argparse
import json
import statistics
import time

from graphician.core.edge import Edge, EdgeKind
from graphician.core.graph import Graph
from graphician.core.node import Node, NodeKind
from graphician.extraction.call_resolution import (
    _resolve_call_placeholders_python,
    resolve_call_placeholders,
)


def build_graph(call_count: int) -> Graph:
    graph = Graph()
    for index in range(call_count):
        caller = graph.add_node(
            Node.new(NodeKind.FUNCTION, f"app::caller_{index}").with_source(
                f"src/group_{index % 100}/caller.py", 0, 1
            )
        )
        placeholder = graph.add_node(
            Node.new(NodeKind.FUNCTION, f"call::target_{index}")
        )
        graph.add_edge(caller, placeholder, Edge.ambiguous(EdgeKind.CALLS))
        graph.add_node(
            Node.new(NodeKind.FUNCTION, f"local::target_{index}").with_source(
                f"src/group_{index % 100}/target.py", 0, 1
            )
        )
        graph.add_node(
            Node.new(NodeKind.FUNCTION, f"remote::target_{index}").with_source(
                f"lib/group_{index % 100}/target.py", 0, 1
            )
        )
    return graph


def measure(base: Graph, iterations: int, resolver) -> list[float]:
    timings = []
    for _ in range(iterations):
        graph = base.clone()
        started = time.perf_counter()
        resolved = resolver(graph)
        timings.append(time.perf_counter() - started)
        if resolved * 3 != graph.node_count():
            raise RuntimeError("resolver produced an unexpected graph")
    return timings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=3_000)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if args.calls < 1 or args.iterations < 1:
        parser.error("calls and iterations must be positive")

    base = build_graph(args.calls)
    native = measure(base, args.iterations, resolve_call_placeholders)
    python = measure(base, args.iterations, _resolve_call_placeholders_python)
    native_median = statistics.median(native)
    python_median = statistics.median(python)
    print(
        json.dumps(
            {
                "calls": args.calls,
                "iterations": args.iterations,
                "native_end_to_end_ms": round(native_median * 1_000, 3),
                "python_end_to_end_ms": round(python_median * 1_000, 3),
                "speedup": round(python_median / native_median, 3),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
