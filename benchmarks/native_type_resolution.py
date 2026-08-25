"""End-to-end benchmark for native and Python type-placeholder resolution."""

from __future__ import annotations

import argparse
import json
import statistics
import time

from graphician.core.edge import Edge, EdgeKind
from graphician.core.graph import Graph
from graphician.core.node import Node, NodeKind
from graphician.extraction.type_resolution import (
    _resolve_type_placeholders_python,
    resolve_type_placeholders,
)


def build_graph(type_count: int) -> Graph:
    graph = Graph()
    for index in range(type_count):
        owner = graph.add_node(Node.new(NodeKind.CLASS, f"owner::{index}"))
        placeholder = graph.add_node(Node.new(NodeKind.CLASS, f"type::Type{index}"))
        graph.add_edge(owner, placeholder, Edge.extracted(EdgeKind.INHERITS))
        graph.add_node(Node.new(NodeKind.CLASS, f"real::Type{index}"))
    return graph


def measure(base: Graph, iterations: int, resolver) -> list[float]:
    timings = []
    for _ in range(iterations):
        graph = base.clone()
        started = time.perf_counter()
        resolved = resolver(graph)
        timings.append(time.perf_counter() - started)
        if resolved * 2 != graph.node_count():
            raise RuntimeError("resolver produced an unexpected graph")
    return timings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--types", type=int, default=1_500)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if args.types < 1 or args.iterations < 1:
        parser.error("types and iterations must be positive")

    base = build_graph(args.types)
    native = measure(base, args.iterations, resolve_type_placeholders)
    python = measure(base, args.iterations, _resolve_type_placeholders_python)
    native_median = statistics.median(native)
    python_median = statistics.median(python)
    print(
        json.dumps(
            {
                "types": args.types,
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
