"""End-to-end benchmark for native and Python SQLite graph persistence."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from graphician.core import Edge, EdgeKind, Graph, Node, NodeKind
from graphician.persistence import store as store_module
from graphician.persistence.store import GraphStore


def build_graph(node_count: int, edge_count: int) -> Graph:
    graph = Graph()
    nodes = [
        graph.add_node(
            Node.new(NodeKind.FUNCTION, f"benchmark::{index}")
            .with_source(f"src/module_{index % 200}.py", index, index + 2)
            .with_property("index", index)
        )
        for index in range(node_count)
    ]
    for index in range(edge_count):
        graph.add_edge(
            nodes[index % node_count],
            nodes[(index * 17 + 1) % node_count],
            Edge.extracted(EdgeKind.CALLS),
        )
    return graph


def timed_save(graph: Graph, root: Path, iterations: int, *, native: bool) -> list[float]:
    operation = store_module.save_graph_sqlite
    timings = []
    try:
        if not native:
            store_module.save_graph_sqlite = None
        for iteration in range(iterations):
            with GraphStore(root / f"{'native' if native else 'python'}-{iteration}.db") as store:
                started = time.perf_counter()
                store.save_graph(graph)
                timings.append(time.perf_counter() - started)
    finally:
        store_module.save_graph_sqlite = operation
    return timings


def timed_load(store: GraphStore, iterations: int, *, native: bool) -> list[float]:
    operation = store_module.load_graph_sqlite
    timings = []
    try:
        if not native:
            store_module.load_graph_sqlite = None
        for _ in range(iterations):
            started = time.perf_counter()
            graph = store.load_graph()
            timings.append(time.perf_counter() - started)
            if graph.node_count() == 0:
                raise RuntimeError("loaded graph is unexpectedly empty")
    finally:
        store_module.load_graph_sqlite = operation
    return timings


def timed_incremental(
    base: Graph,
    updated: Graph,
    root: Path,
    iterations: int,
    *,
    native: bool,
) -> list[float]:
    operation = store_module.save_graph_incremental_sqlite
    timings = []
    try:
        if not native:
            store_module.save_graph_incremental_sqlite = None
        for iteration in range(iterations):
            with GraphStore(
                root / f"incremental-{'native' if native else 'python'}-{iteration}.db"
            ) as store:
                store.save_graph(base)
                started = time.perf_counter()
                store.save_graph_incremental(updated)
                timings.append(time.perf_counter() - started)
    finally:
        store_module.save_graph_incremental_sqlite = operation
    return timings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=5_000)
    parser.add_argument("--edges", type=int, default=15_000)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if args.nodes < 1 or args.edges < 0 or args.iterations < 1:
        parser.error("nodes and iterations must be positive; edges must be non-negative")

    graph = build_graph(args.nodes, args.edges)
    updated = graph.clone()
    for node_id, node in list(updated.nodes())[: max(1, args.nodes // 100)]:
        node.name = f"updated_{node_id.value}"
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        native_save = timed_save(graph, root, args.iterations, native=True)
        python_save = timed_save(graph, root, args.iterations, native=False)
        with GraphStore(root / "load.db") as store:
            store.save_graph(graph)
            native_load = timed_load(store, args.iterations, native=True)
            python_load = timed_load(store, args.iterations, native=False)
        native_incremental = timed_incremental(
            graph, updated, root, args.iterations, native=True
        )
        python_incremental = timed_incremental(
            graph, updated, root, args.iterations, native=False
        )

    native_save_median = statistics.median(native_save)
    python_save_median = statistics.median(python_save)
    native_load_median = statistics.median(native_load)
    python_load_median = statistics.median(python_load)
    native_incremental_median = statistics.median(native_incremental)
    python_incremental_median = statistics.median(python_incremental)
    print(
        json.dumps(
            {
                "nodes": args.nodes,
                "edges": args.edges,
                "iterations": args.iterations,
                "native_save_ms": round(native_save_median * 1_000, 3),
                "python_save_ms": round(python_save_median * 1_000, 3),
                "save_speedup": round(python_save_median / native_save_median, 3),
                "native_load_ms": round(native_load_median * 1_000, 3),
                "python_load_ms": round(python_load_median * 1_000, 3),
                "load_speedup": round(python_load_median / native_load_median, 3),
                "native_incremental_ms": round(native_incremental_median * 1_000, 3),
                "python_incremental_ms": round(python_incremental_median * 1_000, 3),
                "incremental_speedup": round(
                    python_incremental_median / native_incremental_median, 3
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
