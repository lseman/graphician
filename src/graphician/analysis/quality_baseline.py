"""Behavioral quality metrics for impact predictions."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.graph import Graph
from .impact import ImpactQuery, find_impact

if TYPE_CHECKING:
    from ..persistence.store import GraphStore


@dataclass(frozen=True)
class ImpactAccuracy:
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total else 0.0

    @property
    def recall(self) -> float:
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    def __add__(self, other: ImpactAccuracy) -> ImpactAccuracy:
        return ImpactAccuracy(
            self.true_positives + other.true_positives,
            self.false_positives + other.false_positives,
            self.false_negatives + other.false_negatives,
        )


@dataclass(frozen=True)
class OperationalBaseline:
    """Repeatable measurements for loading and querying a built graph."""

    node_count: int
    edge_count: int
    embedding_count: int
    embedding_model: str | None
    fts_indexed_nodes: int
    store_load_p50_ms: float
    store_load_p95_ms: float
    search_p50_ms: float
    search_p95_ms: float
    call_resolution_rate: float
    sample_count: int


def benchmark_store(
    store: GraphStore,
    *,
    samples: int = 11,
    query: str = "graph query",
    timer: Callable[[], float] = time.perf_counter,
) -> OperationalBaseline:
    """Measure already-built store operations without hiding build setup."""
    if samples < 1:
        raise ValueError("samples must be at least 1")

    load_ms: list[float] = []
    for _ in range(samples):
        started = timer()
        graph = store.load_graph()
        load_ms.append((timer() - started) * 1000.0)

    search_ms: list[float] = []
    for _ in range(samples):
        started = timer()
        store.fts_search(query, 25)
        search_ms.append((timer() - started) * 1000.0)

    from .structure import call_resolution_stats

    calls = call_resolution_stats(graph)
    embedding_count, embedding_model = store.get_embedding_stats() or (0, None)
    return OperationalBaseline(
        node_count=graph.node_count(),
        edge_count=graph.edge_count(),
        embedding_count=embedding_count,
        embedding_model=embedding_model,
        fts_indexed_nodes=store.fts_stats(),
        store_load_p50_ms=_percentile(load_ms, 50),
        store_load_p95_ms=_percentile(load_ms, 95),
        search_p50_ms=_percentile(search_ms, 50),
        search_p95_ms=_percentile(search_ms, 95),
        call_resolution_rate=round(float(calls["rate"]), 3),
        sample_count=samples,
    )


def _percentile(samples: list[float], percentile: int) -> float:
    """Nearest-rank percentile matching the Rust evaluation harness."""
    ordered = sorted(samples)
    index = ((len(ordered) - 1) * percentile + 99) // 100
    return round(ordered[index], 3)


def impact_accuracy(predicted: Iterable[str], relevant: Iterable[str]) -> ImpactAccuracy:
    """Score predicted affected files against co-change ground truth."""
    predicted_set = set(predicted)
    relevant_set = set(relevant)
    return ImpactAccuracy(
        true_positives=len(predicted_set & relevant_set),
        false_positives=len(predicted_set - relevant_set),
        false_negatives=len(relevant_set - predicted_set),
    )


def changed_files(repo_root: Path, commit: str) -> set[str]:
    """Return files changed by *commit* relative to its first parent."""
    process = subprocess.run(
        [
            "git",
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            f"{commit}^",
            commit,
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return {_normalize_path(line) for line in process.stdout.splitlines() if line.strip()}


def predict_affected_files(
    graph: Graph,
    seed_file: str,
    *,
    max_hops: int = 3,
    limit: int = 10,
) -> set[str]:
    """Predict affected files by combining impact walks for symbols in a file."""
    seed_ids = [
        node_id
        for node_id, node in graph.nodes()
        if node.source_uri and _path_matches(node.source_uri, seed_file)
    ]
    scores: dict[str, float] = {}
    for seed_id in seed_ids:
        for hit in find_impact(
            graph,
            ImpactQuery(seed_id=seed_id, max_hops=max_hops, limit=graph.node_count()),
        ):
            uri = hit.node.source_uri
            if uri and not _path_matches(uri, seed_file):
                path = _normalize_path(uri)
                scores[path] = max(scores.get(path, 0.0), hit.score)

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return {path for path, _score in ranked[:limit]}


def evaluate_cochange(graph: Graph, changed: Iterable[str]) -> ImpactAccuracy:
    """Micro-average predictions using every indexed co-changed file as a seed."""
    indexed = {
        _normalize_path(path)
        for path in changed
        if any(
            node.source_uri and _path_matches(node.source_uri, path)
            for _node_id, node in graph.nodes()
        )
    }
    total = ImpactAccuracy(0, 0, 0)
    for seed in indexed:
        relevant = indexed - {seed}
        if relevant:
            total += impact_accuracy(predict_affected_files(graph, seed), relevant)
    return total


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").removeprefix("./")


def _path_matches(indexed: str, git_path: str) -> bool:
    indexed = _normalize_path(indexed)
    git_path = _normalize_path(git_path)
    return indexed == git_path or indexed.endswith(f"/{git_path}")
