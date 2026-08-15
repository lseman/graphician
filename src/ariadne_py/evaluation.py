"""Deterministic benchmark runner compatible with Rust ``ariadne-eval``."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import time
import tomllib
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from .analysis.impact.engine import find_impact
from .analysis.impact.types import ImpactQuery
from .analysis.paths import callees_of, callers_of, max_depth_from
from .analysis.structure import call_resolution_stats
from .extraction.flows import compute_flows
from .persistence.store import GraphStore

SAMPLE_QUESTIONS = (
    "how does authentication work",
    "what is the main entry point",
    "how are database connections managed",
    "what error handling patterns are used",
    "how do tests verify core functionality",
)
BENCHMARKS = (
    "token_efficiency",
    "flow_completeness",
    "impact_accuracy",
    "search_quality",
    "build_performance",
    "multi_hop_retrieval",
    "agent_baseline",
)


def load_config(configs_dir: Path, name: str) -> dict[str, Any]:
    path = configs_dir / f"{name}.toml"
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    for key in ("name", "url", "commit"):
        if not config.get(key):
            raise ValueError(f"{path}: config must have a non-empty {key}")
    commits = config.get("test_commits", [])
    if commits and config["commit"] != commits[-1].get("sha"):
        raise ValueError(f"{path}: commit must equal the latest test_commit sha")
    return config


def load_all_configs(configs_dir: Path) -> list[dict[str, Any]]:
    if not configs_dir.exists():
        return []
    return [load_config(configs_dir, path.stem) for path in sorted(configs_dir.glob("*.toml"))]


def run_eval(
    repos: list[str],
    benchmarks: list[str],
    output_dir: Path | None = None,
    embed: bool = False,
) -> list[dict[str, Any]]:
    """Run the same named benchmark registry and emit per-benchmark CSV/JSON."""
    output = output_dir or Path("evaluate/results")
    output.mkdir(parents=True, exist_ok=True)
    configs_dir = Path("evaluate/configs")
    configs = (
        [load_config(configs_dir, repo) for repo in repos]
        if repos
        else load_all_configs(configs_dir)
    )
    selected = benchmarks or list(BENCHMARKS)
    unknown = sorted(set(selected) - set(BENCHMARKS))
    if unknown:
        raise ValueError(f"unknown benchmarks: {', '.join(unknown)}")

    results: list[dict[str, Any]] = []
    for config in configs:
        repo_root = _repo_root(config)
        db_path = repo_root / "ariadne.db"
        if not db_path.exists():
            continue
        with GraphStore(db_path) as store:
            if embed and store.embedding_stats()[0] == 0:
                store.rebuild_embeddings()
            for benchmark in selected:
                rows = _RUNNERS[benchmark](repo_root, store, config)
                _write_results(output, config["name"], benchmark, rows)
                results.extend(
                    {"repo": config["name"], "benchmark": benchmark, "data": row} for row in rows
                )
    return results


def _repo_root(config: dict[str, Any]) -> Path:
    current = Path.cwd()
    return current if current.name == config["name"] else Path(config["name"])


def _search(store: GraphStore, query: str, limit: int) -> list[str]:
    return [qualified_name for qualified_name, _score in store.fts_search(query, limit)]


def _token_efficiency(
    repo: Path, store: GraphStore, config: dict[str, Any]
) -> list[dict[str, Any]]:
    graph = store.load_graph()
    naive = sum(
        max(1, len(path.read_text(encoding="utf-8", errors="replace")) // 4)
        for path in _source_files(repo)
    )
    rows: list[dict[str, Any]] = []
    for question in SAMPLE_QUESTIONS:
        hits = _search(store, question, 50)
        graph_tokens = 0
        for qname in hits:
            node_id = graph.find_by_qname(qname)
            node = graph.node(node_id) if node_id is not None else None
            if node is not None:
                graph_tokens += max(1, len(node.qualified_name) // 4)
                graph_tokens += max(1, len(node.source_uri or "") // 4)
        ratio = round(naive / graph_tokens, 1) if graph_tokens else 0.0
        rows.append(
            {
                "question": question,
                "naive_tokens": naive,
                "graph_tokens": graph_tokens,
                "hit_count": len(hits),
                "reduction_ratio": ratio,
            }
        )
    average = round(sum(row["reduction_ratio"] for row in rows) / max(1, len(rows)), 1)
    rows.append(
        {
            "_aggregate": True,
            "naive_corpus_tokens": naive,
            "average_reduction_ratio": average,
            "summary": (
                f"Graph queries use ~{average:.0f}x fewer tokens "
                "than reading all source files"
            ),
        }
    )
    return rows


def _flow_completeness(
    _repo: Path, store: GraphStore, config: dict[str, Any]
) -> list[dict[str, Any]]:
    graph = store.load_graph()
    count = compute_flows(graph)
    known = config.get("entry_points", [])
    detected = [node.qualified_name for _, node in graph.nodes() if node.kind.value == "flow"]
    found = sum(any(item in hit or hit in item for hit in detected) for item in known)
    depths = []
    for qname in detected:
        node_id = graph.find_by_qname(qname)
        if node_id is not None:
            depths.append(max_depth_from(graph, node_id))
    return [
        {
            "known_entry_points": len(known),
            "detected_entry_points": found,
            "detected_flows": count,
            "recall": round(found / len(known), 3) if known else 1.0,
            "avg_flow_depth": sum(depths) // len(depths) if depths else 0,
            "max_flow_depth": max(depths, default=0),
        }
    ]


def _search_quality(_repo: Path, store: GraphStore, config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for query in config.get("search_queries", []):
        hits = _search(store, query["query"], 20)
        expected = query["expected"].lower()
        expected_name = expected.rsplit("::", 1)[-1]
        rank = next(
            (
                index
                for index, hit in enumerate(hits, 1)
                if expected in hit.lower()
                or hit.lower() in expected
                or hit.lower().rsplit("::", 1)[-1] == expected_name
            ),
            0,
        )
        rows.append(
            {
                "query": query["query"],
                "expected": query["expected"],
                "rank": rank,
                "reciprocal_rank": round(1 / rank, 3) if rank else 0.0,
            }
        )
    mrr = round(sum(row["reciprocal_rank"] for row in rows) / max(1, len(rows)), 3)
    rows.append(
        {
            "_aggregate": True,
            "mrr": mrr,
            "query_count": max(1, len(rows)),
            "summary": f"Mean Reciprocal Rank: {mrr:.3f}",
        }
    )
    return rows


def _build_performance(
    _repo: Path, store: GraphStore, _config: dict[str, Any]
) -> list[dict[str, Any]]:
    loads: list[int] = []
    searches: list[int] = []
    for _ in range(11):
        start = time.perf_counter_ns()
        store.load_graph()
        loads.append((time.perf_counter_ns() - start) // 1_000)
    graph = store.load_graph()
    for _ in range(11):
        start = time.perf_counter_ns()
        _search(store, "graph query", 25)
        searches.append((time.perf_counter_ns() - start) // 1_000)
    status = store.status()
    calls = call_resolution_stats(graph)
    return [
        {
            "_aggregate": True,
            "node_count": status["node_count"],
            "edge_count": status["edge_count"],
            "embedding_count": store.embedding_stats()[0],
            "embedding_model": store.embedding_stats()[1],
            "fts_indexed_nodes": store.fts_stats(),
            "store_load_p50_ms": _percentile(loads, 50),
            "store_load_p95_ms": _percentile(loads, 95),
            "search_p50_ms": _percentile(searches, 50),
            "search_p95_ms": _percentile(searches, 95),
            "call_resolution_rate": round(calls["rate"], 3),
            "sample_count": 11,
            "note": (
                "Cold build time and peak RSS are emitted by the build command, "
                "not inferred here"
            ),
        }
    ]


def _multi_hop(_repo: Path, store: GraphStore, config: dict[str, Any]) -> list[dict[str, Any]]:
    graph = store.load_graph()
    rows = []
    for task in config.get("multi_hop_tasks", []):
        hits = _search(store, task["nl_query"], 10)
        suffix = task["anchor_qualified_suffix"].lower()
        rank = next(
            (
                i
                for i, qname in enumerate(hits, 1)
                if qname.lower().endswith(suffix) or qname.lower().rsplit("::", 1)[-1] == suffix
            ),
            0,
        )
        neighbors = []
        if rank:
            node_id = graph.find_by_qname(hits[rank - 1])
            if node_id is not None:
                fn = callees_of if task.get("traversal_pattern") == "callees_of" else callers_of
                neighbors = [graph.node(item).qualified_name.lower() for item in fn(graph, node_id)]
        expected = [name.lower() for name in task.get("expected_neighbor_names", [])]
        recall = sum(any(name in neighbor for neighbor in neighbors) for name in expected) / max(
            1, len(expected)
        )
        rows.append(
            {
                "task_id": task["id"],
                "nl_query": task["nl_query"],
                "anchor_qualified_suffix": task["anchor_qualified_suffix"],
                "traversal_pattern": task.get("traversal_pattern", "callers_of"),
                "anchor_found": bool(rank),
                "anchor_rank": rank,
                "neighbor_count": len(neighbors),
                "neighbor_recall": round(recall, 3),
                "score": round(recall if rank else 0.0, 3),
            }
        )
    score = round(sum(row["score"] for row in rows) / max(1, len(rows)), 3)
    rows.append(
        {
            "_aggregate": True,
            "average_score": score,
            "task_count": max(1, len(rows)),
            "summary": f"Multi-hop retrieval score: {score:.3f}",
        }
    )
    return rows


def _impact_accuracy(repo: Path, store: GraphStore, config: dict[str, Any]) -> list[dict[str, Any]]:
    graph = store.load_graph()
    rows: list[dict[str, Any]] = []
    totals = [0, 0, 0, 0]
    for commit in config.get("test_commits", []):
        process = subprocess.run(
            [
                "git",
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                f"{commit['sha']}^",
                commit["sha"],
            ],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode:
            rows.append(
                {
                    "commit_sha": commit["sha"],
                    "status": "error",
                    "error": process.stderr.strip(),
                    "ground_truth_mode": "co-change",
                }
            )
            continue
        changed = {_normalize_path(path) for path in process.stdout.splitlines()}
        represented = {
            path
            for path in changed
            if any(
                node.source_uri and _paths_equal(node.source_uri, path) for _, node in graph.nodes()
            )
        }
        for seed in represented:
            relevant = represented - {seed}
            if not relevant:
                continue
            predicted = _affected_files(graph, seed)
            tp, fp, fn = (
                len(predicted & relevant),
                len(predicted - relevant),
                len(relevant - predicted),
            )
            totals[0] += 1
            totals[1] += tp
            totals[2] += fp
            totals[3] += fn
            rows.append(
                {
                    "commit_sha": commit["sha"],
                    "seed_file": seed,
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn,
                    "precision": _ratio(tp, tp + fp),
                    "recall": _ratio(tp, tp + fn),
                    "f1": _f1(tp, fp, fn),
                    "ground_truth_mode": "co-change",
                    "status": "ok",
                }
            )
    rows.append(
        {
            "_aggregate": True,
            "evaluated_seeds": totals[0],
            "true_positives": totals[1],
            "false_positives": totals[2],
            "false_negatives": totals[3],
            "micro_precision": _ratio(totals[1], totals[1] + totals[2]),
            "micro_recall": _ratio(totals[1], totals[1] + totals[3]),
            "micro_f1": _f1(totals[1], totals[2], totals[3]),
            "ground_truth_mode": "co-change",
        }
    )
    return rows


def _agent_baseline(repo: Path, store: GraphStore, config: dict[str, Any]) -> list[dict[str, Any]]:
    graph = store.load_graph()
    rows = []
    for query in config.get("search_queries", []):
        expected = query["expected"].lower()
        relevant = {
            _normalize_path(node.source_uri)
            for _, node in graph.nodes()
            if node.source_uri
            and (expected in node.name.lower() or expected in node.qualified_name.lower())
        }
        if not relevant:
            continue
        qnames = _search(store, query["query"], 20)
        graph_files = {
            _normalize_path(graph.node(graph.find_by_qname(qname)).source_uri)
            for qname in qnames
            if graph.find_by_qname(qname) is not None
            and graph.node(graph.find_by_qname(qname)).source_uri
        }
        lexical = _lexical_files(repo, query["query"], 10)
        rows.append(
            {
                "query": query["query"],
                "expected": query["expected"],
                "relevant_file_count": len(relevant),
                "graph_file_count": len(graph_files),
                "lexical_file_count": len(lexical),
                "graph_file_recall": _file_recall(graph_files, relevant),
                "lexical_file_recall": _file_recall(set(lexical), relevant),
            }
        )
    rows.append(
        {
            "_aggregate": True,
            "query_count": len(rows),
            "mean_graph_file_recall": round(
                sum(r["graph_file_recall"] for r in rows) / max(1, len(rows)), 3
            ),
            "mean_lexical_file_recall": round(
                sum(r["lexical_file_recall"] for r in rows) / max(1, len(rows)), 3
            ),
        }
    )
    return rows


def _affected_files(graph: Any, seed: str) -> set[str]:
    seed_ids = [
        node_id
        for node_id, node in graph.nodes()
        if node.source_uri and _paths_equal(node.source_uri, seed)
    ]
    scores: dict[str, float] = {}
    for seed_id in seed_ids:
        query = ImpactQuery(seed_id=seed_id, max_hops=3, limit=graph.node_count())
        for hit in find_impact(graph, query):
            uri = hit.node.source_uri
            if uri and not _paths_equal(uri, seed):
                path = _normalize_path(uri)
                scores[path] = max(scores.get(path, 0.0), hit.score)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return {path for path, _score in ranked[:10]}


def _source_files(root: Path) -> list[Path]:
    extensions = {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".go",
        ".rs",
        ".java",
        ".c",
        ".cpp",
        ".cc",
        ".cxx",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".scala",
        ".h",
        ".hpp",
        ".hxx",
    }
    excluded = {
        "target",
        "node_modules",
        ".git",
        "vendor",
        "build",
        "dist",
        ".venv",
        "__pycache__",
        ".tox",
        ".mypy_cache",
    }
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix in extensions and not excluded.intersection(path.parts)
    ]


def _lexical_files(root: Path, query: str, limit: int) -> list[str]:
    terms = {term.lower() for term in query.replace("_", " ").split() if len(term) > 2}
    hits = []
    for path in _source_files(root):
        source = path.read_text(encoding="utf-8", errors="replace").lower()
        score = sum(source.count(term) for term in terms)
        if score:
            hits.append((score, _normalize_path(str(path))))
    hits.sort(key=lambda item: (-item[0], item[1]))
    return [path for _score, path in hits[:limit]]


def _write_results(output: Path, repo: str, benchmark: str, rows: list[dict[str, Any]]) -> None:
    stem = f"{repo}_{benchmark}_{date.today().isoformat()}"
    (output / f"{stem}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with (output / f"{stem}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _percentile(samples: list[int], percentile: int) -> float:
    ordered = sorted(samples)
    index = math.ceil((len(ordered) - 1) * percentile / 100)
    return round(ordered[index] / 1000, 3)


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").removeprefix("./")


def _paths_equal(left: str, right: str) -> bool:
    left, right = _normalize_path(left), _normalize_path(right)
    return left == right or left.endswith(f"/{right}") or right.endswith(f"/{left}")


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 3) if denominator else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    return round(2 * precision * recall / (precision + recall), 3) if precision + recall else 0.0


def _file_recall(retrieved: set[str], relevant: set[str]) -> float:
    found = sum(any(_paths_equal(got, wanted) for got in retrieved) for wanted in relevant)
    return round(found / max(1, len(relevant)), 3)


_RUNNERS: dict[str, Callable[[Path, GraphStore, dict[str, Any]], list[dict[str, Any]]]] = {
    "token_efficiency": _token_efficiency,
    "flow_completeness": _flow_completeness,
    "impact_accuracy": _impact_accuracy,
    "search_quality": _search_quality,
    "build_performance": _build_performance,
    "multi_hop_retrieval": _multi_hop,
    "agent_baseline": _agent_baseline,
}
