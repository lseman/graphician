"""Token reduction benchmark.

Measures graph query efficiency vs naive file reading. Runs a set of
sample questions through the search pipeline, measures the token cost
of the graph-structured results (plus neighbor edges), and compares
against the naive cost of reading every source file.

Produces a per-question breakdown and an average reduction ratio.

Mirrors the Rust ``token_benchmark.rs`` module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Sample questions for benchmarking
SAMPLE_QUESTIONS: list[str] = [
    "how does authentication work",
    "what is the main entry point",
    "how are database connections managed",
    "what error handling patterns are used",
    "how do tests verify core functionality",
]


def run_token_benchmark(
    db_path: str,
    graph,
    repo_root: str | Path,
    questions: list[str] | None = None,
) -> dict[str, Any]:
    """Run a token reduction benchmark.

    Scans for source files to compute the naive corpus token count. For
    each question the search pipeline is run, the token cost of returned
    nodes + neighbor edges is estimated, and the reduction ratio is
    computed.

    Args:
        db_path: Path to the database (for embedding lookup).
        graph: The code graph.
        repo_root: Repository root for naive corpus scanning.
        questions: Questions to benchmark. Defaults to SAMPLE_QUESTIONS.

    Returns:
        Benchmark result dict.
    """
    if questions is None:
        questions = list(SAMPLE_QUESTIONS)

    repo_path = Path(repo_root)
    naive_corpus = compute_naive_tokens(repo_path)

    # Check if graph has embeddings
    has_embeddings = _check_embeddings(db_path)
    using_defaults = questions == list(SAMPLE_QUESTIONS)
    warning = None
    if not has_embeddings and using_defaults:
        warning = (
            "No embeddings found in this graph. The default sample questions "
            "are natural language and will not match via FTS5/LIKE alone — "
            "every reduction ratio is likely to be 0. Run `ariadne embed` "
            "first, or pass keyword-matching `--question` flags."
        )

    results: list[dict[str, Any]] = []
    for question in questions:
        hits = _search_hits(graph, question)
        graph_tokens = estimate_graph_tokens(graph, hits)
        hit_count = len(hits)
        ratio = naive_corpus / graph_tokens if graph_tokens > 0 else 0.0

        results.append({
            "question": question,
            "naive_tokens": naive_corpus,
            "graph_tokens": graph_tokens,
            "hit_count": hit_count,
            "reduction_ratio": round(ratio * 10.0) / 10.0,
        })

    avg_ratio = 0.0
    if results:
        total = sum(r["reduction_ratio"] for r in results)
        avg_ratio = round(total / len(results) * 10.0) / 10.0

    return {
        "naive_corpus_tokens": naive_corpus,
        "per_question": results,
        "average_reduction_ratio": avg_ratio,
        "summary": f"Graph queries use ~{avg_ratio:.0f}x fewer tokens than reading all source files",
        "warning": warning,
    }


def benchmark_json(result: dict[str, Any]) -> dict[str, Any]:
    """Format benchmark results as JSON.

    Args:
        result: BenchmarkResult dict from run_token_benchmark.

    Returns:
        JSON-serializable benchmark dict.
    """
    obj: dict[str, Any] = {
        "operation": "token_benchmark",
        "naive_corpus_tokens": result["naive_corpus_tokens"],
        "average_reduction_ratio": result["average_reduction_ratio"],
        "summary": result["summary"],
        "per_question": [
            {
                "question": q["question"],
                "naive_tokens": q["naive_tokens"],
                "graph_tokens": q["graph_tokens"],
                "hit_count": q["hit_count"],
                "reduction_ratio": q["reduction_ratio"],
            }
            for q in result["per_question"]
        ],
    }
    if result.get("warning"):
        obj["warning"] = result["warning"]
    return obj


def benchmark_cli_panel(result: dict[str, Any]) -> str:
    """Format benchmark results as a box-drawing CLI panel.

    Args:
        result: BenchmarkResult dict from run_token_benchmark.

    Returns:
        Formatted panel string.
    """
    lines: list[str] = []
    width = 52
    lines.append("┌" + "─" * width + "Token Benchmark" + "─" * width + "┐")
    lines.append(f"│ Naive corpus tokens: {result['naive_corpus_tokens']:>10} tokens │")
    lines.append(f"│ Avg reduction ratio:  {result['average_reduction_ratio']:>10}x       │")
    lines.append("├" + "─" * width + "┤")

    for q in result.get("per_question", []):
        question = q["question"]
        if len(question) > 42:
            question = question[:42]
        lines.append(f'│ "{question:<42}" │')
        lines.append(
            f'│   → {q["hit_count"]} hits, {q["graph_tokens"]} graph tokens ({q["reduction_ratio"]:.0f}x reduction)  │'
        )

    lines.append("└" + "─" * width + "┘")
    return "\n".join(lines)


# ── Internal helpers ───────────────────────────────────────────────


def _check_embeddings(db_path: str) -> bool:
    """Check if the graph has embeddings.

    Args:
        db_path: Path to the database.

    Returns:
        True if embeddings exist.
    """
    try:
        from ....persistence.store import GraphStore

        store = GraphStore(db_path)
        try:
            stats = store.get_embedding_stats()
            return stats is not None and len(stats) > 0
        finally:
            store.close()
    except Exception:
        return False


def _search_hits(graph, question: str) -> list[dict[str, Any]]:
    """Search for hits matching a question.

    Args:
        graph: The code graph.
        question: Search query.

    Returns:
        List of hit dicts with 'id' and 'score'.
    """
    try:
        from ....analysis.search import hybrid_search

        result = hybrid_search(graph, question, limit=50)
        return result.get("results", [])
    except Exception:
        # Fallback: simple fuzzy match
        return _fuzzy_search(graph, question)


def _fuzzy_search(graph, question: str) -> list[dict[str, Any]]:
    """Fallback fuzzy search for hits.

    Args:
        graph: The code graph.
        question: Search query.

    Returns:
        List of hit dicts with 'id' and 'score'.
    """
    question_lower = question.lower()
    hits: list[dict[str, Any]] = []

    for nid, node in graph.nodes():
        name = (node.name or "").lower()
        qname = (node.qualified_name or "").lower()

        score = 0.0
        if name == question_lower:
            score = 1.0
        elif name.startswith(question_lower) or qname.startswith(question_lower):
            score = 0.9
        elif question_lower in name or question_lower in qname:
            score = 0.7
        elif any(word in name for word in question_lower.split()):
            score = 0.4

        if score > 0:
            hits.append({"id": nid, "score": score})

    hits.sort(key=lambda h: -h["score"])
    return hits


def estimate_graph_tokens(graph, hits: list[dict[str, Any]]) -> int:
    """Estimate token count for graph search results.

    Counts tokens for node metadata + neighbor edges (up to 5 per
    direction).

    Args:
        graph: The code graph.
        hits: List of hit dicts from search.

    Returns:
        Total estimated token count.
    """
    total = 0
    for hit in hits:
        nid = hit.get("id")
        if nid is None:
            continue

        node = graph.node(nid) if hasattr(graph, "node") else None
        if node is None:
            for _, n in graph.nodes():
                if _ids_match(n, nid):
                    node = n
                    break
        if node is None:
            continue

        # Token cost of the node metadata
        total += approx_tokens(node.qualified_name)
        total += approx_tokens(str(node.kind))
        if node.source_uri:
            total += approx_tokens(node.source_uri)

        # Add neighbor edges
        neighbor_count = 0
        if hasattr(graph, "out_neighbors"):
            for neighbor_id, edge in graph.out_neighbors(nid):
                if neighbor_count >= 5:
                    break
                if neighbor_id is not None:
                    neighbor = graph.node(neighbor_id)
                    if neighbor:
                        total += approx_tokens(neighbor.qualified_name)
                    total += approx_tokens(str(edge.kind) if hasattr(edge, "kind") else "unknown")
                neighbor_count += 1

        neighbor_count = 0
        if hasattr(graph, "in_neighbors"):
            for neighbor_id, edge in graph.in_neighbors(nid):
                if neighbor_count >= 5:
                    break
                if neighbor_id is not None:
                    neighbor = graph.node(neighbor_id)
                    if neighbor:
                        total += approx_tokens(neighbor.qualified_name)
                    total += approx_tokens(str(edge.kind) if hasattr(edge, "kind") else "unknown")
                neighbor_count += 1

    return total


def compute_naive_tokens(repo_root: Path) -> int:
    """Count approximate tokens across all parseable source files.

    Args:
        repo_root: Repository root directory.

    Returns:
        Total token count for all source files.
    """
    extensions = [
        "py", "js", "ts", "tsx", "jsx", "go", "rs", "java", "c", "cpp",
        "cc", "cxx", "rb", "php", "swift", "kt", "scala", "h", "hpp", "hxx",
    ]
    skip_dirs = {"target", "node_modules", ".git", "vendor", "build", "dist", ".venv", "__pycache__", ".tox", ".mypy_cache"}
    total = 0

    for entry in _walk_directory(repo_root, extensions, skip_dirs):
        try:
            content = entry.read_text(encoding="utf-8", errors="replace")
            total += approx_tokens(content)
        except OSError:
            pass

    return total


def _walk_directory(dir_path: Path, extensions: list[str], skip_dirs: set[str]) -> list[Path]:
    """Recursively walk directory, yielding paths with matching extensions.

    Args:
        dir_path: Directory to walk.
        extensions: List of extensions (without dot) to match.
        skip_dirs: Directory names to skip.

    Returns:
        List of matching file paths.
    """
    out: list[Path] = []
    if not dir_path.is_dir():
        return out

    try:
        entries = sorted(dir_path.iterdir())
    except OSError:
        return out

    for entry in entries:
        if entry.is_dir():
            if entry.name not in skip_dirs:
                out.extend(_walk_directory(entry, extensions, skip_dirs))
        elif entry.is_file():
            ext = entry.suffix.lstrip(".")
            if ext.lower() in extensions:
                out.append(entry)

    return out


def approx_tokens(s: str) -> int:
    """Approximate token count (same heuristic as token_savings)."""
    return max(len(s) // 4, 1)


def _ids_match(node: Any, node_id: Any) -> bool:
    """Check if a node matches a node ID."""
    if not hasattr(node, "id"):
        return False
    return node.id == node_id
