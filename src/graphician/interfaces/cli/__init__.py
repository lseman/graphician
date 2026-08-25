"""CLI interface for Graphician.

Subcommands mirror the Rust version:
- build, update, status, search, impact, paths
- callers, callees, flows, architecture
- detect-changes, risk, test-coverage
- communities, bridge-nodes, cycles, god-nodes
- tool (JSON tool interface)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from ...analysis.changes import compute_risk, compute_test_coverage, detect_changes
from ...analysis.communities import (
    detect_communities,
    find_bridge_nodes,
    find_hub_nodes,
)
from ...analysis.coverage import graph_coverage
from ...analysis.diff import graph_diff
from ...analysis.impact import compute_impact
from ...analysis.search import hybrid_search
from ...analysis.structure import (
    find_articulation_points,
    find_counterfactual,
    find_cycles,
    find_dead_code,
    find_god_nodes,
    find_large_functions,
    find_motifs,
)
from ...core.graph import Graph
from ...extraction.compiler import apply_compiler_evidence, load_compiler_evidence
from ...extraction.jedi import enrich_jedi_calls
from ...extraction.languages import LanguageRegistry
from ...extraction.pipeline import ExtractionPipeline
from ...extraction.rust_analyzer import RustAnalyzerOptions, enrich_with_rust_analyzer
from ...extraction.spring_di import resolve_spring_injections
from ...persistence.embeddings import build_external_embeddings, build_local_embeddings
from ...persistence.store import GraphStore
from .git import git_commit_hash, graph_freshness
from .response import _minimal_context
from .response.paths import handle_paths

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the graphician CLI."""
    sys.argv[1:] = _normalize_grouped_argv(sys.argv[1:])
    parser = argparse.ArgumentParser(
        prog="graphician",
        description="A local-first code graph for navigating, reviewing, and reasoning about a codebase.",
    )
    parser.add_argument("-d", "--db", default="graphician.db", help="Path to SQLite database")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # build
    build_p = subparsers.add_parser("build", help="Build graph from a project")
    build_p.add_argument("path", nargs="?", default=".", help="Project root")

    # update
    update_p = subparsers.add_parser("update", help="Incrementally update graph")
    update_p.add_argument("path", nargs="?", default=".", help="Project root")

    # status
    subparsers.add_parser("status", help="Graph statistics")

    coverage_p = subparsers.add_parser(
        "coverage", help="Graph extraction and relationship coverage"
    )
    coverage_p.add_argument(
        "--top", type=int, default=20, help="Maximum missing/isolated examples"
    )

    ra_p = subparsers.add_parser(
        "rust-analyzer-enrich", help="Enrich Rust calls using rust-analyzer"
    )
    ra_p.add_argument("path", nargs="?", default=".", help="Repository root")
    ra_p.add_argument("--binary", default="rust-analyzer", help="rust-analyzer executable")
    ra_p.add_argument("--max-symbols", type=int, default=2_000)

    # search
    search_p = subparsers.add_parser("search", help="Hybrid search")
    search_p.add_argument("query", help="Search query")
    search_p.add_argument("--intent", help="Search intent")
    search_p.add_argument("--limit", type=int, default=20)

    # impact
    impact_p = subparsers.add_parser("impact", help="Impact analysis")
    impact_p.add_argument("target", help="Target symbol")
    impact_p.add_argument("--max-hops", type=int, default=4)
    impact_p.add_argument("--top", "--limit", dest="limit", type=int, default=25)

    # paths
    paths_p = subparsers.add_parser("paths", help="Find paths between symbols")
    paths_p.add_argument("source", help="Source symbol")
    paths_p.add_argument("target", help="Target symbol")
    paths_p.add_argument("--max-hops", type=int, default=5)
    paths_p.add_argument("--top", type=int, default=10)
    paths_p.add_argument("--structural-only", action="store_true")

    # callers
    callers_p = subparsers.add_parser("callers", help="Find callers of a symbol")
    callers_p.add_argument("target", help="Target symbol")

    # callees
    callees_p = subparsers.add_parser("callees", help="Find callees of a symbol")
    callees_p.add_argument("target", help="Target symbol")

    # flows
    flows_p = subparsers.add_parser("flows", help="List execution flows")
    flows_p.add_argument("--top", type=int, default=20)

    # architecture
    architecture_p = subparsers.add_parser("architecture", help="Architecture overview")
    architecture_p.add_argument("--detail-level", default="standard")

    # communities
    comm_p = subparsers.add_parser("communities", help="Community detection")
    comm_p.add_argument("--algorithm", default="louvain", choices=["louvain", "leiden", "infomap"])
    comm_p.add_argument("--top", type=int, default=20)

    # bridge-nodes
    bridge_p = subparsers.add_parser("bridge-nodes", help="Find bridge/chokepoint nodes")
    bridge_p.add_argument("--top", type=int, default=25)

    # god-nodes
    god_p = subparsers.add_parser("god-nodes", help="Find top PageRank nodes")
    god_p.add_argument("--top", type=int, default=10)
    god_p.add_argument("--seed")

    # cycles
    cycles_p = subparsers.add_parser("cycles", help="Find dependency cycles")
    cycles_p.add_argument("--top", type=int, default=25)

    # articulation
    articulation_p = subparsers.add_parser("articulation", help="Find articulation points")
    articulation_p.add_argument("--top", type=int, default=25)

    # large-functions
    lf_p = subparsers.add_parser("large-functions", help="Find large functions")
    lf_p.add_argument("--min-lines", type=int, default=80)
    lf_p.add_argument("--top", type=int, default=50)

    # dead-code
    subparsers.add_parser("dead-code", help="Find unreachable code")

    # counterfactual
    cf_p = subparsers.add_parser("counterfactual", help="What breaks if removed?")
    cf_p.add_argument("target", help="Target symbol")
    cf_p.add_argument("--direction", default="out")
    cf_p.add_argument("--max-depth", type=int, default=5)

    # context (minimal_context)
    ctx_p = subparsers.add_parser("context", help="Bounded bidirectional neighborhood")
    ctx_p.add_argument("target", help="Target symbol")
    ctx_p.add_argument("--max-hops", type=int, default=2)
    ctx_p.add_argument("--mode", default="review")

    # motifs
    motif_p = subparsers.add_parser("motifs", help="Subgraph motif matching")
    motif_p.add_argument("--built-in", "--pattern", dest="pattern", default="security_audit")
    motif_p.add_argument("--limit", type=int, default=50)

    # detect-changes
    dc_p = subparsers.add_parser("detect-changes", help="Detect changes from Git diff")
    dc_p.add_argument("--base", help="Git base reference")
    dc_p.add_argument("--max-depth", type=int, default=2)
    dc_p.add_argument("--brief", action="store_true", help="Brief output")

    # risk
    risk_p = subparsers.add_parser("risk", help="Risk assessment")
    risk_p.add_argument("--base", help="Git base reference")
    risk_p.add_argument("--top", type=int, default=25)

    # test-coverage
    coverage_p = subparsers.add_parser("test-coverage", help="Test coverage analysis")
    coverage_p.add_argument("target", nargs="?")
    coverage_p.add_argument("--base")

    generic_operations = {
        "traverse": "Budgeted graph traversal",
        "review-context": "Token-budgeted review context",
        "affected-flows": "Flows affected by changes",
        "blast-radius": "Compact impact summary",
        "core": "Rank nodes by graph coreness",
        "gaps": "Structural weaknesses",
        "diagnostics": "Graph health diagnostics",
        "surprises": "Unexpected relationships",
        "suggested-questions": "Suggested review questions",
        "dedup": "Deduplicate semantic nodes",
        "patterns": "Detect framework patterns",
        "rename-preview": "Preview a symbol rename",
        "token-savings": "Estimate token savings",
        "token-benchmark": "Compare retrieval token costs",
    }
    for name, help_text in generic_operations.items():
        operation_p = subparsers.add_parser(name, help=help_text)
        operation_p.add_argument("--params", default="{}", help="JSON operation parameters")
        if name in {"traverse", "rename-preview"}:
            operation_p.add_argument("target")
        if name == "rename-preview":
            operation_p.add_argument("new_name")
        if name in {"review-context", "affected-flows", "blast-radius", "suggested-questions"}:
            operation_p.add_argument("--base", default="HEAD~1")
        if name in {"affected-flows", "blast-radius", "core", "gaps", "diagnostics", "surprises", "suggested-questions"}:
            operation_p.add_argument("--top", type=int)
        if name == "traverse":
            operation_p.add_argument("--direction", default="both")
            operation_p.add_argument("--max-depth", type=int, default=3)
            operation_p.add_argument("--token-budget", type=int, default=1200)
        if name == "review-context":
            operation_p.add_argument("--max-lines-per-file", type=int, default=200)
            operation_p.add_argument("--token-budget", type=int, default=1600)
        if name == "blast-radius":
            operation_p.add_argument("--max-depth", type=int, default=2)
        if name == "dedup":
            operation_p.add_argument("--threshold", type=float, default=0.92)
            operation_p.add_argument("--community-boost", type=float, default=0.05)
            operation_p.add_argument("--community-algo")
        if name == "patterns":
            operation_p.add_argument("--format", default="json")
        if name == "token-savings":
            operation_p.add_argument("--mode", default="json")
            operation_p.add_argument("--include-files", action="store_true")
        if name == "token-benchmark":
            operation_p.add_argument("-q", "--question", action="append")
            operation_p.add_argument("--mode", default="json")

    diff_p = subparsers.add_parser("graph-diff", help="Diff two stored revisions")
    diff_p.add_argument("base_pos", nargs="?")
    diff_p.add_argument("head_pos", nargs="?")
    diff_p.add_argument("--base", default="HEAD~1")
    diff_p.add_argument("--head", default="HEAD")
    diff_p.add_argument("--top", type=int, default=50)

    snapshot_p = subparsers.add_parser("snapshot-diff", help="Diff two graph databases")
    snapshot_p.add_argument("databases", nargs="+")
    snapshot_p.add_argument("--top", type=int, default=50)

    subparsers.add_parser("snapshots", help="List stored graph snapshots")
    subparsers.add_parser("rebuild-fts", help="Rebuild the FTS index")

    embed_p = subparsers.add_parser("embed", help="Build persistent local embeddings")
    embed_p.add_argument("--model", default="all-MiniLM-L6-v2")

    external_p = subparsers.add_parser("embed-external", help="Build persistent remote embeddings")
    external_p.add_argument(
        "--provider",
        required=True,
        choices=["openai", "google", "ollama", "openai-embedding", "google-embedding", "ollama-embedding"],
    )
    external_p.add_argument("--model")
    external_p.add_argument("--api-key")
    external_p.add_argument("--api-key-env")
    external_p.add_argument("--base-url")
    external_p.add_argument("--dimension", type=int)
    external_p.add_argument("--batch-size", type=int, default=64)

    subparsers.add_parser("jedi-enrich", help="Enrich Python call edges using Jedi")
    subparsers.add_parser("spring-di-resolve", help="Resolve Spring dependency injection")

    eval_p = subparsers.add_parser("eval", help="Run evaluation benchmarks")
    eval_p.add_argument("--repos", nargs="*")
    eval_p.add_argument("--benchmarks", help="Comma-separated benchmark names")
    eval_p.add_argument("--output-dir", default="evaluate/results")
    eval_p.add_argument("--embed", action="store_true")

    # watch
    watch_p = subparsers.add_parser("watch", help="Watch project and update graph")
    watch_p.add_argument("path", nargs="?", default=".", help="Project root")
    watch_p.add_argument("--interval", type=int, default=2, help="Poll interval (s, fallback)")

    # serve
    serve_p = subparsers.add_parser("serve", help="Start graph explorer HTTP server")
    serve_p.add_argument("--bind", help="Address:port to bind")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8787)
    serve_p.add_argument("--algorithm", default="louvain", choices=["louvain", "leiden", "infomap"])

    # tool
    tool_p = subparsers.add_parser("tool", help="JSON tool interface")
    tool_p.add_argument("operation", help="Operation name")
    tool_p.add_argument("--params", help="JSON parameters", default="{}")

    # mcp-server
    subparsers.add_parser("mcp-server", help="Start MCP server")

    # wiki
    wiki_p = subparsers.add_parser("wiki", help="Generate markdown wiki from communities")
    wiki_p.add_argument("--output", default="docs/wiki", help="Output directory")
    wiki_p.add_argument("--force", action="store_true", help="Overwrite existing files")

    # daemon
    daemon_p = subparsers.add_parser("daemon", help="Multi-repo daemon")
    daemon_p.add_argument("subcommand", choices=["add", "start", "status"], help="Daemon subcommand")
    daemon_p.add_argument("path", nargs="?", default=".", help="Repository path")
    daemon_p.add_argument("--alias", default="", help="Repository alias")
    daemon_p.add_argument("--interval", type=int, default=5, help="Poll interval (s)")

    # report
    report_p = subparsers.add_parser("report", help="Write a Markdown report to a file")
    report_p.add_argument("output", help="Output file path")
    report_p.add_argument("--top", type=int, default=25, help="Max items per section")

    # install
    install_p = subparsers.add_parser("install", help="Install git hooks and agent configs")
    install_p.add_argument("path", nargs="?", default=".", help="Repository root")
    install_p.add_argument("--repo")
    install_p.add_argument("--force", action="store_true", help="Overwrite existing")
    install_p.add_argument("--agents", action="store_true", help="Install AGENTS.md")
    install_p.add_argument("--mcp", action="store_true", help="Install MCP configs")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    if not args.command:
        parser.print_help()
        return

    # Execute command
    commands = {
        "build": cmd_build,
        "update": cmd_update,
        "status": cmd_status,
        "coverage": cmd_coverage,
        "rust-analyzer-enrich": cmd_rust_analyzer_enrich,
        "search": cmd_search,
        "impact": cmd_impact,
        "paths": cmd_paths,
        "callers": cmd_callers,
        "callees": cmd_callees,
        "flows": cmd_flows,
        "architecture": cmd_architecture,
        "communities": cmd_communities,
        "bridge-nodes": cmd_bridge_nodes,
        "god-nodes": cmd_god_nodes,
        "cycles": cmd_cycles,
        "articulation": cmd_articulation,
        "large-functions": cmd_large_functions,
        "dead-code": cmd_dead_code,
        "counterfactual": cmd_counterfactual,
        "context": cmd_context,
        "motifs": cmd_motifs,
        "detect-changes": cmd_detect_changes,
        "risk": cmd_risk,
        "test-coverage": cmd_test_coverage,
        "tool": cmd_tool,
        "mcp-server": cmd_mcp_server,
        "report": cmd_report,
        **{name: cmd_generic_operation for name in generic_operations},
        "graph-diff": cmd_graph_diff,
        "snapshot-diff": cmd_snapshot_diff,
        "snapshots": cmd_snapshots,
        "rebuild-fts": cmd_rebuild_fts,
        "embed": cmd_embed,
        "embed-external": cmd_embed_external,
        "jedi-enrich": cmd_jedi_enrich,
        "spring-di-resolve": cmd_spring_di_resolve,
        "eval": cmd_eval,
        "watch": cmd_watch,
        "serve": cmd_serve,
        "wiki": cmd_wiki,
        "daemon": cmd_daemon,
        "install": cmd_install,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


def _load_store(args: argparse.Namespace) -> GraphStore:
    """Load or create a GraphStore."""
    db_path = Path(args.db)
    if db_path.exists():
        return GraphStore(db_path)
    return GraphStore(db_path)


def cmd_build(args: argparse.Namespace) -> None:
    """Build a graph from a project root."""
    root = Path(args.path).resolve()
    if not root.exists():
        print(f"Error: {root} does not exist", file=sys.stderr)
        sys.exit(1)

    store = _load_store(args)
    registry = LanguageRegistry()
    pipeline = ExtractionPipeline(registry, strict=True)

    print(f"Building graph from {root}...", file=sys.stderr)
    graph = pipeline.build(root)

    evidence_path = root / ".graphician" / "compiler-evidence.json"
    if evidence_path.exists():
        evidence = load_compiler_evidence(evidence_path)
        report = apply_compiler_evidence(graph, evidence)
        logger.info(
            "Compiler enrichment (%s): added=%d upgraded=%d unresolved=%d",
            evidence.provider,
            report.added,
            report.upgraded,
            report.unresolved,
        )

    commit = git_commit_hash(root)
    if commit is not None:
        _stamp_valid_from(graph, commit)
    store.save_graph(graph, pipeline._file_hashes)
    store.set_metadata("repository_root", str(root))
    if commit is not None:
        store.set_metadata("indexed_commit", commit)
    print(f"Built: {graph.node_count()} nodes, {graph.edge_count()} edges", file=sys.stderr)
    store.close()


def cmd_update(args: argparse.Namespace) -> None:
    """Incrementally update graph."""
    root = Path(args.path).resolve()
    store = _load_store(args)

    registry = LanguageRegistry()
    pipeline = ExtractionPipeline(registry, strict=True)

    # Get current file hashes
    files = pipeline.discover_files(root)
    current_hashes: dict[str, str] = {}
    for f in files:
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
            current_hashes[str(f.relative_to(root))] = _hash(content)
        except OSError:
            pass

    changed, deleted = store.get_changed_files(current_hashes)
    print(f"Changed: {len(changed)}, Deleted: {len(deleted)}", file=sys.stderr)

    if not changed and not deleted:
        print("No changes detected.", file=sys.stderr)
        store.close()
        return

    previous_revision = store.get_metadata("indexed_commit") or store.get_metadata("last_updated")
    if previous_revision and store.status()["node_count"]:
        store.create_snapshot(previous_revision)

    existing = store.load_graph()
    graph = pipeline.update(root, existing, changed, deleted)
    evidence_path = root / ".graphician" / "compiler-evidence.json"
    if evidence_path.exists():
        apply_compiler_evidence(graph, load_compiler_evidence(evidence_path))
    commit = git_commit_hash(root)
    if commit is not None:
        _stamp_valid_from(graph, commit)
    store.save_graph_incremental(graph, pipeline._file_hashes or current_hashes)
    store.set_metadata("repository_root", str(root))
    if commit is not None:
        store.set_metadata("indexed_commit", commit)
    print(f"Updated: {graph.node_count()} nodes, {graph.edge_count()} edges", file=sys.stderr)
    store.close()


def cmd_status(args: argparse.Namespace) -> None:
    """Show graph statistics."""
    store = _load_store(args)
    status = store.status()
    status["graph_freshness"] = graph_freshness(store)
    print(json.dumps(status, indent=2))
    store.close()


def cmd_coverage(args: argparse.Namespace) -> None:
    """Show graph extraction and relationship coverage."""
    if args.top < 0:
        raise SystemExit("coverage: --top must be non-negative")
    with _load_store(args) as store:
        result = graph_coverage(
            store.load_graph(),
            expected_files=store.get_file_hashes(),
            example_limit=args.top,
        )
    print(json.dumps(result, indent=2))


def cmd_rust_analyzer_enrich(args: argparse.Namespace) -> None:
    """Enrich persisted Rust call edges through rust-analyzer."""
    store = _load_store(args)
    graph = store.load_graph()
    report = enrich_with_rust_analyzer(
        graph,
        Path(args.path),
        RustAnalyzerOptions(Path(args.binary), args.max_symbols),
    )
    store.save_graph(graph, store.get_file_hashes())
    print(
        json.dumps(
            {
                "symbols_queried": report.symbols_queried,
                "compiler_edges": report.compiler_edges,
                "added": report.enrichment.added,
                "upgraded": report.enrichment.upgraded,
                "unresolved": report.adapter_unresolved + report.enrichment.unresolved,
            },
            indent=2,
        )
    )
    store.close()


def cmd_search(args: argparse.Namespace) -> None:
    """Run hybrid search."""
    store = _load_store(args)
    graph = store.load_graph()

    result = hybrid_search(
        graph,
        args.query,
        intent=args.intent,
        limit=args.limit,
    )

    print(json.dumps(result, indent=2))
    store.close()


def cmd_impact(args: argparse.Namespace) -> None:
    """Run impact analysis."""
    store = _load_store(args)
    graph = store.load_graph()

    result = compute_impact(
        graph,
        args.target,
        max_hops=args.max_hops,
        limit=args.limit,
    )

    print(json.dumps(result, indent=2))
    store.close()


def cmd_paths(args: argparse.Namespace) -> None:
    """Find paths between symbols."""
    store = _load_store(args)
    graph = store.load_graph()

    result = handle_paths(graph, {
        "from": args.source,
        "to": args.target,
        "max_hops": args.max_hops,
        "limit": args.top,
    })

    print(json.dumps(result, indent=2))
    store.close()


def cmd_callers(args: argparse.Namespace) -> None:
    """Find callers of a symbol."""
    store = _load_store(args)
    graph = store.load_graph()

    target_id = graph.find_by_qname(args.target)
    if target_id is None:
        print(f"Symbol not found: {args.target}")
        store.close()
        return

    callers = []
    for src, edge in graph.in_neighbors(target_id):
        src_node = graph.node(src)
        if src_node:
            callers.append({
                "qualified_name": src_node.qualified_name,
                "kind": src_node.kind.value,
                "edge_kind": edge.kind.value,
            })

    print(json.dumps({"callers": callers, "total": len(callers)}, indent=2))
    store.close()


def cmd_callees(args: argparse.Namespace) -> None:
    """Find callees of a symbol."""
    store = _load_store(args)
    graph = store.load_graph()

    target_id = graph.find_by_qname(args.target)
    if target_id is None:
        print(f"Symbol not found: {args.target}")
        store.close()
        return

    callees = []
    for dst, edge in graph.out_neighbors(target_id):
        dst_node = graph.node(dst)
        if dst_node:
            callees.append({
                "qualified_name": dst_node.qualified_name,
                "kind": dst_node.kind.value,
                "edge_kind": edge.kind.value,
            })

    print(json.dumps({"callees": callees, "total": len(callees)}, indent=2))
    store.close()


def cmd_flows(args: argparse.Namespace) -> None:
    """List execution flows."""
    store = _load_store(args)
    graph = store.load_graph()

    flows = []
    for _nid, node in graph.nodes():
        if node.kind.value == "flow":
            flows.append({
                "qualified_name": node.qualified_name,
                "entry": node.properties.get("entry", "unknown"),
                "size": node.properties.get("size", 0),
            })

    flows.sort(key=lambda f: f["size"], reverse=True)
    print(json.dumps({"flows": flows[:args.top], "total": len(flows)}, indent=2))
    store.close()


def cmd_architecture(args: argparse.Namespace) -> None:
    """Architecture overview."""
    store = _load_store(args)
    graph = store.load_graph()

    comm = detect_communities(graph)
    hubs = find_hub_nodes(graph)
    gods = find_god_nodes(graph)

    result = {
        "communities": comm["community_count"],
        "modularity": comm["quality"],
        "top_hubs": hubs["hub_nodes"][:10],
        "top_god_nodes": gods["god_nodes"][:10],
    }

    print(json.dumps(result, indent=2))
    store.close()


def cmd_communities(args: argparse.Namespace) -> None:
    """Community detection."""
    store = _load_store(args)
    graph = store.load_graph()

    result = detect_communities(graph, algorithm=args.algorithm)
    print(json.dumps(result, indent=2))
    store.close()


def cmd_bridge_nodes(args: argparse.Namespace) -> None:
    """Find bridge nodes."""
    store = _load_store(args)
    graph = store.load_graph()

    result = find_bridge_nodes(graph)
    print(json.dumps(result, indent=2))
    store.close()


def cmd_god_nodes(args: argparse.Namespace) -> None:
    """Find god nodes."""
    store = _load_store(args)
    graph = store.load_graph()

    result = find_god_nodes(graph)
    print(json.dumps(result, indent=2))
    store.close()


def cmd_cycles(args: argparse.Namespace) -> None:
    """Find cycles."""
    store = _load_store(args)
    graph = store.load_graph()

    result = find_cycles(graph)
    print(json.dumps(result, indent=2))
    store.close()


def cmd_articulation(args: argparse.Namespace) -> None:
    """Find articulation points."""
    store = _load_store(args)
    graph = store.load_graph()

    result = find_articulation_points(graph)
    print(json.dumps(result, indent=2))
    store.close()


def cmd_large_functions(args: argparse.Namespace) -> None:
    """Find large functions."""
    store = _load_store(args)
    graph = store.load_graph()

    result = find_large_functions(graph, min_lines=args.min_lines)
    print(json.dumps(result, indent=2))
    store.close()


def cmd_dead_code(args: argparse.Namespace) -> None:
    """Find dead code."""
    store = _load_store(args)
    graph = store.load_graph()

    result = find_dead_code(graph)
    print(json.dumps(result, indent=2))
    store.close()


def cmd_counterfactual(args: argparse.Namespace) -> None:
    """Counterfactual analysis."""
    store = _load_store(args)
    graph = store.load_graph()

    result = find_counterfactual(graph, args.target)
    print(json.dumps(result, indent=2))
    store.close()


def cmd_context(args: argparse.Namespace) -> None:
    """Minimal context: bounded bidirectional neighborhood."""
    store = _load_store(args)
    graph = store.load_graph()

    result = _minimal_context(graph, {
        "target": args.target,
        "max_hops": args.max_hops,
        "mode": args.mode,
    })
    print(json.dumps(result, indent=2))
    store.close()


def cmd_motifs(args: argparse.Namespace) -> None:
    """Motif detection."""
    store = _load_store(args)
    graph = store.load_graph()

    result = find_motifs(graph, pattern=args.pattern)
    print(json.dumps(result, indent=2))
    store.close()


def cmd_detect_changes(args: argparse.Namespace) -> None:
    """Detect changes from Git diff."""
    store = _load_store(args)
    graph = store.load_graph()

    # Read diff from stdin or use --base
    diff_text = ""
    if sys.stdin.isatty():
        if args.base:
            import subprocess
            diff_result = subprocess.run(
                ["git", "diff", args.base],
                capture_output=True, text=True
            )
            diff_text = diff_result.stdout
        else:
            print("Error: provide --base or pipe diff to stdin", file=sys.stderr)
            sys.exit(1)
    else:
        diff_text = sys.stdin.read()

    result = detect_changes(graph, diff_text, base=args.base)
    print(json.dumps(result, indent=2))
    store.close()


def cmd_risk(args: argparse.Namespace) -> None:
    """Risk assessment."""
    store = _load_store(args)
    graph = store.load_graph()

    result = compute_risk(graph, base=args.base, top=args.top)
    print(json.dumps(result, indent=2))
    store.close()


def cmd_test_coverage(args: argparse.Namespace) -> None:
    """Test coverage analysis."""
    store = _load_store(args)
    graph = store.load_graph()

    result = compute_test_coverage(graph, base=args.base)
    print(json.dumps(result, indent=2))
    store.close()


def cmd_tool(args: argparse.Namespace) -> None:
    """JSON tool interface."""
    params = json.loads(args.params)
    from .response import tool_response

    result = tool_response(
        str(args.db),
        args.operation.replace("-", "_"),
        params,
    )

    print(json.dumps(result, indent=2))


def cmd_generic_operation(args: argparse.Namespace) -> None:
    args.operation = args.command.replace("-", "_")
    params = json.loads(args.params)
    ignored = {"command", "db", "verbose", "params", "operation"}
    for key, value in vars(args).items():
        if key not in ignored and value is not None:
            params[key] = value
    if "question" in params:
        params["questions"] = params.pop("question")
    args.params = json.dumps(params)
    cmd_tool(args)


def _normalize_grouped_argv(argv: list[str]) -> list[str]:
    """Accept the Rust command hierarchy while retaining flat aliases."""
    result = list(argv)
    index = 0
    while index < len(result):
        if result[index] in {"-d", "--db"}:
            index += 2
            continue
        if result[index] in {"-v", "--verbose"}:
            index += 1
            continue
        break
    if index >= len(result):
        return result
    group = result[index]
    grouped = {"analysis", "git", "structure", "advanced", "agent", "maintenance", "utility"}
    if group in grouped and index + 1 < len(result):
        result.pop(index)
    elif group == "build" and index + 1 < len(result):
        nested = result[index + 1]
        if nested in {"update", "watch", "daemon", "install", "serve", "status"}:
            result[index : index + 2] = [nested]
    return result


def cmd_graph_diff(args: argparse.Namespace) -> None:
    with _load_store(args) as store:
        base = args.base_pos or args.base
        head = args.head_pos or args.head
        print(json.dumps(graph_diff(store.load_graph_at(base), store.load_graph_at(head)), indent=2))


def cmd_snapshot_diff(args: argparse.Namespace) -> None:
    if len(args.databases) == 1:
        base_db, head_db = args.db, args.databases[0]
    elif len(args.databases) == 2:
        base_db, head_db = args.databases
    else:
        raise ValueError("snapshot-diff expects HEAD_DB or BASE_DB HEAD_DB")
    with GraphStore(base_db) as base, GraphStore(head_db) as head:
        print(json.dumps(graph_diff(base.load_graph(), head.load_graph()), indent=2))


def cmd_snapshots(args: argparse.Namespace) -> None:
    with _load_store(args) as store:
        print(json.dumps({"snapshots": store.list_snapshots()}, indent=2))


def cmd_rebuild_fts(args: argparse.Namespace) -> None:
    with _load_store(args) as store:
        print(json.dumps({"indexed": store.rebuild_fts()}))


def cmd_embed(args: argparse.Namespace) -> None:
    with _load_store(args) as store:
        vectors = build_local_embeddings(store.load_graph(), args.model)
        store.save_embeddings(args.model, vectors)
        print(json.dumps({"model": args.model, "indexed": len(vectors)}))


def cmd_embed_external(args: argparse.Namespace) -> None:
    with _load_store(args) as store:
        provider = args.provider.removesuffix("-embedding")
        model = args.model or {
            "openai": "text-embedding-3-small",
            "google": "text-embedding-004",
            "ollama": "nomic-embed-text",
        }[provider]
        api_key = args.api_key or (os.getenv(args.api_key_env) if args.api_key_env else None)
        vectors = build_external_embeddings(
            store.load_graph(),
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=args.base_url,
            batch_size=args.batch_size,
        )
        model_key = f"{provider}:{model}"
        store.save_embeddings(model_key, vectors)
        print(json.dumps({"model": model_key, "indexed": len(vectors)}))


def cmd_jedi_enrich(args: argparse.Namespace) -> None:
    with _load_store(args) as store:
        graph = store.load_graph()
        root = Path(store.get_metadata("repository_root") or ".")
        added = enrich_jedi_calls(graph, root)
        store.save_graph_incremental(graph, store.get_file_hashes())
        print(json.dumps({"operation": "jedi_enrich", "added": added}))


def cmd_spring_di_resolve(args: argparse.Namespace) -> None:
    with _load_store(args) as store:
        graph = store.load_graph()
        added = resolve_spring_injections(graph)
        store.save_graph_incremental(graph, store.get_file_hashes())
        print(json.dumps({"operation": "spring_di_resolve", "added": added}))


def cmd_eval(args: argparse.Namespace) -> None:
    from ...evaluation import run_eval

    benchmarks = args.benchmarks.split(",") if args.benchmarks else []
    results = run_eval(
        repos=args.repos or [],
        benchmarks=benchmarks,
        output_dir=Path(args.output_dir),
        embed=args.embed,
    )
    print(json.dumps({"operation": "eval", "results": results}, indent=2))


def cmd_mcp_server(args: argparse.Namespace) -> None:
    """Start MCP server."""
    from ..transport.mcp import GraphicianMCP
    mcp = GraphicianMCP()
    mcp.run()


def cmd_report(args: argparse.Namespace) -> None:
    """Write a Markdown report to a file."""
    from .response import generate_report_markdown

    md = generate_report_markdown(str(args.db), top=args.top)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    print(f"Report written to {output_path}")


def cmd_watch(args: argparse.Namespace) -> None:
    """Watch project and incrementally update graph."""
    from .watch import cmd_watch as _watch
    _watch(args.db, args.path, args.interval)


def cmd_serve(args: argparse.Namespace) -> None:
    """Start graph explorer HTTP server."""
    from .serve import cmd_serve as _serve
    _serve(args.db, args.bind or f"{args.host}:{args.port}", args.algorithm)


def cmd_wiki(args: argparse.Namespace) -> None:
    """Generate markdown wiki from communities."""
    from .wiki import cmd_wiki as _wiki
    _wiki(args.db, args.output, args.force)


def cmd_daemon(args: argparse.Namespace) -> None:
    """Multi-repo daemon."""
    from .daemon import cmd_daemon as _daemon
    _daemon(args.subcommand, args.db, args.path, args.alias, args.interval)


def cmd_install(args: argparse.Namespace) -> None:
    """Install git hooks and agent configs."""
    from .daemon import cmd_install as _install
    _install(args.db, args.repo or args.path, args.force, args.agents, args.mcp)


def _node_to_dict(graph: Graph, qname: str) -> dict[str, Any] | None:
    """Convert a node to a dict."""
    nid = graph.find_by_qname(qname)
    if nid is None:
        return None
    node = graph.node(nid)
    if node is None:
        return None
    return {
        "qualified_name": node.qualified_name,
        "kind": node.kind.value,
        "name": node.name,
        "source_uri": node.source_uri,
    }


def _hash(text: str) -> str:
    """Compute SHA-256 hash."""
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def _stamp_valid_from(graph: Graph, revision: str) -> None:
    """Stamp newly introduced graph rows with their indexed revision."""
    for _, node in graph.nodes():
        if node.valid_from is None:
            node.valid_from = revision
    for _, _, _, edge in graph.edges():
        if edge.valid_from is None:
            edge.valid_from = revision
