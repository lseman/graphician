"""Control flow analysis: per-function CFG, def-use chains, and intra-procedural analysis.

Builds Control Flow Graphs (CFGs) from function source code, then performs:
- Def-use chain analysis (where variables are defined and used)
- Reachable block analysis (dead code within functions)
- Post-dominator analysis (critical path identification)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.graph import Graph
from ..core.id import NodeId
from ..core.node import NodeKind
from ..extraction.data_flow import extract_params

logger = logging.getLogger(__name__)


# ── CFG node types ───────────────────────────────────────────────────────


@dataclass
class CFGNode:
    """A node in a Control Flow Graph."""
    label: str
    line: int
    kind: str
    node_id: str = ""
    data_defs: list[str] = field(default_factory=list)
    data_uses: list[str] = field(default_factory=list)
    successors: list[str] = field(default_factory=list)
    predecessors: list[str] = field(default_factory=list)


@dataclass
class DefUseChain:
    """A def-use chain for a variable."""
    variable: str
    definition_line: int
    definition_block: str
    uses: list[tuple[int, str]]


@dataclass
class ReachableBlocks:
    """Result of reachable block analysis."""
    total_blocks: int
    reachable_blocks: int
    dead_blocks: list[tuple[str, int, str]]
    dead_line_count: int


@dataclass
class CriticalPath:
    """The critical path through a function's CFG."""
    path: list[tuple[str, int, str]]
    path_length: int
    dominated_by: list[str]


# ── CFG Construction ────────────────────────────────────────────────────


def build_cfg(
    source_text: str,
    source_path: str = "",
    function_qname: str = "",
) -> list[CFGNode]:
    """Build a Control Flow Graph from function source code."""
    lines = source_text.splitlines()
    dialect = _detect_dialect(source_path)
    cfg_nodes: list[CFGNode] = []

    # Entry node
    entry = CFGNode(label="__ENTRY__", line=0, kind="entry")
    cfg_nodes.append(entry)

    # Analyze each line
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        line_num = i + 1

        if not stripped or stripped.startswith("#") or stripped.startswith("//") or stripped.startswith("*"):
            i += 1
            continue

        kind = _classify_line(stripped, dialect)
        defs, uses = _extract_defs_uses(stripped, dialect)

        node = CFGNode(
            label=f"L{line_num}",
            line=line_num,
            kind=kind,
            data_defs=defs,
            data_uses=uses,
        )
        cfg_nodes.append(node)

        if kind in ("branch", "loop", "block"):
            i = _skip_block(lines, i, dialect)
        else:
            i += 1

    # Add exit node FIRST, then build edges (so cfg_nodes[-1] is the exit)
    exit_node = CFGNode(label="__EXIT__", line=len(lines), kind="exit")
    cfg_nodes.append(exit_node)

    # Build successor/predecessor edges
    _build_cfg_edges(cfg_nodes, dialect)

    return cfg_nodes


def _detect_dialect(source_path: str) -> str:
    ext = Path(source_path).suffix.lower() if source_path else ""
    if ext == ".py":
        return "python"
    elif ext == ".rs":
        return "rust"
    elif ext in (".js", ".jsx", ".ts", ".tsx"):
        return "typescript"
    elif ext in (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"):
        return "cpp"
    return "python"


def _classify_line(stripped: str, dialect: str) -> str:
    if stripped.startswith("def ") or stripped.startswith("fn "):
        return "entry"
    if stripped.startswith("return") or stripped.startswith("yield"):
        return "return"

    branch_patterns = ["if ", "elif", "else:", "switch", "case ", "when "]
    if dialect in ("rust", "cpp", "typescript"):
        branch_patterns.extend(["if let", "match ", "else if"])
    if any(stripped.startswith(p) for p in branch_patterns):
        return "branch"

    loop_patterns = ["for ", "while ", "loop {", "for <", "loop "]
    if dialect in ("rust", "cpp", "typescript"):
        loop_patterns.extend(["do {", "foreach"])
    if any(stripped.startswith(p) for p in loop_patterns):
        return "loop"

    if "=" in stripped and not stripped.startswith("#") and not stripped.startswith("//") and not stripped.startswith("assert"):
        return "assign"

    call_patterns = [".", "::", "->"]
    if any(p in stripped for p in call_patterns) and (stripped.endswith(";") or stripped.endswith(")") or stripped.endswith("}")):
        return "call"

    return "block"


def _extract_defs_uses(stripped: str, dialect: str) -> tuple[list[str], list[str]]:
    if dialect == "python":
        return _extract_python_defs(stripped), _extract_python_uses(stripped)
    elif dialect == "rust":
        return _extract_rust_defs(stripped), _extract_rust_uses(stripped)
    elif dialect in ("typescript", "cpp"):
        return _extract_ts_cpp_defs(stripped), _extract_ts_cpp_uses(stripped)
    return [], []


def _extract_python_defs(line: str) -> list[str]:
    defs = []
    if "=" in line and not line.startswith("#"):
        parts = line.split("=", 1)
        target = parts[0].strip()
        for part in target.replace("(", "").replace(")", "").split(","):
            part = part.strip()
            if part and part.isidentifier() and not part.startswith("_"):
                defs.append(part)
    if line.startswith("for "):
        rest = line[4:].strip()
        if " in " in rest:
            target = rest.split(" in ", 1)[0].strip()
            for part in target.replace("(", "").replace(")", "").split(","):
                part = part.strip()
                if part and part.isidentifier():
                    defs.append(part)
    return defs


def _extract_python_uses(line: str) -> list[str]:
    uses = []
    skip = {"def", "class", "if", "else", "elif", "for", "while", "return",
            "yield", "import", "from", "as", "with", "try", "except",
            "finally", "raise", "pass", "break", "continue", "and", "or",
            "not", "in", "is", "lambda", "None", "True", "False", "self",
            "cls", "print", "range", "len", "str", "int", "float", "list",
            "dict", "set", "tuple", "type", "super"}
    for token in line.replace("=", " ").replace(",", " ").replace("(", " ").replace(")", " ").split():
        if token and token.isidentifier() and token not in skip:
            uses.append(token)
    return uses


def _extract_rust_defs(line: str) -> list[str]:
    defs = []
    if line.startswith("let "):
        rest = line[4:].strip()
        if rest.startswith("mut "):
            rest = rest[4:].strip()
        target = rest.split("=", 1)[0].strip().rstrip(":").strip()
        for part in target.replace("(", "").replace(")", "").replace("&", "").split(","):
            part = part.strip()
            if part and part.isidentifier() and part != "mut":
                defs.append(part)
    return defs


def _extract_rust_uses(line: str) -> list[str]:
    return [t for t in line.replace("=", " ").replace(";", " ").replace(",", " ").split()
            if t and t[0].islower() and t.isidentifier() and not t.startswith("#")]


def _extract_ts_cpp_defs(line: str) -> list[str]:
    defs = []
    for kw in ["let ", "const ", "var "]:
        if line.startswith(kw):
            rest = line[len(kw):].strip()
            if "=" in rest:
                target = rest.split("=", 1)[0].strip().rstrip(":").strip()
                for part in target.replace("(", "").replace(")", "").split(","):
                    part = part.strip()
                    if part and part.isidentifier():
                        defs.append(part)
            break
    return defs


def _extract_ts_cpp_uses(line: str) -> list[str]:
    keywords = {"let", "const", "var", "if", "else", "for", "while", "return",
                "function", "class", "new", "this", "null", "undefined", "true",
                "false", "async", "await", "import", "export", "from", "typeof",
                "switch", "case", "break", "continue", "throw", "try", "catch",
                "do", "in", "of", "public", "private", "protected", "static",
                "readonly", "enum", "interface", "type", "extends", "implements"}
    uses = []
    for token in line.replace("=", " ").replace(";", " ").replace(",", " ").split():
        clean = token.strip("(){}[]:;,.")
        if clean and clean[0].islower() and clean.isidentifier() and clean not in keywords:
            uses.append(clean)
    return uses


def _skip_block(lines: list[str], start: int, dialect: str) -> int:
    brace_count = 0
    i = start
    while i < len(lines):
        line = lines[i]
        if "{" in line:
            brace_count += line.count("{")
        if "}" in line:
            brace_count -= line.count("}")
        if brace_count <= 0 and i > start:
            return i
        if dialect == "python" and not line.startswith("#") and line.strip() and i > start and lines[i].startswith(("def ", "class ")):
            return i
        i += 1
    return i


def _build_cfg_edges(cfg_nodes: list[CFGNode], dialect: str) -> None:
    """Build successor/predecessor edges in the CFG.

    The exit node must be the last element of cfg_nodes when called.
    """
    exit_node = cfg_nodes[-1]
    stack: list[int] = [0]  # Index of last node at each nesting level

    for i in range(1, len(cfg_nodes) - 1):
        node = cfg_nodes[i]
        parent_idx = stack[-1] if stack else 0
        cfg_nodes[parent_idx].successors.append(node.label)
        node.predecessors.append(cfg_nodes[parent_idx].label)

        if node.kind in ("branch", "loop"):
            stack.append(i)
        elif node.kind == "return":
            node.successors.append(exit_node.label)
            exit_node.predecessors.append(node.label)
            # Don't update stack; return exits the function
        else:
            stack[-1] = i


# ── Def-Use Chain Analysis ──────────────────────────────────────────────


def compute_def_use_chains(
    cfg_nodes: list[CFGNode],
    function_params: list[str] | None = None,
) -> list[DefUseChain]:
    def_points: dict[str, list[tuple[int, str]]] = {}
    use_points: dict[str, list[tuple[int, str]]] = {}

    for node in cfg_nodes:
        for var in node.data_defs:
            def_points.setdefault(var, []).append((node.line, node.label))
        for var in node.data_uses:
            use_points.setdefault(var, []).append((node.line, node.label))

    if function_params:
        for var in function_params:
            def_points.setdefault(var, []).append((0, "__ENTRY__"))

    chains: list[DefUseChain] = []
    all_vars = set(list(def_points.keys()) + list(use_points.keys()))

    for var in sorted(all_vars):
        defs = def_points.get(var, [])
        uses = use_points.get(var, [])
        if defs:
            first_def = defs[0]
            subsequent_uses = [(line, block) for line, block in uses if line >= first_def[0]]
            chains.append(DefUseChain(
                variable=var,
                definition_line=first_def[0],
                definition_block=first_def[1],
                uses=subsequent_uses,
            ))

    return chains


# ── Reachable Block Analysis ────────────────────────────────────────────


def analyze_reachable_blocks(
    cfg_nodes: list[CFGNode],
) -> ReachableBlocks:
    entry = cfg_nodes[0]
    reachable: set[str] = {entry.label}
    queue = [entry.label]

    while queue:
        current = queue.pop(0)
        current_node = next((n for n in cfg_nodes if n.label == current), None)
        if current_node is None:
            continue
        for succ_label in current_node.successors:
            if succ_label not in reachable:
                reachable.add(succ_label)
                queue.append(succ_label)

    dead_blocks = []
    for node in cfg_nodes[1:-1]:  # Skip entry and exit
        if node.label not in reachable:
            dead_blocks.append((node.label, node.line, node.label))

    return ReachableBlocks(
        total_blocks=len(cfg_nodes),
        reachable_blocks=len(reachable) - 2,
        dead_blocks=dead_blocks,
        dead_line_count=len(dead_blocks),
    )


# ── Post-Dominator & Critical Path ──────────────────────────────────────


def compute_post_dominators(
    cfg_nodes: list[CFGNode],
) -> dict[str, str]:
    node_map = {n.label: n for n in cfg_nodes}
    exit_label = cfg_nodes[-1].label
    post_dom: dict[str, str] = {exit_label: exit_label}
    all_labels = [n.label for n in cfg_nodes]

    for label in all_labels:
        if label != exit_label:
            post_dom[label] = label

    changed = True
    for _ in range(100):
        if not changed:
            break
        changed = False
        for label in reversed(all_labels[:-1]):
            node = node_map.get(label)
            if node is None:
                continue
            succ_pdoms = [post_dom[s] for s in node.successors if s in post_dom]
            if succ_pdoms:
                new_pdom = succ_pdoms[0]
                for pdom in succ_pdoms[1:]:
                    if pdom == new_pdom:
                        continue
                    path_a = _trace_to_exit(node_map, new_pdom, exit_label)
                    path_b = _trace_to_exit(node_map, pdom, exit_label)
                    pdom_set = set(path_b)
                    for candidate in path_a:
                        if candidate in pdom_set:
                            new_pdom = candidate
                            break
                if new_pdom != post_dom.get(label):
                    post_dom[label] = new_pdom
                    changed = True

    return post_dom


def _trace_to_exit(node_map: dict[str, CFGNode], start: str, exit_label: str) -> list[str]:
    path = [start]
    current = start
    visited = {start}
    while current != exit_label and current in node_map:
        node = node_map[current]
        found = False
        for succ in node.successors:
            if succ not in visited:
                visited.add(succ)
                path.append(succ)
                current = succ
                found = True
                break
        if not found:
            break
    return path


def find_critical_path(
    cfg_nodes: list[CFGNode],
) -> CriticalPath:
    entry = cfg_nodes[0]
    exit_label = cfg_nodes[-1].label
    best_path: list[tuple[str, int, str]] = []

    def dfs(current_label: str, path: list[tuple[str, int, str]], visited: set[str]) -> None:
        nonlocal best_path
        node = next((n for n in cfg_nodes if n.label == current_label), None)
        if node is None:
            return

        current_path = [*path, (node.label, node.line, node.label)]

        if current_label == exit_label:
            if len(current_path) > len(best_path):
                best_path = current_path
            return

        for succ_label in node.successors:
            if succ_label not in visited:
                visited.add(succ_label)
                dfs(succ_label, current_path, visited)
                visited.discard(succ_label)

    dfs(entry.label, [], {entry.label})
    critical_set = {item[0] for item in best_path}

    return CriticalPath(
        path=best_path,
        path_length=len(best_path),
        dominated_by=list(critical_set),
    )


# ── Integration with Graph ──────────────────────────────────────────────


def analyze_function_flow(
    graph: Graph,
    node_id: NodeId,
    limit: int = 50,
) -> dict[str, Any]:
    node = graph.node(node_id)
    if node is None or node.source_text is None:
        return {"error": "Function not found or has no source text"}

    source_text = node.source_text
    source_path = node.source_uri or ""
    function_qname = node.qualified_name

    cfg_nodes = build_cfg(source_text, source_path, function_qname)
    params = extract_params(source_text, source_path)
    chains = compute_def_use_chains(cfg_nodes, params)
    reachable = analyze_reachable_blocks(cfg_nodes)
    _post_doms = compute_post_dominators(cfg_nodes)
    critical = find_critical_path(cfg_nodes)

    return {
        "function": function_qname,
        "source_path": source_path,
        "cfg_nodes": len(cfg_nodes),
        "dead_blocks": reachable.dead_line_count,
        "critical_path_length": critical.path_length,
        "variables_tracked": len(chains),
        "chains": [
            {
                "variable": c.variable,
                "defined_at": c.definition_line,
                "used_at": [u[0] for u in c.uses],
            }
            for c in chains[:limit]
        ],
        "dead_code_blocks": [
            {"line": d[1], "label": d[0]}
            for d in reachable.dead_blocks[:20]
        ],
        "critical_path": [
            {"line": p[1], "label": p[0]}
            for p in critical.path[:20]
        ],
    }


def analyze_all_functions(
    graph: Graph,
    limit: int = 100,
) -> dict[str, Any]:
    functions = [
        (nid, node)
        for nid, node in graph.nodes()
        if node.kind in (NodeKind.FUNCTION, NodeKind.METHOD)
        and node.source_text
        and len(node.source_text.splitlines()) > 1
    ]

    results = []
    total_dead = 0
    total_critical = 0

    for nid, _node in functions[:limit]:
        analysis = analyze_function_flow(graph, nid)
        if "error" not in analysis:
            total_dead += analysis["dead_blocks"]
            total_critical += analysis["critical_path_length"]
            results.append({
                "function": analysis["function"],
                "source_path": analysis["source_path"],
                "dead_blocks": analysis["dead_blocks"],
                "critical_path_length": analysis["critical_path_length"],
                "variables_tracked": analysis["variables_tracked"],
            })

    return {
        "functions_analyzed": len(results),
        "total_dead_blocks": total_dead,
        "average_critical_path_length": total_critical / max(1, len(results)),
        "results": sorted(results, key=lambda r: -r["dead_blocks"]),
    }
