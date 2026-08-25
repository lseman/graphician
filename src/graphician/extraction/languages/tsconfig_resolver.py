"""Post-extraction resolver for TypeScript path aliases.

Tree-sitter parses import source strings verbatim — `@/components/foo`
becomes `module::@/components/foo`. This pass walks the graph, finds
module nodes whose names start with known TS alias prefixes (`@`, `~`,
`#`, or `@src`), loads the nearest `tsconfig.json`, resolves the path
alias, probes for the target file, and renames the module node to the
canonical `file::/abs/path` so that `importers_of` and `impact` work
correctly across files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...core.graph import Graph
from ...core.id import NodeId
from ...core.node import NodeKind

# Extensions probed when resolving an alias target
PROBE_EXTENSIONS: list[str] = [
    ".ts", ".tsx", ".js", ".jsx", ".vue", ".mjs", ".cjs",
]

# Tsconfig filenames to look for when walking up the directory tree
TSCONFIG_NAMES: list[str] = [
    "tsconfig.json", "tsconfig.app.json", "tsconfig.node.json",
]

# Known path aliases that are NOT project-internal aliases but rather
# npm-package imports. These should NOT be resolved to file paths.
BUILTIN_ALIASES: list[str] = [
    "@types", "@angular", "@ionic", "@nx", "@nrwl", "@babel",
    "@testing-library", "@storybook", "@mui", "@next", "@vue",
    "@sveltejs", "@remix-run", "@emotion", "@radix-ui", "@tanstack",
    "@headlessui", "@chakra-ui",
]


def resolve_ts_path_aliases(
    graph: Graph,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Resolve TypeScript path aliases in the graph.

    Walks module nodes, finds TS alias imports, loads tsconfig.json,
    resolves aliases, and renames module nodes to canonical file paths.

    Returns summary of resolution results.
    """
    repo_root = Path(repo_root)
    candidates: list[tuple[NodeId, str]] = []

    for node_id, node in graph.nodes():
        if node.kind != NodeKind.MODULE:
            continue
        qn = node.qualified_name
        if not qn.startswith("module::"):
            continue

        mod_name = qn[len("module::"):]
        if is_ts_alias_path(mod_name) and not is_builtin_alias(mod_name):
            candidates.append((node_id, mod_name))

    if not candidates:
        return {"resolved": 0, "skipped": 0, "errors": 0, "details": []}

    # Build a cache: directory → tsconfig data
    tsconfig_cache: dict[str, TsconfigData] = {}
    resolved = 0
    skipped = 0
    errors = 0
    details: list[dict[str, Any]] = []

    for mod_id, mod_name in candidates:
        data = _find_tsconfig_for_module(tsconfig_cache, mod_id, graph, repo_root)
        if data is None:
            skipped += 1
            details.append({"module": mod_name, "reason": "no_tsconfig"})
            continue

        resolved_path = _resolve_alias(mod_name, data)
        if resolved_path is None:
            skipped += 1
            details.append({"module": mod_name, "reason": "alias_not_found"})
            continue

        resolved_path = Path(resolved_path)
        resolved_qn = f"file::{resolved_path}"
        stem = resolved_path.stem if resolved_path.stem else resolved_path.name
        graph.rename_node(mod_id, resolved_qn, stem)
        resolved += 1
        details.append({
            "module": mod_name,
            "resolved": str(resolved_path),
            "reason": "success",
        })

    return {
        "resolved": resolved,
        "skipped": skipped,
        "errors": errors,
        "details": details,
    }


def is_ts_alias_path(name: str) -> bool:
    """Check if a module name looks like a TypeScript alias path."""
    return (
        name.startswith("@")
        or name.startswith("~")
        or name.startswith("#")
        or name.startswith("@src")
        or name.startswith("@app")
        or name.startswith("@lib")
        or name.startswith("@shared")
    )


def is_builtin_alias(name: str) -> bool:
    """Check if a module name is a known npm package alias."""
    return any(
        name == alias or name.startswith(f"{alias}/")
        for alias in BUILTIN_ALIASES
    )


@dataclass
class TsconfigData:
    """Parsed tsconfig.json data."""
    base_url: str | None = None
    paths: dict[str, list[str]] = field(default_factory=dict)
    tsconfig_dir: str = ""


def _find_tsconfig_for_module(
    cache: dict[str, TsconfigData],
    mod_id: NodeId,
    graph: Graph,
    repo_root: Path,
) -> TsconfigData | None:
    """Find tsconfig data for a module by walking up from the importer's file."""
    # Get the source file of the importer (the file that has this import edge)
    file_uri = None
    for src_id, _ in graph.in_neighbors(mod_id):
        src_node = graph.node(src_id)
        if src_node and src_node.source_uri:
            file_uri = src_node.source_uri
            break

    if file_uri is None:
        return None

    file_dir = str(Path(file_uri).parent)

    if file_dir in cache:
        return cache[file_dir]

    # Resolve the file path relative to repo_root if it's not absolute
    file_path = Path(file_uri)
    if not file_path.is_absolute():
        file_path = repo_root / file_path

    tsconfig_path = _find_nearest_tsconfig(file_path.parent, repo_root)
    if tsconfig_path is None:
        return None

    data = _parse_tsconfig(tsconfig_path)
    if data is None:
        return None

    result = TsconfigData(
        base_url=data.base_url,
        paths=data.paths,
        tsconfig_dir=str(tsconfig_path.parent),
    )

    if data.base_url:
        result.base_url = str(tsconfig_path.parent / data.base_url)

    cache[file_dir] = result
    return result


def _find_nearest_tsconfig(start: Path, repo_root: Path) -> Path | None:
    """Walk up from start directory to find nearest tsconfig.json."""
    try:
        current = start.resolve()
    except (OSError, ValueError):
        return None

    try:
        root = repo_root.resolve()
    except (OSError, ValueError):
        return None

    while True:
        for name in TSCONFIG_NAMES:
            candidate = current / name
            if candidate.is_file():
                return candidate

        if not current.parent or current == root:
            break
        current = current.parent

    # Check root one final time
    for name in TSCONFIG_NAMES:
        candidate = root / name
        if candidate.is_file():
            return candidate

    return None


@dataclass
class RawTsconfig:
    """Raw parsed tsconfig data before path resolution."""
    base_url: str | None = None
    paths: dict[str, list[str]] = field(default_factory=dict)


def _parse_tsconfig(path: Path) -> RawTsconfig | None:
    """Parse a tsconfig.json file, stripping JSONC comments."""
    try:
        content = path.read_text()
    except (OSError, UnicodeDecodeError):
        return None

    stripped = _strip_jsonc_comments(content)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return None

    compiler_options = data.get("compilerOptions", {})
    if not isinstance(compiler_options, dict):
        return None

    base_url = compiler_options.get("baseUrl")
    if not isinstance(base_url, str):
        base_url = None

    paths_map = compiler_options.get("paths")
    if not isinstance(paths_map, dict):
        paths_map = {}

    paths: dict[str, list[str]] = {}
    for key, value in paths_map.items():
        if isinstance(value, list):
            paths[key] = [
                v for v in value if isinstance(v, str)
            ]

    return RawTsconfig(base_url=base_url, paths=paths)


def _strip_jsonc_comments(text: str) -> str:
    """Strip C-style comments from JSONC (JSON with Comments).

    Preserves strings — comments inside quoted strings are not stripped.
    """
    result: list[str] = []
    chars = list(text)
    i = 0
    in_string = False
    escape = False

    while i < len(chars):
        ch = chars[i]

        if escape:
            escape = False
            result.append(ch)
            i += 1
            continue

        if ch == '\\' and in_string:
            escape = True
            result.append(ch)
            i += 1
            continue

        if ch == '"':
            in_string = not in_string
            result.append(ch)
            i += 1
            continue

        if in_string:
            result.append(ch)
            i += 1
            continue

        if ch == '/' and i + 1 < len(chars):
            next_ch = chars[i + 1]
            if next_ch == '/':
                # Line comment — skip to newline
                while i < len(chars) and chars[i] != '\n':
                    i += 1
                continue
            elif next_ch == '*':
                # Block comment — skip to */
                i += 2
                while i + 1 < len(chars):
                    if chars[i] == '*' and chars[i + 1] == '/':
                        i += 2
                        break
                    i += 1
                else:
                    i = len(chars)
                continue

        result.append(ch)
        i += 1

    return ''.join(result)


def _resolve_alias(mod_name: str, config: TsconfigData) -> str | None:
    """Resolve a module name against tsconfig paths.

    Returns the resolved file path as a string, or None if not found.
    """
    # Try exact match first
    for pattern, targets in config.paths.items():
        if pattern == mod_name:
            if targets:
                target_path = _probe_target(config.base_url, config.tsconfig_dir, targets[0])
                if target_path:
                    return target_path
            continue

        # Try wildcard pattern: @/components/* → @/components/foo
        if '*' in pattern:
            prefix, suffix = pattern.split('*', 1)
            if mod_name.startswith(prefix) and mod_name.endswith(suffix):
                replacement = mod_name[len(prefix):len(mod_name) - len(suffix)]
                if targets:
                    target_pattern = targets[0].replace('*', replacement)
                    target_path = _probe_target(config.base_url, config.tsconfig_dir, target_pattern)
                    if target_path:
                        return target_path

    # If the module name starts with @ and there's no paths entry,
    # try baseUrl as fallback (common in simple setups)
    if config.base_url and mod_name.startswith("@"):
        base_path = config.base_url
        target_path = _probe_with_extensions(f"{base_path}/{mod_name}")
        if target_path:
            return target_path

    return None


def _probe_target(
    base_url: str | None,
    tsconfig_dir: str,
    target_pattern: str,
) -> str | None:
    """Probe a target pattern against tsconfig base URL and extensions."""
    base_path = base_url or tsconfig_dir

    return _probe_with_extensions(f"{base_path}/{target_pattern}")


def _probe_with_extensions(path_str: str) -> str | None:
    """Try a path as-is, then with each extension, then as directory/index."""
    path = Path(path_str)

    # Try the path as-is
    if path.is_file():
        return str(path)

    # Try with each extension
    for ext in PROBE_EXTENSIONS:
        candidate = path.with_suffix(ext)
        if candidate.is_file():
            return str(candidate)

    # Try as directory with index.{ext}
    if path.is_dir():
        for ext in PROBE_EXTENSIONS:
            candidate = path / f"index{ext}"
            if candidate.is_file():
                return str(candidate)

    return None
