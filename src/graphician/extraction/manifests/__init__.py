"""Manifest parsers: package.json, pyproject.toml, cargo.toml.

Extracts dependency relationships from project manifest files and
emits Package nodes with DependsOn edges.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...core.edge import Edge, EdgeKind
from ...core.graph import Graph
from ...core.id import NodeId
from ...core.node import Node, NodeKind


def extract_manifest(
    path: str | Path,
    graph: Graph,
) -> dict[str, Any]:
    """Auto-detect manifest type and extract dependencies.

    Returns summary of extraction results.
    """
    path = Path(path)
    name = path.name.lower()
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        return {"error": str(error), "packages": 0, "edges": 0}

    manifest_id = graph.add_node(
        Node.new(NodeKind.FILE, f"file::{path}")
        .with_source(str(path), 1, max(1, len(source.splitlines())))
        .with_source_text(source)
        .with_property("role", "manifest")
    )
    packages_before = {
        node_id for node_id, node in graph.nodes() if node.kind == NodeKind.PACKAGE
    }

    if name == "package.json":
        result = _extract_package_json(path, graph)
    elif name in ("pyproject.toml", "setup.py", "setup.cfg"):
        result = _extract_pyproject_toml(path, graph)
    elif name == "cargo.toml":
        result = _extract_cargo_toml(path, graph)
    else:
        return {"error": f"Unsupported manifest: {name}", "packages": 0, "edges": 0}

    for package_id, package_node in graph.nodes():
        if package_node.kind == NodeKind.PACKAGE and package_id not in packages_before:
            graph.add_edge(manifest_id, package_id, Edge.extracted(EdgeKind.DEFINES))
    return result


def _add_package_node(graph: Graph, name: str) -> NodeId:
    """Add or get a Package node for a dependency name."""
    qn = f"package::{name}"
    existing = graph.find_by_qname(qn)
    if existing is not None:
        return existing
    node = Node(kind=NodeKind.PACKAGE, name=name, qualified_name=qn)
    return graph.add_node(node)


def _add_depends_on(
    graph: Graph,
    from_id: NodeId,
    to_id: NodeId,
    version: str | None,
) -> None:
    """Add a DependsOn edge between two package nodes."""
    if not _has_edge_kind(graph, from_id, to_id, EdgeKind.DEPENDS_ON):
        edge = Edge.extracted(EdgeKind.DEPENDS_ON)
        if version:
            edge = edge.with_property("version_req", version)
        graph.add_edge(from_id, to_id, edge)


def _has_edge_kind(graph: Graph, src: NodeId, dst: NodeId, kind: EdgeKind) -> bool:
    for neighbor, edge in graph.out_neighbors(src):
        if neighbor.value == dst.value and edge.kind == kind:
            return True
    return False


def _extract_package_json(
    path: Path,
    graph: Graph,
) -> dict[str, Any]:
    """Extract dependencies from package.json."""
    try:
        content = path.read_text()
    except (OSError, UnicodeDecodeError) as e:
        return {"error": str(e), "packages": 0, "edges": 0}

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {e}", "packages": 0, "edges": 0}

    name = data.get("name")
    if not name:
        return {"packages": 0, "edges": 0}

    own_id = _add_package_node(graph, name)
    dep_fields = ["dependencies", "devDependencies", "peerDependencies"]
    packages = 1
    edges = 0

    for field in dep_fields:
        deps = data.get(field)
        if not isinstance(deps, dict):
            continue
        for dep_name, version in deps.items():
            dep_id = _add_package_node(graph, dep_name)
            version_str = version if isinstance(version, str) else None
            _add_depends_on(graph, own_id, dep_id, version_str)
            edges += 1
            packages += 1

    return {"packages": packages, "edges": edges, "manifest": str(path)}


def _extract_pyproject_toml(
    path: Path,
    graph: Graph,
) -> dict[str, Any]:
    """Extract dependencies from pyproject.toml.

    Supports both PEP 621 ([project] dependencies = [...]) and
    Poetry ([tool.poetry.dependencies]).
    """
    try:
        content = path.read_text()
    except (OSError, UnicodeDecodeError) as e:
        return {"error": str(e), "packages": 0, "edges": 0}

    data = _parse_toml(content)
    if data is None:
        return {"error": "Invalid TOML", "packages": 0, "edges": 0}

    # Get project name
    own_name = None
    project = data.get("project", {})
    tool = data.get("tool", {})
    if isinstance(project, dict):
        own_name = project.get("name")

    if not own_name and isinstance(tool, dict):
        poetry = tool.get("poetry", {})
        if isinstance(poetry, dict):
            own_name = poetry.get("name")

    if not own_name:
        return {"packages": 0, "edges": 0}

    own_id = _add_package_node(graph, own_name)
    packages = 1
    edges = 0

    # PEP 621: project.dependencies
    deps_list = []
    if isinstance(project, dict):
        raw_deps = project.get("dependencies")
        if isinstance(raw_deps, list):
            deps_list.extend(raw_deps)

    for dep in deps_list:
        if isinstance(dep, str):
            dep_name, version = _split_pep508(dep)
            dep_id = _add_package_node(graph, dep_name)
            _add_depends_on(graph, own_id, dep_id, version)
            edges += 1
            packages += 1

    # Poetry: tool.poetry.dependencies
    if isinstance(tool, dict):
        poetry = tool.get("poetry", {})
        if isinstance(poetry, dict):
            poetry_deps = poetry.get("dependencies", {})
            if isinstance(poetry_deps, dict):
                for dep_name, dep_value in poetry_deps.items():
                    if dep_name == "python":
                        continue
                    version = None
                    if isinstance(dep_value, str):
                        version = dep_value
                    elif isinstance(dep_value, dict):
                        version = dep_value.get("version")
                        if isinstance(version, (int, float)):
                            version = str(version)
                    if version is None:
                        version = None
                    dep_id = _add_package_node(graph, dep_name)
                    _add_depends_on(graph, own_id, dep_id, version)
                    edges += 1
                    packages += 1

    return {"packages": packages, "edges": edges, "manifest": str(path)}


def _extract_cargo_toml(
    path: Path,
    graph: Graph,
) -> dict[str, Any]:
    """Extract dependencies from Cargo.toml.

    Parses [package] for name, and [dependencies], [dev-dependencies],
    [build-dependencies] for dependency edges.
    """
    try:
        content = path.read_text()
    except (OSError, UnicodeDecodeError) as e:
        return {"error": str(e), "packages": 0, "edges": 0}

    data = _parse_toml(content)
    if data is None:
        return {"error": "Invalid TOML", "packages": 0, "edges": 0}

    # Get package name
    package = data.get("package", {})
    if not isinstance(package, dict):
        return {"packages": 0, "edges": 0}

    own_name = package.get("name")
    if not own_name:
        return {"packages": 0, "edges": 0}

    own_id = _add_package_node(graph, own_name)
    packages = 1
    edges = 0

    dep_sections = ["dependencies", "dev-dependencies", "build-dependencies"]
    for section in dep_sections:
        deps = data.get(section)
        if not isinstance(deps, dict):
            continue
        for dep_name, dep_value in deps.items():
            version = None
            if isinstance(dep_value, str):
                version = dep_value
            elif isinstance(dep_value, dict):
                version = dep_value.get("version")
                if isinstance(version, (int, float)):
                    version = str(version)

            dep_id = _add_package_node(graph, dep_name)
            _add_depends_on(graph, own_id, dep_id, version)
            edges += 1
            packages += 1

    return {"packages": packages, "edges": edges, "manifest": str(path)}


def _parse_toml(content: str) -> dict[str, Any] | None:
    """Parse TOML content. Uses stdlib tomllib on Python 3.11+, otherwise toml package."""
    try:
        import tomllib
    except ImportError:
        try:
            import toml as tomllib
        except ImportError:
            return None

    try:
        return tomllib.loads(content)
    except Exception:  # noqa: BLE001 -- malformed manifest TOML must not crash extraction
        return None


def _split_pep508(req: str) -> tuple[str, str | None]:
    """Split a PEP 508 requirement string into (name, version_specifier).

    E.g. "requests>=2.28" -> ("requests", ">=2.28")
    "click" -> ("click", None)
    "foo[bar]>=1.0" -> ("foo", ">=1.0")
    """
    req = req.split(";")[0].strip()
    split_at = len(req)
    for i, c in enumerate(req):
        if c in "><=!~":
            split_at = i
            break
        if c == "[":
            # Skip extras
            j = req.find("]", i)
            if j != -1:
                req = req[:i] + req[j + 1:]
                split_at = len(req)
                for k, c2 in enumerate(req):
                    if c2 in "><=!~":
                        split_at = k
                        break
            break

    name = req[:split_at].strip()
    version = req[split_at:].strip() if split_at < len(req) else None
    if not version:
        version = None
    return (name, version if version else None)
