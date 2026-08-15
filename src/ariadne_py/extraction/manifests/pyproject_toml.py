"""pyproject.toml manifest extraction.

Parses pyproject.toml to extract Python package metadata:
- Project name, description, version, authors, license
- Dependencies (requires-dist)
- Entry points (console_scripts, gui_scripts, plugins)
- Build system configuration
- Dynamic metadata fields

Creates PACKAGE nodes and DEPENDS_ON edges in the graph.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:
    # Python < 3.11
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]

from ...core.edge import Edge, EdgeKind
from ...core.graph import Graph
from ...core.node import Node, NodeKind

logger = logging.getLogger(__name__)


def extract_pyproject_toml(path: Path, graph: Graph) -> dict[str, Any]:
    """Extract metadata from a pyproject.toml file.

    Args:
        path: Path to the pyproject.toml file.
        graph: The graph to add nodes and edges to.

    Returns:
        Summary dict with extracted data.
    """
    if tomllib is None:
        logger.warning("tomli/tomllib not available, skipping pyproject.toml extraction")
        return {"error": "tomli/tomllib not available", "extracted": 0}

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Could not read pyproject.toml at %s: %s", path, e)
        return {"error": str(e), "extracted": 0}

    try:
        data = tomllib.loads(content)
    except Exception as e:
        logger.warning("Could not parse pyproject.toml at %s: %s", path, e)
        return {"error": str(e), "extracted": 0}

    file_key = f"file::{path.as_posix()}"
    result: dict[str, Any] = {"extracted": 0}

    # Extract project metadata
    project = data.get("project", {})
    if project:
        result["extracted"] += _extract_project_metadata(project, file_key, graph)

    # Extract dependencies
    if project.get("dependencies"):
        result["extracted"] += _extract_dependencies(project["dependencies"], file_key, graph)

    # Extract dynamic dependencies (e.g. from poetry)
    if project.get("optional-dependencies"):
        result["extracted"] += _extract_optional_dependencies(
            project["optional-dependencies"], file_key, graph
        )

    # Extract entry points
    if project.get("entry-points"):
        result["extracted"] += _extract_entry_points(
            project["entry-points"], file_key, graph
        )
    elif "entry-points" in data:
        result["extracted"] += _extract_entry_points(data["entry-points"], file_key, graph)
    elif "tool" in data and "poetry" in data["tool"]:
        poetry = data["tool"]["poetry"]
        if poetry.get("dependencies"):
            result["extracted"] += _extract_poetry_dependencies(poetry["dependencies"], file_key, graph)
        if poetry.get("plugins"):
            result["extracted"] += _extract_poetry_entry_points(poetry["plugins"], file_key, graph)

    # Extract build system
    build_system = data.get("build-system", {})
    if build_system:
        result["extracted"] += _extract_build_system(build_system, file_key, graph)

    # Extract scripts (console entry points)
    if project.get("scripts"):
        result["extracted"] += _extract_scripts(project["scripts"], file_key, graph)

    result["project_name"] = project.get("name", "unknown")
    result["version"] = project.get("version", "unknown")

    return result


def _extract_project_metadata(
    project: dict[str, Any],
    file_key: str,
    graph: Graph,
) -> int:
    """Extract project metadata nodes and edges. Creates a PACKAGE node."""
    count = 0
    name = project.get("name", "unknown")
    qn = f"package::{name}"

    # Create or find the package node
    existing = graph.find_by_qname(qn)
    if existing is None:
        node = Node.new(NodeKind.PACKAGE, qn).with_property(
            "name", name
        ).with_property(
            "description", project.get("description", "")
        ).with_property(
            "version", project.get("version", "unknown")
        ).with_property(
            "authors", _format_authors(project.get("authors", []))
        ).with_property(
            "license", project.get("license", "")
        ).with_property(
            "urls", project.get("urls", {})
        ).with_property(
            "keywords", project.get("keywords", [])
        )
        existing = graph.add_node(node)
        count += 1

    # Add depends_on edge from the file to the package
    file_id = graph.find_by_qname(file_key)
    if file_id is not None and existing is not None:
        graph.add_edge(file_id, existing, Edge.extracted(EdgeKind.DEPENDS_ON))
        count += 1

    return count


def _format_authors(authors: list[dict[str, str]]) -> str:
    """Format author list as a comma-separated string."""
    if not authors:
        return ""
    names = []
    for author in authors:
        name = author.get("name", "")
        email = author.get("email", "")
        if email:
            names.append(f"{name} <{email}>")
        elif name:
            names.append(name)
    return ", ".join(names)


def _parse_dependency(dep: str) -> tuple[str, str | None]:
    """Parse a PEP 508 dependency string.

    Returns (package_name, extra_spec) or (package_name, None).
    """
    dep = dep.strip()
    if not dep:
        return ("unknown", None)

    # Remove version specifier
    name = dep.split(";")[0].split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip()

    # Extract extras
    extras = None
    if "[" in dep:
        start = dep.index("[")
        end = dep.index("]", start)
        extras = dep[start + 1:end]

    return (name, extras)


def _extract_dependencies(
    dependencies: list[str],
    file_key: str,
    graph: Graph,
) -> int:
    """Extract dependency edges from project.dependencies."""
    count = 0
    for dep_str in dependencies:
        name, extras = _parse_dependency(dep_str)
        pkg_qn = f"package::{name}"

        # Create or find the dependency package node
        pkg_id = graph.find_by_qname(pkg_qn)
        if pkg_id is None:
            pkg_node = Node.new(NodeKind.PACKAGE, pkg_qn).with_property("name", name)
            if extras:
                pkg_node = pkg_node.with_property("extras", extras)
            pkg_id = graph.add_node(pkg_node)
            count += 1

        # Add depends_on edge
        file_id = graph.find_by_qname(file_key)
        if file_id is not None and pkg_id is not None:
            edge = Edge.extracted(EdgeKind.DEPENDS_ON)
            if extras:
                edge.properties["extras"] = extras
            graph.add_edge(file_id, pkg_id, edge)
            count += 1

    return count


def _extract_optional_dependencies(
    optional_deps: dict[str, list[str]],
    file_key: str,
    graph: Graph,
) -> int:
    """Extract optional dependency groups."""
    count = 0
    for group_name, deps in optional_deps.items():
        group_qn = f"package::{group_name}"
        group_id = graph.find_by_qname(group_qn)
        if group_id is None:
            group_node = Node.new(
                NodeKind.PACKAGE, group_qn
            ).with_property("group", group_name)
            group_id = graph.add_node(group_node)
            count += 1

        # Connect each dependency in this group
        for dep_str in deps:
            name, _ = _parse_dependency(dep_str)
            pkg_qn = f"package::{name}"
            pkg_id = graph.find_by_qname(pkg_qn)
            if pkg_id is None:
                pkg_node = Node.new(NodeKind.PACKAGE, pkg_qn).with_property("name", name)
                pkg_id = graph.add_node(pkg_node)
                count += 1

            if group_id is not None and pkg_id is not None:
                graph.add_edge(pkg_id, group_id, Edge.extracted(EdgeKind.DEPENDS_ON))
                count += 1

        # Connect group to the file
        file_id = graph.find_by_qname(file_key)
        if file_id is not None and group_id is not None:
            graph.add_edge(file_id, group_id, Edge.extracted(EdgeKind.DEPENDS_ON))
            count += 1

    return count


def _extract_entry_points(
    entry_points: dict[str, dict[str, str]],
    file_key: str,
    graph: Graph,
) -> int:
    """Extract entry point edges from [entry-points] section."""
    count = 0
    for section, commands in entry_points.items():
        section_qn = f"entry_point::{section}"
        section_id = graph.find_by_qname(section_qn)
        if section_id is None:
            section_node = Node.new(
                NodeKind.PACKAGE, section_qn
            ).with_property("section", section)
            section_id = graph.add_node(section_node)
            count += 1

        for name, target in commands.items():
            ep_qn = f"entry_point::{section}::{name}"
            ep_node = Node.new(
                NodeKind.FUNCTION, ep_qn
            ).with_property("target", target)
            ep_id = graph.add_node(ep_node)
            count += 1

            if section_id is not None:
                graph.add_edge(section_id, ep_id, Edge.extracted(EdgeKind.DEFINES))
                count += 1

    file_id = graph.find_by_qname(file_key)
    if file_id is not None:
        for section in entry_points:
            section_qn = f"entry_point::{section}"
            section_id = graph.find_by_qname(section_qn)
            if section_id is not None:
                graph.add_edge(file_id, section_id, Edge.extracted(EdgeKind.DEPENDS_ON))
                count += 1

    return count


def _extract_scripts(
    scripts: dict[str, str],
    file_key: str,
    graph: Graph,
) -> int:
    """Extract console script entry points from project.scripts."""
    count = 0
    for name, target in scripts.items():
        ep_qn = f"entry_point::console::{name}"
        ep_node = Node.new(
            NodeKind.FUNCTION, ep_qn
        ).with_property("target", target).with_property("type", "console_script")
        ep_id = graph.add_node(ep_node)
        count += 1

        file_id = graph.find_by_qname(file_key)
        if file_id is not None:
            graph.add_edge(file_id, ep_id, Edge.extracted(EdgeKind.DEPENDS_ON))
            count += 1

    return count


def _extract_poetry_dependencies(
    deps: dict[str, Any],
    file_key: str,
    graph: Graph,
) -> int:
    """Extract Poetry-style dependencies."""
    count = 0
    for name, version_info in deps.items():
        if name == "python":
            continue

        version = ""
        extras_list: list[str] = []
        if isinstance(version_info, str):
            version = version_info
        elif isinstance(version_info, dict):
            version = version_info.get("version", "")
            extras_list = version_info.get("extras", [])
        elif isinstance(version_info, bool):
            # "package = true" means any version
            version = "*"

        pkg_qn = f"package::{name}"
        pkg_id = graph.find_by_qname(pkg_qn)
        if pkg_id is None:
            node = Node.new(NodeKind.PACKAGE, pkg_qn).with_property("name", name)
            if version:
                node = node.with_property("version_spec", version)
            for extra in extras_list:
                node = node.with_property(f"extra_{extra}", True)
            pkg_id = graph.add_node(node)
            count += 1

        file_id = graph.find_by_qname(file_key)
        if file_id is not None and pkg_id is not None:
            graph.add_edge(file_id, pkg_id, Edge.extracted(EdgeKind.DEPENDS_ON))
            count += 1

    return count


def _extract_poetry_entry_points(
    plugins: dict[str, dict[str, str]],
    file_key: str,
    graph: Graph,
) -> int:
    """Extract Poetry plugin entry points."""
    count = 0
    for group, commands in plugins.items():
        group_qn = f"entry_point::poetry::{group}"
        group_id = graph.find_by_qname(group_qn)
        if group_id is None:
            group_node = Node.new(
                NodeKind.PACKAGE, group_qn
            ).with_property("group", group)
            group_id = graph.add_node(group_node)
            count += 1

        for name, target in commands.items():
            ep_qn = f"entry_point::poetry::{group}::{name}"
            ep_node = Node.new(
                NodeKind.FUNCTION, ep_qn
            ).with_property("target", target)
            ep_id = graph.add_node(ep_node)
            count += 1

            if group_id is not None:
                graph.add_edge(group_id, ep_id, Edge.extracted(EdgeKind.DEFINES))
                count += 1

    file_id = graph.find_by_qname(file_key)
    if file_id is not None and group_id is not None:
        graph.add_edge(file_id, group_id, Edge.extracted(EdgeKind.DEPENDS_ON))
        count += 1

    return count


def _extract_build_system(
    build_system: dict[str, Any],
    file_key: str,
    graph: Graph,
) -> int:
    """Extract build system configuration."""
    count = 0
    requires = build_system.get("requires", [])
    backend = build_system.get("build-backend", "")

    if requires:
        for req in requires:
            name, _ = _parse_dependency(req)
            pkg_qn = f"package::{name}"
            pkg_id = graph.find_by_qname(pkg_qn)
            if pkg_id is None:
                pkg_node = Node.new(NodeKind.PACKAGE, pkg_qn).with_property("name", name)
                pkg_id = graph.add_node(pkg_node)
                count += 1

            file_id = graph.find_by_qname(file_key)
            if file_id is not None and pkg_id is not None:
                graph.add_edge(file_id, pkg_id, Edge.extracted(EdgeKind.DEPENDS_ON))
                count += 1

    if backend:
        backend_qn = f"build_backend::{backend}"
        backend_id = graph.find_by_qname(backend_qn)
        if backend_id is None:
            backend_node = Node.new(
                NodeKind.PACKAGE, backend_qn
            ).with_property("name", backend).with_property("type", "build_backend")
            backend_id = graph.add_node(backend_node)
            count += 1

        file_id = graph.find_by_qname(file_key)
        if file_id is not None and backend_id is not None:
            graph.add_edge(file_id, backend_id, Edge.extracted(EdgeKind.DEPENDS_ON))
            count += 1

    return count
