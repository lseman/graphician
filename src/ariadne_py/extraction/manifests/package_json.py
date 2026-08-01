"""Extract Ariadne graph nodes from package.json manifest files.

Parses package.json files to extract:
- Package metadata (name, version, description, author)
- Dependencies (dependencies, devDependencies, peerDependencies, optionalDependencies)
- Scripts
- Engines
- Config fields
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ...core.edge import Edge, EdgeKind
from ...core.graph import Graph
from ...core.id import NodeId
from ...core.node import Node, NodeKind

logger = logging.getLogger(__name__)

# Node kinds for manifest elements
_KIND_PACKAGE = "package"
_KIND_DEPENDENCY = "dependency"
_KIND_SCRIPT = "script"
_KIND_ENGINE = "engine"
_KIND_CONFIG = "config"


def extract_file(path: str | Path) -> Graph:
    """Extract nodes and edges from a package.json file.

    Args:
        path: Path to the package.json file.

    Returns:
        Graph containing nodes for the package, dependencies, scripts,
        and configuration.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("package.json not found: %s", path)
        return Graph()

    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        logger.error("Failed to parse %s: %s", path, e)
        return Graph()

    graph = Graph()
    source_uri = str(path)

    # Extract package metadata
    name = data.get("name", "")
    version = data.get("version", "")
    description = data.get("description", "")
    author = data.get("author", "")
    license_name = data.get("license", "")
    homepage = data.get("homepage", "")
    repository = data.get("repository", "")
    keywords = data.get("keywords", [])
    bugs = data.get("bugs", "")

    # Package node
    pkg_qn = f"package::{name}" if name else "package::unknown"
    pkg_node = Node(
        kind=NodeKind.PACKAGE,
        name=name or "unknown",
        qualified_name=pkg_qn,
        source_uri=source_uri,
        properties={
            "version": version,
            "description": description,
            "author": author,
            "license": license_name,
            "homepage": homepage,
            "repository": repository,
            "keywords": keywords,
            "bugs": bugs,
        },
    )
    graph.add_node(pkg_node)

    # Extract dependencies
    _extract_dependencies(graph, data.get("dependencies", {}), source_uri, "dependency")
    _extract_dependencies(graph, data.get("devDependencies", {}), source_uri, "dev-dependency")
    _extract_dependencies(graph, data.get("peerDependencies", {}), source_uri, "peer-dependency")
    _extract_dependencies(graph, data.get("optionalDependencies", {}), source_uri, "optional-dependency")

    # Extract scripts
    scripts = data.get("scripts", {})
    for script_name, script_cmd in scripts.items():
        _extract_script(graph, script_name, script_cmd, source_uri, name)

    # Extract engines
    engines = data.get("engines", {})
    for engine_name, engine_version in engines.items():
        _extract_engine(graph, engine_name, engine_version, source_uri, name)

    # Extract config
    config = data.get("config", {})
    for config_key, config_value in config.items():
        _extract_config(graph, config_key, config_value, source_uri, name)

    # Extract workspaces
    workspaces = data.get("workspaces", [])
    if isinstance(workspaces, list):
        for workspace in workspaces:
            _extract_workspace(graph, workspace, source_uri, name)
    elif isinstance(workspaces, dict):
        patterns = workspaces.get("packages", [])
        for pattern in patterns:
            _extract_workspace(graph, pattern, source_uri, name)

    logger.info(
        "Extracted %d nodes, %d edges from %s",
        graph.node_count(),
        graph.edge_count(),
        path,
    )
    return graph


def _extract_dependencies(
    graph: Graph,
    deps: dict[str, str],
    source_uri: str,
    dep_type: str,
) -> None:
    """Extract dependency nodes and edges."""
    pkg_name = Path(source_uri).stem
    for dep_name, dep_version in deps.items():
        dep_qn = f"package::{dep_name}"

        dep_node = Node(
            kind=NodeKind.DEPENDENCY,
            name=dep_name,
            qualified_name=dep_qn,
            source_uri=source_uri,
            properties={
                "type": dep_type,
                "version_range": dep_version,
            },
        )
        graph.add_node(dep_node)

        # Link to package
        pkg_qn = f"package::{pkg_name}"
        pkg_id = graph.find_by_qname(pkg_qn)
        dep_id = graph.find_by_qname(dep_qn)
        if pkg_id and dep_id:
            graph.add_edge(pkg_id, dep_id, Edge.extracted(EdgeKind.DEPENDS_ON))


def _extract_script(
    graph: Graph,
    script_name: str,
    script_cmd: str,
    source_uri: str,
    pkg_name: str,
) -> None:
    """Extract a script node."""
    script_qn = f"package::{pkg_name}::script::{script_name}"

    script_node = Node(
        kind=NodeKind.SCRIPT,
        name=script_name,
        qualified_name=script_qn,
        source_uri=source_uri,
        properties={"command": script_cmd},
    )
    graph.add_node(script_node)

    # Link to package
    pkg_qn = f"package::{pkg_name}"
    pkg_id = graph.find_by_qname(pkg_qn)
    script_id = graph.find_by_qname(script_qn)
    if pkg_id and script_id:
        graph.add_edge(pkg_id, script_id, Edge.extracted(EdgeKind.PROPERTY))


def _extract_engine(
    graph: Graph,
    engine_name: str,
    engine_version: str,
    source_uri: str,
    pkg_name: str,
) -> None:
    """Extract an engine requirement node."""
    engine_qn = f"package::{pkg_name}::engine::{engine_name}"

    engine_node = Node(
        kind=NodeKind.ENGINE,
        name=engine_name,
        qualified_name=engine_qn,
        source_uri=source_uri,
        properties={"version_range": engine_version},
    )
    graph.add_node(engine_node)

    # Link to package
    pkg_qn = f"package::{pkg_name}"
    pkg_id = graph.find_by_qname(pkg_qn)
    engine_id = graph.find_by_qname(engine_qn)
    if pkg_id and engine_id:
        graph.add_edge(pkg_id, engine_id, Edge.extracted(EdgeKind.PROPERTY))


def _extract_config(
    graph: Graph,
    config_key: str,
    config_value: Any,
    source_uri: str,
    pkg_name: str,
) -> None:
    """Extract a config field node."""
    config_qn = f"package::{pkg_name}::config::{config_key}"

    config_node = Node(
        kind=NodeKind.CONFIG,
        name=config_key,
        qualified_name=config_qn,
        source_uri=source_uri,
        properties={"value": str(config_value)},
    )
    graph.add_node(config_node)

    # Link to package
    pkg_qn = f"package::{pkg_name}"
    pkg_id = graph.find_by_qname(pkg_qn)
    config_id = graph.find_by_qname(config_qn)
    if pkg_id and config_id:
        graph.add_edge(pkg_id, config_id, Edge.extracted(EdgeKind.PROPERTY))


def _extract_workspace(
    graph: Graph,
    workspace: str,
    source_uri: str,
    pkg_name: str,
) -> None:
    """Extract a workspace member."""
    workspace_qn = f"package::{workspace}"

    workspace_node = Node(
        kind=NodeKind.MODULE,
        name=workspace,
        qualified_name=workspace_qn,
        source_uri=source_uri,
    )
    graph.add_node(workspace_node)

    # Link to package
    pkg_qn = f"package::{pkg_name}"
    pkg_id = graph.find_by_qname(pkg_qn)
    workspace_id = graph.find_by_qname(workspace_qn)
    if pkg_id and workspace_id:
        graph.add_edge(pkg_id, workspace_id, Edge.extracted(EdgeKind.CONTAINS))
