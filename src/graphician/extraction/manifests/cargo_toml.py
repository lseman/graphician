"""Extract code graph nodes from Cargo.toml manifest files.

Parses Cargo.toml files to extract:
- Package metadata (name, version, description, authors, license)
- Dependencies (dependencies, dev-dependencies, build-dependencies)
- Features
- Workspace members and resolver settings
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[import-not-found]

from ...core.edge import Edge, EdgeKind
from ...core.graph import Graph
from ...core.id import NodeId
from ...core.node import Node, NodeKind

logger = logging.getLogger(__name__)

# Node kinds for manifest elements
_KIND_PACKAGE = "package"
_KIND_DEPENDENCY = "dependency"
_KIND_FEATURE = "feature"
_KIND_WORKSPACE = "workspace"
_KIND_TARGET = "target"


def extract_file(path: str | Path) -> Graph:
    """Extract nodes and edges from a Cargo.toml file.

    Args:
        path: Path to the Cargo.toml file.

    Returns:
        Graph containing nodes for the package, dependencies, features,
        and workspace configuration.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("Cargo.toml not found: %s", path)
        return Graph()

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        logger.error("Failed to parse %s: %s", path, e)
        return Graph()

    graph = Graph()
    source_uri = str(path)

    # Extract package metadata
    pkg = data.get("package", {})
    if pkg:
        _extract_package(graph, pkg, source_uri)

    # Extract dependencies
    deps = data.get("dependencies", {})
    for dep_name, dep_spec in deps.items():
        _extract_dependency(graph, dep_name, dep_spec, source_uri, "dependency")

    # Extract dev-dependencies
    dev_deps = data.get("dev-dependencies", {})
    for dep_name, dep_spec in dev_deps.items():
        _extract_dependency(graph, dep_name, dep_spec, source_uri, "dev-dependency")

    # Extract build-dependencies
    build_deps = data.get("build-dependencies", {})
    for dep_name, dep_spec in build_deps.items():
        _extract_dependency(graph, dep_name, dep_spec, source_uri, "build-dependency")

    # Extract optional-dependencies
    opt_deps = data.get("optional-dependencies", {})
    for group_name, group_deps in opt_deps.items():
        for dep_name, dep_spec in group_deps.items():
            _extract_dependency(
                graph, dep_name, dep_spec, source_uri, "optional-dependency"
            )

    # Extract features
    features = data.get("features", {})
    for feature_name, feature_deps in features.items():
        _extract_feature(graph, feature_name, feature_deps, source_uri)

    # Extract workspace
    workspace = data.get("workspace", {})
    if workspace:
        _extract_workspace(graph, workspace, source_uri)

    # Extract target-specific dependencies
    targets = data.get("target", {})
    for target_triple, target_data in targets.items():
        _extract_target(graph, target_triple, target_data, source_uri)

    logger.info(
        "Extracted %d nodes, %d edges from %s",
        graph.node_count(),
        graph.edge_count(),
        path,
    )
    return graph


def _extract_package(graph: Graph, pkg: dict[str, Any], source_uri: str) -> None:
    """Extract package metadata as nodes."""
    name = pkg.get("name", "")
    version = pkg.get("version", "")
    description = pkg.get("description", "")
    authors = pkg.get("authors", [])
    license_name = pkg.get("license", "")
    rust_version = pkg.get("rust-version", "")

    # Package node
    pkg_node = Node(
        kind=NodeKind.PACKAGE,
        name=name,
        qualified_name=f"package::{name}",
        source_uri=source_uri,
        properties={
            "version": version,
            "description": description,
            "authors": authors,
            "license": license_name,
            "rust_version": rust_version,
        },
    )
    graph.add_node(pkg_node)

    # Version node
    if version:
        ver_node = Node(
            kind=NodeKind.VARIABLE,
            name="version",
            qualified_name=f"package::{name}::version",
            source_uri=source_uri,
            properties={"value": version},
        )
        graph.add_node(ver_node)
        pkg_id = graph.find_by_qname(f"package::{name}")
        ver_id = graph.find_by_qname(f"package::{name}::version")
        if pkg_id and ver_id:
            graph.add_edge(pkg_id, ver_id, Edge.extracted(EdgeKind.DEFINES))

    # License node
    if license_name:
        lic_node = Node(
            kind=NodeKind.CONCEPT,
            name="license",
            qualified_name=f"package::{name}::license",
            source_uri=source_uri,
            properties={"value": license_name},
        )
        graph.add_node(lic_node)
        pkg_id = graph.find_by_qname(f"package::{name}")
        lic_id = graph.find_by_qname(f"package::{name}::license")
        if pkg_id and lic_id:
            graph.add_edge(pkg_id, lic_id, Edge.extracted(EdgeKind.DEFINES))

    # Author nodes
    for author in authors:
        author_node = Node(
            kind=NodeKind.AUTHOR,
            name=author,
            qualified_name=f"package::{name}::author::{author}",
            source_uri=source_uri,
        )
        graph.add_node(author_node)
        pkg_id = graph.find_by_qname(f"package::{name}")
        author_id = graph.find_by_qname(f"package::{name}::author::{author}")
        if pkg_id and author_id:
            graph.add_edge(pkg_id, author_id, Edge.extracted(EdgeKind.DEFINES))


def _extract_dependency(
    graph: Graph,
    dep_name: str,
    dep_spec: Any,
    source_uri: str,
    dep_type: str,
) -> None:
    """Extract a single dependency."""
    # Dependency node
    dep_qn = f"package::{dep_name}"
    dep_node = Node(
        kind=NodeKind.PACKAGE,
        name=dep_name,
        qualified_name=dep_qn,
        source_uri=source_uri,
        properties={"type": dep_type},
    )
    graph.add_node(dep_node)

    # If dep_spec is a dict, extract version and other metadata
    if isinstance(dep_spec, dict):
        version = dep_spec.get("version", "")
        optional = dep_spec.get("optional", False)
        default_features = dep_spec.get("default-features", True)
        features = dep_spec.get("features", [])
        package = dep_spec.get("package", dep_name)
        target = dep_spec.get("target")

        props: dict[str, Any] = {
            "type": dep_type,
            "optional": optional,
            "default_features": default_features,
        }
        if version:
            props["version"] = version
        if features:
            props["features"] = features
        if package != dep_name:
            props["package"] = package

        dep_node = Node(
            kind=NodeKind.PACKAGE,
            name=dep_name,
            qualified_name=dep_qn,
            source_uri=source_uri,
            properties=props,
        )
        graph.add_node(dep_node)

        # Version node
        if version:
            ver_node = Node(
                kind=NodeKind.VARIABLE,
                name="version",
                qualified_name=f"{dep_qn}::version",
                source_uri=source_uri,
                properties={"value": version},
            )
            graph.add_node(ver_node)
            dep_id = graph.find_by_qname(dep_qn)
            ver_id = graph.find_by_qname(f"{dep_qn}::version")
            if dep_id and ver_id:
                graph.add_edge(dep_id, ver_id, Edge.extracted(EdgeKind.DEFINES))

    # If dep_spec is a string, treat as version shorthand
    elif isinstance(dep_spec, str):
        dep_node = Node(
            kind=NodeKind.PACKAGE,
            name=dep_name,
            qualified_name=dep_qn,
            source_uri=source_uri,
            properties={"type": dep_type, "version": dep_spec},
        )
        graph.add_node(dep_node)

    # Link to package if this is a dependency of a package
    # (the source_uri points to the Cargo.toml, so we link to the package node)
    pkg_qn = f"package::{Path(source_uri).stem}"
    pkg_id = graph.find_by_qname(pkg_qn)
    dep_id = graph.find_by_qname(dep_qn)
    if pkg_id and dep_id:
        graph.add_edge(pkg_id, dep_id, Edge.extracted(EdgeKind.DEPENDS_ON))


def _extract_feature(
    graph: Graph, feature_name: str, feature_deps: list[str], source_uri: str
) -> None:
    """Extract a feature node and its dependencies."""
    pkg_name = Path(source_uri).stem
    feature_qn = f"package::{pkg_name}::feature::{feature_name}"

    feature_node = Node(
        kind=NodeKind.VARIABLE,
        name=feature_name,
        qualified_name=feature_qn,
        source_uri=source_uri,
        properties={"deps": feature_deps},
    )
    graph.add_node(feature_node)

    # Link to package
    pkg_qn = f"package::{pkg_name}"
    pkg_id = graph.find_by_qname(pkg_qn)
    feat_id = graph.find_by_qname(feature_qn)
    if pkg_id and feat_id:
        graph.add_edge(pkg_id, feat_id, Edge.extracted(EdgeKind.DEFINES))

    # Link feature to its dependency features
    for dep in feature_deps:
        # Features can reference other features on the same or other packages
        dep_qn = f"package::{dep}" if "::" not in dep else dep
        dep_id = graph.find_by_qname(dep_qn)
        if dep_id:
            graph.add_edge(feat_id, dep_id, Edge.extracted(EdgeKind.DEPENDS_ON))


def _extract_workspace(graph: Graph, workspace: dict[str, Any], source_uri: str) -> None:
    """Extract workspace configuration."""
    ws_qn = f"workspace::{Path(source_uri).parent.name}"
    ws_node = Node(
        kind=NodeKind.MODULE,
        name=Path(source_uri).parent.name,
        qualified_name=ws_qn,
        source_uri=source_uri,
    )
    graph.add_node(ws_node)

    # Workspace members
    members = workspace.get("members", [])
    for member in members:
        member_qn = f"workspace::{member}"
        member_node = Node(
            kind=NodeKind.MODULE,
            name=member,
            qualified_name=member_qn,
            source_uri=source_uri,
        )
        graph.add_node(member_node)
        ws_id = graph.find_by_qname(ws_qn)
        member_id = graph.find_by_qname(member_qn)
        if ws_id and member_id:
            graph.add_edge(ws_id, member_id, Edge.extracted(EdgeKind.DEFINES))

    # Workspace resolver
    resolver = workspace.get("resolver")
    if resolver:
        resolver_node = Node(
            kind=NodeKind.VARIABLE,
            name="resolver",
            qualified_name=f"{ws_qn}::resolver",
            source_uri=source_uri,
            properties={"value": resolver},
        )
        graph.add_node(resolver_node)
        ws_id = graph.find_by_qname(ws_qn)
        res_id = graph.find_by_qname(f"{ws_qn}::resolver")
        if ws_id and res_id:
            graph.add_edge(ws_id, res_id, Edge.extracted(EdgeKind.DEFINES))

    # Workspace dependencies
    workspace_deps = workspace.get("dependencies", {})
    for dep_name, dep_spec in workspace_deps.items():
        dep_qn = f"workspace::dep::{dep_name}"
        dep_node = Node(
            kind=NodeKind.PACKAGE,
            name=dep_name,
            qualified_name=dep_qn,
            source_uri=source_uri,
            properties={"workspace": True},
        )
        graph.add_node(dep_node)
        ws_id = graph.find_by_qname(ws_qn)
        dep_id = graph.find_by_qname(dep_qn)
        if ws_id and dep_id:
            graph.add_edge(ws_id, dep_id, Edge.extracted(EdgeKind.DEFINES))


def _extract_target(graph: Graph, target_triple: str, target_data: dict, source_uri: str) -> None:
    """Extract target-specific configuration."""
    target_qn = f"target::{target_triple}"
    target_node = Node(
        kind=NodeKind.VARIABLE,
        name=target_triple,
        qualified_name=target_qn,
        source_uri=source_uri,
    )
    graph.add_node(target_node)

    # Target dependencies
    deps = target_data.get("dependencies", {})
    for dep_name, dep_spec in deps.items():
        dep_qn = f"target::{target_triple}::dep::{dep_name}"
        dep_node = Node(
            kind=NodeKind.PACKAGE,
            name=dep_name,
            qualified_name=dep_qn,
            source_uri=source_uri,
            properties={"target": target_triple},
        )
        graph.add_node(dep_node)
        target_id = graph.find_by_qname(target_qn)
        dep_id = graph.find_by_qname(dep_qn)
        if target_id and dep_id:
            graph.add_edge(target_id, dep_id, Edge.extracted(EdgeKind.DEPENDS_ON))
