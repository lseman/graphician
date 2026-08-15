from __future__ import annotations

import json
from pathlib import Path

import pytest

from ariadne_py.core import EdgeKind, Graph, Node, NodeKind
from ariadne_py.extraction.languages.parsers import cpp, java, javascript, python, typescript
from ariadne_py.extraction.manifests import cargo_toml, package_json, pyproject_toml


def _nodes(graph: Graph, kind: NodeKind | None = None):
    return [node for _, node in graph.nodes() if kind is None or node.kind is kind]


def _edge_kinds(graph: Graph) -> set[EdgeKind]:
    return {edge.kind for _, _, _, edge in graph.edges()}


@pytest.mark.parametrize(
    ("filename", "source", "extractor", "expected_names", "expected_kinds"),
    [
        (
            "service.py",
            """
import os
from tools import helper

@decorator
class Service(Base):
    def run(self, value):
        helper(value)
        self.finish()

    def finish(self):
        return value

def test_entry():
    Service()
""",
            python.extract_file,
            {"Service", "run", "finish", "test_entry"},
            {EdgeKind.DEFINES, EdgeKind.CALLS, EdgeKind.IMPORTS, EdgeKind.INHERITS},
        ),
        (
            "service.js",
            """
import { helper } from './tools.js';
export class Service extends Base {
  run(value) { helper(value); this.finish(); }
  finish() { return value; }
}
export const handler = (event) => helper(event);
""",
            javascript.extract_file,
            {"Service", "run", "finish", "handler"},
            {EdgeKind.DEFINES, EdgeKind.CALLS, EdgeKind.IMPORTS, EdgeKind.INHERITS},
        ),
        (
            "service.ts",
            """
import { helper } from './tools';
interface Runner { run(value: string): void; }
type Identifier = string | number;
enum State { Ready, Done }
@sealed
class Service extends Base implements Runner {
  run(value: string): void { helper(value); }
}
const handler = (event: Identifier) => helper(event);
""",
            typescript.extract_file,
            {"Runner", "Identifier", "State", "Service", "run", "handler"},
            {EdgeKind.DEFINES, EdgeKind.CALLS, EdgeKind.IMPORTS},
        ),
        (
            "Service.java",
            """
package demo;
import demo.tools.Helper;
@Controller
public class Service extends Base implements Runner {
  @Override public void run(String value) { Helper.process(value); finish(); }
  private void finish() {}
}
interface Runner { void run(String value); }
""",
            java.extract_file,
            {"Service", "Runner", "run", "finish"},
            {EdgeKind.DEFINES, EdgeKind.CALLS, EdgeKind.IMPORTS},
        ),
        (
            "service.cpp",
            """
#include <vector>
namespace demo {
struct Runner { virtual void run() = 0; };
class Service : public Runner {
public:
  void run() { helper(); }
};
int entry() { Service service; service.run(); return 0; }
}
""",
            cpp.extract_file,
            {"demo", "Runner", "Service", "run", "entry"},
            {EdgeKind.DEFINES, EdgeKind.CALLS, EdgeKind.IMPORTS},
        ),
    ],
)
def test_language_parser_contracts(
    tmp_path: Path,
    filename: str,
    source: str,
    extractor,
    expected_names: set[str],
    expected_kinds: set[EdgeKind],
) -> None:
    path = tmp_path / filename
    path.write_text(source)
    graph = Graph()

    extractor(path, graph)

    names = {node.name for node in _nodes(graph)}
    assert expected_names <= names
    assert expected_kinds <= _edge_kinds(graph)
    assert all(node.source_uri is not None for node in _nodes(graph, NodeKind.FILE))


def test_package_json_extracts_full_manifest_contract(tmp_path: Path) -> None:
    path = tmp_path / "package.json"
    path.write_text(
        json.dumps(
            {
                "name": "demo",
                "version": "1.2.3",
                "description": "demo package",
                "author": {"name": "Ariadne"},
                "license": "MIT",
                "dependencies": {"httpx": "^1.0"},
                "devDependencies": {"pytest": "^8"},
                "peerDependencies": {"react": ">=18"},
                "optionalDependencies": {"fsevents": "^2"},
                "scripts": {"test": "pytest", "build": "python -m build"},
                "engines": {"node": ">=20"},
                "config": {"port": 8080},
                "workspaces": {"packages": ["packages/*"]},
            }
        )
    )

    graph = package_json.extract_file(path)

    assert graph.find_by_qname("package::demo") is not None
    assert {NodeKind.PACKAGE, NodeKind.FUNCTION, NodeKind.VARIABLE, NodeKind.MODULE} <= {
        node.kind for node in _nodes(graph)
    }
    assert EdgeKind.DEPENDS_ON in _edge_kinds(graph)


def test_cargo_manifest_extracts_features_workspace_targets_and_dependencies(
    tmp_path: Path,
) -> None:
    path = tmp_path / "Cargo.toml"
    path.write_text(
        """
[package]
name = "demo"
version = "0.1.0"
edition = "2024"

[dependencies]
serde = { version = "1", features = ["derive"], optional = true }
local = { path = "../local" }

[dev-dependencies]
tempfile = "3"

[features]
default = ["serde"]
full = ["serde/derive"]

[workspace]
members = ["crates/*"]

[target.'cfg(unix)'.dependencies]
nix = "0.30"
"""
    )

    graph = cargo_toml.extract_file(path)

    assert graph.find_by_qname("package::demo") is not None
    assert {NodeKind.PACKAGE, NodeKind.VARIABLE, NodeKind.MODULE} <= {
        node.kind for node in _nodes(graph)
    }
    assert EdgeKind.DEPENDS_ON in _edge_kinds(graph)


def test_pyproject_extracts_pep621_poetry_scripts_groups_and_build_system(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text(
        """
[project]
name = "demo"
version = "0.2.0"
description = "demo project"
authors = [{name = "Ariadne", email = "dev@example.com"}]
dependencies = ["httpx>=0.27", "rich; python_version >= '3.11'"]

[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.8"]

[project.scripts]
demo = "demo.cli:main"

[project.entry-points."demo.plugins"]
sample = "demo.plugin:Plugin"

[tool.poetry.dependencies]
python = ">=3.11"
anyio = "^4"

[tool.poetry.scripts]
poetry-demo = "demo.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""
    )
    graph = Graph()

    graph.add_node(Node.new(NodeKind.FILE, f"file::{path.as_posix()}"))
    metadata = pyproject_toml.extract_pyproject_toml(path, graph)

    assert metadata["project_name"] == "demo"
    assert graph.find_by_qname("package::demo") is not None
    assert {NodeKind.PACKAGE, NodeKind.FUNCTION} <= {node.kind for node in _nodes(graph)}
    assert EdgeKind.DEPENDS_ON in _edge_kinds(graph)


@pytest.mark.parametrize("extractor, filename", [(package_json.extract_file, "package.json"), (cargo_toml.extract_file, "Cargo.toml")])
def test_manifest_extractors_handle_missing_and_invalid_files(
    tmp_path: Path, extractor, filename: str
) -> None:
    assert extractor(tmp_path / filename).node_count() == 0
    path = tmp_path / filename
    path.write_text("not valid manifest data")
    assert extractor(path).node_count() == 0
