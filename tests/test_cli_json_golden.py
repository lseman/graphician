"""Golden tests: CLI/tool JSON response shape must be stable across source languages.

Builds a real graph per language from the extractor contracts fixtures, then asserts
that ``tool_response`` returns the same JSON schema (keys, types, structure) for
equivalent queries regardless of which language produced the underlying graph.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from graphician.core import Graph
from graphician.extraction.languages.parsers import cpp, java, javascript, python, typescript
from graphician.interfaces.cli.response import tool_response
from graphician.persistence.store import GraphStore

# (language, filename, source, entry_qualified_name_suffix)
_LANGUAGE_FIXTURES = [
    (
        "python",
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
        "run",
    ),
    (
        "javascript",
        "service.js",
        """
import { helper } from './tools.js';
export class Service extends Base {
  run(value) { helper(value); this.finish(); }
  finish() { return value; }
}
export const handler = (event) => helper(event);
""",
        "run",
    ),
    (
        "typescript",
        "service.ts",
        """
import { helper } from './tools';
interface Runner { run(value: string): void; }
class Service extends Base implements Runner {
  run(value: string): void { helper(value); }
}
const handler = (event: string) => helper(event);
""",
        "run",
    ),
    (
        "java",
        "Service.java",
        """
package demo;
import demo.tools.Helper;
public class Service extends Base implements Runner {
  public void run(String value) { Helper.process(value); finish(); }
  private void finish() {}
}
interface Runner { void run(String value); }
""",
        "run",
    ),
    (
        "cpp",
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
        "run",
    ),
]

_EXTRACTORS = {
    "python": python.extract_file,
    "javascript": javascript.extract_file,
    "typescript": typescript.extract_file,
    "java": java.extract_file,
    "cpp": cpp.extract_file,
}


def _build_db(tmp_path: Path, language: str, filename: str, source: str) -> str:
    path = tmp_path / filename
    path.write_text(source)
    graph = Graph()
    _EXTRACTORS[language](path, graph)

    db_path = tmp_path / "graph.db"
    with GraphStore(db_path) as store:
        store.save_graph(graph)
    return str(db_path)


def _find_qname(db_path: str, suffix: str) -> str:
    with GraphStore(db_path) as store:
        graph = store.load_graph()
    matches = [
        node.qualified_name
        for _, node in graph.nodes()
        if node.qualified_name.endswith(suffix) or node.qualified_name.endswith(f"::{suffix}")
    ]
    assert matches, f"no node found with suffix {suffix!r}"
    return matches[0]


@pytest.mark.parametrize(
    ("language", "filename", "source", "entry_suffix"),
    _LANGUAGE_FIXTURES,
    ids=[row[0] for row in _LANGUAGE_FIXTURES],
)
def test_impact_response_schema_is_stable_across_languages(
    tmp_path: Path, language: str, filename: str, source: str, entry_suffix: str
) -> None:
    db_path = _build_db(tmp_path, language, filename, source)
    target = _find_qname(db_path, entry_suffix)

    response = tool_response(db_path, "impact", {"target": target})

    assert "error" not in response
    assert set(response) >= {"operation", "target", "impacted", "total"}
    assert response["operation"] == "impact"
    assert response["target"] == target
    assert isinstance(response["impacted"], list)
    assert response["total"] == len(response["impacted"])
    for hit in response["impacted"]:
        assert set(hit) >= {"qualified_name", "kind", "score", "distance", "source_uri"}
        assert isinstance(hit["score"], (int, float))
        assert isinstance(hit["distance"], int)


@pytest.mark.parametrize(
    ("language", "filename", "source", "entry_suffix"),
    _LANGUAGE_FIXTURES,
    ids=[row[0] for row in _LANGUAGE_FIXTURES],
)
def test_search_response_schema_is_stable_across_languages(
    tmp_path: Path, language: str, filename: str, source: str, entry_suffix: str
) -> None:
    db_path = _build_db(tmp_path, language, filename, source)

    response = tool_response(db_path, "search", {"query": "run"})

    assert "error" not in response
    assert response["operation"] == "search"
    assert "hits" in response
    assert isinstance(response["hits"], list)
    for hit in response["hits"]:
        assert "qualified_name" in hit
        assert "kind" in hit
        assert "score" in hit
