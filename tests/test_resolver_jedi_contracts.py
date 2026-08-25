from __future__ import annotations

import json
from pathlib import Path

from graphician.core import Edge, EdgeKind, Graph, Node, NodeKind
from graphician.extraction.jedi import enrich_jedi_calls
from graphician.extraction.jedi.parse import parse_jedi_results
from graphician.extraction.jedi.scan import find_dropped_calls
from graphician.extraction.jedi.script_gen import build_jedi_script
from graphician.extraction.languages.tsconfig_resolver import (
    TsconfigData,
    _find_nearest_tsconfig,
    _parse_tsconfig,
    _resolve_alias,
    _strip_jsonc_comments,
    is_builtin_alias,
    is_ts_alias_path,
    resolve_ts_path_aliases,
)


def test_tsconfig_alias_resolution_supports_jsonc_wildcards_and_index_files(tmp_path: Path) -> None:
    source_dir = tmp_path / "src"
    components = source_dir / "components"
    components.mkdir(parents=True)
    (components / "button.ts").write_text("export const button = true")
    shared = source_dir / "shared"
    shared.mkdir()
    (shared / "index.tsx").write_text("export const shared = true")
    config_path = tmp_path / "tsconfig.json"
    config_path.write_text(
        '{\n// paths\n"compilerOptions": {"baseUrl": ".", "paths": {'
        '"@/*": ["src/*"], "#shared": ["src/shared"]}}}\n'
    )

    raw = _parse_tsconfig(config_path)
    assert raw is not None and raw.base_url == "."
    config = TsconfigData(str(tmp_path), raw.paths, str(tmp_path))
    assert _resolve_alias("@/components/button", config).endswith("button.ts")
    assert _resolve_alias("#shared", config).endswith("index.tsx")
    assert _find_nearest_tsconfig(components, tmp_path) == config_path
    assert 'https://example.test' in _strip_jsonc_comments('{"url":"https://example.test"}// x')


def test_tsconfig_graph_pass_renames_internal_alias_and_skips_packages(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    importer_path = tmp_path / "src" / "app.ts"
    importer_path.write_text("import x from '@/target'")
    target_path = tmp_path / "src" / "target.ts"
    target_path.write_text("export default 1")
    (tmp_path / "tsconfig.json").write_text(
        json.dumps({"compilerOptions": {"baseUrl": ".", "paths": {"@/*": ["src/*"]}}})
    )
    graph = Graph()
    importer = graph.add_node(Node.new(NodeKind.FILE, f"file::{importer_path}").with_source(str(importer_path), 1, 1))
    alias = graph.add_node(Node.new(NodeKind.MODULE, "module::@/target"))
    builtin = graph.add_node(Node.new(NodeKind.MODULE, "module::@angular/core"))
    graph.add_edge(importer, alias, Edge.extracted(EdgeKind.IMPORTS))
    graph.add_edge(importer, builtin, Edge.extracted(EdgeKind.IMPORTS))

    result = resolve_ts_path_aliases(graph, tmp_path)

    assert result["resolved"] == 1
    assert graph.find_by_qname(f"file::{target_path}") is not None
    assert graph.find_by_qname("module::@angular/core") == builtin
    assert is_ts_alias_path("@/target")
    assert is_builtin_alias("@angular/core")
    assert not is_builtin_alias("@internal/core")


def test_tsconfig_parser_rejects_invalid_and_missing_configs(tmp_path: Path) -> None:
    assert _parse_tsconfig(tmp_path / "missing.json") is None
    invalid = tmp_path / "tsconfig.json"
    invalid.write_text("{invalid")
    assert _parse_tsconfig(invalid) is None
    invalid.write_text('{"compilerOptions": []}')
    assert _parse_tsconfig(invalid) is None
    assert _find_nearest_tsconfig(tmp_path / "outside", tmp_path) == invalid


def test_jedi_scanner_finds_only_dropped_receiver_patterns() -> None:
    source = """def outer():
    service.authenticate()
    self.keep()
    Service.keep()
    getattr(service, "refresh")()
    getattr(service, method_name)()
    type(service).close()
    callable(service).invoke()
"""
    calls = find_dropped_calls(source, [("app::outer", 1, 8)])
    names = {call[2] for call in calls}
    assert {"authenticate", "refresh", "getattr_variable(method_name)", "close", "invoke"} <= names
    assert "keep" not in names
    assert all(call[3] == "app::outer" for call in calls)


def test_jedi_result_parser_adds_only_resolved_nonduplicate_calls() -> None:
    graph = Graph()
    source = graph.add_node(Node.new(NodeKind.FUNCTION, "app::source"))
    target = graph.add_node(Node.new(NodeKind.METHOD, "app::Service::target"))
    payload = json.dumps([
        ["app.py", 4, "app::source", "app::Service::target"],
        ["app.py", 5, "app::missing", "app::Service::target"],
        ["malformed"],
    ])
    assert parse_jedi_results(payload, graph, set()) == 1
    assert parse_jedi_results(payload, graph, set()) == 0
    assert parse_jedi_results(payload, graph, {("app::source", 4)}) == 0
    assert parse_jedi_results("not-json", graph, set()) == 0
    assert next(graph.out_neighbors(source))[0] == target


def test_jedi_script_is_valid_python_and_empty_graph_is_noop(tmp_path: Path) -> None:
    script = build_jedi_script(
        [(str(tmp_path / "app.py"), 2, 4, "target", "app::source")], tmp_path
    )
    compile(script, "<jedi-enrichment>", "exec")
    assert "jedi.Project" in script
    assert enrich_jedi_calls(Graph(), tmp_path) == 0


def test_jedi_enrichment_handles_existing_edges_without_crashing(tmp_path: Path) -> None:
    path = tmp_path / "app.py"
    path.write_text("def source():\n    service.target()\n\ndef target():\n    pass\n")
    graph = Graph()
    source = graph.add_node(Node.new(NodeKind.FUNCTION, "app::source").with_source(str(path), 1, 2))
    target = graph.add_node(Node.new(NodeKind.FUNCTION, "app::target").with_source(str(path), 4, 5))
    graph.add_edge(source, target, Edge.extracted(EdgeKind.CALLS))

    # The existing edge should be inspected safely; no enrichment is needed.
    assert enrich_jedi_calls(graph, tmp_path) == 0
