from pathlib import Path

from ariadne_py.core import EdgeKind, NodeKind
from ariadne_py.extraction.languages import LanguageRegistry
from ariadne_py.extraction.pipeline import ExtractionPipeline


def test_python_extraction_scopes_methods_and_uses_call_placeholders(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sample.py"
    source.write_text(
        """class A:
    def helper(self):
        return 1

    def run(self):
        self.helper()
        missing_api()
        print('noise')

class B:
    def helper(self):
        return 2
"""
    )

    graph = ExtractionPipeline(LanguageRegistry()).build(tmp_path)

    methods = {
        node.qualified_name: (node_id, node)
        for node_id, node in graph.nodes()
        if node.kind is NodeKind.METHOD
    }
    run_qname = "file::sample.py::sample::A::run"
    a_helper = "file::sample.py::sample::A::helper"
    b_helper = "file::sample.py::sample::B::helper"
    assert {run_qname, a_helper, b_helper} <= methods.keys()
    assert methods[run_qname][1].source_uri == "sample.py"
    assert methods[run_qname][1].line_end == 8

    calls = {
        graph.node(target).qualified_name
        for target, edge in graph.out_neighbors(methods[run_qname][0])
        if edge.kind is EdgeKind.CALLS and graph.node(target) is not None
    }
    assert a_helper in calls
    assert "call::missing_api" in calls
    assert "call::print" not in calls
    assert not any("self.helper" in qname for qname in calls)
