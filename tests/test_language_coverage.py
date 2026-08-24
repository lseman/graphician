from __future__ import annotations

from pathlib import Path

import pytest

from graphician.analysis.coverage import graph_coverage
from graphician.extraction.languages import LanguageRegistry
from graphician.extraction.pipeline import ExtractionPipeline


@pytest.mark.parametrize(
    ("filename", "source"),
    [
        ("module.py", "def helper():\n    return 1\ndef run():\n    return helper()\n"),
        ("module.rs", "fn helper() -> i32 { 1 }\nfn run() -> i32 { helper() }\n"),
        (
            "module.ts",
            "function helper(): number { return 1; }\n"
            "function run(): number { return helper(); }\n",
        ),
        (
            "module.js",
            "function helper() { return 1; }\nfunction run() { return helper(); }\n",
        ),
        (
            "Module.java",
            "class Module { static int helper(){ return 1; } "
            "static int run(){ return helper(); } }\n",
        ),
        ("module.cpp", "int helper(){ return 1; }\nint run(){ return helper(); }\n"),
    ],
)
def test_supported_language_baseline_has_complete_definition_and_call_coverage(
    tmp_path: Path, filename: str, source: str
) -> None:
    (tmp_path / filename).write_text(source)

    graph = ExtractionPipeline(LanguageRegistry(), strict=True).build(tmp_path)
    coverage = graph_coverage(graph)

    assert coverage["source_location"]["rate"] == 1.0
    assert coverage["call_resolution"] == {
        "resolved": 1,
        "unresolved": 0,
        "total": 1,
        "rate": 1.0,
    }
    assert coverage["function_connectivity"]["total"] == 2
    assert coverage["function_connectivity"]["with_callers"] == 1
    assert coverage["function_connectivity"]["with_callees"] == 1
