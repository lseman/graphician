"""Compare dedicated Python and native extraction throughput.

Run with: PYTHONPATH=src python benchmarks/native_extractors.py
"""

from __future__ import annotations

import statistics
import tempfile
import time
from pathlib import Path

from graphician._extract.extractors import extract_rust_file as native_extract
from graphician.core.graph import Graph
from graphician.extraction.languages.parsers.rust import extract_file as python_extract

SOURCE = "\n".join(
    f"fn function_{index}() {{ function_{max(0, index - 1)}(); }}"
    for index in range(500)
)


def measure(extractor, path: Path, rounds: int = 20) -> float:
    samples = []
    for _ in range(rounds):
        graph = Graph()
        started = time.perf_counter()
        extractor(path, graph, file_qn="file::benchmark.rs")
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "benchmark.rs"
        path.write_text(SOURCE, encoding="utf-8")
        python_seconds = measure(python_extract, path)
        native_seconds = measure(native_extract, path)
    print(f"Python median: {python_seconds * 1000:.2f} ms")
    print(f"Native median: {native_seconds * 1000:.2f} ms")
    print(f"Speedup: {python_seconds / native_seconds:.2f}x")


if __name__ == "__main__":
    main()
