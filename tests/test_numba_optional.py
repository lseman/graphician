"""Tests for operation without the optional Numba accelerator."""

import builtins
import importlib.util
from pathlib import Path


def test_numba_acceleration_module_imports_without_numba(monkeypatch) -> None:
    module_path = (
        Path(__file__).parents[1]
        / "src"
        / "graphician"
        / "analysis"
        / "communities"
        / "numba_accel.py"
    )
    original_import = builtins.__import__

    def import_without_numba(name, *args, **kwargs):
        if name == "numba":
            raise ImportError("numba intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_numba)
    spec = importlib.util.spec_from_file_location("numba_accel_without_numba", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    assert module.has_numba() is False
