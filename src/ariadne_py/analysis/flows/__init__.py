"""Execution flow detection."""

from __future__ import annotations

from .types import FlowOptions
from .detection import (
    compute_flows_with_options as compute_flows,
    _detect_entry_points,
)
from .entry_points import (
    _is_framework_entry,
    _is_python_framework_entry,
    _is_js_ts_framework_entry,
    _is_java_framework_entry,
    _is_generic_event_entry,
    _check_decorators,
)
from .trace import _trace_flow, _compute_criticality, _is_test_node

__all__ = [
    "FlowOptions",
    "compute_flows",
    "_detect_entry_points",
    "_is_framework_entry",
    "_is_python_framework_entry",
    "_is_js_ts_framework_entry",
    "_is_java_framework_entry",
    "_is_generic_event_entry",
    "_check_decorators",
    "_trace_flow",
    "_compute_criticality",
    "_is_test_node",
]
