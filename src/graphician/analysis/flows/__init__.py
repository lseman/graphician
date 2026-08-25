"""Execution flow detection."""

from __future__ import annotations

from .detection import (
    _detect_entry_points,
)
from .detection import (
    compute_flows_with_options as compute_flows,
)
from .entry_points import (
    _check_decorators,
    _is_framework_entry,
    _is_generic_event_entry,
    _is_java_framework_entry,
    _is_js_ts_framework_entry,
    _is_python_framework_entry,
)
from .trace import _compute_criticality, _is_test_node, _trace_flow
from .types import FlowOptions

__all__ = [
    "FlowOptions",
    "_check_decorators",
    "_compute_criticality",
    "_detect_entry_points",
    "_is_framework_entry",
    "_is_generic_event_entry",
    "_is_java_framework_entry",
    "_is_js_ts_framework_entry",
    "_is_python_framework_entry",
    "_is_test_node",
    "_trace_flow",
    "compute_flows",
]
