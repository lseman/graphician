"""Types for flow detection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FlowOptions:
    """Tunable limits for flow tracing."""
    max_depth: int = 6
    max_nodes_per_flow: int = 200
    min_flow_size: int = 3
