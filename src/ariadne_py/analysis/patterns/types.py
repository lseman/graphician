"""Types for pattern detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ...core.edge import EdgeKind
from ...core.id import NodeId
from ...core.node import NodeKind


class PatternCategory(str, Enum):
    """Category of framework pattern."""
    DEPENDENCY_INJECTION = "dependency_injection"
    ROUTING = "routing"
    LIFECYCLE = "lifecycle"
    STATE_MANAGEMENT = "state_management"
    VALIDATION = "validation"
    MIDDLEWARE = "middleware"
    DATA_MAPPING = "data_mapping"
    TESTING = "testing"
    COMMAND_LINE = "command_line"
    GENERIC = "generic"


@dataclass
class FrameworkPattern:
    """A single framework pattern definition."""
    id: str
    display_name: str
    description: str
    framework: str
    category: PatternCategory
    min_confidence: float = 0.5
    required_node_kinds: list[NodeKind] = field(default_factory=list)
    required_edge_kinds: list[EdgeKind] = field(default_factory=list)
    signature_names: list[str] = field(default_factory=list)
    import_patterns: list[str] = field(default_factory=list)
    min_nodes: int = 1
    max_nodes: int = 500
    requires_embeddings: bool = False


@dataclass
class PatternMatch:
    """A detected framework pattern match."""
    pattern_id: str
    display_name: str
    framework: str
    category: str
    confidence: float
    matched_nodes: list[dict[str, Any]]
    matched_edges: list[dict[str, Any]] = field(default_factory=list)
