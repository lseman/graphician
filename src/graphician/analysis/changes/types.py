"""Types for change analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Change:
    """A single change detected in a diff."""
    file_path: str
    line_start: int
    line_end: int
    change_type: str  # "added", "removed", "modified"
    content: str
    affected_symbols: list[str] = field(default_factory=list)
    risk_score: float = 0.0


@dataclass
class RiskFactor:
    """A single risk factor and its computed contribution."""
    name: str
    weight: float
    score: float


@dataclass
class RiskScore:
    """Risk assessment for a symbol.

    Uses the CRG-style 5-factor model:
    - flow_participation (max 0.25)
    - community_crossing (max 0.15)
    - test_coverage (max 0.30)
    - security_sensitivity (max 0.20)
    - caller_count (max 0.10)
    """
    qualified_name: str
    kind: str
    overall: float
    level: str = "LOW"
    factors: list[dict[str, Any]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    # Legacy fields for API compatibility
    structural: float = 0.0
    test: float = 0.0
    security: float = 0.0
