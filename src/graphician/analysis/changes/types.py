"""Types for change analysis."""

from __future__ import annotations

from dataclasses import dataclass, field


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
class RiskScore:
    """Risk assessment for a symbol."""
    qualified_name: str
    kind: str
    overall: float
    structural: float
    test: float
    security: float
    reasons: list[str] = field(default_factory=list)
