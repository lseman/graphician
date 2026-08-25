"""Change analysis: detect changes, risk scores, test coverage."""

from __future__ import annotations

from .coverage import compute_test_coverage
from .detection import detect_changes
from .risk import compute_risk
from .types import Change, RiskScore

__all__ = [
    "Change",
    "RiskScore",
    "compute_risk",
    "compute_test_coverage",
    "detect_changes",
]
