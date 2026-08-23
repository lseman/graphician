"""Change analysis: detect changes, risk scores, test coverage."""

from __future__ import annotations

from .types import Change, RiskScore
from .detection import detect_changes
from .risk import compute_risk
from .coverage import compute_test_coverage

__all__ = [
    "Change",
    "RiskScore",
    "detect_changes",
    "compute_risk",
    "compute_test_coverage",
]
