"""Framework pattern detection."""

from __future__ import annotations

from .builtin import built_in_patterns
from .matcher import _match_pattern, detect_patterns
from .types import FrameworkPattern, PatternCategory, PatternMatch

__all__ = [
    "FrameworkPattern",
    "PatternCategory",
    "PatternMatch",
    "_match_pattern",
    "built_in_patterns",
    "detect_patterns",
]
