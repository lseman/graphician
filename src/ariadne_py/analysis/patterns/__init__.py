"""Framework pattern detection."""

from __future__ import annotations

from .types import PatternCategory, FrameworkPattern, PatternMatch
from .builtin import built_in_patterns
from .matcher import detect_patterns, _match_pattern

__all__ = [
    "PatternCategory",
    "FrameworkPattern",
    "PatternMatch",
    "detect_patterns",
    "built_in_patterns",
    "_match_pattern",
]
