"""Framework pattern detection."""

from __future__ import annotations

from .types import PatternCategory, FrameworkPattern, PatternMatch
from .builtin import _builtin_patterns
from .matcher import detect_patterns, _match_pattern

__all__ = [
    "PatternCategory",
    "FrameworkPattern",
    "PatternMatch",
    "detect_patterns",
    "_builtin_patterns",
    "_match_pattern",
]
