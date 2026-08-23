"""Pass 1-2: Normalization and entropy gate."""

from __future__ import annotations

import math
import re


def normalize_label(label: str) -> str:
    """Normalize a label for comparison.

    Steps:
    1. Lowercase
    2. Strip version suffixes (v2, 1.0, _2, etc.)
    3. Collapse non-alphanumeric characters to single underscores
    4. Trim leading/trailing underscores
    """
    lower = label.lower()
    # Strip version suffixes: "method v2" -> "method", "fn 1.0" -> "fn"
    stripped = re.sub(r'\b[vV]?[ _]?\d+\.?\d*$', '', lower)
    stripped = re.sub(r'([a-zA-Z_])[ _]?\d+$', r'\1', stripped)
    # Collapse non-alphanumeric to underscores
    collapsed = re.sub(r'[^a-z0-9]+', '_', stripped)
    trimmed = collapsed.strip('_')
    return trimmed if trimmed else label


def shannon_entropy(s: str) -> float:
    """Compute Shannon entropy (bits/char) of a string."""
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    length = len(s)
    entropy = 0.0
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


def passes_entropy_gate(normalized: str, threshold: float) -> bool:
    """Check if a normalized label passes the entropy gate."""
    length = len(normalized)
    if length < 3:
        return True
    if length <= 5:
        unique_chars = len(set(normalized))
        if unique_chars <= 3:
            return False
    return shannon_entropy(normalized) >= threshold
