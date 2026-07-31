"""Fuzzy string matching utilities."""

from __future__ import annotations


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if len(b) == 0:
        return len(a)

    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr_row = [i + 1]
        for j, cb in enumerate(b):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (ca != cb)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def _fuzzy_score(query: str, target: str) -> float:
    """Compute fuzzy match score between query and target."""
    if query == target:
        return 1.0
    if not query or not target:
        return 0.0

    # Normalize
    q = query.lower()
    t = target.lower()

    # Substring match bonus
    if q in t:
        return 0.8 + 0.2 * (len(q) / len(t))

    # Fuzzy match
    dist = _levenshtein(q, t)
    max_len = max(len(q), len(t))
    if max_len == 0:
        return 0.0

    similarity = 1.0 - (dist / max_len)
    return max(0.0, similarity)
