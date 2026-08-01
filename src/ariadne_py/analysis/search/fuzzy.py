"""Fuzzy string matching utilities.

Ported from ariadne-graph's ``analysis/search/fuzzy.rs``.
"""

from __future__ import annotations


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein distance between two strings.

    Plain O(n*m) DP, equivalent to rust's ``levenshtein`` (fuzzy.rs:173-193)
    minus the Myers bit-vector fast path, which is a perf-only optimization.
    """
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev_row = list(range(len(b) + 1))
    curr_row = [0] * (len(b) + 1)
    for i, ca in enumerate(a):
        curr_row[0] = i + 1
        for j, cb in enumerate(b):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (ca != cb)
            curr_row[j + 1] = min(insertions, deletions, substitutions)
        prev_row, curr_row = curr_row, prev_row

    return prev_row[-1]


def _compact(s: str) -> str:
    """Remove all whitespace. Mirrors rust's ``compact`` (fuzzy.rs:169-171)."""
    return "".join(c for c in s if not c.isspace())


def _ratio(a: str, b: str) -> float:
    """Normalized similarity in [0, 1]. Mirrors rust's ``ratio`` (fuzzy.rs:51-57)."""
    if not a and not b:
        return 1.0
    distance = _levenshtein(a, b)
    return 1.0 - distance / max(len(a), len(b))


def _partial_ratio(shorter: str, longer: str) -> float:
    """Best ratio over sliding windows. Mirrors rust's ``partial_ratio`` (fuzzy.rs:67-93)."""
    if not shorter or not longer:
        return 0.0
    needle, haystack = (shorter, longer) if len(shorter) <= len(longer) else (longer, shorter)
    needle_len = len(needle)
    if needle_len >= len(haystack):
        return _ratio(needle, haystack)
    best = 0.0
    for start in range(len(haystack) - needle_len + 1):
        window = haystack[start : start + needle_len]
        best = max(best, _ratio(needle, window))
        if best >= 1.0:
            break
    return best


def _sorted_tokens(s: str) -> list[str]:
    """Mirrors rust's ``sorted_tokens`` (fuzzy.rs:163-167)."""
    return sorted(s.split())


def _token_sort_ratio(a: str, b: str) -> float:
    """Mirrors rust's ``token_sort_ratio`` (fuzzy.rs:114-116)."""
    return _ratio(" ".join(_sorted_tokens(a)), " ".join(_sorted_tokens(b)))


def _token_set_ratio(a: str, b: str) -> float:
    """Mirrors rust's ``token_set_ratio`` (fuzzy.rs:118-133)."""
    a_tokens = list(dict.fromkeys(_sorted_tokens(a)))
    b_tokens = list(dict.fromkeys(_sorted_tokens(b)))
    common = [t for t in a_tokens if t in b_tokens]
    if not common:
        return 0.0
    common_text = " ".join(common)
    return max(_ratio(common_text, a), _ratio(common_text, b))


def _acronym_ratio(query: str, candidate: str) -> float:
    """Mirrors rust's ``acronym_ratio`` (fuzzy.rs:135-141)."""
    acronym = "".join(token[0] for token in candidate.split() if token)
    return _ratio(_compact(query), acronym)


def _subsequence_ratio(query: str, candidate: str) -> float:
    """Mirrors rust's ``subsequence_ratio`` (fuzzy.rs:143-161)."""
    qchars = iter(query)
    current = next(qchars, None)
    matched = 0
    for c in candidate:
        if current is not None and c == current:
            matched += 1
            current = next(qchars, None)
            if current is None:
                break
    if current is None:
        return matched / max(len(candidate), 1)
    return 0.0


def _fuzzy_score(query: str, target: str) -> float:
    """Compute fuzzy match score between query and target.

    Faithful port of rust's ``fuzzy_score`` (fuzzy.rs:32-49): the max of seven
    similarity signals — ratio(raw), ratio(compact), partial_ratio(compact),
    token_sort_ratio(raw), token_set_ratio(raw), acronym_ratio(raw),
    subsequence_ratio(compact).
    """
    if not query or not target:
        return 0.0

    compact_query = _compact(query)
    compact_target = _compact(target)

    signals = [
        _ratio(query, target),
        _ratio(compact_query, compact_target),
        _partial_ratio(compact_query, compact_target),
        _token_sort_ratio(query, target),
        _token_set_ratio(query, target),
        _acronym_ratio(query, target),
        _subsequence_ratio(compact_query, compact_target),
    ]
    return max(signals)
