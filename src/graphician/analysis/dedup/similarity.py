"""Pass 4: Jaro-Winkler similarity."""

from __future__ import annotations


def jaro_winkler(s1: str, s2: str) -> float:
    """Compute Jaro-Winkler similarity between two strings."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    match_count = 0
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            match_count += 1
            break

    if match_count == 0:
        return 0.0

    # Count transpositions
    matched_s1 = [char for char, matched in zip(s1, s1_matches, strict=True) if matched]
    matched_s2 = [char for char, matched in zip(s2, s2_matches, strict=True) if matched]
    transpositions = sum(a != b for a, b in zip(matched_s1, matched_s2, strict=True)) / 2

    # Count common prefix
    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    jaro = (
        match_count / len1
        + match_count / len2
        + (match_count - transpositions) / match_count
    ) / 3.0

    winkler = jaro + prefix * 0.1 * (1 - jaro)
    return min(winkler, 1.0)
