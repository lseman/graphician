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

    matches: list[str] = []
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches.append(s1[i])
            break

    if len(matches) == 0:
        return 0.0

    # Count transpositions
    t = 0
    match_iter = iter(matches)
    for i in range(len1):
        if not s1_matches[i]:
            continue
        j = next(match_iter)
        if s1[i] != s2[j]:
            t += 1
    t //= 2

    # Count common prefix
    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    jaro = (
        len1 / (2 * len1)
        + len2 / (2 * len2)
        + (len(matches) - t) / len(matches)
    ) / 3.0

    winkler = jaro + prefix * 0.1 * (1 - jaro)
    return min(winkler, 1.0)
