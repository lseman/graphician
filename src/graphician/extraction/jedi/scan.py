"""Char-level scanning of Python source for method calls that
tree-sitter's grammar-based extraction drops.

Tree-sitter keeps calls with ``self/cls/super`` receivers and
uppercase-receiver calls (e.g. ``MyClass.method()``). This module finds
calls with lowercase receivers that were dropped, plus ``getattr(obj,
"method")`` and ``type(x).method()`` patterns.

Returns a list of ``(line, col, method_name, enclosing_qname)`` tuples.
Line is 1-indexed, col is 0-indexed (Jedi format).
"""

from __future__ import annotations


def find_dropped_calls(
    source: str,
    func_nodes: list[tuple[str, int, int]],
) -> list[tuple[int, int, str, str]]:
    """Walk Python source to find method calls that tree-sitter drops.

    Args:
        source: The Python source code.
        func_nodes: List of (qualified_name, line_start, line_end) for
            every function/method node in the graph.

    Returns:
        List of (line, col, method_name, enclosing_qname) tuples for
        dropped calls. Line is 1-indexed, col is 0-indexed.
    """
    results: list[tuple[int, int, str, str]] = []
    lines = source.splitlines()

    for line_idx, line in enumerate(lines):
        line_num = line_idx + 1

        # 1. Attribute calls: receiver.method(...)
        results.extend(_find_attribute_calls(line, line_num, func_nodes))

        # 2. getattr calls: getattr(obj, "method")
        results.extend(_find_getattr_calls(line, line_num, func_nodes))

        # 3. type() calls: type(x).method()
        results.extend(_find_wrapper_call_calls(line, line_num, func_nodes, "type("))

        # 4. callable() calls: callable(x).method()
        results.extend(_find_wrapper_call_calls(line, line_num, func_nodes, "callable("))

    # Deduplicate by (line, method_name, enclosing)
    seen: set[tuple[int, str, str]] = set()
    deduped = []
    for r in results:
        key = (r[0], r[2], r[3])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


def _find_identifier_before(chars: list[str], pos: int) -> tuple[int, str] | None:
    """Find an identifier ending just before *pos* in the char array.

    The identifier must start with a lowercase letter.
    """
    if pos == 0:
        return None

    # Skip whitespace before the dot
    i = pos
    while i > 0 and chars[i - 1].isspace():
        i -= 1
    if i == 0:
        return None

    # Check if the character before whitespace is alphanumeric
    if not (chars[i - 1].isalnum() or chars[i - 1] == "_"):
        return None

    # Find the start of the identifier
    start = i
    while i > 0 and (chars[i - 1].isalnum() or chars[i - 1] == "_"):
        i -= 1
    name = "".join(chars[i:start])
    if not name or not name[0].islower():
        return None
    return (i, name)


def _find_identifier_after(chars: list[str], pos: int) -> tuple[int, str] | None:
    """Find an identifier starting just after *pos* (the dot)."""
    i = pos + 1
    # Skip whitespace after the dot
    while i < len(chars) and chars[i].isspace():
        i += 1
    if i >= len(chars) or not (chars[i].isalpha() or chars[i] == "_"):
        return None

    start = i
    while i < len(chars) and (chars[i].isalnum() or chars[i] == "_"):
        i += 1
    name = "".join(chars[start:i])
    if not name:
        return None
    return (pos + 1, name)


def _is_dropped_receiver(name: str) -> bool:
    """Check if a receiver name should be dropped by tree-sitter.

    Keeps self/cls/super and uppercase receivers.
    """
    if name in ("self", "cls", "super"):
        return False
    return not (name[0:1] and name[0].isupper())


def _find_enclosing_func(line: int, func_nodes: list[tuple[str, int, int]]) -> str:
    """Find the qualified name of the function enclosing *line*.

    Returns the innermost (narrowest) enclosing function, or empty string.
    """
    best: str | None = None
    best_span = float("inf")
    for qname, start, end in func_nodes:
        if start <= line <= end:
            span = end - start
            if span < best_span:
                best_span = span
                best = qname
    return best or ""


def _find_attribute_calls(
    line: str,
    line_num: int,
    func_nodes: list[tuple[str, int, int]],
) -> list[tuple[int, int, str, str]]:
    """Find attribute calls: ``receiver.method(...)``."""
    results: list[tuple[int, int, str, str]] = []
    chars = list(line)
    i = 0
    while i < len(chars):
        if chars[i] == "." and i + 2 < len(chars):
            recv_result = _find_identifier_before(chars, i)
            if recv_result:
                _recv_start, recv_name = recv_result
                if _is_dropped_receiver(recv_name):
                    method_result = _find_identifier_after(chars, i)
                    if method_result:
                        method_col, method_name = method_result
                        enclosing = _find_enclosing_func(line_num, func_nodes)
                        results.append((line_num, method_col, method_name, enclosing))
            i += 1
        else:
            i += 1
    return results


def _find_getattr_calls(
    line: str,
    line_num: int,
    func_nodes: list[tuple[str, int, int]],
) -> list[tuple[int, int, str, str]]:
    """Find ``getattr(obj, "method")`` patterns."""
    results: list[tuple[int, int, str, str]] = []
    chars = list(line)

    # Find `getattr(` or `getattr (`
    for idx in range(len(chars)):
        if line[idx:].startswith("getattr"):
            # Check it's the function call, not part of a name
            if idx > 0 and (chars[idx - 1].isalnum() or chars[idx - 1] == "_"):
                continue

            # Check it's followed by `(`
            after = idx + 7
            j = after
            while j < len(chars) and chars[j].isspace():
                j += 1
            if j >= len(chars) or chars[j] != "(":
                continue

            # Extract method name from getattr call
            method_name = _extract_getattr_method_name(line[idx:])
            if method_name:
                enclosing = _find_enclosing_func(line_num, func_nodes)
                results.append((line_num, idx, method_name, enclosing))
            break
    return results


def _extract_getattr_method_name(s: str) -> str | None:
    """Extract the method name from a ``getattr`` call string.

    Handles: ``getattr(obj, "method")``, ``getattr(obj, 'method')``,
    ``getattr(obj, "method", default)``, and ``getattr(obj, attr_name)``.
    """
    in_string: str | None = None
    comma_count = 0
    chars = list(s)
    j = 0

    # Skip past ``getattr(`` or ``getattr (``
    while j < len(chars) and (chars[j].isspace() or chars[j] == "("):
        j += 1

    # Skip past first argument (obj)
    while j < len(chars) and comma_count < 1:
        ch = chars[j]
        if ch in ('"', "'") and in_string is None:
            in_string = ch
        elif in_string in ('"', "'") and ch == in_string:
            in_string = None
        elif ch == "," and in_string is None:
            comma_count += 1
            in_string = None
        elif ch == ")" and in_string is None:
            return None
        j += 1

    # Skip whitespace, looking for the method name
    while j < len(chars) and chars[j].isspace():
        j += 1

    if j >= len(chars):
        return None

    # Quoted string literal
    if chars[j] in ('"', "'"):
        quote = chars[j]
        start = j + 1
        end = start
        while end < len(chars) and chars[end] != quote:
            end += 1
        if end < len(chars):
            return "".join(chars[start:end])

    # Variable name
    if chars[j].isalpha() or chars[j] == "_":
        start = j
        while j < len(chars) and (chars[j].isalnum() or chars[j] == "_"):
            j += 1
        name = "".join(chars[start:j])
        return f"getattr_variable({name})"

    return None


def _find_wrapper_call_calls(
    line: str,
    line_num: int,
    func_nodes: list[tuple[str, int, int]],
    prefix: str,
) -> list[tuple[int, int, str, str]]:
    """Find ``<prefix>x).method()`` patterns (e.g. ``type(x).method()``)."""
    results: list[tuple[int, int, str, str]] = []
    chars = list(line)

    for idx in range(len(chars)):
        if line[idx:].startswith(prefix):
            # Check it's the function call, not part of a name
            if idx > 0 and (chars[idx - 1].isalnum() or chars[idx - 1] == "_"):
                continue

            # Find the closing paren (handle nesting)
            depth = 1  # start after opening ( of the prefix
            end = idx + len(prefix)
            found_close = False
            while end < len(chars):
                if chars[end] == "(":
                    depth += 1
                elif chars[end] == ")":
                    depth -= 1
                    if depth == 0:
                        # Check for .method() after closing paren
                        after = end + 1
                        if after < len(chars) and chars[after] == ".":
                            method_result = _find_identifier_after(chars, after)
                            if method_result:
                                method_col, method_name = method_result
                                enclosing = _find_enclosing_func(line_num, func_nodes)
                                results.append((line_num, method_col, method_name, enclosing))
                        found_close = True
                        break
                end += 1

            if found_close:
                break
    return results
