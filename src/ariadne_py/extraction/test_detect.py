"""Heuristics for recognising test code.

Two complementary signals are exposed:

- ``is_test_file_path`` — does the file live in a test directory or
  follow a test-naming convention?
- ``is_test_name`` — is the function/method name itself test-shaped?

Language-specific extractors layer their own signals on top (e.g.
``#[test]`` attributes in Rust). The shared name/path patterns here are
deliberately conservative: anything they flag has a strong external
convention pointing at "this is a test".
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any


def is_test_file_path(file_path: str) -> bool:
    """Return ``True`` if *file_path* lives in a test directory or has a
    test file-name convention.

    Supports Rust ``tests/`` (integration), generic ``test/``, Python
    ``tests/``, Go ``_test.go``, JS ``__tests__``, Java ``Test``/``Tests``
    suffix, Kotlin/Swift/Scala conventions, Dart/Lua conventions, and more.
    """
    # Normalise to forward slashes for path-component checks.
    path = file_path.replace("\\", "/")

    # Directory components.
    if any(marker in path for marker in ("/tests/", "/test/", "/__tests__/")):
        return True
    if any(path.startswith(marker + "/") for marker in ("tests", "test", "__tests__")):
        return True
    if any(path.startswith(marker + "/") for marker in ("spec",)):
        return True

    # File-name components.
    p = PurePosixPath(file_path)
    name = p.name
    stem = p.stem
    ext = p.suffix.lstrip(".")

    # Universal stem/name patterns.
    if name.startswith("test_"):
        return True
    if stem.endswith("_test"):
        return True
    if stem.endswith("_spec"):
        return True
    if stem.endswith(".test"):
        return True
    if stem.endswith(".spec"):
        return True

    stem_lower = stem.lower()
    if stem_lower.startswith("test_helper") or stem_lower.startswith("test_helpers"):
        return True

    # Extension-gated suffix conventions (avoid false positives like
    # production ``Contest`` or ``Latest`` modules).
    ext_lower = ext.lower()
    if ext_lower in ("java", "cs", "php"):
        return stem.endswith("Test") or stem.endswith("Tests")
    if ext_lower in ("kt", "swift"):
        return (
            stem.endswith("Test")
            or stem.endswith("Tests")
            or stem.endswith("Spec")
        )
    if ext_lower == "scala":
        return (
            stem.endswith("Spec")
            or stem.endswith("Suite")
            or stem.endswith("Test")
        )
    if ext_lower == "dart":
        return stem_lower.startswith("test_") or stem_lower.endswith("_test")
    if ext_lower == "lua":
        return (
            stem_lower.startswith("test_")
            or stem_lower.endswith("_test")
            or stem_lower.endswith("_spec")
        )

    return False


def is_test_name(node_name: str, node_props: dict[str, Any] | None = None) -> bool:
    """Return ``True`` if *node_name* looks like a test function/method name.

    Checks for common test prefixes and patterns used across languages:
    ``test_``, ``test``, ``spec``, ``it_``, ``should_``, ``expect``,
    ``describe``, ``context``, ``fixture``, ``setup``, ``teardown``, etc.

    Language-specific extractors layer their own signals on top (e.g.
    ``#[test]`` attributes in Rust).
    """
    name_lower = node_name.lower().strip()

    # Remove common prefixes from qualified names (e.g. ``MyClass.test_foo``).
    stem = name_lower.split(".")[-1].split("::")[-1].split("::")[-1]
    stem = stem.strip()

    if not stem:
        return False

    # Test prefixes.
    if stem.startswith("test_") or stem.startswith("test"):
        return True
    if stem.startswith("spec_") or stem.startswith("spec"):
        return True
    if stem.startswith("it_") or stem.startswith("it"):
        return True
    if stem.startswith("should_") or stem.startswith("should"):
        return True
    if stem.startswith("expect"):
        return True
    if stem.startswith("describe"):
        return True
    if stem.startswith("context"):
        return True
    if stem.startswith("fixture"):
        return True
    if stem.startswith("setup") or stem.startswith("teardown"):
        return True
    if stem.startswith("before") or stem.startswith("after"):
        return True
    if stem.startswith("mock") or stem.startswith("stub"):
        return True

    # Decorator/annotation check (language-specific signals).
    if node_props:
        for key in ("decorators", "annotations"):
            val = node_props.get(key)
            if isinstance(val, list):
                for dec in val:
                    if isinstance(dec, str):
                        dec_lower = dec.lower()
                        if any(
                            marker in dec_lower
                            for marker in (
                                "@test", "@pytest", "@junit", "@spec",
                                "@describe", "@it_", "@should",
                                "@before", "@after", "@fixture",
                                "@gtest", "@catch", "@benchmark",
                            )
                        ):
                            return True

    return False
