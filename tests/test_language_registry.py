"""Compatibility tests for declared tree-sitter language packages."""

import pytest

from ariadne_py.extraction.languages import Language, LanguageRegistry


@pytest.mark.parametrize("language", list(Language))
def test_builtin_parser_factories_use_supported_package_apis(language: Language) -> None:
    registry = LanguageRegistry()
    spec = registry.get_spec_by_name(language)

    assert spec is not None
    assert spec.parser_factory() is not None
