"""Node kinds and Node dataclass.

Kinds split into three families:
- Code: File, Module, Class, Function, Method, Trait, Impl, Variable, Type
- Prose: Document, Section, Concept
- Visual: Diagram, Image
- Provenance: Commit, Author
- Synthetic: Hyperedge, Flow, Package
"""

from __future__ import annotations

import enum
import textwrap
from dataclasses import dataclass, field
from typing import Any


class NodeKind(enum.StrEnum):
    """Type of entity a node represents."""
    # Code
    FILE = "file"
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    TRAIT = "trait"
    IMPL = "impl"
    VARIABLE = "variable"
    TYPE = "type"
    # Prose
    DOCUMENT = "document"
    SECTION = "section"
    CONCEPT = "concept"
    # Visual
    DIAGRAM = "diagram"
    IMAGE = "image"
    # Provenance
    COMMIT = "commit"
    AUTHOR = "author"
    # Synthetic
    HYPEREDGE = "hyperedge"
    FLOW = "flow"
    PACKAGE = "package"


@dataclass
class Node:
    """A property-bag node in the graph.

    qualified_name is the canonical key — must be unique within the graph.
    Built from <file>::<module-path>::<name> for source-derived nodes.
    """
    kind: NodeKind
    name: str
    qualified_name: str
    source_uri: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    valid_from: str | None = None
    valid_to: str | None = None
    source_text: str | None = None

    @classmethod
    def new(cls, kind: NodeKind, qualified_name: str) -> Node:
        """Create a node. Name is derived from qualified_name."""
        name = qualified_name.rsplit("::", 1)[-1] if "::" in qualified_name else qualified_name
        return cls(
            kind=kind,
            name=name,
            qualified_name=qualified_name,
        )

    def with_source(
        self,
        uri: str,
        line_start: int,
        line_end: int,
    ) -> Node:
        """Attach source location."""
        self.source_uri = uri
        self.line_start = line_start
        self.line_end = line_end
        return self

    def with_source_text(self, text: str) -> Node:
        """Attach source text, truncated to 10KB (UTF-8 safe)."""
        max_bytes = 10_000
        encoded = text.encode("utf-8")[:max_bytes]
        self.source_text = encoded.decode("utf-8", errors="ignore")
        return self

    def with_property(self, key: str, value: Any) -> Node:
        """Attach a property."""
        self.properties[key] = value
        return self
