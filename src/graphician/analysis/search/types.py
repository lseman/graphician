"""Types for search."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ...core.id import NodeId
from ...core.node import Node


class SearchIntent(Enum):
    """Classification of search query intent."""
    LOOKUP = "lookup"
    DEBUG = "debug"
    IMPACT = "impact"
    IMPLEMENTATION = "implementation"
    REVIEW = "review"
    ARCHITECTURE = "architecture"
    DOCUMENTATION = "documentation"

    @classmethod
    def classify(cls, query: str) -> SearchIntent:
        """Classify a query string into an intent."""
        q = query.lower()
        if any(term in q for term in (
            "why", "bug", "error", "fail", "panic",
            "exception", "debug",
        )):
            return cls.DEBUG
        if any(term in q for term in (
            "impact", "depend", "break", "affected",
            "blast radius",
        )):
            return cls.IMPACT
        if any(term in q for term in (
            "implement", "add", "change", "modify", "create",
        )):
            return cls.IMPLEMENTATION
        if any(term in q for term in (
            "review", "test", "coverage", "risk",
        )):
            return cls.REVIEW
        if any(term in q for term in (
            "architecture", "design", "structure", "pattern",
            "module", "package",
        )):
            return cls.ARCHITECTURE
        if any(term in q for term in (
            "document", "readme", "guide", "explain",
        )):
            return cls.DOCUMENTATION
        return cls.LOOKUP


@dataclass
class SearchHit:
    """A single search result."""
    id: NodeId
    node: Node
    score: float
    rank: int = 0
    reasons: list[str] = field(default_factory=list)
