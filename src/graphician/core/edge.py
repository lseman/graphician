"""Edge kinds, confidence, and Edge dataclass.

Edge kinds split into:
- Structural: defines, calls, imports, inherits, implements, data_flow,
  reads_writes, tested_by, member_of, entry_of, depends_on
- Semantic: mentions, describes, similar_to, rationale_for
- Cross-modal: illustrates, documented_by
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class EdgeKind(enum.StrEnum):
    """Type of relationship between two nodes."""
    # Structural
    DEFINES = "defines"
    CALLS = "calls"
    IMPORTS = "imports"
    INHERITS = "inherits"
    IMPLEMENTS = "implements"
    DATA_FLOW = "data_flow"
    READS_WRITES = "reads_writes"
    TESTED_BY = "tested_by"
    MEMBER_OF = "member_of"
    ENTRY_OF = "entry_of"
    DEPENDS_ON = "depends_on"
    # Semantic
    MENTIONS = "mentions"
    DESCRIBES = "describes"
    SIMILAR_TO = "similar_to"
    RATIONALE_FOR = "rationale_for"
    # Cross-modal
    ILLUSTRATES = "illustrates"
    DOCUMENTED_BY = "documented_by"


class Confidence(enum.StrEnum):
    """Confidence level for an edge."""
    EXTRACTED = "extracted"
    INFERRED = "inferred"
    AMBIGUOUS = "ambiguous"

    def score(self) -> float:
        if self == Confidence.EXTRACTED:
            return 1.0
        return 0.0


@dataclass
class Edge:
    """A typed relationship between two nodes."""
    kind: EdgeKind
    confidence: Confidence = Confidence.EXTRACTED
    properties: dict[str, Any] = field(default_factory=dict)
    valid_from: str | None = None
    valid_to: str | None = None

    @staticmethod
    def extracted(kind: EdgeKind) -> Edge:
        return Edge(kind=kind, confidence=Confidence.EXTRACTED)

    @staticmethod
    def inferred(kind: EdgeKind, score: float) -> Edge:
        return Edge(kind=kind, confidence=Confidence.INFERRED, properties={"score": score})

    @staticmethod
    def ambiguous(kind: EdgeKind) -> Edge:
        return Edge(kind=kind, confidence=Confidence.AMBIGUOUS)

    def with_property(self, key: str, value: Any) -> Edge:
        self.properties[key] = value
        return self
