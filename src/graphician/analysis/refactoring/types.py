"""Types for refactoring operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Confidence(str, Enum):
    """Confidence level for a rename edit."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @classmethod
    def from_str(cls, value: str) -> "Confidence":
        """Convert string to Confidence enum."""
        for c in cls:
            if c.value == value:
                return c
        return cls.LOW


@dataclass
class RenameEdit:
    """A single rename edit suggestion."""
    file: str | None = None
    line: int | None = None
    old: str = ""
    new: str = ""
    confidence: Confidence | str = Confidence.HIGH


@dataclass
class RenameStats:
    """Stats for rename preview."""
    high: int = 0
    medium: int = 0
    low: int = 0
    total: int = 0

    @classmethod
    def from_edits(cls, edits: list[RenameEdit]) -> "RenameStats":
        """Compute stats from a list of edits."""
        stats = cls()
        for edit in edits:
            conf = edit.confidence
            if isinstance(conf, Confidence):
                conf = conf.value
            if conf == "high":
                stats.high += 1
            elif conf == "medium":
                stats.medium += 1
            else:
                stats.low += 1
        stats.total = len(edits)
        return stats


@dataclass
class RenamePreview:
    """Rename preview result."""
    target_qname: str = ""
    target_name: str = ""
    new_name: str = ""
    target_kind: str = ""
    edits: list[RenameEdit] = field(default_factory=list)
    stats: RenameStats = field(default_factory=RenameStats)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "target_qname": self.target_qname,
            "target_name": self.target_name,
            "new_name": self.new_name,
            "target_kind": self.target_kind,
            "edits": [
                {
                    "file": e.file,
                    "line": e.line,
                    "old": e.old,
                    "new": e.new,
                    "confidence": e.confidence if isinstance(e.confidence, str) else e.confidence.value,
                }
                for e in self.edits
            ],
            "stats": {
                "high": self.stats.high,
                "medium": self.stats.medium,
                "low": self.stats.low,
                "total": self.stats.total,
            },
        }
