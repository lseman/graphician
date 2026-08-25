"""Subgraph motif matching with VF2-style subgraph isomorphism."""

from __future__ import annotations

from .builtins import (
    diamond_inheritance_motif,
    doc_function_triangle,
    security_audit_motif,
)
from .dsl import (
    Motif,
    MotifBuilder,
    MotifEdge,
    MotifNode,
    MotifNodeBuilder,
    NamePattern,
)
from .engine import MotifMatch, find_motifs

__all__ = [
    "Motif",
    "MotifBuilder",
    "MotifEdge",
    "MotifMatch",
    "MotifNode",
    "MotifNodeBuilder",
    "NamePattern",
    "diamond_inheritance_motif",
    "doc_function_triangle",
    "find_motifs",
    "security_audit_motif",
]
