"""Subgraph motif matching with VF2-style subgraph isomorphism."""

from __future__ import annotations

from .dsl import (
    NamePattern,
    MotifNode,
    MotifEdge,
    Motif,
    MotifBuilder,
    MotifNodeBuilder,
)
from .engine import MotifMatch, find_motifs
from .builtins import (
    security_audit_motif,
    diamond_inheritance_motif,
    doc_function_triangle,
)

__all__ = [
    "NamePattern",
    "MotifNode",
    "MotifEdge",
    "Motif",
    "MotifBuilder",
    "MotifNodeBuilder",
    "MotifMatch",
    "find_motifs",
    "security_audit_motif",
    "diamond_inheritance_motif",
    "doc_function_triangle",
]
