"""Motif DSL: NamePattern, MotifNode, MotifEdge, Motif, MotifBuilder."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class NamePattern:
    """How to match a node's name field."""

    def __init__(self, kind: str, pattern: str) -> None:
        self.kind = kind  # "exact", "contains", "glob", "regex"
        self.pattern = pattern

    @classmethod
    def exact(cls, name: str) -> NamePattern:
        return cls("exact", name)

    @classmethod
    def contains(cls, name: str) -> NamePattern:
        return cls("contains", name)

    @classmethod
    def glob(cls, pat: str) -> NamePattern:
        return cls("glob", pat)

    @classmethod
    def regex(cls, pat: str) -> NamePattern:
        return cls("regex", pat)

    def matches(self, name: str) -> bool:
        if self.kind == "exact":
            return name == self.pattern
        elif self.kind == "contains":
            return self.pattern.lower() in name.lower()
        elif self.kind == "glob":
            regex_pat = _glob_to_regex(self.pattern)
            try:
                return bool(re.match(regex_pat, name))
            except re.error:
                return False
        elif self.kind == "regex":
            try:
                return bool(re.search(self.pattern, name))
            except re.error:
                return False
        return False


def _glob_to_regex(pat: str) -> str:
    """Convert a simple glob pattern (only * wildcards) to regex."""
    out = "^"
    for ch in pat:
        if ch == "*":
            out += "[^:]*"
        elif ch in r".()+?[]{}|\\^$#@":
            out += "\\" + ch
        else:
            out += ch
    out += "$"
    return out


@dataclass
class MotifNode:
    """Constraint on a pattern node."""
    id: int = 0
    kind: Any = None  # NodeKind
    name: NamePattern | None = None
    min_degree: int | None = None


@dataclass
class MotifEdge:
    """Constraint on a pattern edge."""
    from_id: int = 0
    to_id: int = 0
    kind: Any = None  # EdgeKind


@dataclass
class Motif:
    """A motif (subgraph pattern) to search for in the graph."""
    nodes: list[MotifNode] = field(default_factory=list)
    edges: list[MotifEdge] = field(default_factory=list)

    @staticmethod
    def builder() -> MotifBuilder:
        return MotifBuilder()

    def validate(self) -> tuple[bool, str]:
        """Validate that the motif is well-formed."""
        node_ids = {n.id for n in self.nodes}
        if len(node_ids) != len(self.nodes):
            return False, "duplicate node id in motif"
        for e in self.edges:
            if e.from_id not in node_ids:
                return False, f"edge from node {e.from_id} not found in motif nodes"
            if e.to_id not in node_ids:
                return False, f"edge to node {e.to_id} not found in motif nodes"
        return True, ""


class MotifBuilder:
    """Builder for motifs."""

    def __init__(self) -> None:
        self._nodes: list[MotifNode] = []
        self._edges: list[MotifEdge] = []

    def add_node(self, f: Callable[[MotifNodeBuilder], MotifNodeBuilder]) -> MotifBuilder:
        id_ = len(self._nodes)
        node = MotifNode(id=id_)
        builder = MotifNodeBuilder(node)
        result = f(builder)
        self._nodes.append(result._node)
        return self

    def add_edge(self, from_id: int, to_id: int, kind: Any) -> MotifBuilder:
        self._edges.append(MotifEdge(from_id=from_id, to_id=to_id, kind=kind))
        return self

    def build(self) -> Motif:
        return Motif(nodes=self._nodes, edges=self._edges)


class MotifNodeBuilder:
    """Helper builder for individual nodes."""

    def __init__(self, node: MotifNode) -> None:
        self._node = node

    def kind(self, kind: Any) -> MotifNodeBuilder:
        self._node.kind = kind
        return self

    def name_exact(self, name: str) -> MotifNodeBuilder:
        self._node.name = NamePattern.exact(name)
        return self

    def name_contains(self, name: str) -> MotifNodeBuilder:
        self._node.name = NamePattern.contains(name)
        return self

    def name_regex(self, pat: str) -> MotifNodeBuilder:
        self._node.name = NamePattern.regex(pat)
        return self

    def name_glob(self, pat: str) -> MotifNodeBuilder:
        self._node.name = NamePattern.glob(pat)
        return self

    def min_degree(self, min_deg: int) -> MotifNodeBuilder:
        self._node.min_degree = min_deg
        return self
