"""Built-in motif queries."""

from __future__ import annotations

from ...core.edge import EdgeKind
from ...core.node import NodeKind
from .dsl import Motif


def security_audit_motif() -> Motif:
    """Two Functions linked by Calls where the first contains 'sql'."""
    return (
        Motif.builder()
        .add_node(lambda n: n.kind(NodeKind.FUNCTION).name_contains("sql"))
        .add_node(lambda n: n.kind(NodeKind.FUNCTION))
        .add_edge(0, 1, EdgeKind.CALLS)
        .build()
    )


def diamond_inheritance_motif() -> Motif:
    """Diamond inheritance: Class with two parent Classes, both inheriting from grandparent."""
    return (
        Motif.builder()
        .add_node(lambda n: n.kind(NodeKind.CLASS))  # child
        .add_node(lambda n: n.kind(NodeKind.CLASS))  # parent 1
        .add_node(lambda n: n.kind(NodeKind.CLASS))  # parent 2
        .add_node(lambda n: n.kind(NodeKind.CLASS))  # grandparent
        .add_edge(1, 3, EdgeKind.INHERITS)
        .add_edge(2, 3, EdgeKind.INHERITS)
        .add_edge(0, 1, EdgeKind.INHERITS)
        .add_edge(0, 2, EdgeKind.INHERITS)
        .build()
    )


def doc_function_triangle() -> Motif:
    """Document → Mentions → Concept → Mentions → Function + Describes."""
    return (
        Motif.builder()
        .add_node(lambda n: n.kind(NodeKind.DOCUMENT))
        .add_node(lambda n: n.kind(NodeKind.FUNCTION))
        .add_node(lambda n: n.kind(NodeKind.CONCEPT))
        .add_edge(0, 2, EdgeKind.MENTIONS)
        .add_edge(2, 1, EdgeKind.MENTIONS)
        .add_edge(0, 1, EdgeKind.DESCRIBES)
        .build()
    )
