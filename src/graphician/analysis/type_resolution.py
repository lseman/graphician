"""Resolution of ``type::<Name>`` placeholder nodes.

``type::<Name>`` placeholder nodes are left by supertype extraction (e.g.
Java's ``emit_superclass`` / ``emit_interfaces``). This pass runs after
the merged graph is assembled: for every placeholder with a unique
same-named ``Class`` / ``Trait`` node elsewhere in the graph, every
``Inherits`` / ``Implements`` edge pointing at the placeholder is rewired
to the real node instead.

Ambiguous names (multiple candidates) are left as placeholders —
matching the project's conservative-resolution philosophy used by
``call_resolution.py``.
"""

from __future__ import annotations

from ..core.graph import Graph


def resolve_type_placeholders(graph: Graph) -> int:
    """Resolve ``type::<Name>`` placeholders left by supertype extraction.

    Returns the number of edges rewired.
    """
    # Keep this historical import path as a shallow compatibility adapter;
    # extraction.type_resolution owns the native dispatch and Python fallback.
    from ..extraction.type_resolution import resolve_type_placeholders as resolve

    return resolve(graph)
