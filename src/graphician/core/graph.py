"""Custom property graph.

Lightweight directed graph with:
- Stable node/edge IDs
- qualified_name → NodeIndex secondary index
- O(1) symbol resolution
- Edge deduplication on merge
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator

from .edge import Edge, EdgeKind
from .id import EdgeId, NodeId
from .node import Node
from .normalize import _normalize_identifier


class Graph:
    """In-memory property graph.

    Maintains a secondary index from qualified_name to node index for O(1)
    symbol resolution. Nodes are stored by stable integer ID.
    """

    def __init__(self) -> None:
        self._nodes: dict[int, Node] = {}
        self._edges: dict[int, tuple[int, int, Edge]] = {}
        self._by_qname: dict[str, int] = {}
        self._out: dict[int, list[tuple[int, int]]] = defaultdict(list)  # src → [(dst, edge_id)]
        self._in: dict[int, list[tuple[int, int]]] = defaultdict(list)  # dst → [(src, edge_id)]
        self._next_node_id: int = 0
        self._next_edge_id: int = 0
        # Native analysis snapshots are cached by ``analysis.native``.  Keep
        # the cache opaque here so the core graph does not depend on the
        # optional extension, while still giving every structural mutation a
        # cheap and authoritative invalidation token.
        self._native_revision: int = 0
        self._native_snapshot: object | None = None
        self._native_snapshot_key: object | None = None

    def _invalidate_native_snapshot(self) -> None:
        """Invalidate the optional native analysis snapshot."""
        self._native_revision += 1
        self._native_snapshot = None
        self._native_snapshot_key = None

    # ── Node operations ──────────────────────────────────────────────

    def add_node(self, node: Node) -> NodeId:
        """Add a node. Duplicate qualified_name merges instead of overwriting.

        Merging strategy: keep AST-sourced attributes (source_uri, line_start,
        line_end, source_text) from the existing node since the AST pass runs
        first and has the most precise source information. Properties from the
        new node fill gaps but never overwrite existing properties. This
        prevents data-flow or call-resolution passes from losing the AST
        source_location of a symbol.
        """
        qn = node.qualified_name
        if qn in self._by_qname:
            idx = self._by_qname[qn]
            existing = self._nodes[idx]
            # Merge: keep AST source info from existing, fill gaps from new
            if existing.source_uri is None:
                existing.source_uri = node.source_uri
            if existing.line_start is None:
                existing.line_start = node.line_start
            if existing.line_end is None:
                existing.line_end = node.line_end
            if existing.source_text is None:
                existing.source_text = node.source_text
            # Merge properties: new fills gaps, doesn't overwrite
            if node.properties:
                for k, v in node.properties.items():
                    if k not in existing.properties:
                        existing.properties[k] = v
            # Update normalized forms only if not already set
            if existing.normalized_name is None and node.normalized_name is not None:
                existing.normalized_name = node.normalized_name
            if existing.normalized_qname is None and node.normalized_qname is not None:
                existing.normalized_qname = node.normalized_qname
            return NodeId(idx)
        idx = self._next_node_id
        self._next_node_id += 1
        self._nodes[idx] = node
        self._by_qname[qn] = idx
        # Pre-compute normalized forms if not already set (search optimization)
        if node.normalized_name is None:
            nn = _normalize_identifier(node.name)
            nq = _normalize_identifier(node.qualified_name)
            if nn is not None and nq is not None:
                node.normalized_name = nn
                node.normalized_qname = nq
        self._invalidate_native_snapshot()
        return NodeId(idx)

    def remove_node(self, id: NodeId) -> None:
        """Remove a node and all incident edges."""
        idx = id.value
        if idx not in self._nodes:
            return
        self._invalidate_native_snapshot()
        qn = self._nodes[idx].qualified_name
        self._by_qname.pop(qn, None)

        # Remove all incident edges
        for dst, eid in self._out.pop(idx, []):
            self._edges.pop(eid, None)
            self._in[dst] = [(s, e) for s, e in self._in[dst] if e != eid]

        for src, eid in self._in.pop(idx, []):
            self._edges.pop(eid, None)
            self._out[src] = [(d, e) for d, e in self._out[src] if e != eid]

        del self._nodes[idx]

    def rename_node(self, id: NodeId, new_qn: str, new_name: str) -> NodeId:
        """Rename a node. Merges if new_qn collides with another node."""
        idx = id.value
        if idx not in self._nodes:
            return id

        # Collision check
        if new_qn in self._by_qname:
            existing = self._by_qname[new_qn]
            if existing != idx:
                return self._merge_into(idx, existing)

        old_qn = self._nodes[idx].qualified_name
        self._by_qname.pop(old_qn, None)
        node = self._nodes[idx]
        node.qualified_name = new_qn
        node.name = new_name
        self._by_qname[new_qn] = idx
        return id

    def _merge_into(self, loser: int, winner: int) -> NodeId:
        """Rewire loser's edges onto winner, then remove loser."""
        self._invalidate_native_snapshot()
        incoming: list[tuple[int, int]] = list(self._in.get(loser, []))
        outgoing: list[tuple[int, int]] = list(self._out.get(loser, []))

        for src, eid in incoming:
            new_src = winner if src == loser else src
            if new_src == winner:
                continue
            _, _, edge = self._edges[eid]
            if not self._has_edge_kind(new_src, winner, edge.kind):
                self.add_edge(NodeId(new_src), NodeId(winner), edge)
            # Remove loser's original edge from all adjacency lists
            self._edges.pop(eid, None)
            self._in[winner] = [(s, e) for s, e in self._in[winner] if e != eid]
            self._out[src] = [(d, e) for d, e in self._out[src] if e != eid]

        for dst, eid in outgoing:
            new_dst = winner if dst == loser else dst
            if winner == new_dst:
                continue
            _, _, edge = self._edges[eid]
            if not self._has_edge_kind(winner, new_dst, edge.kind):
                self.add_edge(NodeId(winner), NodeId(new_dst), edge)
            # Remove loser's original edge from all adjacency lists
            self._edges.pop(eid, None)
            self._out[winner] = [(d, e) for d, e in self._out[winner] if e != eid]
            self._in[dst] = [(s, e) for s, e in self._in[dst] if e != eid]

        loser_qn = self._nodes[loser].qualified_name
        del self._nodes[loser]
        self._by_qname.pop(loser_qn, None)
        self._out.pop(loser, None)
        self._in.pop(loser, None)
        return NodeId(winner)

    # ── Edge operations ──────────────────────────────────────────────

    def add_edge(self, src: NodeId, dst: NodeId, edge: Edge) -> EdgeId:
        """Add a directed edge, preserving self-loops and parallel edges."""
        eid = self._next_edge_id
        self._next_edge_id += 1
        self._edges[eid] = (src.value, dst.value, edge)
        self._out[src.value].append((dst.value, eid))
        self._in[dst.value].append((src.value, eid))
        self._invalidate_native_snapshot()
        return EdgeId(eid)

    def remove_edges_by_id(self, ids: list[EdgeId]) -> None:
        """Remove a batch of edges by stable id."""
        for eid_obj in ids:
            eid = eid_obj.value
            if eid in self._edges:
                self._invalidate_native_snapshot()
                src, dst, _ = self._edges[eid]
                self._out[src] = [(d, e) for d, e in self._out[src] if e != eid]
                self._in[dst] = [(s, e) for s, e in self._in[dst] if e != eid]
                del self._edges[eid]

    def edge_by_id(self, id: EdgeId) -> tuple[int, int, Edge] | None:
        """Return (src, dst, edge) for an edge id."""
        return self._edges.get(id.value)

    # ── Lookup ───────────────────────────────────────────────────────

    def find_by_qname(self, qname: str) -> NodeId | None:
        idx = self._by_qname.get(qname)
        return NodeId(idx) if idx is not None else None

    def node(self, id: NodeId) -> Node | None:
        return self._nodes.get(id.value)

    def node_mut(self, id: NodeId) -> Node | None:
        return self._nodes.get(id.value)

    def node_by_value(self, idx: int) -> Node | None:
        """Look up a node by its integer index."""
        return self._nodes.get(idx)

    def edge(self, id: EdgeId) -> Edge | None:
        e = self._edges.get(id.value)
        return e[2] if e else None

    def edge_weight(self, id: EdgeId) -> Edge | None:
        """Return the Edge object for an edge id."""
        return self.edge(id)

    # ── Iteration ────────────────────────────────────────────────────

    def nodes(self) -> Iterator[tuple[NodeId, Node]]:
        # Node IDs are assigned monotonically and never reused, so dict
        # insertion order is identical to sorted ID order.  Loaders insert
        # rows in ID order (``ORDER BY id``), keeping the invariant intact.
        for idx, node in self._nodes.items():
            yield NodeId(idx), node

    def edges(self) -> Iterator[tuple[EdgeId, NodeId, NodeId, Edge]]:
        for eid, (src, dst, edge) in self._edges.items():
            yield EdgeId(eid), NodeId(src), NodeId(dst), edge

    def out_neighbors(self, id: NodeId) -> Iterator[tuple[NodeId, Edge]]:
        for dst, eid in self._out.get(id.value, []):
            yield NodeId(dst), self._edges[eid][2]

    def in_neighbors(self, id: NodeId) -> Iterator[tuple[NodeId, Edge]]:
        for src, eid in self._in.get(id.value, []):
            yield NodeId(src), self._edges[eid][2]

    # ── Counts ───────────────────────────────────────────────────────

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return len(self._edges)

    def in_degree_by_kind(self, kind: EdgeKind) -> dict[int, int]:
        """Count in-edges of ``kind`` per node, keyed by raw node index.

        One O(E) pass replaces repeated per-node ``in_neighbors`` scans in
        phases that only need the aggregate count.
        """
        counts: dict[int, int] = {}
        for dst, neighbors in self._in.items():
            total = 0
            for _src, eid in neighbors:
                if self._edges[eid][2].kind == kind:
                    total += 1
            if total:
                counts[dst] = total
        return counts

    # ── Merge ────────────────────────────────────────────────────────

    def merge(self, other: Graph) -> None:
        """Merge all nodes and edges from other into self.

        Nodes with matching qualified names are deduplicated. Edges are
        added with original semantics — duplicates skipped.
        """
        by_qname = self._by_qname
        remap: dict[int, NodeId] = {}

        # ``_by_qname`` is the same mapping the old implementation rebuilt
        # from scratch on every merge, so reuse it directly.
        for other_id, node in other._nodes.items():
            mapped_index = by_qname.get(node.qualified_name)
            remap[other_id] = (
                self.add_node(node) if mapped_index is None else NodeId(mapped_index)
            )

        # Existing (src, dst, kind) triples in one O(E) pass replace a
        # per-edge adjacency scan; new edges added during the merge are
        # recorded as they go so later duplicates are caught too.
        existing_kinds: set[tuple[int, int, EdgeKind]] = {
            (src, dst, edge.kind) for _, (src, dst, edge) in self._edges.items()
        }

        for _other_eid, (other_src, other_dst, edge) in other._edges.items():
            si = remap.get(other_src)
            di = remap.get(other_dst)
            if si is None or di is None or si == di:
                continue
            if (si.value, di.value, edge.kind) in existing_kinds:
                continue
            existing_kinds.add((si.value, di.value, edge.kind))
            self.add_edge(si, di, edge)

    # ── Internal helpers ─────────────────────────────────────────────

    def clone(self) -> Graph:
        """Deep-clone this graph."""
        import copy
        new = Graph()
        new._nodes = {idx: copy.deepcopy(node) for idx, node in self._nodes.items()}
        new._edges = {
            eid: (src, dst, copy.deepcopy(edge))
            for eid, (src, dst, edge) in self._edges.items()
        }
        new._by_qname = dict(self._by_qname)
        new._out = defaultdict(
            list,
            {src: list(neighbors) for src, neighbors in self._out.items()},
        )
        new._in = defaultdict(
            list,
            {dst: list(neighbors) for dst, neighbors in self._in.items()},
        )
        new._next_node_id = self._next_node_id
        new._next_edge_id = self._next_edge_id
        return new

    def edge_index(self, id: EdgeId) -> int | None:
        """Return the internal index of an edge by its stable id, or None."""
        if id.value in self._edges:
            return id.value
        return None

    def remove_edge(self, id: EdgeId) -> None:
        """Remove a single edge by its stable id."""
        eid = id.value
        if eid not in self._edges:
            return
        self._invalidate_native_snapshot()
        src, dst, _ = self._edges[eid]
        self._out[src] = [(d, e) for d, e in self._out[src] if e != eid]
        self._in[dst] = [(s, e) for s, e in self._in[dst] if e != eid]
        del self._edges[eid]

    def remove_edge_by_kind(self, src: NodeId, dst: NodeId, kind: EdgeKind) -> None:
        """Remove the first edge matching (src, dst, kind)."""
        src_idx, dst_idx = src.value, dst.value
        for d, eid in self._out.get(src_idx, []):
            if d == dst_idx:
                _, _, edge = self._edges[eid]
                if edge.kind == kind:
                    self._invalidate_native_snapshot()
                    self._out[src_idx] = [
                        (dd, ee) for dd, ee in self._out[src_idx] if ee != eid
                    ]
                    self._in[dst_idx] = [
                        (ss, ee) for ss, ee in self._in[dst_idx] if ee != eid
                    ]
                    del self._edges[eid]
                    return

    def _has_edge_kind(self, src: int, dst: int, kind: EdgeKind) -> bool:
        for d, eid in self._out.get(src, []):
            _, _, edge = self._edges[eid]
            if d == dst and edge.kind == kind:
                return True
        return False


def scope_key(graph: Graph, function_id: NodeId) -> str:
    """Stable identifier for the function scope owning scoped nodes.

    Variable/parameter/return nodes are qualified by the enclosing
    function's qualified name rather than its insertion-order integer
    NodeId, which would shift whenever node emission order changes
    (different extractor implementations, worker scheduling, file sets)
    and break QN stability across builds.
    """
    node = graph.node(function_id)
    if node is not None:
        return node.qualified_name
    return f"func::{function_id.value}"
