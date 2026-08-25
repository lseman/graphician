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
        """Add a node. Duplicate qualified_name updates in place."""
        qn = node.qualified_name
        if qn in self._by_qname:
            idx = self._by_qname[qn]
            self._nodes[idx] = node
            return NodeId(idx)
        idx = self._next_node_id
        self._next_node_id += 1
        self._nodes[idx] = node
        self._by_qname[qn] = idx
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
        for idx, node in self._nodes.items():
            yield NodeId(idx), node

    def edges(self) -> Iterator[tuple[EdgeId, NodeId, NodeId, Edge]]:
        for eid, (src, dst, edge) in sorted(self._edges.items()):
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

    # ── Merge ────────────────────────────────────────────────────────

    def merge(self, other: Graph) -> None:
        """Merge all nodes and edges from other into self.

        Nodes with matching qualified names are deduplicated. Edges are
        added with original semantics — duplicates skipped.
        """
        own_qn: dict[str, NodeId] = {
            node.qualified_name: NodeId(index)
            for index, node in self._nodes.items()
        }
        remap: dict[int, NodeId] = {}

        for other_id, node in other.nodes():
            qn = node.qualified_name
            if qn in own_qn:
                mapped = own_qn[qn]
            else:
                mapped = self.add_node(node)
                own_qn[qn] = mapped
            remap[other_id.value] = mapped

        for _, src, dst, edge in other.edges():
            si = remap.get(src.value)
            di = remap.get(dst.value)
            if si is None or di is None:
                continue
            if si == di:
                continue
            if not self._has_edge_kind(si.value, di.value, edge.kind):
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

    def _has_edge_kind(self, src: int, dst: int, kind: EdgeKind) -> bool:
        for d, eid in self._out.get(src, []):
            _, _, edge = self._edges[eid]
            if d == dst and edge.kind == kind:
                return True
        return False
