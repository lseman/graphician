"""LSH index for MinHash signatures."""

from __future__ import annotations

from ..._extract import dedup_candidate_pairs as _native_candidate_pairs
from ...core.id import NodeId
from ...core.node import Node
from .minhash import MinHash, shingle
from .types import DedupOptions


class LshIndex:
    """LSH index for MinHash signatures.

    One hash table per band. Maps band signature -> list of node IDs.
    """

    def __init__(self, num_bands: int, row_length: int) -> None:
        self.num_bands = num_bands
        self.row_length = row_length
        self.tables: list[dict[tuple[int, ...], list[NodeId]]] = [
            {} for _ in range(num_bands)
        ]

    def add(self, signature: MinHash, node_id: NodeId) -> None:
        """Add a MinHash signature with its node ID."""
        sig = signature.signature
        for band in range(self.num_bands):
            start = band * self.row_length
            end = start + self.row_length
            if end > len(sig):
                continue
            band_sig = tuple(sig[start:end])
            self.tables[band].setdefault(band_sig, []).append(node_id)

    def get_candidates(self, signature: MinHash) -> set[NodeId]:
        """Find candidate node IDs that share at least one band."""
        candidates: set[NodeId] = set()
        sig = signature.signature
        for band in range(self.num_bands):
            start = band * self.row_length
            end = start + self.row_length
            if end > len(sig):
                continue
            band_sig = tuple(sig[start:end])
            ids = self.tables[band].get(band_sig)
            if ids:
                candidates.update(ids)
        return candidates


def lsh_candidate_pairs(
    nodes: list[Node],
    node_ids: list[NodeId],
    options: DedupOptions,
) -> list[tuple[NodeId, NodeId, float]]:
    """Run MinHash/LSH to find candidate pairs.

    Returns list of (node_id_a, node_id_b, jaccard_estimate) for pairs
    sharing at least one LSH band with Jaccard >= threshold.
    """
    if _native_candidate_pairs is not None:
        native_pairs = _native_candidate_pairs(
            [(node_id.value, node.name) for node, node_id in zip(nodes, node_ids, strict=True)],
            options.shingle_size,
            options.num_permutations,
            options.num_bands,
            options.row_length,
            options.jaccard_threshold,
        )
        return [(NodeId(left), NodeId(right), score) for left, right, score in native_pairs]

    lsh = LshIndex(options.num_bands, options.row_length)

    # Build MinHash signatures
    signatures: dict[NodeId, MinHash] = {}
    for node, nid in zip(nodes, node_ids, strict=True):
        shingles = shingle(node.name, options.shingle_size)
        signatures[nid] = MinHash.from_iter(shingles, options.num_permutations)

    # Add all signatures to LSH index
    for nid, sig in signatures.items():
        lsh.add(sig, nid)

    # Find candidate pairs
    pairs: list[tuple[NodeId, NodeId, float]] = []
    pair_seen: set[tuple[NodeId, NodeId]] = set()

    for id_a, sig_a in signatures.items():
        candidates = lsh.get_candidates(sig_a)
        for id_b in candidates:
            if id_b == id_a:
                continue
            pair = (
                (id_a, id_b)
                if id_a.value < id_b.value
                else (id_b, id_a)
            )
            if pair in pair_seen:
                continue
            pair_seen.add(pair)
            sig_b = signatures.get(id_b)
            if sig_b is None:
                continue
            jaccard = sig_a.jaccard(sig_b)
            if jaccard >= options.jaccard_threshold:
                pairs.append((id_a, id_b, jaccard))

    return pairs
