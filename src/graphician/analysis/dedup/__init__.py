"""Entity deduplication pipeline."""

from __future__ import annotations

from .lsh import LshIndex, lsh_candidate_pairs
from .minhash import MinHash, shingle
from .normalize import normalize_label, passes_entropy_gate, shannon_entropy
from .similarity import jaro_winkler
from .types import DedupOptions, DedupResult
from .union_find import UnionFind, deduplicate_nodes

__all__ = [
    "DedupOptions",
    "DedupResult",
    "LshIndex",
    "MinHash",
    "UnionFind",
    "deduplicate_nodes",
    "jaro_winkler",
    "lsh_candidate_pairs",
    "normalize_label",
    "passes_entropy_gate",
    "shannon_entropy",
    "shingle",
]
