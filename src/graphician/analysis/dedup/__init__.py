"""Entity deduplication pipeline."""

from __future__ import annotations

from .normalize import normalize_label, shannon_entropy, passes_entropy_gate
from .minhash import MinHash, shingle
from .lsh import LshIndex, lsh_candidate_pairs
from .similarity import jaro_winkler
from .union_find import UnionFind, deduplicate_nodes
from .types import DedupOptions, DedupResult

__all__ = [
    "normalize_label",
    "shannon_entropy",
    "passes_entropy_gate",
    "MinHash",
    "shingle",
    "LshIndex",
    "lsh_candidate_pairs",
    "jaro_winkler",
    "UnionFind",
    "deduplicate_nodes",
    "DedupOptions",
    "DedupResult",
]
