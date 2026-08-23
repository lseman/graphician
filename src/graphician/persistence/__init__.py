"""Persistence layer exports.

Mirrors the Rust ``persistence/mod.rs`` re-exports.
"""

from .store import (
    GraphStore,
    IncompatibleDatabaseError,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL,
    EdgeIdentity,
    edge_identity,
    parse_confidence,
)
from .fts import FTSIndex, build_fts5_query
from .embeddings import (
    ExternalEmbeddingConfig,
    build_external_embeddings,
    build_local_embeddings,
    validate_config,
    external_embedding_from_config,
)
from .embeddings.local import (
    cosine_similarity,
    decode_embedding,
    embedding_source_text,
    semantic_embedding,
)

__all__ = [
    "GraphStore",
    "IncompatibleDatabaseError",
    "EdgeIdentity",
    "FTSIndex",
    "build_external_embeddings",
    "build_local_embeddings",
    "edge_identity",
    "parse_confidence",
    "cosine_similarity",
    "decode_embedding",
    "embedding_source_text",
    "semantic_embedding",
    "build_fts5_query",
    "ExternalEmbeddingConfig",
    "validate_config",
    "external_embedding_from_config",
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_EMBEDDING_MODEL",
]

# Re-export type resolution from analysis for convenience
try:
    from graphician.analysis.type_resolution import resolve_type_placeholders
    __all__.append("resolve_type_placeholders")
except ImportError:
    pass
