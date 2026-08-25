"""Persistence layer exports.

Mirrors the Rust ``persistence/mod.rs`` re-exports.
"""

from .embeddings import (
    ExternalEmbeddingConfig,
    build_external_embeddings,
    build_local_embeddings,
    external_embedding_from_config,
    validate_config,
)
from .embeddings.local import (
    cosine_similarity,
    decode_embedding,
    embedding_source_text,
    semantic_embedding,
)
from .fts import FTSIndex, build_fts5_query
from .store import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL,
    EdgeIdentity,
    GraphStore,
    IncompatibleDatabaseError,
    edge_identity,
    parse_confidence,
)

__all__ = [
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_EMBEDDING_MODEL",
    "EdgeIdentity",
    "ExternalEmbeddingConfig",
    "FTSIndex",
    "GraphStore",
    "IncompatibleDatabaseError",
    "build_external_embeddings",
    "build_fts5_query",
    "build_local_embeddings",
    "cosine_similarity",
    "decode_embedding",
    "edge_identity",
    "embedding_source_text",
    "external_embedding_from_config",
    "parse_confidence",
    "semantic_embedding",
    "validate_config",
]

# Re-export type resolution from analysis for convenience
try:
    from graphician.analysis.type_resolution import resolve_type_placeholders  # noqa: F401
    __all__.append("resolve_type_placeholders")
except ImportError:
    pass
