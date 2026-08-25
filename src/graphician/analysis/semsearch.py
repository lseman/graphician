"""Semantic search using local embeddings.

Uses sentence-transformers for local embedding computation.
Finds semantically similar nodes based on source text content.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from ..core.graph import Graph
from ..core.id import NodeId
from ..core.node import NodeKind

logger = logging.getLogger(__name__)


class EmbeddingIndex:
    """In-memory embedding index for semantic search.

    Stores normalized embeddings for nodes with source text.
    Uses cosine similarity for retrieval.
    """

    def __init__(self) -> None:
        self._embeddings: dict[int, list[float]] = {}
        self._texts: dict[int, str] = {}
        self._model: Any = None
        self._initialized = False

    def initialize(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._initialized = True
            logger.info("Initialized embedding model: %s", model_name)
        except ImportError:
            logger.warning("sentence-transformers not available. Skipping semantic search.")
            self._initialized = False

    def index_node(self, nid: NodeId, text: str) -> None:
        """Index a node's source text."""
        if not self._initialized:
            return
        self._texts[nid.value] = text
        embedding = self._model.encode(text, normalize_embeddings=True)
        self._embeddings[nid.value] = embedding.tolist()

    def index_graph(self, graph: Graph) -> None:
        """Index all nodes in the graph that have source text."""
        if not self._initialized:
            return

        nodes_with_text = [
            (nid, node)
            for nid, node in graph.nodes()
            if node.source_text and node.kind in (
                NodeKind.FUNCTION,
                NodeKind.METHOD,
                NodeKind.CLASS,
                NodeKind.FILE,
            )
        ]

        if not nodes_with_text:
            return

        # Batch encode for efficiency
        texts = [node.source_text or "" for _, node in nodes_with_text]
        nids = [nid.value for nid, _ in nodes_with_text]

        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")

        embeddings = self._model.encode(texts, normalize_embeddings=True)

        for nid, text, embedding in zip(nids, texts, embeddings, strict=True):
            self._embeddings[nid] = embedding.tolist()
            self._texts[nid] = text

        logger.info("Indexed %d nodes for semantic search", len(nids))

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[tuple[int, float]]:
        """Search for semantically similar nodes.

        Returns list of (node_id, similarity_score) sorted by score.
        """
        if not self._initialized or not self._embeddings:
            return []

        query_embedding = self._model.encode(query, normalize_embeddings=True)
        query_vec = query_embedding.tolist()

        scores: list[tuple[int, float]] = []
        for nid, vec in self._embeddings.items():
            sim = _cosine_similarity(query_vec, vec)
            scores.append((nid, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:limit]

    def clear(self) -> None:
        """Clear all indexed embeddings."""
        self._embeddings.clear()
        self._texts.clear()


def semantic_search(
    graph: Graph,
    query: str,
    embedding_index: EmbeddingIndex,
    limit: int = 10,
) -> dict[str, Any]:
    """Execute semantic search against the embedding index.

    Returns nodes most similar to the query based on source text.
    """
    results = embedding_index.search(query, limit=limit)

    search_results: list[dict[str, Any]] = []
    for nid, score in results:
        node = graph.node(NodeId(nid))
        if node:
            search_results.append({
                "qualified_name": node.qualified_name,
                "kind": node.kind.value,
                "name": node.name,
                "score": round(score, 4),
                "source_uri": node.source_uri,
            })

    return {
        "query": query,
        "results": search_results,
        "total": len(search_results),
    }


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
