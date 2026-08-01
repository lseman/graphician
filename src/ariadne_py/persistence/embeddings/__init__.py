"""Build local or remote embeddings for durable graph search indexes.

Mirrors the Rust persistence/embeddings module including the
ExternalEmbeddingConfig dataclass and validate_config function.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

from ...core.graph import Graph


@dataclass
class ExternalEmbeddingConfig:
    """Embedding provider configuration.

    Mirrors the Rust ``ExternalEmbeddingConfig`` from
    ``persistence/embeddings/external.rs``.
    """

    provider: str
    """Provider name: ``openai-embedding``, ``google-embedding``, or ``ollama-embedding``."""
    api_key: str | None = None
    """API key (required for OpenAI and Google; optional for Ollama)."""
    base_url: str | None = None
    """Base URL for the API."""
    model: str | None = None
    """Model name."""
    dimension: int = 384
    """Embedding dimension (must match the model's output)."""


def validate_config(config: ExternalEmbeddingConfig) -> tuple[bool, str | None]:
    """Validate that an external embedding config has required fields.

    Returns ``(is_valid, error_message)``.
    Mirrors the Rust ``validate_config`` from ``external.rs``.
    """
    if config.dimension <= 0:
        return False, "embedding dimension must be > 0"

    if config.provider == "openai-embedding":
        if not config.api_key:
            return False, "openai-embedding requires api_key"
    elif config.provider == "google-embedding":
        if not config.api_key:
            return False, "google-embedding requires api_key"
    elif config.provider == "ollama-embedding":
        # Ollama can work without an API key (local server)
        pass
    else:
        return False, f"unsupported provider: {config.provider}"

    return True, None


def embeddable_texts(graph: Graph) -> dict[str, str]:
    return {
        node.qualified_name: node.source_text
        for _, node in graph.nodes()
        if node.source_text and node.qualified_name
    }


def external_embedding_from_config(
    config: ExternalEmbeddingConfig,
    text: str,
) -> list[float]:
    """Build a single external embedding from a config.

    Mirrors the Rust ``build_external_embedding`` function.
    """
    is_valid, error = validate_config(config)
    if not is_valid:
        raise ValueError(error or "invalid config")

    if text.strip().isspace() or not text.strip():
        return [0.0] * config.dimension

    with httpx.Client(timeout=30.0) as client:
        provider = config.provider.replace("-embedding", "")
        base = (config.base_url or _DEFAULT_BASE_URLS.get(provider, "http://localhost:11434")).rstrip("/")

        if provider == "openai":
            key = config.api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY or --api-key is required")
            model = config.model or "text-embedding-3-small"
            resp = client.post(
                f"{base}/embeddings",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "input": text, "encoding_format": "float"},
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            # Sort by index to handle any reordering
            data.sort(key=lambda d: d.get("index", 0))
            return data[0]["embedding"]

        elif provider == "google":
            key = config.api_key or os.getenv("GOOGLE_API_KEY")
            if not key:
                raise ValueError("GOOGLE_API_KEY or --api-key is required")
            model = config.model or "text-embedding-004"
            resp = client.post(
                f"{base}/models/{model}:embedContent?key={key}",
                headers={"Content-Type": "application/json"},
                json={"content": {"parts": [{"text": text}]}},
            )
            resp.raise_for_status()
            values = resp.json()["embedding"]["values"]
            return [float(v) for v in values]

        elif provider == "ollama":
            model = config.model or "nomic-embed-text"
            resp = client.post(
                f"{base}/api/embed",
                headers={"Content-Type": "application/json"},
                json={
                    "model": model,
                    "input": text,
                    "options": {"num_ctx": 2048},
                    "truncate": True,
                },
            )
            resp.raise_for_status()
            embeddings = resp.json()["embeddings"]
            if embeddings:
                return [float(v) for v in embeddings[0]]
            return [0.0] * config.dimension

        else:
            raise ValueError(f"unsupported provider: {config.provider}")


_DEFAULT_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta",
    "ollama": "http://localhost:11434",
}


def build_local_embeddings(
    graph: Graph, model: str = "all-MiniLM-L6-v2"
) -> dict[str, list[float]]:
    from sentence_transformers import SentenceTransformer

    texts = embeddable_texts(graph)
    if not texts:
        return {}
    vectors = SentenceTransformer(model).encode(
        list(texts.values()), normalize_embeddings=True
    )
    return {qname: vector.tolist() for qname, vector in zip(texts, vectors)}


def build_external_embeddings(
    graph: Graph,
    *,
    provider: str,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    batch_size: int = 64,
    timeout: float = 60.0,
) -> dict[str, list[float]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    items = list(embeddable_texts(graph).items())
    output: dict[str, list[float]] = {}
    with httpx.Client(timeout=timeout) as client:
        for offset in range(0, len(items), batch_size):
            batch = items[offset : offset + batch_size]
            vectors = _embed_batch(
                client,
                provider=provider,
                model=model,
                texts=[text for _, text in batch],
                api_key=api_key,
                base_url=base_url,
            )
            if len(vectors) != len(batch):
                raise RuntimeError("embedding provider returned an unexpected vector count")
            output.update((qname, vector) for (qname, _), vector in zip(batch, vectors))
    return output


def _embed_batch(
    client: httpx.Client,
    *,
    provider: str,
    model: str,
    texts: list[str],
    api_key: str | None,
    base_url: str | None,
) -> list[list[float]]:
    provider = provider.lower()
    if provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY or --api-key is required")
        response = client.post(
            (base_url or "https://api.openai.com/v1").rstrip("/") + "/embeddings",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "input": texts},
        )
        response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in data]
    if provider == "ollama":
        response = client.post(
            (base_url or "http://localhost:11434").rstrip("/") + "/api/embed",
            json={"model": model, "input": texts},
        )
        response.raise_for_status()
        values = response.json()["embeddings"]
        return [[float(value) for value in vector] for vector in values]
    if provider == "google":
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY or --api-key is required")
        root = (base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        response = client.post(
            f"{root}/models/{model}:batchEmbedContents?key={key}",
            json={
                "requests": [
                    {"model": f"models/{model}", "content": {"parts": [{"text": text}]}}
                    for text in texts
                ]
            },
        )
        response.raise_for_status()
        return [
            [float(value) for value in item["values"]]
            for item in response.json()["embeddings"]
        ]
    raise ValueError(f"unsupported embedding provider: {provider}")
