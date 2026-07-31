"""Build local or remote embeddings for durable graph search indexes."""

from __future__ import annotations

import os

import httpx

from ..core.graph import Graph


def embeddable_texts(graph: Graph) -> dict[str, str]:
    return {
        node.qualified_name: node.source_text
        for _, node in graph.nodes()
        if node.source_text and node.qualified_name
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
