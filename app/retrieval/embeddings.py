"""Embedding backends behind one small interface.

sentence-transformers runs locally (free, no data leaves the server). The OpenAI-compatible backend
works with any provider that serves /embeddings. The hashing backend is deterministic and offline;
it exists for tests and CI, and the settings refuse it in production.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import re
from typing import Protocol

import numpy as np

from app.core.config import Settings


class Embedder(Protocol):
    model_name: str
    dim: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """E5 models expect 'query: ' and 'passage: ' prefixes; other models get the text as is."""

    def __init__(self, model_name: str, dim: int, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")
        self.dim = self.model.get_sentence_embedding_dimension() or dim
        if self.dim != dim:
            raise ValueError(f"{model_name} produces {self.dim}-d vectors but EMBEDDING_DIM is {dim}")
        self.batch_size = batch_size
        self._e5 = "e5" in model_name.lower()

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.astype(np.float32).tolist()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prepared = [f"passage: {t}" for t in texts] if self._e5 else texts
        return await asyncio.to_thread(self._encode, prepared)

    async def embed_query(self, text: str) -> list[float]:
        prepared = f"query: {text}" if self._e5 else text
        return (await asyncio.to_thread(self._encode, [prepared]))[0]


class OpenAICompatibleEmbedder:
    def __init__(self, settings: Settings) -> None:
        from openai import AsyncOpenAI

        self.model_name = settings.embedding_model
        self.dim = settings.embedding_dim
        key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else "not-needed"
        self.client = AsyncOpenAI(api_key=key, base_url=settings.llm_base_url, timeout=settings.llm_timeout_s)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), 96):
            response = await self.client.embeddings.create(model=self.model_name, input=texts[start : start + 96])
            out.extend(item.embedding for item in response.data)
        return out

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]


_TOKEN = re.compile(r"\w+", re.UNICODE)


class HashingEmbedder:
    """Feature hashing of word unigrams and bigrams into a fixed-size, L2-normalised vector."""

    def __init__(self, dim: int) -> None:
        self.model_name = f"hashing-{dim}"
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        tokens = _TOKEN.findall(text.lower())
        features = tokens + [f"{a} {b}" for a, b in itertools.pairwise(tokens)]
        vec = np.zeros(self.dim, dtype=np.float32)
        for feature in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            vec[index] += 1.0 if digest[4] & 1 else -1.0
        norm = np.linalg.norm(vec)
        return (vec / norm if norm else vec).tolist()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "hashing":
        return HashingEmbedder(settings.embedding_dim)
    if settings.embedding_backend == "openai":
        return OpenAICompatibleEmbedder(settings)
    return SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_dim)
