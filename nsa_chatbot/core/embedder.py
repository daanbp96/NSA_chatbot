"""Embedding wrapper. EMBEDDING_PROVIDER + EMBEDDING_MODEL drive selection."""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import lru_cache

from nsa_chatbot.config import EMBEDDING_MODEL, EMBEDDING_PROVIDER


class Embedder:
    def __init__(
        self,
        provider: str = EMBEDDING_PROVIDER,
        model: str = EMBEDDING_MODEL,
    ):
        self.provider = provider.lower()
        self.model = model
        if self.provider == "openai":
            from openai import OpenAI

            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY not set.")
            self._client = OpenAI()
        elif self.provider in {"sentence-transformers", "st", "local"}:
            from sentence_transformers import SentenceTransformer

            self._client = SentenceTransformer(self.model)
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    @property
    def model_id(self) -> str:
        return f"{self.provider}:{self.model}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self.provider == "openai":
            out: list[list[float]] = []
            for i in range(0, len(texts), 96):
                resp = self._client.embeddings.create(
                    model=self.model, input=list(texts[i : i + 96])
                )
                out.extend(d.embedding for d in resp.data)
            return out
        # normalize_embeddings=True so local vectors are unit-length, matching
        # OpenAI's normalized output and the index's cosine space.
        return [
            list(map(float, v))
            for v in self._client.encode(list(texts), normalize_embeddings=True)
        ]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()
