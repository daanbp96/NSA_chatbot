"""OpenAI embedding wrapper. The model is set by ``EMBEDDING_MODEL``; the API
key is read from ``OPENAI_API_KEY`` by the SDK."""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import lru_cache

from nsa_chatbot.config import EMBEDDING_MODEL


class Embedder:
    def __init__(self, model: str = EMBEDDING_MODEL):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set.")
        from openai import OpenAI

        self.model = model
        self._client = OpenAI()

    @property
    def model_id(self) -> str:
        # Stamped into the Chroma collection at build time; query() refuses a
        # mismatch (see store/index.py).
        return f"openai:{self.model}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 96):
            resp = self._client.embeddings.create(
                model=self.model, input=list(texts[i : i + 96])
            )
            out.extend(d.embedding for d in resp.data)
        return out


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()
