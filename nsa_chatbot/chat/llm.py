"""Streaming chat-completion wrapper. ANSWER_PROVIDER drives selection."""

from __future__ import annotations

import os
from collections.abc import Iterator

from nsa_chatbot.config import (
    ANSWER_MODEL_ANTHROPIC,
    ANSWER_MODEL_OPENAI,
    ANSWER_PROVIDER,
)


class LLM:
    def __init__(self, provider: str = ANSWER_PROVIDER, model: str | None = None):
        self.provider = provider.lower()
        if self.provider in {"anthropic", "claude"}:
            import anthropic

            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY not set.")
            self._client = anthropic.Anthropic()
            self.model = model or ANSWER_MODEL_ANTHROPIC
        elif self.provider in {"openai", "gpt"}:
            from openai import OpenAI

            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY not set.")
            self._client = OpenAI()
            self.model = model or ANSWER_MODEL_OPENAI
        else:
            raise ValueError(f"Unknown answer provider: {provider}")

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def stream(self, system: str, user: str) -> Iterator[str]:
        if self.provider in {"anthropic", "claude"}:
            with self._client.messages.stream(
                model=self.model,
                max_tokens=2000,
                system=system,
                messages=[{"role": "user", "content": user}],
            ) as s:
                yield from s.text_stream
        else:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                stream=True,
            )
            for event in resp:
                delta = event.choices[0].delta
                if delta and delta.content:
                    yield delta.content
