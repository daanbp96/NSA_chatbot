"""Streaming chat-completion wrapper. ANSWER_PROVIDER drives selection."""

from __future__ import annotations

import os
from collections.abc import Iterator

from nsa_chatbot.config import (
    ANSWER_MAX_TOKENS,
    ANSWER_MODEL_ANTHROPIC,
    ANSWER_MODEL_OPENAI,
    ANSWER_PROVIDER,
)


class LLM:
    def __init__(self, provider: str = ANSWER_PROVIDER, model: str | None = None):
        self.provider = provider.lower()
        # Raw provider stop/finish reason from the last stream() call:
        # "max_tokens"/"length" means the answer was cut off at ANSWER_MAX_TOKENS.
        self.last_stop_reason: str | None = None
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
        self.last_stop_reason = None
        if self.provider in {"anthropic", "claude"}:
            with self._client.messages.stream(
                model=self.model,
                max_tokens=ANSWER_MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}],
            ) as s:
                yield from s.text_stream
                self.last_stop_reason = s.get_final_message().stop_reason
        else:
            resp = self._client.chat.completions.create(
                model=self.model,
                max_tokens=ANSWER_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                stream=True,
            )
            for event in resp:
                choice = event.choices[0]
                delta = choice.delta
                if delta and delta.content:
                    yield delta.content
                if choice.finish_reason:
                    self.last_stop_reason = choice.finish_reason
