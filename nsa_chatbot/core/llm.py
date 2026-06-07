"""Streaming Anthropic answer model.

The project commits to Anthropic for answers; the model is set by
``ANSWER_MODEL_ANTHROPIC`` and the API key is read from ``ANTHROPIC_API_KEY``
by the SDK. (Embeddings are a separate concern — see ``core.embedder``.)
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator

import anthropic

from nsa_chatbot.config import ANSWER_MAX_TOKENS, ANSWER_MODEL_ANTHROPIC


class LLM:
    def __init__(self, model: str | None = None):
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set.")
        self.model = model or ANSWER_MODEL_ANTHROPIC
        self._client = anthropic.Anthropic()
        # Stop reason from the last stream() call: "max_tokens" means the answer
        # was cut off at ANSWER_MAX_TOKENS.
        self.last_stop_reason: str | None = None

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    def complete(self, system: str, user: str, max_tokens: int = 128) -> str:
        """Non-streaming single completion — for short, internal calls like
        follow-up query rewriting. Returns the concatenated text content.
        """
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")

    def run_tool_loop(
        self,
        system: str,
        user: str,
        tools: list[dict],
        dispatch: Callable[[str, dict], str],
        *,
        max_iters: int = 3,
        max_tokens: int = 1024,
        disable_parallel_tool_use: bool = True,
    ) -> Iterator[dict]:
        """Drive a non-streaming tool-use loop. The model issues tool calls;
        ``dispatch(name, input) -> str`` runs each and returns the text result
        fed back to the model. Yields each ``tool_use`` block's ``input`` dict
        as it happens (for progress display). Stops when the model ends its turn
        without a tool call, or after ``max_iters`` rounds of tool calls.

        With ``disable_parallel_tool_use`` (default), the model issues at most one
        tool call per round, so total tool calls are bounded by ``max_iters`` —
        without it a single round can fan out to many parallel calls.

        Generation is intentionally *not* produced here — this loop only decides
        what to gather; a separate step writes the grounded answer.
        """
        tool_choice = {"type": "auto", "disable_parallel_tool_use": disable_parallel_tool_use}
        messages: list[dict] = [{"role": "user", "content": user}]
        for _ in range(max_iters):
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )
            if msg.stop_reason != "tool_use":
                return
            messages.append({"role": "assistant", "content": msg.content})
            results: list[dict] = []
            for block in msg.content:
                if block.type != "tool_use":
                    continue
                yield dict(block.input) if block.input else {}
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": dispatch(block.name, dict(block.input or {})),
                    }
                )
            messages.append({"role": "user", "content": results})

    def stream(self, system: str, user: str) -> Iterator[str]:
        self.last_stop_reason = None
        with self._client.messages.stream(
            model=self.model,
            max_tokens=ANSWER_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as s:
            yield from s.text_stream
            self.last_stop_reason = s.get_final_message().stop_reason
