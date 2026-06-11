"""Cheap LLM-generated conversation titles (ChatGPT-style).

A short, specific title summarizing the chat's topic, generated once (on the
first turn) with the cheap model. Fail-safe: returns None on any error or empty
output, so the caller falls back to the truncated-first-message title.
"""

from __future__ import annotations

from functools import lru_cache

from nsa_chatbot.config import REWRITE_MODEL_ANTHROPIC
from nsa_chatbot.chat.history import _as_text
from nsa_chatbot.core.llm import LLM

_SYSTEM = (
    "You write a very short, specific title (3-6 words) summarizing the TOPIC of a "
    "chat, for a conversation list. Output ONLY the title: no quotes, no surrounding "
    "punctuation, no trailing period, no preamble. For a bare greeting or small talk "
    "with no real topic, output a single fitting word (e.g. 'Greeting')."
)

_MAX_TITLE = 60


@lru_cache(maxsize=1)
def _llm() -> LLM:
    return LLM(model=REWRITE_MODEL_ANTHROPIC)


def suggest_title(messages: list[dict]) -> str | None:
    """A short topic title from the first exchange, or None on failure/empty
    (so the caller falls back to the truncated first-message title)."""
    user_text = answer_text = ""
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        if not user_text and m.get("role") == "user":
            user_text = _as_text(m.get("content")).strip()
        elif not answer_text and m.get("role") == "assistant":
            answer_text = _as_text(m.get("content")).strip()
        if user_text and answer_text:
            break
    if not user_text:
        return None

    user = (
        f"User's first message: {user_text[:500]}\n"
        f"Assistant's reply (excerpt): {answer_text[:300]}\n\n"
        "Title:"
    )
    try:
        out = _llm().complete(_SYSTEM, user, max_tokens=16)
    except Exception:
        return None
    title = " ".join(out.split()).strip().strip("\"'").rstrip(".").strip()
    return title[:_MAX_TITLE] or None
