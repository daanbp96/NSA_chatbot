"""History-aware query rewriting.

A follow-up like "what about California?" has no content to embed. Before
retrieval, we rewrite it into a standalone search query using recent
conversation, with a cheap model (Haiku). Memory is **retrieval-only**: this
output feeds jurisdiction detection + retrieval; the answer generation stays
single-turn (each answer grounded in its own freshly-retrieved sources).
"""

from __future__ import annotations

from functools import lru_cache

from nsa_chatbot.config import REWRITE_MODEL_ANTHROPIC
from nsa_chatbot.core.llm import LLM

_SYSTEM = (
    "You rewrite the user's latest message into a single standalone search query "
    "for a legal-document retrieval system. Use the conversation to resolve "
    'pronouns, ellipsis, and references (e.g. "what about California?" becomes the '
    "full topic from the prior turn, scoped to California). If the latest message "
    "is already a complete standalone question, return it unchanged. Output ONLY "
    "the query text — no preamble, no quotes."
)


@lru_cache(maxsize=1)
def _rewriter() -> LLM:
    return LLM(model=REWRITE_MODEL_ANTHROPIC)


def rewrite_query(message: str, history: list[dict], *, max_turns: int = 4) -> str:
    """Rewrite ``message`` into a standalone query using recent ``history``.

    Returns ``message`` unchanged when there's no history (first turn) or on any
    error — so a Haiku hiccup degrades cleanly to today's behavior.
    """
    message = (message or "").strip()
    if not history or not message:
        return message

    lines: list[str] = []
    for turn in history[-max_turns:]:
        role = turn.get("role", "")
        content = turn.get("content") or ""
        if role == "assistant":
            content = content[:400]  # keep the rewrite call cheap
        lines.append(f"{role}: {content}")
    user = (
        f"Conversation so far:\n{chr(10).join(lines)}\n\n"
        f"Latest message: {message}\n\n"
        "Standalone query:"
    )
    try:
        out = _rewriter().complete(_SYSTEM, user, max_tokens=128).strip()
        return out or message
    except Exception:
        return message
