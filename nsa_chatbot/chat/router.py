"""Front router + small-talk responder.

A cheap Haiku call classifies each fresh turn as conversational vs legal (and,
for legal, tags difficulty). Classification is deliberately **tiny** — it returns
only ``{route, difficulty}`` (no drafted reply), so it's fast. For the
conversational bucket, :func:`converse` streams a warm reply in a separate call,
so chit-chat types out token-by-token instead of landing as a delayed lump.

This keeps greetings, "what can you do", "what's in your corpus", thanks, and
out-of-scope chit-chat off the grounded Opus/Sonnet pipeline entirely, and tags
substantive questions ``simple`` vs ``hard`` for answer-model tiering.

It deliberately does NOT decide corpus coverage: any in-domain surprise-billing /
NSA / IDR question routes to ``legal`` and goes through the grounded pipeline, so
a real corpus gap still produces the grounded refusal rather than a router guess.
On any error it fails safe to ``legal``/``hard`` (the most capable, grounded path).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

from nsa_chatbot.config import MODEL_POLICY
from nsa_chatbot.core.llm import LLM

_CLASSIFY_SYSTEM = """You are the front-door router for the Clearest Health assistant — a
tool the sales team uses to answer questions about the federal No Surprises Act
(NSA) and US state surprise-billing / IDR (independent dispute resolution) law
(any state). Scope is by topic, not by a fixed list of states: a surprise-billing
question about any state is in scope — whether the corpus actually has that state
is decided later by retrieval. Users are salespeople, not lawyers.

Classify the user's latest message into exactly one route:

- "conversational": greetings, thanks, small talk, questions about what you are
  or what you can do, questions about what topics/jurisdictions you cover, AND
  anything OUT OF SCOPE (not about surprise billing / the NSA / IDR / balance
  billing / out-of-network payment).
- "legal": any substantive question about the NSA, surprise billing, balance
  billing, IDR/open negotiation, the qualifying payment amount, eligibility,
  out-of-network payment, or a specific state's surprise-billing law — EVEN IF
  you suspect it may not be covered (retrieval decides coverage).

For "legal" messages, also rate difficulty:
- "hard": compares or spans multiple states, involves federal-vs-state preemption,
  needs multi-provision or multi-step eligibility/strategy reasoning, or has
  ambiguous facts.
- "simple": a single-provision lookup, a direct definitional question, or one
  jurisdiction with a straightforward answer.
For "conversational" messages, difficulty is "simple".

Respond with ONLY a JSON object, no preamble, no code fences:
{"route": "conversational"|"legal", "difficulty": "simple"|"hard"}
"""

_CONVERSE_SYSTEM = """You are the Clearest Health assistant for the internal sales team.
You help with the federal No Surprises Act (NSA) and US state surprise-billing / IDR
(independent dispute resolution) law for any state. Reply warmly, briefly (1-3
sentences), in plain language.

- For greetings, thanks, or small talk, respond naturally.
- For questions about what you are or what you cover, briefly explain that you answer
  surprise-billing / NSA / IDR questions grounded in primary legal sources, and invite one.
- If the message is OUT OF SCOPE (not about surprise billing / the NSA / IDR / balance
  billing / out-of-network payment), say so politely, name what you do cover, and invite an
  in-scope question.

Never tell the user to go read or check a statute, regulation, or document — they have no
access to source documents.
"""


@dataclass
class RouteResult:
    route: str          # "conversational" | "legal"
    difficulty: str     # "simple" | "hard"


# Safe default: treat as a hard legal question → the most capable grounded path.
_FALLBACK = RouteResult(route="legal", difficulty="hard")


@lru_cache(maxsize=1)
def _llm() -> LLM:
    """The shared Haiku instance used for both classification and chit-chat."""
    return LLM(model=MODEL_POLICY["router"])


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _context(history: list[dict] | None, max_turns: int) -> str:
    """Recent conversation as a compact prefix (assistant turns clipped)."""
    lines: list[str] = []
    for turn in (history or [])[-max_turns:]:
        role = turn.get("role", "")
        content = turn.get("content") or ""
        if role == "assistant":
            content = content[:300]  # keep the call cheap
        lines.append(f"{role}: {content}")
    return ("Conversation so far:\n" + "\n".join(lines) + "\n\n") if lines else ""


def route(message: str, history: list[dict] | None = None, *, max_turns: int = 4) -> RouteResult:
    """Classify ``message`` (in the context of recent ``history``). Returns only
    the route + difficulty (no reply); chit-chat replies come from
    :func:`converse`. Fails safe to legal/hard on any error.
    """
    message = (message or "").strip()
    if not message:
        return _FALLBACK
    user = f"{_context(history, max_turns)}Latest message: {message}\n\nJSON:"
    try:
        raw = _llm().complete(_CLASSIFY_SYSTEM, user, max_tokens=64)
        data = json.loads(_strip_fences(raw))
        route_val = data.get("route")
        if route_val not in ("conversational", "legal"):
            return _FALLBACK
        difficulty = data.get("difficulty")
        if difficulty not in ("simple", "hard"):
            difficulty = "hard" if route_val == "legal" else "simple"
        return RouteResult(route=route_val, difficulty=difficulty)
    except Exception:
        return _FALLBACK


def converse(message: str, history: list[dict] | None = None, *, max_turns: int = 4) -> Iterator[str]:
    """Stream a warm conversational reply (greetings / meta / out-of-scope), token
    by token, on the cheap Haiku model. Used only after :func:`route` returns
    ``conversational``.
    """
    message = (message or "").strip()
    user = f"{_context(history, max_turns)}Latest message: {message}"
    yield from _llm().stream(_CONVERSE_SYSTEM, user)
