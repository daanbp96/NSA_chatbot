"""Front router: a cheap human/legal split before the expensive pipeline.

A single Haiku call classifies each fresh turn and, for the conversational
bucket, drafts the reply in the same round trip. This keeps greetings, "what can
you do", "what's in your corpus", thanks, and out-of-scope chit-chat off the
grounded Opus/Sonnet pipeline entirely (they get a human answer for ~$0.001),
and it tags substantive questions ``simple`` vs ``hard`` so the answer model can
be tiered (Sonnet by default, Opus for hard ones).

It deliberately does NOT decide corpus coverage: any in-domain surprise-billing /
NSA / IDR question routes to ``legal`` and goes through the grounded pipeline, so
a real corpus gap still produces the grounded refusal rather than a router guess.
On any error it fails safe to ``legal``/``hard`` (the most capable, grounded path).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from nsa_chatbot.config import MODEL_POLICY
from nsa_chatbot.core.llm import LLM

_SYSTEM = """You are the front-door router for the Clearest Health assistant — a
tool the sales team uses to answer questions about the federal No Surprises Act
(NSA) and US state surprise-billing / IDR (independent dispute resolution) law
(any state). Scope is by topic, not by a fixed list of states: a surprise-billing
question about any state is in scope — whether the corpus actually has that state
is decided later by retrieval. Users are salespeople, not lawyers, and cannot
access primary legal documents.

Classify the user's latest message into exactly one route:

- "conversational": greetings, thanks, small talk, questions about what you are
  or what you can do, questions about what topics/jurisdictions you cover ("what
  do you have in your corpus?"), and anything OUT OF SCOPE (not about surprise
  billing / the NSA / IDR / balance billing / out-of-network payment). For these,
  write a warm, brief, plain-language reply yourself. For out-of-scope messages,
  politely say what you do cover and invite an in-scope question. Never tell the
  user to go read or check a statute, regulation, or document.

- "legal": any substantive question about the NSA, surprise billing, balance
  billing, IDR/open negotiation, the qualifying payment amount, eligibility,
  out-of-network payment, or a specific state's surprise-billing law — EVEN IF
  you suspect it may not be covered. Do not answer these yourself; leave "reply"
  null. Another grounded step handles them.

For "legal" messages, also rate difficulty:
- "hard": compares or spans multiple states, involves federal-vs-state preemption,
  needs multi-provision or multi-step eligibility/strategy reasoning, or has
  ambiguous facts.
- "simple": a single-provision lookup, a direct definitional question, or one
  jurisdiction with a straightforward answer.
For "conversational" messages, difficulty is "simple".

Respond with ONLY a JSON object, no preamble, no code fences:
{"route": "conversational"|"legal", "difficulty": "simple"|"hard", "reply": <string for conversational, null for legal>}
"""


@dataclass
class RouteResult:
    route: str          # "conversational" | "legal"
    difficulty: str     # "simple" | "hard"
    reply: str | None   # human reply when conversational; None when legal


# Safe default: treat as a hard legal question → the most capable grounded path.
_FALLBACK = RouteResult(route="legal", difficulty="hard", reply=None)


@lru_cache(maxsize=1)
def _router() -> LLM:
    return LLM(model=MODEL_POLICY["router"])


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    return text.strip()


def route(message: str, history: list[dict] | None = None, *, max_turns: int = 4) -> RouteResult:
    """Classify ``message`` (in the context of recent ``history``) and, when
    conversational, draft the reply. Fails safe to legal/hard on any error."""
    message = (message or "").strip()
    if not message:
        return _FALLBACK

    lines: list[str] = []
    for turn in (history or [])[-max_turns:]:
        role = turn.get("role", "")
        content = turn.get("content") or ""
        if role == "assistant":
            content = content[:300]  # keep the call cheap
        lines.append(f"{role}: {content}")
    convo = ("Conversation so far:\n" + "\n".join(lines) + "\n\n") if lines else ""
    user = f"{convo}Latest message: {message}\n\nJSON:"

    try:
        raw = _router().complete(_SYSTEM, user, max_tokens=512)
        data = json.loads(_strip_fences(raw))
        route_val = data.get("route")
        if route_val not in ("conversational", "legal"):
            return _FALLBACK
        difficulty = data.get("difficulty")
        if difficulty not in ("simple", "hard"):
            difficulty = "hard" if route_val == "legal" else "simple"
        reply = data.get("reply")
        if route_val == "conversational" and not (reply and reply.strip()):
            # Classified as conversational but no usable reply — fall back to the
            # grounded path rather than emitting an empty message.
            return _FALLBACK
        return RouteResult(
            route=route_val,
            difficulty=difficulty,
            reply=reply.strip() if (route_val == "conversational" and reply) else None,
        )
    except Exception:
        return _FALLBACK
