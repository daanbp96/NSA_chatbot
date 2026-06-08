"""Source-discovery agent.

A conversational agent that searches a whitelist of trusted legal domains
(Anthropic server-side ``web_search``) and proposes ``sources.yaml`` entries.
It does not write anything — proposals are returned to the UI for approval;
the UI calls :func:`nsa_chatbot.ingest.registry.append_source`, and the user
rebuilds via the existing ``ingest()`` + ``build_index()``.
"""

from __future__ import annotations

import os

import anthropic

from nsa_chatbot.config import MODEL_POLICY
from nsa_chatbot.ingest.domains import read_domains

_SYSTEM = """You help a legal-research team find primary-source documents to add
to a No Surprises Act / surprise-billing corpus (federal NSA + the states IL,
CA, NY, NJ, FL, TN). You can only search the whitelisted domains.

Rules:
- Propose ONLY primary sources: statutes, regulations, or official agency
  guidance, on the allowed domains. Never propose law-firm blogs, news, or
  summaries.
- When you've found a specific authoritative page and are confident it's in
  scope, call `propose_source` with a complete entry. Otherwise ask a brief
  clarifying question.
- `id` is kebab-case and descriptive (e.g. `ca-hsc-1371-9`, `cms-idr-overview`).
- Infer `fetcher`: an eCFR URL -> `ecfr`; a `.pdf` URL -> `pdf`; otherwise `html`.
  For `ecfr`, always include the `url` AND set `title`, `part`, and (if the page
  is a single section) `section_prefix` — read them from the eCFR URL, e.g.
  `.../title-45/.../part-149/.../section-149.110` -> title 45, part 149,
  section_prefix "149.110".
- `jurisdiction` is `federal` or one of CA/IL/NY/NJ/FL/TN. `kind` is statute,
  regulation, or guidance.
- You may propose more than one source. Proposals accumulate for the user's
  review, so when the request clearly calls for several (e.g. "add the CA and NY
  balance-billing statutes"), make a separate `propose_source` call for each in
  the same turn rather than one at a time.
- BE CONCISE. Do not narrate your search process or internal steps (no "Let me
  search…", "Let me fix the parsing", "the search returned no results", "the
  tool hit its limit"). Reply only with the result: a one-line confirmation of
  what you propose, or a brief clarifying question.
- If the request is vague or out of scope (e.g. "find me something" or "a random
  article"), don't just refuse — proactively suggest 2-3 specific in-scope
  primary sources you could add (name the citation), or name a state whose
  coverage is thin, and ask which to pursue.
"""

_PROPOSE_TOOL = {
    "name": "propose_source",
    "description": (
        "Propose one primary-source document to add to the corpus registry. "
        "Call when you've found a specific authoritative page on an allowed "
        "domain and are confident it is in scope."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "kebab-case unique id"},
            "jurisdiction": {
                "type": "string",
                "enum": ["federal", "CA", "IL", "NY", "NJ", "FL", "TN"],
            },
            "kind": {"type": "string", "enum": ["statute", "regulation", "guidance"]},
            "citation": {"type": "string", "description": "canonical legal citation"},
            "fetcher": {"type": "string", "enum": ["ecfr", "html", "pdf"]},
            "url": {
                "type": "string",
                "description": "authoritative URL — required for html/pdf; for ecfr, the ecfr.gov page URL",
            },
            "title": {"type": "integer", "description": "ecfr only: CFR title number (e.g. 45)"},
            "part": {"type": "integer", "description": "ecfr only: CFR part number (e.g. 149)"},
            "section_prefix": {
                "type": "string",
                "description": "ecfr only, optional: limit to a section, e.g. '149.110'",
            },
            "short": {"type": "string", "description": "short display label"},
            "notes": {"type": "string"},
        },
        "required": ["id", "jurisdiction", "kind", "citation", "fetcher"],
    },
}

def _web_search_tool() -> dict:
    # Built per call so Admin-tab edits to the whitelist take effect immediately.
    return {
        "type": "web_search_20260209",
        "name": "web_search",
        "allowed_domains": read_domains(),
        "max_uses": 10,
    }


_MAX_ITERS = 6


def _client() -> anthropic.Anthropic:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set.")
    return anthropic.Anthropic()


def discover(
    user_message: str, history: list[dict] | None = None
) -> tuple[str, list[dict], list[dict]]:
    """Run one discovery turn.

    Returns ``(reply_text, proposals, messages)``: the assistant's text, any
    structured source proposals it made this turn (each a dict ready for
    ``append_source``), and the updated message list to thread into the next call.
    """
    client = _client()
    messages: list[dict] = list(history or [])
    messages.append({"role": "user", "content": user_message})

    tools = [_web_search_tool(), _PROPOSE_TOOL]
    proposals: list[dict] = []
    resp = None
    for _ in range(_MAX_ITERS):
        resp = client.messages.create(
            model=MODEL_POLICY["discover"],
            max_tokens=4000,
            system=_SYSTEM,
            tools=tools,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "pause_turn":
            continue  # server-side web_search paused; resume

        if resp.stop_reason == "tool_use":
            tool_results = []
            for block in resp.content:
                if block.type == "tool_use" and block.name == "propose_source":
                    proposals.append(dict(block.input))
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Recorded — awaiting the user's approval.",
                    })
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            break  # surface proposals / reply to the user

        break  # end_turn

    reply = "".join(
        b.text for b in (resp.content if resp else []) if b.type == "text"
    )
    return reply, proposals, messages
