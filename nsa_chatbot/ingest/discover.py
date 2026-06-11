"""Source-discovery agent.

A conversational agent that searches the open web for primary-source documents
and proposes ``sources.yaml`` entries. It writes nothing — proposals are returned
to the UI, where a human reviews each candidate link and approves it one by one
(``ingest.registry.append_source``); the user then rebuilds via the existing
``ingest()`` + ``build_index()``. The human is the trust gate: there is no domain
whitelist, so the agent is told to bias hard toward primary/official sources and
the reviewer makes the final call per link.
"""

from __future__ import annotations

import os

import anthropic

from nsa_chatbot.config import MODEL_POLICY

_SYSTEM = """You help a legal-research team find primary-source documents to add
to a No Surprises Act / surprise-billing corpus: the federal NSA plus ANY US
state's surprise-billing / balance-billing / out-of-network dispute-resolution
law the user asks about.

You search the open web. A human reviews every candidate you propose and decides
whether to keep it, so your job is to surface a good shortlist — not to be the
final gatekeeper.

Rules:
- Propose ONLY primary sources: statutes, regulations, or official agency
  guidance (e.g. a state legislature, a state insurance department, .gov
  agencies, official codes). Strongly avoid law-firm blogs, news articles,
  summaries, and commercial aggregators — prefer the authoritative original.
- When the user asks for material, propose SEVERAL candidates (a handful of the
  best links), each via a `propose_source` call, so the reviewer can pick. In
  your text reply, list them briefly with a one-line "why it's authoritative"
  for each. Make all the calls in the same turn rather than one at a time.
- `id` is kebab-case and descriptive (e.g. `tx-tdi-idr-faq`, `cms-idr-overview`).
  `jurisdiction` is `federal` or the US state (its name or 2-letter code, e.g.
  `Texas` or `TX`). `kind` is statute, regulation, or guidance.
- Infer `fetcher`: an eCFR URL -> `ecfr`; a `.pdf` URL -> `pdf`; otherwise
  `html`. For `ecfr`, include the `url` AND set `title`, `part`, and (if the
  page is a single section) `section_prefix`, read from the eCFR URL, e.g.
  `.../title-45/.../part-149/.../section-149.110` -> title 45, part 149,
  section_prefix "149.110".
- BE CONCISE. Don't narrate your search process. Reply only with the result: a
  short list of what you propose, or a brief clarifying question.
- If the request is vague or out of scope, don't just refuse — suggest 2-3
  specific in-scope primary sources you could add (name the citation) and ask
  which to pursue.
"""

_PROPOSE_TOOL = {
    "name": "propose_source",
    "description": (
        "Propose one primary-source document to add to the corpus registry. "
        "Call once per candidate; propose several per request so the reviewer "
        "can choose. The human approves or rejects each link."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "kebab-case unique id"},
            "jurisdiction": {
                "type": "string",
                "description": (
                    "'federal', or the US state as its name or 2-letter code "
                    "(e.g. 'Colorado' or 'CO'). Normalized to a canonical code on save."
                ),
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
    # Open web search (no allowed_domains): the agent locates authoritative
    # sources anywhere, and the human approves each proposed link.
    return {
        "type": "web_search_20260209",
        "name": "web_search",
        "max_uses": 10,
    }


_MAX_ITERS = 6


def _client() -> anthropic.Anthropic:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set.")
    return anthropic.Anthropic()


def discover(user_message: str, history: list[dict] | None = None):
    """Run one discovery turn, as a **generator** so the UI can show progress
    (open-web search over several rounds can take up to a minute).

    Yields ``("progress", note)`` before each model round, then exactly one final
    ``("result", (reply_text, proposals, messages))`` — the assistant's text, any
    ``propose_source`` entries (each a dict ready for ``append_source``), and the
    updated message list to thread into the next call.
    """
    client = _client()
    messages: list[dict] = list(history or [])
    messages.append({"role": "user", "content": user_message})

    tools = [_web_search_tool(), _PROPOSE_TOOL]
    proposals: list[dict] = []
    resp = None
    for step in range(_MAX_ITERS):
        yield ("progress", f"Searching the web for primary sources… (step {step + 1})")
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
    yield ("result", (reply, proposals, messages))
