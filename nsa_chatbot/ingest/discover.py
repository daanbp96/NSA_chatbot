"""Source-discovery agent.

A conversational agent that helps find primary-source documents and proposes
``sources.yaml`` entries. It searches the open web to locate authoritative
sources, but may only *propose* a source from a domain on the trusted whitelist;
for a good source on an off-list domain it first proposes the domain
(``propose_domain``) for the user's approval. It writes nothing — proposals are
returned to the UI: the UI adds approved domains (``ingest.domains.add_domain``)
and approved sources (``ingest.registry.append_source``), then the user rebuilds
via the existing ``ingest()`` + ``build_index()``.
"""

from __future__ import annotations

import os

import anthropic

from nsa_chatbot.config import MODEL_POLICY
from nsa_chatbot.ingest.domains import read_domains


def _system(whitelist: list[str]) -> str:
    """System prompt, rebuilt per call so the live whitelist (Admin-tab editable)
    is always reflected."""
    listed = "\n".join(f"  - {d}" for d in whitelist) or "  (none yet)"
    return f"""You help a legal-research team find primary-source documents to add
to a No Surprises Act / surprise-billing corpus: the federal NSA plus ANY US
state's surprise-billing / balance-billing / out-of-network dispute-resolution
law the user asks about.

You can search the open web to locate authoritative sources. These domains are
already TRUSTED (pre-approved):
{listed}

Rules:
- Propose ONLY primary sources: statutes, regulations, or official agency
  guidance, published on a government / official domain (e.g. a state
  legislature, a state insurance department, .gov agencies). NEVER propose
  law-firm blogs, news articles, summaries, or commercial aggregators.
- A `propose_source` call is allowed ONLY when the source's domain is on the
  trusted list above. If you find a good primary source on a domain that is NOT
  on the list, do NOT propose the source yet — instead call `propose_domain`
  with the host and a short reason (why it's authoritative and what you expect
  to find there). Once the user approves it, the domain becomes trusted and you
  can `propose_source` from it on the next turn.
- You may call `propose_domain` for several hosts and `propose_source` for
  several documents; proposals accumulate for the user's review. When a request
  clearly calls for multiple sources, make a separate call for each in the same
  turn rather than one at a time.
- `id` is kebab-case and descriptive (e.g. `co-doi-oon-arbitration`,
  `cms-idr-overview`). `jurisdiction` is `federal` or the US state (its name or
  2-letter code, e.g. `Colorado` or `CO`). `kind` is statute, regulation, or
  guidance.
- Infer `fetcher`: an eCFR URL -> `ecfr`; a `.pdf` URL -> `pdf`; otherwise
  `html`. For `ecfr`, include the `url` AND set `title`, `part`, and (if the
  page is a single section) `section_prefix`, read from the eCFR URL, e.g.
  `.../title-45/.../part-149/.../section-149.110` -> title 45, part 149,
  section_prefix "149.110".
- BE CONCISE. Do not narrate your search process or internal steps (no "Let me
  search…", "the search returned no results"). Reply only with the result: a
  one-line confirmation of what you propose, or a brief clarifying question.
- If the request is vague or out of scope (e.g. "find me a random article"),
  don't just refuse — proactively suggest 2-3 specific in-scope primary sources
  you could add (name the citation), or name a state whose coverage is thin, and
  ask which to pursue.
"""


_PROPOSE_TOOL = {
    "name": "propose_source",
    "description": (
        "Propose one primary-source document to add to the corpus registry. "
        "Call when you've found a specific authoritative page whose domain is on "
        "the trusted whitelist and you're confident it is in scope."
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

_PROPOSE_DOMAIN_TOOL = {
    "name": "propose_domain",
    "description": (
        "Suggest adding an authoritative domain to the search whitelist, when a "
        "relevant primary source lives on a domain that is not yet trusted. The "
        "user must approve before it's added; after approval you can propose "
        "sources from it. Use a bare host (e.g. 'doi.colorado.gov')."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "host": {
                "type": "string",
                "description": "bare host to add, e.g. 'doi.colorado.gov'",
            },
            "reason": {
                "type": "string",
                "description": "why this domain is authoritative and what primary source(s) you expect to find there",
            },
        },
        "required": ["host", "reason"],
    },
}


def _web_search_tool() -> dict:
    # Open web search (no allowed_domains) so the agent can locate authoritative
    # sources anywhere; the whitelist instead gates which domains it may PROPOSE
    # sources from (enforced by the system prompt + propose_domain approval).
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


def discover(
    user_message: str, history: list[dict] | None = None
) -> tuple[str, list[dict], list[dict], list[dict]]:
    """Run one discovery turn.

    Returns ``(reply_text, source_proposals, domain_proposals, messages)``: the
    assistant's text, any ``propose_source`` entries (each a dict ready for
    ``append_source``), any ``propose_domain`` suggestions (``{host, reason}``,
    ready for ``add_domain``), and the updated message list to thread forward.
    """
    client = _client()
    messages: list[dict] = list(history or [])
    messages.append({"role": "user", "content": user_message})

    tools = [_web_search_tool(), _PROPOSE_TOOL, _PROPOSE_DOMAIN_TOOL]
    proposals: list[dict] = []
    domain_proposals: list[dict] = []
    resp = None
    for _ in range(_MAX_ITERS):
        resp = client.messages.create(
            model=MODEL_POLICY["discover"],
            max_tokens=4000,
            system=_system(read_domains()),
            tools=tools,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "pause_turn":
            continue  # server-side web_search paused; resume

        if resp.stop_reason == "tool_use":
            tool_results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                if block.name == "propose_source":
                    proposals.append(dict(block.input))
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Recorded — awaiting the user's approval.",
                    })
                elif block.name == "propose_domain":
                    domain_proposals.append(dict(block.input))
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Domain suggestion recorded — awaiting the user's approval.",
                    })
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            break  # surface proposals / reply to the user

        break  # end_turn

    reply = "".join(
        b.text for b in (resp.content if resp else []) if b.type == "text"
    )
    return reply, proposals, domain_proposals, messages
