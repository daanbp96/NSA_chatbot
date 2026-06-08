"""Add-source tab: chat-driven discovery → approve → append to sources.yaml.

The agent (`ingest.discover`) searches whitelisted domains and proposes entries;
the user approves to append them (`ingest.registry.append_source`); when done,
"Fetch + rebuild" runs the existing `ingest()` + `build_index()`.
"""

from __future__ import annotations

import gradio as gr

from nsa_chatbot.ingest.discover import discover
from nsa_chatbot.ingest.domains import add_domain, read_domains, remove_domain
from nsa_chatbot.ingest.registry import append_source
from nsa_chatbot.ingest.run import ingest, preview_source
from nsa_chatbot.store.index import build_index

_NONE = "_No pending proposals._"


def _domain_add(raw: str):
    """Add a domain to the whitelist with visible feedback. ``add_domain`` no-ops
    on an unparseable host (bare TLD, junk), so detect that and say so."""
    before = read_domains()
    domains = add_domain(raw)
    raw = (raw or "").strip()
    if not raw:
        status = ""
    elif len(domains) > len(before):
        status = f"✓ Added `{domains[-1]}`."
    elif raw:
        status = (
            f"⚠️ `{raw}` isn't a valid host to add. Use a full domain like "
            "`tn.gov` (bare TLDs and `*` wildcards aren't accepted), or it's "
            "already in the list."
        )
    return gr.update(choices=domains, value=[]), "", status


def _domain_remove(selected: list[str]):
    domains = read_domains()
    removed = [d for d in (selected or []) if d in domains]
    for d in removed:
        domains = remove_domain(d)
    status = (
        "✓ Removed " + ", ".join(f"`{d}`" for d in removed) if removed
        else "_Check a domain first, then Remove selected._"
    )
    return gr.update(choices=domains, value=[]), status


def _merge_proposals(existing: list[dict], new: list[dict]) -> list[dict]:
    """Accumulate proposals across discovery turns, deduped by ``id`` (a later
    proposal with the same id replaces the earlier one, position preserved). This
    is what lets the operator build up several sources and approve them together —
    without it, each new turn (even a plain follow-up) overwrites the pending set.
    """
    by_id: dict[str, dict] = {p.get("id"): p for p in existing}
    for p in new:
        by_id[p.get("id")] = p
    return list(by_id.values())


def _proposal_dropdown(proposals: list[dict]):
    """Dropdown update listing pending proposal ids (first selected)."""
    ids = [p.get("id") for p in proposals]
    return gr.update(choices=ids, value=(ids[0] if ids else None))


def _fmt_proposals(proposals: list[dict]) -> str:
    if not proposals:
        return _NONE
    lines = ["**Proposed sources** — review, then Approve to add to `sources.yaml`:", ""]
    for p in proposals:
        lines.append(
            f"- **{p.get('citation', '?')}** "
            f"({p.get('jurisdiction')}/{p.get('kind')}, `{p.get('fetcher')}`)  \n"
            f"  id `{p.get('id')}` — {p.get('url', '(no url)')}"
        )
    return "\n".join(lines)


def discover_fn(
    message: str, chat_history: list[dict], agent_msgs: list[dict], pending: list[dict]
):
    """One discovery turn. New proposals are *merged* into ``pending`` (not
    overwritten), so the operator can gather several across turns and approve
    them together. A turn with no new proposal — or an error — leaves ``pending``
    untouched. Returns (chat, agent_msgs, proposals_md, pending, preview_dropdown).
    """
    message = (message or "").strip()
    if not message:
        return chat_history, agent_msgs, _fmt_proposals(pending), pending, _proposal_dropdown(pending)
    # The user bubble was already added by the pre-step (_accept) so the textbox
    # could clear immediately; here we only append the agent's reply.
    try:
        reply, proposals, new_msgs = discover(message, agent_msgs)
    except Exception as exc:
        # Keep any pending proposals; only append the error to the chat.
        return (
            chat_history + [{"role": "assistant", "content": f"_Error:_ {exc}"}],
            agent_msgs,
            _fmt_proposals(pending),
            pending,
            _proposal_dropdown(pending),
        )
    merged = _merge_proposals(pending, proposals)
    if not reply:
        reply = "Found a candidate — review the proposal below." if proposals else "(no response)"
    chat_history = chat_history + [{"role": "assistant", "content": reply}]
    return chat_history, new_msgs, _fmt_proposals(merged), merged, _proposal_dropdown(merged)


def approve_fn(proposals: list[dict]):
    """Append every pending proposal to sources.yaml. Returns
    (status, proposals_md, pending, preview_dropdown). Clears the pending set."""
    if not proposals:
        return "_Nothing to approve._", _NONE, [], _proposal_dropdown([])
    added, errors = [], []
    for p in proposals:
        try:
            append_source(p)
            added.append(p.get("id"))
        except ValueError as exc:
            errors.append(f"`{p.get('id')}`: {exc}")
        except Exception as exc:  # don't let one bad entry silently no-op the click
            errors.append(f"`{p.get('id')}`: unexpected error: {exc}")
    lines = []
    if added:
        n = len(added)
        lines.append(
            f"✅ **Added {n} source{'s' if n != 1 else ''} to `sources.yaml`:** "
            + ", ".join(f"`{i}`" for i in added)
        )
        lines.append("")
        lines.append("_Now click **Fetch + rebuild index** below to fetch and index them._")
    if errors:
        lines.append("**⚠️ Skipped:**")
        lines += [f"- {e}" for e in errors]
    return "\n".join(lines), _NONE, [], _proposal_dropdown([])


def preview_fn(proposals: list[dict], selected_id: str | None) -> str:
    """Fetch-preview a single proposal (the one picked in the dropdown, or the
    first if none selected), so multiple proposals don't crowd into one blob."""
    if not proposals:
        return "(nothing to preview)"
    chosen = next((p for p in proposals if p.get("id") == selected_id), proposals[0])
    head = preview_source(chosen)
    return f"### {chosen.get('id')} — {chosen.get('url', '(no url)')}\n{head}"


def dismiss_fn():
    return "_Dismissed._", _NONE, [], _proposal_dropdown([])


def rebuild_fn() -> str:
    try:
        r = ingest()
        n = build_index()
    except Exception as exc:
        return f"_Rebuild failed:_ `{exc}`"
    note = ""
    if r.failures:
        note = " _(some sources failed to fetch — see Admin for details)_"
    return (
        f"Ingest: **{len(r.succeeded)}** ok, **{len(r.failures)}** failed, "
        f"**{len(r.skipped)}** skipped. Index rebuilt with **{n}** chunks.{note}"
    )


def build_discover_tab() -> None:
    """Create and wire the Add-source tab inside the active Blocks context."""
    gr.Markdown("### Add a source")
    gr.Markdown(
        "_Describe what you need; I search trusted legal domains and propose sources. "
        "Approve to add them to `sources.yaml`, then rebuild._"
    )

    agent_state = gr.State([])   # anthropic-format discovery conversation
    pending = gr.State([])       # proposals awaiting approval

    chatbot = gr.Chatbot(height=360)
    msg = gr.Textbox(
        placeholder="e.g. 'Find the CMS guidance on the federal IDR process'",
        show_label=False,
        autofocus=True,
    )
    with gr.Row():
        send = gr.Button("Send", variant="primary")
        clear = gr.Button("Clear")

    proposals_md = gr.Markdown(_NONE)
    with gr.Row():
        # Pick which pending proposal to preview (avoids cramming several into one box).
        preview_select = gr.Dropdown(
            choices=[], label="Proposal to preview", scale=2, interactive=True
        )
        preview = gr.Button("Preview fetch", scale=1)
        approve = gr.Button("Approve & add", variant="primary", scale=1)
        dismiss = gr.Button("Dismiss", scale=1)
    preview_acc = gr.Accordion("Preview (what would be indexed)", open=False)
    with preview_acc:
        # Equal lines/max_lines = fixed-height box that scrolls internally
        # instead of growing with the (now longer) preview text.
        preview_code = gr.Code(value="", language=None, lines=14, max_lines=14)
    status = gr.Markdown()

    gr.Markdown("---")
    rebuild = gr.Button("Fetch + rebuild index")
    rebuild_status = gr.Markdown()

    with gr.Accordion("Search domains (the agent only searches these)", open=False):
        gr.Markdown("_Add by URL or host; check one + Remove selected to drop it._")
        domains_group = gr.CheckboxGroup(
            choices=read_domains(), value=[], label="Whitelisted domains"
        )
        with gr.Row():
            domain_input = gr.Textbox(
                label="Add domain (URL or host)",
                placeholder="e.g. https://advance.tn.gov or advance.tn.gov",
                scale=3,
            )
            add_domain_btn = gr.Button("Add", scale=1)
        remove_domain_btn = gr.Button("Remove selected", size="sm")
        domain_status = gr.Markdown()

    sent = gr.State("")  # stashes the submitted message so the box clears first

    def _accept(message: str, history: list[dict]):
        """Clear the textbox and echo the user bubble immediately, before the
        (slow) discovery call — matches the Chat tab's responsiveness."""
        message = (message or "").strip()
        if not message:
            return "", "", history
        return "", message, history + [{"role": "user", "content": message}]

    # Two-step: `_accept` runs first with queue=False so the textbox clears and
    # the user message appears the instant Enter/Send is pressed; then
    # `discover_fn` runs the agent from the stashed message.
    for trigger in (send.click, msg.submit):
        trigger(
            _accept,
            inputs=[msg, chatbot],
            outputs=[msg, sent, chatbot],
            queue=False,
        ).then(
            discover_fn,
            inputs=[sent, chatbot, agent_state, pending],
            outputs=[chatbot, agent_state, proposals_md, pending, preview_select],
        )

    preview.click(preview_fn, [pending, preview_select], [preview_code]).then(
        lambda: gr.update(open=True), outputs=preview_acc
    )
    approve.click(approve_fn, [pending], [status, proposals_md, pending, preview_select])
    dismiss.click(dismiss_fn, None, [status, proposals_md, pending, preview_select])
    clear.click(
        lambda: ([], [], _NONE, [], gr.update(choices=[], value=None)),
        outputs=[chatbot, agent_state, proposals_md, pending, preview_select],
    )
    rebuild.click(rebuild_fn, None, [rebuild_status])

    add_domain_btn.click(
        _domain_add, inputs=domain_input, outputs=[domains_group, domain_input, domain_status]
    )
    domain_input.submit(
        _domain_add, inputs=domain_input, outputs=[domains_group, domain_input, domain_status]
    )
    remove_domain_btn.click(
        _domain_remove, inputs=domains_group, outputs=[domains_group, domain_status]
    )
