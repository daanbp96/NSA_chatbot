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
    return gr.update(choices=add_domain(raw), value=[]), ""


def _domain_remove(selected: list[str]):
    domains = read_domains()
    for d in selected or []:
        domains = remove_domain(d)
    return gr.update(choices=domains, value=[])


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


def discover_fn(message: str, chat_history: list[dict], agent_msgs: list[dict]):
    message = (message or "").strip()
    if not message:
        return chat_history, agent_msgs, _NONE, []
    # The user bubble was already added by the pre-step (_accept) so the textbox
    # could clear immediately; here we only append the agent's reply.
    try:
        reply, proposals, new_msgs = discover(message, agent_msgs)
    except Exception as exc:
        return (
            chat_history + [{"role": "assistant", "content": f"_Error:_ {exc}"}],
            agent_msgs,
            _NONE,
            [],
        )
    if not reply:
        reply = "Found a candidate — review the proposal below." if proposals else "(no response)"
    chat_history = chat_history + [{"role": "assistant", "content": reply}]
    return chat_history, new_msgs, _fmt_proposals(proposals), proposals


def approve_fn(proposals: list[dict]):
    if not proposals:
        return "_Nothing to approve._", _NONE, []
    added, errors = [], []
    for p in proposals:
        try:
            append_source(p)
            added.append(p.get("id"))
        except ValueError as exc:
            errors.append(f"`{p.get('id')}`: {exc}")
    lines = []
    if added:
        lines.append("**Added to `sources.yaml`:** " + ", ".join(f"`{i}`" for i in added))
        lines.append("")
        lines.append("_Done adding? Click **Fetch + rebuild** below (or use the Admin tab)._")
    if errors:
        lines.append("**Skipped:**")
        lines += [f"- {e}" for e in errors]
    return "\n".join(lines), _NONE, []


def preview_fn(proposals: list[dict]) -> str:
    if not proposals:
        return "(nothing to preview)"
    blocks = []
    for p in proposals:
        head = preview_source(p)
        blocks.append(f"### {p.get('id')} — {p.get('url', '(no url)')}\n{head}")
    return "\n\n========\n\n".join(blocks)


def dismiss_fn():
    return "_Dismissed._", _NONE, []


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
        preview = gr.Button("Preview fetch")
        approve = gr.Button("Approve & add", variant="primary")
        dismiss = gr.Button("Dismiss")
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
            inputs=[sent, chatbot, agent_state],
            outputs=[chatbot, agent_state, proposals_md, pending],
        )

    preview.click(preview_fn, [pending], [preview_code]).then(
        lambda: gr.update(open=True), outputs=preview_acc
    )
    approve.click(approve_fn, [pending], [status, proposals_md, pending])
    dismiss.click(dismiss_fn, None, [status, proposals_md, pending])
    clear.click(
        lambda: ([], [], _NONE, []),
        outputs=[chatbot, agent_state, proposals_md, pending],
    )
    rebuild.click(rebuild_fn, None, [rebuild_status])

    add_domain_btn.click(_domain_add, inputs=domain_input, outputs=[domains_group, domain_input])
    domain_input.submit(_domain_add, inputs=domain_input, outputs=[domains_group, domain_input])
    remove_domain_btn.click(_domain_remove, inputs=domains_group, outputs=domains_group)
