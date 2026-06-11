"""Add-source tab: chat-driven discovery → per-link approval → append to sources.yaml.

The agent (`ingest.discover`) searches the open web and proposes candidate primary
sources. The reviewer opens each link to verify it and **Approves or Rejects it
one by one** (`ingest.registry.append_source`); when done, "Fetch + rebuild" runs
the existing `ingest()` + `build_index()`. There is no domain whitelist — the
human is the trust gate. The post-rebuild report shows per-source chunk counts so
a link that fetched to little/garbage is obvious without a slow pre-approve preview.
"""

from __future__ import annotations

import gradio as gr

from nsa_chatbot.ingest.discover import discover
from nsa_chatbot.ingest.registry import append_source, source_overview
from nsa_chatbot.ingest.run import ingest
from nsa_chatbot.store.index import build_index, chunk_counts_by_source


def _merge_proposals(existing: list[dict], new: list[dict]) -> list[dict]:
    """Accumulate proposals across discovery turns, deduped by ``id`` (a later
    proposal with the same id replaces the earlier one, position preserved), so
    the reviewer can gather candidates over several turns before approving.
    """
    by_id: dict[str, dict] = {p.get("id"): p for p in existing}
    for p in new:
        by_id[p.get("id")] = p
    return list(by_id.values())


def discover_fn(
    message: str, chat_history: list[dict], agent_msgs: list[dict], pending: list[dict]
):
    """One discovery turn (generator, so the chat streams progress). Shows a live
    "searching…" bubble while the open-web agent works — which can take up to a
    minute — then replaces it with the reply. New candidates are *merged* into
    ``pending`` (not overwritten); a turn with no new candidate (or an error)
    leaves ``pending`` untouched. Yields (chat, agent_msgs, pending).
    """
    message = (message or "").strip()
    if not message:
        yield chat_history, agent_msgs, pending
        return

    # The user bubble was already added by the pre-step (_accept). Stream the
    # agent's progress into a placeholder assistant bubble so the tab never looks
    # frozen during the (slow) multi-round web search.
    reply, proposals, new_msgs = "", [], agent_msgs
    try:
        for kind, payload in discover(message, agent_msgs):
            if kind == "progress":
                yield (
                    chat_history + [{"role": "assistant", "content": f"🔎 {payload}"}],
                    agent_msgs,
                    pending,
                )
            else:  # ("result", (reply, proposals, messages))
                reply, proposals, new_msgs = payload
    except Exception as exc:
        yield (
            chat_history + [{"role": "assistant", "content": f"_Error:_ {exc}"}],
            agent_msgs,
            pending,
        )
        return

    merged = _merge_proposals(pending, proposals)
    if not reply:
        reply = (
            "Found candidates — review them below." if proposals
            else "I couldn't find primary sources to propose for that. Try naming a "
            "specific statute, agency, or state — or note that the state may have no "
            "surprise-billing law."
        )
    yield chat_history + [{"role": "assistant", "content": reply}], new_msgs, merged


def _approve_one(pid: str, proposals: list[dict]):
    """Append the one candidate with id ``pid`` to sources.yaml, then drop it from
    the pending list. On a validation error (duplicate id / bad jurisdiction) keep
    it and surface the message. Returns (pending, status)."""
    proposals = proposals or []
    target = next((p for p in proposals if p.get("id") == pid), None)
    if target is None:
        return proposals, "_That candidate is no longer pending._"
    try:
        append_source(target)
    except ValueError as exc:
        return proposals, f"⚠️ `{pid}`: {exc}"
    except Exception as exc:
        return proposals, f"⚠️ `{pid}`: unexpected error: {exc}"
    remaining = [p for p in proposals if p.get("id") != pid]
    return remaining, (
        f"✅ Added `{pid}` to `sources.yaml`. Click **Fetch + rebuild index** "
        "when you're done approving."
    )


def _reject_one(pid: str, proposals: list[dict]):
    """Drop the candidate with id ``pid`` from the pending list. Returns
    (pending, status)."""
    proposals = proposals or []
    return [p for p in proposals if p.get("id") != pid], f"Rejected `{pid}`."


def rebuild_fn(refetch_all: bool, progress=gr.Progress()) -> str:
    """Fetch sources + rebuild the index, with a live progress bar. By default
    fetches only sources missing from the corpus (the ones you just approved);
    tick *Re-fetch all* to re-download everything. The index is always rebuilt
    wholesale from the corpus on disk.

    Reports per-source chunk counts so a link that fetched to little/garbage (a
    JS-only page, a redirect, a non-PDF served as PDF) is obvious even though it
    didn't raise — this replaces the old slow pre-approve preview.
    """
    try:
        only = None if refetch_all else {
            r["id"] for r in source_overview() if not r["in_corpus"]
        }
        progress(0, desc="Fetching sources…")
        r = ingest(
            only_ids=only,
            on_progress=lambda i, n, sid: progress((i, n), desc=f"Fetching {sid}"),
        )
        count = build_index(
            on_progress=lambda done, n: progress((done, n), desc="Embedding + indexing"),
        )
    except Exception as exc:
        return f"_Rebuild failed:_ `{exc}`"

    counts = chunk_counts_by_source()
    scope = "all sources" if refetch_all else "new/missing sources"
    lines = [
        f"Fetched {scope}: **{len(r.succeeded)}** ok, **{len(r.failures)}** failed, "
        f"**{len(r.skipped)}** skipped. Index rebuilt with **{count}** chunks."
    ]
    if r.succeeded:
        lines += ["", "**Fetched sources (chunks indexed):**"]
        for sid in sorted(r.succeeded):
            c = counts.get(sid, 0)
            flag = "  ⚠️ fetched little — re-check the link" if c <= 1 else ""
            lines.append(f"- `{sid}`: {c}{flag}")
    if r.failures:
        lines += ["", "**⚠️ Failed to fetch:**"]
        lines += [f"- `{f.source_id}`: {f.reason}" for f in r.failures]
    if r.warnings:
        lines += ["", "**Warnings:**"]
        lines += [f"- {w}" for w in r.warnings]
    return "\n".join(lines)


def build_discover_tab() -> None:
    """Create and wire the Add-source tab inside the active Blocks context."""
    gr.Markdown("### Add a source")
    gr.Markdown(
        "_Describe what you need; I search the open web and propose candidate "
        "primary sources. **Open each link to verify it**, then Approve or Reject — "
        "approved sources are added to `sources.yaml`. When done, rebuild._"
    )

    agent_state = gr.State([])   # anthropic-format discovery conversation
    pending = gr.State([])       # source candidates awaiting per-link approval

    chatbot = gr.Chatbot(height=360)
    msg = gr.Textbox(
        placeholder="e.g. 'I need IDR documentation for the state of Texas'",
        show_label=False,
        autofocus=True,
    )
    with gr.Row():
        send = gr.Button("Send", variant="primary")
        clear = gr.Button("Clear")

    status = gr.Markdown()

    # Candidate list: one row per proposal, each with its own Approve/Reject.
    # Re-renders whenever `pending` changes (e.g. after an approve/reject).
    @gr.render(inputs=[pending])
    def render_candidates(proposals):
        if not proposals:
            gr.Markdown("_No candidates yet — describe what you need above._")
            return
        n = len(proposals)
        gr.Markdown(
            f"**{n} candidate{'s' if n != 1 else ''}** — open each link to verify, "
            "then Approve or Reject:"
        )
        for p in proposals:
            pid = p.get("id")
            with gr.Row():
                gr.Markdown(
                    f"**{p.get('citation', '?')}**  \n"
                    f"[{p.get('url', '(no url)')}]({p.get('url', '')}) — "
                    f"`{pid}` ({p.get('jurisdiction')}/{p.get('kind')}, `{p.get('fetcher')}`)"
                )
                approve_btn = gr.Button("✓ Approve", variant="primary", scale=0, min_width=120)
                reject_btn = gr.Button("✗ Reject", scale=0, min_width=120)
            # Default-arg binds this row's id into each handler's closure.
            approve_btn.click(
                lambda cur, _pid=pid: _approve_one(_pid, cur),
                inputs=[pending], outputs=[pending, status],
            )
            reject_btn.click(
                lambda cur, _pid=pid: _reject_one(_pid, cur),
                inputs=[pending], outputs=[pending, status],
            )

    gr.Markdown("---")
    refetch_all = gr.Checkbox(
        value=False,
        label="Re-fetch all sources (default: only new/missing ones)",
    )
    rebuild = gr.Button("Fetch + rebuild index")
    rebuild_status = gr.Markdown()

    sent = gr.State("")  # stashes the submitted message so the box clears first

    def _accept(message: str, history: list[dict]):
        """Clear the textbox and echo the user bubble immediately, before the
        (slow) discovery call — matches the Chat tab's responsiveness."""
        message = (message or "").strip()
        if not message:
            return "", "", history
        return "", message, history + [{"role": "user", "content": message}]

    # Two-step: `_accept` clears the box and shows the user bubble instantly
    # (queue=False); then `discover_fn` runs the agent from the stashed message.
    for trigger in (send.click, msg.submit):
        trigger(
            _accept,
            inputs=[msg, chatbot],
            outputs=[msg, sent, chatbot],
            queue=False,
        ).then(
            discover_fn,
            inputs=[sent, chatbot, agent_state, pending],
            outputs=[chatbot, agent_state, pending],
        )

    clear.click(
        lambda: ([], [], [], ""),
        outputs=[chatbot, agent_state, pending, status],
    )
    rebuild.click(rebuild_fn, [refetch_all], [rebuild_status])
