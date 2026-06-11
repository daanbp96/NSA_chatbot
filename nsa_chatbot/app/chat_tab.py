"""Chat tab: user-facing Q&A wired to the retrieval pipeline.

No jurisdiction dropdown — the scope is inferred from the question. A
*deterministic* in-text detector (`chat/jurisdiction.py`) finds any state named
in the (rewritten) question; the follow-up rewrite (`chat/followup.py`) carries
that state across turns, so once a state is mentioned, later follow-ups stay
scoped to it. Behavior by signal:

- one state named (or carried over) → that state's sources + federal, state first
- two+ states named → all of them + federal (e.g. a comparison)
- no state anywhere → answer from the federal NSA baseline, with a nudge to name
  a state for its specifics
- a named state with no relevant material → offer a federal-only answer (yes/no)

Used sources are folded into each assistant message as a collapsible
``<details>`` block (built from the ``("sources", footer)`` event the pipeline
emits at the end of generation), so every answer carries its own provenance and
scrollback shows what grounded each earlier turn.
"""

from __future__ import annotations

import gradio as gr

from nsa_chatbot.chat.jurisdiction import (
    STATE_NAMES,
    detect_jurisdictions,
    is_affirmative,
)
from nsa_chatbot.chat import history
from nsa_chatbot.chat.followup import rewrite_query
from nsa_chatbot.chat.titling import suggest_title
from nsa_chatbot.chat.rag import answer_agentic, retrieve, retrieve_split
from nsa_chatbot.chat.router import converse, route
from nsa_chatbot.store.index import CollectionNotFoundError

_THINKING = "💭 Thinking…"

_BASELINE_NUDGE = (
    "\n\n_That's the federal rule. Name a state (e.g. \"in Texas…\") and I'll give "
    "you its state-specific specifics._"
)


def _sources_block(footer: str) -> str:
    """Wrap the pipeline's ``[S#]`` footer in a collapsible block appended to the
    answer text, so each assistant message carries its own provenance. The footer
    goes in a fenced code block (monospaced, and keeps ``[S#]`` from being read as
    broken markdown links). Empty footer → nothing appended.
    """
    if not footer.strip():
        return ""
    n = sum(1 for ln in footer.splitlines() if ln.strip())
    label = "Source" if n == 1 else "Sources"
    return (
        f"\n\n<details><summary>{label} ({n})</summary>\n\n"
        f"```\n{footer}\n```\n\n</details>"
    )


def _stream_answer(
    history: list[dict], question: str, states: list[str],
    difficulty: str = "simple", federal_only: bool = False,
):
    """Shared tail: retrieve → coverage gate → stream. Yields (history, pending).

    - ``federal_only``: answer from federal sources only (the "answer federally
      instead?" path).
    - empty ``states``: no jurisdiction named → answer from the federal baseline
      and append a nudge to name a state.
    - non-empty ``states``: retrieve those states + federal (states first); if no
      state has relevant material, offer a federal-only answer via ``pending``.
    """
    baseline = not states and not federal_only
    try:
        if federal_only or not states:
            chunks = retrieve(question, jurisdiction="federal")
            jurisdiction: str | None = "federal"
            if not chunks:
                history[-1]["content"] = "I don't have that in my corpus."
                yield history, None
                return
        else:
            state_chunks, federal_chunks = retrieve_split(question, states)
            if not state_chunks:
                if federal_chunks:
                    names = ", ".join(STATE_NAMES.get(s, s) for s in states)
                    history[-1]["content"] = (
                        f"I didn't find {names}-specific material for that question. "
                        f"Answer based on **federal** law instead? (yes/no)"
                    )
                    yield history, {"kind": "offer_federal", "question": question}
                    return
                history[-1]["content"] = "I don't have that in my corpus."
                yield history, None
                return
            chunks = state_chunks + federal_chunks  # state first (SYSTEM_PROMPT rule 3)
            # One state → scope the agent's gap-fill searches to it; multiple →
            # leave broad (None) so it can fill from any of the named states.
            jurisdiction = states[0] if len(states) == 1 else None
    except CollectionNotFoundError:
        history[-1]["content"] = (
            "_The corpus pipeline (ingest → build) needs to run before the "
            "chat can answer. Open the Add source tab._"
        )
        yield history, None
        return
    except Exception as exc:
        history[-1]["content"] = f"_Error:_ {exc}"
        yield history, None
        return

    answer = ""
    progress = ""
    try:
        for kind, payload in answer_agentic(
            question, jurisdiction=jurisdiction, seed_chunks=chunks, difficulty=difficulty
        ):
            if kind == "progress":
                # Show "searching…" notes while the model gathers sources; the
                # first answer token replaces them.
                progress = f"{progress}\n{payload}" if progress else payload
                history[-1]["content"] = progress
                yield history, None
            elif kind == "token":
                answer += payload
                history[-1]["content"] = answer
                yield history, None
            elif kind == "sources":
                # Final event: fold the [S#] footer into this answer as a
                # collapsible block (and, for a baseline answer, the name-a-state
                # nudge), so the provenance travels with the message.
                note = _BASELINE_NUDGE if baseline else ""
                history[-1]["content"] = answer + note + _sources_block(payload)
                yield history, None
    except Exception as exc:
        # Append under the partial answer instead of clobbering it.
        note = f"_Error:_ {exc}"
        history[-1]["content"] = f"{answer}\n\n{note}" if answer else note
        yield history, None


def chat_fn(message: str, history: list[dict], pending: dict | None):
    """Stream the answer. Scope is inferred from the question (no dropdown).
    Yields (history, pending); ``pending`` carries a federal-fallback offer across
    turns. Used sources are appended into each assistant message.
    """
    message = (message or "").strip()
    if not message:
        yield history, pending  # keep any pending offer
        return

    prior = history  # turns before this message — used for follow-up rewriting
    history = history + [{"role": "user", "content": message}]
    yield history, pending
    # Instant feedback: a "thinking" placeholder so the bubble isn't blank during
    # the router call (and, for legal turns, the rewrite + retrieval before the
    # first "searching…" note). Every downstream path overwrites this content.
    history = history + [{"role": "assistant", "content": _THINKING}]
    yield history, pending

    # 1. Resolve a federal-fallback offer made on the previous turn.
    if pending and pending.get("kind") == "offer_federal":
        if is_affirmative(message):
            # Reframe as federal-only — otherwise the still-state-worded question
            # makes the grounded model refuse ("no <State> in my corpus") even
            # though relevant federal sources were retrieved.
            fed_q = (
                "Answer under federal law (the No Surprises Act) only, ignoring "
                f"state-specific rules: {pending['question']}"
            )
            yield from _stream_answer(history, fed_q, [], "simple", federal_only=True)
        else:
            history[-1]["content"] = (
                "Okay — rephrase, or name a state, whenever you like."
            )
            yield history, None
        return

    # 2. Front router: a cheap Haiku call decides whether this is small talk /
    # meta / out-of-scope (answer like a human, no retrieval, no Opus) or a
    # substantive legal question (run the grounded pipeline). It also tags legal
    # questions simple/hard for answer-model tiering. Fails safe to legal/hard.
    routed = route(message, prior)
    if routed.route == "conversational":
        # Stream the chit-chat reply (cheap Haiku) so it types out instead of
        # landing as a delayed lump; the first token replaces the placeholder.
        reply = ""
        try:
            for tok in converse(message, prior):
                reply += tok
                history[-1]["content"] = reply
                yield history, None
        except Exception:
            reply = ""
        if not reply.strip():
            history[-1]["content"] = (
                "Hi! I can help with the federal No Surprises Act and state "
                "surprise-billing / IDR questions — ask away."
            )
            yield history, None
        return

    # 3. Legal question. Rewrite a follow-up into a standalone query using prior
    # turns (so an implicit state carries over); detection, retrieval, and the
    # answer all run on the rewritten query. The displayed user turn keeps the
    # original text.
    question = rewrite_query(message, prior)
    states = sorted(detect_jurisdictions(question))
    # Hard if the router judged it so, or the question spans >=2 states (a
    # deterministic multi-jurisdiction signal) → escalate to the Opus tier.
    difficulty = "hard" if (routed.difficulty == "hard" or len(states) >= 2) else "simple"
    yield from _stream_answer(history, question, states, difficulty)


def _convo_choices(rows: list[dict]) -> list[tuple[str, str]]:
    """(label, value) pairs for the conversation list — title shown, id returned."""
    return [(r.get("title") or "Untitled", r["id"]) for r in rows]


def persist_fn(messages: list[dict], current_id: str | None):
    """Autosave the conversation after a completed turn (assigning an id on first
    save), and refresh the sidebar. Returns (current_id, conversation-list update).
    Setting the list's value programmatically does not fire its change event, so
    this doesn't trigger a reload."""
    if not messages:
        return current_id, gr.update()
    is_new = current_id is None
    cid = current_id or history.new_id()
    # Generate a smart title once, on the first turn; on later turns keep the
    # stored one. `title=None` makes history.save fall back to the truncated
    # first-message title (also the fallback if the titling call fails).
    title = suggest_title(messages) if is_new else None
    history.save(cid, messages, title=title)
    return cid, gr.update(choices=_convo_choices(history.list_conversations()), value=cid)


def new_chat_fn():
    """Start a fresh, unsaved conversation. Returns
    (chatbot, pending, current_id, sent, conversation-list update)."""
    return [], None, None, "", gr.update(value=None)


def select_fn(conv_id: str | None):
    """Load a saved conversation. Returns (chatbot, pending, current_id)."""
    if not conv_id:
        return [], None, None
    conv = history.load(conv_id)
    return (conv.get("messages", []) if conv else []), None, conv_id


def delete_current_fn(current_id: str | None):
    """Delete the open conversation and clear the view. Returns
    (chatbot, pending, current_id, conversation-list update)."""
    if current_id:
        history.delete(current_id)
    return [], None, None, gr.update(choices=_convo_choices(history.list_conversations()), value=None)


def build_chat_tab() -> None:
    """Create and wire the Chat tab's components inside the active Blocks context."""
    # Holds a pending federal-fallback offer across turns; the open conversation's
    # id (None = unsaved new chat); and the just-submitted message (so the textbox
    # can clear before the slow streaming answer runs).
    pending = gr.State(None)
    current_id = gr.State(None)
    sent = gr.State("")

    with gr.Row():
        # --- Sidebar: saved conversations (ChatGPT-style) ---
        with gr.Column(scale=1, min_width=210):
            new_btn = gr.Button("➕ New chat", variant="primary", size="sm")
            convo_list = gr.Radio(
                choices=_convo_choices(history.list_conversations()),
                value=None,
                label="Conversations",
                interactive=True,
            )
            delete_btn = gr.Button("🗑 Delete current", size="sm")

        # --- Main chat ---
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(
                placeholder="Ask about IDR / surprise billing — name a state (e.g. \"in Texas…\") for state-specific rules",
                show_label=False,
                autofocus=True,
            )
            send = gr.Button("Send", variant="primary")

    def _accept(message: str):
        """Clear the textbox immediately and stash the message for chat_fn."""
        return "", message

    # Send/submit: `_accept` clears the box (queue=False, instant) → `chat_fn`
    # streams the answer → `persist_fn` autosaves + refreshes the sidebar.
    for trigger in (send.click, msg.submit):
        trigger(
            _accept,
            inputs=[msg],
            outputs=[msg, sent],
            queue=False,
        ).then(
            chat_fn,
            inputs=[sent, chatbot, pending],
            outputs=[chatbot, pending],
        ).then(
            persist_fn,
            inputs=[chatbot, current_id],
            outputs=[current_id, convo_list],
        )

    new_btn.click(
        new_chat_fn,
        outputs=[chatbot, pending, current_id, sent, convo_list],
    )
    convo_list.change(
        select_fn,
        inputs=[convo_list],
        outputs=[chatbot, pending, current_id],
    )
    delete_btn.click(
        delete_current_fn,
        inputs=[current_id],
        outputs=[chatbot, pending, current_id, convo_list],
    )
