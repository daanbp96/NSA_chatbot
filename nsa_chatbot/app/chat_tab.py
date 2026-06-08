"""Chat tab: user-facing Q&A wired to the retrieval pipeline.

Streaming chatbot + jurisdiction dropdown, plus a **jurisdiction guard**: if the
question names a state that disagrees with the dropdown, the bot asks which to
use; if the chosen state has no coverage, it refuses and offers a federal-only
answer. The pending clarification is carried across turns in a ``gr.State`` (the
chat is otherwise stateless).

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
from nsa_chatbot.chat.followup import rewrite_query
from nsa_chatbot.chat.rag import answer_agentic, retrieve, retrieve_split
from nsa_chatbot.chat.router import route
from nsa_chatbot.store.index import CollectionNotFoundError

STATE_CHOICES = ["all", "federal", "IL", "CA", "NY", "NJ", "FL", "TN"]


def _normalize_state(choice: str) -> str | None:
    if choice == "all":
        return None
    if choice == "federal":
        return "federal"
    return choice.upper()


def _label(jur: str | None) -> str:
    """Human label for a jurisdiction value (None = all)."""
    if jur is None:
        return "all jurisdictions"
    if jur == "federal":
        return "federal"
    return STATE_NAMES.get(jur, jur)


def _parse_choice(reply: str) -> str | None:
    """Interpret a reply to a conflict question → 'all' | 'federal' | a state
    code, or None if it can't be read as a jurisdiction choice.
    """
    low = (reply or "").strip().lower()
    if low in ("all", "everything", "both", "any", "all states"):
        return "all"
    if "federal" in low or low == "fed":
        return "federal"
    detected = detect_jurisdictions(reply)
    if len(detected) == 1:
        return next(iter(detected))
    return None


def _resolve_fresh(detected: set[str], dd: str | None):
    """Decide for a fresh question: ('conflict', None) or ('answer', effective)."""
    if not detected:
        return ("answer", dd)  # no signal from text → dropdown wins (today's behavior)
    if len(detected) == 1:
        x = next(iter(detected))
        if dd is None or dd == x:
            return ("answer", x)  # 'all' + state → scope to the state; or they agree
        return ("conflict", None)  # dropdown is a different state, or 'federal'
    return ("conflict", None)  # 2+ states named


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
    history: list[dict], question: str, effective: str | None,
    difficulty: str = "simple",
):
    """Shared tail: retrieve → coverage gate → stream. Yields (history, pending)
    tuples; the sources footer is appended onto the assistant message inline.
    Sets pending only to ask the federal-fallback question. ``difficulty``
    ("simple"/"hard") selects the tiered answer model.
    """
    try:
        if effective and effective != "federal":
            # Specific state: retrieve the state's own material separately from
            # federal so federal volume can't crowd it out. No relevant state
            # material -> offer federal rather than answering the state question
            # from federal law silently.
            state_chunks, federal_chunks = retrieve_split(question, effective)
            if not state_chunks:
                if federal_chunks:
                    name = STATE_NAMES.get(effective, effective)
                    history[-1]["content"] = (
                        f"I didn't find {name}-specific material for that question. "
                        f"Answer based on **federal** law instead? (yes/no)"
                    )
                    yield history, {"kind": "offer_federal", "question": question}
                    return
                history[-1]["content"] = "I don't have that in my corpus."
                yield history, None
                return
            chunks = state_chunks + federal_chunks  # state first (SYSTEM_PROMPT rule 3)
        else:
            chunks = retrieve(question, jurisdiction=effective)
            if not chunks:
                history[-1]["content"] = "I don't have that in my corpus."
                yield history, None
                return
    except CollectionNotFoundError:
        history[-1]["content"] = (
            "_The corpus pipeline (ingest → build) needs to run before the "
            "chat can answer. Open the Admin tab._"
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
            question, jurisdiction=effective, seed_chunks=chunks, difficulty=difficulty
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
                # collapsible block, so the provenance travels with the message.
                history[-1]["content"] = answer + _sources_block(payload)
                yield history, None
    except Exception as exc:
        # Append under the partial answer instead of clobbering it.
        note = f"_Error:_ {exc}"
        history[-1]["content"] = f"{answer}\n\n{note}" if answer else note
        yield history, None


def chat_fn(message: str, history: list[dict], state: str, pending: dict | None):
    """Stream the answer, mediated by the jurisdiction guard. Yields
    (history, pending); ``pending`` carries a clarification across turns. Used
    sources are appended into each assistant message, not a separate output.
    """
    message = (message or "").strip()
    if not message:
        yield history, pending  # keep any pending clarification
        return

    prior = history  # turns before this message — used for follow-up rewriting
    history = history + [{"role": "user", "content": message}]
    yield history, pending
    history = history + [{"role": "assistant", "content": ""}]

    # 1. Resolve a clarification asked on the previous turn.
    if pending and pending.get("kind") == "conflict":
        choice = _parse_choice(message)
        if choice is not None:
            effective = None if choice == "all" else choice
            # A conflict is inherently multi-jurisdiction → treat as hard.
            yield from _stream_answer(history, pending["question"], effective, "hard")
            return
        pending = None  # unreadable → treat this message as a fresh question
    elif pending and pending.get("kind") == "offer_federal":
        if is_affirmative(message):
            # Reframe as federal-only — otherwise the still-state-worded question
            # makes the grounded model refuse ("no <State> in my corpus") even
            # though relevant federal sources were retrieved.
            fed_q = (
                "Answer under federal law (the No Surprises Act) only, ignoring "
                f"state-specific rules: {pending['question']}"
            )
            yield from _stream_answer(history, fed_q, "federal", "simple")
        else:
            history[-1]["content"] = (
                "Okay — rephrase, or pick a jurisdiction from the selector, whenever you like."
            )
            yield history, None
        return

    # 2. Front router: a cheap Haiku call decides whether this is small talk /
    # meta / out-of-scope (answer like a human, no retrieval, no Opus) or a
    # substantive legal question (run the grounded pipeline). It also tags legal
    # questions simple/hard for answer-model tiering. Fails safe to legal/hard.
    routed = route(message, prior)
    if routed.route == "conversational":
        history[-1]["content"] = routed.reply
        yield history, None
        return

    # 3. Legal question. Rewrite a follow-up into a standalone query using prior
    # turns (history-aware retrieval); detection, retrieval, and the answer all
    # run on the rewritten query. The displayed user turn keeps the original text.
    question = rewrite_query(message, prior)
    detected = detect_jurisdictions(question)
    dd = _normalize_state(state)
    kind, effective = _resolve_fresh(detected, dd)
    # Hard if the router judged it so, or the question spans >=2 states (a
    # deterministic multi-jurisdiction signal) → escalate to the Opus tier.
    difficulty = "hard" if (routed.difficulty == "hard" or len(detected) >= 2) else "simple"

    if kind == "conflict":
        states = sorted(detected)
        if len(states) >= 2:
            opts = " or ".join(f'"{s}"' for s in states)
            ask = (
                f"Your question mentions more than one state ({', '.join(_label(s) for s in states)}). "
                f"Which should I use — reply {opts}, or \"federal\"?"
            )
        else:
            x = states[0]
            ask = (
                f"Your question mentions {_label(x)}, but the selector is set to "
                f"{_label(dd)}. Which should I use — reply \"{x}\" or \"{dd}\"?"
            )
        history[-1]["content"] = ask
        yield history, {"kind": "conflict", "question": question}
        return

    yield from _stream_answer(history, question, effective, difficulty)


def build_chat_tab() -> None:
    """Create and wire the Chat tab's components inside the active Blocks context."""
    state = gr.Dropdown(
        choices=STATE_CHOICES,
        value="all",
        label="Jurisdiction",
        info="`all` = no filter. Any state pulls that state + federal.",
    )
    # Holds a pending clarification (conflict / offer_federal) across turns.
    pending = gr.State(None)

    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(
        placeholder="Ask a question about IDR / surprise billing…",
        show_label=False,
        autofocus=True,
    )
    with gr.Row():
        send = gr.Button("Send", variant="primary")
        clear = gr.Button("Clear")

    # Holds the just-submitted message so the visible textbox can be cleared
    # *before* the (slow, streaming) answer runs.
    sent = gr.State("")

    def _accept(message: str):
        """Clear the textbox immediately and stash the message for chat_fn."""
        return "", message

    # Two-step wiring: `_accept` runs first with queue=False so the textbox
    # clears the instant Enter/Send is pressed (not queued behind generation),
    # stashing the message into `sent`; then `chat_fn` streams from `sent`.
    for trigger in (send.click, msg.submit):
        trigger(
            _accept,
            inputs=[msg],
            outputs=[msg, sent],
            queue=False,
        ).then(
            chat_fn,
            inputs=[sent, chatbot, state, pending],
            outputs=[chatbot, pending],
        )

    clear.click(lambda: ([], None), outputs=[chatbot, pending])
