"""Chat tab: user-facing Q&A wired to :func:`nsa_chatbot.chat.rag.answer`.

Streaming chatbot + jurisdiction dropdown + a foldable sources panel.
"""

from __future__ import annotations

import gradio as gr

from nsa_chatbot.chat.rag import answer

STATE_CHOICES = ["all", "federal", "IL", "CA", "NY", "NJ", "FL", "TN"]


def _normalize_state(choice: str) -> str | None:
    if choice == "all":
        return None
    if choice == "federal":
        return "federal"
    return choice.upper()


def chat_fn(message: str, history: list[dict], state: str):
    """Stream tokens into the chatbot; emit sources footer at end."""
    message = (message or "").strip()
    if not message:
        yield history, ""
        return

    jurisdiction = _normalize_state(state)
    history = history + [{"role": "user", "content": message}]
    yield history, ""

    history = history + [{"role": "assistant", "content": ""}]
    answer_text = ""
    sources_text = ""

    try:
        for kind, payload in answer(message, jurisdiction=jurisdiction):
            if kind == "token":
                answer_text += payload
                history[-1]["content"] = answer_text
                yield history, sources_text
            elif kind == "sources":
                sources_text = payload
                yield history, sources_text
    except Exception as exc:
        msg = str(exc)
        hint = ""
        if "Collection" in msg and "does not exist" in msg:
            hint = (
                "\n\n_The corpus pipeline (ingest → build) needs to run "
                "before the chat can answer. Open the Admin tab._"
            )
        history[-1]["content"] = f"_Error:_ {exc}{hint}"
        yield history, sources_text


def build_chat_tab() -> None:
    """Create and wire the Chat tab's components inside the active Blocks context."""
    state = gr.Dropdown(
        choices=STATE_CHOICES,
        value="all",
        label="Jurisdiction",
        info="`all` = no filter. Any state pulls that state + federal.",
    )

    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(
        placeholder="Ask a question about IDR / surprise billing…",
        show_label=False,
        autofocus=True,
    )
    with gr.Row():
        send = gr.Button("Send", variant="primary")
        clear = gr.Button("Clear")

    with gr.Accordion("Sources for last answer", open=False):
        sources = gr.Code(value="", language=None, lines=10)

    send.click(
        chat_fn,
        inputs=[msg, chatbot, state],
        outputs=[chatbot, sources],
    ).then(lambda: "", outputs=msg)

    msg.submit(
        chat_fn,
        inputs=[msg, chatbot, state],
        outputs=[chatbot, sources],
    ).then(lambda: "", outputs=msg)

    clear.click(lambda: ([], ""), outputs=[chatbot, sources])
