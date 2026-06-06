"""Gradio frontend for the NSA chatbot.

Run with::

    python -m nsa_chatbot
    # or for dev mode with hot reload:
    gradio nsa_chatbot/app.py --watch-dirs nsa_chatbot

Two tabs:
  - **Chat**: user-facing. Wraps :func:`nsa_chatbot.chat.rag.answer` with a
    streaming chatbot, jurisdiction dropdown, and a foldable sources panel.
  - **Admin**: operator controls. Live index stats + buttons to run the
    corpus pipeline (ingest + rebuild) without touching the terminal.
"""

from __future__ import annotations

import gradio as gr

from nsa_chatbot.chat.rag import answer
from nsa_chatbot.ingest.run import ingest
from nsa_chatbot.store.index import build_index, stats

STATE_CHOICES = ["all", "federal", "IL", "CA", "NY", "NJ", "FL", "TN"]


def _normalize_state(choice: str) -> str | None:
    if choice == "all":
        return None
    if choice == "federal":
        return "federal"
    return choice.upper()


# ---------- Chat handler ---------------------------------------------------


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


# ---------- Admin handlers -------------------------------------------------


def stats_fn() -> str:
    """Render current index stats as markdown."""
    s = stats()
    if "error" in s:
        return f"_No index yet:_ `{s['error']}`. Run ingest and rebuild."
    lines = [f"**Total chunks:** {s['count']}", "", "**By jurisdiction:**"]
    for j, n in sorted(s["jurisdictions"].items()):
        lines.append(f"- `{j}`: {n}")
    return "\n".join(lines)


def ingest_fn(only_ids_text: str) -> str:
    """Run the ingest step. Comma-separated source ids; blank = all."""
    only_ids: set[str] | None = None
    if only_ids_text and only_ids_text.strip():
        only_ids = {s.strip() for s in only_ids_text.split(",") if s.strip()}
    try:
        result = ingest(only_ids=only_ids)
    except Exception as exc:
        return f"_Ingest failed:_ `{exc}`"

    label = f"`{sorted(only_ids)}`" if only_ids else "all sources in `sources.yaml`"
    lines = [
        f"**Ingest complete** ({label}).",
        "",
        f"- Succeeded: **{len(result.succeeded)}**",
        f"- Failed: **{len(result.failures)}**",
        f"- Skipped: **{len(result.skipped)}**",
    ]
    if result.failures:
        lines.append("")
        lines.append("**Failures:**")
        for f in result.failures:
            lines.append(f"- `{f.source_id}` -- {f.reason}")
    if result.warnings:
        lines.append("")
        lines.append("**Warnings:**")
        for w in result.warnings:
            lines.append(f"- {w}")
    if result.skipped:
        skipped_list = ", ".join(f"`{s}`" for s in result.skipped)
        lines.append("")
        lines.append(f"**Skipped (hand-maintained):** {skipped_list}")
    lines.append("")
    lines.append("_New content isn't queryable until the index is rebuilt._")
    return "\n".join(lines)


def build_fn() -> str:
    """Re-chunk corpus and rebuild the Chroma index. Costs embedding API calls."""
    try:
        n = build_index()
    except Exception as exc:
        return f"_Build failed:_ `{exc}`"
    return f"Index rebuilt with **{n}** chunks."


# ---------- UI -------------------------------------------------------------


with gr.Blocks(title="NSA IDR Assistant") as demo:
    gr.Markdown("## NSA IDR Assistant")

    with gr.Tabs():
        with gr.Tab("Chat"):
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

        with gr.Tab("Admin"):
            gr.Markdown("### Index status")
            status_md = gr.Markdown()
            refresh_btn = gr.Button("Refresh", size="sm")

            gr.Markdown("---")
            gr.Markdown("### Ingest sources")
            gr.Markdown(
                "_Fetches sources declared in `sources.yaml`, writes "
                "plain-text files under `corpus/`. Safe to re-run — files "
                "are overwritten in place._"
            )
            only_input = gr.Textbox(
                label="Source IDs",
                placeholder="leave blank for all, or e.g. 45-cfr-149, cms-idr-tips",
            )
            ingest_btn = gr.Button("Run ingest", variant="primary")
            ingest_md = gr.Markdown()

            gr.Markdown("---")
            gr.Markdown("### Rebuild index")
            gr.Markdown(
                "_Re-chunks everything in `corpus/`, re-embeds, replaces "
                "the Chroma collection. Costs OpenAI embedding tokens "
                "(~$0.01 per ~200 chunks)._"
            )
            build_btn = gr.Button("Rebuild index", variant="primary")
            build_md = gr.Markdown()

            # Wiring
            refresh_btn.click(stats_fn, outputs=status_md)
            ingest_btn.click(ingest_fn, inputs=only_input, outputs=ingest_md)
            build_btn.click(build_fn, outputs=build_md).then(
                stats_fn, outputs=status_md
            )
            demo.load(stats_fn, outputs=status_md)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1")
