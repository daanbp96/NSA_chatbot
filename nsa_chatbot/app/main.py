"""Gradio app assembly + entry point.

Run with::

    python -m nsa_chatbot
    # or for dev mode with hot reload:
    gradio nsa_chatbot/app/main.py --watch-dirs nsa_chatbot

Three tabs:
  - **Chat** (:mod:`nsa_chatbot.app.chat_tab`): user-facing Q&A.
  - **Add source** (:mod:`nsa_chatbot.app.discover_tab`): chat-driven source discovery,
    fetch + rebuild, and the search-domain whitelist.
  - **Source overview** (:mod:`nsa_chatbot.app.admin_tab`): read-only corpus registry status.
"""

from __future__ import annotations

import gradio as gr

from nsa_chatbot.app.admin_tab import build_admin_tab, source_overview_fn
from nsa_chatbot.app.chat_tab import build_chat_tab
from nsa_chatbot.app.discover_tab import build_discover_tab

with gr.Blocks(title="NSA IDR Assistant") as demo:
    gr.Markdown("## NSA IDR Assistant")

    with gr.Tabs():
        with gr.Tab("Chat"):
            build_chat_tab()
        with gr.Tab("Add source"):
            build_discover_tab()
        with gr.Tab("Source overview"):
            overview_md = build_admin_tab()

    demo.load(source_overview_fn, outputs=overview_md)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1")
