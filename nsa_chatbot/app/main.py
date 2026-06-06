"""Gradio app assembly + entry point.

Run with::

    python -m nsa_chatbot
    # or for dev mode with hot reload:
    gradio nsa_chatbot/app/main.py --watch-dirs nsa_chatbot

Two tabs:
  - **Chat** (:mod:`nsa_chatbot.app.chat_tab`): user-facing.
  - **Admin** (:mod:`nsa_chatbot.app.admin_tab`): operator controls.
"""

from __future__ import annotations

import gradio as gr

from nsa_chatbot.app.admin_tab import build_admin_tab, stats_fn
from nsa_chatbot.app.chat_tab import build_chat_tab

with gr.Blocks(title="NSA IDR Assistant") as demo:
    gr.Markdown("## NSA IDR Assistant")

    with gr.Tabs():
        with gr.Tab("Chat"):
            build_chat_tab()
        with gr.Tab("Admin"):
            status_md = build_admin_tab()

    demo.load(stats_fn, outputs=status_md)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1")
