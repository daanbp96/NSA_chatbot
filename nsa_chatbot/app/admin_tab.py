"""Admin tab: operator controls — live index stats + buttons to run the
corpus pipeline (ingest + rebuild) without touching the terminal.
"""

from __future__ import annotations

import gradio as gr

from nsa_chatbot.ingest.run import ingest
from nsa_chatbot.store.index import build_index, stats


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


def build_admin_tab() -> gr.Markdown:
    """Create and wire the Admin tab. Returns the status markdown component so
    the caller can hook it to ``demo.load`` for an initial stats render.
    """
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
    build_btn.click(build_fn, outputs=build_md).then(stats_fn, outputs=status_md)

    return status_md
