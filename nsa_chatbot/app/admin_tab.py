"""Source overview tab: a read-only view of the corpus registry.

Shows how many **sources** are declared and how many are actually indexed (per
jurisdiction), plus a per-source table of fetched/indexed status. It deliberately
does NOT report chunk counts or expose operator buttons — discovery, fetch +
rebuild, and the search-domain whitelist all live on the Add-source tab.
"""

from __future__ import annotations

from collections import Counter

import gradio as gr

from nsa_chatbot.ingest.registry import source_overview
from nsa_chatbot.store.index import chunk_counts_by_source


def source_overview_fn() -> str:
    """Render source counts + a per-source fetched/indexed table (no chunks)."""
    rows = source_overview()
    counts = chunk_counts_by_source()  # used only as an "is it indexed?" signal

    def is_indexed(rid: str) -> bool:
        return counts.get(rid, 0) > 0

    total = len(rows)
    indexed = sum(1 for r in rows if is_indexed(r["id"]))
    by_jur = Counter(r["jurisdiction"] for r in rows)

    lines = [
        f"**{total} sources** declared · **{indexed} indexed** "
        f"· **{total - indexed} not yet indexed**",
        "",
        "**Sources by jurisdiction:**",
    ]
    for j, n in sorted(by_jur.items()):
        lines.append(f"- `{j}`: {n}")

    lines += [
        "",
        "| source | jurisdiction | citation | fetched | indexed |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        fetched = "✓" if r["in_corpus"] else "✗"
        if is_indexed(r["id"]):
            ix = "✓"
        elif r["fetcher"] == "skip":
            ix = "✗ — hand-supply a file"  # skip = never auto-fetched
        else:
            ix = "✗ ⚠️"  # declared but not indexed → needs fetch + rebuild
        lines.append(
            f"| `{r['id']}` | {r['jurisdiction']} | {r.get('citation') or '?'} "
            f"| {fetched} | {ix} |"
        )
    return "\n".join(lines)


def build_admin_tab() -> gr.Markdown:
    """Create the Source-overview tab. Returns the overview markdown component so
    the caller can hook it to ``demo.load`` for an initial render.
    """
    gr.Markdown("### Source overview")
    gr.Markdown(
        "_Every source declared in `sources.yaml`: whether it's been fetched to "
        "`corpus/` and indexed for retrieval. **⚠️ = declared but not indexed** "
        "(add via the Add-source tab, then Fetch + rebuild there)._"
    )
    overview_md = gr.Markdown(source_overview_fn())
    refresh_btn = gr.Button("Refresh", size="sm")
    refresh_btn.click(source_overview_fn, outputs=overview_md)
    return overview_md
