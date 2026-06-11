"""Persistent chat history — ChatGPT-style saved conversations.

One JSON file per conversation under ``CONVERSATIONS_DIR`` (local-only, gitignored,
like ``corpus/``/``index/``). A conversation's ``messages`` is exactly the Gradio
``gr.Chatbot`` list — ``[{"role", "content"}, ...]`` — which is also the shape the
Anthropic ``messages[]`` API wants, so this format is reusable as-is if we later
move to multi-turn generation ("Option B").

Single-turn generation is unchanged: assistant ``content`` includes the folded-in
``<details>`` sources block. That's fine for display + the follow-up rewrite
(`chat.followup.rewrite_query` reads prior turns); a future multi-turn path can
re-derive the per-turn ``[S#]`` source set from that footer or by re-retrieval.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from nsa_chatbot.config import CONVERSATIONS_DIR

_TITLE_MAX = 48


def new_id() -> str:
    """A fresh conversation id."""
    return uuid.uuid4().hex


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _path(conv_id: str):
    return CONVERSATIONS_DIR / f"{conv_id}.json"


def _as_text(content) -> str:
    """Flatten a message's ``content`` to plain text. Gradio's messages format
    may hand back a string, a list of parts, or structured dicts — be tolerant."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                parts.append(p.get("text") or "")
        return " ".join(x for x in parts if x)
    if isinstance(content, dict):
        return content.get("text") or ""
    return ""


def _title_from(messages: list[dict]) -> str:
    """Derive a title from the first user message (trimmed)."""
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") == "user":
            text = " ".join(_as_text(m.get("content")).split())
            if text:
                return text[:_TITLE_MAX] + ("…" if len(text) > _TITLE_MAX else "")
    return "New chat"


def save(conv_id: str, messages: list[dict], *, title: str | None = None) -> dict:
    """Write (or overwrite) a conversation. Creates the dir on first save.
    Preserves ``created_at`` across saves; bumps ``updated_at``. Title is derived
    from the first user message unless one is given (or already stored)."""
    CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
    existing = load(conv_id)
    created = existing.get("created_at") if existing else None
    if title is None:
        title = (existing or {}).get("title") or _title_from(messages)
    conv = {
        "id": conv_id,
        "title": title or _title_from(messages),
        "created_at": created or _now(),
        "updated_at": _now(),
        "messages": messages or [],
    }
    _path(conv_id).write_text(
        json.dumps(conv, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return conv


def load(conv_id: str) -> dict | None:
    """Read one conversation, or None if it doesn't exist / is unreadable."""
    p = _path(conv_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def list_conversations() -> list[dict]:
    """Summaries (id, title, updated_at) of all saved conversations, newest first.
    Tolerates a missing dir (returns [])."""
    if not CONVERSATIONS_DIR.exists():
        return []
    rows: list[dict] = []
    for p in CONVERSATIONS_DIR.glob("*.json"):
        try:
            conv = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        rows.append({
            "id": conv.get("id") or p.stem,
            "title": conv.get("title") or "Untitled",
            "updated_at": conv.get("updated_at") or "",
        })
    rows.sort(key=lambda r: r["updated_at"], reverse=True)
    return rows


def delete(conv_id: str) -> None:
    """Remove a conversation file (no-op if absent)."""
    _path(conv_id).unlink(missing_ok=True)
