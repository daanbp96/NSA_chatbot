"""Append a discovered source to ``sources.yaml``, preserving the file's
comments and layout (ruamel round-trip). The full fetch + index still happens
later via the existing ``ingest()`` + ``build_index()`` — this only edits the
registry.
"""

from __future__ import annotations

from ruamel.yaml import YAML

from nsa_chatbot.config import CORPUS_DIR, SOURCES_YAML
from nsa_chatbot.core import us_states
from nsa_chatbot.ingest.schemas import SourceSpec

_yaml = YAML()  # round-trip mode preserves comments
_yaml.preserve_quotes = True
_yaml.indent(mapping=2, sequence=4, offset=2)  # match sources.yaml's style


def _load():
    """Parsed sources.yaml, or an empty mapping if the file doesn't exist yet.
    sources.yaml is local-only (untracked) — a fresh clone has none and builds it
    up through the Add source tab, so every reader must tolerate its absence.
    """
    if not SOURCES_YAML.exists():
        return {}
    with SOURCES_YAML.open() as fh:
        return _yaml.load(fh) or {}


def existing_ids() -> set[str]:
    """All source ids currently in sources.yaml (federal + every state)."""
    return _ids(_load())


def _ids(cfg) -> set[str]:
    ids: set[str] = set()
    for raw in cfg.get("federal") or []:
        if raw.get("id"):
            ids.add(raw["id"])
    for sources in (cfg.get("states") or {}).values():
        for raw in sources or []:
            if raw.get("id"):
                ids.add(raw["id"])
    return ids


def source_overview() -> list[dict]:
    """Every declared source with whether its corpus file exists (fetched).
    Index chunk counts are joined in by the caller (store.chunk_counts_by_source).
    """
    cfg = _load()
    rows: list[dict] = []
    for raw in cfg.get("federal") or []:
        rows.append(_row(raw, CORPUS_DIR / "federal" / f"{raw.get('id')}.txt"))
    for slug, sources in (cfg.get("states") or {}).items():
        for raw in sources or []:
            rows.append(_row(raw, CORPUS_DIR / "states" / slug / f"{raw.get('id')}.txt"))
    return rows


def _row(raw, path) -> dict:
    return {
        "id": raw.get("id"),
        "jurisdiction": raw.get("jurisdiction"),
        "citation": raw.get("citation"),
        "fetcher": raw.get("fetcher"),
        "in_corpus": path.exists(),
    }


def append_source(entry: dict) -> None:
    """Validate ``entry`` and append it under the right bucket in sources.yaml.

    Raises ``ValueError`` on a malformed or duplicate entry. Comments and
    formatting in the file are preserved.
    """
    try:
        spec = SourceSpec.from_dict(entry)  # raises on missing required fields
    except TypeError as exc:
        raise ValueError(f"invalid source entry: {exc}") from exc

    cfg = _load()

    if spec.id in _ids(cfg):
        raise ValueError(f"duplicate source id: {spec.id!r}")

    # Canonicalize the loosely-labeled jurisdiction here, at the write seam, so
    # the stored value matches what the chat dropdown / detection / retrieval
    # filter use. Discovery may propose "Colorado", "CO", "Tex." etc.
    canon = us_states.canonicalize(spec.jurisdiction)
    if canon is None:
        raise ValueError(
            f"unrecognized jurisdiction {spec.jurisdiction!r} — use 'federal' or a "
            "US state (full name or 2-letter code, e.g. 'Colorado' or 'CO')"
        )
    new_entry = {k: v for k, v in entry.items() if v is not None}
    new_entry["jurisdiction"] = canon  # persist the canonical form
    if canon == "federal":
        cfg.setdefault("federal", [])
        cfg["federal"].append(new_entry)
    else:
        slug = us_states.slug_for(canon)
        states = cfg.setdefault("states", {})
        states.setdefault(slug, [])
        states[slug].append(new_entry)

    with SOURCES_YAML.open("w") as fh:
        _yaml.dump(cfg, fh)
