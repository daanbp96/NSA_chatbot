"""Append a discovered source to ``sources.yaml``, preserving the file's
comments and layout (ruamel round-trip). The full fetch + index still happens
later via the existing ``ingest()`` + ``build_index()`` — this only edits the
registry.
"""

from __future__ import annotations

from ruamel.yaml import YAML

from nsa_chatbot.config import CORPUS_DIR, SOURCES_YAML
from nsa_chatbot.ingest.schemas import SourceSpec

# State code -> the `states:` bucket slug used in sources.yaml / the corpus dir.
_STATE_SLUG = {
    "CA": "california",
    "IL": "illinois",
    "NY": "new-york",
    "NJ": "new-jersey",
    "FL": "florida",
    "TN": "tennessee",
}

_yaml = YAML()  # round-trip mode preserves comments
_yaml.preserve_quotes = True
_yaml.indent(mapping=2, sequence=4, offset=2)  # match sources.yaml's style


def existing_ids() -> set[str]:
    """All source ids currently in sources.yaml (federal + every state)."""
    with SOURCES_YAML.open() as fh:
        cfg = _yaml.load(fh) or {}
    return _ids(cfg)


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
    with SOURCES_YAML.open() as fh:
        cfg = _yaml.load(fh) or {}
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

    with SOURCES_YAML.open() as fh:
        cfg = _yaml.load(fh) or {}

    if spec.id in _ids(cfg):
        raise ValueError(f"duplicate source id: {spec.id!r}")

    new_entry = {k: v for k, v in entry.items() if v is not None}
    if spec.jurisdiction == "federal":
        cfg.setdefault("federal", [])
        cfg["federal"].append(new_entry)
    else:
        slug = _STATE_SLUG.get(spec.jurisdiction)
        if slug is None:
            raise ValueError(f"unknown jurisdiction {spec.jurisdiction!r}")
        states = cfg.setdefault("states", {})
        states.setdefault(slug, [])
        states[slug].append(new_entry)

    with SOURCES_YAML.open("w") as fh:
        _yaml.dump(cfg, fh)
