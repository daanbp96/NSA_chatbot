"""Ingestion orchestrator. Reads ``sources.yaml``, fetches each source, and
writes plain-text + YAML-frontmatter files to ``corpus/``.

Re-running is safe: each source is overwritten in place. Pass ``only_ids``
to restrict to a subset of source ids (useful when adding a new state law).
Returns an :class:`IngestResult` so callers (the Admin UI) can surface
per-source failure reasons.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from nsa_chatbot.config import CORPUS_DIR, SOURCES_YAML
from nsa_chatbot.ingest.fetcher import FetchedDoc, fetch_ecfr, fetch_html, fetch_pdf
from nsa_chatbot.ingest.schemas import (
    Frontmatter,
    IngestError,
    IngestFailure,
    IngestResult,
    SourceSpec,
)


def _write_doc(path: Path, source: SourceSpec, doc: FetchedDoc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = Frontmatter(
        id=source.id,
        jurisdiction=source.jurisdiction,
        kind=source.kind,
        citation=source.citation,
        short=source.short or source.citation,
        source_url=doc.source_url,
        fetched_at=doc.fetched_at,
        notes=source.notes,
    )

    with path.open("w", encoding="utf-8") as fh:
        fh.write("---\n")
        yaml.safe_dump(fm.to_yaml_dict(), fh, sort_keys=False, allow_unicode=True)
        fh.write("---\n\n")
        fh.write(doc.text)
        fh.write("\n")


def _fetch_one(source: SourceSpec) -> FetchedDoc | None:
    """Dispatch to the right fetcher. Raises :class:`IngestError` on bad spec.

    Returns ``None`` for ``fetcher: skip`` sources (hand-maintained files
    already on disk that should not be overwritten).
    """
    fetcher = source.fetcher or "html"
    if fetcher == "ecfr":
        if source.title is None or source.part is None:
            raise IngestError("ecfr source missing title/part")
        return fetch_ecfr(
            title=source.title,
            part=source.part,
            section_prefix=source.section_prefix,
        )
    if fetcher == "html":
        if not source.url:
            raise IngestError("html source missing url")
        return fetch_html(source.url)
    if fetcher == "pdf":
        if not source.url:
            raise IngestError("pdf source missing url")
        return fetch_pdf(source.url)
    if fetcher == "skip":
        return None
    raise IngestError(f"unknown fetcher {fetcher!r}")


def ingest(only_ids: set[str] | None = None) -> IngestResult:
    """Fetch every source in ``sources.yaml`` (or the ``only_ids`` subset).

    Returns an :class:`IngestResult` summarising successes, failures (each
    with the reason), and non-fatal warnings (e.g. suspiciously thin HTML).
    """
    with SOURCES_YAML.open() as fh:
        cfg = yaml.safe_load(fh) or {}

    todo: list[tuple[Path, SourceSpec]] = []
    for raw in cfg.get("federal", []) or []:
        source = SourceSpec.from_dict(raw)
        todo.append((CORPUS_DIR / "federal" / f"{source.id}.txt", source))
    for state, sources in (cfg.get("states") or {}).items():
        for raw in sources or []:
            source = SourceSpec.from_dict(raw)
            todo.append(
                (CORPUS_DIR / "states" / state / f"{source.id}.txt", source)
            )

    result = IngestResult()
    for path, source in todo:
        if only_ids and source.id not in only_ids:
            continue
        try:
            doc = _fetch_one(source)
            if doc is None:
                # fetcher: skip -- hand-maintained file, not a failure.
                result.skipped.append(source.id)
                continue
            _write_doc(path, source, doc)
        except IngestError as exc:
            result.failures.append(IngestFailure(source.id, str(exc)))
            continue
        except Exception as exc:  # unexpected; surface but keep going
            result.failures.append(
                IngestFailure(source.id, f"unexpected error: {exc}")
            )
            continue
        result.succeeded.append(source.id)
        if doc.warning:
            result.warnings.append(f"{source.id}: {doc.warning}")

    return result
