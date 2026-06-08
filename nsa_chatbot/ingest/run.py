"""Ingestion orchestrator. Reads ``sources.yaml``, fetches each source, and
writes plain-text + YAML-frontmatter files to ``corpus/``.

Re-running is safe: each source is overwritten in place. Pass ``only_ids``
to restrict to a subset of source ids (useful when adding a new state law).
Returns an :class:`IngestResult` so callers (the Admin UI) can surface
per-source failure reasons.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import yaml

from nsa_chatbot.config import CORPUS_DIR, SOURCES_YAML
from nsa_chatbot.ingest.fetcher import (
    FetchedDoc,
    fetch_ecfr,
    fetch_html,
    fetch_pdf,
    fetch_via_anthropic,
    is_connection_error,
    parse_ecfr_url,
)
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
        title, part, section_prefix = source.title, source.part, source.section_prefix
        # The discovery agent proposes eCFR sources by URL without the structured
        # title/part the Versioner API needs — derive them from the URL.
        if (title is None or part is None) and source.url:
            parsed = parse_ecfr_url(source.url)
            if parsed:
                title, part, url_section = parsed
                section_prefix = section_prefix or url_section
        if title is None or part is None:
            raise IngestError(
                "ecfr source needs title/part (or an ecfr.gov URL to derive them from)"
            )
        return fetch_ecfr(title=title, part=part, section_prefix=section_prefix)
    if fetcher == "html":
        if not source.url:
            raise IngestError("html source missing url")
        return _local_or_web_fetch(fetch_html, source.url)
    if fetcher == "pdf":
        if not source.url:
            raise IngestError("pdf source missing url")
        return _local_or_web_fetch(fetch_pdf, source.url)
    if fetcher == "web_fetch":
        if not source.url:
            raise IngestError("web_fetch source missing url")
        return fetch_via_anthropic(source.url)
    if fetcher == "skip":
        return None
    raise IngestError(f"unknown fetcher {fetcher!r}")


def _local_or_web_fetch(local_fetch, url: str) -> FetchedDoc:
    """Local fetch first; on a connection failure (host unreachable), fall back
    to Anthropic's server-side web_fetch. Genuine errors (404/parse) still raise.
    """
    try:
        return local_fetch(url)
    except IngestError as exc:
        if is_connection_error(exc):
            return fetch_via_anthropic(url)
        raise


def preview_source(entry: dict, max_chars: int = 4000) -> str:
    """Fetch a proposed source and return the head of the extracted text — so a
    reviewer sees what would actually land in the corpus *before* approving.
    Reuses the same fetch dispatch as ingest; returns a bracketed message on
    failure (e.g. unreachable host) instead of raising.
    """
    try:
        spec = SourceSpec.from_dict(entry)
        doc = _fetch_one(spec)
    except IngestError as exc:
        return f"[could not fetch: {exc}]"
    except Exception as exc:
        return f"[fetch error: {exc}]"
    if doc is None:
        return "[fetcher: skip — nothing to fetch/preview]"
    text = doc.text.strip()
    prefix = f"[warning: {doc.warning}]\n\n" if doc.warning else ""
    suffix = f"\n\n… ({len(text)} chars total)" if len(text) > max_chars else ""
    return prefix + text[:max_chars] + suffix


def ingest(
    only_ids: set[str] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> IngestResult:
    """Fetch every source in ``sources.yaml`` (or the ``only_ids`` subset).

    ``on_progress(done, total, source_id)`` is called once per source before it
    is fetched, for a UI progress bar. Returns an :class:`IngestResult`
    summarising successes, failures (each with the reason), and non-fatal
    warnings (e.g. suspiciously thin HTML).
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
    if only_ids is not None:
        todo = [(p, s) for p, s in todo if s.id in only_ids]

    result = IngestResult()
    total = len(todo)
    for i, (path, source) in enumerate(todo, start=1):
        if on_progress:
            on_progress(i, total, source.id)
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
