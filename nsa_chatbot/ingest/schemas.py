"""Typed schemas for the dict shapes that flow through the pipeline.

These were previously inline dicts keyed by string. Using dataclasses gives us
one definitive home for the field set and lets type-checkers + IDEs follow the
data end to end.

Shapes:

* :class:`SourceSpec` -- one entry in ``sources.yaml``. Conditional fields
  (``title``/``part`` for eCFR, ``url`` for html/pdf) are optional; the
  orchestrator validates them per ``fetcher`` at use time.
* :class:`Frontmatter` -- the YAML block at the top of each ``corpus/**.txt``
  file. Written by the ingest step, read back by the chunker.
* :class:`IngestResult` / :class:`IngestFailure` -- structured summary returned
  by :func:`nsa_chatbot.ingest.run.ingest` so the Admin tab can show per-source
  failure reasons rather than just a count.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields


class IngestError(Exception):
    """Raised by fetchers/parsers when a source can't be turned into text.

    The orchestrator catches this and records the message against the source
    id so it can be surfaced in the Admin UI.
    """


@dataclass
class SourceSpec:
    """One entry in ``sources.yaml``.

    Fields differ by ``fetcher``: ``ecfr`` needs ``title``/``part``;
    ``html``/``pdf`` need ``url``; ``skip`` needs nothing extra. Validation
    happens in :mod:`nsa_chatbot.ingest.run` where the fetcher is dispatched.
    """

    id: str
    jurisdiction: str
    kind: str
    citation: str
    fetcher: str
    short: str | None = None
    notes: str | None = None
    # html / pdf
    url: str | None = None
    # ecfr
    title: int | None = None
    part: int | None = None
    section_prefix: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> SourceSpec:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Frontmatter:
    """YAML frontmatter block at the top of a corpus file."""

    id: str
    jurisdiction: str
    kind: str
    citation: str
    short: str
    source_url: str
    fetched_at: str
    notes: str | None = None

    def to_yaml_dict(self) -> dict:
        """Dict suitable for ``yaml.safe_dump`` (drops ``None`` fields)."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict) -> Frontmatter:
        known = {f.name for f in fields(cls)}
        # Preserve robustness: missing fields default to "" so a partial
        # frontmatter doesn't crash chunking.
        defaults = {
            "id": "",
            "jurisdiction": "unknown",
            "kind": "unknown",
            "citation": "",
            "short": "",
            "source_url": "",
            "fetched_at": "",
        }
        merged = {**defaults, **{k: v for k, v in (data or {}).items() if k in known}}
        return cls(**merged)


@dataclass
class IngestFailure:
    """One failed source. ``reason`` is the human-readable message."""

    source_id: str
    reason: str


@dataclass
class IngestResult:
    """Summary returned from :func:`nsa_chatbot.ingest.run.ingest`.

    ``succeeded`` and ``skipped`` are lists of source ids; ``failures`` carries
    a per-source reason so the Admin tab can show exactly which sources broke
    and why. ``warnings`` carries non-fatal flags (e.g. suspiciously short
    HTML) for sources that did write successfully.
    """

    succeeded: list[str] = field(default_factory=list)
    failures: list[IngestFailure] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> int:
        return len(self.succeeded)

    @property
    def fail(self) -> int:
        return len(self.failures)
