"""Source fetchers: ``fetch_ecfr``, ``fetch_html``, ``fetch_pdf``,
``fetch_via_anthropic``.

These are thin orchestrators around HTTP + format-specific parsers in
:mod:`nsa_chatbot.ingest.formats`. Each raises :class:`IngestError` on failure;
the orchestrator catches and records the reason per source. ``fetch_via_anthropic``
fetches through Anthropic's server-side ``web_fetch`` — used as a fallback for
hosts the local network can't reach.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass

import anthropic
import requests
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from nsa_chatbot.config import REWRITE_MODEL_ANTHROPIC
from nsa_chatbot.ingest.formats.ecfr import parse_ecfr_xml
from nsa_chatbot.ingest.formats.html_pages import extract_html
from nsa_chatbot.ingest.formats.pdf import extract_pdf
from nsa_chatbot.ingest.schemas import IngestError

# Browser UA -- some state legislature sites 403 non-browser clients.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
# (connect, read): a short connect timeout so an unreachable host (e.g. ilga.gov)
# fails in seconds and the web_fetch fallback kicks in, instead of hanging.
CONNECT_TIMEOUT = 5
TIMEOUT = 30
SHORT_BODY_THRESHOLD = 200


def _retryable(exc: BaseException) -> bool:
    """Retry transient HTTP failures (read timeouts, 5xx), but NOT a failure to
    *connect* — an unreachable host won't recover on retry; retrying just delays
    the web_fetch fallback by the full backoff (~90s for nothing)."""
    return isinstance(exc, requests.RequestException) and not isinstance(
        exc, requests.ConnectionError
    )


@dataclass
class FetchedDoc:
    text: str
    fetched_at: str
    source_url: str
    warning: str | None = None


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


@retry(
    retry=retry_if_exception(_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _get(url: str, **kwargs) -> requests.Response:
    headers = {"User-Agent": USER_AGENT, **kwargs.pop("headers", {})}
    resp = requests.get(
        url, headers=headers, timeout=(CONNECT_TIMEOUT, TIMEOUT), **kwargs
    )
    resp.raise_for_status()
    return resp


# ---------- eCFR fetcher ---------------------------------------------------


def _ecfr_latest_issue_date(title: int) -> str | None:
    """Resolve the latest available issue date for a CFR title.

    eCFR data lags real time by a few days, so today's date won't work. We hit
    the versioner metadata endpoint. Returns ``None`` if the lookup fails;
    callers fall back to a date a week ago.
    """
    url = "https://www.ecfr.gov/api/versioner/v1/titles.json"
    try:
        resp = _get(url)
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None
    for t in data.get("titles", []):
        if t.get("number") == title:
            return t.get("up_to_date_as_of") or t.get("latest_issue_date")
    return None


# eCFR human-facing URLs encode the structured fields the Versioner API needs,
# e.g. .../title-45/.../part-149/.../section-149.110 → title 45, part 149,
# section_prefix "149.110". The discovery agent proposes eCFR sources by URL, so
# parse those out rather than requiring it to also supply title/part by hand.
_ECFR_TITLE = re.compile(r"/title-(\d+)")
_ECFR_PART = re.compile(r"/part-(\d+)")
_ECFR_SECTION = re.compile(r"/section-([\d.]+)")


def parse_ecfr_url(url: str | None) -> tuple[int, int, str | None] | None:
    """Extract ``(title, part, section_prefix)`` from an ecfr.gov URL, or ``None``
    if it isn't a parseable eCFR URL (missing title or part)."""
    if not url or "ecfr.gov" not in url:
        return None
    t = _ECFR_TITLE.search(url)
    p = _ECFR_PART.search(url)
    if not (t and p):
        return None
    s = _ECFR_SECTION.search(url)
    section = s.group(1).rstrip(".") if s else None
    return int(t.group(1)), int(p.group(1)), section


def fetch_ecfr(
    title: int,
    part: int,
    section_prefix: str | None = None,
    date: str | None = None,
) -> FetchedDoc:
    """Fetch a CFR part (or sub-prefix) from the eCFR Versioner API."""
    if date is None:
        date = _ecfr_latest_issue_date(title) or (
            _dt.date.today() - _dt.timedelta(days=7)
        ).isoformat()
    url = (
        f"https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml"
        f"?part={part}"
    )
    try:
        resp = _get(url)
    except requests.RequestException as exc:
        raise IngestError(f"eCFR HTTP fetch failed: {exc}") from exc

    text = parse_ecfr_xml(resp.content, section_prefix=section_prefix)
    if not text:
        raise IngestError(
            f"eCFR returned no sections for title {title} part {part}"
        )

    return FetchedDoc(text=text, fetched_at=_now_iso(), source_url=url)


# ---------- Generic HTML fetcher -------------------------------------------


def fetch_html(url: str) -> FetchedDoc:
    """Fetch an HTML page and extract its main text content."""
    try:
        resp = _get(url)
    except requests.RequestException as exc:
        raise IngestError(f"HTTP fetch failed: {exc}") from exc

    cleaned = extract_html(resp.content, url, raw_text=resp.text)
    warning = None
    if len(cleaned) < SHORT_BODY_THRESHOLD:
        warning = f"HTML body suspiciously short ({len(cleaned)} chars)"
    return FetchedDoc(
        text=cleaned, fetched_at=_now_iso(), source_url=url, warning=warning
    )


# ---------- PDF fetcher ----------------------------------------------------


def fetch_pdf(url: str) -> FetchedDoc:
    """Download a PDF and extract plain text."""
    try:
        resp = _get(url)
    except requests.RequestException as exc:
        raise IngestError(f"HTTP fetch failed: {exc}") from exc

    text = extract_pdf(resp.content, url=url)
    return FetchedDoc(text=text, fetched_at=_now_iso(), source_url=url)


# ---------- Anthropic web_fetch fallback ------------------------------------


def is_connection_error(exc: Exception) -> bool:
    """True if ``exc`` was caused by a failure to *connect* (timeout / refused)
    — i.e. the host is unreachable, not a 404/parse error. Used to decide
    whether to fall back to ``fetch_via_anthropic``.
    """
    return isinstance(exc.__cause__, (requests.ConnectionError, requests.Timeout))


def fetch_via_anthropic(url: str) -> FetchedDoc:
    """Fetch a URL through Anthropic's server-side ``web_fetch`` tool — runs on
    Anthropic's network, so it reaches hosts the local client can't. Pulls the
    raw fetched text out of the ``web_fetch_tool_result`` block.
    """
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=REWRITE_MODEL_ANTHROPIC,
        max_tokens=128,
        tools=[{
            "type": "web_fetch_20260209",
            "name": "web_fetch",
            "max_uses": 1,
            "allowed_callers": ["direct"],
        }],
        extra_headers={"anthropic-beta": "web-fetch-2025-09-10"},
        messages=[{"role": "user", "content": f"Fetch {url}"}],
    )
    for block in resp.content:
        b = block.model_dump()
        if b.get("type") != "web_fetch_tool_result":
            continue
        result = b.get("content") or {}
        if result.get("type") == "web_fetch_result":
            data = (((result.get("content") or {}).get("source")) or {}).get("data")
            if data and data.strip():
                return FetchedDoc(text=data, fetched_at=_now_iso(), source_url=url)
        raise IngestError(
            f"web_fetch failed for {url}: {result.get('error_code') or result.get('type')}"
        )
    raise IngestError(f"web_fetch returned no result for {url}")
