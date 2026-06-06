"""Source fetchers: ``fetch_ecfr``, ``fetch_html``, ``fetch_pdf``.

These are thin orchestrators around HTTP + format-specific parsers in
:mod:`nsa_chatbot.formats`. Each raises :class:`IngestError` on failure;
the orchestrator catches and records the reason per source.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from nsa_chatbot.formats.ecfr import parse_ecfr_xml
from nsa_chatbot.formats.html_pages import extract_html
from nsa_chatbot.formats.pdf import extract_pdf
from nsa_chatbot.schemas import IngestError

# Browser UA -- some state legislature sites 403 non-browser clients.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
TIMEOUT = 30
SHORT_BODY_THRESHOLD = 200


@dataclass
class FetchedDoc:
    text: str
    fetched_at: str
    source_url: str
    warning: str | None = None


def _now_iso() -> str:
    return _dt.datetime.utcnow().isoformat() + "Z"


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _get(url: str, **kwargs) -> requests.Response:
    headers = {"User-Agent": USER_AGENT, **kwargs.pop("headers", {})}
    resp = requests.get(url, headers=headers, timeout=TIMEOUT, **kwargs)
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
