"""HTML page extraction: site-specific selectors + chrome stripping.

Different statute hosts wrap content differently, so we try a ranked list of
selectors from most-specific (uscode.house.gov, leg.state.fl.us, etc.) to
most-general (``<main>``, ``<article>``, ``<pre>``) before falling back to
``<body>``. Site-specific chrome (USC edition lists, etc.) is then stripped
from the candidate text.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup


# Sites we fetch differ in how they wrap content. Ranked from most-specific
# to most-general. Comments call out which site each selector targets.
_CONTENT_SELECTORS = [
    # uscode.house.gov - statute text is in a div with class "main-content"
    {"name": "div", "attrs": {"class": "main-content"}},
    # nysenate.gov - law text in section.law-content or div.c-block-law
    {"name": "div", "attrs": {"class": re.compile(r"law-doc|c-block-law|law-content")}},
    # leg.state.fl.us - statute body in <div class="Section">
    {"name": "div", "attrs": {"class": re.compile(r"Section|StatuteBody")}},
    # leginfo.legislature.ca.gov - <div id="manylawsections"> or <div id="law_section">
    {"name": "div", "attrs": {"id": re.compile(r"manylawsections|law_section")}},
    # ilga.gov uses <td> tables; grab the largest text block
    {"name": "td", "attrs": {"class": re.compile(r"xsl|content")}},
    # njleg.state.nj.us - <body>; we'll fall through
    # law.justia.com - <div id="codes-content"> or <article>
    {"name": "div", "attrs": {"id": "codes-content"}},
    {"name": "article", "attrs": {}},
    # cms.gov - <main>
    {"name": "main", "attrs": {}},
    # congress.gov plaws - <pre> blocks contain statute text
    {"name": "pre", "attrs": {}},
]


# USC pages start with edition lists; we drop everything before the first
# real provision marker (§NNN. ...). Editorial / References-in-text trailers
# are also boilerplate, but we KEEP "Statutory Notes" because effective-date
# notes there can matter for compliance answers.
_USC_PROVISION_RE = re.compile(
    r"(?m)^§\s?\d+(gg|usc)?[–\-]?\w*\.?\s+\S"
)


def extract_html(html_bytes: bytes, url: str, raw_text: str = "") -> str:
    """Return cleaned main-text content for ``url``.

    Walks ``_CONTENT_SELECTORS`` ranked most-specific to most-general; falls
    back to ``<body>``; falls back to ``raw_text`` (typically ``resp.text``)
    if there's no ``<body>``. Output is always whitespace-normalized and
    chrome-stripped. ``url`` drives site-specific chrome rules (USC has
    different chrome from CA leginfo, etc.).
    """
    soup = BeautifulSoup(html_bytes, "lxml")

    # Drop script/style/nav noise globally.
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    candidate_text: str | None = None
    for sel in _CONTENT_SELECTORS:
        nodes = soup.find_all(sel["name"], attrs=sel["attrs"])
        if not nodes:
            continue
        best = max(nodes, key=lambda n: len(n.get_text(" ", strip=True)))
        text = best.get_text("\n", strip=True)
        if len(text) > 400:  # ignore tiny matches
            candidate_text = text
            break

    if candidate_text is None:
        body = soup.find("body")
        candidate_text = body.get_text("\n", strip=True) if body else raw_text

    cleaned = normalize_whitespace(candidate_text)
    cleaned = _strip_site_chrome(cleaned, url)
    return cleaned


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace; cap blank lines at one."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_site_chrome(text: str, url: str) -> str:
    if "uscode.house.gov" in url:
        m = _USC_PROVISION_RE.search(text)
        if m:
            text = text[m.start() :]
        for marker in ("\nEditorial Notes\n", "\nReferences In Text\n"):
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx]
                break
    return text
