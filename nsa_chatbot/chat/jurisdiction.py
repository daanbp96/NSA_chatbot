"""Deterministic jurisdiction detection from a question's text.

Used by the chat flow to catch the case where a user names a state *in the
question* (e.g. "in California…", "under 215 ILCS 124/10") while the dropdown
says something else. A miss (no signal found) degrades to dropdown-only
behavior — never worse — so the detector errs toward precision over recall.
Oblique references ("the Golden State") are intentionally not handled.

The citation-prefix patterns here are also a natural seed for the future
exact-citation lookup.
"""

from __future__ import annotations

import re

# 2-letter code -> display name, for the six in-scope states.
STATE_NAMES: dict[str, str] = {
    "CA": "California",
    "IL": "Illinois",
    "NY": "New York",
    "NJ": "New Jersey",
    "FL": "Florida",
    "TN": "Tennessee",
}

# Case-insensitive name / citation-prefix signals per state code.
_NAME_PATTERNS: dict[str, list[str]] = {
    "CA": [r"\bcalifornia\b", r"\bcalif\b", r"\bcal\.?\s*health", r"\bcal\.?\s*ins",
           r"\bhealth (?:and|&) safety code\b"],
    "IL": [r"\billinois\b", r"\bilcs\b"],
    "NY": [r"\bnew york\b", r"\bn\.y\.", r"\bfinancial services law\b"],
    "NJ": [r"\bnew jersey\b", r"\bn\.j\."],
    "FL": [r"\bflorida\b", r"\bfla\."],
    "TN": [r"\btennessee\b", r"\btenn\."],
}
_NAME_RES: dict[str, list[re.Pattern]] = {
    code: [re.compile(p, re.IGNORECASE) for p in pats]
    for code, pats in _NAME_PATTERNS.items()
}

# Postal codes matched case-sensitively as standalone UPPERCASE tokens, so "CA"
# fires but "ca"/"because" do not.
_POSTAL_RES: dict[str, re.Pattern] = {
    code: re.compile(rf"\b{code}\b") for code in STATE_NAMES
}

_AFFIRMATIVE = {
    "yes", "y", "yeah", "yep", "yup", "sure", "ok", "okay", "okey",
    "please", "do", "federal", "fed",
}


def detect_jurisdictions(text: str) -> set[str]:
    """Return the set of in-scope state codes the text references (excludes
    "federal"; empty when there is no clear signal).
    """
    text = text or ""
    found: set[str] = set()
    for code, regexes in _NAME_RES.items():
        if any(r.search(text) for r in regexes):
            found.add(code)
    for code, rx in _POSTAL_RES.items():
        if rx.search(text):
            found.add(code)
    return found


def is_affirmative(text: str) -> bool:
    """Loose yes/no for the 'answer based on federal?' confirmation."""
    t = (text or "").strip().lower()
    if not t:
        return False
    first = re.split(r"[\s,.!]+", t, maxsplit=1)[0]
    return t in _AFFIRMATIVE or first in _AFFIRMATIVE


# Section-number-shaped references in a question, for exact-citation retrieval.
# Decimal sections (149.110, 1371.9, 627.64194, 54.9816-1T); ILCS refs
# (215 ILCS 124/10 -> "124/10"); and §/section/sec./CFR/USC-prefixed numbers
# (incl. bare integers like NY's "§ 603"). Tokens are lowercased for matching.
_DECIMAL_RE = re.compile(r"\b\d+\.\d+[\w.\-/]*")
_ILCS_RE = re.compile(r"\b\d+\s+ILCS\s+(\d+(?:/\d+)?)", re.IGNORECASE)
_SECTION_RE = re.compile(
    r"(?:§|\bsection\b|\bsec\.|\bcfr\b|\bu\.?s\.?c\.?\b)\s*§?\s*(\d+(?:\.\d+)?[\w\-/]*)",
    re.IGNORECASE,
)


def extract_citation_tokens(text: str) -> list[str]:
    """Return normalized section-number tokens referenced in ``text`` (empty when
    none). Used to boost the exact provision into retrieval; a token that matches
    nothing in the index is simply ignored, so over-extraction is harmless.
    """
    text = text or ""
    raw: list[str] = []
    raw += [m.group(1) for m in _ILCS_RE.finditer(text)]
    raw += [m.group(0) for m in _DECIMAL_RE.finditer(text)]
    raw += [m.group(1) for m in _SECTION_RE.finditer(text)]

    seen: set[str] = set()
    out: list[str] = []
    for t in raw:
        t = t.strip().strip(".,;:").lower()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out
