"""Canonical US jurisdiction table — the single source of truth shared by the
ingest, chat, and app layers so a new state works through the Add source tab
with no per-state code edits.

Lives in ``core`` (the shared kernel): it depends on nothing and everything
else imports *from* it. The 2-letter code is what retrieval filters on
(``ChunkMetadata.jurisdiction``); the slug is the ``states:`` bucket key in
sources.yaml and the ``corpus/states/<slug>/`` directory.
"""

from __future__ import annotations

# code -> (display name, sources.yaml / corpus slug)
US_STATES: dict[str, tuple[str, str]] = {
    "AL": ("Alabama", "alabama"),
    "AK": ("Alaska", "alaska"),
    "AZ": ("Arizona", "arizona"),
    "AR": ("Arkansas", "arkansas"),
    "CA": ("California", "california"),
    "CO": ("Colorado", "colorado"),
    "CT": ("Connecticut", "connecticut"),
    "DE": ("Delaware", "delaware"),
    "DC": ("District of Columbia", "district-of-columbia"),
    "FL": ("Florida", "florida"),
    "GA": ("Georgia", "georgia"),
    "HI": ("Hawaii", "hawaii"),
    "ID": ("Idaho", "idaho"),
    "IL": ("Illinois", "illinois"),
    "IN": ("Indiana", "indiana"),
    "IA": ("Iowa", "iowa"),
    "KS": ("Kansas", "kansas"),
    "KY": ("Kentucky", "kentucky"),
    "LA": ("Louisiana", "louisiana"),
    "ME": ("Maine", "maine"),
    "MD": ("Maryland", "maryland"),
    "MA": ("Massachusetts", "massachusetts"),
    "MI": ("Michigan", "michigan"),
    "MN": ("Minnesota", "minnesota"),
    "MS": ("Mississippi", "mississippi"),
    "MO": ("Missouri", "missouri"),
    "MT": ("Montana", "montana"),
    "NE": ("Nebraska", "nebraska"),
    "NV": ("Nevada", "nevada"),
    "NH": ("New Hampshire", "new-hampshire"),
    "NJ": ("New Jersey", "new-jersey"),
    "NM": ("New Mexico", "new-mexico"),
    "NY": ("New York", "new-york"),
    "NC": ("North Carolina", "north-carolina"),
    "ND": ("North Dakota", "north-dakota"),
    "OH": ("Ohio", "ohio"),
    "OK": ("Oklahoma", "oklahoma"),
    "OR": ("Oregon", "oregon"),
    "PA": ("Pennsylvania", "pennsylvania"),
    "RI": ("Rhode Island", "rhode-island"),
    "SC": ("South Carolina", "south-carolina"),
    "SD": ("South Dakota", "south-dakota"),
    "TN": ("Tennessee", "tennessee"),
    "TX": ("Texas", "texas"),
    "UT": ("Utah", "utah"),
    "VT": ("Vermont", "vermont"),
    "VA": ("Virginia", "virginia"),
    "WA": ("Washington", "washington"),
    "WV": ("West Virginia", "west-virginia"),
    "WI": ("Wisconsin", "wisconsin"),
    "WY": ("Wyoming", "wyoming"),
}


# Lowercased full name -> code, for canonicalize().
_NAME_TO_CODE: dict[str, str] = {
    name.lower(): code for code, (name, _slug) in US_STATES.items()
}

# Common legal/colloquial shorthand -> code (keys lowercased, periods removed).
# Full names and 2-letter codes are handled directly; these only add shorthand a
# model or user might emit ("Tex.", "Calif.", "Fla.").
_ALIAS_TO_CODE: dict[str, str] = {
    "cal": "CA", "calif": "CA", "colo": "CO", "conn": "CT", "fla": "FL",
    "ill": "IL", "ind": "IN", "kan": "KS", "kans": "KS", "mass": "MA",
    "mich": "MI", "minn": "MN", "miss": "MS", "mont": "MT", "neb": "NE",
    "nebr": "NE", "nev": "NV", "okla": "OK", "ore": "OR", "oreg": "OR",
    "penn": "PA", "penna": "PA", "tenn": "TN", "tex": "TX", "wash": "WA",
    "wis": "WI", "wisc": "WI", "wyo": "WY",
}
_FEDERAL = {"federal", "fed", "us", "usa", "national"}


def canonicalize(value: str) -> str | None:
    """Normalize a free-form jurisdiction label to its canonical stored form:
    ``"federal"`` or a 2-letter US state code. Returns None if unrecognizable.

    Accepts codes ("co", "CO"), full names ("Colorado"), federal aliases
    ("federal", "fed", "US"), and common abbreviations ("Tex.", "Calif."). This
    is the single seam where a loosely-labeled proposal becomes the canonical
    value the chat dropdown / detection / retrieval filter all share.
    """
    if not value:
        return None
    s = " ".join(value.strip().lower().replace(".", "").split())
    if not s:
        return None
    if s in _FEDERAL:
        return "federal"
    if s.upper() in US_STATES:
        return s.upper()
    if s in _NAME_TO_CODE:
        return _NAME_TO_CODE[s]
    return _ALIAS_TO_CODE.get(s)


def slug_for(code: str) -> str | None:
    """sources.yaml / corpus slug for a code, or None if it isn't a US state."""
    entry = US_STATES.get((code or "").upper())
    return entry[1] if entry else None


def choices() -> list[tuple[str, str]]:
    """(label, value) pairs for the jurisdiction dropdown: all + federal, then
    every state by display name. Gradio renders the label and returns the value.
    """
    states = sorted(
        ((name, code) for code, (name, _slug) in US_STATES.items()),
        key=lambda nc: nc[0],
    )
    return [("All jurisdictions", "all"), ("Federal", "federal")] + [
        (name, code) for name, code in states
    ]
