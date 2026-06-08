"""How legal text is structured: section markers, subsection markers.

The eCFR fetcher *emits* ``[SECTION § N — H]`` markers (via
:func:`render_section_marker`); the chunker *consumes* the same markers (via
:data:`SECTION_MARKER_RE`). Keeping both ends here means the convention is
defined in exactly one place.
"""

from __future__ import annotations

import re

# Marker the eCFR fetcher emits at the start of every section, and that the
# chunker splits on. The em-dash between number and heading is intentional.
SECTION_MARKER_RE = re.compile(
    r"\[SECTION § (?P<num>[\w\-.]+)\s*—\s*(?P<head>[^\]]+)\]"
)

# Subsection labels: (a), (1), (i), at the start of a line.
SUBSECTION_RE = re.compile(r"(?m)^\s*\((?P<label>[a-zA-Z0-9]{1,3})\)\s+")


def render_section_marker(section_num: str, heading: str) -> str:
    """Build the ``[SECTION § N — H]`` marker line.

    Symmetric counterpart to :data:`SECTION_MARKER_RE`. Used by the eCFR
    fetcher when emitting structured text the chunker can split on.
    """
    return f"[SECTION § {section_num} — {heading}]"
