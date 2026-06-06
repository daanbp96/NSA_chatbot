"""eCFR XML schema knowledge.

eCFR's versioner API returns nested ``<DIVn>`` elements:
``DIV5`` = Part, ``DIV6`` = Subpart, ``DIV8`` = Section. Sections wrap a
``<HEAD>`` plus ``<P>`` / ``<FP>`` paragraph elements. This module knows
how to walk that tree and emit a flat text representation that downstream
chunking can split on.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from nsa_chatbot.ingest.formats.legal_text import render_section_marker


def parse_ecfr_xml(xml_bytes: bytes, section_prefix: str | None = None) -> str:
    """Parse an eCFR versioner XML payload into chunker-ready text.

    Returns a flat string with one ``[SECTION § N — H]`` marker per section,
    followed by that section's paragraph text. Returns an empty string if no
    matching ``DIV8`` sections were found (caller treats that as a fetch
    failure).
    """
    soup = BeautifulSoup(xml_bytes, features="xml")
    chunks: list[str] = []

    for sec in soup.find_all("DIV8"):
        section_num = sec.get("N", "").strip()
        if section_prefix and not section_num.startswith(section_prefix):
            continue

        head = sec.find("HEAD")
        head_text = head.get_text(" ", strip=True) if head else ""
        # Drop the section number prefix so we don't double-print it.
        head_text = re.sub(rf"^§\s*{re.escape(section_num)}\s*", "", head_text)

        body_parts: list[str] = []
        for p in sec.find_all(["P", "FP"]):
            body_parts.append(p.get_text(" ", strip=True))
        body = "\n\n".join(b for b in body_parts if b)

        marker = render_section_marker(section_num, head_text)
        chunks.append(f"\n\n{marker}\n\n{body}".rstrip())

    return "\n".join(chunks).strip()
