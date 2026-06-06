"""PDF text extraction via pypdf.

No section-marker emission -- PDFs don't carry that structure, so the
chunker falls through to subsection / token-window splits.
"""

from __future__ import annotations

import io

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from nsa_chatbot.formats.html_pages import normalize_whitespace
from nsa_chatbot.schemas import IngestError

SHORT_BODY_THRESHOLD = 200


def extract_pdf(pdf_bytes: bytes, url: str = "") -> str:
    """Return cleaned plain text from a PDF.

    Raises :class:`IngestError` on parse failure or suspiciously short output.
    ``url`` is only used to enrich the error message.
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except (PdfReadError, ValueError, OSError) as exc:
        raise IngestError(f"PDF parse failed: {exc}") from exc

    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            # Individual page failures are common (encrypted pages, malformed
            # streams); skip the page and keep going.
            continue

    text = normalize_whitespace("\n\n".join(p for p in parts if p))
    if len(text) < SHORT_BODY_THRESHOLD:
        raise IngestError(
            f"PDF text suspiciously short ({len(text)} chars) for {url}"
        )
    return text
