"""The cross-flow bridge types: :class:`Chunk` and :class:`ChunkMetadata`.

A ``Chunk`` is produced by the chunker (corpus-building), embedded + stored by
``store``, and consumed by ``chat`` to build cited answers. Its ``metadata`` is
a typed :class:`ChunkMetadata` end to end; the flat scalar ``dict`` form Chroma
requires exists *only* at the storage boundary in :mod:`nsa_chatbot.store.index`
(via :meth:`ChunkMetadata.to_chroma` / :meth:`ChunkMetadata.from_chroma`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")

# Metadata fields that are optional/derived; on the way back out of Chroma an
# empty string for these means "absent" and is restored to ``None``.
_OPTIONAL_FIELDS = ("section", "section_heading", "subsection")


@dataclass
class ChunkMetadata:
    """Per-chunk metadata. Carried on :class:`Chunk` and, flattened, on each
    row in the Chroma collection.

    ``section`` / ``section_heading`` / ``subsection`` are populated when the
    chunker can identify them from section / subsection markers; otherwise
    ``None``.
    """

    source_id: str
    jurisdiction: str
    kind: str
    citation: str
    short: str
    source_url: str
    section: str | None = None
    section_heading: str | None = None
    subsection: str | None = None

    def to_chroma(self) -> dict[str, Any]:
        """Scalar-only dict for Chroma. ``None`` becomes ``""``."""
        return {k: ("" if v is None else v) for k, v in asdict(self).items()}

    @classmethod
    def from_chroma(cls, data: dict) -> ChunkMetadata:
        """Rebuild from a Chroma metadata dict. Unknown keys are ignored and
        empty strings for the optional fields are restored to ``None``.
        """
        known = {f.name for f in fields(cls)}
        kept = {k: v for k, v in (data or {}).items() if k in known}
        for k in _OPTIONAL_FIELDS:
            if kept.get(k) == "":
                kept[k] = None
        return cls(**kept)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: ChunkMetadata

    def n_tokens(self) -> int:
        return len(_ENC.encode(self.text))
