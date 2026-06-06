"""Legal-aware chunker.

Legal text has natural structure (sections, subsections, paragraphs). We chunk
along that structure rather than blindly slicing at character boundaries, so
each chunk can be cited back to a real provision (e.g. ``45 CFR § 149.110(a)``).

Strategy, in order of preference:

1. Split on explicit ``[SECTION § X — Y]`` markers emitted by the eCFR fetcher.
2. Split on inline section markers like ``§ 149.110`` or ``Section 1371.9``.
3. If a section is still too long, split on subsection markers ``(a)``, ``(1)``.
4. Final fallback: token-window split with overlap (rare, for unstructured guidance).

Format-specific regexes live in :mod:`nsa_chatbot.formats.legal_text` so the
fetcher and chunker share one source of truth for section-marker syntax.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken
import yaml

from nsa_chatbot.config import CHUNK_OVERLAP_TOKENS, CHUNK_TARGET_TOKENS
from nsa_chatbot.formats.legal_text import SECTION_MARKER_RE, SUBSECTION_RE
from nsa_chatbot.schemas import ChunkMetadata, Frontmatter

_ENC = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)

    def n_tokens(self) -> int:
        return len(_ENC.encode(self.text))


def _read_frontmatter(path: Path) -> tuple[Frontmatter, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return Frontmatter.from_dict({"id": path.stem}), text
    end = text.find("\n---", 3)
    if end == -1:
        return Frontmatter.from_dict({"id": path.stem}), text
    raw = yaml.safe_load(text[3:end]) or {}
    raw.setdefault("id", path.stem)
    body = text[end + 4 :].lstrip("\n")
    return Frontmatter.from_dict(raw), body


def _token_split(text: str, target: int, overlap: int) -> list[str]:
    """Token-window split with overlap. Used only for unstructured fallback."""
    ids = _ENC.encode(text)
    if len(ids) <= target:
        return [text]
    out: list[str] = []
    start = 0
    while start < len(ids):
        end = min(start + target, len(ids))
        out.append(_ENC.decode(ids[start:end]))
        if end == len(ids):
            break
        start = end - overlap
    return out


def _split_subsections(text: str, target_tokens: int) -> list[tuple[str, str]]:
    """Split a section's body on subsection markers. Returns (label, text) pairs."""
    matches = list(SUBSECTION_RE.finditer(text))
    if not matches:
        return [("", text)]
    parts: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.start() : end].strip()
        parts.append((m.group("label"), body))

    # Greedily group small subsections together until we approach target_tokens
    # so we don't over-fragment.
    grouped: list[tuple[str, str]] = []
    cur_label: list[str] = []
    cur_text: list[str] = []
    cur_tokens = 0
    for label, body in parts:
        n = len(_ENC.encode(body))
        if cur_tokens + n > target_tokens and cur_text:
            grouped.append((",".join(cur_label), "\n\n".join(cur_text)))
            cur_label, cur_text, cur_tokens = [], [], 0
        cur_label.append(label)
        cur_text.append(body)
        cur_tokens += n
    if cur_text:
        grouped.append((",".join(cur_label), "\n\n".join(cur_text)))
    return grouped


def _split_by_section_markers(
    body: str,
) -> list[tuple[str | None, str | None, str]]:
    """Split body on explicit ``[SECTION § ... ]`` markers from the eCFR fetcher.

    Returns list of (section_number, section_heading, text). When no markers
    are present, returns a single (None, None, body) entry.
    """
    matches = list(SECTION_MARKER_RE.finditer(body))
    if not matches:
        return [(None, None, body)]
    out: list[tuple[str | None, str | None, str]] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        section_text = body[m.end() : end].strip()
        out.append((m.group("num"), m.group("head").strip(), section_text))
    return out


def chunk_file(path: Path) -> list[Chunk]:
    fm, body = _read_frontmatter(path)
    if not body.strip():
        return []

    base_meta = ChunkMetadata.from_frontmatter(fm)

    # Holds (metadata, text, section, subsection) before chunk_id assignment.
    raw_chunks: list[tuple[ChunkMetadata, str, str | None, str | None]] = []

    for sec_num, sec_head, sec_body in _split_by_section_markers(body):
        section_header = ""
        if sec_num:
            section_header = f"§ {sec_num}"
            if sec_head:
                section_header += f" — {sec_head}"
            section_header += "\n\n"

        full_text = section_header + sec_body
        # If the section is short enough, emit as a single chunk.
        if len(_ENC.encode(full_text)) <= CHUNK_TARGET_TOKENS:
            raw_chunks.append(
                (
                    _meta_with_section(base_meta, sec_num, sec_head, None),
                    full_text,
                    sec_num,
                    None,
                )
            )
            continue

        # Otherwise split on subsections.
        for label, sub_text in _split_subsections(sec_body, CHUNK_TARGET_TOKENS):
            chunk_text = section_header + sub_text
            sub_label = label or None
            # If still too long (rare - long subsection), token-window split.
            for piece in _token_split(
                chunk_text, CHUNK_TARGET_TOKENS, CHUNK_OVERLAP_TOKENS
            ):
                raw_chunks.append(
                    (
                        _meta_with_section(base_meta, sec_num, sec_head, sub_label),
                        piece,
                        sec_num,
                        sub_label,
                    )
                )

    # Final safety net for sources with zero structure detected.
    if not raw_chunks:
        for piece in _token_split(body, CHUNK_TARGET_TOKENS, CHUNK_OVERLAP_TOKENS):
            raw_chunks.append((base_meta, piece, None, None))

    # Assign deterministic positional indices to disambiguate chunk ids.
    # IL 215 ILCS 124/10, for example, contains multiple revisions of the
    # same subsection from different Public Acts; their text can collide.
    return [
        _make_chunk(meta, text, section, label, idx)
        for idx, (meta, text, section, label) in enumerate(raw_chunks)
    ]


def _meta_with_section(
    base: ChunkMetadata,
    section: str | None,
    heading: str | None,
    subsection: str | None,
) -> ChunkMetadata:
    """Return a copy of ``base`` with section/heading/subsection populated."""
    return ChunkMetadata(
        source_id=base.source_id,
        jurisdiction=base.jurisdiction,
        kind=base.kind,
        citation=base.citation,
        short=base.short,
        source_url=base.source_url,
        section=section,
        section_heading=heading,
        subsection=subsection,
    )


def _make_chunk(
    meta: ChunkMetadata,
    text: str,
    section: str | None,
    label: str | None,
    index: int,
) -> Chunk:
    cid_parts = [meta.source_id]
    if section:
        cid_parts.append(section)
    if label:
        cid_parts.append(label)
    cid_parts.append(str(index))
    chunk_id = ":".join(cid_parts)
    return Chunk(chunk_id=chunk_id, text=text, metadata=meta.to_chroma())


def chunk_corpus(corpus_dir: Path) -> Iterable[Chunk]:
    for path in sorted(corpus_dir.rglob("*.txt")):
        yield from chunk_file(path)
