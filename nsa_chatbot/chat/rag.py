"""RAG pipeline: retrieve grounded chunks, build prompt, stream answer."""

from __future__ import annotations

from collections.abc import Iterator

from nsa_chatbot.core.llm import LLM
from nsa_chatbot.config import TOP_K
from nsa_chatbot.core.chunk import Chunk
from nsa_chatbot.store.index import query as vector_query

SYSTEM_PROMPT = """You are a careful legal research assistant for the
Clearest Health internal sales team. You answer questions about the federal
No Surprises Act (NSA) and surprise-billing laws in IL, CA, NY, NJ, FL, and TN.

You operate under strict rules:

1. GROUNDING. Base your answer ONLY on the SOURCES block provided in the user
   message. Do not rely on outside knowledge of statutes or regulations even
   if you think you know them. If the sources do not contain enough
   information to answer, say exactly: "I don't have that in my corpus."
   Then suggest what document the user should check.

2. CITATIONS. Every factual claim must end with a bracketed citation that
   matches a source id from the SOURCES block, like [S3] or [S1, S4]. Use the
   smallest set of sources that supports the claim.

3. JURISDICTION. Federal NSA preempts weaker state law but state law applies
   where it provides equal or greater consumer protection, where it governs
   reimbursement between insurers and out-of-network providers, or where the
   plan is fully insured (state-regulated). When a question implicates a
   specific state, use that state's sources first; cite federal law only when
   it adds something the state law doesn't.

4. SCOPE. You should break answers into two sections - 1. You can describe what the law says,
    2. You can provide practical sales/client advice based off of what the law explicitly states.

5. STYLE. Be terse and concrete. Use bullet points. Quote short fragments of
   statutory text in double-quotes when helpful. Do not invent section numbers
   or paragraph letters - if you cite a subsection like (a)(2), it must
   appear in the sources you were given.
"""


def _format_sources(chunks: list[Chunk]) -> str:
    """Numbered SOURCES block for the LLM prompt."""
    lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        meta = c.metadata
        citation = meta.citation or "?"
        section = meta.section
        sub = meta.subsection
        cite_full = citation
        if section and section not in citation:
            cite_full = f"{citation}, § {section}"
        if sub:
            cite_full += f"({sub})"
        url = meta.source_url
        header = f"[S{i}] {cite_full}"
        if url:
            header += f"  <{url}>"
        lines.append(header)
        lines.append(c.text.strip())
        lines.append("")
    return "\n".join(lines)


def _format_source_footer(chunks: list[Chunk]) -> str:
    """Compact footer for the UI's sources panel."""
    lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        meta = c.metadata
        citation = meta.citation or "?"
        url = meta.source_url
        line = f"[S{i}] {citation}"
        if url:
            line += f"  {url}"
        lines.append(line)
    return "\n".join(lines)


def _build_user_prompt(question: str, chunks: list[Chunk]) -> str:
    return (
        "QUESTION:\n"
        f"{question.strip()}\n\n"
        "SOURCES (primary legal text retrieved for this question):\n"
        f"{_format_sources(chunks)}\n\n"
        "Answer the question using only the sources above. Cite with [S#] "
        "tags. If insufficient, say so."
    )


def retrieve(
    question: str,
    k: int = TOP_K,
    jurisdiction: str | None = None,
) -> list[Chunk]:
    where: dict | None = None
    if jurisdiction == "federal":
        where = {"jurisdiction": "federal"}
    elif jurisdiction:
        # state + federal so the model can compare for preemption
        where = {"jurisdiction": {"$in": [jurisdiction, "federal"]}}
    return vector_query(question, k=k, where=where)


def answer(
    question: str,
    *,
    jurisdiction: str | None = None,
    llm: LLM | None = None,
    k: int = TOP_K,
) -> Iterator[tuple[str, str]]:
    """Stream ("token", text) tuples, then a final ("sources", footer)."""
    chunks = retrieve(question, k=k, jurisdiction=jurisdiction)
    if not chunks:
        yield (
            "token",
            "I don't have that in my corpus. Open the Admin tab to ingest "
            "sources and rebuild the index.",
        )
        return

    llm = llm or LLM()
    for tok in llm.stream(SYSTEM_PROMPT, _build_user_prompt(question, chunks)):
        yield ("token", tok)

    yield ("sources", _format_source_footer(chunks))
