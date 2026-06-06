"""Chroma-backed vector index. Local, persistent, no server required.

Public functions (intended to be called from API endpoints or tests):
  - :func:`build_index` -- drop and recreate the collection from ``./corpus``.
  - :func:`query` -- vector search with optional metadata filter.
  - :func:`stats` -- collection size and sample jurisdictions.
"""

from __future__ import annotations

import shutil
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import chromadb
from chromadb.config import Settings

from nsa_chatbot.config import CORPUS_DIR, INDEX_DIR
from nsa_chatbot.ingest.chunker import Chunk, chunk_corpus
from nsa_chatbot.store.embedder import Embedder, get_embedder

COLLECTION = "nsa_corpus"


def _client() -> chromadb.PersistentClient:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(INDEX_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def _batched(it: Iterable, n: int) -> Iterable[list]:
    batch: list = []
    for x in it:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def build_index(
    corpus_dir: Path = CORPUS_DIR, embedder: Embedder | None = None
) -> int:
    """Re-build the Chroma index from scratch. Returns chunk count.

    Order matters:
      1. ``delete_collection`` -- clears Chroma's *in-memory* collection
         cache (the ``PersistentClient`` is cached per-path within a process,
         so wiping only disk would leave a stale collection entry that
         ``create_collection`` then refuses to overwrite).
      2. Remove orphan UUID-named HNSW directories -- ``delete_collection``
         removes the collection's row from sqlite but leaves its vector
         files on disk, and they pile up across rebuilds.
      3. ``create_collection`` -- fresh start.
    """
    embedder = embedder or get_embedder()
    client = _client()

    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    if INDEX_DIR.exists():
        for child in INDEX_DIR.iterdir():
            if child.is_dir():
                shutil.rmtree(child)

    coll = client.create_collection(
        name=COLLECTION,
        metadata={"embedder": embedder.model_id},
    )

    total = 0
    for batch in _batched(chunk_corpus(corpus_dir), 64):
        texts = [c.text for c in batch]
        embeddings = embedder.embed(texts)
        coll.add(
            ids=[c.chunk_id for c in batch],
            documents=texts,
            metadatas=[_clean_meta(c.metadata) for c in batch],
            embeddings=embeddings,
        )
        total += len(batch)

    return total


def _clean_meta(meta: dict) -> dict:
    """Chroma requires scalar metadata values."""
    return {k: ("" if v is None else v) for k, v in meta.items()}


def get_collection() -> chromadb.Collection:
    return _client().get_collection(COLLECTION)


def query(
    text: str,
    k: int = 12,
    where: dict | None = None,
    embedder: Embedder | None = None,
) -> list[Chunk]:
    embedder = embedder or get_embedder()
    coll = get_collection()
    [emb] = embedder.embed([text])
    res = coll.query(
        query_embeddings=[emb],
        n_results=k,
        where=where or None,
    )
    out: list[Chunk] = []
    for cid, doc, meta in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0]
    ):
        out.append(Chunk(chunk_id=cid, text=doc, metadata=meta or {}))
    return out


def stats() -> dict:
    """Real per-jurisdiction chunk counts across the whole collection.

    Walks all metadata in the collection, so the result reflects the full
    distribution rather than a sample. Cheap at current corpus sizes; if
    the collection grows past tens of thousands of chunks, consider caching.
    """
    try:
        coll = get_collection()
    except Exception as exc:
        return {"error": str(exc), "count": 0}
    count = coll.count()
    metadatas = coll.get(include=["metadatas"]).get("metadatas") or []
    jurisdictions = Counter(m.get("jurisdiction", "?") for m in metadatas)
    return {"count": count, "jurisdictions": dict(jurisdictions)}
