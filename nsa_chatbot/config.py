"""Project config. Provider/model selection via env (.env supported)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus"
INDEX_DIR = ROOT / "index"
SOURCES_YAML = ROOT / "sources.yaml"

# Embeddings — swap providers by setting EMBEDDING_PROVIDER + EMBEDDING_MODEL.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# Answer model — swap providers by setting ANSWER_PROVIDER and the matching model env.
ANSWER_PROVIDER = os.getenv("ANSWER_PROVIDER", "anthropic")
ANSWER_MODEL_ANTHROPIC = os.getenv("ANSWER_MODEL_ANTHROPIC", "claude-opus-4-7")
ANSWER_MODEL_OPENAI = os.getenv("ANSWER_MODEL_OPENAI", "gpt-4.1")

# Retrieval / chunking knobs. Edit here if tuning; not env-overridable.
TOP_K = 12
CHUNK_TARGET_TOKENS = 550
CHUNK_OVERLAP_TOKENS = 80
