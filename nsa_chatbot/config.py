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
# Output cap for an answer. Answers are streamed, so this can be generous —
# the two-section cited format runs long, and 2000 truncated mid-citation.
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "8000"))

# Retrieval / chunking knobs. Edit here if tuning; not env-overridable.
TOP_K = 12
CHUNK_TARGET_TOKENS = 550
CHUNK_OVERLAP_TOKENS = 80

# Max cosine distance (0..2) for a retrieved chunk to count as relevant. Chunks
# farther than this are dropped, so an out-of-corpus question retrieves nothing
# and the pipeline answers "I don't have that in my corpus" instead of citing
# weak hits. Only meaningful with a cosine-space index. Raise toward 2.0 to
# effectively disable filtering. Default 0.65 is the midpoint of the measured
# gap on this corpus: in-corpus questions hit ~0.35-0.43, out-of-corpus ~0.89+.
RELEVANCE_MAX_DISTANCE = float(os.getenv("RELEVANCE_MAX_DISTANCE", "0.65"))
