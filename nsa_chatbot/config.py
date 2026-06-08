"""Project configuration.

Three kinds of settings, kept deliberately separate:

* **Developer defaults** (paths, model ids, retrieval/chunking knobs) — plain
  constants below. Change them here and commit; they are code, not deployment
  config, so they are intentionally *not* env-overridable.
* **Secrets** (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``) — NOT defined here.
  The SDKs read them straight from the environment; ``load_dotenv()`` loads a
  local ``.env`` (see ``.env.example``) so they're present in development.
* **Runtime-editable state** (the search-domain whitelist) lives in a file
  (``source_domains.txt``, via ``ingest.domains``), not here — these constants
  freeze at import, so anything that changes while the app runs belongs in a file.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# Load secrets (API keys) from a local .env in development — see .env.example.
# This is the only thing the environment is used for.
load_dotenv()

# --- Paths ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus"
INDEX_DIR = ROOT / "index"
SOURCES_YAML = ROOT / "sources.yaml"
SOURCE_DOMAINS_FILE = ROOT / "source_domains.txt"

# --- Models -----------------------------------------------------------------
# Answers are always Anthropic; embeddings always OpenAI (Anthropic has no
# embeddings model). Changing EMBEDDING_MODEL requires a full index rebuild.
EMBEDDING_MODEL = "text-embedding-3-large"
ANSWER_MODEL_ANTHROPIC = "claude-opus-4-8"  # default LLM() model; hard-tier answers
REWRITE_MODEL_ANTHROPIC = "claude-haiku-4-5"  # cheap follow-up query rewriting
ANSWER_MAX_TOKENS = 8000  # streamed, so generous — the two-section cited answers run long

# Which model handles which step of the chat pipeline — the "which agent, which
# model" routing policy in one place (plain Python, no orchestration framework).
# Cheap models do the cheap work (classify, plan searches); the grounded answer
# is tiered: Sonnet by default, Opus only for hard questions. Read at the call
# sites via LLM(MODEL_POLICY["planner"]) / LLM(MODEL_POLICY["answer"][difficulty]).
MODEL_POLICY = {
    "router": "claude-haiku-4-5",   # classify conversational vs legal (+ difficulty) + draft chit-chat
    "planner": "claude-haiku-4-5",  # agentic search-loop query planning
    "answer": {
        "simple": "claude-sonnet-4-6",  # single-provision / one-jurisdiction questions
        "hard": "claude-opus-4-8",      # multi-state / multi-provision / strategy reasoning
    },
    "discover": "claude-sonnet-4-6",  # source-discovery agent: web_search + propose a structured entry
}

# --- Retrieval / chunking ---------------------------------------------------
TOP_K = 12
# Over-fetch this multiple of k before de-duplicating, so collapsing near-
# identical chunks (e.g. the tri-agency CFR parallels) still leaves enough
# distinct survivors to fill the top-k.
RETRIEVAL_OVERFETCH = 3
# Two chunks are treated as near-duplicates when their 5-gram shingle Jaccard
# similarity is >= this. 0.85 collapses the tri-agency restatements (26/29/45
# CFR say the same thing) while keeping genuinely distinct subsections.
NEAR_DUP_THRESHOLD = 0.85
# Agentic-loop budget, tiered by question difficulty. `searches` caps the
# self-directed search rounds (one search per round, parallelism disabled, so it
# also caps total searches); `sources` caps the chunks fed to generation after
# dedup. Simple questions stay lean (most traffic, where the cost win lives);
# hard ones (multi-provision / multi-state reasoning) need broad recall, so they
# get a larger budget — a too-small budget starves them and the model hedges
# ("I don't have that") even though the determinant provisions are in the corpus.
SEARCH_BUDGET = {
    "simple": {"searches": 3, "sources": 12},
    "hard": {"searches": 6, "sources": 30},
}
CHUNK_TARGET_TOKENS = 550
CHUNK_OVERLAP_TOKENS = 80
# Max cosine distance (0..2) for a chunk to count as relevant. 0.65 is the
# midpoint of the measured gap on this corpus (in-corpus ~0.35-0.43,
# out-of-corpus ~0.89+); beyond it, off-topic questions retrieve nothing and
# the bot answers "I don't have that in my corpus" instead of citing weak hits.
RELEVANCE_MAX_DISTANCE = 0.65

# --- Source-discovery whitelist (seed) --------------------------------------
# Seeds source_domains.txt on first run; the live list is managed from the Admin
# tab (ingest.domains). Trusted primary-source hosts only; bare domains also
# match their subdomains (www., etc.).
SOURCE_DOMAINS = [
    "ecfr.gov",                    # federal CFR (eCFR)
    "uscode.house.gov",            # federal USC
    "cms.gov",                     # CMS NSA / IDR guidance
    "govinfo.gov",                 # federal (GPO) — statutes, regs, the Register
    "congress.gov",                # federal (USC, public laws)
    "leginfo.legislature.ca.gov",  # CA
    "nysenate.gov",                # NY
    "newyork.public.law",          # NY mirror (nysenate often 403s)
    "leg.state.fl.us",             # FL
    "flsenate.gov",                # FL
    "ilga.gov",                    # IL
    "njleg.state.nj.us",           # NJ
    # State government base domains — a base domain auto-covers its subdomains
    # (the search API rejects bare TLDs / host wildcards), so e.g. tn.gov covers
    # advance.tn.gov where the TN code lives. tn.gov unblocks TN (the gap state).
    "tn.gov",
    "ca.gov",
    "ny.gov",
    "nj.gov",
    "il.gov",
]
