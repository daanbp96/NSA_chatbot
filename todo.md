# TODO

Pending cleanup and decisions. Add items as they come up; remove when done.

## Next up — START HERE (handoff to a fresh session)

The two big retrieval-layer builds below are **DONE** (see "Shipped — dedup + agentic loop").
Pick the next work from **Smaller open items** — the highest-value being the **test suite**
(now that the retrieval layer has real moving parts) and the **`SYSTEM_PROMPT` IDR-first
tightening**.

### Smaller open items (carry forward)
- **TN corpus gap:** `tn-summary` is `fetcher: skip` with no file — hand-supply one (or add a TN
  domain to the whitelist + discover it). `il-public-act-102-901`: one IL page web_fetch couldn't
  retrieve (Admin → Sources shows both flagged).
- `SYSTEM_PROMPT` IDR-first tightening; consolidate `STATE_CHOICES` vs the jurisdiction map (DRY).
- **No tests/linter exist** — a small suite around the invariants (threshold, jurisdiction guard,
  citation match, YAML append, the new loop) would catch regressions; we've relied on manual + fan-out validation.

---

## Shipped — cost tiering + front router + non-legal register

Motivated by two real complaints: a legal question cost ~€1 (Opus did *both* the search planning
and the long answer over 46–76 sources), and the bot was too strict / wrong-register for the
audience (meta questions like "what's in your corpus" got "I don't have that in my corpus", and
answers told non-lawyer salespeople to "check these documents" they can't access).

**Orchestration decision:** kept plain-Python control flow (no LangChain/LangGraph — it cuts
against `core/llm.py`'s "no framework ceremony" and adds nothing for a single-provider app). The
"which agent, which model" routing is centralized declaratively in `config.MODEL_POLICY`.

- **`config.MODEL_POLICY`** — one map: `router`/`planner` → Haiku 4.5; `answer` → `{simple: Sonnet
  4.6, hard: Opus 4.8}`. Plus `SOURCE_CAP=15`. Call sites read `LLM(MODEL_POLICY[...])`.
- **`chat/router.py` (new)** — `route(message, history) -> RouteResult{route, difficulty, reply}`,
  one cheap Haiku call. Conversational bucket (greetings / "what can you do" / "what's in your
  corpus" / out-of-scope chit-chat) gets a human reply drafted in the same call — **no retrieval,
  no Opus**. Legal bucket goes to the grounded pipeline, tagged simple/hard. Fails safe to
  legal/hard. Crucially it does NOT predict corpus coverage — any in-domain question still hits the
  grounded refusal path, so the router can't hallucinate a "we don't cover that".
- **Answer tiering** — `chat/rag.py:answer_agentic(..., difficulty=...)` runs the planner on Haiku
  and generation on `MODEL_POLICY["answer"][difficulty]`. `app/chat_tab.py` computes
  `difficulty = hard if router.hard or >=2 states detected else simple`.
- **Cost caps** — `core/llm.py:run_tool_loop` gained `disable_parallel_tool_use=True` (one search
  per round → total searches ≤ `MAX_AGENT_SEARCHES=3`, was fanning out to 7); accumulated sources
  capped to `SOURCE_CAP=15` after dedup (was 46–76).
- **Register** — `SYSTEM_PROMPT` rule 1 no longer says "suggest what document to check"; new
  **rule 7 (AUDIENCE)** forbids telling the salesperson to read/check/consult any document or
  emitting a "what to check next" list. `[S#]` tags + sources panel kept as provenance/trust.
- **Wiring** — `chat_fn` calls `route()` on a fresh turn (after the `pending` clarification
  branches, which are replies, not fresh questions); conversational → short-circuit; legal →
  existing rewrite → guard → agentic loop with the chosen difficulty. Pending conflict → hard,
  offer-federal → simple.

Verified (live, real API): router classifies hi / capabilities / "what's in your corpus" /
pizza → conversational with human replies; ASC → legal/hard → **Opus**, 3 searches, 15 sources,
cited, no "go check" phrases; QPA → legal/simple → **Sonnet**, 2 searches, 15 sources, cited.
Cost shape: trivial turns are Haiku-only; simple legal is Haiku+Haiku+Sonnet over ≤15 sources;
hard is the same with Opus over ≤15 (vs Opus-on-everything over 46–76 before).

**Follow-up fix (recall + synthesis regression).** After shipping the cost caps, the ASC motivating
case went back to hedging ("the sources don't address physician ownership"). Diagnosis (verified):
two layers — (1) the flat caps (≤3 searches / 15 sources) starved *hard* broad questions of recall;
(2) even with the right chunks, the model framed an absent factor as a gap instead of reasoning over
the enumerated criteria. Fixes:
- **`config.SEARCH_BUDGET`** replaces the flat `MAX_AGENT_SEARCHES`/`SOURCE_CAP` with a
  difficulty-tiered budget: `simple {searches:3, sources:12}`, `hard {searches:6, sources:30}`.
  `answer_agentic` derives `max_searches`/`source_cap` from `difficulty`. (Verified the hard tier now
  gathers the nonparticipating-definition / participating-facility / qualified-IDR-item chunks.)
- **`SYSTEM_PROMPT` rule 8 (REASON FROM THE CRITERIA)** — when the question asks whether a factor
  affects an outcome and the sources enumerate that outcome's governing criteria, answer by applying
  them ("Ownership does not affect IDR eligibility; eligibility turns on [criteria] [S#]") instead of
  retreating to "the sources don't address that factor". Scoped: still refuse when the corpus lacks
  the governing criteria themselves; don't guess at out-of-corpus bodies of law.
- Verified: ASC (hard) now lands the confident grounded conclusion (no hedge); a control probe
  ("does a Medicare Star rating change IDR eligibility?") correctly answers "not a factor — eligibility
  turns on [criteria]" rather than inventing a rule.

**Possible follow-ups:** difficulty is judged pre-retrieval (router + ≥2-states); a simple-looking
but actually-hard question lands on Sonnet + the lean budget (rephrasing or the ≥2-states signal
escalates). `run_tool_loop` still has no per-call network timeout (a flaky connection hung an earlier
verify run; re-run worked). Haiku planner is non-deterministic in how many searches it issues (2–5
observed on the same hard question) — fine, but recall varies run-to-run.

---

## Shipped — dedup + agentic retrieval loop

Both queued retrieval-layer builds, motivated by the ASC/eligibility manual-test case (asked
whether a physician-owner ASC's cases are IDR-eligible, the bot used to reply "I don't have that
in my corpus" + a "what to check" list, even though the eligibility provisions are indexed — a
**recall gap from duplication**: ~11 near-identical tri-agency restatements crowded out the
distinct eligibility/open-negotiation chunks).

**Task 1 — near-duplicate dedup** (`chat/rag.py`, `config.py`):
- `_collapse_near_dups(chunks, threshold)` — 5-gram shingle Jaccard; drops the `[SECTION …]`
  marker before shingling so the tri-agency parallels (26/29/45 CFR) collapse to one while
  distinct subsections survive. `_shingles` lowercases, so the marker regex is `re.IGNORECASE`.
- `retrieve` + `retrieve_split` now over-fetch `k * RETRIEVAL_OVERFETCH` (=3), collapse, then
  `_merge_citation_hits` (which now **always caps to k** — it previously returned the dense list
  uncapped when no citation token was present, which leaked the over-fetch). Citation-boost still
  prepends ahead of dedup.
- Config: `RETRIEVAL_OVERFETCH=3`, `NEAR_DUP_THRESHOLD=0.85`.
- Verified: federal top-12 for the ASC question went from **1 → 5** eligibility/open-negotiation
  chunks; caps hold (12, and split 6/8).

**Task 2 — agentic retrieval loop, two-phase** (`core/llm.py`, `chat/rag.py`, `app/chat_tab.py`,
`config.py`):
- `core/llm.py:run_tool_loop(system, user, tools, dispatch, *, max_iters, max_tokens)` — a
  non-streaming Anthropic tool-use loop; yields each `tool_use` input (for progress), runs
  `dispatch`, stops on `end_turn` or after `max_iters` **rounds**. (Note: the model issues
  *parallel* tool calls per round, so a round ≠ one search — the ASC case did 7 searches over 3
  rounds.) Generation is deliberately NOT done here.
- `chat/rag.py:answer_agentic(question, *, jurisdiction, seed_chunks, llm, max_searches)` —
  Phase 1: seed the accumulator with the guard's chunks, then let the planner model issue
  `search_corpus` calls (scoped to `jurisdiction` via closure — **not** exposed to the model, so
  the jurisdiction guard's decision can't be overridden); each search uses `k=TOP_K//2`. Phase 2:
  `_collapse_near_dups` the accumulated union (the parallels re-enter across searches with
  distinct ids), then reuse the existing **`answer_from_chunks`** for streamed, `[S#]`-cited
  generation. Empty union → `OFF_CORPUS_MESSAGE`. `answer()` now seeds via `retrieve` then
  delegates here; `answer_from_chunks` kept as the phase-2 primitive.
- `app/chat_tab.py` — `_stream_answer` swaps `answer_from_chunks` → `answer_agentic(question,
  jurisdiction=effective, seed_chunks=chunks)`; renders new `("progress", …)` events as
  "🔎 Searching: …" lines, replaced by the first answer token. Jurisdiction guard + `pending`
  state machine + follow-up rewrite all unchanged.
- Config: `MAX_AGENT_SEARCHES=3`.
- Verified end-to-end against the live index (541 chunks, real Anthropic+OpenAI calls):
  - **ASC/eligibility** → self-searches (7) and answers correctly ("ownership is irrelevant;
    eligibility turns on OON status / participating facility / notice-and-consent / open
    negotiation [S#]") instead of "what to check"; 46 sources after dedup (was 76 pre-refinement).
  - **Off-topic** ("best pizza in Chicago") → 0 searches, `OFF_CORPUS_MESSAGE` (grounding holds).
  - **Normal in-corpus** (emergency balance-billing) → 5 searches, answers + cites.

**Possible follow-ups (not blocking):** the accumulated source set can still be large (~46 on a
broad question) — could cap the union or lower `MAX_AGENT_SEARCHES`/per-search `k` further if the
sources panel feels heavy or cost matters. Note `run_tool_loop` has no per-call network timeout;
a flaky API connection hung one verification run (re-run succeeded) — consider an SDK timeout.

---

## Shipped this session

**Package refactor (legibility — two-flows split).** `app.py` → `app/` package
(`chat_tab` / `admin_tab` / `main`); `LLM` + `Embedder` + `Chunk`/`ChunkMetadata`
moved into a shared `core/`; root `schemas.py` → `ingest/schemas.py`; top-level
`formats/` → `ingest/formats/`. `Chunk.metadata` is now a typed `ChunkMetadata`
end-to-end — the scalar dict Chroma needs is confined to `store/index.py`
(`to_chroma`/`from_chroma`). Import direction: `core/` depends only on `config`;
everything imports *from* core, never back.

**Flow-B (chat) hardening** — from the review fan-out, highest-severity first:
- [x] **Relevance threshold** — `RELEVANCE_MAX_DISTANCE` (cosine space, default 0.65, tuned from measured in-corpus ~0.4 vs out-of-corpus ~0.9). Off-topic questions now retrieve nothing and get the canonical "I don't have that in my corpus" instead of a confidently mis-cited answer. **This closed the grounding hole** (previously `query` always returned `k` chunks, so the refusal path only fired on an empty index).
- [x] **Embedder-match guard** — `query()` raises if the live embedder ≠ the one that built the index (was silent-garbage on drift).
- [x] **`ANSWER_MAX_TOKENS`** (default 8000, was a hardcoded 2000) + a truncation notice when the model hits the cap (was silently cut off mid-citation).
- [x] **Prompt-injection** — user question wrapped in `<question>` tags (early-close neutralized) + `SYSTEM_PROMPT` rule 6 treats it as data; refusal string unified to the exact canonical phrase.
- [x] **Dropped the OpenAI answer branch** — answers are Anthropic-only now (OpenAI stays for *embeddings*). Removed `ANSWER_PROVIDER`/`ANSWER_MODEL_OPENAI`. (Resolves the old "drop the OpenAI provider" item below.)
- [x] **Answer model default → `claude-opus-4-8`.**
- [x] **Typed `CollectionNotFoundError`** (re-exported from `store.index`) replaces a brittle substring match; mid-stream errors now **append** under the partial answer instead of clobbering it.

The index was **rebuilt to 413 chunks** (federal + CA/FL/NY) for the cosine-space
change — which also cleared the stale 215-chunk/federal-only index.

---

## Cleanup

- [x] ~~Remove stray `Sources YAML:` header from `sources.yaml` (line 1).~~
- [x] ~~Handle `fetcher: pdf` entries in `sources.yaml`.~~ Implemented `fetch_pdf` in `fetcher.py` using `pypdf`; wired into `_fetch_one`. Verified by ingesting `cms-idr-guidance-disputing-parties`.
- [x] ~~**Complete the corpus**~~ — mostly done via the `web_fetch` fallback. IL (`ilga.gov`), NJ
  (`njleg.state.nj.us`), and the 8 federal USC sections (`uscode.house.gov`) were unreachable from
  this environment (`ConnectTimeout`); they're now `fetcher: web_fetch` (fetched through Anthropic's
  server-side web_fetch) and ingested. Index is **~541 chunks**, **5/6 states** (CA/FL/IL/NJ/NY).
  **Remaining:** TN (`tn-summary`, `fetcher: skip` — hand-supply a file) and `il-public-act-102-901`
  (one IL public-act page web_fetch couldn't retrieve — revisit / try a different URL).

---

## Feature: Chat pipeline frontend (Gradio) — ✅ SHIPPED

The Gradio rebuild is complete and is the project's single runtime surface. The old click CLI + rich REPL is gone.

- [x] `gradio` added to `pyproject.toml`; `click` and `rich` removed (no leftover imports anywhere).
- [x] `nsa_chatbot/app.py` exists — `gr.Blocks` with jurisdiction dropdown, streaming `gr.Chatbot` wired to `chat.rag.answer`, and a foldable "Sources for last answer" panel showing the `[S#]` footer.
- [x] `python -m nsa_chatbot` launches the UI on `127.0.0.1:7860`.
- [x] `cli.py` deleted; operator actions (`ingest`, `build_index`, `stats`) are plain Python functions.
- [x] `CLAUDE.md` updated to describe the Gradio entry point.

The answer pipeline (`SYSTEM_PROMPT`, streaming tuples, `[S#]` citations) was unchanged by the rebuild.

### Still open from this track

- [x] ~~**Phase 6 — conversation memory decision.**~~ Resolved: **retrieval-only** memory (history
  rewrites the query, not threaded into the LLM) — keeps citations fresh/grounded while making
  follow-ups work. See "History-aware retrieval / statefulness" below.
- [ ] **Polish:** graceful error when the provider API key is missing (show a clear message instead of a stack trace); confirm streaming feels responsive for long answers (Gradio queue tuning).

### Post-rebuild improvements (deferred — revisit now that the rebuild works end-to-end)

- [ ] **Tighten `SYSTEM_PROMPT` toward IDR-first scope.** Currently generic about "surprise-billing laws"; the actual product is IDR strategy support.
- [ ] **Consolidate `STATE_CHOICES` and jurisdiction names** into a single source of truth.
- [ ] **Infer jurisdiction from the question** — promoted to its own feature, see **Jurisdiction guard** below.
- [x] ~~**Decide whether to drop the OpenAI provider.**~~ Done — answer path is Anthropic-only (OpenAI kept for embeddings).

---

## Remaining from the chat (Flow-B) review

The high-severity findings are fixed (see Shipped above). Still open, all optional and
roughly prioritized:

- [x] ~~**Jurisdiction guard**~~ — built this session; see its own section below.
- [x] ~~**Exact-citation / hybrid retrieval.**~~ Built. `extract_citation_tokens` (in
  `chat/jurisdiction.py`) pulls section-number tokens; `store.index.lookup_by_citation` scans
  metadata (`section` for federal, the `citation` string for state — Chroma can't substring-match)
  and `rag._merge_citation_hits` prepends exact matches ahead of the dense top-k (bypassing the
  distance threshold), in both `retrieve` and `retrieve_split`. Gated on a detected citation, so
  generic queries are unchanged. Verified: "Cal. Health & Safety Code § 1371.9" lands the right
  provision at position 0 and the answer cites it.
- [x] ~~**Dedup / MMR.**~~ Was discarded, then **built** (see "Shipped — dedup + agentic loop")
  once the ASC/eligibility case showed the tri-agency parallels causing a real *recall* failure
  (not just redundancy): they crowded the distinct eligibility chunks out of the top-k. Implemented
  as text near-duplicate collapse (`_collapse_near_dups`), not MMR.
- [x] ~~**Prompt caching** on `SYSTEM_PROMPT`.~~ Skip — it's a no-op as the app is shaped. The
  system prompt is ~700 tokens, far below Opus 4.8's 4096-token cache minimum (shorter prefixes
  silently don't cache); the large content (the SOURCES block) is volatile per question; and the
  chat is stateless, so there's no growing cacheable prefix. **Still a no-op even now:**
  memory shipped as *retrieval-only* (history is used to rewrite the query, not threaded into the
  model), so there's still no stable prefix. Caching would only matter under *full-conversational*
  memory — which we deliberately declined.
- [x] ~~**History-aware retrieval / statefulness.**~~ Built (retrieval-only). A follow-up like
  "what about California?" is rewritten into a standalone query by a cheap Haiku call
  (`chat/followup.py:rewrite_query`, gated on prior history, fail-safe to the raw message) before
  detection/retrieval; generation stays single-turn so citations stay grounded in each answer's own
  sources. Verified: T1 federal emergency-billing → T2 "what about California?" rewrote to a CA query,
  scoped CA, and answered from California AB-72 law; a standalone follow-up isn't derailed.

---

## Feature: Jurisdiction guard (in-text state safety) — ✅ SHIPPED

**Built this session.** Deterministic detection + in-chat clarification, per the locked decisions.

- New `nsa_chatbot/chat/jurisdiction.py` — `detect_jurisdictions()` (state names + postal codes
  + citation prefixes, word-boundary), `STATE_NAMES`, `is_affirmative()`.
- `chat/rag.py` — split `answer()` into `retrieve` + `answer_from_chunks`; added **`retrieve_split`**
  (state-only + federal-only retrieval).
- `app/chat_tab.py` — clarification state machine over a new `gr.State`: conflict → ask which;
  no relevant state material → offer federal (yes/no); "all" + named state → scope silently.

**Design note (changed during build):** the approved "partition a single mixed `[state,federal]`
top-k" gate over-fired — federal volume crowds a covered state out of the top-k, so CA questions
were wrongly told "no California material." Fixed with **`retrieve_split`**: retrieve the state's
own chunks separately from federal (each thresholded). Coverage now means "no *relevant state*
material" (judged on the state-only retrieval), which is accurate **and** surfaces state law the
mixed top-k had buried — this partially addresses the per-jurisdiction-representation item under
the review list. Verified: conflict ask/resolve, IL→offer-federal→yes/no, generic, and CA-in-text
all behave correctly; UI boots.

The original design write-up is kept below for reference.

**Status (original): designed, not built.** Decisions are locked; one fork left to pick at build time.

### Problem

Jurisdiction is taken **only from the dropdown**; the question text is ignored. The
relevance threshold guards *topical* relevance, not *jurisdictional* — federal NSA
chunks are topically close to almost any surprise-billing question, so they clear the
threshold. Consequences:

- Dropdown "all" + "in California…" in the text → can be answered from federal/other-state
  law and read as a California answer. Nothing trips the refusal path.
- Dropdown = state A + question about state B → filter excludes B; either refuses (safe-ish)
  or answers from A mislabeled (bad).
- Worst case: **IL / NJ / TN have zero indexed chunks**, so an in-text question about them is
  answered from federal and looks fully legitimate. (Compounds with the IL/NJ/TN corpus gap above.)

A confidently-wrong *legal* answer is the worst failure mode for this tool, so this ranks
above exact-citation and statefulness.

### Locked decisions

1. **Conflict** — question names a state that differs from a specific dropdown selection →
   **pause and ask** which to use (do not silently override).
2. **No coverage** — the named state has no chunks for this question → **refuse**, then
   **ask whether to answer based on federal law**, and act on the yes/no.

### Open fork (decide at build)

- **Detection mechanism:** deterministic mapping (state names + postal codes + citation
  prefixes like `Cal. Health & Safety` / `215 ILCS` → code; no API call; misses oblique
  phrasing; the citation→state map is reusable by the future exact-citation lookup) vs LLM
  extraction vs hybrid (deterministic first, LLM fallback). Leaning deterministic to start.
- **Interaction:** in-chat yes/no follow-up (bot asks, user replies in chat; needs cross-turn
  `gr.State` in `chat_fn` to hold the pending question) vs "resolve via dropdown + re-send"
  (no stateful parse, simpler, clunkier). The locked decisions imply the in-chat version.

### Design sketch

- `detect_jurisdictions(question) -> set[str]` in `chat/rag.py` (+ a small state/citation→code map).
- Pre-retrieval **conflict check** (detected vs dropdown) → ask. "all" dropdown + a named state →
  scope to that state (no nag); only ask on genuine disagreement.
- Post-retrieval **coverage check**: partition retrieved chunks into state-specific vs federal;
  if a specific state was requested and no state chunks came back but federal did → refuse + offer federal.
- Carry the pending question across turns via a new `gr.State` threaded through
  `send.click`/`msg.submit`, reset on Clear. Touches `chat/rag.py` and `app/chat_tab.py`.

---

## Feature: Agentic source discovery (Step 0 of corpus pipeline) — ✅ SHIPPED (lean v1)

Built to the user's lean model, **not** the 6-phase spec below (kept for reference / deferred items):

- `config.SOURCE_DOMAINS` — whitelist of trusted primary-source domains; drives `web_search`'s `allowed_domains`.
- `ingest/registry.py:append_source` — `ruamel.yaml` round-trip append (preserves comments), validates via `SourceSpec`, rejects duplicate ids, routes to the `federal`/`states:<slug>` bucket.
- `ingest/discover.py:discover` — Anthropic tool-use loop: server-side `web_search` (scoped to the whitelist) + a `propose_source` structured tool; returns reply + validated proposals + threaded messages.
- `app/discover_tab.py` — third "Add source" tab: chat → render proposals → **Approve & add** (append) / Dismiss, + a **Fetch + rebuild** button reusing `ingest()` + `build_index()`.

**Deliberately dropped from the 6-phase plan** (batch-append + final full rebuild made them unnecessary): incremental single-file indexing, reject/refine loop, audit log, atomic temp-file writes. **Preview-fetch is now built** — `ingest.run.preview_source` (reuses `_fetch_one`) + a "Preview fetch" button in the Add-source tab shows the head of the extracted text (catches bad extractions / nav-chrome / unreachable hosts before approve). Still dropped as low-value: refine-on-reject (the chat is already conversational), dup hints, agent-added log (git covers provenance).

**Caveat:** discovery (server-side search) works regardless of local network, but the rebuild's *fetch* runs locally — so blocked hosts (IL/NJ/USC) still fail at ingest. Not verified live end-to-end through the UI (the rebuild re-fetches all + re-embeds, mutating the real corpus/index) — components verified in isolation.

**Fix (eCFR-by-URL preview/ingest).** The discovery agent proposes eCFR sources as
`{fetcher: ecfr, url: <ecfr.gov page>}`, but `_fetch_one`'s ecfr branch ignored `url` and required
`title`/`part`, so Preview died with `ecfr source missing title/part`. Fixed: `fetcher.parse_ecfr_url`
derives `(title, part, section_prefix)` from an ecfr.gov URL (`.../title-45/.../part-149/.../section-149.110`),
and `_fetch_one` falls back to it when an ecfr source lacks the structured fields. Also added
`title`/`part`/`section_prefix` to the `propose_source` tool schema + a prompt note so the agent fills
them directly (persisted entries match the hand-authored convention). Verified: previewing the exact
failed proposal (45 CFR 149.110) now returns the section text.

**Operator-UI rework + Add-source agent polish (done).**
- **Admin → "Source overview"** (`app/admin_tab.py`, `app/main.py`): now read-only — shows **source
  counts** (declared / indexed / not-yet, per-jurisdiction source counts) and a per-source table
  (id, citation, fetched ✓/✗, indexed ✓/✗ with ⚠️/skip flags). **No chunk counts.** Removed the
  Ingest, Rebuild-index, and Index-status (chunk) sections and the `stats_fn`/`ingest_fn`/`build_fn`
  functions. (`registry._row` now carries `citation`.)
- **Add-source tab** (`app/discover_tab.py`) now owns the **search-domain whitelist** (moved from
  Admin, in a "Search domains" accordion) alongside discovery + Fetch+rebuild. Preview is longer
  (`preview_source` default 800 → 4000 chars) and **scrollable** (`gr.Code(lines=14, max_lines=14)`,
  accordion auto-opens on Preview).
- **Discovery agent** (`ingest/discover.py`): `_SYSTEM` now forbids narrating the search process
  (kills the "Let me fix the parsing / search hit its limit" leakage) and tells it to proactively
  suggest 2–3 concrete in-scope sources on vague/out-of-scope input instead of flatly refusing;
  `web_search max_uses` 5 → 10.
- Verified: app builds, overview renders source counts (no "chunks"), domains relocated, preview
  default raised. Live browser pass (tab labels, scroll feel, terser replies) left to a manual run.

### Goal (original spec — reference)

Let a sales-team user (non-developer) add a new source via a chat interface instead of editing `sources.yaml` by hand. A specialized agent searches for primary-source legal documents in scope (IDR + tangentially-relevant rules at federal + 6-state level), proposes a structured `sources.yaml` entry, the user approves, and the system fetches + indexes the new source.

Mode: **agent proposes, user approves, system persists**. No autonomous writes.

### Architectural fit

This becomes Step 0 of the corpus pipeline:

```
[NEW] user chats with agent → agent proposes sources.yaml entry → user approves
                                       ↓
                              append to sources.yaml
                                       ↓
                              ingest --only <new-id>
                                       ↓
                              chunk + embed + insert into Chroma
```

The answer pipeline is unchanged. Only the corpus pipeline gains a new entry point.

### Prerequisites

- [x] ~~Decide whether we're committing to Gradio as the frontend.~~ Yes — shipped (see above).
- [x] ~~Basic Gradio chat tab for the answer pipeline working first~~ — the "Ask" UI in `app.py` is the pattern the agent tab will mirror.
- [x] ~~PDF fetcher handled~~ — `fetch_pdf` implemented in `fetcher.py`.

All prerequisites are met; this feature is now unblocked and is the next major build.

### Plan

#### Phase 1 — agent foundations

- [ ] Choose web search provider. Default: **Anthropic native `web_search` tool** (one fewer dep, one fewer API key). Alternatives: Tavily, Brave. Decide based on quality + cost.
- [ ] Define the tool surface the agent has access to:
  - `web_search(query)` — find candidate URLs
  - `preview_fetch(url)` — run `fetch_html(url)` and return the first ~2000 chars so the agent can verify the page actually contains primary-source text
  - `list_existing_sources()` — return citations + ids from `sources.yaml` so the agent doesn't duplicate
  - `propose_source(entry: dict)` — structured-output tool the agent calls when it's confident; returns the proposal to the UI for user approval
- [ ] Write the agent system prompt. Must encode:
  - IDR domain scope (federal NSA + 6 states; tangentially-relevant rules per CLAUDE.md)
  - `sources.yaml` schema (required fields, fetcher options, id format)
  - **Strict scope filter**: only `.gov` domains, eCFR, official state legislature mirrors, CMS, and known accurate mirrors (e.g. `newyork.public.law` because nysenate.gov 403s). No law firm blogs, no Wikipedia, no Justia summaries.
  - Fetcher constraints (`ecfr` is federal CFR only; `html` needs an extractable page; PDFs not yet supported as of this writing).
  - Output format: always call `propose_source` with a complete YAML-shaped object; never write to disk directly.

#### Phase 2 — agent loop

- [ ] Implement a stateful conversation function `find_source_chat(user_message, history) -> (assistant_response, proposal_or_none)`:
  - Calls Anthropic with the system prompt + history + tools defined above
  - Loops on tool calls until the model either asks a clarifying question or calls `propose_source`
  - Returns (assistant text, structured proposal if any)
- [ ] Validate proposals before showing to user:
  - `id` is kebab-case and unique
  - All required fields present
  - `fetcher` is one of the supported values
  - URL parses and is reachable (HEAD request)

#### Phase 3 — preview + approval UI

- [ ] Gradio tab "Add source" with:
  - Chat panel (left/top) — user converses with the agent
  - Proposal panel (right/bottom) — appears when agent calls `propose_source`. Shows:
    - The proposed YAML entry rendered as a form
    - Citation, jurisdiction, kind dropdowns (so user can correct)
    - Test-fetch preview: first ~500 chars of what `fetch_html` extracted from the URL, so the user sees what would actually be indexed
    - [Approve] [Reject and continue] [Edit] buttons
- [ ] On Approve, kick off Phase 4.
- [ ] On Reject, send a message back into the agent loop ("user rejected this candidate, find another") so the agent can refine.

#### Phase 4 — persistence + ingest

- [ ] Implement `append_source_to_yaml(entry: dict) -> None`:
  - Loads `sources.yaml` (preserve comments via ruamel.yaml, not pyyaml — pyyaml strips comments)
  - Appends the entry under the correct jurisdiction bucket
  - Writes back atomically (temp file + rename)
- [ ] Implement `ingest_single_source(source_id: str) -> Path`:
  - Wrapper around `ingest.run.ingest(only_ids={source_id})`
  - Returns path to the new corpus file
- [ ] Implement `index_single_corpus_file(path: Path) -> int`:
  - Chunk the file via `chunker.chunk_file`
  - Embed via `get_embedder().embed`
  - `collection.add(...)` — does NOT drop the collection (unlike `build_index`)
  - Returns chunk count added
- [ ] Show status in the UI: "Fetching… Chunking… Embedding… Done. Added N chunks."

#### Phase 5 — failure modes

- [ ] Web search returns nothing → agent says "I couldn't find a primary source for that; can you describe it more?"
- [ ] `preview_fetch` returns < 200 chars → agent rejects the candidate and tries another
- [ ] User rejects 3+ candidates → agent gives up and asks the user to provide a URL directly
- [ ] `ingest_single_source` fails (network, blocked) → do NOT write to `sources.yaml`. Show error.
- [ ] `index_single_corpus_file` fails → corpus file exists but isn't queryable. Either retry or clean up the orphan file.
- [ ] Duplicate id → agent should catch this via `list_existing_sources`; if it slips through, raise before write.

#### Phase 6 — quality controls

- [ ] Log every agent-added source to a separate file (e.g. `corpus/_added_via_agent.log`) so curated additions are auditable.
- [ ] Show a small "Sources added via agent" list in the UI tab so the user can see history.
- [ ] (Stretch) Periodically re-validate agent-added sources — has the URL gone dead? Has the statute been amended?

### Open questions to resolve during build

- **Confidence threshold for `propose_source`**: should the agent only propose when confident, or always propose its best guess and let the user reject? Current plan: confident-only, on the theory that the user's time is the constraint.
- **What about updates to existing sources?** If a state amends its statute, does the agent suggest re-fetching? For now: out of scope, manual re-ingest. Revisit if it becomes a friction point.
- **Conversation memory across sessions**: should the agent remember past rejections so it doesn't propose the same bad source twice? For now: no, sessions are stateless. Revisit.
- **Cost guardrails**: each agent turn is a Claude call with tool use; rapid back-and-forth could be expensive. Add a turn limit (e.g. 10 turns max per source) and a "spent so far" indicator?

### Risk to be aware of

The whole product invariant is "every claim grounded in a primary source." If a non-primary source (law firm summary, blog post) gets into the corpus, the model will cite it as authoritative because the citation *exists in the index*. The strict-scope system prompt + manual approval are the safeguards — both must be in place before this feature ships.
