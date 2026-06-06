# TODO

Pending cleanup and decisions. Add items as they come up; remove when done.

## Cleanup

- [x] ~~Remove stray `Sources YAML:` header from `sources.yaml` (line 1).~~
- [x] ~~Handle `fetcher: pdf` entries in `sources.yaml`.~~ Implemented `fetch_pdf` in `fetcher.py` using `pypdf`; wired into `_fetch_one`. Verified by ingesting `cms-idr-guidance-disputing-parties`.
- [ ] **Ingest IL, NJ, TN.** Their sources are defined in `sources.yaml` (`illinois`, `new-jersey`, `tennessee` slugs) but have no corpus files — only `california`, `florida`, `new-york` are built. Run `ingest(only_ids={...})` for the IL/NJ/TN ids, then `build_index()`. (`tn-summary` is `fetcher: skip` and is hand-maintained.) Until this is done the bot can't answer for half its target states.

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

- [ ] **Phase 6 — conversation memory decision.** Each `answer()` call is still independent; no prior-turn context is threaded into the LLM. Decide: keep stateless (simpler, citations always fresh) or thread prior turns in (more natural, slightly higher cost, risk of stale citation refs). If threading, define summarize-vs-verbatim — don't append the whole transcript indefinitely.
- [ ] **Polish:** graceful error when the provider API key is missing (show a clear message instead of a stack trace); confirm streaming feels responsive for long answers (Gradio queue tuning).

### Post-rebuild improvements (deferred — revisit now that the rebuild works end-to-end)

- [ ] **Tighten `SYSTEM_PROMPT` toward IDR-first scope.** Currently generic about "surprise-billing laws"; the actual product is IDR strategy support.
- [ ] **Consolidate `STATE_CHOICES` and `JURISDICTION_NAMES`** into a single source of truth.
- [ ] **Infer jurisdiction from the question.** Today the dropdown is the only way to scope; would be more natural if the model could detect "in California, ..." in the question itself and either set the filter or ask a clarifying question. Likely needs a small extraction step (or just relies on the model to retrieve broadly and select the right state in its answer).
- [ ] **Decide whether to drop the OpenAI provider.** `llm.py` supports both Anthropic and OpenAI; only Anthropic is in use. Dropping the branch shrinks ~30 lines.

---

## Feature: Agentic source discovery (Step 0 of corpus pipeline)

### Goal

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
