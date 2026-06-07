# NSA IDR Assistant

A grounded chatbot for the federal **No Surprises Act** and state surprise-billing / IDR law
(CA, IL, NY, NJ, FL). Every answer is grounded in primary-source legal text and cited `[S#]`;
if the corpus doesn't cover something, it says so rather than guessing.

Runs locally as a web app (Gradio) with three tabs: **Chat**, **Add source**, **Source overview**.

---

## Setup

You need: a Mac (or Linux), the two API keys (Anthropic + OpenAI), and access to this private repo
(ask Daan to add you as a collaborator on GitHub first).

### 1. Install the tools (one time)

```bash
# uv — the Python environment manager this project uses
curl -LsSf https://astral.sh/uv/install.sh | sh
# (restart your terminal after this, or run:  source ~/.zshrc )
```

Git is already on most Macs. If `git` says "command not found", install Xcode tools: `xcode-select --install`.

### 2. Get the code

```bash
git clone https://github.com/daanbp96/NSA_chatbot.git
cd NSA_chatbot
uv sync
```

### 3. Add your API keys

```bash
cp .env.example .env
```

Open `.env` in any text editor and paste the two keys after the `=` signs:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

(`.env` is private — it is never uploaded to GitHub.)

### 4. Build the corpus + index (one time, ~a few minutes)

The legal documents and the search index are **not** stored in the repo — you build them once.
This downloads the source documents and embeds them (costs a few cents of OpenAI usage):

```bash
uv run python -c "from nsa_chatbot.ingest.run import ingest; from nsa_chatbot.store.index import build_index; r=ingest(); print('fetched', r.ok, 'sources,', r.fail, 'failed'); print('indexed', build_index(), 'chunks')"
```

A couple of sources may fail to fetch (some government sites block automated access) — that's
expected; the rest still work.

### 5. Run it

```bash
uv run python -m nsa_chatbot
```

Open the link it prints (**http://127.0.0.1:7860**) in your browser. To stop it, press `Ctrl+C`.

---

## Day-to-day

- **Just run it again:** `cd NSA_chatbot && uv run python -m nsa_chatbot` (steps 1–4 are one-time).
- **Get updates:** `git pull` then `uv sync`. If the update changed how documents are processed,
  re-run step 4 to rebuild the index.
- **Add a new legal source:** use the **Add source** tab in the app — describe what you need, approve
  the proposal, then click **Fetch + rebuild index**.

## Troubleshooting

- *"ANTHROPIC_API_KEY not set"* or an OpenAI auth error → check step 3 (`.env` keys filled in, no spaces).
- *The chat says the index needs to be built* → run step 4.
- *Clone fails with "repository not found"* → you need collaborator access to the private repo.

> Tip: to skip step 4, Daan can share a prebuilt `index/` folder — drop it into the project root and
> the app will use it directly.
