# NSA IDR Assistant

A grounded chatbot for the federal **No Surprises Act** and **state surprise-billing / IDR law for
any US state you add**. Every answer is grounded in primary-source legal text and cited `[S#]`;
if the corpus doesn't cover something, it says so rather than guessing.

Runs locally as a web app (Gradio) with three tabs: **Chat**, **Add source**, **Source overview**.

The legal corpus is **not shipped** — you build it on your own machine through the **Add source**
tab (search → verify each link → approve → rebuild). Nothing about which states are covered is
hardcoded; you add the ones you need.

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

### 4. Run it

```bash
uv run python -m nsa_chatbot
```

Open the link it prints (**http://127.0.0.1:7860**) in your browser. To stop it, press `Ctrl+C`.

### 5. Build your corpus (in the app, one time)

A fresh install has no legal documents yet, so the Chat tab will say it has nothing in its corpus.
Fill it from the **Add source** tab:

1. Describe what you need, e.g. *"I need IDR documentation for the state of Texas."*
2. The assistant searches the web and proposes a few candidate links.
3. **Open each link to check it's a legit primary source**, then click **✓ Approve** (or **✗ Reject**)
   on the ones you want — one by one.
4. Click **Fetch + rebuild index**. This downloads the approved documents and embeds them (a few
   cents of OpenAI usage) and shows how many chunks each source produced.

Repeat for any state or topic. A source that fetched almost nothing (flagged with ⚠️ in the rebuild
report, e.g. a JavaScript-only page) should be dropped and replaced — open its link and find a
plain-HTML or PDF version.

---

## Day-to-day

- **Just run it again:** `cd NSA_chatbot && uv run python -m nsa_chatbot`.
- **Get updates:** `git pull` then `uv sync`. If the update changed how documents are processed,
  click **Fetch + rebuild index** (tick *Re-fetch all*) to rebuild.
- **Add more sources:** the **Add source** tab, anytime — approve links, then **Fetch + rebuild index**.

Your corpus (`sources.yaml`, `corpus/`, `index/`) stays on your machine and is never uploaded.

## Troubleshooting

- *"ANTHROPIC_API_KEY not set"* or an OpenAI auth error → check step 3 (`.env` keys filled in, no spaces).
- *The chat says it has nothing in its corpus* → add sources via the **Add source** tab (step 5).
- *Clone fails with "repository not found"* → you need collaborator access to the private repo.

> Tip: Daan can share a prebuilt `sources.yaml` + `index/` — drop them into the project root and the
> app will use them directly, skipping step 5.
