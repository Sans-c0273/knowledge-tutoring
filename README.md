# socratic-tutor-azure

A standalone, wired-together copy of three independent POCs, all running on
Azure AI Foundry instead of their original providers. This repo is a fresh
`git init` with **no history or remotes from the source repos** — nothing here
can ever be pushed back to `kg_reasoner`, `knowledge-graph-poc`, or
`socratic-tutor-poc`'s original GitHub repos.

```
socratic-tutor-azure/
├── kg_reasoner/            # graph reasoning library (Tier-2 rerank now Azure-backed)
├── knowledge-graph-poc/    # doc-ingestion CLI (chat completions now Azure-backed)
├── socratic-tutor-poc/     # the tutor app: FastAPI backend + React frontend
│                           # (chat completions AND embeddings now Azure-backed)
└── azure-foundry-adapter/  # shared Azure REST client + per-project adapters
```

## What actually changed vs. the originals

Unlike the earlier `azure-foundry-adapter/` sibling folder (which deliberately
made *zero* edits to the 3 original repos), this repo's copies are edited
directly and verified against a real Azure endpoint:

| Project | What was added | Verified how |
|---|---|---|
| `knowledge-graph-poc` | `azure_foundry` provider (`kg.yaml`, `config.py`, new adapter file, `cli.py` choices) | `uv run kg selftest` (1002 passed) + a real `adapter_factory(cfg).complete(...)` call against Azure |
| `socratic-tutor-poc` | `azure_foundry` chat provider + `azure_foundry` embeddings backend (`config.py`, `providers/__init__.py`, new provider file, `embeddings.py`) | `uv run pytest` (1120 passed) + a real server boot (`uv run poc serve`, health check OK) + real `get_provider()`/`get_embedder()` calls against Azure |
| `kg_reasoner` | Nothing — it's provider-agnostic by design; only its own test suite (30 passed) plus a real `rerank()` call using `AzureFoundryLLMCall` | Live call correctly dropped an irrelevant candidate and kept the relevant one |

All three currently point at one Azure resource (`gpt-4o-mini` for chat,
`text-embedding-3-large` for embeddings) via each project's own `.env`
(gitignored — never committed).

## Running each one

**socratic-tutor-poc** (the only long-running service):
```bash
cd socratic-tutor-poc
uv sync --extra dev
uv run poc serve          # :8000
cd web && npm install && npm run dev   # :5173, separate terminal, dev only
```

> **Node version note:** `web/`'s Vite 8 / rolldown needs Node ^20.19 or
> >=22.12. If the system `node` is older (this dev box ships 18.20.6), `npm
> run dev` fails with `SyntaxError: ... does not provide an export named
> 'styleText'`. Fix: download a standalone newer Node (no sudo/system change
> needed) and use it just for this folder:
> ```bash
> mkdir -p ~/.local/node22 && cd ~/.local/node22
> curl -sL https://nodejs.org/dist/v22.14.0/node-v22.14.0-linux-x64.tar.xz | tar -xJ --strip-components=1
> cd web && ~/.local/node22/bin/node ~/.local/node22/bin/npm install --legacy-peer-deps
> ~/.local/node22/bin/node node_modules/vite/bin/vite.js
> ```
> (`--legacy-peer-deps` also works around an unrelated npm 10.9.x arborist bug —
> `Cannot read properties of null (reading 'edgesOut')` — hit on a clean install.)

**knowledge-graph-poc** (CLI, run on demand):
```bash
cd knowledge-graph-poc
uv sync
uv run kg ingest           # put source files in data/inbox/ first
```

**kg_reasoner** (library, no server — invoke from your own script):
```python
from azure_foundry_adapter.client import AzureFoundrySettings
from azure_foundry_adapter.adapters.kg_reasoner import AzureFoundryLLMCall
from kg_reasoner.pipeline import reason

call = AzureFoundryLLMCall(AzureFoundrySettings.from_env())
guidance = reason(ctx, index, llm_call=call)
```

These three still aren't wired to *each other* (no shared docker-compose, no
inter-service HTTP calls) — that was true of the originals and remains true
here; this repo only replaces *who each one talks to for its LLM calls*
(Azure, instead of Claude/OpenRouter/local BGE-M3).

## Credentials

Each project's own `.env` (gitignored) holds real Azure endpoint/key/deployment
values. `.env.example` in each project shows the shape without secrets. If you
rotate the Azure key, update all three `.env` files (they currently share one
Azure resource) plus `azure-foundry-adapter/.env`.

## Dimension note (embeddings)

`socratic-tutor-poc` embeds via Azure `text-embedding-3-large` (3072-dim) now,
not local BGE-M3 (1024-dim). If `data/chroma/` already has BGE-M3 vectors in
it, delete it before ingesting again — the dimensions aren't compatible.
