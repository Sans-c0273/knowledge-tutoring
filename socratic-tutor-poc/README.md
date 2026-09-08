# Socratic AI Tutor — POC

A rule-driven **Pedagogy Layer** that turns an LLM from an answer machine into a teaching system.
Every teaching decision (intent, level, strategy, whether a direct answer is permitted, response structure)
is made by deterministic rules and data **before** the LLM is called. The LLM only classifies intent on the
way in and renders the decided plan on the way out.

## Specs (contract of record)

`~/TK-PKA/02-works/Eddi Learning Platform/03-outputs/Socratic AI/`

Read in this order:
1. `2026-09-01-socratic-ai-poc-build-kickoff-addendum.md` — settled decisions (supersedes model/deployment choices elsewhere)
2. `2026-09-01-socratic-ai-poc-implementation-plan.md` — requirements R1–R21 with acceptance criteria
3. `2026-09-01-socratic-ai-poc-technical-specification.md` — schemas, rule tables, call contracts
4. `2026-09-01-socratic-ai-poc-architecture.md` — decision rationale

## Stack

- **Backend**: Python 3.11/3.12, FastAPI, uv
- **Frontend**: React + Vite SPA (glass-box UI: ingestion progress, chat, per-turn inspector)
- **LLM**: Claude via subscription OAuth (`ant auth login`) — Haiku 4.5 for intent/evaluation, Sonnet 5 for generation
- **Embeddings**: local BGE-M3 · **Vector store**: local Chroma
- **Access**: localhost, single user, no auth

## Setup

```bash
uv sync --extra dev
ant auth login          # one-time: Claude credentials from your subscription
uv run poc serve        # http://localhost:8000
```

## Layout

```
src/socratic_tutor/
  models/         Pydantic schemas (the spec's JSON contracts)
  pedagogy/       Intent, student level, strategy selection, response planner, guardrail
  domain/         KL Map (graph + BFS), RAG (ingestion, embeddings, retrieval)
  providers/      LLM provider abstraction (Anthropic native, OpenAI-compatible adapter)
  api/            FastAPI routes + SSE
  cli.py
web/              React + Vite SPA
tests/
data/             Runtime state (gitignored): chroma, uploads, sessions
content/          Course content + rule tables (versioned)
```
