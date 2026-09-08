# kg-mapper-poc

**What it does:** A proof of concept that ingests mixed-format source files (slides, PDFs, Word docs, spreadsheets, web pages, images) and produces a **knowledge graph of atomic notes** — Zettelkasten-style Markdown files with typed, traceable relationships between ideas. The graph is validated against three success criteria (manual inspection, stability ≥0.70 Jaccard, relevance-score monotonicity) and rendered as a browsable, single-file HTML view.

**Why:** Course materials and knowledge documents arrive as monolithic files. Learners and knowledge workers cannot see the *structure* inside them — which ideas depend on which, which ideas recur across sources, and how strongly ideas relate. This POC tests whether an LLM can decompose sources into atomic notes reliably enough to build on.

**Status:** Design approved 2026-08-25; rebuild in progress. Never run against a live LLM yet — first real ingest is the next step.

---

## Quickstart

### Setup

```sh
brew install uv                   # if not installed
uv sync                           # Python 3.12 via .python-version; writes .venv
cp .env.example .env              # only if using OpenRouter or Anthropic API (see below)
uv run kg selftest                # offline test suite, spends no tokens
```

### Choose a provider

Edit `kg.yaml` → `llm.provider` to select one of:

| provider | credential | how to set up | notes |
|---|---|---|---|
| `claude_subscription` (default) | none | Run `claude login` once in Claude Code. Uses your existing Claude Pro/Max subscription login. | *Terms note (§0.1, DESIGN.md):* subscription usage is permitted for your own individual use only. The policy is in flux; if it tightens, the API-key adapter is a one-line switch. |
| `openrouter` | `OPENROUTER_API_KEY=<key>` in `.env` | Visit `https://openrouter.ai`, create an account, copy your API key into `.env`. | Per-call costs; soft budget cap in `kg.yaml`. Per-stage models are configurable. |
| `anthropic_api` | `ANTHROPIC_API_KEY=<key>` in `.env` | `uv sync --extra anthropic-api`, then set the key in `.env`. | Console API key; per-call costs; prices in `kg.yaml` `llm.anthropic_api.pricing_usd_per_mtok`. |

### Ingest and render

```sh
# Drop files into data/inbox/ and optionally list URLs in data/inbox/urls.txt
# Then:

uv run kg ingest --dry-run        # Convert only; estimate cost; write converted/ folder
uv run kg ingest                  # Full pipeline: convert → extract → edges → dedup → render
uv run kg render                  # Rebuild graph.html from the notes (view-only)
open data/graph/graph.html        # Browse the graph
```

---

## Commands

| command | model calls | what it does | exit code |
|---|---|---|---|
| `kg ingest [--dry-run] [--provider PROVIDER] [--write-same-as]` | yes (no in dry-run) | Run the full pipeline (S0–S8): discover → convert → chunk → extract → consolidate → edges → dedup → write. `--provider` overrides `kg.yaml` for this run. `--write-same-as` commits duplicate edges; default queues them for review. | 0 only when every file ingested (`status: ok`); 1 when any file was deferred or failed (`partial` / `failed`) or the run could not complete. A report is written either way. |
| `kg gates` | no | Validate the notes on disk against the schema. | 0 if all pass; 1 if any gate fails. |
| `kg render [--stamp]` | no | Rebuild `_index.json` and `graph.html` from the notes. View-only; never modifies notes. `--stamp` adds the current time to the HTML metadata (default: omitted for deterministic output). | 0 on success; 1 on error. |
| `kg stability [--keep] [--allow-served-drift]` | yes (2× ingest) | Ingest the inbox twice into clean folders and report node-set agreement (Jaccard ≥0.70). No `--provider` flag: both legs share the config. `--keep` retains both legs' output folders. `--allow-served-drift` treats served-model mismatches between legs as a warning instead of invalidating. | 0 if passed and valid; 1 if failed or invalid. |
| `kg sample-relevance [--n N] [--seed S]` | no | Draw a stratified sample of `related_to` edges (by relevance band) and write a review file for human judgment. | 0 on success; 1 on error. |
| `kg summarise-relevance <file>` | no | Read a filled review file and report agreement rate per band and the monotonicity verdict (§8(c), DESIGN.md). | 0 on success; 1 on error. |
| `kg selftest` | no | Run the offline test suite with credentials unset. Spends no tokens. Any further arguments are passed to pytest (e.g. `kg selftest -x -k sandbox`; a leading `--` is accepted but not required). | pytest exit code. |

---

## Supported inputs and locator formats

`kg.yaml` → `paths.inbox` is the inbox folder. Web URLs are listed in `inbox/urls.txt` (one URL per line; comments and blank lines ignored).

| format | source | how it's processed | locator in the notes | example |
|---|---|---|---|---|
| `.txt` | plain text | split into paragraph groups | `file.txt#lines=12-48` | — |
| `.md` | Markdown | split by headings (ATX style); preamble by line range | `file.md#heading=Intro > Terms` or `file.md#lines=1-15` | — |
| `.pdf` | PDF (text-based, one unit per page) | extracted page by page; empty pages flagged | `file.pdf#page=4` | — |
| `.pptx` | PowerPoint | split on slide boundaries; speaker notes included | `file.pptx#slide=7` | — |
| `.docx` | Word | split by heading sections; falls back to paragraph blocks | `file.docx#heading=1 Scope > 1.2 Terms` or `file.docx#para=31-45` | — |
| `.xlsx` | Excel (cached formula values only; merged cells use top-left) | split by sheet; blocks split at blank rows; tables with row/col headers | `file.xlsx#Sheet1!A3:F40` | — |
| `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp` | images and diagrams | described by a vision model; description becomes the Markdown source | `file.png` (whole-image locator) | — |
| URL in `urls.txt` | web pages | fetched and converted to Markdown; boilerplate removed | `https://example.com/path#heading=Section > Sub` (original URL, not stem) | — |
| *any other extension* | — | rejected (unsupported, never skipped) | — | `.xls`, `.xlsx-old`, `document.zip` → stay in inbox, listed in report |

**Locator validity:** A script can resolve ≥95% of locators back to the source. Every node's frontmatter carries `sources: [{locator, file}]`; the `file` field holds the original filename or URL (unchanged by stem de-duplication). The rendered `## Source` section quotes the verbatim source text at that locator.

---

## Output layout

After a successful `kg ingest`:

```
data/graph/
├── nodes/<id>-<slug>.md          # atomic notes: frontmatter + definition + relations + source
├── _registry.yaml                # master registry: every node's id, title, status, file path
├── _index.json                   # derived index for the HTML view (regenerated on render)
├── _review/                      # human-review queues
│   ├── duplicates.md             # near-duplicate pairs (if any) queued for manual judgment
│   ├── grounding.md              # quotes flagged as not found verbatim in source
│   └── relevance-sample-<ts>.md  # stratified sample of related_to edges for verdict entry
├── _reports/
│   ├── run-<ts>.md               # human-readable run report (inputs, schema, cost, events)
│   └── run-<ts>.json             # machine-readable equivalent
└── graph.html                    # single self-contained HTML file; open in a browser

data/processed/                   # source files moved here after successful ingestion
data/converted/                   # intermediate Markdown + metadata (from S1 convert stage)
data/runs/                        # stability legs and Agent SDK workspace
```

---

## The two edge schemas

Determined by `kg.yaml` → `corpus.schema`. Every run uses exactly one schema; edges of the other type are rejected and logged.

### `education` schema (for course materials)

| edge type | direction | scored? | meaning | notes |
|---|---|---|---|---|
| `prerequisite_of` | A → B | no | A must be understood before B | DAG enforced (no cycles) |
| `part_of` | A → B | no | A is a component of the larger topic B | — |
| `example_of` | A → B | no | A is a concrete instance of concept B | — |
| `refines` | A → B | no | A is a more precise/specialised statement of B | — |
| `supersedes` | A → B | no | A replaces B (newer version of the idea) | — |
| `same_as` | A ↔ B | no | A and B are the same idea (duplicate) | symmetric |

Every node also carries `origin: course_material | external_knowledge` (frontmatter). In this POC all ingested nodes are `course_material`; the field exists so future exploratory nodes can be distinguished.

### `general` schema (for knowledge bases)

| edge type | direction | scored? | meaning | notes |
|---|---|---|---|---|
| `related_to` | A ↔ B | **yes** | A and B are meaningfully connected; score = model confidence (0–100) | symmetric; scored edges only |
| `part_of` | A → B | no | A belongs to the logical grouping / topic B | — |
| `same_as` | A ↔ B | no | A and B are the same idea (duplicate) | symmetric |

---

## Validation workflow (success criteria §8)

Three gates must pass before the POC is deemed reliable:

### (a) Manual read — end-to-end runs, output reads sensibly
- Run `kg ingest --dry-run` on your files first to inspect `data/converted/*.md`.
- Run `kg ingest` to extract the full graph.
- Open `data/graph/graph.html` and spot-check ≥20 notes for accuracy and definition quality.
- Look for systematic nonsense (hallucinations, mangled text, missing locators). A few isolated errors are expected.

### (b) Stability — Jaccard ≥ 0.70 per corpus
- Run `kg stability` to ingest the same inbox twice into clean folders.
- Node-set agreement is computed as Jaccard (|intersection| / |union|) on normalised titles.
- **Bar: ≥ 0.70 Jaccard.** If stability is lower, node identity is not stable enough; the fix is human-authored identifiers, not more prompt tuning.
- The report shows `only_in_a` and `only_in_b` with locators so you can see which nodes diverged.

### (c) Relevance judgment — agreement rate monotonic across bands
- Run `kg sample-relevance` to draw a stratified sample of `related_to` edges (split into low/medium/high relevance bands).
- Fill in the `verdict` column manually: `agree` or `disagree` for each edge.
- Run `kg summarise-relevance <file>` to compute agreement rates per band.
- **Bar: agreement rate is monotonic** — high-relevance edges agreed-with more often than low ones.
  - `monotonic`: rates are non-decreasing from low to high band. ✓
  - `not monotonic`: some higher band agrees less often than a lower one. ✗
  - `indistinguishable`: every band within ±10% of the others; the score carries no information. ✗
  - `insufficient data`: fewer than 5 judged rows in any configured band. ✗ (rerun with a larger sample.)

---

## Cost and rate limits

### Subscription provider (`claude_subscription`)
- **Cost:** Shared across all Claude apps on your plan's 5-hour/weekly rolling window. No per-token charge. Equivalent list-price estimate (for sizing): ~$0.31 per sample ingest (≈500k tokens total).
- **Rate limits:** If a 5-hour window is exhausted mid-run, the pipeline waits once (if the window resets within `llm.claude_subscription.rate_limit_wait_minutes`; default 30). A second rate-limit hit defers remaining files, and `kg ingest` is re-run later to resume.
- **Full course:** ≈1.3M input + 0.3M output tokens; likely spans >1 window on Pro. Max plan (5× more per month) is comfortable.

### OpenRouter (`openrouter`)
- **Cost:** Per-call billing; run report shows cost per stage. Default models: ~$0.30 per sample ingest ($0.002 describe + $0.28 atomize/edges + $0.01 dedup). Full course ~$5–6.
- **Soft budget:** `kg.yaml` → `limits.soft_budget_usd` (default 2.00). Once priced spend exceeds it, model stages stop and the remaining files are deferred (a `budget_exceeded` event in the report); re-run `kg ingest` later to resume. The check only counts priced calls, so it never applies to the subscription provider.
- **Rate limits:** OpenRouter retries 429 responses up to 3 times with backoff; persistent 429s defer the file.

### Anthropic Console API (`anthropic_api`)
- **Cost:** Per-token pricing from `kg.yaml` → `llm.anthropic_api.pricing_usd_per_mtok`. Prompt caching reduces input cost ~⅓ once warm. ~$0.31 per sample ingest at Sonnet 5 list price.
- **Rate limits:** SDK retries 429 responses up to 3 times; persistent 429s defer the file.

---

## Sandbox rules (PRD §3, §9; DESIGN §23)

The pipeline is fully sandboxed:

- **Reads/writes only in `data/`:** Every filesystem operation resolves through `kg.yaml` → `paths.*`; roots must be inside the project directory.
- **Never touches TK-PKA or the knowledge base:** Hard-coded denylist refuses any path containing `/TK-PKA/` or `01-knowledge-base`, or resolving under `~/knowledge-base` (symlinks are resolved before the check). Violation raises `SandboxViolation`.
- **Agent SDK isolation (subscription provider):** The subprocess runs with `tools=[]`, `setting_sources=[]`, an empty `cwd` under `data/runs/`, and the other providers' API keys blanked in its environment (`ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`), so it loads no `~/.claude` settings, hooks, memory, or `CLAUDE.md` and cannot read or write files.
- **Transcript cleanup (subscription provider):** After each call, the transcript is removed from `~/.claude/projects/<slug>/` so no prompt text lingers.

---

## Design rules worth knowing

These are built in and not configurable; they're recorded here for context:

- **Notes are append-only, IDs permanent:** Nodes get a permanent `<corpus-slug>-<NNNN>` ID on first write. IDs are never reused, even if a note is rejected and discarded.
- **Wikilinks are idempotent:** Definitions are stored plain (no links); `## Relations` and `## Source` are regenerated on every `kg ingest`, so running ingest twice over an already-ingested graph produces byte-identical output (R11, D32).
- **Gates block edges, never delete notes:** Edge rejections (out-of-schema, cycle, dangling target, etc.) are logged to `_review/rejected-edges.jsonl`; the edge is not written. No note or edge is ever deleted — only deferred or queued.
- **Duplicate handling is queue-first:** By default, near-duplicate pairs are queued for human review (not written as `same_as` edges). Use `--write-same-as` to commit edges without review.
- **Sources are moved, never deleted:** After successful ingestion, files move to `data/processed/`. Before ingestion, they are listed in the inbox. File counts before and after are verified in the report.

---

## Full documentation

Design and decision records live outside the repo, in the project workspace:

- **`docs/PRD.md`** — requirements and success criteria (§8).
- **`docs/DESIGN.md`** — architecture, stack, data model (§8), LLM boundary (§7), gates (§10), CLI (§18), sandbox (§23).
- **`docs/DECISIONS.md`** — append-only log of key decisions (D1–D34).
- **`docs/TECH_DEBT.md`** — known out-of-scope improvements and build-time items.
- **`docs/CHANGELOG.md`** — human-readable history of shipped changes.

Path: `~/claude-workspace/kg-mapper-poc/docs/`

---

## Not built in this POC

These are recorded in TECH_DEBT.md or PRD §4 "Out of scope":

- Per-student records, mastery, or coverage.
- Any database, server, or live service — files and CLI only.
- OCR of exact text from images (vision-model descriptions are the bar).
- `origin` tag enforcement (descriptive only).
- Cross-course node merging (workflow is one corpus at a time).
- Exploratory-learning node expansion beyond source materials.
- Any link to TK-PKA, `01-knowledge-base/`, or `kb-ingest`.
- Embedding-based duplicate detection (token similarity for now).
- Per-student anything.
- Batch API (full-course cost optimisation).
- `kg report` and `kg providers` convenience commands (data already in run reports).
- Parallel model calls.
- Embedded images in PPTX/DOCX (alt text only; full image extraction future).
- OCR path for scanned PDFs.

---

## If stability or relevance fails

**If stability (b) is below 0.70:** Node identity is not stable enough. The answer is not more prompt tuning but human-authored identifiers, or evidence that the prompt + model combination is fundamentally unreliable on your domain. The diff in `data/runs/stability-<ts>/{a,b}/` shows exactly which nodes appeared in only one leg, with their locators and sources, so you can diagnose whether the disagreement is granularity (one run splits finer) or substance.

**If relevance (c) is not monotonic:** The relevance scoring is not informative. Either the model is guessing (try a stronger model), or the scoring prompt needs revision. Without monotonicity, the feature should be dropped.

**Provider-specific effects:** Both legs use the same provider and models. If you suspect a provider effect, re-run stability on a different provider by editing `kg.yaml` and running again. The run reports record provider and per-stage models, so past results stay interpretable.

---

## Exit codes

| command | 0 | 1 |
|---|---|---|
| `ingest` | every file ingested (`ok`) | any file deferred or failed (`partial` / `failed`), or the run aborted |
| `gates` | all pass | any gate fails |
| `render` | success | error |
| `stability` | passed and valid | failed or invalid |
| `sample-relevance` | success | error |
| `summarise-relevance` | success | error |
| `selftest` | all tests pass | any test fails |

---

## Development and testing

```sh
uv run kg selftest          # offline suite, no tokens

# with pytest flags:
uv run kg selftest -x       # stop on first failure
uv run kg selftest -k gates # run only tests matching "gates"

# manual single-stage test:
uv run python -m pytest tests/test_WU5_stability.py -xvs
```

All tests run without credentials and with fake adapters, so no real API calls are made.
