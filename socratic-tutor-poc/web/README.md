# web/ — glass-box UI

React + Vite + TypeScript SPA. Four views in one shell:

| Route | Requirement | What it does |
|---|---|---|
| `#/ingest` | R17 | Upload PDF/PPTX/DOCX/MD/TXT/transcripts or a URL; per-file progress over `parse → chunk → embed → kl-extract → review-ready`, fed by SSE |
| `#/tutor` | R19 + R20 | Three panels: chat · TurnTrace inspector · knowledge map |
| `#/klmap` | — | The live course map, layered by prerequisite depth, with the selected turn's nodes highlighted |
| `#/review` | R18 | The human gate: extracted drafts with their validation report, graph + tables, approve/reject |

All routes except `#/tutor` are operator-only, and `#/tutor` itself drops to a bare chat when
operator mode is off — see below.

## Run

```bash
npm install
npm run dev        # http://localhost:5173, /api proxied to http://localhost:8000
npm run build      # → web/dist, served by FastAPI
npm run typecheck
npm test           # vitest
```

Every upload and chat turn is filed under one course id, `SESSION.course_id` in `src/config.ts`
(default `MATH-SEED-01`, the seed course). Set `VITE_COURSE_ID=<id>` at dev or build time to
ingest into, and teach from, a course other than the seed — otherwise the first approved upload
replaces the seed's map in chat for that id.

## Operator mode — a security boundary, not a preference

`src/config.ts` exports `OPERATOR_MODE`. When it is off the app is **the conversation and
nothing else**: no inspector, no knowledge-map panels, no ingestion, no map review, no turn ids,
no developer scenario bar, and operator routes redirect to the chat.

Default: **on** for a local single-user run (`localhost`, `127.0.0.1`, `::1`), **off** anywhere
else, so nobody has to remember to hide the glass box when a student first sits down. Override
explicitly with `VITE_OPERATOR_MODE=true|false`.

Why the boundary exists: the trace is an operator artefact. Guardrail events describe what was
withheld and why, retrieval shows course evidence, and the review screen can make a map live.
The inspector sits beside the chat, which is where a student would be reading.

### This flag is not the boundary — read this before relying on it

`OPERATOR_MODE` hides surfaces. It does not protect data. The trace still arrives over the wire,
so anyone with the browser's network tab sees everything the inspector would have shown, in both
modes. **A client-side gate presented as a security control is worse than none, because it stops
someone building the real one.**

In a multi-user deployment the server is the boundary:

- send `trace` frames only to an authenticated operator session;
- keep `/api/ingest/*` and `/api/klmap/drafts/*` behind the same check — the review endpoints can
  make a knowledge map live, which is a privileged write, not a read;
- then expose `GET /api/session/capabilities` → `{ operator: boolean }`, which the SPA reads at
  boot and uses in place of the host-name rule (approved shape, to be routed to the backend once
  auth exists).

Keeping the host-name rule as the offline default is deliberate: if the capabilities call fails
or is missing, a non-local client resolves to student view, so a misconfigured deployment fails
safe rather than open.

### What the inspector actually renders, worst payload first

Whoever builds the authenticated version needs to know which sections carry what, because the
answer is not uniform — some of it is metadata and some of it is course material.

| Section | Payload | Why it matters |
|---|---|---|
| Retrieval | **Full text of every retrieved chunk**, plus its source reference | The heaviest payload on the panel and the one worth reconsidering first: for a cohort this is course material rendered per turn, whether or not that turn needed it. A data-minimisation question, not just an access one — source ref + score + a snippet would serve the same diagnostic purpose. |
| Guardrail | What was withheld and what the guardrail did about it | Post-C1 the detail is a non-reversible handle, so the risk is disclosure of *policy behaviour* rather than the answer. Still tells a student that something was withheld and when. |
| Strategy · Response plan | Which rules fired, and the constraints on the turn | Reveals the teaching policy, including that a full solution was withheld — enough to work out what to say next to get a different one. |
| Knowledge map · Topic | Course structure and the student's own level | Course structure is not secret; the stored level is personal data about that student. |
| Intent | Classifier output about the student's message | Personal data, low volume. |
| Timing · Models & versions | Latencies, model ids, table versions | Operational metadata; no student or course content. |

Two related rules the UI now follows:

- **No field in the trace may carry the withheld answer.** The guardrail's `detail` is a
  non-reversible handle (`canonical_answer matched as exact (P03, 5 chars)`), so the inspector
  takes each event's *meaning and severity from its type*, not from the detail string. The
  earlier "reveal the canonical tokens" affordance is gone — it assumed the secret reaches the
  browser, which is the thing being fixed.
- **`deterministic_checker_used` is rendered prominently.** When a deterministic check decided a
  verdict, no amount of student push-back can move it; that is the strongest guarantee in the
  pipeline and reads differently from an LLM comparison.

## Mock backend

`src/config.ts` exports `USE_MOCKS`, default **false**: the app talks to the real API unless
you opt into the mocks. `src/mocks/` provides realistic `TurnTrace` fixtures (including the
Design Document §10 worked example, the homework `x = 7` turn), a bilingual chat simulator
that streams tokens and emits the trace stage by stage, an ingestion progress simulator, the
seed-pack KL map, and two extraction drafts — one deliberately unapprovable (prerequisite
cycle, a relation declared both ways, an edge naming a concept that does not exist).

Opt in for UI work that must not touch the backend — no component touches the flag:

```bash
VITE_USE_MOCKS=true npm run dev
```

Never build with it set. `USE_MOCKS` is a build-time constant: when it folds to `true`,
Rollup tree-shakes `src/api/http.ts` out of the bundle and `web/dist` becomes a simulation
that never calls `fetch` while looking exactly like a working client. That is what the
2026-09-02 "UI looks great but does not really work" report was. `npm run build` now ends
with `scripts/check-bundle.mjs`, which fails if the real client's route strings are missing
from `dist/assets/*.js`. The header badge (`mock data` / `live API`) says which one you are
looking at.

Endpoints the real implementation must provide (see `src/api/types.ts` for the contract):

- `POST /api/ingest/upload` — multipart, returns `[{file_id, filename}]`
- `POST /api/ingest/url` — `{url}` → `{file_id, filename}`
- `GET  /api/ingest/events` — SSE, `{file_id, filename, stage, percent, status, message}`
- `POST /api/chat/turn` — SSE, events `generating` / `token` / `trace` / `done` / `error`.
  `trace` arrives repeatedly as each pipeline step completes (a full trace or a patch — the UI
  merges either), `done` carries the complete trace so a client that missed one converges, and
  `generating` marks the start of the model call.

  **Generation is buffered:** the backend drafts the whole reply, runs the guardrail, and only
  then streams, so a withheld answer never reaches the student even for an instant. That leaves
  a silent 1–2 s gap between the last decision patch and the first token. The chat panel shows a
  composing line and the inspector shows a banner, driven by `generating` when it arrives and
  otherwise derived from "a response plan exists but no token has arrived yet" — so the state
  holds either way. Guardrail events can land **before** any token; the merge is order-independent.
- `GET  /api/klmap?course_id=…` — `KlMap`
- `GET  /api/klmap/drafts` · `GET /api/klmap/drafts/{course_id}` — `KlMapDraft`, which carries
  the raw extracted nodes and edges **plus** the `ValidationReport`, including when validation
  failed: the reviewer has to see what the model proposed in order to fix it
- `POST /api/klmap/drafts/{course_id}/approve` — `{approve, note?}`; must refuse to approve a
  draft with blocking errors, whatever the client sends
- `POST /api/klmap/drafts/{course_id}/revalidate` — re-read the YAML from disk after an edit

Validation issue codes come from `domain/klmap/loader.py` (`KL0xx` errors, `KL1xx` warnings) and
`domain/rag/kl_extract.py` (`KX1xx`, proposals the extractor dropped). Each issue's `location`
(`edges[7]`, `nodes[2]`) anchors it to a row in the tables and to the node or edge in the graph.

**One thing worth sending from the backend:** for `KL009` (prerequisite cycle), a structured
`path: string[]` on the issue. The UI highlights the whole cycle; without that field it parses
the path out of the message text, which works today but breaks silently if the wording changes.
See `cyclePath` in `src/components/review/issues.ts`.

## Tests

`npm test` covers the two things that fail silently:

- `src/components/inspector/summary.test.ts` — the summary strip is the system's own account of
  what it did. It must never claim a constraint the trace does not carry (or omit one it does).
- `src/api/sse.test.ts` — the POST-SSE parser: partial frames, multi-line `data`, CRLF, comments,
  UTF-8 split across chunk boundaries, error responses.
- `src/components/review/issues.test.ts` — issue anchoring and the cycle-path fallback parse.

Presentational components are covered by build + typecheck + a manual browser pass, not by tests.

## Conventions

- Colours follow the design document: amber = LLM steps, blue = deterministic logic,
  green = data/knowledge, red = policy overrides and guardrail. Monospace for structured values.
  Relation types get their own five-colour encoding in the graphs — downstream the lookup
  collapses `related_to`/`part_of`/`uses` into one bucket, but a reviewer is checking exactly
  that distinction, so the graph never collapses it.
- Thai text carries `lang="th"` so the `"Noto Sans Thai", "Sarabun"` stack and its line height apply.
- Plain CSS in `src/styles.css`; no component library, no graph library (both SVG renderers
  compute their own layout, including cycle-safe layering for drafts under review).
