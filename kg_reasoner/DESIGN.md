# kg_reasoner — design

**What this is:** the missing piece between your knowledge graph (`kg-mapper-poc`) and the Pedagogy Layer design docs ("Teacher-like chatbot — final v2", Parts 1 & 2). Given one student turn's pedagogical situation, it identifies *which graph nodes* should support the teaching feedback, and *why* — the `knowledge_guidance` object that Strategy Selection consumes.

It does **not** choose the teaching strategy (S01–S08). That stays exactly where the design docs put it — the Trigger Matrix, driven by `intent` + `learner_state` + `special_handling` + `student_level`. This module answers a narrower question: *"which concept(s), if any, should back that strategy up?"* (Part 2, p.4).

## Why a reasoner, not just BFS

Part 1 of the design doc is explicit that MVP1 doesn't need "an AI reasoner" — relation-filtered BFS plus a handful of `KM` rules is enough, because the worked examples there use a 5-node toy graph. Your actual graph is different in a way that matters: `kg-mapper-poc` ingests real course material into potentially hundreds of atomic notes with two possible edge vocabularies (`education` vs `general` — see `src/kg/schemas.py`), and which edges exist depends on which schema a corpus was ingested under. A fixed 5-rule table tuned to one toy graph doesn't survive that. What you actually need is:

1. The same "traversal isn't reasoning" discipline the docs already insist on (Part 1, p.36).
2. Rules that are schema-aware, because `education` gives you a directed `prerequisite_of` DAG and `general` doesn't — the two schemas need different logic to answer the same pedagogical question ("what's the gap?", "what's next?").
3. Room to fall back to LLM judgment *only* when topology genuinely can't decide (too many similarly-plausible candidates) — not as the default path.

This design keeps Tier 1 (rules) as the default, exactly as the source docs recommend, and adds Tier 2 (LLM) as strictly optional.

## Data flow

```
Student Message
   |
Intent Detection + Student-Level Selection      (unchanged — your existing Pedagogy Layer)
   |
   v
StudentContext { topic_query, intent, learner_state, student_level, special_handling, exclude_node_ids, mastered_node_ids }
   |
   v
+----------------------------- kg_reasoner --------------------------------+
|  1. anchor.py     topic_query -> one graph node (exact title/alias,      |
|                    then token-overlap fallback)                         |
|  2. rules.py       KM01-KM05, schema-aware, evaluated in priority order  |
|                    -> list[CandidateNode] with a Role and a rule trace   |
|  3. llm_reasoner.py (OPTIONAL) re-rank/narrow when Tier 1 returns more   |
|                    similarly-plausible candidates than you want to show |
+----------------------------------------------------------------------------+
   |
   v
KnowledgeGuidance { anchor, selections: [CandidateNode...], notes }
   |
   v
Strategy Selection (Trigger Matrix — unchanged)   -> primary/supporting strategy
   |
   v
Response Planner (unchanged) -> LLM -> student-facing response
```

`kg_reasoner` reads `data/graph/_index.json`, the file `kg render` / `kg ingest` already produce. It has **no import-time dependency on the `kg` package** — deliberately, so it can be deployed as a separate service/process from the ingester. The trade-off: `graph_index.norm_title` is a simplified re-implementation of `kg.normalise.norm_title` (no plural-stripping, no Thai trigram handling). If you'd rather have exact parity with the ingester's own title matching, add `kg-mapper-poc` as an editable dependency (`uv add --editable ../knowledge-graph-poc` from this project) and import `kg.normalise.norm_title` / `kg.graph_io.load_graph` directly instead — `graph_index.py` is a thin enough shim that the swap is mechanical.

## Node anchoring

Unchanged from Part 1's recommendation (p.31): exact normalised title/alias match first ("you don't need sophisticated reasoning" for this step, and it's still true at 224 nodes). A token-Jaccard fallback (`best_fuzzy_match`, threshold 0.5) catches near-misses like "carbon dioxide gas" → **Carbon Dioxide**. No embeddings, no LLM call for this step — add embedding similarity only if you start seeing topic strings that genuinely don't share vocabulary with any title/alias (e.g. paraphrased student questions rather than teacher-curated topic labels).

## Traversal

`traverse.bfs` is exactly the Part 1 recommendation: breadth-first, relation-filtered, depth-limited (1–2 hops). The one addition beyond the docs' toy example is direction-awareness, because `education`'s edges are directed and mean different things forwards vs backwards:

- `prerequisite_of` walked **backwards** from the anchor = "what should this student already know" (a gap candidate).
- `prerequisite_of` walked **forwards** = "what does this unlock next" (an extension candidate).

`general`'s `related_to` is symmetric, so direction doesn't apply there — but relevance score does, and every general-schema rule sorts by it.

## The rule table (Tier 1)

Generalises KM01–KM05 (Part 2, p.4) from the toy graph to your two real schemas. Full mapping in `rules.py`; summary:

| Rule | Condition | Role | `education` schema source | `general` schema source |
|---|---|---|---|---|
| KM02 | `learner_state == confused` | `prerequisite_gap` | nearest unmastered `prerequisite_of` ancestor (depth ≤2) | `part_of` parent (broader/simpler framing), else strongest `related_to` — **explicitly flagged as a proxy**, never asserted as a diagnosis |
| KM03 | `intent == clarify` | `alternative_explanation` | `refines`/`supersedes`/`same_as` partner; else a `part_of` sibling | strongest unused `related_to` neighbor(s) |
| KM04 | `intent == practice_quiz` | `practice_target` | `example_of` children, grounded (has a source quote) preferred | `part_of` children, else `related_to`, grounded preferred |
| KM05 | `learner_state == normal` and `intent == summarize_review` | `next_challenge` | forward `prerequisite_of` (this topic unlocks X) | **no candidate** — `general` has no directed progression edge; see below |
| KM01 | default (nothing else fired) | `supporting_context` | `part_of` parent + `example_of` children | top `related_to` by relevance |

Rules combine (a confused student asking to clarify fires both KM02 and KM03) and are capped/deduplicated at `max_selections`, evaluated in the precedence order `special_handling > learner_state > intent`, matching the Trigger Matrix's own precedence (Part 1, p.16–17).

Every "gap"/"alternative" selection carries Part 2's own warning forward literally: **a prerequisite (or its proxy) existing does not mean the student has a problem with it.** That framing has to survive into however you prompt the response-generating LLM — this module hands you a candidate to *check or scaffold*, never a diagnosis.

### student_level: breadth/depth, not tone

`student_level` (beginner/intermediate/advanced) tunes how much of the graph a turn draws on — never wording or which KM rule fires; that split stays exactly where the rest of this doc puts it (Strategy Selection's job). Three levers, set per level in `rules.LEVEL_PROFILES`:

| Level | max_candidates (overall cap) | max_depth (KM02 `education` BFS) | min_relevance (`general` `related_to` floor) |
|---|---|---|---|
| beginner | 2 | 1 hop | 70 |
| intermediate | 3 | 2 hops | 50 |
| advanced | 5 | 2 hops | 0 (no floor) |

`max_candidates` is enforced once, centrally, in `apply_rules` — not inside each rule function — so an explicit `max_selections` override always wins regardless of level, and per-rule functions stay free to return everything that qualifies. `min_relevance` has a floor-relaxation safety net (`_relevance_filtered`): if the floor would leave a rule with zero candidates, it falls back to the single strongest neighbor anyway, tagged in `relation` as relaxed — a stricter level must never end up with *less* grounding than a looser one would have offered, just a higher bar most of the time. `max_depth` only affects KM02's `education`-schema BFS today; every other rule is single-hop by construction. `KM05` (`next_challenge`) is deliberately excluded from level-scaling — "what's next" is singular at every level, not a breadth question.

### Two things this surfaced about your schema that are worth a decision

1. **`general` schema has no progression/"next topic" edge.** If you want the "student understands X, suggest what's next" behavior from Part 1's Student-Level Selection story for a `general`-schema corpus, you need one of: (a) ingest that corpus under `education` instead so `prerequisite_of` exists, or (b) keep topic sequencing entirely in the teacher-authored syllabus list (outside the graph, as Part 1's Student-Level Selection already does) rather than expecting the graph to answer it. `rules.py` documents this in a `notes` entry rather than silently returning nothing.
2. **Neither schema has a misconception edge or node type**, and Part 1 (Table C, NOTE1) explicitly deferred misconception handling for the same reason. If/when you add one — either a `misconception_of` edge in the `education` vocabulary, or a `kind: misconception` frontmatter field — add a `KM06` rule keyed on `learner_state == confused` with a misconception at distance ≤1, role `misconception_check`. The rest of this design (anchor → traverse → rank) doesn't change; only the rule table grows.

## Tier 2: optional LLM re-ranking

`llm_reasoner.rerank` exists for the case Tier 1 alone can't resolve well: several `related_to` neighbors within a few relevance points of each other, or several `part_of` siblings, where a semantic judgment call ("which of these is actually the best alternative explanation for *this* student") beats a fixed sort order. It is opt-in (`pipeline.reason(..., llm_call=...)`) and only invoked when Tier 1 returns more candidates than you asked to show.

The contract mirrors `kg-mapper-poc`'s own prompting convention in `prompts/edges.py` / `prompts/dedup.py`: the model is handed a **roster** of candidate ids and definitions and must choose only from it. Any id it returns that wasn't in the roster is silently dropped — never trusted — the same posture as `kg.gates` rejecting edges that reference unknown ids. If the call raises or returns nothing usable, Tier 1's own ordering is returned unchanged; Tier 2 can only narrow or reorder, never fail the turn.

`llm_reasoner.py` has no SDK dependency — you implement `LLMCall` against whatever provider you use. If you want to reuse `kg-mapper-poc`'s own adapters (`src/kg/llm/adapters/*.py`, which already implement a `complete(system, messages, schema_json, model, max_tokens)` surface for structured output), wrap one of those to match the `LLMCall` protocol rather than writing a new client.

## What Strategy Selection does with this

Exactly the pattern in Part 2 (pp.2–3, 6–11): `knowledge_guidance` is read *alongside* the existing Trigger Matrix output, not instead of it. The Trigger Matrix still decides the strategy (S01–S08) from `intent` + `learner_state` + `special_handling` + `student_level`; `knowledge_guidance.selections` tells the Response Planner *which concept(s)* to reference while executing that strategy — e.g. "Re-explain Differently, using **Light Energy**" rather than "Re-explain Differently" with no target. `special_handling` (homework/assessment) is deliberately **not** re-implemented here — Teaching Policy (P01–P04) already owns those constraints (don't reveal the answer, hint-first, etc.); this module only adds a note reminding the caller that policy still applies, so the two layers don't get entangled (Part 2, p.4: "your Knowledge Map and Strategy Selection become unnecessarily entangled" is exactly the failure mode being avoided).

## Testing

`tests/fixtures/education_index.json` and `general_index.json` are hand-built graphs shaped like `_index.json`, deliberately reconstructed from the design docs' own worked examples (the Basic Arithmetic → Inverse Operations → Linear Equations chain from Part 1 p.30–38, and the Photosynthesis graph from Part 2 p.5–11) — so the tests double as a check that this implementation reproduces the source docs' own reasoning, not just an internally-consistent one. `test_pipeline.py` covers the end-to-end path including Tier 2 failure modes (provider error, hallucinated id). Run with `PYTHONPATH=src python3 -m pytest`.

The package was also smoke-tested against your real `knowledge-graph-poc` sample corpus (224 nodes / 326 edges, `general` schema) — see `DEMO.md`.

## Suggested next steps

1. Point `--index` at a real course corpus (ingested under whichever schema fits) instead of the Mario sample, and sanity-check `selections` against a handful of the same $\ge$20 notes you already spot-check manually per the ingester's own validation workflow.
2. Decide on the `general`-schema progression gap above (education schema vs. syllabus-order) before you rely on KM05 for a `general` corpus.
3. Only if Tier 1 output feels topologically-correct-but-tone-deaf for some intents, wire up Tier 2 with your existing LLM adapter and compare a sample of turns with/without it — don't turn it on by default.
