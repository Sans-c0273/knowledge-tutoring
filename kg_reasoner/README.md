# kg_reasoner

Identifies which nodes in a `kg-mapper-poc` knowledge graph should support teaching feedback for one student turn — the reasoner between your knowledge graph and the Pedagogy Layer's Strategy Selection. See `DESIGN.md` for the full write-up (why a reasoner, how the rules map to your two edge schemas, what's still a gap).

## Setup

Requires Python 3.10+. No runtime dependencies — only `pytest` for running the tests.

**With [uv](https://docs.astral.sh/uv/) (recommended):**

```sh
git clone https://github.com/nvannaprathip/kg_reasoner.git
cd kg_reasoner
uv sync --extra dev
```

**With plain venv + pip:**

```sh
git clone https://github.com/nvannaprathip/kg_reasoner.git
cd kg_reasoner
python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Verify the install:

```sh
PYTHONPATH=src python3 -m pytest    # should show 25 passed
```

To run against a real knowledge graph instead of the test fixtures, clone `kg-mapper-poc` as a sibling folder and run `kg ingest` / `kg render` there first — `kg_reasoner` just needs the resulting `_index.json` path (see Quickstart below). It has no import-time dependency on the `kg` package itself.

## Quickstart

```sh
PYTHONPATH=src python3 -m pytest          # offline, no network, no LLM calls (25 tests)

PYTHONPATH=src python3 -m kg_reasoner.cli \
  --index tests/fixtures/education_index.json \
  --topic "Linear Equations" \
  --intent solve \
  --learner-state confused
```

Or against a real graph once you've run `kg ingest` / `kg render` in `knowledge-graph-poc`:

```sh
PYTHONPATH=src python3 -m kg_reasoner.cli \
  --index ../knowledge-graph-poc/data/graph/_index.json \
  --topic "Photosynthesis" \
  --intent clarify \
  --learner-state confused
```

## Library usage

```python
from kg_reasoner import StudentContext, reason
from kg_reasoner.graph_index import load_index

index = load_index("../knowledge-graph-poc/data/graph/_index.json")
ctx = StudentContext(topic_query="Photosynthesis", intent="clarify", learner_state="confused")
guidance = reason(ctx, index)
print(guidance.to_dict())
```

`guidance.to_dict()` returns the `knowledge_guidance` object described in Part 2 of the design docs — hand it to Strategy Selection alongside the existing Trigger Matrix output.

## Layout

```
src/kg_reasoner/
  models.py         StudentContext, CandidateNode, KnowledgeGuidance (pure data)
  graph_index.py     loads data/graph/_index.json into a traversal-ready index
  anchor.py          topic text -> anchor node
  traverse.py        relation-filtered, depth-limited BFS
  rules.py           Tier 1: deterministic KM01-KM05, schema-aware
  llm_reasoner.py     Tier 2 (optional): LLM re-rank, roster-constrained
  pipeline.py        orchestrates the above -> KnowledgeGuidance
  cli.py             python -m kg_reasoner.cli ...
tests/               25 tests, fixtures rebuilt from the design docs' own worked examples
DESIGN.md            full design write-up
DEMO.md              a run against the real knowledge-graph-poc sample corpus
```

## Status

Prototype: Tier 1 (rules) is implemented and tested against both `education` and `general` schema fixtures, plus smoke-tested against the real `knowledge-graph-poc` sample corpus (see `DEMO.md`). Tier 2 (LLM re-rank) has its contract and tests but no wired-up provider yet — plug in your own `LLMCall`, or adapt one of `kg-mapper-poc`'s existing `src/kg/llm/adapters/*.py`.
