# Demo: run against a real graph

Everything below is from an actual run against `knowledge-graph-poc`'s sample corpus (`data/graph/_index.json`, 224 nodes / 326 edges, `general` schema, ingested from the Mario game-design PDFs) — not the tests' synthetic fixtures. Confirms the pipeline works end to end against real, non-toy ingester output.

## Confused student, asking to clarify

```sh
PYTHONPATH=src python3 -m kg_reasoner.cli \
  --index ../knowledge-graph-poc/data/graph/_index.json \
  --topic "Basic Enemy" \
  --intent clarify \
  --learner-state confused
```

```json
{
  "knowledge_guidance": {
    "schema": "general",
    "anchor": { "id": "sample-0074", "title": "Basic Enemy", "role": "anchor", "distance": 0 },
    "selections": [
      {
        "id": "sample-0057",
        "title": "MVP Scope",
        "role": "prerequisite_gap",
        "relation": "part_of (broader concept, proxy)",
        "distance": 1
      },
      {
        "id": "sample-0076",
        "title": "Enemy State Machine",
        "role": "alternative_explanation",
        "relation": "related_to",
        "distance": 1,
        "relevance": 80
      },
      {
        "id": "sample-0072",
        "title": "Stomp",
        "role": "alternative_explanation",
        "relation": "related_to",
        "distance": 1,
        "relevance": 75
      }
    ],
    "notes": [
      "KM02 (confused): 1 prerequisite_gap candidate(s)",
      "KM03 (clarify): 2 alternative_explanation candidate(s)"
    ]
  }
}
```

KM02 and KM03 both fired (confused + clarify). Since this corpus is `general` schema, the "gap" is the `part_of` parent (MVP Scope) — flagged explicitly as a proxy, not a diagnosis, per the design doc's own warning. The two `alternative_explanation` candidates are the anchor's two highest-relevance `related_to` neighbors (80, 75).

## Student wants practice

```sh
PYTHONPATH=src python3 -m kg_reasoner.cli \
  --index ../knowledge-graph-poc/data/graph/_index.json \
  --topic "Enemy State Machine" \
  --intent practice_quiz
```

```json
{
  "knowledge_guidance": {
    "schema": "general",
    "anchor": { "id": "sample-0076", "title": "Enemy State Machine", "role": "anchor", "distance": 0 },
    "selections": [
      { "id": "sample-0022", "title": "Koopa Troopa", "role": "practice_target", "relation": "related_to", "relevance": 70 },
      { "id": "sample-0023", "title": "Piranha Plant", "role": "practice_target", "relation": "related_to", "relevance": 70 }
    ],
    "notes": ["KM04 (practice_quiz): 2 practice_target candidate(s)"]
  }
}
```

Both practice targets carry source quotes (`grounded: true`, omitted above for brevity) — safe to generate a quiz question from without inventing content.
