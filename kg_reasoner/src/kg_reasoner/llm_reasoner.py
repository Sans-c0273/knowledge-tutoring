"""Tier 2 (optional): LLM-based re-ranking when graph topology alone is ambiguous.

Per both source docs, Tier 1 (rules.py) is sufficient for MVP — "You don't
need that sophisticated reasoner in MVP1" (Part 2, p.36). Reach for this
module only when Tier 1 hands back too many similarly-plausible candidates
for the same role (e.g. four `related_to` neighbors within a few relevance
points of each other) and a semantic judgment call would help.

Contract, deliberately mirrored on kg-mapper-poc's own prompts/edges.py and
prompts/dedup.py convention ("never invent ids"): the model is given a roster
of candidate ids and MUST choose only from it. Any id it returns that is not
in the roster is dropped, never trusted — same spirit as kg.gates rejecting
edges that reference unknown ids. This module has no dependency on a specific
provider SDK; you plug in your own `call` callable. If you want to reuse
kg-mapper-poc's own adapters (src/kg/llm/adapters/*.py implement
`complete(system, messages, schema_json, model, max_tokens) -> RawCompletion`),
wrap one of those to match the `LLMCall` signature below.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from kg_reasoner.models import CandidateNode, StudentContext


class LLMCall(Protocol):
    """Minimal adapter surface this module needs. Implement this against whatever
    provider you use; kg-mapper-poc's adapters already speak a very similar shape."""

    def __call__(self, *, system: str, user: str, schema_json: dict[str, Any]) -> dict[str, Any]: ...


#: JSON schema for the constrained response — deliberately tiny and closed,
#: same profile discipline as kg.schemas.WireModel (extra fields forbidden).
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["selections"],
    "properties": {
        "selections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "reason"],
                "properties": {
                    "id": {"type": "string", "description": "A node id taken verbatim from the roster. Never invent one."},
                    "reason": {"type": "string", "description": "One short sentence: why this node helps right now."},
                },
            },
        }
    },
}


def _build_prompt(ctx: StudentContext, anchor: CandidateNode, candidates: list[CandidateNode], top_n: int) -> tuple[str, str]:
    system = (
        "You are ranking which knowledge-graph notes would most help a teaching "
        "chatbot respond to one student right now. Choose only from the given "
        f"roster of node ids; pick at most {top_n}. Never invent an id, never "
        "reorder based on anything not stated below. If none genuinely help, "
        "return an empty list."
    )
    roster = [
        {
            "id": c.id,
            "title": c.title,
            "relation_to_anchor": c.relation,
            "distance": c.distance,
            "relevance": c.relevance,
            "definition": c.definition_snippet,
        }
        for c in candidates
    ]
    user = json.dumps(
        {
            "student_intent": ctx.intent,
            "learner_state": ctx.learner_state,
            "student_level": ctx.student_level,
            "anchor_topic": {"id": anchor.id, "title": anchor.title, "definition": anchor.definition_snippet},
            "roster": roster,
        },
        ensure_ascii=False,
        indent=2,
    )
    return system, user


def rerank(
    ctx: StudentContext,
    anchor: CandidateNode,
    candidates: list[CandidateNode],
    call: LLMCall,
    *,
    top_n: int = 3,
) -> list[CandidateNode]:
    """Ask the LLM to pick the most useful subset of `candidates`; falls back safely.

    Returns candidates in the model's chosen order, restricted to ids that were
    actually offered (a hallucinated id is dropped and the rest kept). If the
    call raises, or the response has zero valid ids, the original `candidates`
    list (Tier 1's own order) is returned unchanged — Tier 2 can only narrow
    or reorder, never fail the turn.
    """
    if not candidates:
        return candidates
    by_id = {c.id: c for c in candidates}
    system, user = _build_prompt(ctx, anchor, candidates, top_n)
    try:
        response = call(system=system, user=user, schema_json=RESPONSE_SCHEMA)
        chosen_ids = [row["id"] for row in response.get("selections", []) if row.get("id") in by_id]
    except Exception:  # noqa: BLE001 - Tier 2 is best-effort; never take the turn down with it
        return candidates
    if not chosen_ids:
        return candidates
    seen: set[str] = set()
    out = []
    for cid in chosen_ids:
        if cid in seen:
            continue
        seen.add(cid)
        out.append(by_id[cid])
    return out[:top_n]


__all__ = ["LLMCall", "RESPONSE_SCHEMA", "rerank"]
