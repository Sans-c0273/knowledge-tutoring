"""End-to-end: StudentContext + GraphIndex -> KnowledgeGuidance.

This is the function Strategy Selection calls. It does NOT choose the teaching
strategy (S01-S08) — that stays exactly where Part 1/2 put it, in the Trigger
Matrix. This only answers "which graph node(s), if any, should support that
strategy" (Part 2, p.4): "Strategy Selection = choose the teaching action +
choose which Knowledge Map concept, if any, should support that action."
"""

from __future__ import annotations

from kg_reasoner.anchor import resolve_anchor
from kg_reasoner.graph_index import GraphIndex
from kg_reasoner.llm_reasoner import LLMCall, rerank
from kg_reasoner.models import KnowledgeGuidance, StudentContext
from kg_reasoner.rules import apply_rules


def reason(
    ctx: StudentContext,
    index: GraphIndex,
    *,
    max_selections: int | None = None,
    llm_call: LLMCall | None = None,
    llm_top_n: int = 3,
) -> KnowledgeGuidance:
    """Run anchor resolution -> relation-filtered rules -> optional LLM re-rank.

    `llm_call` is optional (Tier 2). Pass one only when Tier 1 tends to return
    more equally-plausible candidates than you want to show in one turn; leave
    it None to stay fully deterministic (recommended default, per source docs).

    `max_selections=None` (the default) lets `apply_rules` derive the cap from
    `ctx.student_level` (see `rules.LEVEL_PROFILES`) — pass an explicit int only
    to override that for one call.
    """
    anchor = resolve_anchor(ctx.topic_query, index)
    if anchor is None:
        return KnowledgeGuidance(
            schema=index.schema or "unknown",
            anchor=None,
            selections=(),
            notes=(f"No node matched topic_query={ctx.topic_query!r}; no knowledge-graph guidance this turn.",),
        )

    result = apply_rules(ctx, index, anchor.id, max_selections=max_selections)
    selections = result.candidates
    notes = list(result.notes)

    if llm_call is not None and len(selections) > llm_top_n:
        selections = rerank(ctx, anchor, selections, llm_call, top_n=llm_top_n)
        notes.append(f"Tier 2 LLM rerank applied: {len(result.candidates)} candidates -> {len(selections)} selected")

    return KnowledgeGuidance(schema=index.schema or "unknown", anchor=anchor, selections=tuple(selections), notes=tuple(notes))


__all__ = ["reason"]
