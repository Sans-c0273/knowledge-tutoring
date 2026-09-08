"""Step 1: find where in the graph to start (Part 1 PDF §"Knowledge Map lookup", step 1).

For MVP this stays deliberately simple, per the source design docs: exact
title/alias match first, then a cheap fuzzy fallback. No embeddings, no LLM
call — teacher-curated topics are a small, known vocabulary.
"""

from __future__ import annotations

from kg_reasoner.graph_index import GraphIndex
from kg_reasoner.models import CandidateNode


def resolve_anchor(topic_query: str, index: GraphIndex, *, fuzzy_min_overlap: float = 0.5) -> CandidateNode | None:
    """Map a topic string (from RAG / the topic mapper) to a single anchor node.

    Returns None if nothing matches closely enough — the caller should treat
    that as "no knowledge-graph guidance available this turn" and fall back
    to RAG-only teaching, not guess.
    """
    exact = index.find_by_title(topic_query)
    node_id = exact[0] if exact else index.best_fuzzy_match(topic_query, min_overlap=fuzzy_min_overlap)
    if node_id is None:
        return None
    node = index.nodes[node_id]
    return CandidateNode(
        id=node.id,
        title=node.title,
        role="anchor",
        relation="self",
        distance=0,
        grounded=node.grounded,
        definition_snippet=node.definition_plain[:280],
    )


__all__ = ["resolve_anchor"]
