"""KL Map lookup output and RAG retrieval hits (Tech Spec §2.3, step 2a/2b).

`KnowledgeContext` is what a relation-filtered BFS (depth 1, widened to 2 for
`prerequisite_of` when the learner is confused) returns. It describes graph
structure only — it never asserts anything about what the student knows
(Tech Spec §3.5, hard rule).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from socratic_tutor.models.enums import Language, Relation


class RelatedTopic(BaseModel):
    """A neighbour node, its BFS distance, and the edge that reached it.

    `relation` exists because the three buckets in `KnowledgeContext` are lossy:
    the vocabulary has five relations, but `related_to`, `part_of` and `uses` all
    land in `related_topics`. Without it the edge type is destroyed at this
    boundary and KM03 cannot name the relation it is re-explaining through.

    Optional, and a null means **unknown**, never `related_to`: a walk that has
    not been taught to record the edge must say nothing rather than let the
    pedagogy layer assert a relation the graph never confirmed. Carried on
    `RelatedTopic` itself, so all three buckets have the same shape even though
    prerequisites and next_topics are single-relation walks where it is redundant.
    """

    model_config = ConfigDict(use_enum_values=False)

    topic: str
    distance: int = 1
    relation: Relation | None = None


class KnowledgeContext(BaseModel):
    """Relation-filtered neighbourhood of the current topic (Tech Spec §2.3).

    `relationships` carries free-text relationship statements drawn from the map
    ("sunlight provides energy for photosynthesis") for use in explanation —
    not the closed `Relation` vocabulary, which lives on each `RelatedTopic`.
    """

    current_topic: str
    prerequisites: list[RelatedTopic] = Field(default_factory=list)
    related_topics: list[RelatedTopic] = Field(default_factory=list)
    next_topics: list[RelatedTopic] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    """One RAG hit with the source reference the response must cite.

    Tech Spec §1 step 2a and §4.2 (`source_reference_required`). An empty
    retrieval set is the anti-hallucination trigger in Teaching Policy row 4:
    no evidence → no direct factual answer.
    """

    model_config = ConfigDict(use_enum_values=False)

    text: str
    source_ref: str
    topic: str = ""
    lang: Language = Language.EN
    score: float = 0.0


__all__ = ["KnowledgeContext", "RelatedTopic", "RetrievedChunk"]
