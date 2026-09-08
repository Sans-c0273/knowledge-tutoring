"""Shared data types for the reasoner.

Pure data — no I/O, no dependency on the kg-mapper-poc package. `StudentContext`
mirrors the Intent Detection / Student-Level Selection output from the Pedagogy
Layer design (intent, learner_state, student_level, special_handling); the
reasoner's output, `KnowledgeGuidance`, is what Strategy Selection consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Intent = Literal[
    "explain", "clarify", "solve", "hint", "check_answer", "practice_quiz", "summarize_review"
]
LearnerState = Literal["normal", "confused"]
StudentLevel = Literal["beginner", "intermediate", "advanced"]
SpecialHandling = Literal["none", "homework", "assessment"]

#: The pedagogical purpose a selected node serves. Kept small and stable so
#: Strategy Selection / Response Planner can switch on it without caring which
#: edge type produced it (education vs general schema disagree on that).
Role = Literal[
    "anchor",
    "supporting_context",
    "prerequisite_gap",
    "alternative_explanation",
    "practice_target",
    "next_challenge",
]


@dataclass(frozen=True)
class StudentContext:
    """One turn's pedagogical situation, as produced by Intent Detection + Student-Level Selection."""

    topic_query: str
    intent: Intent
    learner_state: LearnerState = "normal"
    student_level: StudentLevel = "beginner"
    special_handling: SpecialHandling = "none"
    #: Node ids already surfaced this session (e.g. the explanation just given) —
    #: excluded from re-explanation candidates so KM03 doesn't repeat itself.
    exclude_node_ids: frozenset[str] = field(default_factory=frozenset)
    #: Node ids the student model already marks as mastered, if you have one —
    #: used only to avoid presenting a "gap" the student has already cleared.
    mastered_node_ids: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class CandidateNode:
    """One graph node selected for this turn, with why it was selected."""

    id: str
    title: str
    role: Role
    relation: str  # edge type that reached it from the anchor, or "self" for the anchor
    distance: int  # hops from the anchor (0 = anchor itself)
    relevance: int | None = None  # only meaningful for scored edges (general schema related_to)
    grounded: bool = False  # True if the node carries >=1 source quote (safe to cite/quiz on)
    definition_snippet: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "role": self.role,
            "relation": self.relation,
            "distance": self.distance,
            "grounded": self.grounded,
        }
        if self.relevance is not None:
            out["relevance"] = self.relevance
        if self.definition_snippet:
            out["definition_snippet"] = self.definition_snippet
        return out


@dataclass(frozen=True)
class KnowledgeGuidance:
    """The reasoner's output: what Strategy Selection reads (Part 2 PDF's `knowledge_guidance`)."""

    schema: str
    anchor: CandidateNode | None
    selections: tuple[CandidateNode, ...]
    #: Human-readable trace of which rule(s) fired — for debugging and for a
    #: teacher-facing "why did it say this" view; never shown to the student.
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "knowledge_guidance": {
                "schema": self.schema,
                "anchor": self.anchor.to_dict() if self.anchor else None,
                "selections": [c.to_dict() for c in self.selections],
                "notes": list(self.notes),
            }
        }


__all__ = [
    "CandidateNode",
    "Intent",
    "KnowledgeGuidance",
    "LearnerState",
    "Role",
    "SpecialHandling",
    "StudentContext",
    "StudentLevel",
]
