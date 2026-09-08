"""Closed vocabularies from the Technical Specification.

Every enum here is a contract value that appears on the wire (LLM structured
output), in a rule table, or in a persisted `TurnTrace`. Adding a member is a
spec change, not an implementation detail.

Spec: Tech Spec §2.1 (intent taxonomy, learner state, special handling),
§2.2 (student level), §2.3 (KL Map relation vocabulary), §2.4 (evaluation,
knowledge guidance actions), §3.2 (strategy library), Addendum §"Language & review".

All members subclass `str`, so `model_dump()` and `json.dumps()` emit the plain
spec string with no custom encoder.
"""

from __future__ import annotations

from enum import Enum


class Intent(str, Enum):
    """What the student wants from this turn (Tech Spec §2.1)."""

    EXPLAIN = "explain"
    CLARIFY = "clarify"
    SOLVE = "solve"
    HINT = "hint"
    CHECK_ANSWER = "check_answer"
    PRACTICE_QUIZ = "practice_quiz"
    SUMMARIZE_REVIEW = "summarize_review"


class LearnerState(str, Enum):
    """Binary confusion signal. MVP1 has no confusion sub-levels (Tech Spec §2.1)."""

    NORMAL = "normal"
    CONFUSED = "confused"


class SpecialHandling(str, Enum):
    """Context that triggers a Teaching Policy override (Tech Spec §2.1, §3.4).

    Detected from the student's own words only, never inferred from topic or
    difficulty. `ASSESSMENT` mode ships present-but-disabled for the POC
    (Addendum §"Language & review", D4).
    """

    NONE = "none"
    HOMEWORK = "homework"
    ASSESSMENT = "assessment"


class StudentLevel(str, Enum):
    """Self-declared per-topic level from onboarding (Tech Spec §2.2).

    No questionnaire completed → `BEGINNER`. The POC never updates a level from
    performance; that is Phase 2.
    """

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class StrategyId(str, Enum):
    """Strategy Library IDs S01–S08 (Tech Spec §3.2).

    The wire value is the ID (`"S01"`); the member name is the readable form.
    `STRATEGY_NAMES` maps an ID to the `strategy_name` string the spec's
    `StrategySelection` example carries alongside it.
    """

    DIRECT_ANSWER_EXPLANATION = "S01"
    EXAMPLE_ANALOGY = "S02"
    STEP_BY_STEP_GUIDANCE = "S03"
    HINT_SCAFFOLD = "S04"
    RE_EXPLAIN_DIFFERENTLY = "S05"
    FEEDBACK = "S06"
    CHECK_UNDERSTANDING = "S07"
    PRACTICE_QUIZ = "S08"


#: Canonical `strategy_name` per strategy ID (Tech Spec §2.4 example, §3.2 table).
STRATEGY_NAMES: dict[StrategyId, str] = {
    StrategyId.DIRECT_ANSWER_EXPLANATION: "direct_answer_explanation",
    StrategyId.EXAMPLE_ANALOGY: "example_analogy",
    StrategyId.STEP_BY_STEP_GUIDANCE: "step_by_step_guidance",
    StrategyId.HINT_SCAFFOLD: "hint_scaffold",
    StrategyId.RE_EXPLAIN_DIFFERENTLY: "re_explain_differently",
    StrategyId.FEEDBACK: "feedback",
    StrategyId.CHECK_UNDERSTANDING: "check_understanding",
    StrategyId.PRACTICE_QUIZ: "practice_quiz",
}


class Evaluation(str, Enum):
    """Answer Evaluation verdict (Tech Spec §2.4, §5.2).

    `CANNOT_EVALUATE` is a legitimate outcome and routes the turn to S07 Check
    Understanding — never to invented feedback.
    """

    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"
    CANNOT_EVALUATE = "cannot_evaluate"


class Relation(str, Enum):
    """Closed KL Map relation vocabulary (Tech Spec §2.3).

    One declared direction per relation. `PREREQUISITE_OF` is DAG-enforced at
    content-build time.
    """

    PREREQUISITE_OF = "prerequisite_of"
    RELATED_TO = "related_to"
    NEXT_TOPIC = "next_topic"
    PART_OF = "part_of"
    USES = "uses"


class KnowledgeAction(str, Enum):
    """How the planner may use the KL Map result (Tech Spec §2.4, §3.5).

    `CHECK_OR_SCAFFOLD_PREREQUISITE` is deliberately named check-*or*-scaffold:
    a prerequisite edge existing does NOT imply the student lacks it. Probe with
    one question; scaffold only on a failed probe. The system must never assert
    "you don't understand X" from graph structure alone (Tech Spec §3.5, hard rule).
    """

    USE_CURRENT_TOPIC_CONTEXT = "use_current_topic_context"
    USE_PREREQUISITE_AS_SCAFFOLD = "use_prerequisite_as_scaffold"
    USE_RELATED_CONCEPT_FOR_REEXPLANATION = "use_related_concept_for_reexplanation"
    PRACTICE_CURRENT_TOPIC_RELATIONSHIP = "practice_current_topic_relationship"
    CHECK_OR_SCAFFOLD_PREREQUISITE = "check_or_scaffold_prerequisite"


class GuardrailEventType(str, Enum):
    """What the output guardrail did (Tech Spec §6).

    Closed rather than free-form because E3 counts hard leaks separately from
    guardrail saves (Tech Spec §9): if the eval harness counted by string match,
    a typo in a later change would silently zero a safety metric. `OTHER` is the
    escape hatch for a check added before this vocabulary catches up — put the
    specifics in `GuardrailEvent.detail`.
    """

    #: A withheld canonical answer appeared in the draft and was stopped.
    LEAK_BLOCKED = "leak_blocked"
    #: Word or question count over the plan by >25%; served anyway, logged.
    PLAN_VIOLATION = "plan_violation"
    #: The draft was rejected and the model asked again with a system note.
    REGENERATED = "regenerated"
    #: Regeneration failed too; the template fallback was served instead.
    FALLBACK_SERVED = "fallback_served"
    OTHER = "other"


class Language(str, Enum):
    """Response language. Bilingual TH/EN from day one (Addendum §"Language & review")."""

    TH = "th"
    EN = "en"


__all__ = [
    "STRATEGY_NAMES",
    "Evaluation",
    "GuardrailEventType",
    "Intent",
    "KnowledgeAction",
    "Language",
    "LearnerState",
    "Relation",
    "SpecialHandling",
    "StrategyId",
    "StudentLevel",
]
