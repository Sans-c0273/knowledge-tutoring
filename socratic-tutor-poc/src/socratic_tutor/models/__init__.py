"""Pydantic schemas — the Technical Specification's JSON contracts.

One module per pipeline artefact, in pipeline order:

| Module | Spec | Produced by |
|---|---|---|
| `enums` | §2.1–§2.4, §3.2 | closed vocabularies shared by everything |
| `intent` | §2.1 | step 1, Call A |
| `student` | §2.2 | step 2 lookup |
| `knowledge` | §2.3 | steps 2a/2b |
| `strategy` | §2.4, §3 | step 3, rule tables |
| `plan` | §4 | step 4, planner |
| `trace` | §7 | every step; rendered by the glass-box UI |
"""

from __future__ import annotations

from socratic_tutor.models.enums import (
    STRATEGY_NAMES,
    Evaluation,
    GuardrailEventType,
    Intent,
    KnowledgeAction,
    Language,
    LearnerState,
    Relation,
    SpecialHandling,
    StrategyId,
    StudentLevel,
)
from socratic_tutor.models.intent import (
    SYSTEM_COMPUTED,
    IntentClassification,
    IntentResult,
    SpecialHandlingResult,
)
from socratic_tutor.models.knowledge import KnowledgeContext, RelatedTopic, RetrievedChunk
from socratic_tutor.models.plan import GLOBAL_RULES, GlobalRules, ResponsePlan
from socratic_tutor.models.strategy import (
    Constraints,
    KnowledgeGuidance,
    StrategyRef,
    StrategySelection,
)
from socratic_tutor.models.student import StudentModel, TopicLevel
from socratic_tutor.models.trace import GuardrailEvent, StepLatency, TurnTrace

__all__ = [
    "GLOBAL_RULES",
    "STRATEGY_NAMES",
    "SYSTEM_COMPUTED",
    "Constraints",
    "Evaluation",
    "GlobalRules",
    "GuardrailEvent",
    "GuardrailEventType",
    "Intent",
    "IntentClassification",
    "IntentResult",
    "KnowledgeAction",
    "KnowledgeContext",
    "KnowledgeGuidance",
    "Language",
    "LearnerState",
    "RelatedTopic",
    "Relation",
    "ResponsePlan",
    "RetrievedChunk",
    "SpecialHandling",
    "SpecialHandlingResult",
    "StepLatency",
    "StrategyId",
    "StrategyRef",
    "StrategySelection",
    "StudentLevel",
    "StudentModel",
    "TopicLevel",
    "TurnTrace",
]
