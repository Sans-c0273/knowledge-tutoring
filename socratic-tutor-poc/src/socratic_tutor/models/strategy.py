"""Step 3 output — Strategy Selection (Tech Spec §2.4, §3).

Produced by four rule tables with no LLM call. `evaluation` is present only when
S06 fires and is produced by the separate Answer Evaluation function (§5.2),
keeping Strategy Selection itself clean.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from socratic_tutor.models.enums import (
    STRATEGY_NAMES,
    Evaluation,
    KnowledgeAction,
    Relation,
    StrategyId,
    StudentLevel,
)


class StrategyRef(BaseModel):
    """An `{id, name}` pair as the spec's `StrategySelection` JSON carries it.

    `strategy_name` is derived from `STRATEGY_NAMES` when omitted, so callers
    write `StrategyRef(strategy_id=StrategyId.FEEDBACK)` and the pair can never
    drift apart.
    """

    model_config = ConfigDict(use_enum_values=False)

    strategy_id: StrategyId
    strategy_name: str = ""

    @model_validator(mode="after")
    def _fill_name(self) -> StrategyRef:
        if not self.strategy_name:
            self.strategy_name = STRATEGY_NAMES[self.strategy_id]
        return self


class Constraints(BaseModel):
    """Teaching Policy permissions for this turn (Tech Spec §3.4).

    **These defaults are deliberately fail-closed. Do not "fix" them to match the
    Normal-learning policy row.** Answer-withholding in this system is structural,
    not behavioural: a turn whose constraints were never populated must degrade
    to hint-first Socratic behaviour, because a permissive default is a silent
    policy breach while a restrictive one is a visible bug. E3's gate is zero
    hard leaks (Tech Spec §9). The planner always sets all three explicitly.
    """

    direct_answer_allowed: bool = False
    full_solution_allowed_now: bool = False
    student_attempt_required: bool = True


class KnowledgeGuidance(BaseModel):
    """How the generator may use the KL Map result (Tech Spec §2.4, §3.5 KM rules).

    `relation` is the closed `Relation` vocabulary, not free text: this field
    drives a teaching decision, and the KM02/KM03 tests assert on *which edge
    type* produced the guidance. The spec's `"used_in"` example is `uses` read in
    the inverse direction, which `direction` records. Human-readable prose for
    the prompt lives in `KnowledgeContext.relationships`; the prompt builder
    composes from that list plus this enum.
    """

    # `extra="forbid"` so the removed `relationship=` name raises instead of being
    # silently dropped. A guidance field that quietly stops being applied is a
    # teaching decision that quietly stops happening.
    model_config = ConfigDict(use_enum_values=False, extra="forbid")

    action: KnowledgeAction | None = None
    target_concept: str | None = None
    #: Null means the walk could not confirm an edge type — never a default
    #: relation. The pedagogy layer must not assert what the graph did not say.
    relation: Relation | None = None
    #: Which way the edge is being read. `forward` follows the declared direction
    #: ("A is a prerequisite of B"); `inverse` reads it backwards ("B uses A").
    direction: Literal["forward", "inverse"] = "forward"
    distance: int | None = None


class StrategySelection(BaseModel):
    """One primary strategy, up to two supporting, plus the policy that applies.

    Tech Spec §2.4. Combination Rules (§3.3) cap this at 1 primary + 2 supporting;
    the cap is enforced by the selector, not by this schema, so an over-long list
    coming out of a table edit is visible in the trace instead of raising.
    """

    model_config = ConfigDict(use_enum_values=False)

    primary_strategy: StrategyRef
    evaluation: Evaluation | None = None
    supporting_strategies: list[StrategyRef] = Field(default_factory=list)
    constraints: Constraints = Field(default_factory=Constraints)
    knowledge_guidance: KnowledgeGuidance | None = None
    context: dict[str, Any] = Field(default_factory=dict)

    @property
    def topic(self) -> str | None:
        """Convenience read of `context["topic"]` (Tech Spec §2.4 example)."""
        value = self.context.get("topic")
        return value if isinstance(value, str) else None

    @property
    def student_level(self) -> StudentLevel | None:
        """Convenience read of `context["student_level"]` (Tech Spec §2.4 example)."""
        value = self.context.get("student_level")
        if isinstance(value, StudentLevel):
            return value
        if isinstance(value, str):
            try:
                return StudentLevel(value)
            except ValueError:
                return None
        return None


__all__ = ["Constraints", "KnowledgeGuidance", "StrategyRef", "StrategySelection"]
