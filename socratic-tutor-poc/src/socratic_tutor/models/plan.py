"""Step 4 output — Response Planning (Tech Spec §4).

`ResponsePlan` is the generation model's entire brief: it becomes the Policy
layer of Call C's system prompt (§5.3). Everything the generator is allowed to
do is stated here; nothing is left to the model's judgement.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from socratic_tutor.models.enums import Language, StudentLevel


class GlobalRules(BaseModel):
    """Rules applied to every response template (Tech Spec §4.2).

    Field defaults are the spec's values verbatim. `GLOBAL_RULES` is the shared
    instance; construct your own only to model a table-version override.
    """

    max_examples: int = 1
    max_analogies: int = 1
    max_questions: int = 1
    source_reference_required: bool = True
    unsupported_claims_allowed: bool = False
    hint_first_and_full_solution_same_turn: bool = False
    quiz_answer_before_student_attempt: bool = False
    tone: str = "clear_respectful_encouraging_non_patronizing"


#: The Tech Spec §4.2 defaults as a ready instance.
GLOBAL_RULES = GlobalRules()


class ResponsePlan(BaseModel):
    """The deterministic brief handed to Call C (Tech Spec §4.3).

    `full_solution_allowed` defaults to False for the same fail-closed reason as
    `Constraints`: when it is False the canonical answer is structurally absent
    from Call C's context (§5.3), so a plan that was never populated cannot leak.
    """

    model_config = ConfigDict(use_enum_values=False)

    structure: list[str] = Field(default_factory=list)
    max_words: int = 120
    max_questions: int = 1
    language_level: StudentLevel = StudentLevel.BEGINNER
    language: Language = Language.EN
    full_solution_allowed: bool = False
    wait_for_student: bool = False


__all__ = ["GLOBAL_RULES", "GlobalRules", "ResponsePlan"]
