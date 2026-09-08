"""Step 1 — Intent Detection (Tech Spec §2.1, §5.1).

Two models, and the difference between them is load-bearing:

- `IntentClassification` is the **wire** model. It is what `strict_json_schema()`
  turns into Call A's tool definition, so every field on it is something the
  model is *obliged to decide*. It carries the four §2.1 fields and nothing else.
- `IntentResult` is the **pipeline** model, built by
  `pedagogy.intent.apply_confidence_floor` from the wire output plus fields the
  system computes about the model. It is what the trace stores and the glass-box
  UI renders.

They must not be merged. `low_confidence_fallback` is a decision this code makes
after reading `confidence`; asking the model to report it would be asking it to
know something it cannot, and it would answer anyway. `SYSTEM_COMPUTED` marks
those fields so `strict_json_schema()` refuses a model carrying them (see
`providers.base`), which makes the mistake impossible rather than merely
discouraged.

Keep both flat, closed-vocabulary and free of validation constraints — ranges
and confidence thresholding are policy, applied by the pedagogy layer, not by
the schema.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from socratic_tutor.models.enums import Intent, LearnerState, SpecialHandling

#: Marker for a field the pipeline derives rather than the model decides.
#: `providers.base.strict_json_schema` raises on any model carrying one, so a
#: pipeline model can never be handed to a provider as a tool schema.
SYSTEM_COMPUTED = {"system_computed": True}


class SpecialHandlingResult(BaseModel):
    """Whether the student's own words put this turn under a policy override.

    Detected from what the student said — "just give me the homework answer" →
    homework; "I'm doing my test" → assessment; otherwise none (Tech Spec §2.1).
    """

    model_config = ConfigDict(use_enum_values=False)

    detected: bool = False
    type: SpecialHandling = SpecialHandling.NONE


class IntentClassification(BaseModel):
    """Call A's wire schema — exactly the four fields of Tech Spec §2.1.

    Everything here is a judgment the classifier is competent to make about the
    student's message. Adding a field means adding a model obligation, so add
    one only if the model is genuinely the right thing to ask.
    """

    model_config = ConfigDict(use_enum_values=False)

    intent: Intent
    learner_state: LearnerState
    special_handling: SpecialHandlingResult
    confidence: float


class IntentResult(BaseModel):
    """The pipeline's record of step 1 — never sent to a model.

    The first four fields are Tech Spec §2.1 verbatim. The last two record what
    `pedagogy.intent.apply_confidence_floor` did: below the threshold (~0.6, see
    `Settings.intent_confidence_threshold`) `intent` is forced to `explain` and
    the classifier's own label is preserved in `raw_intent`, so the glass-box UI
    can show "classifier said practice_quiz at 0.41 → fell back to explain"
    instead of presenting a guess as a classification.
    """

    model_config = ConfigDict(use_enum_values=False)

    intent: Intent = Intent.EXPLAIN
    learner_state: LearnerState = LearnerState.NORMAL
    special_handling: SpecialHandlingResult = Field(default_factory=SpecialHandlingResult)
    confidence: float = 0.0

    #: True when `intent` is the safe default rather than the classifier's label.
    low_confidence_fallback: bool = Field(default=False, json_schema_extra=SYSTEM_COMPUTED)
    #: What the classifier actually returned, set only when the fallback fired.
    raw_intent: Intent | None = Field(default=None, json_schema_extra=SYSTEM_COMPUTED)


__all__ = [
    "SYSTEM_COMPUTED",
    "IntentClassification",
    "IntentResult",
    "SpecialHandlingResult",
]
