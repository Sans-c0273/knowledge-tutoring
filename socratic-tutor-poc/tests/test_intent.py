"""Intent Detection (R6) — Tech Spec §2.1, §5.1. Offline against a fake provider.

What is asserted here is everything that does not depend on model quality:
prompt assembly, the wire schema's shape, the no-RAG guarantee, the history
window, and the low-confidence fallback. Classification accuracy is E1's job and
needs live credentials (`eval/run_e1_intent.py`).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from socratic_tutor.config import Settings
from socratic_tutor.models.enums import Intent, LearnerState, SpecialHandling
from socratic_tutor.models.intent import SpecialHandlingResult
from socratic_tutor.pedagogy.intent import (
    MAX_CONTEXT_TURNS,
    SYSTEM_PROMPT,
    IntentClassification,
    apply_confidence_floor,
    build_user_message,
    detect_intent,
    normalize_context,
)
from socratic_tutor.providers.base import (
    LLMProvider,
    Msg,
    StructuredResult,
    Usage,
    strict_json_schema,
)

SETTINGS = Settings()


class FakeProvider(LLMProvider):
    """Records the call and returns a canned structured value. Never touches a network."""

    name = "fake"

    def __init__(self, value: BaseModel) -> None:
        self.value = value
        self.calls: list[dict[str, Any]] = []

    async def complete_structured(self, **kwargs: Any) -> StructuredResult:
        self.calls.append(kwargs)
        return StructuredResult(
            value=self.value,
            usage=Usage(10, 5),
            latency_ms=1.0,
            model=kwargs["model"],
            provider=self.name,
        )

    def stream_text(self, **kwargs: Any):  # pragma: no cover - unused by Call A
        raise NotImplementedError

    @property
    def last(self) -> dict[str, Any]:
        return self.calls[-1]

    @property
    def sent_text(self) -> str:
        return "\n".join(m.content for m in self.last["messages"])


def classification(
    intent: Intent = Intent.EXPLAIN,
    *,
    confidence: float = 0.9,
    learner_state: LearnerState = LearnerState.NORMAL,
    special: SpecialHandling = SpecialHandling.NONE,
) -> IntentClassification:
    return IntentClassification(
        intent=intent,
        learner_state=learner_state,
        special_handling=SpecialHandlingResult(
            detected=special is not SpecialHandling.NONE, type=special
        ),
        confidence=confidence,
    )


# ------------------------------------------------------------- wire schema


def test_wire_schema_is_exactly_the_spec_shape() -> None:
    """Tech Spec §2.1: these four fields and no others reach the model."""
    payload = classification().model_dump(mode="json")
    assert set(payload) == {"intent", "learner_state", "special_handling", "confidence"}
    assert set(payload["special_handling"]) == {"detected", "type"}


def test_wire_schema_survives_the_strict_builder() -> None:
    schema = strict_json_schema(IntentClassification)
    assert set(schema["required"]) == {
        "intent",
        "learner_state",
        "special_handling",
        "confidence",
    }
    assert schema["properties"]["intent"]["enum"] == [i.value for i in Intent]
    assert "low_confidence_fallback" not in json.dumps(schema)


# --------------------------------------------------------- prompt assembly


def test_system_prompt_encodes_the_taxonomy_and_the_evidence_rules() -> None:
    for intent in Intent:
        assert intent.value in SYSTEM_PROMPT
    # Confusion evidence: a wrong answer is a knowledge gap, not confusion (§2.1).
    assert "knowledge gap" in SYSTEM_PROMPT
    # Special handling comes from the student's own words only (§2.1).
    assert "Never infer homework or assessment" in SYSTEM_PROMPT
    # Bilingual, classified in place (Addendum §"Language & review").
    assert "Do not translate" in SYSTEM_PROMPT


def test_user_message_carries_history_and_the_current_message() -> None:
    text = build_user_message(
        "ok what next",
        ["assistant: subtract 4 from both sides first. What do you get?", "user: 2x = 6"],
    )
    assert "<conversation_history>" in text
    assert "assistant: subtract 4 from both sides first. What do you get?" in text
    assert "user: 2x = 6" in text
    assert "<student_message>\nok what next\n</student_message>" in text


def test_first_turn_says_so_rather_than_sending_an_empty_block() -> None:
    assert "(this is the first turn)" in build_user_message("What is a linear equation?")


def test_history_window_keeps_the_most_recent_turns() -> None:
    context = [f"user: turn {i}" for i in range(12)]
    turns = normalize_context(context)
    assert len(turns) == MAX_CONTEXT_TURNS
    assert turns[-1].content == "turn 11"
    assert turns[0].content == f"turn {12 - MAX_CONTEXT_TURNS}"


def test_context_accepts_both_prefixed_strings_and_msg_objects() -> None:
    turns = normalize_context(["user: hi", "assistant: hello", Msg(role="user", content="x = 3")])
    assert [(t.role, t.content) for t in turns] == [
        ("user", "hi"),
        ("assistant", "hello"),
        ("user", "x = 3"),
    ]


def test_unprefixed_context_entries_are_treated_as_tutor_turns() -> None:
    """The seed sets use unprefixed entries for narrative placeholders."""
    turns = normalize_context(["...several turns about two-step equations..."])
    assert turns[0].role == "assistant"


# ---------------------------------------------------------------- the call


async def test_detect_intent_sends_the_expected_request() -> None:
    provider = FakeProvider(classification(Intent.SOLVE, confidence=0.95))
    result = await detect_intent(
        "How do I solve 2x + 4 = 10?", provider=provider, settings=SETTINGS
    )

    call = provider.last
    assert call["schema"] is IntentClassification
    assert call["temperature"] == 0.0
    assert call["model"] == "claude-haiku-4-5"
    assert call["system"] == SYSTEM_PROMPT
    assert len(call["messages"]) == 1
    assert result.intent is Intent.SOLVE
    assert result.low_confidence_fallback is False
    assert result.raw_intent is None


async def test_call_a_never_sees_retrieved_content() -> None:
    """§5.1: Call A never sees RAG content — there is no parameter to pass it through."""
    import inspect

    parameters = set(inspect.signature(detect_intent).parameters)
    assert not parameters & {"evidence", "chunks", "retrieved_chunks", "context_chunks"}

    provider = FakeProvider(classification())
    await detect_intent("What is a linear equation?", provider=provider, settings=SETTINGS)
    assert "source_ref" not in provider.sent_text


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("สมการเชิงเส้นคืออะไรครับ", Intent.EXPLAIN),
        ("ยังไม่เข้าใจเลยครับ อธิบายใหม่อีกทีได้ไหม", Intent.CLARIFY),
        ("What is a linear equation?", Intent.EXPLAIN),
    ],
)
async def test_thai_and_english_both_produce_a_valid_result(message: str, expected: Intent) -> None:
    """The Thai message is sent through untranslated and parses back the same way."""
    provider = FakeProvider(classification(expected, confidence=0.9))
    result = await detect_intent(message, provider=provider, settings=SETTINGS)
    assert result.intent is expected
    assert message in provider.sent_text


# ------------------------------------------------------- confidence floor


def test_low_confidence_falls_back_to_explain_and_records_it() -> None:
    result = apply_confidence_floor(classification(Intent.PRACTICE_QUIZ, confidence=0.41), 0.6)
    assert result.intent is Intent.EXPLAIN
    assert result.low_confidence_fallback is True
    assert result.raw_intent is Intent.PRACTICE_QUIZ
    assert result.confidence == 0.41


def test_confident_classification_is_left_alone() -> None:
    result = apply_confidence_floor(classification(Intent.PRACTICE_QUIZ, confidence=0.88), 0.6)
    assert result.intent is Intent.PRACTICE_QUIZ
    assert result.low_confidence_fallback is False
    assert result.raw_intent is None


def test_fallback_keeps_the_policy_switch_and_the_confusion_signal() -> None:
    """A classifier wobble must not discard "this is my homework" — that is a policy breach."""
    raw = classification(
        Intent.SOLVE,
        confidence=0.3,
        learner_state=LearnerState.CONFUSED,
        special=SpecialHandling.HOMEWORK,
    )
    result = apply_confidence_floor(raw, 0.6)
    assert result.intent is Intent.EXPLAIN
    assert result.learner_state is LearnerState.CONFUSED
    assert result.special_handling.type is SpecialHandling.HOMEWORK
    assert result.special_handling.detected is True


async def test_threshold_comes_from_settings() -> None:
    provider = FakeProvider(classification(Intent.HINT, confidence=0.5))
    lenient = await detect_intent(
        "hint please", provider=provider, settings=Settings(intent_confidence_threshold=0.4)
    )
    strict = await detect_intent(
        "hint please", provider=provider, settings=Settings(intent_confidence_threshold=0.7)
    )
    assert lenient.intent is Intent.HINT
    assert strict.intent is Intent.EXPLAIN
    assert strict.raw_intent is Intent.HINT


def test_result_round_trips_with_the_fallback_fields() -> None:
    from socratic_tutor.models.intent import IntentResult

    result = apply_confidence_floor(classification(Intent.SOLVE, confidence=0.2), 0.6)
    assert IntentResult.model_validate_json(result.model_dump_json()) == result
