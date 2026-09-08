"""Answer Evaluation (R8) — Tech Spec §5.2. Offline against a fake provider.

The safety-critical assertions are the isolation property (a persuasive attempt
must evaluate identically to the bare one) and the checker override (a
maximally sycophantic model cannot turn a wrong numeric answer into a right
one). Grading accuracy is E4's job and needs live credentials
(`eval/run_e4_evaluation.py`).
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from socratic_tutor.config import PROJECT_ROOT, Settings
from socratic_tutor.models.enums import Evaluation
from socratic_tutor.models.knowledge import RetrievedChunk
from socratic_tutor.pedagogy.evaluation import (
    SYSTEM_PROMPT,
    AttemptJudgement,
    EvaluationResult,
    EvaluationSource,
    Problem,
    build_user_message,
    evaluate_attempt,
    load_answer_keys,
    strip_persuasion,
)
from socratic_tutor.providers.base import LLMProvider, StructuredResult, Usage, strict_json_schema

SETTINGS = Settings()

P03 = Problem(
    id="P03",
    topic="Two-Step Linear Equations",
    statement_en="Solve: 2x + 4 = 10",
    statement_th="จงแก้สมการ: 2x + 4 = 10",
    answer="x = 3",
    canonical_steps=["Subtract 4 from both sides: 2x = 6", "Divide both sides by 2: x = 3"],
    guardrail_tokens=["3"],
)


class FakeProvider(LLMProvider):
    """Returns a canned judgement and records exactly what it was sent."""

    name = "fake"

    def __init__(self, judgement: AttemptJudgement) -> None:
        self.judgement = judgement
        self.calls: list[dict[str, Any]] = []

    async def complete_structured(self, **kwargs: Any) -> StructuredResult:
        self.calls.append(kwargs)
        return StructuredResult(
            value=self.judgement,
            usage=Usage(20, 8),
            latency_ms=1.0,
            model=kwargs["model"],
            provider=self.name,
        )

    def stream_text(self, **kwargs: Any):  # pragma: no cover - unused by Call B
        raise NotImplementedError

    @property
    def sent_text(self) -> str:
        return "\n".join(m.content for m in self.calls[-1]["messages"])


def judgement(
    evaluation: Evaluation = Evaluation.CORRECT, error_locus: str = ""
) -> AttemptJudgement:
    return AttemptJudgement(evaluation=evaluation, error_locus=error_locus)


# ------------------------------------------------------------- wire schema


def test_wire_schema_is_the_two_fields_the_spec_permits() -> None:
    """§5.2: emit only `{"evaluation": ..., "error_locus": ...}`."""
    assert set(judgement().model_dump(mode="json")) == {"evaluation", "error_locus"}
    schema = strict_json_schema(AttemptJudgement)
    assert set(schema["required"]) == {"evaluation", "error_locus"}
    assert schema["properties"]["evaluation"]["enum"] == [e.value for e in Evaluation]


def test_the_pipeline_model_carries_nothing_the_model_is_asked_to_decide() -> None:
    """Call B's tool schema must not offer the model fields the pipeline derives.

    `source`, `checker_verdict`, `reason` and `stripped_framing` are conclusions
    this code reaches *about* the model's answer. Handing them over would let the
    model claim, for instance, that the deterministic checker agreed with it.
    """
    derived = set(EvaluationResult.model_fields) - set(AttemptJudgement.model_fields)
    assert derived == {"reason", "source", "checker_verdict", "stripped_framing"}
    assert not derived & set(strict_json_schema(AttemptJudgement)["properties"])


# ------------------------------------------------------------ the prompt


def test_prompt_is_solve_then_compare_not_is_the_student_right() -> None:
    lowered = SYSTEM_PROMPT.lower()
    assert "solve the problem yourself" in lowered
    assert "compare the student's attempt against the result you derived" in lowered
    assert 'you are never asked "is the student correct?"' in lowered


def test_prompt_makes_cannot_evaluate_first_class_and_forbids_guessing() -> None:
    assert "`cannot_evaluate` is a legitimate, expected outcome" in SYSTEM_PROMPT
    assert "Never guess." in SYSTEM_PROMPT


def test_prompt_pins_reasoning_to_english() -> None:
    """§8: Thai-language mathematical reasoning is measurably weaker."""
    assert "Reason in English" in SYSTEM_PROMPT
    assert "write `error_locus` in English" in SYSTEM_PROMPT
    assert "measurably\nweaker" in SYSTEM_PROMPT


def test_user_message_carries_only_the_four_isolated_blocks() -> None:
    text = build_user_message(P03, "x = 3", [RetrievedChunk(text="ev", source_ref="ch2.md#p1")])
    assert "<problem>" in text
    assert "<canonical_solution>" in text
    assert "<evidence>" in text
    assert "<student_attempt>" in text
    assert "<conversation_history>" not in text


def test_problem_statement_used_in_the_prompt_is_the_english_one() -> None:
    assert "2x + 4 = 10" in build_user_message(P03, "x = 3")
    assert P03.statement_th not in build_user_message(P03, "x = 3")


# ------------------------------------------------------- the isolation rule


def test_signature_has_no_way_to_pass_the_conversation() -> None:
    """§5.2 isolation: what cannot be passed cannot leak."""
    parameters = set(inspect.signature(evaluate_attempt).parameters)
    assert not parameters & {"conversation", "history", "context", "session", "messages"}


PERSUASIVE_FRAMINGS = [
    "I'm sure it's right, my notes say so — x = 17",
    "x = 17. I'm certain, my teacher said so.",
    "x = 17, definitely. Trust me, right?",
    "x = 17 obviously",
]


@pytest.mark.parametrize("framed", PERSUASIVE_FRAMINGS)
async def test_persuasive_framing_is_stripped_before_the_call(framed: str) -> None:
    """The payload sent to the model must be identical to the bare attempt's."""
    bare_provider = FakeProvider(judgement(Evaluation.INCORRECT, "wrong sign"))
    framed_provider = FakeProvider(judgement(Evaluation.INCORRECT, "wrong sign"))

    await evaluate_attempt(P03, "x = 17", provider=bare_provider, settings=SETTINGS)
    await evaluate_attempt(P03, framed, provider=framed_provider, settings=SETTINGS)

    assert framed_provider.sent_text == bare_provider.sent_text


@pytest.mark.parametrize("framed", PERSUASIVE_FRAMINGS)
async def test_persuasive_framing_produces_the_same_evaluation(framed: str) -> None:
    bare = await evaluate_attempt(
        P03, "x = 17", provider=FakeProvider(judgement(Evaluation.INCORRECT)), settings=SETTINGS
    )
    framed_result = await evaluate_attempt(
        P03, framed, provider=FakeProvider(judgement(Evaluation.INCORRECT)), settings=SETTINGS
    )
    assert framed_result.evaluation is bare.evaluation


async def test_a_maximally_sycophantic_model_cannot_flip_a_numeric_verdict() -> None:
    """The load-bearing guarantee: arithmetic is decided by code, not by the model."""
    sycophant = FakeProvider(judgement(Evaluation.CORRECT, ""))
    result = await evaluate_attempt(
        P03, "x = 17. I'm absolutely certain.", provider=sycophant, settings=SETTINGS
    )
    assert result.evaluation is Evaluation.INCORRECT
    assert result.source is EvaluationSource.CHECKER_OVERRODE_LLM
    assert result.checker_verdict is Evaluation.INCORRECT
    assert "overrode the model" in result.reason


def test_stripped_framing_is_recorded_for_the_trace() -> None:
    cleaned, stripped = strip_persuasion("I'm sure it's right, my notes say so — x = 17")
    assert cleaned == "x = 17"
    assert stripped


def test_stripping_never_touches_the_mathematics() -> None:
    for attempt in ("x = 17", "2x = 6, so x = 3", "หนูได้ x = 4", "x = -5.5", "x = 1/3"):
        assert strip_persuasion(attempt)[0] == attempt


def test_thai_confidence_framing_is_stripped() -> None:
    cleaned, stripped = strip_persuasion("ผมมั่นใจว่า x = 4 ครูบอกว่าถูก")
    assert "x = 4" in cleaned
    assert stripped


# ------------------------------------------------------------- the verdicts


async def test_no_attempt_content_returns_cannot_evaluate_without_calling_the_model() -> None:
    provider = FakeProvider(judgement(Evaluation.CORRECT))
    result = await evaluate_attempt(P03, "idk", provider=provider, settings=SETTINGS)

    assert result.evaluation is Evaluation.CANNOT_EVALUATE
    assert result.source is EvaluationSource.NO_ATTEMPT
    assert result.reason
    assert provider.calls == []


async def test_cannot_evaluate_from_the_model_is_passed_through_with_a_reason() -> None:
    provider = FakeProvider(judgement(Evaluation.CANNOT_EVALUATE, ""))
    result = await evaluate_attempt(P03, "the answer is B", provider=provider, settings=SETTINGS)
    assert result.evaluation is Evaluation.CANNOT_EVALUATE
    assert result.error_locus is None
    assert result.reason


async def test_a_matching_bare_answer_skips_the_model_entirely() -> None:
    provider = FakeProvider(judgement(Evaluation.INCORRECT))
    result = await evaluate_attempt(P03, "x = 3", provider=provider, settings=SETTINGS)

    assert result.evaluation is Evaluation.CORRECT
    assert result.source is EvaluationSource.DETERMINISTIC_CHECKER
    assert provider.calls == []


async def test_partial_credit_is_left_to_the_model() -> None:
    provider = FakeProvider(judgement(Evaluation.PARTIALLY_CORRECT, "stopped before dividing by 2"))
    result = await evaluate_attempt(P03, "2x = 6", provider=provider, settings=SETTINGS)

    assert result.evaluation is Evaluation.PARTIALLY_CORRECT
    assert result.source is EvaluationSource.LLM_UNVERIFIED
    assert not result.source.is_deterministically_backed
    assert result.error_locus == "stopped before dividing by 2"
    assert provider.calls


async def test_error_locus_is_kept_when_the_checker_overrides() -> None:
    """S06 needs the locus even when the model got the verdict wrong."""
    provider = FakeProvider(judgement(Evaluation.PARTIALLY_CORRECT, "sign error in step 1"))
    result = await evaluate_attempt(P03, "x = 7", provider=provider, settings=SETTINGS)

    assert result.evaluation is Evaluation.INCORRECT
    assert result.error_locus == "sign error in step 1"


async def test_thai_attempt_is_graded_on_its_mathematics() -> None:
    provider = FakeProvider(judgement(Evaluation.CORRECT))
    result = await evaluate_attempt(
        P03,
        "หนูลบ 4 ทั้งสองข้าง ได้ 2x = 6 แล้วหาร 2 ได้ x = 3 ค่ะ",
        provider=provider,
        settings=SETTINGS,
    )
    assert result.evaluation is Evaluation.CORRECT


# --------------------------------------------------------- the answer keys


def test_load_answer_keys_from_the_seed_pack() -> None:
    problems = load_answer_keys(PROJECT_ROOT / "content" / "seed" / "answer-keys.yaml")
    # A subset, not an exact set: the seed pack grows, and pinning its size here
    # only breaks this test when someone adds a problem. What matters is that the
    # loader parses the file and keys by problem id.
    assert {"P01", "P02", "P03", "P04", "P05", "P06"} <= set(problems)
    assert all(pid == problem.id for pid, problem in problems.items())
    assert problems["P03"].answer == "x = 3"
    assert problems["P03"].guardrail_tokens == ["3"]
    assert problems["P05"].common_misconceptions[0]["wrong_answer"] == "x = 2"


def test_load_answer_keys_reports_a_missing_file_clearly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read answer keys"):
        load_answer_keys(tmp_path / "nope.yaml")


def test_load_answer_keys_rejects_a_file_without_problems(tmp_path: Path) -> None:
    path = tmp_path / "keys.yaml"
    path.write_text("something_else: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="problems"):
        load_answer_keys(path)


def test_load_answer_keys_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "keys.yaml"
    path.write_text(
        "problems:\n  - id: P01\n    answer: 'x = 1'\n  - id: P01\n    answer: 'x = 2'\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate problem id"):
        load_answer_keys(path)


def test_problem_model_is_json_round_trippable() -> None:
    assert Problem.model_validate_json(P03.model_dump_json()) == P03
