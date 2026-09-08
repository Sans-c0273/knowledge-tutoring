"""R12 — the turn pipeline and its record (Tech Spec §1, §7).

Everything here runs offline: the two LLM calls are injected fakes, the rule
tables, syllabus and KL map are the real seed content. The headline test is
`test_full_teaching_decision_reconstructs_from_the_record`, which is R12's
acceptance criterion stated as an assertion.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from socratic_tutor.config import Settings
from socratic_tutor.domain.klmap import load_klmap
from socratic_tutor.models.enums import (
    Evaluation,
    GuardrailEventType,
    Intent,
    Language,
    LearnerState,
    SpecialHandling,
    StrategyId,
    StudentLevel,
)
from socratic_tutor.models.intent import IntentResult, SpecialHandlingResult
from socratic_tutor.models.knowledge import RetrievedChunk
from socratic_tutor.models.plan import ResponsePlan
from socratic_tutor.models.strategy import StrategyRef, StrategySelection
from socratic_tutor.orchestrator import (
    AnswerKeyLookup,
    InMemorySessionRepository,
    JsonFileSessionRepository,
    SessionState,
    TurnDependencies,
    TurnRecord,
    TurnRequest,
    _may_state_solution,
    _policy_applied,
    execute_turn,
    redact_answer,
    run_turn,
)
from socratic_tutor.pedagogy.evaluation import EvaluationResult, EvaluationSource, Problem
from socratic_tutor.pedagogy.student_model import (
    InMemoryStudentModelRepository,
    StudentModelService,
)
from socratic_tutor.pedagogy.tables import get_tables
from socratic_tutor.pedagogy.topics import load_syllabus

SEED = Path(__file__).resolve().parents[1] / "content" / "seed"
COURSE = "MATH-SEED-01"

CANONICAL = Problem(
    id="P03",
    topic="Two-Step Linear Equations",
    statement_en="Solve: 2x + 4 = 10",
    answer="x = 3",
    canonical_steps=["Subtract 4 from both sides: 2x = 6", "Divide both sides by 2: x = 3"],
    guardrail_tokens=["3"],
)

CHUNKS = [
    RetrievedChunk(
        text="Undo addition by subtracting the same number from both sides.",
        source_ref="ch1-inverse-operations.md#L12",
        topic="Inverse Operations",
        lang=Language.EN,
        score=0.91,
    )
]


def fake_intent(
    intent: Intent = Intent.SOLVE,
    learner_state: LearnerState = LearnerState.NORMAL,
    special: SpecialHandling = SpecialHandling.NONE,
    confidence: float = 0.93,
) -> Any:
    async def detect(message: str, context: Any = (), **kwargs: Any) -> IntentResult:
        return IntentResult(
            intent=intent,
            learner_state=learner_state,
            special_handling=SpecialHandlingResult(
                detected=special is not SpecialHandling.NONE, type=special
            ),
            confidence=confidence,
        )

    return detect


def fake_generation(*replies: str) -> Any:
    """A generator that streams `replies[n]` on the n-th call, last one repeating."""
    calls: list[dict[str, Any]] = []

    async def generate(plan: Any, message: str, **kwargs: Any) -> AsyncIterator[str]:
        index = min(len(calls), len(replies) - 1)
        calls.append({"plan": plan, "message": message, **kwargs})
        for word in replies[index].split(" "):
            yield word + " "

    generate.calls = calls  # type: ignore[attr-defined]
    return generate


def build_deps(**overrides: Any) -> TurnDependencies:
    defaults: dict[str, Any] = {
        "syllabus": load_syllabus(SEED / "syllabus-linear-equations.yaml"),
        "tables": get_tables(),
        "klmap": load_klmap(SEED / "kl-map-linear-equations.yaml"),
        "students": StudentModelService(InMemoryStudentModelRepository()),
        "sessions": InMemorySessionRepository(),
        "settings": Settings(),
        "retrieve": lambda course_id, topic, message: list(CHUNKS),
        "canonical_lookup": lambda course_id, topic, message: None,
        "detect_intent": fake_intent(),
        "generate": fake_generation("What could you subtract from both sides first?"),
    }
    return TurnDependencies(**{**defaults, **overrides})


def request(message: str = "How do I solve two-step linear equations?") -> TurnRequest:
    return TurnRequest(
        session_id="sess-1", student_id="student-1", course_id=COURSE, message=message
    )


async def test_full_turn_populates_every_pipeline_step() -> None:
    record = await execute_turn(request(), build_deps())
    trace = record.trace

    assert trace.error is None
    assert trace.intent_result is not None and trace.intent_result.intent is Intent.SOLVE
    assert trace.topic == "Two-Step Linear Equations"
    assert trace.student_level is StudentLevel.BEGINNER
    assert trace.retrieved_chunks and trace.retrieved_chunks[0].source_ref
    assert trace.knowledge_context is not None
    assert trace.strategy_selection is not None
    assert trace.response_plan is not None
    assert trace.generated_text
    assert trace.matched_trigger_rows


async def test_full_teaching_decision_reconstructs_from_the_record() -> None:
    """R12's acceptance criterion, as an assertion.

    Everything a reviewer needs to answer "why did the tutor say that?" must be
    in the stored record, with no other source consulted.
    """
    deps = build_deps(detect_intent=fake_intent(Intent.SOLVE, LearnerState.CONFUSED))
    record = await execute_turn(request("I'm confused about two-step linear equations"), deps)

    replayed = TurnRecord.model_validate_json(record.model_dump_json())
    trace, topic = replayed.trace, replayed.topic_resolution

    # what the student was understood to want
    assert trace.intent_result is not None
    assert trace.intent_result.learner_state is LearnerState.CONFUSED
    assert trace.intent_result.confidence > 0

    # what it was about, and how sure we were
    assert topic.topic == "Two-Step Linear Equations"
    assert topic.node_id == "C009"
    assert topic.matched_by is not None
    assert topic.level_source == "default_beginner"
    assert trace.student_level is StudentLevel.BEGINNER

    # what evidence was in front of the model
    assert all(chunk.source_ref for chunk in trace.retrieved_chunks)
    assert trace.knowledge_context is not None
    assert trace.knowledge_context.prerequisites, "confused learner must get prerequisites"

    # which rules fired, including the ones that lost
    assert trace.matched_trigger_rows
    assert trace.strategy_selection is not None
    assert trace.strategy_selection.primary_strategy.strategy_id
    assert "trigger_matrix" in trace.table_versions

    # what the model was allowed to do, and what it did
    assert trace.response_plan is not None
    assert trace.response_plan.structure
    assert trace.generated_text

    # provenance and cost
    assert {
        "intent",
        "topic_level",
        "retrieval",
        "kl_lookup",
        "strategy_selection",
        "response_planning",
        "generation",
        "guardrail",
    } <= set(trace.latencies_ms)
    assert trace.latencies_ms["pre_stream"] > 0
    assert trace.model_ids["intent"] and trace.model_ids["generation"]
    assert trace.prompt_versions["intent"] and trace.prompt_versions["generation"]


async def test_the_trigger_contest_is_recorded_whole() -> None:
    """ "Which rule chose this, and what beat what" is the panel's question.

    The row was previously flattened to `"TM12 (selected)"` here and taken apart
    again with `partition` at the wire edge — one row in three representations
    across two files. It now travels as the selector produced it.
    """
    deps = build_deps(
        detect_intent=fake_intent(Intent.CHECK_ANSWER, special=SpecialHandling.HOMEWORK)
    )
    record = await execute_turn(request("homework: I got x = 3, right?"), deps)

    matches = record.trigger_matches
    assert len(matches) >= 2, "a homework check_answer turn matches more than one row"

    winner = next(match for match in matches if match.selected)
    loser = next(match for match in matches if not match.selected)

    assert winner.row_id == "TM12"
    assert winner.lost_to is None
    assert loser.lost_to == winner.row_id
    assert loser.lost_on == "priority", "which dimension decided, not just which row won"
    assert loser.primary, "what the losing row would have done instead"
    assert loser.key["special_handling"] == "homework", "the key is a mapping, not a 4-list"
    assert winner.source in {"spec", "extension"}


async def test_the_spec_field_is_projected_from_the_contest() -> None:
    """§7 names `matched_trigger_rows`, so it stays — derived, not maintained."""
    deps = build_deps(
        detect_intent=fake_intent(Intent.CHECK_ANSWER, special=SpecialHandling.HOMEWORK)
    )
    record = await execute_turn(request("homework: I got x = 3, right?"), deps)

    assert record.trace.matched_trigger_rows == [match.row_id for match in record.trigger_matches]
    assert all("(" not in entry for entry in record.trace.matched_trigger_rows), (
        "no labels baked into strings for someone downstream to parse back out"
    )


async def test_the_km_contest_is_recorded_whole() -> None:
    """§3.5 O2: KM02 beats KM03 and both must be logged, with the reason.

    Previously flattened to `"KM03 (matched, lost tie-break to KM02)"` here and
    unpacked with `partition` at the wire edge — the same one-fact-three-places
    shape as the trigger rows.
    """
    deps = build_deps(detect_intent=fake_intent(Intent.CLARIFY, LearnerState.CONFUSED))
    record = await execute_turn(request("I still don't get two-step linear equations"), deps)

    matches = record.km_matches
    assert matches, "a confused clarify turn on a mapped topic matches a KM rule"
    assert sum(match.applied for match in matches) <= 1

    winner = next(match for match in matches if match.applied)
    assert winner.rule_id == "KM02"
    assert winner.requirement == "immediate_prerequisite", (
        "the rule fired because the map has a prerequisite, not because of a "
        "conclusion about this student"
    )

    losers = [match for match in matches if not match.applied]
    if losers:
        assert all(match.lost_to == winner.rule_id for match in losers)
        assert all(match.lost_on == "priority" for match in losers), (
            "a table edit flattening the priorities would show as row_id here"
        )


async def test_matched_km_rules_is_projected_from_the_contest() -> None:
    deps = build_deps(detect_intent=fake_intent(Intent.CLARIFY, LearnerState.CONFUSED))
    record = await execute_turn(request("I still don't get two-step linear equations"), deps)

    assert record.trace.matched_km_rules == [match.rule_id for match in record.km_matches]
    assert all("(" not in entry for entry in record.trace.matched_km_rules)


async def test_ambiguous_topic_is_carried_into_the_record() -> None:
    record = await execute_turn(request("linear equations"), build_deps())

    resolution = record.topic_resolution
    assert resolution.ambiguous is True
    assert set(resolution.candidates) == {
        "One-Step Linear Equations",
        "Two-Step Linear Equations",
    }
    assert resolution.confidence < 0.5


async def test_second_turn_uses_the_previous_topic_as_context() -> None:
    deps = build_deps()
    await execute_turn(request("How do I solve two-step linear equations?"), deps)
    record = await execute_turn(request("linear equations"), deps)

    assert record.topic_resolution.topic == "Two-Step Linear Equations"
    assert record.topic_resolution.resolved_by_context is True
    assert record.topic_resolution.ambiguous is False


async def test_session_accumulates_messages_and_records() -> None:
    deps = build_deps()
    await execute_turn(request("How do I solve two-step linear equations?"), deps)
    await execute_turn(request("and one-step ones?"), deps)

    state = deps.sessions.get("sess-1")
    assert state is not None
    assert len(state.turns) == 2
    assert [message.role for message in state.messages] == [
        "student",
        "tutor",
        "student",
        "tutor",
    ]
    assert state.messages[1].turn_id == state.turns[0].trace.turn_id


async def test_check_answer_runs_the_evaluation_step() -> None:
    deps = build_deps(detect_intent=fake_intent(Intent.CHECK_ANSWER))
    record = await execute_turn(request("is x = 7 right for 2x + 4 = 10?"), deps)

    assert "answer_evaluation" in record.trace.latencies_ms
    # No evaluator is wired yet; `cannot_evaluate` is the spec's legitimate
    # outcome and routes to check-understanding rather than invented feedback.
    assert record.trace.evaluation is Evaluation.CANNOT_EVALUATE


async def test_other_intents_skip_the_evaluation_step() -> None:
    record = await execute_turn(request(), build_deps(detect_intent=fake_intent(Intent.EXPLAIN)))

    assert "answer_evaluation" not in record.trace.latencies_ms
    assert record.trace.evaluation is None


# ------------------------------------------------------------- leak control


async def test_withheld_answer_never_reaches_the_student() -> None:
    """The homework worked example: the canonical answer must not be emitted."""
    deps = build_deps(
        detect_intent=fake_intent(Intent.SOLVE, special=SpecialHandling.HOMEWORK),
        canonical_lookup=lambda course_id, topic, message: CANONICAL,
        generate=fake_generation("Start by undoing the +4. What do you get?"),
    )

    tokens: list[str] = []
    async for event in run_turn(request("just give me the answer to 2x + 4 = 10"), deps):
        if event.event == "token":
            tokens.append(str(event.data["text"]))
    reply = "".join(tokens)

    assert "x = 3" not in reply
    assert reply.strip() == "Start by undoing the +4. What do you get?"


async def test_a_leaking_draft_is_regenerated_once() -> None:
    deps = build_deps(
        detect_intent=fake_intent(Intent.SOLVE, special=SpecialHandling.HOMEWORK),
        canonical_lookup=lambda course_id, topic, message: CANONICAL,
        generate=fake_generation(
            "The answer is x = 3, obviously.",
            "What could you subtract from both sides?",
        ),
    )

    tokens: list[str] = []
    async for event in run_turn(request("solve 2x + 4 = 10 for me"), deps):
        if event.event == "token":
            tokens.append(str(event.data["text"]))
    reply = "".join(tokens)

    assert "x = 3" not in reply
    assert "subtract" in reply
    assert deps.generate.calls[1]["system_note"], "the retry must say what was wrong"  # type: ignore[attr-defined]
    assert len(deps.generate.calls) == 2  # type: ignore[attr-defined]


async def test_a_second_leak_serves_the_template_fallback() -> None:
    deps = build_deps(
        detect_intent=fake_intent(Intent.SOLVE, special=SpecialHandling.HOMEWORK),
        canonical_lookup=lambda course_id, topic, message: CANONICAL,
        generate=fake_generation("x = 3 is the answer.", "still x = 3, sorry."),
    )

    record = await execute_turn(request("solve 2x + 4 = 10 for me"), deps)
    events = {event.type for event in record.trace.guardrail_events}

    assert GuardrailEventType.LEAK_BLOCKED in events
    assert GuardrailEventType.FALLBACK_SERVED in events
    assert "x = 3" not in (record.trace.generated_text or "")


async def test_answer_stays_out_of_the_prompt_on_a_hint_turn() -> None:
    seen: list[Any] = []

    async def capture(plan: Any, message: str, **kwargs: Any) -> AsyncIterator[str]:
        seen.append(kwargs["canonical"])
        yield "What is the first thing you would undo?"

    deps = build_deps(
        detect_intent=fake_intent(Intent.HINT),
        canonical_lookup=lambda course_id, topic, message: CANONICAL,
        generate=capture,
    )
    record = await execute_turn(request("hint for 2x + 4 = 10?"), deps)

    assert record.trace.response_plan is not None
    assert record.trace.response_plan.full_solution_allowed is False
    assert seen == [None], "a hint turn never receives the answer"


def test_may_state_solution_needs_a_block_that_actually_states_it() -> None:
    """Defence in depth for the flag and the structure disagreeing.

    The planner now clears `full_solution_allowed` whenever it drops the
    solution blocks, so the two agree today. This gate is what makes a future
    disagreement harmless rather than a leak: the prompt, the buffering and the
    scan all read this one value.
    """
    deps = build_deps()
    hint_only = ResponsePlan(
        structure=["single_hint", "student_attempt_prompt"], full_solution_allowed=True
    )
    with_solution = ResponsePlan(
        structure=["direct_answer", "explanation"], full_solution_allowed=True
    )

    assert _may_state_solution(hint_only, deps) is False
    assert _may_state_solution(with_solution, deps) is True
    assert (
        _may_state_solution(with_solution.model_copy(update={"full_solution_allowed": False}), deps)
        is False
    )


async def test_protected_turns_emit_tokens_only_after_the_guardrail() -> None:
    """An append-only stream cannot be retracted, so a withheld answer buffers."""
    deps = build_deps(
        canonical_lookup=lambda course_id, topic, message: CANONICAL,
        generate=fake_generation("The answer is x = 3."),
    )

    order = [event.event async for event in run_turn(request("solve it"), deps)]

    assert "token" in order
    assert order.index("trace") < order.index("token")
    assert order[-1] == "done"


async def test_turns_with_nothing_withheld_stream_live() -> None:
    """An explain turn may state the solution, so there is nothing to buffer for."""
    deps = build_deps(
        detect_intent=fake_intent(Intent.EXPLAIN),
        generate=fake_generation("one two three four"),
    )

    tokens = [
        str(event.data["text"])
        async for event in run_turn(request(), deps)
        if event.event == "token"
    ]

    assert len(tokens) == 4, "live streaming preserves the model's own token boundaries"


async def test_guardrail_logs_a_plan_violation_without_blocking() -> None:
    deps = build_deps(generate=fake_generation(" ".join(["word"] * 300)))

    record = await execute_turn(request(), deps)

    assert GuardrailEventType.PLAN_VIOLATION in {
        event.type for event in record.trace.guardrail_events
    }
    assert record.trace.generated_text, "a plan violation is logged, not blocked"


# ------------------------------------------------------------------ failure


async def test_failure_at_step_one_still_produces_a_serialisable_record() -> None:
    async def broken(message: str, context: Any = (), **kwargs: Any) -> IntentResult:
        raise RuntimeError("intent provider unreachable")

    deps = build_deps(detect_intent=broken)
    events = [event async for event in run_turn(request(), deps)]

    assert [event.event for event in events] == ["error", "done"]
    record = events[-1].record
    assert record is not None
    assert record.trace.error is not None
    assert "intent provider unreachable" in record.trace.error
    assert record.trace.intent_result is None
    assert record.trace.generated_text is None

    replayed = TurnRecord.model_validate_json(record.model_dump_json())
    assert replayed.trace.error == record.trace.error


async def test_a_failed_turn_is_still_persisted() -> None:
    async def broken(message: str, context: Any = (), **kwargs: Any) -> IntentResult:
        raise RuntimeError("nope")

    deps = build_deps(detect_intent=broken)
    await execute_turn(request(), deps)

    state = deps.sessions.get("sess-1")
    assert state is not None
    assert len(state.turns) == 1
    assert state.turns[0].trace.error


async def test_generation_failure_keeps_the_decisions_already_made() -> None:
    async def broken(plan: Any, message: str, **kwargs: Any) -> AsyncIterator[str]:
        raise RuntimeError("generation provider down")
        yield ""  # pragma: no cover - makes this an async generator

    record = await execute_turn(request(), build_deps(generate=broken))

    assert record.trace.error is not None
    assert record.trace.strategy_selection is not None, "step 3 ran before the failure"
    assert record.trace.response_plan is not None


async def test_trace_is_emitted_after_every_decision_step() -> None:
    events = [event.event async for event in run_turn(request(), build_deps())]

    assert events.count("trace") >= 6, "the inspector fills in live, not once at the end"
    assert events[0] == "trace"
    assert events[-1] == "done"


# --------------------------------------------------------------- persistence


def test_session_repository_round_trips(tmp_path: Path) -> None:
    repository = JsonFileSessionRepository(tmp_path)
    repository.save(SessionState(session_id="sess-1", student_id="s1", course_id=COURSE))

    reopened = JsonFileSessionRepository(tmp_path).get("sess-1")
    assert reopened is not None
    assert reopened.course_id == COURSE
    assert JsonFileSessionRepository(tmp_path).get("absent") is None


def test_session_repository_rejects_unsafe_ids(tmp_path: Path) -> None:
    from socratic_tutor.pedagogy.student_model import StudentModelError

    with pytest.raises(StudentModelError, match="safe path segment"):
        JsonFileSessionRepository(tmp_path).get("../../etc/passwd")


async def test_declared_level_is_used_when_the_student_onboarded() -> None:
    students = StudentModelService(InMemoryStudentModelRepository())
    syllabus = load_syllabus(SEED / "syllabus-linear-equations.yaml")
    students.record_onboarding("student-1", COURSE, {"C009": StudentLevel.ADVANCED}, syllabus)

    record = await execute_turn(request(), build_deps(students=students))

    assert record.trace.student_level is StudentLevel.ADVANCED
    assert record.topic_resolution.level_source == "onboarding_questionnaire"
    assert record.trace.response_plan is not None
    assert record.trace.response_plan.language_level is StudentLevel.ADVANCED


async def test_thai_message_produces_a_thai_plan() -> None:
    record = await execute_turn(request("ช่วยอธิบายสมการเชิงเส้นสองขั้นตอนหน่อย"), build_deps())

    assert record.trace.response_plan is not None
    assert record.trace.response_plan.language is Language.TH
    assert record.trace.topic == "Two-Step Linear Equations"


# ------------------------------------------------- H3: the protection stays on


def answer_keys() -> AnswerKeyLookup:
    return AnswerKeyLookup(SEED / "answer-keys.yaml")


@pytest.mark.parametrize(
    "message",
    [
        "this is homework: solve 2x + 4 = 10",
        "solve 2x+4=10",
        "จงแก้สมการ 2x + ๔ = ๑๐",
        "2x plus 4 equals 10",
    ],
)
def test_answer_key_matching_folds_the_way_the_scanner_folds(message: str) -> None:
    """Thai numerals and spacing are ordinary student input, not edge cases."""
    problem = answer_keys()("MATH-SEED-01", None, message)

    assert problem is not None
    assert problem.id == "P03"


def test_answer_key_does_not_match_an_unrelated_message() -> None:
    assert answer_keys()("MATH-SEED-01", None, "just give me the answer") is None


async def test_the_problem_stays_in_scope_across_turns() -> None:
    """The follow-up turn that used to disarm the guardrail."""
    deps = build_deps(
        detect_intent=fake_intent(Intent.SOLVE, special=SpecialHandling.HOMEWORK),
        canonical_lookup=answer_keys(),
        # Leaks on every draft, so the second turn's scan has something to catch.
        generate=fake_generation("The answer is x = 3."),
    )

    await execute_turn(request("this is homework: solve 2x + 4 = 10"), deps)
    state = deps.sessions.get("sess-1")
    assert state is not None and state.active_problem_id == "P03"

    tokens: list[str] = []
    async for event in run_turn(request("just give me the answer"), deps):
        if event.event == "token":
            tokens.append(str(event.data["text"]))

    second = state.turns[1].trace
    assert "x = 3" not in "".join(tokens), "the answer must not survive a follow-up turn"
    assert GuardrailEventType.LEAK_BLOCKED in {event.type for event in second.guardrail_events}


async def test_a_different_problem_replaces_the_one_in_scope() -> None:
    deps = build_deps(canonical_lookup=answer_keys())

    await execute_turn(request("solve 2x + 4 = 10"), deps)
    await execute_turn(request("now solve 5x - 3 = 2x + 9"), deps)

    state = deps.sessions.get("sess-1")
    assert state is not None
    assert state.active_problem_id == "P05"


async def test_withholding_without_a_known_answer_still_buffers() -> None:
    """Fail closed: an unresolved answer must not downgrade to the live path."""
    deps = build_deps(
        detect_intent=fake_intent(Intent.HINT),
        canonical_lookup=lambda course_id, topic, message: None,
        generate=fake_generation("one two three four"),
    )

    tokens = [
        str(event.data["text"])
        async for event in run_turn(request("give me a hint"), deps)
        if event.event == "token"
    ]
    record = await execute_turn(request("give me another hint"), deps)

    assert len(tokens) == 1, "buffered and re-emitted, not streamed as generated"
    unscanned = [
        event
        for event in record.trace.guardrail_events
        if event.action_taken == "buffered_unscanned"
    ]
    assert unscanned, "a turn that could not be scanned must say so in the trace"
    assert "no leak scan was possible" in unscanned[0].detail


# --------------------------------------------------------------- C1 redaction


async def test_guardrail_events_never_carry_the_answer() -> None:
    """A save must stay visible without publishing what it suppressed.

    The guardrail already builds a non-reversible `detail`; this asserts the
    property at the boundary that streams and persists it, so a future change on
    either side cannot reintroduce the value.
    """
    deps = build_deps(
        detect_intent=fake_intent(Intent.SOLVE, special=SpecialHandling.HOMEWORK),
        canonical_lookup=lambda course_id, topic, message: CANONICAL,
        generate=fake_generation("The answer is x = 3.", "What would you undo first?"),
    )

    record = await execute_turn(request("solve 2x + 4 = 10 for me"), deps)
    dumped = record.model_dump_json()

    assert record.trace.guardrail_events, "the save must still be visible"
    assert any("(P03)" in event.detail for event in record.trace.guardrail_events), (
        "an operator needs a handle to look the problem up server-side"
    )
    assert "x = 3" not in dumped
    assert "x=3" not in dumped.replace(" ", "")


async def test_the_persisted_session_never_carries_the_answer(tmp_path: Path) -> None:
    deps = build_deps(
        detect_intent=fake_intent(Intent.SOLVE, special=SpecialHandling.HOMEWORK),
        canonical_lookup=lambda course_id, topic, message: CANONICAL,
        sessions=JsonFileSessionRepository(tmp_path),
        generate=fake_generation("The answer is x = 3.", "What would you undo first?"),
    )

    await execute_turn(request("solve 2x + 4 = 10 for me"), deps)
    on_disk = (tmp_path / "sess-1.json").read_text(encoding="utf-8")

    assert "x = 3" not in on_disk
    assert "P03" in on_disk, "the id is kept; only the answer is withheld"


async def test_evaluation_reason_is_redacted() -> None:
    async def judge(problem: Any, attempt: str, evidence: Any = (), **kwargs: Any) -> Any:
        return EvaluationResult(
            evaluation=Evaluation.INCORRECT,
            reason="deterministic match: 5 != 3",
            error_locus="the constant was added instead of subtracted; x = 3",
            source=EvaluationSource.DETERMINISTIC_CHECKER,
        )

    deps = build_deps(
        detect_intent=fake_intent(Intent.CHECK_ANSWER),
        canonical_lookup=lambda course_id, topic, message: CANONICAL,
        evaluate=judge,
    )

    record = await execute_turn(request("is x = 5 right for 2x + 4 = 10?"), deps)

    assert record.answer_evaluation is not None
    assert "3" not in record.answer_evaluation.reason
    assert "5 !=" in record.answer_evaluation.reason, "only the answer goes, not the message"
    assert "x = 3" not in (record.answer_evaluation.error_locus or "")
    assert "added instead of subtracted" in (record.answer_evaluation.error_locus or "")


def test_redaction_covers_every_form_of_the_answer() -> None:
    """Bare numeric keys go wherever they stand alone, even at some cost to prose.

    "step 3" losing its 3 makes a debug string clumsy; a numeric answer surviving
    in a field that is streamed and persisted is the failure this exists to
    prevent. Over-redaction is the safe direction.
    """
    assert redact_answer("the answer is x = 3", CANONICAL) == "the answer is [redacted]"
    assert redact_answer("Divide both sides by 2: x = 3", CANONICAL) == "[redacted]"
    assert redact_answer("step 3 of 4", CANONICAL) == "step [redacted] of 4"
    assert redact_answer("13 apples", CANONICAL) == "13 apples", "not a standalone match"
    assert redact_answer("anything", None) == "anything"


# ------------------------------------------------- the quiz item the tutor asks


async def test_a_quiz_turn_asks_a_known_problem_and_arms_on_it() -> None:
    """A practice question is authored by the system, so its answer is knowable."""
    seen: list[Any] = []

    async def capture(plan: Any, message: str, **kwargs: Any) -> AsyncIterator[str]:
        seen.append(kwargs.get("quiz_item"))
        yield "Try this one: solve 2x + 4 = 10."

    deps = build_deps(
        detect_intent=fake_intent(Intent.PRACTICE_QUIZ),
        canonical_lookup=answer_keys(),
        generate=capture,
    )
    await execute_turn(request("quiz me on two-step equations"), deps)

    state = deps.sessions.get("sess-1")
    assert state is not None
    assert state.active_problem_id in {"P03", "P04"}
    assert seen[0] is not None and seen[0].id == state.active_problem_id


async def test_a_quiz_item_stays_in_scope_until_it_is_answered() -> None:
    """PS12: swapping the question mid-thread armed the scan on the wrong answer."""
    deps = build_deps(
        detect_intent=fake_intent(Intent.PRACTICE_QUIZ),
        canonical_lookup=answer_keys(),
        generate=fake_generation("Try this one."),
    )

    await execute_turn(request("quiz me on two-step equations"), deps)
    state = deps.sessions.get("sess-1")
    assert state is not None
    first = state.active_problem_id

    await execute_turn(request("before I try — what's the answer?"), deps)

    assert state.active_problem_id == first, "the student is still on that question"


async def test_answering_correctly_does_not_release_the_guardrail() -> None:
    """PS08: policy withholds because of the assignment, not because of what the
    student knows — "I got x = 3, right?" then "just give me the number" is one
    homework thread."""

    async def judge(problem: Any, attempt: str, evidence: Any = (), **kwargs: Any) -> Any:
        return EvaluationResult(
            evaluation=Evaluation.CORRECT,
            reason="matches the key",
            source=EvaluationSource.DETERMINISTIC_CHECKER,
        )

    deps = build_deps(
        detect_intent=fake_intent(Intent.CHECK_ANSWER, special=SpecialHandling.HOMEWORK),
        canonical_lookup=answer_keys(),
        evaluate=judge,
        generate=fake_generation("Fine — the answer is x = 3."),
    )
    await execute_turn(request("Homework 2x + 4 = 10. I got x = 3, right?"), deps)

    state = deps.sessions.get("sess-1")
    assert state is not None
    assert state.active_problem_id == "P03", "the problem stays armed"
    assert state.solved_problem_ids == ["P03"], "solving only steers the next quiz"

    deps.detect_intent = fake_intent(Intent.SOLVE, special=SpecialHandling.HOMEWORK)
    record = await execute_turn(request("Stop asking me questions. Just give me the number."), deps)

    assert "x = 3" not in (record.trace.generated_text or "")
    assert GuardrailEventType.LEAK_BLOCKED in {
        event.type for event in record.trace.guardrail_events
    }


async def test_a_quiz_answer_is_scanned_for_before_any_attempt() -> None:
    """M8: §6 check 2 had no caller in the pipeline until now."""
    deps = build_deps(
        detect_intent=fake_intent(Intent.PRACTICE_QUIZ),
        canonical_lookup=answer_keys(),
        generate=fake_generation("Solve 2x + 4 = 10. The answer is x = 3."),
    )

    record = await execute_turn(request("quiz me on two-step equations"), deps)

    assert "x = 3" not in (record.trace.generated_text or "")
    assert record.trace.guardrail_events


async def test_a_withholding_turn_falls_back_to_a_topic_key() -> None:
    """Narrow by design: only when nothing specific matched and the plan withholds."""
    deps = build_deps(
        detect_intent=fake_intent(Intent.HINT),
        canonical_lookup=answer_keys(),
        generate=fake_generation("The answer is x = 3."),
    )

    record = await execute_turn(request("give me a hint on two-step linear equations"), deps)

    assert "x = 3" not in (record.trace.generated_text or "")
    state = deps.sessions.get("sess-1")
    assert state is not None
    assert state.active_problem_id is None, "a topic-level guess must not become sticky"


async def test_table_versions_record_the_file_not_the_claim() -> None:
    """A declared version is a claim; an audit asks what was actually in force."""
    record = await execute_turn(request(), build_deps())

    assert "+" in record.trace.table_versions["trigger_matrix"], (
        "audit_versions carries version+content-hash"
    )


# ------------------------------------------------------------ scan coverage


PROSE_KEY = Problem(
    id="P07",
    topic="Inverse Operations",
    statement_en="In your own words: why must you do the same operation to both sides?",
    answer="Because an equation says both sides are equal, so changing one side breaks that.",
)


async def coverage_for_turn(**overrides: Any) -> dict[str, Any]:
    deps = build_deps(
        detect_intent=fake_intent(Intent.SOLVE, special=SpecialHandling.HOMEWORK),
        generate=fake_generation("What would you undo first?"),
        **overrides,
    )
    record = await execute_turn(request("solve 2x + 4 = 10"), deps)
    return record.scan_coverage


async def test_coverage_distinguishes_a_numeric_key_from_a_prose_one() -> None:
    """A PASS should mean "checked these forms", not just "found nothing"."""
    numeric = await coverage_for_turn(canonical_lookup=lambda c, t, m: CANONICAL)
    prose = await coverage_for_turn(canonical_lookup=lambda c, t, m: PROSE_KEY)

    assert numeric["scanned"] is True
    assert "number_words_th" in numeric["forms"], "digits and number words, both languages"
    assert prose["scanned"] is True
    assert "number_words_th" in prose["not_detected"], "a prose key degrades to substring"
    assert set(prose["forms"]) < set(numeric["forms"])


async def test_coverage_says_plainly_when_nothing_was_scanned() -> None:
    coverage = await coverage_for_turn(canonical_lookup=lambda c, t, m: None)

    assert coverage["scanned"] is False
    assert coverage["forms"] == []


async def test_coverage_comes_from_the_guardrail_not_from_here() -> None:
    """Derived, never declared: a list written in this file would already be stale.

    Asserted by comparing against the scanner's own answer for the same key, so
    the day the scanner's scope changes this test follows it rather than pinning
    a snapshot of what it used to cover.
    """
    from socratic_tutor.pedagogy.guardrail import coverage_for

    coverage = await coverage_for_turn(canonical_lookup=lambda c, t, m: CANONICAL)

    assert coverage == coverage_for(CANONICAL.answer).as_dict()


async def test_unscanned_means_two_different_things_and_says_which() -> None:
    """`scanned: false` is a pass on one turn and a warning on another.

    The guardrail reports coverage only for checks that actually ran, so a turn
    permitted to state the solution comes back unscanned for the innocent reason
    that there was nothing to check. A withholding turn with no answer key comes
    back unscanned for the dangerous one. Read alone the two are identical, and
    that conflation is the shape of every disarm bug in this build.
    """
    permitted = build_deps(
        detect_intent=fake_intent(Intent.EXPLAIN),
        canonical_lookup=lambda c, t, m: CANONICAL,
        generate=fake_generation("Here is how it works."),
    )
    withheld_known = build_deps(
        detect_intent=fake_intent(Intent.SOLVE, special=SpecialHandling.HOMEWORK),
        canonical_lookup=lambda c, t, m: CANONICAL,
        generate=fake_generation("What would you undo first?"),
    )
    withheld_unknown = build_deps(
        detect_intent=fake_intent(Intent.SOLVE, special=SpecialHandling.HOMEWORK),
        canonical_lookup=lambda c, t, m: None,
        generate=fake_generation("What would you undo first?"),
    )

    nothing_to_check = await execute_turn(request("explain two-step equations"), permitted)
    checked = await execute_turn(request("solve 2x + 4 = 10"), withheld_known)
    could_not_check = await execute_turn(request("solve 2x + 4 = 10"), withheld_unknown)

    assert (nothing_to_check.withheld_solution, nothing_to_check.scan_coverage["scanned"]) == (
        False,
        False,
    )
    assert (checked.withheld_solution, checked.scan_coverage["scanned"]) == (True, True)
    assert (could_not_check.withheld_solution, could_not_check.scan_coverage["scanned"]) == (
        True,
        False,
    )
    assert any(
        event.action_taken == "buffered_unscanned"
        for event in could_not_check.trace.guardrail_events
    ), "the dangerous case is also flagged as an event, not only as a field"


# ------------------------------------------- the policy decision, as a decision


async def test_the_governing_policy_row_is_a_field_not_a_version_string() -> None:
    """ "Why did it withhold?" is the panel's most important question.

    It was answerable only from `teaching_policy_row` buried in the versions map,
    which is why the inspector's policy card could never render.
    """
    deps = build_deps(detect_intent=fake_intent(Intent.SOLVE, special=SpecialHandling.HOMEWORK))
    record = await execute_turn(request("homework: solve 2x + 4 = 10"), deps)

    policy = record.teaching_policy
    assert policy is not None
    assert policy.row == "TP02"
    assert policy.context == "homework"
    assert policy.direct_answer == "Not initially - hint first"
    assert policy.attempt == "Required first"
    assert "TP01" in policy.also_applied, "lower-precedence rows that also applied"
    assert "S01" in policy.prohibited_strategies


async def test_table_versions_carry_versions_only() -> None:
    """Provenance and decisions are different kinds of fact.

    Mixing them is what let the policy row hide in a versions map for as long as
    it did.
    """
    record = await execute_turn(request(), build_deps())

    assert "teaching_policy_row" not in record.trace.table_versions
    assert "response_template" not in record.trace.table_versions
    assert record.response_template, "the template is recorded, as a decision"
    assert all("+" in value for value in record.trace.table_versions.values())


async def test_the_combination_verdict_is_recorded() -> None:
    record = await execute_turn(request(), build_deps())

    combination = record.combination
    assert combination is not None
    assert combination.rule
    assert combination.allowed is True, "nothing dropped on an ordinary turn"


async def test_assessment_shows_what_it_prohibits_even_with_nothing_to_override() -> None:
    """The panel must show the constraint, not only the moments it bites.

    Under assessment the Trigger Matrix already selects S07 for every intent, so
    the policy has nothing to veto and `override_applied` is False. That is a
    property worth seeing rather than a gap: the tables agree, so the override
    path is never exercised in normal operation.
    """
    deps = build_deps(detect_intent=fake_intent(Intent.HINT, special=SpecialHandling.ASSESSMENT))
    record = await execute_turn(request("give me a hint, I'm in a test"), deps)

    policy = record.teaching_policy
    assert policy is not None
    assert policy.context == "assessment"
    assert policy.direct_answer == "Prohibited"
    assert set(policy.prohibited_strategies) >= {"S01", "S02", "S03", "S04"}
    assert policy.override_applied is False
    assert record.trace.strategy_selection is not None
    assert (
        record.trace.strategy_selection.primary_strategy.strategy_id
        is StrategyId.CHECK_UNDERSTANDING
    )


def test_an_override_is_reported_when_the_policy_does_change_something() -> None:
    """Exercised directly, since the shipped tables never need the veto."""
    selection = StrategySelection(
        primary_strategy=StrategyRef(strategy_id=StrategyId.CHECK_UNDERSTANDING),
        context={
            "decision_trace": {
                "policy_row": "TP03",
                "policy_rows_applied": ["TP03", "TP01"],
                "policy_actions": [
                    {"rule": "veto_primary", "reason": "S04 is prohibited under assessment"}
                ],
                "policy_prohibited_strategies": ["S01", "S04"],
            }
        },
    )

    policy = _policy_applied(selection, build_deps())

    assert policy is not None
    assert policy.override_applied is True
    assert policy.override_notes == ["S04 is prohibited under assessment"]
