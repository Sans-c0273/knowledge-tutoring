"""Schema contract tests — Tech Spec §2, §4, §7.

Every model must survive a JSON round trip (traces are persisted and replayed
into the glass-box UI), and `TurnTrace` must serialize when only the first step
ran (a turn that failed at step 1 still has to render).
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ValidationError

from socratic_tutor.models import (
    GLOBAL_RULES,
    STRATEGY_NAMES,
    Constraints,
    Evaluation,
    GlobalRules,
    GuardrailEvent,
    GuardrailEventType,
    Intent,
    IntentResult,
    KnowledgeAction,
    KnowledgeContext,
    KnowledgeGuidance,
    Language,
    LearnerState,
    RelatedTopic,
    Relation,
    ResponsePlan,
    RetrievedChunk,
    SpecialHandling,
    SpecialHandlingResult,
    StepLatency,
    StrategyId,
    StrategyRef,
    StrategySelection,
    StudentLevel,
    StudentModel,
    TopicLevel,
    TurnTrace,
)


def round_trip(instance: BaseModel) -> BaseModel:
    """Serialize to JSON text and validate it back — what persistence really does."""
    return type(instance).model_validate_json(instance.model_dump_json())


SAMPLES: list[BaseModel] = [
    SpecialHandlingResult(detected=True, type=SpecialHandling.HOMEWORK),
    IntentResult(
        intent=Intent.CHECK_ANSWER,
        learner_state=LearnerState.CONFUSED,
        special_handling=SpecialHandlingResult(detected=True, type=SpecialHandling.HOMEWORK),
        confidence=0.91,
    ),
    TopicLevel(topic="Cell Structure", level=StudentLevel.INTERMEDIATE),
    StudentModel(
        course_id="BIO101",
        student_topic_levels=[
            TopicLevel(topic="Cell Structure", level=StudentLevel.INTERMEDIATE),
            TopicLevel(topic="Photosynthesis", level=StudentLevel.BEGINNER),
        ],
    ),
    RelatedTopic(topic="Inverse Operations", distance=1),
    KnowledgeContext(
        current_topic="Linear Equations",
        prerequisites=[RelatedTopic(topic="Basic Arithmetic", distance=2)],
        related_topics=[RelatedTopic(topic="Equation Balance")],
        next_topics=[RelatedTopic(topic="Variables on Both Sides")],
        relationships=["sunlight provides energy for photosynthesis"],
    ),
    RetrievedChunk(
        text="สมการเชิงเส้นคือ...",
        source_ref="unit2.pdf#p4",
        topic="Linear Equations",
        lang=Language.TH,
        score=0.82,
    ),
    StrategyRef(strategy_id=StrategyId.FEEDBACK),
    Constraints(),
    KnowledgeGuidance(
        action=KnowledgeAction.USE_PREREQUISITE_AS_SCAFFOLD,
        target_concept="Light Energy",
        relation=Relation.USES,
        direction="inverse",
        distance=1,
    ),
    RelatedTopic(topic="Equation Balance", distance=1, relation=Relation.RELATED_TO),
    StrategySelection(
        primary_strategy=StrategyRef(strategy_id=StrategyId.FEEDBACK),
        evaluation=Evaluation.PARTIALLY_CORRECT,
        supporting_strategies=[StrategyRef(strategy_id=StrategyId.HINT_SCAFFOLD)],
        context={"topic": "Linear Equations", "student_level": "beginner"},
    ),
    GlobalRules(),
    ResponsePlan(
        structure=["single_hint", "student_attempt_prompt"],
        max_words=50,
        language_level=StudentLevel.BEGINNER,
        language=Language.TH,
        wait_for_student=True,
    ),
    GuardrailEvent(
        type=GuardrailEventType.LEAK_BLOCKED,
        detail="canonical answer 'x = 4'",
        action_taken="regenerate",
    ),
    StepLatency(step="intent", ms=182.5),
]


@pytest.mark.parametrize("instance", SAMPLES, ids=lambda i: type(i).__name__)
def test_round_trips_through_json(instance: BaseModel) -> None:
    assert round_trip(instance) == instance


@pytest.mark.parametrize("instance", SAMPLES, ids=lambda i: type(i).__name__)
def test_dumps_to_plain_json(instance: BaseModel) -> None:
    """`model_dump(mode="json")` must be `json.dumps`-able with no custom encoder."""
    json.dumps(instance.model_dump(mode="json"))


# --------------------------------------------------------- spec §2.1 / §2.2


def test_intent_result_matches_spec_shape() -> None:
    """Tech Spec §2.1 field names, nesting and values, verbatim.

    The exact-shape assertion (these four fields and no others) belongs to the
    wire schema and lives in `test_intent.py::test_wire_schema_is_exactly_the_spec_shape`;
    `IntentResult` additionally carries the fallback bookkeeping, which must stay
    inert until `apply_confidence_floor` sets it.
    """
    payload = IntentResult().model_dump(mode="json")
    assert set(payload["special_handling"]) == {"detected", "type"}
    spec_fields = {"intent", "learner_state", "special_handling", "confidence"}
    assert spec_fields <= set(payload)
    assert {key: payload[key] for key in spec_fields} == {
        "intent": "explain",
        "learner_state": "normal",
        "special_handling": {"detected": False, "type": "none"},
        "confidence": 0.0,
    }
    assert payload["low_confidence_fallback"] is False
    assert payload["raw_intent"] is None


def test_intent_result_parses_raw_spec_json() -> None:
    raw = """
    {"intent": "hint", "learner_state": "confused",
     "special_handling": {"detected": true, "type": "homework"}, "confidence": 0.72}
    """
    result = IntentResult.model_validate_json(raw)
    assert result.intent is Intent.HINT
    assert result.special_handling.type is SpecialHandling.HOMEWORK


def test_level_for_known_topic() -> None:
    student = StudentModel(
        course_id="BIO101",
        student_topic_levels=[TopicLevel(topic="Cell Structure", level=StudentLevel.ADVANCED)],
    )
    assert student.level_for("Cell Structure") is StudentLevel.ADVANCED


def test_level_for_unknown_topic_defaults_to_beginner() -> None:
    """ "No questionnaire completed → default = beginner" (Tech Spec §2.2)."""
    student = StudentModel(course_id="BIO101")
    assert student.level_for("Photosynthesis") is StudentLevel.BEGINNER
    assert (
        StudentModel(
            course_id="BIO101",
            student_topic_levels=[TopicLevel(topic="Cell Structure", level=StudentLevel.ADVANCED)],
        ).level_for("Mitosis")
        is StudentLevel.BEGINNER
    )


def test_level_for_ignores_case_and_surrounding_space() -> None:
    student = StudentModel(
        course_id="MATH101",
        student_topic_levels=[
            TopicLevel(topic="Linear Equations", level=StudentLevel.INTERMEDIATE)
        ],
    )
    assert student.level_for("  linear equations ") is StudentLevel.INTERMEDIATE


# ------------------------------------------------------------- spec §2.4 / §3


def test_strategy_name_is_derived_from_the_id() -> None:
    assert StrategyRef(strategy_id=StrategyId.FEEDBACK).strategy_name == "feedback"
    assert StrategyRef(strategy_id=StrategyId.HINT_SCAFFOLD).strategy_name == "hint_scaffold"


def test_strategy_library_covers_s01_to_s08() -> None:
    assert [s.value for s in StrategyId] == [f"S0{n}" for n in range(1, 9)]
    assert set(STRATEGY_NAMES) == set(StrategyId)


def test_constraints_default_to_the_restrictive_policy_row() -> None:
    """Fail-closed: an unset plan must not permit a direct answer (E3 gate, Tech Spec §9)."""
    constraints = Constraints()
    assert not constraints.direct_answer_allowed
    assert not constraints.full_solution_allowed_now
    assert constraints.student_attempt_required
    assert not ResponsePlan().full_solution_allowed


def test_strategy_selection_context_accessors() -> None:
    selection = StrategySelection(
        primary_strategy=StrategyRef(strategy_id=StrategyId.STEP_BY_STEP_GUIDANCE),
        context={"topic": "Linear Equations", "student_level": "advanced"},
    )
    assert selection.topic == "Linear Equations"
    assert selection.student_level is StudentLevel.ADVANCED
    assert (
        StrategySelection(
            primary_strategy=StrategyRef(strategy_id=StrategyId.FEEDBACK)
        ).student_level
        is None
    )


def test_knowledge_guidance_relation_is_the_closed_vocabulary() -> None:
    """KM02/KM03 assertions need the edge type, not a string that can drift."""
    guidance = KnowledgeGuidance(
        action=KnowledgeAction.CHECK_OR_SCAFFOLD_PREREQUISITE,
        target_concept="Inverse Operations",
        relation=Relation.PREREQUISITE_OF,
        distance=1,
    )
    assert guidance.relation is Relation.PREREQUISITE_OF
    assert guidance.direction == "forward"
    with pytest.raises(ValidationError):
        KnowledgeGuidance(relation="used_in")


def test_knowledge_guidance_no_longer_accepts_the_pre_rename_name() -> None:
    """The compatibility alias is gone: `relationship=` must not silently do nothing."""
    with pytest.raises(ValidationError):
        KnowledgeGuidance(relationship="related_to")


def test_a_null_relation_means_unknown_not_a_default() -> None:
    """A walk that cannot confirm the edge type must say nothing, not guess."""
    assert KnowledgeGuidance().relation is None
    assert RelatedTopic(topic="Equation Balance").relation is None


def test_related_topic_carries_the_edge_that_reached_it() -> None:
    """The three KnowledgeContext buckets are lossy; the relation survives per item."""
    context = KnowledgeContext(
        current_topic="Combining Like Terms",
        related_topics=[
            RelatedTopic(topic="Variables and Expressions", distance=1, relation=Relation.USES),
            RelatedTopic(topic="Evaluating Expressions", distance=1, relation=Relation.PART_OF),
        ],
    )
    assert [t.relation for t in context.related_topics] == [Relation.USES, Relation.PART_OF]
    assert round_trip(context) == context


def test_guardrail_event_type_is_closed_with_an_escape_hatch() -> None:
    """E3 counts leaks and saves by type; a typo must fail loudly, not silently."""
    assert {t.value for t in GuardrailEventType} == {
        "leak_blocked",
        "plan_violation",
        "regenerated",
        "fallback_served",
        "other",
    }
    assert GuardrailEvent(type=GuardrailEventType.OTHER, detail="new check").type is (
        GuardrailEventType.OTHER
    )
    with pytest.raises(ValidationError):
        GuardrailEvent(type="leak_bloked")


def test_relation_vocabulary_is_closed_to_the_five_spec_relations() -> None:
    assert {r.value for r in Relation} == {
        "prerequisite_of",
        "related_to",
        "next_topic",
        "part_of",
        "uses",
    }


# --------------------------------------------------------------- spec §4.2


def test_global_rules_match_the_spec_defaults() -> None:
    assert GLOBAL_RULES.model_dump() == {
        "max_examples": 1,
        "max_analogies": 1,
        "max_questions": 1,
        "source_reference_required": True,
        "unsupported_claims_allowed": False,
        "hint_first_and_full_solution_same_turn": False,
        "quiz_answer_before_student_attempt": False,
        "tone": "clear_respectful_encouraging_non_patronizing",
    }


# ----------------------------------------------------------------- spec §7


def test_turn_trace_serializes_when_only_step_one_ran() -> None:
    """A turn that failed at step 1 still has to persist and still has to render."""
    trace = TurnTrace(session_id="s1", student_message="ช่วยอธิบายหน่อย")
    trace.intent_result = IntentResult(intent=Intent.EXPLAIN, confidence=0.4)
    trace.error = "topic mapping failed: no syllabus match"

    restored = round_trip(trace)
    assert restored.intent_result is not None
    assert restored.strategy_selection is None
    assert restored.response_plan is None
    assert restored.generated_text is None
    assert restored.retrieved_chunks == []
    assert restored.turn_id == trace.turn_id
    assert restored.error == trace.error


def test_turn_trace_carries_every_pipeline_step() -> None:
    trace = TurnTrace(session_id="s1", student_message="Is x = 4 right?")
    trace.intent_result = IntentResult(intent=Intent.CHECK_ANSWER, confidence=0.95)
    trace.topic = "Linear Equations"
    trace.student_level = StudentLevel.BEGINNER
    trace.retrieved_chunks = [RetrievedChunk(text="...", source_ref="unit2.pdf#p4")]
    trace.knowledge_context = KnowledgeContext(current_topic="Linear Equations")
    trace.evaluation = Evaluation.INCORRECT
    trace.matched_trigger_rows = ["check_answer+normal"]
    trace.matched_km_rules = ["KM02", "KM03"]
    trace.strategy_selection = StrategySelection(
        primary_strategy=StrategyRef(strategy_id=StrategyId.FEEDBACK)
    )
    trace.response_plan = ResponsePlan(structure=["state_evaluation"])
    trace.generated_text = "Not quite — check the second step."
    trace.guardrail_events = [
        GuardrailEvent(type=GuardrailEventType.PLAN_VIOLATION, detail="word count 132/100")
    ]
    trace.model_ids = {"intent": "claude-haiku-4-5", "generation": "claude-sonnet-5"}
    trace.table_versions = {"trigger_matrix": "v1"}

    restored = round_trip(trace)
    assert restored == trace
    assert restored.matched_km_rules == ["KM02", "KM03"]
    assert restored.model_ids["generation"] == "claude-sonnet-5"


def test_step_latency_records_even_when_the_step_raises() -> None:
    trace = TurnTrace(session_id="s1", student_message="hi")
    with trace.step("intent"):
        pass
    with pytest.raises(RuntimeError), trace.step("retrieval"):
        raise RuntimeError("chroma is down")

    assert set(trace.latencies_ms) == {"intent", "retrieval"}
    assert all(ms >= 0.0 for ms in trace.latencies_ms.values())
    assert trace.total_latency_ms == pytest.approx(sum(trace.latencies_ms.values()))


def test_record_usage_accumulates_per_role() -> None:
    trace = TurnTrace(session_id="s1", student_message="hi")
    trace.record_usage("generation", input_tokens=1200, output_tokens=90)
    trace.record_usage("generation", input_tokens=300, output_tokens=40)
    assert trace.token_usage["generation"] == {"input": 1500, "output": 130}


def test_prompt_versions_identify_a_prompt_by_content() -> None:
    """§7 wants prompt versions so a behaviour change is attributable to an edit."""
    from socratic_tutor.pedagogy.evaluation import SYSTEM_PROMPT as EVALUATION_PROMPT
    from socratic_tutor.pedagogy.intent import SYSTEM_PROMPT as INTENT_PROMPT
    from socratic_tutor.providers.base import prompt_version

    assert prompt_version(INTENT_PROMPT) == prompt_version(INTENT_PROMPT)
    assert prompt_version(INTENT_PROMPT) != prompt_version(EVALUATION_PROMPT)
    # An edit of any size changes the version; nothing has to be bumped by hand.
    assert prompt_version(INTENT_PROMPT) != prompt_version(INTENT_PROMPT + " ")

    trace = TurnTrace(session_id="s", student_message="hi")
    trace.prompt_versions = {
        "intent": prompt_version(INTENT_PROMPT),
        "evaluation": prompt_version(EVALUATION_PROMPT),
    }
    assert round_trip(trace) == trace
