"""Response Planner (R9, evaluation suite E2 — no LLM).

The property under test throughout: a constraint conflict is impossible by
construction. `full_solution_allowed=false` does not merely set a flag, it
removes the solution blocks from the emitted structure, so a generator handed
the plan has no section that tells it to state the answer.
"""

from __future__ import annotations

import pytest

from socratic_tutor.models.enums import (
    Evaluation,
    Intent,
    Language,
    LearnerState,
    SpecialHandling,
    StrategyId,
    StudentLevel,
)
from socratic_tutor.models.intent import IntentResult, SpecialHandlingResult
from socratic_tutor.models.plan import ResponsePlan
from socratic_tutor.models.strategy import Constraints, StrategyRef, StrategySelection
from socratic_tutor.pedagogy.planner import (
    PlannerError,
    PlanTrace,
    plan_conflicts,
    plan_response,
)
from socratic_tutor.pedagogy.strategy import select_strategy
from socratic_tutor.pedagogy.tables import RuleTables, load_tables

#: Seed problem P03, `2x + 4 = 10` — the spec's §10 worked example. See the note
#: on the same constant in `test_strategy_selection.py`.
TOPIC = "Two-Step Linear Equations"


@pytest.fixture(scope="module")
def tables() -> RuleTables:
    return load_tables()


def selection_for(
    primary: StrategyId,
    supporting: list[StrategyId] | None = None,
    *,
    direct_answer_allowed: bool = True,
    full_solution_allowed_now: bool = True,
    student_attempt_required: bool = False,
    student_level: StudentLevel = StudentLevel.BEGINNER,
) -> StrategySelection:
    """A `StrategySelection` built directly, so the planner is tested alone."""
    return StrategySelection(
        primary_strategy=StrategyRef(strategy_id=primary),
        supporting_strategies=[StrategyRef(strategy_id=s) for s in supporting or []],
        constraints=Constraints(
            direct_answer_allowed=direct_answer_allowed,
            full_solution_allowed_now=full_solution_allowed_now,
            student_attempt_required=student_attempt_required,
        ),
        context={"topic": TOPIC, "student_level": student_level.value},
    )


def end_to_end(
    tables: RuleTables,
    intent: Intent,
    *,
    learner_state: LearnerState = LearnerState.NORMAL,
    special_handling: SpecialHandling = SpecialHandling.NONE,
    student_level: StudentLevel = StudentLevel.BEGINNER,
    evaluation: Evaluation | None = None,
    rag_evidence_present: bool = True,
    language: Language = Language.EN,
    trace: PlanTrace | None = None,
) -> ResponsePlan:
    """Strategy selection into planning, the way the pipeline runs it."""
    selection = select_strategy(
        IntentResult(
            intent=intent,
            learner_state=learner_state,
            special_handling=SpecialHandlingResult(
                detected=special_handling is not SpecialHandling.NONE, type=special_handling
            ),
        ),
        student_level,
        TOPIC,
        None,
        evaluation,
        rag_evidence_present,
        tables,
    )
    return plan_response(selection, tables, language=language, trace=trace)


def end_to_end_with_trace(
    tables: RuleTables, intent: Intent, *, trace: PlanTrace, **kwargs
) -> ResponsePlan:
    return end_to_end(tables, intent, trace=trace, **kwargs)


# --- §4.1 templates -------------------------------------------------------


@pytest.mark.parametrize(
    ("strategy", "structure", "max_words"),
    [
        (
            StrategyId.DIRECT_ANSWER_EXPLANATION,
            ["direct_answer", "explanation", "source_reference", "check_understanding"],
            120,
        ),
        (StrategyId.STEP_BY_STEP_GUIDANCE, ["ordered_steps", "next_step_prompt"], 100),
        (StrategyId.HINT_SCAFFOLD, ["single_hint", "student_attempt_prompt"], 50),
        (
            StrategyId.RE_EXPLAIN_DIFFERENTLY,
            ["reframed_explanation", "one_example_or_analogy", "single_check_question"],
            120,
        ),
        (StrategyId.FEEDBACK, ["state_evaluation", "explain_feedback", "next_action"], 100),
        (StrategyId.PRACTICE_QUIZ, ["single_practice_question"], 60),
    ],
)
def test_spec_templates_are_transcribed_verbatim(
    tables: RuleTables, strategy: StrategyId, structure: list[str], max_words: int
) -> None:
    template = tables.response_templates.template(strategy)
    assert template.source == "spec"
    assert list(template.structure) == structure
    assert template.max_words == max_words


def test_wait_for_student_matches_the_spec_column(tables: RuleTables) -> None:
    templates = tables.response_templates
    assert templates.template(StrategyId.DIRECT_ANSWER_EXPLANATION).wait_for_student is False
    assert templates.template(StrategyId.HINT_SCAFFOLD).wait_for_student is True
    assert templates.template(StrategyId.PRACTICE_QUIZ).wait_for_student is True
    # §4.1's "Depends" for S06: it waits exactly when policy requires an attempt.
    s06 = templates.template(StrategyId.FEEDBACK)
    assert s06.wait_for_student == "depends_on_attempt_required"
    assert s06.waits(student_attempt_required=True) is True
    assert s06.waits(student_attempt_required=False) is False


def test_global_rules_transcribe_section_4_2(tables: RuleTables) -> None:
    rules = tables.global_rules
    assert rules.max_examples == 1
    assert rules.max_analogies == 1
    assert rules.max_questions == 1
    assert rules.source_reference_required is True
    assert rules.unsupported_claims_allowed is False
    assert rules.hint_first_and_full_solution_same_turn is False
    assert rules.quiz_answer_before_student_attempt is False
    assert rules.tone == "clear_respectful_encouraging_non_patronizing"


def test_student_level_wording_covers_every_level(tables: RuleTables) -> None:
    for level in StudentLevel:
        assert tables.student_level_wording[level.value]


# --- the six planning steps ----------------------------------------------


def test_primary_template_drives_the_structure(tables: RuleTables) -> None:
    plan = plan_response(selection_for(StrategyId.DIRECT_ANSWER_EXPLANATION), tables)
    assert plan.structure == [
        "direct_answer",
        "explanation",
        "source_reference",
        "check_understanding",
    ]
    assert plan.max_words == 120


def test_supporting_blocks_are_added_before_the_closing_question(
    tables: RuleTables,
) -> None:
    """§3.3's sequence is Answer -> Example -> Check, not Answer -> Check -> Example."""
    plan = plan_response(
        selection_for(
            StrategyId.DIRECT_ANSWER_EXPLANATION,
            [StrategyId.EXAMPLE_ANALOGY, StrategyId.CHECK_UNDERSTANDING],
        ),
        tables,
    )
    assert plan.structure == [
        "direct_answer",
        "explanation",
        "source_reference",
        "one_example_or_analogy",
        "check_understanding",
    ]


def test_supporting_block_needing_a_response_is_deferred(tables: RuleTables) -> None:
    """ "Hint -> Student tries -> Guide": the guide is the *next* turn (§3.3)."""
    trace = PlanTrace()
    plan = plan_response(
        selection_for(
            StrategyId.HINT_SCAFFOLD,
            [StrategyId.STEP_BY_STEP_GUIDANCE, StrategyId.CHECK_UNDERSTANDING],
            direct_answer_allowed=False,
            full_solution_allowed_now=False,
            student_attempt_required=True,
        ),
        tables,
        trace=trace,
    )
    assert "ordered_steps" not in plan.structure
    assert any(action["rule"] == "deferred_to_next_turn" for action in trace.actions)


def test_quiz_feedback_block_is_deferred_to_the_next_turn(tables: RuleTables) -> None:
    """ "Question -> Answer -> Feedback": S06 evaluates an answer not yet given."""
    plan = plan_response(selection_for(StrategyId.PRACTICE_QUIZ, [StrategyId.FEEDBACK]), tables)
    assert plan.structure == ["single_practice_question"]


def test_student_level_sets_the_language_level_and_wording_rule(
    tables: RuleTables,
) -> None:
    trace = PlanTrace()
    plan = plan_response(
        selection_for(StrategyId.DIRECT_ANSWER_EXPLANATION, student_level=StudentLevel.ADVANCED),
        tables,
        trace=trace,
    )
    assert plan.language_level is StudentLevel.ADVANCED
    assert trace.wording_rule == "Concise, technical"


def test_language_reaches_the_plan(tables: RuleTables) -> None:
    plan = plan_response(selection_for(StrategyId.HINT_SCAFFOLD), tables, language=Language.TH)
    assert plan.language is Language.TH


def test_missing_template_raises_rather_than_planning_an_empty_turn(
    tables: RuleTables,
) -> None:
    with pytest.raises(PlannerError, match="S02"):
        plan_response(selection_for(StrategyId.EXAMPLE_ANALOGY), tables)


# --- constraints are structural, not advisory ----------------------------


def test_withheld_solution_removes_the_solution_block_from_the_structure(
    tables: RuleTables,
) -> None:
    """The flag is not the protection; the missing block is."""
    trace = PlanTrace()
    plan = plan_response(
        selection_for(
            StrategyId.DIRECT_ANSWER_EXPLANATION,
            direct_answer_allowed=False,
            full_solution_allowed_now=False,
        ),
        tables,
        trace=trace,
    )
    assert "direct_answer" not in plan.structure
    assert plan.full_solution_allowed is False
    assert any(action["rule"] == "full_solution_withheld" for action in trace.actions)
    assert plan_conflicts(plan, tables.response_templates) == []


def test_a_plan_never_holds_a_hint_and_a_solution_together(tables: RuleTables) -> None:
    """§4.2 `hint_first_and_full_solution_same_turn: false`, enforced structurally."""
    plan = plan_response(
        selection_for(
            StrategyId.FEEDBACK,
            [StrategyId.HINT_SCAFFOLD],
            full_solution_allowed_now=True,
            direct_answer_allowed=True,
            student_attempt_required=True,
        ),
        tables,
    )
    blocks = [tables.response_templates.block(name) for name in plan.structure]
    assert any(block.is_hint for block in blocks)
    assert not any(block.is_solution or block.is_steps for block in blocks)
    assert plan_conflicts(plan, tables.response_templates) == []


def test_a_quiz_plan_never_contains_the_answer(tables: RuleTables) -> None:
    plan = plan_response(selection_for(StrategyId.PRACTICE_QUIZ), tables)
    blocks = [tables.response_templates.block(name) for name in plan.structure]
    assert not any(block.is_solution for block in blocks)
    assert plan.structure == ["single_practice_question"]


# --- `full_solution_allowed` describes the turn, not the context ----------
#
# Three mechanisms key off this one flag: the guardrail gates its entire
# canonical scan on it, the orchestrator decides between buffer-and-scan and
# live streaming on it, and the generation prompt tells the model whether it may
# state a solution. Reported permissively for a hint-only turn, all three fail
# open at once — with no adversarial input, on the most ordinary turn there is.


def test_a_hint_only_turn_reports_false_even_when_policy_permits_a_solution(
    tables: RuleTables,
) -> None:
    """The exact shipped-table path: "I'm stuck on 2x + 4 = 10, can you help?"

    solve + confused -> TM08 -> S04 Hint -> hint-only template, under TP01
    normal_learning, which leaves `full_solution_allowed_now` True.
    """
    trace = PlanTrace()
    plan = end_to_end_with_trace(
        tables, Intent.SOLVE, learner_state=LearnerState.CONFUSED, trace=trace
    )
    blocks = [tables.response_templates.block(name) for name in plan.structure]
    assert not any(block.is_solution for block in blocks)
    assert plan.full_solution_allowed is False
    assert any(action["rule"] == "full_solution_not_stated_this_turn" for action in trace.actions)


@pytest.mark.parametrize("intent", list(Intent))
@pytest.mark.parametrize(
    "special_handling",
    [SpecialHandling.NONE, SpecialHandling.HOMEWORK, SpecialHandling.ASSESSMENT],
)
@pytest.mark.parametrize("rag_evidence_present", [True, False])
def test_no_solution_block_means_no_full_solution_allowed(
    tables: RuleTables,
    intent: Intent,
    special_handling: SpecialHandling,
    rag_evidence_present: bool,
) -> None:
    """The generalising sweep, over every reachable turn shape."""
    for learner_state in LearnerState:
        plan = end_to_end(
            tables,
            intent,
            learner_state=learner_state,
            special_handling=special_handling,
            rag_evidence_present=rag_evidence_present,
        )
        states_solution = any(
            tables.response_templates.block(name).is_solution
            for name in plan.structure
            if name in tables.response_templates.blocks
        )
        if not states_solution:
            assert plan.full_solution_allowed is False, (
                f"{intent.value}/{learner_state.value}/{special_handling.value}: "
                "plan claims a full solution is allowed but emits no block that states one"
            )


@pytest.mark.parametrize("intent", list(Intent))
def test_the_prompt_policy_layer_never_invites_a_solution_the_plan_omits(
    tables: RuleTables, intent: Intent
) -> None:
    """The downstream consequence, asserted where the model would read it."""
    from socratic_tutor.pedagogy.generation import _policy_layer

    for learner_state in LearnerState:
        plan = end_to_end(tables, intent, learner_state=learner_state)
        states_solution = any(
            tables.response_templates.block(name).is_solution
            for name in plan.structure
            if name in tables.response_templates.blocks
        )
        policy = _policy_layer(plan)
        if not states_solution:
            assert "You may give the complete solution this turn." not in policy, (
                f"{intent.value}/{learner_state.value}: the prompt invites a solution "
                "the plan has no block for"
            )
            assert "may NOT give the final answer" in policy


def test_question_blocks_are_capped_at_the_global_maximum(tables: RuleTables) -> None:
    trace = PlanTrace()
    plan = plan_response(
        selection_for(StrategyId.STEP_BY_STEP_GUIDANCE, [StrategyId.CHECK_UNDERSTANDING]),
        tables,
        trace=trace,
    )
    questions = [
        name for name in plan.structure if tables.response_templates.block(name).is_question
    ]
    assert len(questions) == tables.global_rules.max_questions == 1
    assert any(action["rule"] == "max_questions" for action in trace.actions)


def test_a_plan_is_never_emitted_empty(tables: RuleTables) -> None:
    """An unconstrained generator is worse than a narrowly constrained one."""
    trace = PlanTrace()
    plan = plan_response(
        selection_for(
            StrategyId.DIRECT_ANSWER_EXPLANATION,
            direct_answer_allowed=False,
            full_solution_allowed_now=False,
        ),
        tables,
        trace=trace,
    )
    assert plan.structure


@pytest.mark.parametrize("intent", list(Intent))
@pytest.mark.parametrize(
    "special_handling", [SpecialHandling.NONE, SpecialHandling.HOMEWORK, SpecialHandling.ASSESSMENT]
)
@pytest.mark.parametrize("rag_evidence_present", [True, False])
def test_no_reachable_turn_produces_a_conflicting_plan(
    tables: RuleTables,
    intent: Intent,
    special_handling: SpecialHandling,
    rag_evidence_present: bool,
) -> None:
    """The invariant across the whole reachable space, not one example of it."""
    for learner_state in LearnerState:
        plan = end_to_end(
            tables,
            intent,
            learner_state=learner_state,
            special_handling=special_handling,
            rag_evidence_present=rag_evidence_present,
        )
        assert plan_conflicts(plan, tables.response_templates) == [], (
            f"{intent.value}/{learner_state.value}/{special_handling.value}"
        )


@pytest.mark.parametrize("intent", list(Intent))
def test_homework_and_assessment_plans_never_carry_a_solution_block(
    tables: RuleTables, intent: Intent
) -> None:
    for special_handling in (SpecialHandling.HOMEWORK, SpecialHandling.ASSESSMENT):
        plan = end_to_end(tables, intent, special_handling=special_handling)
        blocks = [tables.response_templates.block(name) for name in plan.structure]
        assert not any(block.is_solution for block in blocks)
        assert plan.full_solution_allowed is False


def test_assessment_plans_carry_no_hint_either(tables: RuleTables) -> None:
    """The §3.1 ruling reaching all the way into the generator's brief."""
    for intent in Intent:
        plan = end_to_end(tables, intent, special_handling=SpecialHandling.ASSESSMENT)
        blocks = [tables.response_templates.block(name) for name in plan.structure]
        assert not any(block.is_hint for block in blocks)
        assert plan.structure == ["single_check_question"]


# --- the worked example, planned -----------------------------------------


def test_worked_example_plan(tables: RuleTables) -> None:
    """The §10 homework turn, carried through step 4."""
    plan = end_to_end(
        tables,
        Intent.CHECK_ANSWER,
        special_handling=SpecialHandling.HOMEWORK,
        evaluation=Evaluation.INCORRECT,
    )
    assert plan.structure == [
        "state_evaluation",
        "explain_feedback",
        "next_action",
        "single_hint",
    ]
    assert plan.max_words == 100
    assert plan.max_questions == 1
    assert plan.language_level is StudentLevel.BEGINNER
    assert plan.full_solution_allowed is False
    assert plan.wait_for_student is True
    assert plan_conflicts(plan, tables.response_templates) == []


def test_planning_is_deterministic(tables: RuleTables) -> None:
    def once() -> str:
        return end_to_end(
            tables,
            Intent.CHECK_ANSWER,
            special_handling=SpecialHandling.HOMEWORK,
            evaluation=Evaluation.INCORRECT,
        ).model_dump_json()

    assert once() == once()
