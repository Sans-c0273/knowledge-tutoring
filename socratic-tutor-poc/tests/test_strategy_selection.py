"""Strategy Selection engine (R7, evaluation suite E2 — 100% pass, no LLM).

Covers all four tables in sequence: every Trigger Matrix row fires for its key,
every Combination Rule is enforced, every Teaching Policy row overrides
correctly, and the §10 worked example reproduces field for field.
"""

from __future__ import annotations

import dataclasses

import pytest

from socratic_tutor.models.enums import (
    Evaluation,
    Intent,
    KnowledgeAction,
    LearnerState,
    Relation,
    SpecialHandling,
    StrategyId,
    StudentLevel,
)
from socratic_tutor.models.intent import IntentResult, SpecialHandlingResult
from socratic_tutor.models.knowledge import KnowledgeContext, RelatedTopic
from socratic_tutor.models.strategy import Constraints, StrategySelection
from socratic_tutor.pedagogy.strategy import UnmatchedLookup, select_strategy
from socratic_tutor.pedagogy.tables import RuleTables, load_tables

#: The spec's §10 worked example is seed problem P03, `2x + 4 = 10`, answer
#: `x = 3`, whose recorded common misconception is `x = 7` — a sign error moving
#: the constant. That is why the student's "I got x = 7" evaluates `incorrect`.
#: P01's topic would be inconsistent here: P01 is `x + 5 = 12`, where `x = 7` is
#: the *correct* answer, so an integration test wiring real evaluation to this
#: fixture would compute `correct` and select a different strategy than these
#: unit tests assert.
TOPIC = "Two-Step Linear Equations"


@pytest.fixture(scope="module")
def tables() -> RuleTables:
    return load_tables()


def intent_result(
    intent: Intent,
    learner_state: LearnerState = LearnerState.NORMAL,
    special_handling: SpecialHandling = SpecialHandling.NONE,
) -> IntentResult:
    return IntentResult(
        intent=intent,
        learner_state=learner_state,
        special_handling=SpecialHandlingResult(
            detected=special_handling is not SpecialHandling.NONE, type=special_handling
        ),
        confidence=0.9,
    )


def run(
    tables: RuleTables,
    intent: Intent,
    *,
    learner_state: LearnerState = LearnerState.NORMAL,
    student_level: StudentLevel = StudentLevel.BEGINNER,
    special_handling: SpecialHandling = SpecialHandling.NONE,
    knowledge_context: KnowledgeContext | None = None,
    evaluation: Evaluation | None = None,
    rag_evidence_present: bool = True,
    on_unmatched=None,
) -> StrategySelection:
    return select_strategy(
        intent_result(intent, learner_state, special_handling),
        student_level,
        TOPIC,
        knowledge_context,
        evaluation,
        rag_evidence_present,
        tables,
        on_unmatched=on_unmatched,
    )


def strategies(selection: StrategySelection) -> tuple[str, list[str]]:
    return (
        selection.primary_strategy.strategy_id.value,
        [ref.strategy_id.value for ref in selection.supporting_strategies],
    )


def trace(selection: StrategySelection) -> dict:
    return selection.context["decision_trace"]


# --- 1. Trigger Matrix ----------------------------------------------------

#: One probe per Trigger Matrix row: (row id, intent, learner state, level, handling).
#: Each probe is chosen so that row wins the priority sort, and so that Teaching
#: Policy leaves its strategies untouched — the assertion is then both "this row
#: fired" and "it produced the strategies the table declares".
TRIGGER_PROBES = [
    ("TM01", Intent.EXPLAIN, LearnerState.NORMAL, StudentLevel.BEGINNER, SpecialHandling.NONE),
    ("TM02", Intent.EXPLAIN, LearnerState.NORMAL, StudentLevel.INTERMEDIATE, SpecialHandling.NONE),
    ("TM03", Intent.EXPLAIN, LearnerState.CONFUSED, StudentLevel.BEGINNER, SpecialHandling.NONE),
    ("TM04", Intent.EXPLAIN, LearnerState.CONFUSED, StudentLevel.ADVANCED, SpecialHandling.NONE),
    ("TM05", Intent.CLARIFY, LearnerState.NORMAL, StudentLevel.BEGINNER, SpecialHandling.NONE),
    ("TM06", Intent.CLARIFY, LearnerState.CONFUSED, StudentLevel.BEGINNER, SpecialHandling.NONE),
    ("TM07", Intent.SOLVE, LearnerState.NORMAL, StudentLevel.BEGINNER, SpecialHandling.NONE),
    ("TM08", Intent.SOLVE, LearnerState.CONFUSED, StudentLevel.BEGINNER, SpecialHandling.NONE),
    ("TM09", Intent.HINT, LearnerState.NORMAL, StudentLevel.BEGINNER, SpecialHandling.NONE),
    ("TM10", Intent.CHECK_ANSWER, LearnerState.NORMAL, StudentLevel.BEGINNER, SpecialHandling.NONE),
    (
        "TM11",
        Intent.CHECK_ANSWER,
        LearnerState.CONFUSED,
        StudentLevel.BEGINNER,
        SpecialHandling.NONE,
    ),
    (
        "TM12",
        Intent.CHECK_ANSWER,
        LearnerState.NORMAL,
        StudentLevel.BEGINNER,
        SpecialHandling.HOMEWORK,
    ),
    (
        "TM13",
        Intent.PRACTICE_QUIZ,
        LearnerState.NORMAL,
        StudentLevel.BEGINNER,
        SpecialHandling.NONE,
    ),
    (
        "TM14",
        Intent.SUMMARIZE_REVIEW,
        LearnerState.NORMAL,
        StudentLevel.BEGINNER,
        SpecialHandling.NONE,
    ),
    (
        "TM15",
        Intent.SUMMARIZE_REVIEW,
        LearnerState.CONFUSED,
        StudentLevel.BEGINNER,
        SpecialHandling.NONE,
    ),
    ("TM16", Intent.SOLVE, LearnerState.NORMAL, StudentLevel.BEGINNER, SpecialHandling.HOMEWORK),
    (
        "TM17",
        Intent.EXPLAIN,
        LearnerState.NORMAL,
        StudentLevel.BEGINNER,
        SpecialHandling.ASSESSMENT,
    ),
]


@pytest.mark.parametrize(
    ("row_id", "intent", "learner_state", "student_level", "special_handling"),
    TRIGGER_PROBES,
    ids=[probe[0] for probe in TRIGGER_PROBES],
)
def test_every_trigger_row_fires_for_its_key(
    tables: RuleTables,
    row_id: str,
    intent: Intent,
    learner_state: LearnerState,
    student_level: StudentLevel,
    special_handling: SpecialHandling,
) -> None:
    row = next(candidate for candidate in tables.trigger_rows if candidate.id == row_id)
    selection = run(
        tables,
        intent,
        learner_state=learner_state,
        student_level=student_level,
        special_handling=special_handling,
    )
    assert trace(selection)["selected_trigger_row"] == row_id
    assert strategies(selection) == (
        row.primary.value,
        [s.value for s in row.supporting],
    )


def test_every_trigger_row_has_a_probe() -> None:
    """A new row without a probe would ship untested."""
    tables = load_tables()
    assert {probe[0] for probe in TRIGGER_PROBES} == {row.id for row in tables.trigger_rows}


def test_higher_priority_row_wins_when_several_match(tables: RuleTables) -> None:
    selection = run(tables, Intent.CHECK_ANSWER, special_handling=SpecialHandling.HOMEWORK)
    matched = [row["row_id"] for row in trace(selection)["matched_trigger_rows"]]
    assert matched == ["TM12", "TM16"]  # 60 beats 55
    assert trace(selection)["selected_trigger_row"] == "TM12"


def test_exact_row_beats_wildcard_row_at_equal_priority(tables: RuleTables) -> None:
    """TM01 (level=beginner) and TM02 (level=any) differ only in specificity."""
    rows = {row.id: row for row in tables.trigger_rows}
    tweaked = dataclasses.replace(rows["TM02"], priority=rows["TM01"].priority)
    patched = dataclasses.replace(
        tables,
        trigger_rows=tuple(
            sorted(
                (tweaked if row.id == "TM02" else row for row in tables.trigger_rows),
                key=lambda row: row.rank,
            )
        ),
    )
    selection = run(patched, Intent.EXPLAIN, student_level=StudentLevel.BEGINNER)
    assert trace(selection)["selected_trigger_row"] == "TM01"


def test_unmatched_lookup_uses_the_fallback_and_emits_a_review_event(
    tables: RuleTables,
) -> None:
    """Table gaps feed a review queue; the table grows from evidence."""
    empty = dataclasses.replace(tables, trigger_rows=())
    events: list[UnmatchedLookup] = []
    selection = run(empty, Intent.PRACTICE_QUIZ, on_unmatched=events.append)

    assert strategies(selection) == (tables.trigger_fallback.primary.value, [])
    assert len(events) == 1
    event = events[0].as_event()
    assert event["event"] == "trigger_matrix_unmatched"
    assert event["intent"] == "practice_quiz"
    assert event["special_handling"] == "none"
    assert event["topic"] == TOPIC
    assert event["fallback_row_id"] == tables.trigger_fallback.id
    assert event["trigger_matrix_version"] == tables.version_map["trigger_matrix"]
    assert trace(selection)["trigger_matched"] is False
    assert trace(selection)["unmatched_lookup"] == event


def test_matched_lookup_emits_no_review_event(tables: RuleTables) -> None:
    events: list[UnmatchedLookup] = []
    selection = run(tables, Intent.EXPLAIN, on_unmatched=events.append)
    assert events == []
    assert trace(selection)["unmatched_lookup"] is None


# --- 2. Strategy Library --------------------------------------------------


def test_strategy_ids_resolve_to_readable_names(tables: RuleTables) -> None:
    selection = run(tables, Intent.CHECK_ANSWER, special_handling=SpecialHandling.HOMEWORK)
    assert selection.primary_strategy.strategy_name == "feedback"
    assert [ref.strategy_name for ref in selection.supporting_strategies] == ["hint_scaffold"]
    assert trace(selection)["strategy_library"]["S06"] == "Feedback"


# --- 3. Combination Rules -------------------------------------------------


def with_row(tables: RuleTables, row_id: str, **changes) -> RuleTables:
    """Swap one Trigger Matrix row, bypassing the loader's cross-validation.

    Used to prove the engine enforces at turn time what the loader rejects at
    build time — the two defences are independent.
    """
    rows = tuple(
        dataclasses.replace(row, **changes) if row.id == row_id else row
        for row in tables.trigger_rows
    )
    return dataclasses.replace(tables, trigger_rows=tuple(sorted(rows, key=lambda r: r.rank)))


def test_disallowed_supporting_strategy_is_dropped_with_a_reason(
    tables: RuleTables,
) -> None:
    # S03 is not in S01's allowed supporting set {S02, S07}.
    patched = with_row(
        tables,
        "TM01",
        supporting=(StrategyId.STEP_BY_STEP_GUIDANCE, StrategyId.CHECK_UNDERSTANDING),
    )
    selection = run(patched, Intent.EXPLAIN)
    assert strategies(selection) == ("S01", ["S07"])
    dropped = [
        action
        for action in trace(selection)["combination_actions"]
        if action.get("dropped") == "S03"
    ]
    assert dropped and dropped[0]["rule"] == "allowed_combinations"
    assert "S02, S07" in dropped[0]["reason"]


def test_max_one_primary_and_two_supporting_is_enforced(tables: RuleTables) -> None:
    patched = with_row(
        tables,
        "TM10",
        supporting=(
            StrategyId.HINT_SCAFFOLD,
            StrategyId.RE_EXPLAIN_DIFFERENTLY,
            StrategyId.CHECK_UNDERSTANDING,
        ),
    )
    selection = run(patched, Intent.CHECK_ANSWER)
    primary, supporting = strategies(selection)
    assert primary == "S06"
    assert supporting == ["S04", "S05"]
    assert len(supporting) == tables.combinations.max_supporting
    dropped = [
        action
        for action in trace(selection)["combination_actions"]
        if action["rule"] == "max_supporting"
    ]
    assert dropped[0]["dropped"] == "S07"


def test_hint_first_beats_an_immediate_full_answer(tables: RuleTables) -> None:
    """S01 and S04 may never share a turn; the hint takes the primary slot."""
    patched = with_row(
        tables,
        "TM01",
        supporting=(StrategyId.HINT_SCAFFOLD, StrategyId.CHECK_UNDERSTANDING),
    )
    selection = run(patched, Intent.EXPLAIN)
    assert strategies(selection) == ("S04", ["S07"])
    actions = trace(selection)["combination_actions"]
    assert {
        "rule": "hint_first_beats_full_answer",
        "promoted": "S04",
        "replaced": "S01",
        "reason": "the hint takes the primary slot",
    } in actions
    assert any(action.get("dropped") == "S01" for action in actions)


def test_primary_repeated_as_supporting_is_dropped(tables: RuleTables) -> None:
    patched = with_row(
        tables,
        "TM07",
        supporting=(StrategyId.STEP_BY_STEP_GUIDANCE, StrategyId.CHECK_UNDERSTANDING),
    )
    selection = run(patched, Intent.SOLVE)
    assert strategies(selection) == ("S03", ["S07"])
    assert any(
        action["rule"] == "no_duplicate_primary"
        for action in trace(selection)["combination_actions"]
    )


def test_every_shipped_row_produces_a_legal_combination(tables: RuleTables) -> None:
    for row_id, intent, learner_state, student_level, special_handling in TRIGGER_PROBES:
        selection = run(
            tables,
            intent,
            learner_state=learner_state,
            student_level=student_level,
            special_handling=special_handling,
        )
        primary = selection.primary_strategy.strategy_id
        supporting = [ref.strategy_id for ref in selection.supporting_strategies]
        assert len(supporting) <= tables.combinations.max_supporting, row_id
        assert set(supporting) <= set(tables.combinations.allowed_supporting(primary)), row_id


# --- evaluation routing (§2.4, §5.2) --------------------------------------


def test_evaluation_rides_only_on_s06(tables: RuleTables) -> None:
    selection = run(tables, Intent.EXPLAIN, evaluation=Evaluation.CORRECT)
    assert selection.primary_strategy.strategy_id is StrategyId.DIRECT_ANSWER_EXPLANATION
    assert selection.evaluation is None
    assert any(
        action["rule"] == "evaluation_requires_s06"
        for action in trace(selection)["combination_actions"]
    )


def test_cannot_evaluate_routes_to_check_understanding_never_invented_feedback(
    tables: RuleTables,
) -> None:
    selection = run(tables, Intent.CHECK_ANSWER, evaluation=Evaluation.CANNOT_EVALUATE)
    assert strategies(selection) == ("S07", [])
    assert selection.evaluation is None
    assert any(
        action["rule"] == "cannot_evaluate_routes_to_check"
        for action in trace(selection)["combination_actions"]
    )


# --- 4. Teaching Policy — the hard override -------------------------------


def test_normal_learning_row_permits_a_direct_answer(tables: RuleTables) -> None:
    selection = run(tables, Intent.EXPLAIN)
    assert trace(selection)["policy_rows_applied"] == ["TP01"]
    assert selection.constraints.direct_answer_allowed is True
    assert selection.constraints.full_solution_allowed_now is True
    assert selection.constraints.student_attempt_required is False


@pytest.mark.parametrize("intent", list(Intent))
def test_homework_always_withholds_the_answer_and_requires_an_attempt(
    tables: RuleTables, intent: Intent
) -> None:
    selection = run(tables, intent, special_handling=SpecialHandling.HOMEWORK)
    assert trace(selection)["policy_row"] == "TP02"
    assert selection.constraints.direct_answer_allowed is False
    assert selection.constraints.full_solution_allowed_now is False
    assert selection.constraints.student_attempt_required is True
    assert selection.primary_strategy.strategy_id is not StrategyId.DIRECT_ANSWER_EXPLANATION


def test_homework_with_a_correct_looking_answer_still_withholds(tables: RuleTables) -> None:
    """The student's answer looking right does not unlock the homework answer."""
    selection = run(
        tables,
        Intent.CHECK_ANSWER,
        special_handling=SpecialHandling.HOMEWORK,
        evaluation=Evaluation.CORRECT,
    )
    assert selection.constraints.direct_answer_allowed is False
    assert selection.constraints.student_attempt_required is True
    assert selection.evaluation is Evaluation.CORRECT
    assert strategies(selection) == ("S06", ["S04"])


def test_homework_turn_always_carries_a_hint(tables: RuleTables) -> None:
    """ "Not initially — hint first" has to be real, not nominal."""
    patched = with_row(tables, "TM12", supporting=(StrategyId.CHECK_UNDERSTANDING,))
    selection = run(patched, Intent.CHECK_ANSWER, special_handling=SpecialHandling.HOMEWORK)
    _, supporting = strategies(selection)
    assert "S04" in supporting
    assert any(
        action["rule"] == "policy_requires_hint_first"
        for action in trace(selection)["policy_actions"]
    )


@pytest.mark.parametrize("intent", list(Intent))
def test_assessment_prohibits_a_direct_answer_for_every_intent(
    tables: RuleTables, intent: Intent
) -> None:
    selection = run(tables, intent, special_handling=SpecialHandling.ASSESSMENT)
    assert trace(selection)["policy_row"] == "TP03"
    assert selection.constraints.direct_answer_allowed is False
    assert selection.constraints.full_solution_allowed_now is False
    assert selection.constraints.student_attempt_required is True
    assert selection.primary_strategy.strategy_id is not StrategyId.DIRECT_ANSWER_EXPLANATION
    assert StrategyId.DIRECT_ANSWER_EXPLANATION not in [
        ref.strategy_id for ref in selection.supporting_strategies
    ]


#: Every strategy that emits something the student could use toward the graded
#: answer: the answer itself (S01), a worked example (S02), the solution steps
#: (S03), and a hint (S04). During an assessment none of them may be selected.
CLUE_EMITTING = {
    StrategyId.DIRECT_ANSWER_EXPLANATION,
    StrategyId.EXAMPLE_ANALOGY,
    StrategyId.STEP_BY_STEP_GUIDANCE,
    StrategyId.HINT_SCAFFOLD,
}


@pytest.mark.parametrize("intent", list(Intent))
@pytest.mark.parametrize("learner_state", list(LearnerState))
@pytest.mark.parametrize("student_level", list(StudentLevel))
def test_assessment_never_selects_a_clue_emitting_strategy(
    tables: RuleTables,
    intent: Intent,
    learner_state: LearnerState,
    student_level: StudentLevel,
) -> None:
    """Safety invariant, not a row test.

    Whatever the intent, learner state or level, an assessment turn may not emit
    a clue toward the answer — a hint toward a graded answer differs from a
    direct answer in degree, not in kind.
    """
    selection = run(
        tables,
        intent,
        learner_state=learner_state,
        student_level=student_level,
        special_handling=SpecialHandling.ASSESSMENT,
    )
    selected = {selection.primary_strategy.strategy_id} | {
        ref.strategy_id for ref in selection.supporting_strategies
    }
    assert not (selected & CLUE_EMITTING), f"{intent.value} leaked {selected & CLUE_EMITTING}"


def test_assessment_vetoes_a_clue_emitting_primary_the_matrix_chose(
    tables: RuleTables,
) -> None:
    """Policy is defence in depth, not a comment on the Trigger Matrix.

    Even if a future row selected S03 for an assessment turn, the policy row
    replaces it with the declared `veto_primary` rather than trusting the table.
    """
    patched = with_row(tables, "TM17", primary=StrategyId.STEP_BY_STEP_GUIDANCE, supporting=())
    selection = run(patched, Intent.SOLVE, special_handling=SpecialHandling.ASSESSMENT)
    assert strategies(selection) == ("S07", [])
    veto = [
        action
        for action in trace(selection)["policy_actions"]
        if action["rule"] == "policy_prohibits_strategy"
    ]
    assert veto and veto[0]["replaced"] == "S03" and veto[0]["promoted"] == "S07"


def test_assessment_hints_are_teacher_configurable(tables: RuleTables) -> None:
    """`allow_hints` is a real switch, and its shipped default is the strict one.

    Held constant: a Trigger Matrix that selects S04 for the assessment turn.
    With hints off (shipped) the policy vetoes it down to S07; with hints on the
    same table keeps it. So the strict behaviour is a policy choice a teacher can
    change, not something hard-coded above them.
    """
    hinting_matrix = with_row(tables, "TM17", primary=StrategyId.HINT_SCAFFOLD, supporting=())

    vetoed = run(hinting_matrix, Intent.SOLVE, special_handling=SpecialHandling.ASSESSMENT)
    assert strategies(vetoed) == ("S07", [])

    permissive = dataclasses.replace(
        hinting_matrix,
        policy_rows=tuple(
            dataclasses.replace(row, allow_hints=True) if row.context == "assessment" else row
            for row in hinting_matrix.policy_rows
        ),
    )
    allowed = run(permissive, Intent.SOLVE, special_handling=SpecialHandling.ASSESSMENT)
    assert strategies(allowed) == ("S04", [])
    # The direct-answer prohibition is not part of the switch and still holds.
    assert allowed.constraints.direct_answer_allowed is False


def test_allow_check_false_drops_the_closing_question(tables: RuleTables) -> None:
    """§3.4 marks the Check column teacher-configurable for assessment."""
    strict = tables.policy_row("normal_learning")
    patched = dataclasses.replace(
        tables,
        policy_rows=tuple(
            dataclasses.replace(strict, allow_check=False)
            if row.context == "normal_learning"
            else row
            for row in tables.policy_rows
        ),
    )
    selection = run(patched, Intent.EXPLAIN)
    assert StrategyId.CHECK_UNDERSTANDING not in [
        ref.strategy_id for ref in selection.supporting_strategies
    ]
    assert any(
        action["rule"] == "policy_disallows_check" for action in trace(selection)["policy_actions"]
    )


def test_no_rag_evidence_prohibits_a_direct_factual_answer(tables: RuleTables) -> None:
    """The anti-hallucination row: no evidence -> no direct factual answer."""
    selection = run(tables, Intent.EXPLAIN, rag_evidence_present=False)
    assert trace(selection)["selected_trigger_row"] == "TM01"  # the intent chose S01
    assert "TP04" in trace(selection)["policy_rows_applied"]
    assert selection.constraints.direct_answer_allowed is False
    assert trace(selection)["policy_effect"]["factual_claims_require_evidence"] is True
    # The planner routes to S07/S04 instead (§3.4 row 4).
    assert strategies(selection) == ("S04", ["S07"])


def test_policy_veto_overrides_what_the_intent_chose(tables: RuleTables) -> None:
    """Intent says "explain -> direct answer"; policy says no. Policy wins."""
    selection = run(tables, Intent.EXPLAIN, rag_evidence_present=False)
    veto = [
        action
        for action in trace(selection)["policy_actions"]
        if action["rule"] == "policy_vetoes_direct_answer"
    ]
    assert veto and veto[0]["replaced"] == "S01" and veto[0]["promoted"] == "S04"
    assert veto[0]["policy_row"] == "TP04"


def test_stacked_policy_rows_combine_restrictively(tables: RuleTables) -> None:
    """Homework and no evidence at once can only tighten, never loosen."""
    selection = run(
        tables,
        Intent.CHECK_ANSWER,
        special_handling=SpecialHandling.HOMEWORK,
        rag_evidence_present=False,
    )
    # TP01 is the session baseline and always applies; TP02 and TP04 tighten it.
    assert trace(selection)["policy_rows_applied"] == ["TP02", "TP04", "TP01"]
    assert selection.constraints.direct_answer_allowed is False
    assert selection.constraints.full_solution_allowed_now is False
    # TP04 is silent on attempts ("—"); TP02's requirement still stands.
    assert selection.constraints.student_attempt_required is True


def test_policy_sits_above_the_precedence_order(tables: RuleTables) -> None:
    assert trace(run(tables, Intent.EXPLAIN))["precedence"] == [
        "teaching_policy",
        "special_handling",
        "learner_state",
        "intent",
        "student_level",
    ]


# --- the self-report may only tighten -------------------------------------
#
# `special_handling` is detected from the student's own words. If saying
# something could *widen* what the tutor may do, the withholding architecture
# would be negotiable by the person it exists to constrain — and "my teacher said
# it's fine for this one" is a sentence students actually say.


def at_least_as_strict_as(tighter: Constraints, baseline: Constraints) -> bool:
    """Permissions may only be withdrawn; requirements may only be added."""
    return (
        (not tighter.direct_answer_allowed or baseline.direct_answer_allowed)
        and (not tighter.full_solution_allowed_now or baseline.full_solution_allowed_now)
        and (tighter.student_attempt_required or not baseline.student_attempt_required)
    )


@pytest.mark.parametrize("intent", list(Intent))
@pytest.mark.parametrize("learner_state", list(LearnerState))
@pytest.mark.parametrize("student_level", list(StudentLevel))
@pytest.mark.parametrize("rag_evidence_present", [True, False])
def test_a_self_report_can_never_widen_permissions(
    tables: RuleTables,
    intent: Intent,
    learner_state: LearnerState,
    student_level: StudentLevel,
    rag_evidence_present: bool,
) -> None:
    """The invariant, over the whole cross-product.

    Whatever the student says about their context, the turn that results is no
    more permissive than the same turn with nothing detected.
    """
    baseline = run(
        tables,
        intent,
        learner_state=learner_state,
        student_level=student_level,
        special_handling=SpecialHandling.NONE,
        rag_evidence_present=rag_evidence_present,
    ).constraints

    for special_handling in (SpecialHandling.HOMEWORK, SpecialHandling.ASSESSMENT):
        reported = run(
            tables,
            intent,
            learner_state=learner_state,
            student_level=student_level,
            special_handling=special_handling,
            rag_evidence_present=rag_evidence_present,
        ).constraints
        assert at_least_as_strict_as(reported, baseline), (
            f"{intent.value}/{special_handling.value}: self-report widened permissions "
            f"from {baseline!r} to {reported!r}"
        )


def test_the_session_baseline_always_applies(tables: RuleTables) -> None:
    """The structural reason the invariant holds, not just its consequence.

    A detected context is *added* to the baseline rather than replacing it, and
    applicable rows combine restrictively — so widening is not an operation the
    engine can perform, whatever a future table says.
    """
    baseline_id = tables.baseline_policy_row().id
    for special_handling in SpecialHandling:
        selection = run(tables, Intent.EXPLAIN, special_handling=special_handling)
        assert baseline_id in trace(selection)["policy_rows_applied"]


def test_tightening_the_baseline_tightens_every_unclassified_turn(
    tables: RuleTables,
) -> None:
    """The seam the assignment-context design change plugs into.

    Point `baseline_context` at `homework` and a student who never mentions
    homework is served hint-first anyway. Asserted so the seam is known to work
    before anyone has to rely on it.
    """
    strict = dataclasses.replace(tables, baseline_context="homework")
    permissive = run(tables, Intent.EXPLAIN).constraints
    assert permissive.direct_answer_allowed is True

    protected = run(strict, Intent.EXPLAIN).constraints
    assert protected.direct_answer_allowed is False
    assert protected.student_attempt_required is True


# --- 5. KM rules ----------------------------------------------------------


def knowledge(**kwargs) -> KnowledgeContext:
    defaults: dict = {
        "current_topic": TOPIC,
        "prerequisites": [],
        "related_topics": [],
        "next_topics": [],
    }
    defaults.update(kwargs)
    return KnowledgeContext(**defaults)


def test_km01_normal_learner_gets_current_topic_context(tables: RuleTables) -> None:
    selection = run(tables, Intent.EXPLAIN, knowledge_context=knowledge())
    assert trace(selection)["km_rule_applied"] == "KM01"
    guidance = selection.knowledge_guidance
    assert guidance is not None
    assert guidance.action is KnowledgeAction.USE_CURRENT_TOPIC_CONTEXT
    assert guidance.target_concept == TOPIC
    assert guidance.distance == 0


def test_km02_confused_with_a_prerequisite_checks_or_scaffolds_it(
    tables: RuleTables,
) -> None:
    """The action is check-or-scaffold; the engine never diagnoses a gap."""
    selection = run(
        tables,
        Intent.EXPLAIN,
        learner_state=LearnerState.CONFUSED,
        knowledge_context=knowledge(
            prerequisites=[RelatedTopic(topic="Inverse Operations", distance=1)]
        ),
    )
    assert trace(selection)["km_rule_applied"] == "KM02"
    guidance = selection.knowledge_guidance
    assert guidance is not None
    assert guidance.action is KnowledgeAction.CHECK_OR_SCAFFOLD_PREREQUISITE
    assert guidance.action is not KnowledgeAction.USE_PREREQUISITE_AS_SCAFFOLD
    assert guidance.target_concept == "Inverse Operations"
    assert guidance.relation is Relation.PREREQUISITE_OF
    assert guidance.direction == "forward"
    assert guidance.distance == 1


def test_km02_ignores_a_distant_prerequisite(tables: RuleTables) -> None:
    """KM02 returns the immediate prerequisite (distance 1) only."""
    selection = run(
        tables,
        Intent.EXPLAIN,
        learner_state=LearnerState.CONFUSED,
        knowledge_context=knowledge(
            prerequisites=[RelatedTopic(topic="Basic Arithmetic", distance=2)]
        ),
    )
    assert "KM02" not in trace(selection)["km_rules_matched"]
    assert selection.knowledge_guidance is None


def test_km03_clarify_with_a_related_concept_offers_a_reexplanation(
    tables: RuleTables,
) -> None:
    selection = run(
        tables,
        Intent.CLARIFY,
        knowledge_context=knowledge(
            related_topics=[RelatedTopic(topic="Equation Balance", distance=1)]
        ),
    )
    assert trace(selection)["km_rule_applied"] == "KM03"
    guidance = selection.knowledge_guidance
    assert guidance is not None
    assert guidance.action is KnowledgeAction.USE_RELATED_CONCEPT_FOR_REEXPLANATION
    assert guidance.target_concept == "Equation Balance"
    assert guidance.distance == 1


def test_km03_does_not_claim_an_edge_type_it_cannot_verify(tables: RuleTables) -> None:
    """`related_topics` folds `related_to`, `part_of` and `uses` into one bucket.

    `RelatedTopic` does not carry the edge type (`domain/klmap/lookup.py`
    collapses all three into one walk), so KM03 cannot tell which produced the
    concept. Guidance therefore leaves `relation` unset rather than asserting
    `related_to`. Set the relation on KM03 and replace this test once
    `RelatedTopic` carries its own.
    """
    selection = run(
        tables,
        Intent.CLARIFY,
        knowledge_context=knowledge(
            related_topics=[RelatedTopic(topic="Equation Balance", distance=1)]
        ),
    )
    assert selection.knowledge_guidance is not None
    assert selection.knowledge_guidance.relation is None


def test_km02_wins_over_km03_and_both_matches_are_logged(tables: RuleTables) -> None:
    """O2 tie-break: prerequisite relation present -> KM02 wins; log both."""
    selection = run(
        tables,
        Intent.CLARIFY,
        learner_state=LearnerState.CONFUSED,
        knowledge_context=knowledge(
            prerequisites=[RelatedTopic(topic="Inverse Operations", distance=1)],
            related_topics=[RelatedTopic(topic="Equation Balance", distance=1)],
        ),
    )
    assert trace(selection)["km_rules_matched"] == ["KM02", "KM03"]
    assert trace(selection)["km_rule_applied"] == "KM02"
    # Read from the structured entries, which are the one representation of the
    # contest — the losing rule names its winner rather than the winner keeping
    # a separate list of who it beat.
    losers = [m for m in trace(selection)["km_matches"] if not m["applied"]]
    assert [m["rule_id"] for m in losers] == ["KM03"]
    assert losers[0]["lost_to"] == "KM02"
    assert selection.knowledge_guidance is not None
    assert selection.knowledge_guidance.action is KnowledgeAction.CHECK_OR_SCAFFOLD_PREREQUISITE


def test_km04_does_not_fire_while_disabled(tables: RuleTables) -> None:
    selection = run(tables, Intent.PRACTICE_QUIZ, knowledge_context=knowledge())
    assert "KM04" not in trace(selection)["km_rules_matched"]
    assert "KM04" in trace(selection)["km_rules_disabled"]
    assert selection.knowledge_guidance is not None
    assert (
        selection.knowledge_guidance.action
        is not KnowledgeAction.PRACTICE_CURRENT_TOPIC_RELATIONSHIP
    )


def test_km05_does_not_fire_while_disabled(tables: RuleTables) -> None:
    selection = run(
        tables,
        Intent.EXPLAIN,
        learner_state=LearnerState.CONFUSED,
        knowledge_context=knowledge(
            next_topics=[RelatedTopic(topic="Variables on Both Sides", distance=1)]
        ),
    )
    assert trace(selection)["km_rules_matched"] == []
    assert "KM05" in trace(selection)["km_rules_disabled"]
    assert selection.knowledge_guidance is None


def test_no_knowledge_context_produces_no_guidance(tables: RuleTables) -> None:
    selection = run(tables, Intent.EXPLAIN, knowledge_context=None)
    assert selection.knowledge_guidance is None
    assert trace(selection)["km_rules_matched"] == []


# --- the R7 acceptance test ----------------------------------------------


def test_worked_example_reproduces_exactly(tables: RuleTables) -> None:
    """Tech Spec §10 / Architecture worked example — the R7 acceptance test.

    Student: "I got x = 7 for this homework problem, is it right?"
    -> intent check_answer + homework, evaluation incorrect
    -> S06 Feedback + S04 Hint, no direct answer, no full solution, attempt required.
    """
    selection = select_strategy(
        IntentResult(
            intent=Intent.CHECK_ANSWER,
            learner_state=LearnerState.NORMAL,
            special_handling=SpecialHandlingResult(detected=True, type=SpecialHandling.HOMEWORK),
            confidence=0.92,
        ),
        StudentLevel.BEGINNER,
        TOPIC,
        None,
        Evaluation.INCORRECT,
        True,
        tables,
    )

    assert selection.primary_strategy.strategy_id is StrategyId.FEEDBACK
    assert selection.primary_strategy.strategy_name == "feedback"
    assert selection.evaluation is Evaluation.INCORRECT
    assert len(selection.supporting_strategies) == 1
    assert selection.supporting_strategies[0].strategy_id is StrategyId.HINT_SCAFFOLD
    assert selection.supporting_strategies[0].strategy_name == "hint_scaffold"
    assert selection.constraints.direct_answer_allowed is False
    assert selection.constraints.full_solution_allowed_now is False
    assert selection.constraints.student_attempt_required is True
    assert selection.knowledge_guidance is None
    assert selection.topic == TOPIC
    assert selection.student_level is StudentLevel.BEGINNER

    decision = trace(selection)
    assert decision["selected_trigger_row"] == "TM12"
    assert decision["policy_row"] == "TP02"
    assert decision["policy_rows_applied"] == ["TP02", "TP01"]
    assert decision["table_versions"] == tables.version_map

    assert selection.model_dump(mode="json")["primary_strategy"] == {
        "strategy_id": "S06",
        "strategy_name": "feedback",
    }


def test_worked_example_is_stable_across_runs(tables: RuleTables) -> None:
    """No LLM, no clock, no dict ordering: the same input gives the same output."""

    def once() -> str:
        selection = run(
            tables,
            Intent.CHECK_ANSWER,
            special_handling=SpecialHandling.HOMEWORK,
            evaluation=Evaluation.INCORRECT,
        )
        return selection.model_dump_json()

    assert once() == once()


# --- the trigger contest, structured for the inspector --------------------
#
# "Which rule chose this strategy, and what beat what" is the question the glass
# box exists to answer. It is a contest between rows, so the trace carries the
# contest rather than just its winner.


def test_the_winner_is_marked_and_carries_no_loss(tables: RuleTables) -> None:
    selection = run(tables, Intent.CHECK_ANSWER, special_handling=SpecialHandling.HOMEWORK)
    rows = trace(selection)["matched_trigger_rows"]
    winners = [row for row in rows if row["selected"]]
    assert len(winners) == 1
    assert winners[0]["row_id"] == trace(selection)["selected_trigger_row"]
    assert winners[0]["lost_to"] is None
    assert winners[0]["lost_on"] is None


def test_every_loser_names_the_row_that_beat_it(tables: RuleTables) -> None:
    selection = run(tables, Intent.CHECK_ANSWER, special_handling=SpecialHandling.HOMEWORK)
    rows = trace(selection)["matched_trigger_rows"]
    winner = next(row for row in rows if row["selected"])
    losers = [row for row in rows if not row["selected"]]
    assert losers, "expected a contest, not a single candidate"
    for loser in losers:
        assert loser["lost_to"] == winner["row_id"]
        assert loser["lost_on"] in {"priority", "specificity", "row_id"}


def test_the_tie_break_dimension_is_named(tables: RuleTables) -> None:
    """A reviewer needs to know whether a chosen priority or alphabetical order decided it."""
    by_priority = trace(
        run(tables, Intent.CHECK_ANSWER, special_handling=SpecialHandling.HOMEWORK)
    )["matched_trigger_rows"]
    assert next(r for r in by_priority if r["row_id"] == "TM16")["lost_on"] == "priority"

    # TM01 and TM02 differ only in specificity once their priorities are levelled.
    rows = {row.id: row for row in tables.trigger_rows}
    levelled = dataclasses.replace(rows["TM02"], priority=rows["TM01"].priority)
    patched = dataclasses.replace(
        tables,
        trigger_rows=tuple(
            sorted(
                (levelled if row.id == "TM02" else row for row in tables.trigger_rows),
                key=lambda row: row.rank,
            )
        ),
    )
    by_specificity = trace(run(patched, Intent.EXPLAIN))["matched_trigger_rows"]
    assert next(r for r in by_specificity if r["row_id"] == "TM02")["lost_on"] == "specificity"


def test_the_key_is_a_mapping_not_a_positional_list(tables: RuleTables) -> None:
    """A four-element list encodes each field's meaning in its index."""
    row = trace(run(tables, Intent.EXPLAIN))["matched_trigger_rows"][0]
    assert set(row["key"]) == {"intent", "learner_state", "student_level", "special_handling"}
    assert row["key"]["intent"] == "explain"


def test_each_entry_carries_what_the_row_selected(tables: RuleTables) -> None:
    """So the inspector can show what the losing rows would have done instead."""
    rows = trace(run(tables, Intent.CHECK_ANSWER, special_handling=SpecialHandling.HOMEWORK))[
        "matched_trigger_rows"
    ]
    loser = next(row for row in rows if not row["selected"])
    assert loser["primary"] == "S04"
    assert loser["supporting"] == ["S03", "S07"]
    assert loser["source"] == "extension"


def test_entries_are_json_safe(tables: RuleTables) -> None:
    """They ride to the browser inside the trace and are persisted to disk."""
    import json

    rows = trace(run(tables, Intent.SOLVE, learner_state=LearnerState.CONFUSED))[
        "matched_trigger_rows"
    ]
    assert json.loads(json.dumps(rows)) == rows


def test_an_unmatched_lookup_produces_an_empty_contest(tables: RuleTables) -> None:
    empty = dataclasses.replace(tables, trigger_rows=())
    selection = run(empty, Intent.EXPLAIN)
    assert trace(selection)["matched_trigger_rows"] == []
    assert trace(selection)["trigger_matched"] is False


# --- the KM contest -------------------------------------------------------


def test_the_o2_tie_break_travels_as_structure(tables: RuleTables) -> None:
    """KM02 beats KM03 and both are logged — §3.5's named decision.

    The point of `lost_on` here: "KM02 beat KM03" does not say whether that was
    a deliberate priority or an accident of ordering. It was deliberate, and the
    trace says so, so a future table edit that flattened the priorities would
    show up as `rule_id` rather than passing quietly.
    """
    selection = run(
        tables,
        Intent.CLARIFY,
        learner_state=LearnerState.CONFUSED,
        knowledge_context=knowledge(
            prerequisites=[RelatedTopic(topic="Inverse Operations", distance=1)],
            related_topics=[RelatedTopic(topic="Equation Balance", distance=1)],
        ),
    )
    matches = {m["rule_id"]: m for m in trace(selection)["km_matches"]}
    assert matches["KM02"]["applied"] is True
    assert matches["KM02"]["lost_to"] is None
    assert matches["KM03"]["applied"] is False
    assert matches["KM03"]["lost_to"] == "KM02"
    assert matches["KM03"]["lost_on"] == "priority"


def test_each_km_entry_names_the_requirement_it_matched_on(tables: RuleTables) -> None:
    """Why the rule fired, not just that it did."""
    selection = run(
        tables,
        Intent.EXPLAIN,
        learner_state=LearnerState.CONFUSED,
        knowledge_context=knowledge(
            prerequisites=[RelatedTopic(topic="Inverse Operations", distance=1)]
        ),
    )
    match = trace(selection)["km_matches"][0]
    assert match["rule_id"] == "KM02"
    assert match["requirement"] == "immediate_prerequisite"
    assert match["knowledge_action"] == "check_or_scaffold_prerequisite"


def test_km_rules_matched_is_projected_from_the_structured_entries(
    tables: RuleTables,
) -> None:
    """Two lists built independently are two lists that can disagree."""
    for context in (
        None,
        knowledge(),
        knowledge(
            prerequisites=[RelatedTopic(topic="Inverse Operations", distance=1)],
            related_topics=[RelatedTopic(topic="Equation Balance", distance=1)],
        ),
    ):
        for state in LearnerState:
            decision = trace(
                run(tables, Intent.CLARIFY, learner_state=state, knowledge_context=context)
            )
            assert decision["km_rules_matched"] == [m["rule_id"] for m in decision["km_matches"]]


def test_no_km_match_gives_an_empty_contest(tables: RuleTables) -> None:
    decision = trace(run(tables, Intent.EXPLAIN, knowledge_context=None))
    assert decision["km_matches"] == []
    assert decision["km_rules_matched"] == []
    assert decision["km_rule_applied"] is None


def test_km_entries_are_json_safe(tables: RuleTables) -> None:
    import json

    decision = trace(
        run(
            tables,
            Intent.CLARIFY,
            learner_state=LearnerState.CONFUSED,
            knowledge_context=knowledge(
                prerequisites=[RelatedTopic(topic="Inverse Operations", distance=1)],
                related_topics=[RelatedTopic(topic="Equation Balance", distance=1)],
            ),
        )
    )
    assert json.loads(json.dumps(decision["km_matches"])) == decision["km_matches"]
