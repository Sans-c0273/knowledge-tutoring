"""Rule-table loading and validation (R5, evaluation suite E2 — no LLM).

Two things are under test: that the shipped tables in `content/rules/` are
internally consistent, and that a *broken* table is a loud build error rather
than a quiet runtime surprise. The second half copies the real tables to a temp
directory and breaks one thing at a time.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from socratic_tutor.models.enums import (
    STRATEGY_NAMES,
    KnowledgeAction,
    Relation,
    StrategyId,
)
from socratic_tutor.pedagogy.tables import (
    ANY,
    DEFAULT_RULES_DIR,
    TABLE_FILES,
    RuleTableError,
    RuleTables,
    load_tables,
)


@pytest.fixture(scope="module")
def tables() -> RuleTables:
    return load_tables()


@pytest.fixture
def rules_dir(tmp_path: Path) -> Path:
    """A writable copy of the shipped rule tables."""
    target = tmp_path / "rules"
    shutil.copytree(DEFAULT_RULES_DIR, target)
    return target


def patch_table(rules_dir: Path, table: str, mutate) -> None:
    """Load one table, apply `mutate` to the parsed data, write it back."""
    path = rules_dir / TABLE_FILES[table]
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def problems_from(rules_dir: Path) -> list[str]:
    with pytest.raises(RuleTableError) as excinfo:
        load_tables(rules_dir)
    return excinfo.value.problems


# --- the shipped tables ---------------------------------------------------


def test_all_five_tables_load_without_warnings(tables: RuleTables) -> None:
    assert set(tables.version_map) == set(TABLE_FILES)
    assert tables.warnings == ()


def test_every_table_carries_a_version_for_the_turn_trace(tables: RuleTables) -> None:
    """`TurnTrace` records table versions so a decision can be replayed (§7)."""
    for table, version in tables.version_map.items():
        assert version, f"{table} has no version"


def test_strategy_library_holds_s01_to_s08_with_spec_text(tables: RuleTables) -> None:
    assert set(tables.strategies) == set(StrategyId)
    for strategy_id, strategy in tables.strategies.items():
        assert strategy.strategy_name == STRATEGY_NAMES[strategy_id]
        assert strategy.name
        assert strategy.required_behavior
        assert strategy.typical_next_action
    assert tables.strategy(StrategyId.HINT_SCAFFOLD).required_behavior == (
        "Give the smallest useful clue, NOT the answer"
    )
    assert tables.strategy(StrategyId.HINT_SCAFFOLD).typical_next_action == (
        "Wait for student attempt"
    )


def test_trigger_matrix_rows_are_unique_and_reference_known_strategies(
    tables: RuleTables,
) -> None:
    ids = [row.id for row in tables.trigger_rows]
    assert len(ids) == len(set(ids))
    for row in (*tables.trigger_rows, tables.trigger_fallback):
        assert row.primary in tables.strategies
        assert all(s in tables.strategies for s in row.supporting)
        assert row.primary not in row.supporting


def test_trigger_matrix_covers_every_seed_row_of_the_spec(tables: RuleTables) -> None:
    """The §3.1 seed rows must all survive any later table edit."""
    seed = {row.id: row for row in tables.trigger_rows if row.source == "spec"}
    expected = {
        "TM01": ("explain", "normal", "beginner", "none", "S01", ["S02", "S07"], 50),
        "TM03": ("explain", "confused", "beginner", "none", "S05", ["S02", "S07"], 70),
        "TM07": ("solve", "normal", ANY, "none", "S03", ["S07"], 60),
        "TM08": ("solve", "confused", ANY, "none", "S04", ["S03", "S07"], 75),
        "TM09": ("hint", ANY, ANY, ANY, "S04", ["S07"], 65),
        "TM10": ("check_answer", "normal", ANY, "none", "S06", ["S07"], 70),
        "TM12": ("check_answer", ANY, ANY, "homework", "S06", ["S04"], 60),
        "TM13": ("practice_quiz", ANY, ANY, "none", "S08", ["S06"], 60),
        "TM14": ("summarize_review", "normal", ANY, "none", "S01", ["S07"], 50),
        # §3.1 defers this row to Teaching Policy row 3 rather than naming a
        # strategy. Ruled S07 alone: during an assessment the tutor stops being
        # a help channel, and a hint toward a graded answer is still assistance
        # toward that answer.
        "TM17": (ANY, ANY, ANY, "assessment", "S07", [], 90),
    }
    assert set(seed) == set(expected)
    for row_id, (intent, state, level, handling, primary, supporting, priority) in expected.items():
        row = seed[row_id]
        assert row.key == (intent, state, level, handling)
        assert row.primary.value == primary
        assert [s.value for s in row.supporting] == supporting
        assert row.priority == priority


def test_specificity_orders_exact_rows_above_wildcard_rows(tables: RuleTables) -> None:
    rows = {row.id: row for row in tables.trigger_rows}
    assert rows["TM01"].specificity == 4
    assert rows["TM02"].specificity == 3
    assert rows["TM17"].specificity == 1
    assert rows["TM01"].rank < rows["TM02"].rank


def test_combination_rules_cap_is_one_primary_and_two_supporting(
    tables: RuleTables,
) -> None:
    assert tables.combinations.max_primary == 1
    assert tables.combinations.max_supporting == 2


def test_combination_rules_declare_the_four_spec_sets(tables: RuleTables) -> None:
    spec_sets = {
        StrategyId.DIRECT_ANSWER_EXPLANATION: {
            StrategyId.EXAMPLE_ANALOGY,
            StrategyId.CHECK_UNDERSTANDING,
        },
        StrategyId.HINT_SCAFFOLD: {
            StrategyId.STEP_BY_STEP_GUIDANCE,
            StrategyId.CHECK_UNDERSTANDING,
        },
        StrategyId.FEEDBACK: {
            StrategyId.HINT_SCAFFOLD,
            StrategyId.RE_EXPLAIN_DIFFERENTLY,
            StrategyId.CHECK_UNDERSTANDING,
        },
        StrategyId.PRACTICE_QUIZ: {StrategyId.FEEDBACK, StrategyId.CHECK_UNDERSTANDING},
    }
    for primary, supporting in spec_sets.items():
        combination = tables.combinations.by_primary[primary]
        assert combination.source == "spec"
        assert set(combination.supporting) == supporting
        assert combination.sequence


def test_hint_first_beats_full_answer_is_configured(tables: RuleTables) -> None:
    assert tables.combinations.hint_first_beats_full_answer is True
    assert tables.combinations.hint_strategies == frozenset({StrategyId.HINT_SCAFFOLD})
    assert tables.combinations.full_answer_strategies == frozenset(
        {StrategyId.DIRECT_ANSWER_EXPLANATION}
    )
    assert not (tables.combinations.hint_strategies & tables.combinations.full_answer_strategies)


def test_every_trigger_row_is_a_legal_combination(tables: RuleTables) -> None:
    """A trigger row whose supporting set the rules forbid would silently lose it."""
    for row in (*tables.trigger_rows, tables.trigger_fallback):
        allowed = tables.combinations.allowed_supporting(row.primary)
        assert row.primary in tables.combinations.by_primary, row.id
        assert set(row.supporting) <= set(allowed), row.id
        assert len(row.supporting) <= tables.combinations.max_supporting, row.id


def test_teaching_policy_contexts_are_exhaustive(tables: RuleTables) -> None:
    contexts = [row.context for row in tables.policy_rows]
    assert contexts == [
        "normal_learning",
        "homework",
        "assessment",
        "insufficient_rag_evidence",
    ]


def test_teaching_policy_rows_transcribe_the_spec_table(tables: RuleTables) -> None:
    normal = tables.policy_row("normal_learning")
    assert (normal.direct_answer_allowed, normal.student_attempt_required) == (True, False)

    homework = tables.policy_row("homework")
    assert homework.direct_answer_allowed is False
    assert homework.full_solution_allowed_now is False
    assert homework.student_attempt_required is True
    assert homework.hint_first is True

    assessment = tables.policy_row("assessment")
    assert assessment.direct_answer_allowed is False
    assert assessment.student_attempt_required is True
    assert assessment.check == "Teacher-configurable"

    no_evidence = tables.policy_row("insufficient_rag_evidence")
    assert no_evidence.direct_answer_allowed is False
    assert no_evidence.factual_claims_require_evidence is True
    # "—" in the spec table: the row is silent on attempts and inherits.
    assert no_evidence.student_attempt_required is None


def test_assessment_ships_with_hints_switched_off(tables: RuleTables) -> None:
    """The teacher-configurable switch exists, and its shipped default is strict."""
    assessment = tables.policy_row("assessment")
    assert assessment.allow_hints is False
    assert assessment.hint_first is False
    assert assessment.allow_check is True
    assert assessment.prohibited_strategies == frozenset(
        {
            StrategyId.DIRECT_ANSWER_EXPLANATION,
            StrategyId.EXAMPLE_ANALOGY,
            StrategyId.STEP_BY_STEP_GUIDANCE,
        }
    )
    assert assessment.veto_primary is StrategyId.CHECK_UNDERSTANDING


def test_every_other_context_still_permits_hints(tables: RuleTables) -> None:
    for context in ("normal_learning", "homework", "insufficient_rag_evidence"):
        assert tables.policy_row(context).allow_hints is True


def test_check_strategies_emit_no_clue(tables: RuleTables) -> None:
    combinations = tables.combinations
    assert combinations.check_strategies == frozenset({StrategyId.CHECK_UNDERSTANDING})
    assert not (
        combinations.check_strategies
        & (combinations.hint_strategies | combinations.full_answer_strategies)
    )


def test_veto_primary_inside_its_own_prohibited_set_is_a_build_error(
    rules_dir: Path,
) -> None:
    """A substitution the same row would then veto is an infinite regress."""

    def mutate(data: dict[str, Any]) -> None:
        for row in data["contexts"]:
            if row["context"] == "assessment":
                row["prohibited_strategies"] = ["S01", "S02", "S03", "S07"]

    patch_table(rules_dir, "teaching_policy", mutate)
    problems = problems_from(rules_dir)
    assert any("veto_primary" in problem and "S07" in problem for problem in problems)


def test_km_rules_km01_to_km03_active_km04_km05_disabled(tables: RuleTables) -> None:
    enabled = {rule.id: rule for rule in tables.km_rules if rule.enabled}
    disabled = {rule.id for rule in tables.km_rules if not rule.enabled}
    assert set(enabled) == {"KM01", "KM02", "KM03"}
    assert disabled == {"KM04", "KM05"}


def test_km02_action_is_check_or_scaffold_never_a_diagnosis(tables: RuleTables) -> None:
    """A prerequisite edge is a hypothesis, not a diagnosis (§3.5 hard rule, A5)."""
    km02 = tables.km_rule("KM02")
    assert km02.knowledge_action is KnowledgeAction.CHECK_OR_SCAFFOLD_PREREQUISITE
    assert km02.knowledge_action is not KnowledgeAction.USE_PREREQUISITE_AS_SCAFFOLD
    assert km02.relation is Relation.PREREQUISITE_OF
    assert km02.max_distance == 1


def test_km_rules_use_the_closed_relation_vocabulary(tables: RuleTables) -> None:
    for rule in tables.km_rules:
        assert rule.relation is None or isinstance(rule.relation, Relation)
        assert rule.direction in ("forward", "inverse")


def test_unknown_km_relation_is_a_build_error(rules_dir: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        for rule in data["rules"]:
            if rule["id"] == "KM02":
                rule["relation"] = "leads_to"

    patch_table(rules_dir, "km_rules", mutate)
    assert any("leads_to" in problem for problem in problems_from(rules_dir))


def test_km02_outranks_km03_for_the_o2_tie_break(tables: RuleTables) -> None:
    assert tables.km_rule("KM02").priority > tables.km_rule("KM03").priority


# --- fail-fast validation -------------------------------------------------


def test_missing_table_file_is_a_build_error(rules_dir: Path) -> None:
    (rules_dir / TABLE_FILES["km_rules"]).unlink()
    problems = problems_from(rules_dir)
    assert any("km_rules" in problem and "missing file" in problem for problem in problems)


def test_missing_version_field_is_a_build_error(rules_dir: Path) -> None:
    patch_table(rules_dir, "strategy_library", lambda data: data.pop("version"))
    assert any("`version`" in problem for problem in problems_from(rules_dir))


def test_unknown_strategy_in_trigger_matrix_is_a_build_error(rules_dir: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["rows"][0]["primary"] = "S99"

    patch_table(rules_dir, "trigger_matrix", mutate)
    assert any("S99" in problem for problem in problems_from(rules_dir))


def test_invalid_enum_value_in_trigger_matrix_is_a_build_error(rules_dir: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["rows"][0]["learner_state"] = "mildly_baffled"

    patch_table(rules_dir, "trigger_matrix", mutate)
    problems = problems_from(rules_dir)
    assert any("mildly_baffled" in problem and "normal" in problem for problem in problems)


def test_supporting_strategy_outside_the_allowed_combination_is_a_build_error(
    rules_dir: Path,
) -> None:
    def mutate(data: dict[str, Any]) -> None:
        # TM01 is primary S01, whose allowed supporting set is {S02, S07}.
        data["rows"][0]["supporting"] = ["S03"]

    patch_table(rules_dir, "trigger_matrix", mutate)
    problems = problems_from(rules_dir)
    assert any("TM01" in problem and "S03" in problem for problem in problems)


def test_hint_paired_with_full_answer_in_one_row_is_a_build_error(rules_dir: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["rows"][0]["supporting"] = ["S02", "S07"]
        data["rows"][0]["primary"] = "S01"
        data["rows"].append(
            {
                "id": "TM99",
                "intent": "explain",
                "learner_state": "normal",
                "student_level": "advanced",
                "special_handling": "none",
                "primary": "S01",
                "supporting": ["S04"],
                "priority": 10,
            }
        )

    patch_table(rules_dir, "trigger_matrix", mutate)
    problems = problems_from(rules_dir)
    assert any("TM99" in problem and "hint-first" in problem for problem in problems)


def test_too_many_supporting_strategies_is_a_build_error(rules_dir: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["rows"].append(
            {
                "id": "TM98",
                "intent": "check_answer",
                "learner_state": "normal",
                "student_level": "advanced",
                "special_handling": "none",
                "primary": "S06",
                "supporting": ["S04", "S05", "S07"],
                "priority": 10,
            }
        )

    patch_table(rules_dir, "trigger_matrix", mutate)
    problems = problems_from(rules_dir)
    assert any("TM98" in problem and "max_supporting" in problem for problem in problems)


def test_missing_policy_context_is_a_build_error(rules_dir: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["contexts"] = [row for row in data["contexts"] if row["context"] != "assessment"]

    patch_table(rules_dir, "teaching_policy", mutate)
    problems = problems_from(rules_dir)
    assert any("assessment" in problem and "missing context" in problem for problem in problems)


def test_disabling_a_required_km_rule_is_a_build_error(rules_dir: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        for rule in data["rules"]:
            if rule["id"] == "KM02":
                rule["enabled"] = False

    patch_table(rules_dir, "km_rules", mutate)
    assert any("KM02" in problem for problem in problems_from(rules_dir))


def test_enabled_km_rule_without_an_action_is_a_build_error(rules_dir: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        for rule in data["rules"]:
            if rule["id"] == "KM03":
                rule["knowledge_action"] = None

    patch_table(rules_dir, "km_rules", mutate)
    assert any(
        "KM03" in problem and "knowledge_action" in problem for problem in problems_from(rules_dir)
    )


def test_strategy_name_drifting_from_the_enum_is_a_build_error(rules_dir: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["strategies"][0]["strategy_name"] = "direct_answer"

    patch_table(rules_dir, "strategy_library", mutate)
    problems = problems_from(rules_dir)
    assert any("STRATEGY_NAMES" in problem for problem in problems)


def test_all_problems_are_reported_in_one_raise(rules_dir: Path) -> None:
    """The loader is a build gate: it lists the whole repair list, not the first item."""
    patch_table(rules_dir, "strategy_library", lambda data: data.pop("description"))

    def mutate(data: dict[str, Any]) -> None:
        data["rows"][0]["primary"] = "S99"
        data["rows"][1]["priority"] = "high"

    patch_table(rules_dir, "trigger_matrix", mutate)
    problems = problems_from(rules_dir)
    assert len(problems) >= 3


def test_duplicate_key_and_priority_warns_but_resolves_deterministically(
    rules_dir: Path,
) -> None:
    def mutate(data: dict[str, Any]) -> None:
        clone = dict(data["rows"][0])
        clone["id"] = "TM00"
        data["rows"].append(clone)

    patch_table(rules_dir, "trigger_matrix", mutate)
    loaded = load_tables(rules_dir)
    assert any("TM00" in warning and "TM01" in warning for warning in loaded.warnings)
    # Lowest id wins, on every machine and every run.
    assert loaded.trigger_rows[0].rank < loaded.trigger_rows[1].rank


# --- audit provenance ------------------------------------------------------


def test_every_table_is_hashed(tables: RuleTables) -> None:
    assert set(tables.content_hashes) == set(TABLE_FILES)
    assert all(len(h) == 12 for h in tables.content_hashes.values())


def test_audit_versions_pair_the_claim_with_the_evidence(tables: RuleTables) -> None:
    """ "Which policy was in force for this turn" must be answerable from the record."""
    for name, value in tables.audit_versions.items():
        version, _, digest = value.partition("+")
        assert version == tables.version_map[name]
        assert digest == tables.content_hashes[name]


def test_an_edit_without_a_version_bump_still_changes_the_hash(rules_dir: Path) -> None:
    """The reason the hash exists: a declared version is a claim, not evidence."""
    before = load_tables(rules_dir)

    def mutate(data: dict[str, Any]) -> None:
        for row in data["rows"]:
            if row["id"] == "TM01":
                row["priority"] = 51  # a real policy change

    patch_table(rules_dir, "trigger_matrix", mutate)
    after = load_tables(rules_dir)

    assert after.version_map["trigger_matrix"] == before.version_map["trigger_matrix"]
    assert after.content_hashes["trigger_matrix"] != before.content_hashes["trigger_matrix"]
    assert after.audit_versions["trigger_matrix"] != before.audit_versions["trigger_matrix"]


def test_the_baseline_context_is_declared_and_real(tables: RuleTables) -> None:
    assert tables.baseline_context == "normal_learning"
    assert tables.baseline_policy_row().context == "normal_learning"


def test_an_unknown_baseline_context_is_a_build_error(rules_dir: Path) -> None:
    patch_table(
        rules_dir, "teaching_policy", lambda data: data.update(baseline_context="strict_mode")
    )
    problems = problems_from(rules_dir)
    assert any("baseline_context" in problem and "strict_mode" in problem for problem in problems)
