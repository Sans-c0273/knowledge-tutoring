"""Rule-table loading and validation (Tech Spec §3, requirement R5).

The five tables under `content/rules/` are versioned data, not code
(Architecture A1: teaching decisions live in rule tables so they are auditable,
testable, and cannot be jailbroken or drift over a long conversation).

Loading is **fail-fast**: every problem found in any table is collected and
raised together as a single `RuleTableError` naming the file, the row and what
to do about it. A bad table is a build error, never a runtime surprise. Problems
that have a deterministic resolution (ambiguous priorities) are collected as
warnings instead and surfaced on `RuleTables.warnings`.

`RuleTables.version_map` is what `TurnTrace` records so a past teaching decision
can be replayed against the exact tables that produced it (Tech Spec §7).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import ValidationError

from socratic_tutor.models.enums import (
    STRATEGY_NAMES,
    Intent,
    KnowledgeAction,
    Language,
    LearnerState,
    Relation,
    SpecialHandling,
    StrategyId,
    StudentLevel,
)
from socratic_tutor.models.plan import GlobalRules

#: Wildcard token in a Trigger Matrix key field.
ANY = "any"

#: `content/rules/` resolved from this file: pedagogy -> socratic_tutor -> src -> repo root.
DEFAULT_RULES_DIR = Path(__file__).resolve().parents[3] / "content" / "rules"

TABLE_FILES: dict[str, str] = {
    "trigger_matrix": "trigger_matrix.yaml",
    "strategy_library": "strategy_library.yaml",
    "combination_rules": "combination_rules.yaml",
    "teaching_policy": "teaching_policy.yaml",
    "km_rules": "km_rules.yaml",
    "response_templates": "response_templates.yaml",
    "global_rules": "global_rules.yaml",
}

#: Block kinds that state or contain the answer, dropped when it is withheld.
SOLUTION_KINDS = frozenset({"solution"})
#: Block kinds that walk the student through the solution path.
STEP_KINDS = frozenset({"steps"})

#: The four Teaching Policy contexts of Tech Spec §3.4. Exhaustiveness is enforced.
REQUIRED_POLICY_CONTEXTS = (
    "normal_learning",
    "homework",
    "assessment",
    "insufficient_rag_evidence",
)

#: KM rules that must be present and enabled for the POC (Tech Spec §3.5).
REQUIRED_ACTIVE_KM_RULES = ("KM01", "KM02", "KM03")


class RuleTableError(ValueError):
    """A rule table is malformed, incomplete, or internally inconsistent.

    Carries every problem found across all five tables so one run of the loader
    reports the whole repair list rather than the first failure.
    """

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        joined = "\n  - ".join(self.problems)
        super().__init__(f"{len(self.problems)} rule-table problem(s):\n  - {joined}")


@dataclass(frozen=True)
class Strategy:
    """One row of the Strategy Library (Tech Spec §3.2)."""

    id: StrategyId
    name: str
    strategy_name: str
    required_behavior: str
    typical_next_action: str


@dataclass(frozen=True)
class TriggerRow:
    """One Trigger Matrix row (Tech Spec §3.1).

    Key fields hold either a spec enum value or `ANY`. `specificity` counts the
    non-wildcard key fields and breaks priority ties so an exact row beats a
    wildcard row.
    """

    id: str
    intent: str
    learner_state: str
    student_level: str
    special_handling: str
    primary: StrategyId
    supporting: tuple[StrategyId, ...]
    priority: int
    source: str = "spec"
    note: str = ""

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.intent, self.learner_state, self.student_level, self.special_handling)

    @property
    def specificity(self) -> int:
        return sum(1 for value in self.key if value != ANY)

    def matches(
        self,
        intent: Intent,
        learner_state: LearnerState,
        student_level: StudentLevel,
        special_handling: SpecialHandling,
    ) -> bool:
        """True when every key field is `ANY` or equals the looked-up value."""
        lookup = (
            intent.value,
            learner_state.value,
            student_level.value,
            special_handling.value,
        )
        return all(row == ANY or row == value for row, value in zip(self.key, lookup, strict=True))

    @property
    def rank(self) -> tuple[int, int, str]:
        """Sort key: highest priority, then most specific, then id (deterministic)."""
        return (-self.priority, -self.specificity, self.id)


@dataclass(frozen=True)
class Combination:
    """One allowed primary -> supporting set (Tech Spec §3.3)."""

    primary: StrategyId
    supporting: frozenset[StrategyId]
    sequence: tuple[str, ...]
    source: str = "spec"
    note: str = ""


@dataclass(frozen=True)
class CombinationRules:
    """The whole Combination Rules table (Tech Spec §3.3)."""

    max_primary: int
    max_supporting: int
    hint_first_beats_full_answer: bool
    hint_strategies: frozenset[StrategyId]
    full_answer_strategies: frozenset[StrategyId]
    check_strategies: frozenset[StrategyId]
    cannot_evaluate_primary: StrategyId
    by_primary: dict[StrategyId, Combination]

    def allowed_supporting(self, primary: StrategyId) -> frozenset[StrategyId]:
        combination = self.by_primary.get(primary)
        return combination.supporting if combination else frozenset()


@dataclass(frozen=True)
class PolicyRow:
    """One Teaching Policy context row (Tech Spec §3.4).

    `student_attempt_required` is `None` when the row is silent on attempts
    ("—" in the spec table); it then inherits from the other applicable rows.

    `allow_hints` and `allow_check` are the teacher-configurable switches §3.4
    calls for. `prohibited_strategies` names the clue-emitting strategies this
    context forbids outright, and `veto_primary` is what the selector substitutes
    when the chosen primary is one of them.
    """

    id: str
    context: str
    precedence: int
    applies_when: dict[str, Any]
    direct_answer: str
    attempt: str
    check: str
    direct_answer_allowed: bool
    full_solution_allowed_now: bool
    student_attempt_required: bool | None
    hint_first: bool
    factual_claims_require_evidence: bool
    allow_hints: bool = True
    allow_check: bool = True
    prohibited_strategies: frozenset[StrategyId] = frozenset()
    veto_primary: StrategyId | None = None
    note: str = ""

    def applies(self, special_handling: SpecialHandling, *, rag_evidence_present: bool) -> bool:
        """True when this row's `applies_when` conditions all hold."""
        wanted_handling = self.applies_when.get("special_handling")
        if wanted_handling is not None and wanted_handling != special_handling.value:
            return False
        wanted_evidence = self.applies_when.get("rag_evidence_present")
        if wanted_evidence is None:
            return True
        return bool(wanted_evidence) is rag_evidence_present


@dataclass(frozen=True)
class KmRule:
    """One KL Map -> Strategy rule (Tech Spec §3.5)."""

    id: str
    enabled: bool
    priority: int
    learner_state: str | None
    intent: str | None
    requires: str | None
    kl_map_action: str
    strategy_effect: str
    knowledge_action: KnowledgeAction | None
    target: str
    relation: Relation | None = None
    direction: Literal["forward", "inverse"] = "forward"
    max_distance: int | None = None
    note: str = ""


@dataclass(frozen=True)
class Block:
    """One structural block a response template can emit (Tech Spec §4.1)."""

    name: str
    kind: str

    @property
    def is_question(self) -> bool:
        return self.kind == "question"

    @property
    def is_hint(self) -> bool:
        return self.kind == "hint"

    @property
    def is_solution(self) -> bool:
        return self.kind in SOLUTION_KINDS

    @property
    def is_steps(self) -> bool:
        return self.kind in STEP_KINDS


@dataclass(frozen=True)
class ResponseTemplate:
    """One primary strategy's response template (Tech Spec §4.1)."""

    strategy: StrategyId
    structure: tuple[str, ...]
    max_words: int
    follow_up: str
    #: True, False, or "depends_on_attempt_required" for §4.1's "Depends" (S06).
    wait_for_student: bool | str
    source: str = "spec"
    note: str = ""

    def waits(self, *, student_attempt_required: bool) -> bool:
        """Resolve §4.1's "Depends" against this turn's Teaching Policy."""
        if isinstance(self.wait_for_student, bool):
            return self.wait_for_student
        return student_attempt_required


@dataclass(frozen=True)
class SupportingBlock:
    """What a strategy contributes when it is supporting rather than primary.

    `requires_student_response` marks a block that acts on an answer the student
    has not given yet; §3.3's sequences put those in the following turn.
    """

    strategy: StrategyId
    block: str
    requires_student_response: bool = False


@dataclass(frozen=True)
class ResponseTemplates:
    """The §4.1 template table plus the block vocabulary it draws on."""

    blocks: dict[str, Block]
    templates: dict[StrategyId, ResponseTemplate]
    supporting_blocks: dict[StrategyId, SupportingBlock]

    def block(self, name: str) -> Block:
        return self.blocks[name]

    def template(self, strategy: StrategyId) -> ResponseTemplate:
        return self.templates[strategy]


@dataclass(frozen=True)
class GuardrailConfig:
    """Thresholds and copy for the §6 output guardrail."""

    count_tolerance: float
    max_regenerations: int
    regeneration_note: str
    fallback_text: dict[str, str]

    def fallback_for(self, language: str) -> str:
        """Fallback copy in `language`, falling back to English."""
        return self.fallback_text.get(language) or self.fallback_text.get("en", "")


@dataclass(frozen=True)
class RuleTables:
    """All seven loaded tables plus the version map `TurnTrace` records."""

    trigger_rows: tuple[TriggerRow, ...]
    trigger_fallback: TriggerRow
    strategies: dict[StrategyId, Strategy]
    combinations: CombinationRules
    policy_rows: tuple[PolicyRow, ...]
    km_rules: tuple[KmRule, ...]
    response_templates: ResponseTemplates
    global_rules: GlobalRules
    student_level_wording: dict[str, str]
    guardrail: GuardrailConfig
    version_map: dict[str, str]
    #: SHA-256 of each table file's bytes, truncated. The declared `version:` is
    #: a human's claim about the file; this is what the file actually was.
    content_hashes: dict[str, str] = field(default_factory=dict)
    #: Baseline policy context for a session, before any self-report (§3.4).
    baseline_context: str = "normal_learning"
    warnings: tuple[str, ...] = ()

    @property
    def audit_versions(self) -> dict[str, str]:
        """`table -> "version+hash"`, for `TurnTrace.table_versions`.

        A declared version is a claim, not evidence: edit a table without
        bumping `version:` and every turn afterwards reports a policy it was
        never run under. For a ministry audit the question "which policy was in
        force for this turn" has to be answerable from the record alone, so the
        record carries what the file *was*, not what it said it was.
        """
        return {
            name: f"{version}+{self.content_hashes.get(name, 'nohash')}"
            for name, version in self.version_map.items()
        }

    def strategy(self, strategy_id: StrategyId) -> Strategy:
        return self.strategies[strategy_id]

    def baseline_policy_row(self) -> PolicyRow:
        """The row that applies before the student says anything (§3.4)."""
        return self.policy_row(self.baseline_context)

    def policy_row(self, context: str) -> PolicyRow:
        for row in self.policy_rows:
            if row.context == context:
                return row
        raise KeyError(context)

    def km_rule(self, rule_id: str) -> KmRule:
        for rule in self.km_rules:
            if rule.id == rule_id:
                return rule
        raise KeyError(rule_id)


@dataclass
class _Loader:
    """Collects problems across all five files so one raise reports them all."""

    rules_dir: Path
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: Content hash per table, so the trace can record what a file *was* rather
    #: than what its `version:` field claimed.
    hashes: dict[str, str] = field(default_factory=dict)

    def fail(self, table: str, message: str) -> None:
        self.problems.append(f"{table}: {message}")

    def warn(self, table: str, message: str) -> None:
        self.warnings.append(f"{table}: {message}")

    def read(self, table: str) -> dict[str, Any]:
        path = self.rules_dir / TABLE_FILES[table]
        if not path.is_file():
            self.fail(table, f"missing file {path}; every rule table must be present")
            return {}
        raw_bytes = path.read_bytes()
        self.hashes[table] = hashlib.sha256(raw_bytes).hexdigest()[:12]
        try:
            data = yaml.safe_load(raw_bytes.decode("utf-8"))
        except yaml.YAMLError as exc:
            self.fail(table, f"{path} is not valid YAML: {exc}")
            return {}
        if not isinstance(data, dict):
            self.fail(table, f"{path} must contain a mapping at the top level")
            return {}
        for required in ("version", "description"):
            value = data.get(required)
            if not isinstance(value, str) or not value.strip():
                self.fail(table, f"{path.name} needs a non-empty `{required}` field")
        return data

    def enum_value(
        self, table: str, where: str, field_name: str, raw: Any, enum: type, *, allow_any: bool
    ) -> str:
        """Validate a key field against an enum, permitting the `any` wildcard."""
        if not isinstance(raw, str):
            self.fail(table, f"{where}: `{field_name}` must be a string, got {raw!r}")
            return ANY if allow_any else ""
        if allow_any and raw == ANY:
            return ANY
        allowed = [member.value for member in enum]
        if raw not in allowed:
            options = ", ".join(allowed + ([ANY] if allow_any else []))
            self.fail(table, f"{where}: `{field_name}={raw!r}` is not one of: {options}")
            return ANY if allow_any else ""
        return raw

    def strategy_id(self, table: str, where: str, raw: Any) -> StrategyId | None:
        try:
            return StrategyId(raw)
        except ValueError:
            valid = ", ".join(member.value for member in StrategyId)
            self.fail(table, f"{where}: unknown strategy {raw!r}; the library holds {valid}")
            return None


def _load_strategy_library(loader: _Loader, data: dict[str, Any]) -> dict[StrategyId, Strategy]:
    table = "strategy_library"
    strategies: dict[StrategyId, Strategy] = {}
    rows = data.get("strategies")
    if not isinstance(rows, list) or not rows:
        loader.fail(table, "`strategies` must be a non-empty list")
        return strategies

    for index, row in enumerate(rows):
        where = f"strategies[{index}]"
        if not isinstance(row, dict):
            loader.fail(table, f"{where}: each entry must be a mapping")
            continue
        strategy_id = loader.strategy_id(table, where, row.get("id"))
        if strategy_id is None:
            continue
        if strategy_id in strategies:
            loader.fail(table, f"{where}: duplicate strategy id {strategy_id.value}")
            continue
        wire_name = row.get("strategy_name", "")
        expected = STRATEGY_NAMES[strategy_id]
        if wire_name != expected:
            loader.fail(
                table,
                f"{where}: `strategy_name={wire_name!r}` must equal {expected!r} "
                "(models.enums.STRATEGY_NAMES is the wire contract)",
            )
        for text_field in ("name", "required_behavior", "typical_next_action"):
            if not str(row.get(text_field, "")).strip():
                loader.fail(table, f"{where}: `{text_field}` must be non-empty")
        strategies[strategy_id] = Strategy(
            id=strategy_id,
            name=str(row.get("name", "")),
            strategy_name=str(wire_name),
            required_behavior=str(row.get("required_behavior", "")),
            typical_next_action=str(row.get("typical_next_action", "")),
        )

    missing = [member.value for member in StrategyId if member not in strategies]
    if missing:
        loader.fail(table, f"missing strategies {', '.join(missing)}; S01-S08 must all be defined")
    return strategies


def _parse_supporting(loader: _Loader, table: str, where: str, raw: Any) -> tuple[StrategyId, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        loader.fail(table, f"{where}: `supporting` must be a list")
        return ()
    resolved: list[StrategyId] = []
    for item in raw:
        strategy_id = loader.strategy_id(table, where, item)
        if strategy_id is None:
            continue
        if strategy_id in resolved:
            loader.fail(table, f"{where}: {strategy_id.value} listed twice in `supporting`")
            continue
        resolved.append(strategy_id)
    return tuple(resolved)


def _load_trigger_matrix(
    loader: _Loader, data: dict[str, Any], strategies: dict[StrategyId, Strategy]
) -> tuple[tuple[TriggerRow, ...], TriggerRow | None]:
    table = "trigger_matrix"
    rows_raw = data.get("rows")
    parsed: list[TriggerRow] = []
    if not isinstance(rows_raw, list) or not rows_raw:
        loader.fail(table, "`rows` must be a non-empty list")
        rows_raw = []

    seen_ids: set[str] = set()
    for index, row in enumerate(rows_raw):
        where = f"rows[{index}]"
        if not isinstance(row, dict):
            loader.fail(table, f"{where}: each row must be a mapping")
            continue
        row_id = str(row.get("id", "")).strip()
        if not row_id:
            loader.fail(table, f"{where}: every row needs an `id` (traces reference it)")
            continue
        where = f"row {row_id}"
        if row_id in seen_ids:
            loader.fail(table, f"{where}: duplicate row id")
            continue
        seen_ids.add(row_id)

        primary = loader.strategy_id(table, where, row.get("primary"))
        supporting = _parse_supporting(loader, table, where, row.get("supporting"))
        priority = row.get("priority")
        if not isinstance(priority, int) or isinstance(priority, bool):
            loader.fail(table, f"{where}: `priority` must be an integer, got {priority!r}")
            priority = 0
        if primary is None:
            continue
        if primary in supporting:
            loader.fail(table, f"{where}: {primary.value} is both primary and supporting")
        for strategy_id in (primary, *supporting):
            if strategy_id not in strategies:
                loader.fail(table, f"{where}: {strategy_id.value} is not in the Strategy Library")

        parsed.append(
            TriggerRow(
                id=row_id,
                intent=loader.enum_value(
                    table, where, "intent", row.get("intent"), Intent, allow_any=True
                ),
                learner_state=loader.enum_value(
                    table,
                    where,
                    "learner_state",
                    row.get("learner_state"),
                    LearnerState,
                    allow_any=True,
                ),
                student_level=loader.enum_value(
                    table,
                    where,
                    "student_level",
                    row.get("student_level"),
                    StudentLevel,
                    allow_any=True,
                ),
                special_handling=loader.enum_value(
                    table,
                    where,
                    "special_handling",
                    row.get("special_handling"),
                    SpecialHandling,
                    allow_any=True,
                ),
                primary=primary,
                supporting=supporting,
                priority=priority,
                source=str(row.get("source", "spec")),
                note=str(row.get("note", "")),
            )
        )

    _check_priority_ambiguity(loader, table, parsed)

    fallback_raw = data.get("fallback")
    fallback: TriggerRow | None = None
    if not isinstance(fallback_raw, dict):
        loader.fail(
            table,
            "`fallback` must be a mapping; an unmatched lookup still has to produce a turn",
        )
    else:
        primary = loader.strategy_id(table, "fallback", fallback_raw.get("primary"))
        supporting = _parse_supporting(loader, table, "fallback", fallback_raw.get("supporting"))
        if primary is not None:
            fallback = TriggerRow(
                id=str(fallback_raw.get("id", "TM_FALLBACK")),
                intent=ANY,
                learner_state=ANY,
                student_level=ANY,
                special_handling=ANY,
                primary=primary,
                supporting=supporting,
                priority=-1,
                source="fallback",
                note=str(fallback_raw.get("note", "")),
            )
    return tuple(sorted(parsed, key=lambda row: row.rank)), fallback


def _check_priority_ambiguity(loader: _Loader, table: str, rows: list[TriggerRow]) -> None:
    """Warn when two rows share an identical key AND priority.

    Not an error: `TriggerRow.rank` still orders them deterministically by id, so
    selection is reproducible. It is a warning because it means whoever edited
    the table probably did not mean to leave the choice to alphabetical order.
    """
    buckets: dict[tuple[tuple[str, ...], int], list[str]] = {}
    for row in rows:
        buckets.setdefault((row.key, row.priority), []).append(row.id)
    for (key, priority), ids in sorted(buckets.items()):
        if len(ids) > 1:
            loader.warn(
                table,
                f"rows {', '.join(sorted(ids))} share key {key} at priority {priority}; "
                f"resolved deterministically to {min(ids)} (lowest id) — give them "
                "distinct priorities to make the intent explicit",
            )


def _load_combination_rules(
    loader: _Loader, data: dict[str, Any], strategies: dict[StrategyId, Strategy]
) -> CombinationRules | None:
    table = "combination_rules"
    max_primary = data.get("max_primary", 1)
    max_supporting = data.get("max_supporting", 2)
    for name, value in (("max_primary", max_primary), ("max_supporting", max_supporting)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            loader.fail(table, f"`{name}` must be a non-negative integer, got {value!r}")

    def strategy_set(field_name: str) -> frozenset[StrategyId]:
        raw = data.get(field_name) or []
        if not isinstance(raw, list):
            loader.fail(table, f"`{field_name}` must be a list")
            return frozenset()
        resolved = [loader.strategy_id(table, field_name, item) for item in raw]
        return frozenset(item for item in resolved if item is not None)

    hint_strategies = strategy_set("hint_strategies")
    full_answer_strategies = strategy_set("full_answer_strategies")
    check_strategies = strategy_set("check_strategies")
    overlap = hint_strategies & full_answer_strategies
    if overlap:
        names = ", ".join(sorted(s.value for s in overlap))
        loader.fail(table, f"{names} cannot be both a hint and a full-answer strategy")
    clue_overlap = check_strategies & (hint_strategies | full_answer_strategies)
    if clue_overlap:
        names = ", ".join(sorted(s.value for s in clue_overlap))
        loader.fail(
            table,
            f"{names} cannot be a check strategy and also emit a clue; a context that "
            "prohibits clue-emitting strategies falls back to the check strategies",
        )
    if not check_strategies:
        loader.fail(
            table,
            "`check_strategies` must name at least one strategy; assessment turns have "
            "nothing to fall back to without it",
        )

    cannot_evaluate_primary = loader.strategy_id(
        table, "cannot_evaluate_primary", data.get("cannot_evaluate_primary", "S07")
    )

    combinations_raw = data.get("combinations")
    by_primary: dict[StrategyId, Combination] = {}
    if not isinstance(combinations_raw, list) or not combinations_raw:
        loader.fail(table, "`combinations` must be a non-empty list")
        combinations_raw = []
    for index, row in enumerate(combinations_raw):
        where = f"combinations[{index}]"
        if not isinstance(row, dict):
            loader.fail(table, f"{where}: each entry must be a mapping")
            continue
        primary = loader.strategy_id(table, where, row.get("primary"))
        if primary is None:
            continue
        if primary in by_primary:
            loader.fail(table, f"{where}: duplicate combination for primary {primary.value}")
            continue
        supporting = _parse_supporting(loader, table, where, row.get("supporting"))
        if primary in supporting:
            loader.fail(table, f"{where}: {primary.value} cannot support itself")
        for strategy_id in (primary, *supporting):
            if strategy_id not in strategies:
                loader.fail(table, f"{where}: {strategy_id.value} is not in the Strategy Library")
        sequence_raw = row.get("sequence") or []
        if not isinstance(sequence_raw, list):
            loader.fail(table, f"{where}: `sequence` must be a list of step labels")
            sequence_raw = []
        by_primary[primary] = Combination(
            primary=primary,
            supporting=frozenset(supporting),
            sequence=tuple(str(step) for step in sequence_raw),
            source=str(row.get("source", "spec")),
            note=str(row.get("note", "")),
        )

    if cannot_evaluate_primary is None:
        return None
    return CombinationRules(
        max_primary=int(max_primary) if isinstance(max_primary, int) else 1,
        max_supporting=int(max_supporting) if isinstance(max_supporting, int) else 2,
        hint_first_beats_full_answer=bool(data.get("hint_first_beats_full_answer", True)),
        hint_strategies=hint_strategies,
        full_answer_strategies=full_answer_strategies,
        check_strategies=check_strategies,
        cannot_evaluate_primary=cannot_evaluate_primary,
        by_primary=by_primary,
    )


def _load_teaching_policy(loader: _Loader, data: dict[str, Any]) -> tuple[PolicyRow, ...]:
    table = "teaching_policy"
    baseline = data.get("baseline_context", "normal_learning")
    if not isinstance(baseline, str) or not baseline.strip():
        loader.fail(table, f"`baseline_context` must be a context name, got {baseline!r}")
    rows_raw = data.get("contexts")
    parsed: list[PolicyRow] = []
    if not isinstance(rows_raw, list) or not rows_raw:
        loader.fail(table, "`contexts` must be a non-empty list")
        return ()

    for index, row in enumerate(rows_raw):
        where = f"contexts[{index}]"
        if not isinstance(row, dict):
            loader.fail(table, f"{where}: each context must be a mapping")
            continue
        row_id = str(row.get("id", "")).strip()
        context = str(row.get("context", "")).strip()
        if not row_id or not context:
            loader.fail(table, f"{where}: needs both `id` and `context`")
            continue
        where = f"context {row_id}"
        applies_when = row.get("applies_when")
        if not isinstance(applies_when, dict) or not applies_when:
            loader.fail(table, f"{where}: `applies_when` must be a non-empty mapping")
            applies_when = {}
        handling = applies_when.get("special_handling")
        if handling is not None:
            loader.enum_value(
                table,
                where,
                "applies_when.special_handling",
                handling,
                SpecialHandling,
                allow_any=False,
            )

        booleans: dict[str, bool] = {}
        for field_name in (
            "direct_answer_allowed",
            "full_solution_allowed_now",
            "hint_first",
            "factual_claims_require_evidence",
        ):
            value = row.get(field_name)
            if not isinstance(value, bool):
                loader.fail(table, f"{where}: `{field_name}` must be true or false, got {value!r}")
                value = False
            booleans[field_name] = value

        attempt_required = row.get("student_attempt_required")
        if attempt_required is not None and not isinstance(attempt_required, bool):
            loader.fail(
                table,
                f"{where}: `student_attempt_required` must be true, false or null "
                "(null = the row is silent and inherits)",
            )
            attempt_required = None

        precedence = row.get("precedence")
        if not isinstance(precedence, int) or isinstance(precedence, bool):
            loader.fail(table, f"{where}: `precedence` must be an integer, got {precedence!r}")
            precedence = 0

        # Teacher-configurable switches (§3.4). Both default permissive so that a
        # row written before these existed keeps its old behaviour; the strict
        # reading is stated explicitly on the assessment row.
        switches: dict[str, bool] = {}
        for field_name in ("allow_hints", "allow_check"):
            value = row.get(field_name, True)
            if not isinstance(value, bool):
                loader.fail(table, f"{where}: `{field_name}` must be true or false, got {value!r}")
                value = True
            switches[field_name] = value

        prohibited = frozenset(
            _parse_supporting(loader, table, where, row.get("prohibited_strategies") or [])
        )
        veto_primary: StrategyId | None = None
        veto_raw = row.get("veto_primary")
        if veto_raw is not None:
            veto_primary = loader.strategy_id(table, where, veto_raw)
            if veto_primary is not None and veto_primary in prohibited:
                loader.fail(
                    table,
                    f"{where}: `veto_primary={veto_primary.value}` is also in this row's "
                    "`prohibited_strategies`; the substitution would be vetoed by the row "
                    "that requested it",
                )

        parsed.append(
            PolicyRow(
                id=row_id,
                context=context,
                precedence=precedence,
                applies_when=dict(applies_when),
                direct_answer=str(row.get("direct_answer", "")),
                attempt=str(row.get("attempt", "")),
                check=str(row.get("check", "")),
                student_attempt_required=attempt_required,
                prohibited_strategies=prohibited,
                veto_primary=veto_primary,
                note=str(row.get("note", "")),
                **booleans,
                **switches,
            )
        )

    contexts = [row.context for row in parsed]
    missing = [name for name in REQUIRED_POLICY_CONTEXTS if name not in contexts]
    if missing:
        loader.fail(
            table,
            f"missing context row(s) {', '.join(missing)}; §3.4 has exactly "
            f"{', '.join(REQUIRED_POLICY_CONTEXTS)} and all four must be present",
        )
    duplicates = {name for name in contexts if contexts.count(name) > 1}
    if duplicates:
        loader.fail(table, f"duplicate context row(s): {', '.join(sorted(duplicates))}")
    if isinstance(baseline, str) and baseline not in contexts:
        loader.fail(
            table,
            f"`baseline_context={baseline!r}` is not one of the declared contexts: "
            f"{', '.join(contexts)}",
        )
    return tuple(parsed)


def _load_km_rules(loader: _Loader, data: dict[str, Any]) -> tuple[KmRule, ...]:
    table = "km_rules"
    rows_raw = data.get("rules")
    parsed: list[KmRule] = []
    if not isinstance(rows_raw, list) or not rows_raw:
        loader.fail(table, "`rules` must be a non-empty list")
        return ()

    for index, row in enumerate(rows_raw):
        where = f"rules[{index}]"
        if not isinstance(row, dict):
            loader.fail(table, f"{where}: each rule must be a mapping")
            continue
        rule_id = str(row.get("id", "")).strip()
        if not rule_id:
            loader.fail(table, f"{where}: every KM rule needs an `id`")
            continue
        where = f"rule {rule_id}"
        enabled = row.get("enabled")
        if not isinstance(enabled, bool):
            loader.fail(table, f"{where}: `enabled` must be true or false, got {enabled!r}")
            enabled = False
        priority = row.get("priority")
        if not isinstance(priority, int) or isinstance(priority, bool):
            loader.fail(table, f"{where}: `priority` must be an integer, got {priority!r}")
            priority = 0

        condition = row.get("condition") or {}
        if not isinstance(condition, dict):
            loader.fail(table, f"{where}: `condition` must be a mapping")
            condition = {}
        learner_state = condition.get("learner_state")
        if learner_state is not None:
            learner_state = loader.enum_value(
                table,
                where,
                "condition.learner_state",
                learner_state,
                LearnerState,
                allow_any=False,
            )
        intent = condition.get("intent")
        if intent is not None:
            intent = loader.enum_value(
                table, where, "condition.intent", intent, Intent, allow_any=False
            )

        action_raw = row.get("knowledge_action")
        action: KnowledgeAction | None = None
        if action_raw is not None:
            try:
                action = KnowledgeAction(action_raw)
            except ValueError:
                valid = ", ".join(member.value for member in KnowledgeAction)
                loader.fail(
                    table, f"{where}: `knowledge_action={action_raw!r}` is not one of: {valid}"
                )
        elif enabled:
            loader.fail(table, f"{where}: an enabled rule must declare a `knowledge_action`")

        relation_raw = row.get("relation")
        relation: Relation | None = None
        if relation_raw is not None:
            try:
                relation = Relation(relation_raw)
            except ValueError:
                valid = ", ".join(member.value for member in Relation)
                loader.fail(
                    table,
                    f"{where}: `relation={relation_raw!r}` is not in the closed KL Map "
                    f"vocabulary: {valid}",
                )
        direction = row.get("direction", "forward")
        if direction not in ("forward", "inverse"):
            loader.fail(
                table, f"{where}: `direction` must be forward or inverse, got {direction!r}"
            )
            direction = "forward"

        parsed.append(
            KmRule(
                id=rule_id,
                enabled=enabled,
                priority=priority,
                learner_state=learner_state,
                intent=intent,
                requires=condition.get("requires"),
                kl_map_action=str(row.get("kl_map_action", "")),
                strategy_effect=str(row.get("strategy_effect", "")),
                knowledge_action=action,
                target=str(row.get("target", "")),
                relation=relation,
                direction=direction,
                max_distance=row.get("max_distance"),
                note=str(row.get("note", "")),
            )
        )

    ids = [rule.id for rule in parsed]
    duplicates = {rule_id for rule_id in ids if ids.count(rule_id) > 1}
    if duplicates:
        loader.fail(table, f"duplicate rule id(s): {', '.join(sorted(duplicates))}")
    enabled_ids = {rule.id for rule in parsed if rule.enabled}
    missing = [rule_id for rule_id in REQUIRED_ACTIVE_KM_RULES if rule_id not in enabled_ids]
    if missing:
        loader.fail(table, f"{', '.join(missing)} must be present and enabled for the POC (§3.5)")
    return tuple(parsed)


def _load_response_templates(
    loader: _Loader, data: dict[str, Any], strategies: dict[StrategyId, Strategy]
) -> ResponseTemplates:
    table = "response_templates"

    blocks: dict[str, Block] = {}
    blocks_raw = data.get("blocks")
    if not isinstance(blocks_raw, dict) or not blocks_raw:
        loader.fail(table, "`blocks` must be a non-empty mapping of block name to kind")
        blocks_raw = {}
    for name, spec in blocks_raw.items():
        where = f"blocks.{name}"
        if not isinstance(spec, dict) or "kind" not in spec:
            loader.fail(table, f"{where}: must be a mapping with a `kind`")
            continue
        blocks[str(name)] = Block(name=str(name), kind=str(spec["kind"]))

    templates: dict[StrategyId, ResponseTemplate] = {}
    templates_raw = data.get("templates")
    if not isinstance(templates_raw, list) or not templates_raw:
        loader.fail(table, "`templates` must be a non-empty list")
        templates_raw = []
    for index, row in enumerate(templates_raw):
        where = f"templates[{index}]"
        if not isinstance(row, dict):
            loader.fail(table, f"{where}: each template must be a mapping")
            continue
        strategy = loader.strategy_id(table, where, row.get("strategy"))
        if strategy is None:
            continue
        where = f"template {strategy.value}"
        if strategy in templates:
            loader.fail(table, f"{where}: duplicate template")
            continue
        if strategy not in strategies:
            loader.fail(table, f"{where}: not in the Strategy Library")

        structure_raw = row.get("structure")
        if not isinstance(structure_raw, list) or not structure_raw:
            loader.fail(table, f"{where}: `structure` must be a non-empty list of block names")
            structure_raw = []
        structure = [str(name) for name in structure_raw]
        for name in structure:
            if name not in blocks:
                loader.fail(table, f"{where}: block {name!r} is not declared in `blocks`")

        max_words = row.get("max_words")
        if not isinstance(max_words, int) or isinstance(max_words, bool) or max_words <= 0:
            loader.fail(
                table, f"{where}: `max_words` must be a positive integer, got {max_words!r}"
            )
            max_words = 100

        wait = row.get("wait_for_student")
        if not isinstance(wait, bool) and wait != "depends_on_attempt_required":
            loader.fail(
                table,
                f"{where}: `wait_for_student` must be true, false, or "
                f"'depends_on_attempt_required', got {wait!r}",
            )
            wait = True

        templates[strategy] = ResponseTemplate(
            strategy=strategy,
            structure=tuple(structure),
            max_words=max_words,
            follow_up=str(row.get("follow_up", "")),
            wait_for_student=wait,
            source=str(row.get("source", "spec")),
            note=str(row.get("note", "")),
        )

    supporting: dict[StrategyId, SupportingBlock] = {}
    supporting_raw = data.get("supporting_blocks")
    if not isinstance(supporting_raw, dict) or not supporting_raw:
        loader.fail(table, "`supporting_blocks` must be a non-empty mapping")
        supporting_raw = {}
    for key, spec in supporting_raw.items():
        where = f"supporting_blocks.{key}"
        strategy = loader.strategy_id(table, where, key)
        if strategy is None:
            continue
        if not isinstance(spec, dict) or "block" not in spec:
            loader.fail(table, f"{where}: must be a mapping with a `block`")
            continue
        name = str(spec["block"])
        if name not in blocks:
            loader.fail(table, f"{where}: block {name!r} is not declared in `blocks`")
        requires = spec.get("requires_student_response", False)
        if not isinstance(requires, bool):
            loader.fail(table, f"{where}: `requires_student_response` must be true or false")
            requires = False
        supporting[strategy] = SupportingBlock(
            strategy=strategy, block=name, requires_student_response=requires
        )

    missing = [member.value for member in StrategyId if member not in supporting]
    if missing:
        loader.fail(
            table,
            f"`supporting_blocks` is missing {', '.join(missing)}; every strategy can be "
            "selected as supporting, so every strategy needs a block",
        )
    return ResponseTemplates(blocks=blocks, templates=templates, supporting_blocks=supporting)


def _load_global_rules(
    loader: _Loader, data: dict[str, Any]
) -> tuple[GlobalRules, dict[str, str], GuardrailConfig]:
    table = "global_rules"

    rules_raw = data.get("rules")
    rules = GlobalRules()
    if not isinstance(rules_raw, dict) or not rules_raw:
        loader.fail(table, "`rules` must be a mapping of the §4.2 global rules")
    else:
        unknown = set(rules_raw) - set(GlobalRules.model_fields)
        if unknown:
            loader.fail(
                table,
                f"unknown global rule(s) {', '.join(sorted(unknown))}; the §4.2 rule set is "
                f"{', '.join(sorted(GlobalRules.model_fields))}",
            )
        try:
            rules = GlobalRules(**{k: v for k, v in rules_raw.items() if k not in unknown})
        except ValidationError as exc:
            loader.fail(table, f"`rules` does not satisfy the GlobalRules schema: {exc}")

    wording_raw = data.get("student_level_wording")
    wording: dict[str, str] = {}
    if not isinstance(wording_raw, dict):
        loader.fail(table, "`student_level_wording` must be a mapping keyed by student level")
    else:
        wording = {str(k): str(v) for k, v in wording_raw.items()}
        missing = [level.value for level in StudentLevel if level.value not in wording]
        if missing:
            loader.fail(
                table,
                f"`student_level_wording` is missing {', '.join(missing)}; §4.2 gives a "
                "wording rule for every level",
            )

    guardrail_raw = data.get("guardrail")
    guardrail = GuardrailConfig(
        count_tolerance=0.25, max_regenerations=1, regeneration_note="", fallback_text={}
    )
    if not isinstance(guardrail_raw, dict) or not guardrail_raw:
        loader.fail(table, "`guardrail` must be a mapping (§6 thresholds and fallback copy)")
    else:
        tolerance = guardrail_raw.get("count_tolerance", 0.25)
        if not isinstance(tolerance, int | float) or isinstance(tolerance, bool) or tolerance < 0:
            loader.fail(table, f"`guardrail.count_tolerance` must be >= 0, got {tolerance!r}")
            tolerance = 0.25
        attempts = guardrail_raw.get("max_regenerations", 1)
        if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 0:
            loader.fail(table, f"`guardrail.max_regenerations` must be >= 0, got {attempts!r}")
            attempts = 1
        note = str(guardrail_raw.get("regeneration_note", "")).strip()
        if not note:
            loader.fail(
                table,
                "`guardrail.regeneration_note` must be non-empty; a retry without a note "
                "just asks the model for the same draft again",
            )
        fallback_raw = guardrail_raw.get("fallback_text")
        fallback_text: dict[str, str] = {}
        if not isinstance(fallback_raw, dict) or not fallback_raw:
            loader.fail(table, "`guardrail.fallback_text` must be a mapping keyed by language")
        else:
            fallback_text = {str(k): str(v) for k, v in fallback_raw.items()}
            missing = [lang.value for lang in Language if lang.value not in fallback_text]
            if missing:
                loader.fail(
                    table,
                    f"`guardrail.fallback_text` is missing {', '.join(missing)}; a leak in "
                    "any supported language still has to be answerable",
                )
        guardrail = GuardrailConfig(
            count_tolerance=float(tolerance),
            max_regenerations=int(attempts),
            regeneration_note=note,
            fallback_text=fallback_text,
        )
    return rules, wording, guardrail


def _cross_validate(
    loader: _Loader,
    trigger_rows: tuple[TriggerRow, ...],
    fallback: TriggerRow | None,
    combinations: CombinationRules | None,
    policy_rows: tuple[PolicyRow, ...],
    templates: ResponseTemplates,
) -> None:
    """Every Trigger Matrix row must be a legal combination.

    Caught here rather than at turn time: a trigger row whose supporting
    strategies the Combination Rules forbid would silently lose them on every
    turn it fires, which is exactly the kind of quiet degradation A1 exists to
    prevent.
    """
    if combinations is None:
        return
    for row in [r for r in (fallback,) if r is not None] + list(trigger_rows):
        if row.primary not in combinations.by_primary:
            loader.fail(
                "combination_rules",
                f"no combination declares primary {row.primary.value}, used by "
                f"trigger row {row.id}; add it or change the trigger row",
            )
            continue
        allowed = combinations.allowed_supporting(row.primary)
        forbidden = [s.value for s in row.supporting if s not in allowed]
        if forbidden:
            allowed_text = ", ".join(sorted(s.value for s in allowed)) or "(none)"
            loader.fail(
                "trigger_matrix",
                f"row {row.id}: supporting {', '.join(forbidden)} is not allowed with "
                f"primary {row.primary.value}; allowed: {allowed_text}",
            )
        if len(row.supporting) > combinations.max_supporting:
            loader.fail(
                "trigger_matrix",
                f"row {row.id}: {len(row.supporting)} supporting strategies exceeds "
                f"max_supporting={combinations.max_supporting}",
            )
        hint = combinations.hint_strategies
        full = combinations.full_answer_strategies
        selected = {row.primary, *row.supporting}
        if combinations.hint_first_beats_full_answer and (selected & hint) and (selected & full):
            loader.fail(
                "trigger_matrix",
                f"row {row.id}: pairs a hint strategy with a full-answer strategy, which "
                "the hint-first rule forbids in the same turn",
            )

    # Every strategy that can reach the primary slot needs a response template,
    # or the planner has nothing to build the turn from.
    reachable_primaries: set[StrategyId] = {row.primary for row in trigger_rows}
    if fallback is not None:
        reachable_primaries.add(fallback.primary)
    reachable_primaries.add(combinations.cannot_evaluate_primary)
    reachable_primaries |= {row.veto_primary for row in policy_rows if row.veto_primary is not None}
    reachable_primaries |= combinations.hint_strategies | combinations.check_strategies
    for strategy_id in sorted(reachable_primaries, key=lambda s: s.value):
        if strategy_id not in templates.templates:
            loader.fail(
                "response_templates",
                f"{strategy_id.value} can hold the primary slot but has no template; "
                "the planner cannot build a turn from it",
            )


def load_tables(rules_dir: Path | str | None = None) -> RuleTables:
    """Load and validate all seven rule tables, or raise `RuleTableError`.

    Args:
        rules_dir: directory holding the YAML files; defaults to
            `content/rules/` in the repository.

    Raises:
        RuleTableError: with every problem found across every table.
    """
    loader = _Loader(rules_dir=Path(rules_dir) if rules_dir else DEFAULT_RULES_DIR)

    raw = {name: loader.read(name) for name in TABLE_FILES}
    strategies = _load_strategy_library(loader, raw["strategy_library"])
    trigger_rows, fallback = _load_trigger_matrix(loader, raw["trigger_matrix"], strategies)
    combinations = _load_combination_rules(loader, raw["combination_rules"], strategies)
    policy_rows = _load_teaching_policy(loader, raw["teaching_policy"])
    baseline_context = str(raw["teaching_policy"].get("baseline_context", "normal_learning"))
    km_rules = _load_km_rules(loader, raw["km_rules"])
    templates = _load_response_templates(loader, raw["response_templates"], strategies)
    global_rules, wording, guardrail = _load_global_rules(loader, raw["global_rules"])
    _cross_validate(loader, trigger_rows, fallback, combinations, policy_rows, templates)

    if loader.problems:
        raise RuleTableError(loader.problems)
    if fallback is None or combinations is None:  # pragma: no cover - guarded above
        raise RuleTableError(["trigger_matrix/combination_rules failed to load"])

    return RuleTables(
        trigger_rows=trigger_rows,
        trigger_fallback=fallback,
        strategies=strategies,
        combinations=combinations,
        policy_rows=policy_rows,
        km_rules=km_rules,
        response_templates=templates,
        global_rules=global_rules,
        student_level_wording=wording,
        guardrail=guardrail,
        version_map={name: str(raw[name].get("version", "")) for name in TABLE_FILES},
        content_hashes=dict(loader.hashes),
        baseline_context=baseline_context,
        warnings=tuple(loader.warnings),
    )


@lru_cache(maxsize=4)
def get_tables(rules_dir: str | None = None) -> RuleTables:
    """Process-wide cached `load_tables`. Tables are immutable data per process."""
    return load_tables(rules_dir)


__all__ = [
    "ANY",
    "DEFAULT_RULES_DIR",
    "Block",
    "Combination",
    "CombinationRules",
    "GuardrailConfig",
    "KmRule",
    "PolicyRow",
    "ResponseTemplate",
    "ResponseTemplates",
    "RuleTableError",
    "RuleTables",
    "Strategy",
    "SupportingBlock",
    "TriggerRow",
    "get_tables",
    "load_tables",
]
