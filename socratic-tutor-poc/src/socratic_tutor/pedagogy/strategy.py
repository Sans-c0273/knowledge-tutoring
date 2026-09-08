"""Strategy Selection — the four rule tables in sequence (Tech Spec §3, R7).

Deterministic and LLM-free (Tech Spec §1 step 3). Every teaching decision this
module makes is reconstructable from `StrategySelection.context["decision_trace"]`,
which is what the glass-box UI renders and what `TurnTrace` persists (§7).

Order of operations::

    1. Trigger Matrix  (§3.1)  which strategy?      -> primary + supporting ids
    2. Strategy Library(§3.2)  what does it mean?   -> readable StrategyRefs
    3. Combination Rules(§3.3) can they coexist?    -> 1 primary + <=2 supporting
    3b. Evaluation routing(§5.2) cannot_evaluate    -> S07, never invented feedback
    4. Teaching Policy (§3.4)  what is permitted?   -> constraints, and a veto
    5. KM rules        (§3.5)  KL Map guidance      -> knowledge_guidance

Steps 1-3 choose; step 4 is the hard override that can overrule all of them.
That is the spec's precedence order read top-down:

    Teaching Policy -> Special Handling -> Learner State -> Intent -> Student Level

Special Handling, Learner State, Intent and Student Level are the four key fields
of the Trigger Matrix, ranked against each other by row priority; Teaching Policy
sits above the whole table and is applied afterwards so it can veto the result
(Architecture A2 — answer-withholding is a context rule, not a model behaviour).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from socratic_tutor.models.enums import (
    Evaluation,
    Intent,
    LearnerState,
    SpecialHandling,
    StrategyId,
    StudentLevel,
)
from socratic_tutor.models.intent import IntentResult
from socratic_tutor.models.knowledge import KnowledgeContext
from socratic_tutor.models.strategy import (
    Constraints,
    KnowledgeGuidance,
    StrategyRef,
    StrategySelection,
)
from socratic_tutor.pedagogy.tables import (
    CombinationRules,
    KmRule,
    PolicyRow,
    RuleTables,
    TriggerRow,
)

#: The spec's precedence order, carried in every trace for the glass-box UI.
PRECEDENCE = (
    "teaching_policy",
    "special_handling",
    "learner_state",
    "intent",
    "student_level",
)


@dataclass(frozen=True)
class TriggerMatch:
    """One Trigger Matrix row that matched, and how it fared in the contest.

    The inspector's job is to answer "which rule chose this strategy, and what
    beat what". That is a *contest* between rows, so the trace carries the
    contest — every candidate, its rank inputs, whether it won, and for a loser
    which row beat it on which tie-break. Emitting only the winner's id would
    make the glass box show a conclusion with its reasoning removed.

    `key` is a mapping rather than a positional list: a four-element list
    encodes each field's meaning in its index, which the next reader has to know
    or guess.
    """

    row_id: str
    key: dict[str, str]
    priority: int
    specificity: int
    primary: str
    supporting: tuple[str, ...]
    selected: bool
    #: Row that outranked this one, and the tie-break it lost on. None for the
    #: winner.
    lost_to: str | None = None
    lost_on: str | None = None
    source: str = ""

    @classmethod
    def of(cls, row: TriggerRow, *, winner: TriggerRow | None) -> TriggerMatch:
        selected = winner is not None and row.id == winner.id
        lost_to = None if selected or winner is None else winner.id
        return cls(
            row_id=row.id,
            key={
                "intent": row.intent,
                "learner_state": row.learner_state,
                "student_level": row.student_level,
                "special_handling": row.special_handling,
            },
            priority=row.priority,
            specificity=row.specificity,
            primary=row.primary.value,
            supporting=tuple(s.value for s in row.supporting),
            selected=selected,
            lost_to=lost_to,
            lost_on=None if lost_to is None else _tie_break(row, winner),
            source=row.source,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "key": dict(self.key),
            "priority": self.priority,
            "specificity": self.specificity,
            "primary": self.primary,
            "supporting": list(self.supporting),
            "selected": self.selected,
            "lost_to": self.lost_to,
            "lost_on": self.lost_on,
            "source": self.source,
        }


def _tie_break(row: TriggerRow, winner: TriggerRow) -> str:
    """Which dimension of `TriggerRow.rank` decided this pair.

    Named rather than inferred downstream, because "TM12 beat TM16" is only half
    the answer — a reviewer asking whether the table is right needs to know
    whether it came down to a priority someone chose or to alphabetical order,
    which is the case worth looking at twice.
    """
    if winner.priority != row.priority:
        return "priority"
    if winner.specificity != row.specificity:
        return "specificity"
    return "row_id"


@dataclass(frozen=True)
class KmMatch:
    """One KM rule that matched, and how it fared against the others (§3.5).

    The same contest shape as `TriggerMatch`, one layer down. The O2 tie-break —
    KM02 beats KM03 when both match, and both are logged — is a decision the
    spec calls out explicitly, so it deserves to survive into the trace as
    structure rather than as a sentence someone reassembles downstream.

    `requirement` names the `requires` clause the rule matched on
    (`immediate_prerequisite`, `related_concept`), and it carries more weight
    than its name suggests.

    **It exists to keep the trace from implying a diagnosis.** The hard rule of
    §3.5 and Architecture A5 is that a prerequisite edge is a hypothesis about
    the *domain*, never a finding about *this student* — the system must never
    assert "you don't understand X" from graph structure alone, which is why
    KM02's action is `check_or_scaffold_prerequisite` rather than
    `use_prerequisite_as_scaffold`.

    A bare "KM02 applied" in the inspector leaves the reader to supply the
    missing clause, and the clause they will supply is the diagnosis. "Matched
    on `immediate_prerequisite`" says the thing that is actually true: a
    prerequisite exists at distance 1 in the map. The rule fired on the graph,
    not on a judgement about the learner, and the trace should not be the place
    that quietly converts one into the other.
    """

    rule_id: str
    applied: bool
    priority: int
    #: The `requires` clause this rule matched on — a fact about the KL Map, not
    #: about the student. See the class docstring; this is load-bearing for KM02.
    requirement: str | None
    knowledge_action: str | None
    kl_map_action: str
    strategy_effect: str
    lost_to: str | None = None
    lost_on: str | None = None

    @classmethod
    def of(cls, rule: KmRule, *, winner: KmRule | None) -> KmMatch:
        applied = winner is not None and rule.id == winner.id
        lost_to = None if applied or winner is None else winner.id
        return cls(
            rule_id=rule.id,
            applied=applied,
            priority=rule.priority,
            requirement=rule.requires,
            knowledge_action=rule.knowledge_action.value if rule.knowledge_action else None,
            kl_map_action=rule.kl_map_action,
            strategy_effect=rule.strategy_effect,
            lost_to=lost_to,
            lost_on=None
            if lost_to is None or winner is None
            else ("priority" if winner.priority != rule.priority else "rule_id"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "applied": self.applied,
            "priority": self.priority,
            "requirement": self.requirement,
            "knowledge_action": self.knowledge_action,
            "kl_map_action": self.kl_map_action,
            "strategy_effect": self.strategy_effect,
            "lost_to": self.lost_to,
            "lost_on": self.lost_on,
        }


@dataclass(frozen=True)
class UnmatchedLookup:
    """A Trigger Matrix miss — the review-queue event.

    Rule-table gaps are expected: the table is seeded, not complete. Every miss
    is emitted so a weekly review can add the row, which is how the table grows
    from evidence rather than from guesswork (Architecture, risk register).
    """

    intent: Intent
    learner_state: LearnerState
    student_level: StudentLevel
    special_handling: SpecialHandling
    topic: str
    fallback_row_id: str
    table_version: str

    def as_event(self) -> dict[str, Any]:
        return {
            "event": "trigger_matrix_unmatched",
            "intent": self.intent.value,
            "learner_state": self.learner_state.value,
            "student_level": self.student_level.value,
            "special_handling": self.special_handling.value,
            "topic": self.topic,
            "fallback_row_id": self.fallback_row_id,
            "trigger_matrix_version": self.table_version,
        }


def select_strategy(
    intent_result: IntentResult,
    student_level: StudentLevel,
    topic: str,
    knowledge_context: KnowledgeContext | None,
    evaluation: Evaluation | None,
    rag_evidence_present: bool,
    tables: RuleTables,
    *,
    on_unmatched: Callable[[UnmatchedLookup], None] | None = None,
) -> StrategySelection:
    """Run the four tables plus the KM rules and return the turn's strategy.

    Args:
        intent_result: step 1 output — intent, learner state, special handling.
        student_level: the student's declared level for `topic` (§2.2).
        topic: the mapped syllabus topic for this turn.
        knowledge_context: KL Map neighbourhood, or None when no map lookup ran.
        evaluation: Answer Evaluation verdict; only meaningful when S06 fires.
        rag_evidence_present: False triggers the anti-hallucination policy row.
        tables: loaded, validated rule tables.
        on_unmatched: called with an `UnmatchedLookup` when no Trigger Matrix row
            matches. The same event is always recorded in the decision trace, so
            the callback is optional.

    Returns:
        A fully populated `StrategySelection`. `context` carries the spec's
        `topic`/`student_level` plus a `decision_trace` dict naming every row
        that matched, the row that was selected, the policy rows that applied and
        every strategy that was dropped or substituted, with the reason.
    """
    intent = intent_result.intent
    learner_state = intent_result.learner_state
    special_handling = intent_result.special_handling.type

    trace: dict[str, Any] = {
        "precedence": list(PRECEDENCE),
        "table_versions": dict(tables.version_map),
        "lookup": {
            "intent": intent.value,
            "learner_state": learner_state.value,
            "student_level": student_level.value,
            "special_handling": special_handling.value,
            "topic": topic,
            "rag_evidence_present": rag_evidence_present,
            "evaluation": evaluation.value if evaluation else None,
        },
    }

    # 1. Trigger Matrix ----------------------------------------------------
    selected_row, unmatched = _match_trigger(
        tables, intent, learner_state, student_level, special_handling, topic, trace
    )
    if unmatched is not None and on_unmatched is not None:
        on_unmatched(unmatched)

    # 2. Strategy Library --------------------------------------------------
    primary = selected_row.primary
    supporting = list(selected_row.supporting)
    trace["strategy_library"] = {
        strategy_id.value: tables.strategy(strategy_id).name
        for strategy_id in (primary, *supporting)
    }

    # 3. Combination Rules -------------------------------------------------
    actions: list[dict[str, Any]] = []
    primary, supporting = _apply_combination_rules(
        tables.combinations, primary, supporting, actions
    )

    # 3b. Evaluation routing (§5.2) ---------------------------------------
    primary, supporting, evaluation = _apply_evaluation_routing(
        tables, primary, supporting, evaluation, actions
    )
    trace["combination_rule"] = primary.value
    trace["combination_sequence"] = list(
        tables.combinations.by_primary[primary].sequence
        if primary in tables.combinations.by_primary
        else ()
    )
    trace["combination_actions"] = actions

    # 4. Teaching Policy — the hard override ------------------------------
    constraints, primary, supporting = _apply_teaching_policy(
        tables, primary, supporting, special_handling, rag_evidence_present, trace
    )

    # 5. KM rules ----------------------------------------------------------
    knowledge_guidance = _apply_km_rules(tables, intent, learner_state, knowledge_context, trace)

    trace["final"] = {
        "primary": primary.value,
        "supporting": [strategy_id.value for strategy_id in supporting],
        "constraints": constraints.model_dump(),
    }

    return StrategySelection(
        primary_strategy=_ref(tables, primary),
        evaluation=evaluation,
        supporting_strategies=[_ref(tables, strategy_id) for strategy_id in supporting],
        constraints=constraints,
        knowledge_guidance=knowledge_guidance,
        context={
            "topic": topic,
            "student_level": student_level.value,
            "decision_trace": trace,
        },
    )


def _ref(tables: RuleTables, strategy_id: StrategyId) -> StrategyRef:
    """Resolve an id to the `{strategy_id, strategy_name}` pair of §2.4."""
    return StrategyRef(
        strategy_id=strategy_id, strategy_name=tables.strategy(strategy_id).strategy_name
    )


def _match_trigger(
    tables: RuleTables,
    intent: Intent,
    learner_state: LearnerState,
    student_level: StudentLevel,
    special_handling: SpecialHandling,
    topic: str,
    trace: dict[str, Any],
) -> tuple[TriggerRow, UnmatchedLookup | None]:
    """Find the winning Trigger Matrix row (§3.1).

    Candidates rank by (priority DESC, specificity DESC, row id ASC): the spec's
    "highest priority wins", then an exact row beating a wildcard row, then the
    id so that an otherwise identical pair still resolves the same way on every
    machine and every run.
    """
    matched = sorted(
        (
            row
            for row in tables.trigger_rows
            if row.matches(intent, learner_state, student_level, special_handling)
        ),
        key=lambda row: row.rank,
    )
    winner = matched[0] if matched else None
    trace["matched_trigger_rows"] = [
        TriggerMatch.of(row, winner=winner).as_dict() for row in matched
    ]

    if matched:
        trace["trigger_matched"] = True
        trace["selected_trigger_row"] = matched[0].id
        trace["unmatched_lookup"] = None
        return matched[0], None

    unmatched = UnmatchedLookup(
        intent=intent,
        learner_state=learner_state,
        student_level=student_level,
        special_handling=special_handling,
        topic=topic,
        fallback_row_id=tables.trigger_fallback.id,
        table_version=tables.version_map.get("trigger_matrix", ""),
    )
    trace["trigger_matched"] = False
    trace["selected_trigger_row"] = tables.trigger_fallback.id
    trace["unmatched_lookup"] = unmatched.as_event()
    return tables.trigger_fallback, unmatched


def _apply_combination_rules(
    rules: CombinationRules,
    primary: StrategyId,
    supporting: list[StrategyId],
    actions: list[dict[str, Any]],
) -> tuple[StrategyId, list[StrategyId]]:
    """Enforce §3.3: hint-first, the allowed sets, and the 1 + 2 cap.

    Every drop or substitution is appended to `actions` with its reason, so the
    trace shows not just what was chosen but what was rejected.
    """
    # Hint-first always wins over an immediate full answer.
    if rules.hint_first_beats_full_answer:
        selected = [primary, *supporting]
        hints = [s for s in selected if s in rules.hint_strategies]
        answers = [s for s in selected if s in rules.full_answer_strategies]
        if hints and answers:
            new_primary = primary if primary in rules.hint_strategies else hints[0]
            for answer in answers:
                actions.append(
                    {
                        "rule": "hint_first_beats_full_answer",
                        "dropped": answer.value,
                        "reason": (
                            f"{answer.value} is a full-answer strategy and cannot share a "
                            f"turn with the hint strategy {hints[0].value}"
                        ),
                    }
                )
            if new_primary != primary:
                actions.append(
                    {
                        "rule": "hint_first_beats_full_answer",
                        "promoted": new_primary.value,
                        "replaced": primary.value,
                        "reason": "the hint takes the primary slot",
                    }
                )
            supporting = [s for s in supporting if s not in answers and s != new_primary]
            primary = new_primary

    # Only the declared primary -> supporting set may coexist.
    allowed = rules.allowed_supporting(primary)
    kept: list[StrategyId] = []
    for strategy_id in supporting:
        if strategy_id == primary:
            actions.append(
                {
                    "rule": "no_duplicate_primary",
                    "dropped": strategy_id.value,
                    "reason": "already the primary strategy",
                }
            )
            continue
        if strategy_id not in allowed:
            allowed_text = ", ".join(sorted(s.value for s in allowed)) or "none"
            actions.append(
                {
                    "rule": "allowed_combinations",
                    "dropped": strategy_id.value,
                    "reason": (
                        f"not an allowed supporting strategy for primary {primary.value} "
                        f"(allowed: {allowed_text})"
                    ),
                }
            )
            continue
        kept.append(strategy_id)

    # Max 1 primary + 2 supporting per turn.
    if len(kept) > rules.max_supporting:
        for strategy_id in kept[rules.max_supporting :]:
            actions.append(
                {
                    "rule": "max_supporting",
                    "dropped": strategy_id.value,
                    "reason": f"exceeds max_supporting={rules.max_supporting}",
                }
            )
        kept = kept[: rules.max_supporting]
    return primary, kept


def _apply_evaluation_routing(
    tables: RuleTables,
    primary: StrategyId,
    supporting: list[StrategyId],
    evaluation: Evaluation | None,
    actions: list[dict[str, Any]],
) -> tuple[StrategyId, list[StrategyId], Evaluation | None]:
    """`evaluation` rides on the selection only when S06 fires (§2.4, §5.2).

    `cannot_evaluate` is a legitimate outcome: it routes the turn to Check
    Understanding rather than to invented feedback (§5.2).
    """
    if primary is not StrategyId.FEEDBACK:
        if evaluation is not None:
            actions.append(
                {
                    "rule": "evaluation_requires_s06",
                    "dropped": evaluation.value,
                    "reason": (
                        "evaluation is carried only when S06 Feedback is the primary "
                        f"strategy; primary here is {primary.value}"
                    ),
                }
            )
        return primary, supporting, None

    if evaluation is Evaluation.CANNOT_EVALUATE:
        replacement = tables.combinations.cannot_evaluate_primary
        actions.append(
            {
                "rule": "cannot_evaluate_routes_to_check",
                "replaced": primary.value,
                "promoted": replacement.value,
                "reason": "cannot_evaluate never produces feedback (Tech Spec §5.2)",
            }
        )
        allowed = tables.combinations.allowed_supporting(replacement)
        kept = [s for s in supporting if s in allowed and s != replacement]
        for strategy_id in supporting:
            if strategy_id not in kept:
                actions.append(
                    {
                        "rule": "allowed_combinations",
                        "dropped": strategy_id.value,
                        "reason": f"not allowed with the substituted primary {replacement.value}",
                    }
                )
        return replacement, kept, None

    return primary, supporting, evaluation


def _applicable_policy_rows(
    tables: RuleTables, special_handling: SpecialHandling, rag_evidence_present: bool
) -> list[PolicyRow]:
    """Rows in force this turn: the session baseline, plus whatever was detected.

    **The baseline is always included, which is what makes a self-report
    one-way.** `special_handling` comes from the student's own words, so if
    detecting it *replaced* the baseline row, a student could widen their own
    permissions by talking — and "my teacher said it's fine this time" is a
    sentence students say. Because every applicable row is combined restrictively
    downstream, adding a row can only ever tighten the turn, never loosen it.

    That makes the guarantee structural rather than a rule someone has to
    remember when editing the table: there is no way to express a
    permission-widening self-report, because widening is not an operation this
    function can perform.
    """
    baseline = tables.baseline_policy_row()
    rows = [baseline]
    rows.extend(
        row
        for row in tables.policy_rows
        if row is not baseline
        and row.applies(special_handling, rag_evidence_present=rag_evidence_present)
    )
    return sorted(rows, key=lambda row: -row.precedence)


def _apply_teaching_policy(
    tables: RuleTables,
    primary: StrategyId,
    supporting: list[StrategyId],
    special_handling: SpecialHandling,
    rag_evidence_present: bool,
    trace: dict[str, Any],
) -> tuple[Constraints, StrategyId, list[StrategyId]]:
    """Apply §3.4 — the hard override at the top of the precedence order.

    When several rows apply (homework *and* no retrieved evidence, say),
    permissions combine restrictively and requirements inclusively, so stacking
    contexts can only ever tighten what the tutor may do. Policy can veto a
    strategy the first three tables chose.
    """
    rows = _applicable_policy_rows(tables, special_handling, rag_evidence_present)
    trace["policy_rows_applied"] = [row.id for row in rows]
    trace["policy_row"] = rows[0].id if rows else None
    trace["policy_contexts"] = [row.context for row in rows]

    if not rows:
        # Unreachable with a validated table (§3.4 is exhaustive over special
        # handling). Fail closed rather than fall through to permissive defaults.
        trace["policy_actions"] = [
            {"rule": "no_policy_row", "reason": "failing closed: nothing is permitted"}
        ]
        return Constraints(), primary, supporting

    direct_answer_allowed = all(row.direct_answer_allowed for row in rows)
    full_solution_allowed_now = all(row.full_solution_allowed_now for row in rows)
    attempt_votes = [
        row.student_attempt_required for row in rows if row.student_attempt_required is not None
    ]
    student_attempt_required = any(attempt_votes) if attempt_votes else False
    hint_first = any(row.hint_first for row in rows)

    policy_actions: list[dict[str, Any]] = []
    rules = tables.combinations
    veto_row = next((row for row in rows if not row.direct_answer_allowed), rows[0])

    allow_hints = all(row.allow_hints for row in rows)
    allow_check = all(row.allow_check for row in rows)

    # Everything the applicable contexts forbid outright. A row that switches
    # hints off (assessment) thereby prohibits the hint strategies too, so the
    # one teacher-facing switch moves both the insertion rule and the ban.
    prohibited: set[StrategyId] = set()
    for row in rows:
        prohibited |= set(row.prohibited_strategies)
        if not row.allow_hints:
            prohibited |= rules.hint_strategies
    if not direct_answer_allowed:
        prohibited |= rules.full_answer_strategies
    trace["policy_prohibited_strategies"] = sorted(s.value for s in prohibited)

    # What a vetoed primary is replaced by: the highest-precedence declared
    # substitute that is not itself prohibited, else a permitted hint, else a
    # check strategy. Never nothing — the turn still has to produce a response.
    def substitute() -> StrategyId | None:
        for row in rows:
            if row.veto_primary is not None and row.veto_primary not in prohibited:
                return row.veto_primary
        for pool in (rules.hint_strategies, rules.check_strategies):
            candidate = next(
                (s for s in sorted(pool, key=lambda s: s.value) if s not in prohibited), None
            )
            if candidate is not None:
                return candidate
        return None

    # Veto: a prohibited strategy cannot hold the primary slot, whatever the
    # Trigger Matrix chose. This is the hard override doing its job.
    if primary in prohibited:
        replacement = substitute()
        if replacement is not None:
            reason = (
                f"{veto_row.context}: direct answer {veto_row.direct_answer}"
                if primary in rules.full_answer_strategies
                else f"{veto_row.context} prohibits {primary.value}"
            )
            policy_actions.append(
                {
                    "rule": "policy_vetoes_direct_answer"
                    if primary in rules.full_answer_strategies
                    else "policy_prohibits_strategy",
                    "policy_row": veto_row.id,
                    "policy_context": veto_row.context,
                    "replaced": primary.value,
                    "promoted": replacement.value,
                    "reason": reason,
                }
            )
            primary = replacement
            supporting = [s for s in supporting if s != replacement]

    for strategy_id in list(supporting):
        if strategy_id in prohibited:
            supporting.remove(strategy_id)
            policy_actions.append(
                {
                    "rule": "policy_vetoes_direct_answer"
                    if strategy_id in rules.full_answer_strategies
                    else "policy_prohibits_strategy",
                    "policy_row": veto_row.id,
                    "dropped": strategy_id.value,
                    "reason": f"{veto_row.context} prohibits {strategy_id.value}",
                }
            )

    # A context may switch off the closing check question (§3.4's
    # "Teacher-configurable" Check column) without prohibiting it as a primary.
    if not allow_check:
        for strategy_id in list(supporting):
            if strategy_id in rules.check_strategies:
                supporting.remove(strategy_id)
                policy_actions.append(
                    {
                        "rule": "policy_disallows_check",
                        "policy_row": veto_row.id,
                        "dropped": strategy_id.value,
                        "reason": f"{veto_row.context}: check is {veto_row.check}",
                    }
                )

    # Hint-first must be real, not nominal: if policy demands a hint first and no
    # hint strategy is in play, add one wherever the combination rules allow it.
    if allow_hints and hint_first and not ({primary, *supporting} & rules.hint_strategies):
        allowed = rules.allowed_supporting(primary)
        candidate = next(
            (s for s in sorted(rules.hint_strategies, key=lambda s: s.value) if s in allowed),
            None,
        )
        if candidate is not None:
            supporting.insert(0, candidate)
            policy_actions.append(
                {
                    "rule": "policy_requires_hint_first",
                    "policy_row": veto_row.id,
                    "added": candidate.value,
                    "reason": f"{veto_row.context}: {veto_row.direct_answer}",
                }
            )

    # Re-apply the combination cap: a substitution or insertion above may have
    # changed the primary or pushed the supporting list over the limit.
    primary, supporting = _apply_combination_rules(rules, primary, supporting, policy_actions)

    trace["policy_actions"] = policy_actions
    trace["policy_effect"] = {
        "direct_answer_allowed": direct_answer_allowed,
        "full_solution_allowed_now": full_solution_allowed_now,
        "student_attempt_required": student_attempt_required,
        "hint_first": hint_first,
        "allow_hints": allow_hints,
        "allow_check": allow_check,
        "factual_claims_require_evidence": any(row.factual_claims_require_evidence for row in rows),
    }

    constraints = Constraints(
        direct_answer_allowed=direct_answer_allowed,
        full_solution_allowed_now=full_solution_allowed_now,
        student_attempt_required=student_attempt_required,
    )
    return constraints, primary, supporting


def _km_rule_matches(
    rule_id: str,
    learner_state: LearnerState,
    intent: Intent,
    knowledge_context: KnowledgeContext | None,
    tables: RuleTables,
) -> bool:
    rule = tables.km_rule(rule_id)
    if not rule.enabled:
        return False
    if rule.learner_state is not None and rule.learner_state != learner_state.value:
        return False
    if rule.intent is not None and rule.intent != intent.value:
        return False
    if rule.requires is None:
        return knowledge_context is not None
    return _requirement_target(rule.requires, knowledge_context, rule.max_distance) is not None


def _requirement_target(
    requires: str, knowledge_context: KnowledgeContext | None, max_distance: int | None
) -> Any:
    """Resolve a KM rule's `requires` clause against the KL Map neighbourhood."""
    if knowledge_context is None:
        return None
    if requires == "immediate_prerequisite":
        limit = max_distance if max_distance is not None else 1
        return next((t for t in knowledge_context.prerequisites if t.distance <= limit), None)
    if requires == "related_concept":
        return next(iter(knowledge_context.related_topics), None)
    if requires == "next_topic":
        return next(iter(knowledge_context.next_topics), None)
    return None


def _apply_km_rules(
    tables: RuleTables,
    intent: Intent,
    learner_state: LearnerState,
    knowledge_context: KnowledgeContext | None,
    trace: dict[str, Any],
) -> KnowledgeGuidance | None:
    """Apply §3.5 to produce the optional `knowledge_guidance`.

    KM02 is `check_or_scaffold_prerequisite` and nothing stronger: the existence
    of a prerequisite edge is a statement about the *domain*, not a diagnosis of
    *this student*. This code must never conclude that the student lacks the
    prerequisite — it asks the generator to probe it with one question and to
    scaffold only if the probe fails (Tech Spec §3.5 hard rule, Architecture A5).

    O2 tie-break: when KM02 and KM03 both match, KM02 wins and both matches are
    logged. That falls out of the priority ordering, and `km_rules_matched`
    keeps the losing match visible for analysis.
    """
    matched = [
        rule
        for rule in tables.km_rules
        if _km_rule_matches(rule.id, learner_state, intent, knowledge_context, tables)
    ]
    matched.sort(key=lambda rule: (-rule.priority, rule.id))

    winner = matched[0] if matched else None
    km_matches = [KmMatch.of(rule, winner=winner).as_dict() for rule in matched]

    trace["km_matches"] = km_matches
    # Projected from the structured entries in the same breath, never built
    # alongside them: two lists assembled independently are two things that can
    # disagree, and the id list is the one §7 reads.
    trace["km_rules_matched"] = [match["rule_id"] for match in km_matches]
    trace["km_rules_disabled"] = [rule.id for rule in tables.km_rules if not rule.enabled]

    if winner is None:
        trace["km_rule_applied"] = None
        trace["km_actions"] = []
        return None

    trace["km_rule_applied"] = winner.id
    trace["km_actions"] = [
        {
            "rule": winner.id,
            "kl_map_action": winner.kl_map_action,
            "strategy_effect": winner.strategy_effect,
            "knowledge_action": winner.knowledge_action.value if winner.knowledge_action else None,
        }
    ]

    if knowledge_context is None or winner.knowledge_action is None:
        return None

    if winner.target == "current_topic":
        return KnowledgeGuidance(
            action=winner.knowledge_action,
            target_concept=knowledge_context.current_topic,
            relation=winner.relation,
            direction=winner.direction,
            distance=0,
        )

    target = _requirement_target(winner.requires or "", knowledge_context, winner.max_distance)
    if target is None:
        return None
    return KnowledgeGuidance(
        action=winner.knowledge_action,
        target_concept=target.topic,
        relation=winner.relation,
        direction=winner.direction,
        distance=target.distance,
    )


__all__ = ["PRECEDENCE", "KmMatch", "TriggerMatch", "UnmatchedLookup", "select_strategy"]
