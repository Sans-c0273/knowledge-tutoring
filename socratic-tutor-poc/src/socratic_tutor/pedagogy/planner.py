"""Response Planning — step 4 of the pipeline (Tech Spec §4, requirement R9).

A template lookup plus rule application. No LLM, no reasoning: the `ResponsePlan`
this emits is the generation model's entire brief (§4.3), and everything the
generator is permitted to do is stated in it.

The six deterministic steps of §4::

    1. read the primary strategy
    2. load its response template            (§4.1)
    3. add the supporting strategy blocks    (§3.3 sequences)
    4. apply the student-level wording rule  (§4.2)
    5. apply the teaching constraints        (§3.4, via StrategySelection)
    6. emit ResponsePlan JSON                (§4.3)

**Constraint conflicts are impossible by construction, not caught afterwards.**
Step 5 rewrites the emitted `structure`: when the full solution is withheld, the
solution blocks are *removed from the plan*, not merely flagged. A generator
handed this plan has no block telling it to state the answer, so a leak requires
the model to invent a section it was never asked for — which is then what the
§6 guardrail exists to catch. Fail-closed all the way down: `Constraints` and
`ResponsePlan.full_solution_allowed` default restrictive, and this module always
sets all of them explicitly rather than relying on those defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from socratic_tutor.models.enums import Language, StrategyId, StudentLevel
from socratic_tutor.models.plan import ResponsePlan
from socratic_tutor.models.strategy import StrategySelection
from socratic_tutor.pedagogy.tables import ResponseTemplates, RuleTables


class PlannerError(ValueError):
    """The plan could not be built from the tables as loaded.

    Only raised for a condition `load_tables` should already have rejected (a
    primary strategy with no template). It exists so the failure is loud at the
    point of use rather than silently producing an empty plan.
    """


@dataclass
class PlanTrace:
    """Why the emitted plan looks the way it does, for the glass-box UI."""

    primary: str = ""
    template: str = ""
    template_structure: list[str] = field(default_factory=list)
    supporting_blocks_added: list[str] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    wording_rule: str = ""
    final_structure: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary,
            "template": self.template,
            "template_structure": list(self.template_structure),
            "supporting_blocks_added": list(self.supporting_blocks_added),
            "actions": list(self.actions),
            "wording_rule": self.wording_rule,
            "final_structure": list(self.final_structure),
        }


def plan_response(
    selection: StrategySelection,
    tables: RuleTables,
    *,
    language: Language = Language.EN,
    student_level: StudentLevel | None = None,
    trace: PlanTrace | None = None,
) -> ResponsePlan:
    """Turn a `StrategySelection` into the `ResponsePlan` handed to Call C.

    Args:
        selection: step 3 output, carrying the strategies and the constraints.
        tables: loaded rule tables (templates and §4.2 global rules).
        language: the student's language; Thai default for Thai courses (§5.3).
        student_level: overrides the level recorded on the selection.
        trace: populated with the planning decisions when supplied.

    Returns:
        A `ResponsePlan` whose `structure` already reflects every constraint.

    Raises:
        PlannerError: the primary strategy has no response template.
    """
    templates = tables.response_templates
    rules = tables.global_rules
    constraints = selection.constraints
    trace = trace if trace is not None else PlanTrace()

    # 1-2. Primary strategy and its template.
    primary = selection.primary_strategy.strategy_id
    template = templates.templates.get(primary)
    if template is None:
        known = ", ".join(sorted(s.value for s in templates.templates))
        raise PlannerError(
            f"no response template for primary strategy {primary.value}; "
            f"templates exist for {known}. Add one to content/rules/response_templates.yaml."
        )
    structure = list(template.structure)
    trace.primary = primary.value
    trace.template = primary.value
    trace.template_structure = list(structure)

    waits = template.waits(student_attempt_required=constraints.student_attempt_required)

    # 3. Supporting strategy blocks.
    for ref in selection.supporting_strategies:
        supporting = templates.supporting_blocks.get(ref.strategy_id)
        if supporting is None:
            trace.actions.append(
                {
                    "rule": "unknown_supporting_strategy",
                    "dropped": ref.strategy_id.value,
                    "reason": "no supporting block declared for this strategy",
                }
            )
            continue
        if supporting.block in structure:
            continue
        if waits and supporting.requires_student_response:
            # §3.3's own sequences put this block in the next turn:
            # "Hint -> Student tries -> Guide", "Question -> Answer -> Feedback".
            trace.actions.append(
                {
                    "rule": "deferred_to_next_turn",
                    "dropped": supporting.block,
                    "strategy": ref.strategy_id.value,
                    "reason": (
                        "acts on a student response this turn is still waiting for "
                        "(Tech Spec §3.3 sequence)"
                    ),
                }
            )
            continue
        structure.append(supporting.block)
        trace.supporting_blocks_added.append(supporting.block)

    # 4. Student-level wording rule (§4.2). The plan carries the level; the
    # prose directive goes to the generation prompt's Policy layer.
    level = student_level or selection.student_level or StudentLevel.BEGINNER
    trace.wording_rule = tables.student_level_wording.get(level.value, "")

    # 5. Teaching constraints, applied to the structure itself.
    structure = _apply_constraints(
        structure,
        templates=templates,
        direct_answer_allowed=constraints.direct_answer_allowed,
        full_solution_allowed=constraints.full_solution_allowed_now,
        forbid_hint_with_solution=not rules.hint_first_and_full_solution_same_turn,
        quiz_answer_before_attempt=rules.quiz_answer_before_student_attempt,
        max_questions=rules.max_questions,
        actions=trace.actions,
    )

    if not structure:
        # Never emit an empty brief: an unconstrained generator is worse than a
        # narrow one. Fall back to the primary template's first question block,
        # or to a bare check question.
        recovery = next(
            (name for name in template.structure if templates.block(name).is_question),
            None,
        )
        structure = [recovery] if recovery else ["single_check_question"]
        trace.actions.append(
            {
                "rule": "empty_structure_recovered",
                "added": structure[0],
                "reason": "every block was removed by the constraints; asking is always safe",
            }
        )

    # 6. Emit.
    #
    # `full_solution_allowed` describes **what this turn actually does**, not
    # what the context would permit. Those are different questions and conflating
    # them disables three safety mechanisms at once.
    #
    # The normal-learning policy row leaves `full_solution_allowed_now` True even
    # for a hint-only turn, because at policy level nothing forbids a solution
    # here. But if step 5 emitted no block that states one, this turn is not
    # going to — and downstream everything keys off this flag: the guardrail
    # gates its entire canonical scan on it, the orchestrator decides whether to
    # buffer-and-scan or stream live on it, and the generation prompt's Policy
    # layer tells the model whether it may give the complete solution. Reported
    # permissively, an ordinary "I'm stuck, help me" hint turn runs with the
    # scan off, tokens unretractable, and the prompt inviting a full solution.
    #
    # Derived here rather than fixed at each of those three call sites so there
    # is one source of truth. `orchestrator._may_state_solution` already asks
    # this narrower question for prompt contents; this makes the plan itself
    # carry the answer.
    states_solution = any(
        templates.block(name).is_solution for name in structure if name in templates.blocks
    )
    full_solution_allowed = constraints.full_solution_allowed_now and states_solution
    if constraints.full_solution_allowed_now and not states_solution:
        trace.actions.append(
            {
                "rule": "full_solution_not_stated_this_turn",
                "reason": (
                    "policy permits a full solution in this context, but the emitted "
                    "structure contains no block that states one, so the plan reports "
                    "False and the guardrail stays armed"
                ),
            }
        )

    trace.final_structure = list(structure)
    return ResponsePlan(
        structure=structure,
        max_words=template.max_words,
        max_questions=rules.max_questions,
        language_level=level,
        language=language,
        full_solution_allowed=full_solution_allowed,
        wait_for_student=waits,
    )


def _apply_constraints(
    structure: list[str],
    *,
    templates: ResponseTemplates,
    direct_answer_allowed: bool,
    full_solution_allowed: bool,
    forbid_hint_with_solution: bool,
    quiz_answer_before_attempt: bool,
    max_questions: int,
    actions: list[dict[str, Any]],
) -> list[str]:
    """Rewrite the structure so the plan cannot express a forbidden turn."""
    blocks = [templates.block(name) for name in structure if name in templates.blocks]

    def drop(rule: str, reason: str, predicate) -> None:
        nonlocal blocks
        kept = []
        for block in blocks:
            if predicate(block):
                actions.append({"rule": rule, "dropped": block.name, "reason": reason})
            else:
                kept.append(block)
        blocks = kept

    if not full_solution_allowed:
        drop(
            "full_solution_withheld",
            "full_solution_allowed_now is false, so the plan carries no block that "
            "states the solution",
            lambda block: block.is_solution,
        )
    if not direct_answer_allowed:
        drop(
            "direct_answer_prohibited",
            "Teaching Policy prohibits a direct answer this turn",
            lambda block: block.is_solution,
        )
    if not quiz_answer_before_attempt:
        drop(
            "quiz_answer_withheld",
            "a practice question's answer is never planned before the student attempts it",
            lambda block: block.is_solution and block.name.endswith("answer_key"),
        )

    # Hint-first and a full solution never share a turn (§4.2). The hint stays;
    # anything that walks the student to the answer goes.
    if forbid_hint_with_solution and any(block.is_hint for block in blocks):
        drop(
            "hint_first_and_full_solution_same_turn",
            "a hint and the solution path cannot share a turn; the solution path "
            "belongs to the turn after the student attempts",
            lambda block: block.is_solution or block.is_steps,
        )

    # Questions last, in the order they were added: §3.3's sequences all close on
    # the question ("Answer -> Example -> Check"), and a supporting block appended
    # after the template's own closing question would otherwise land behind it.
    ordered = [block for block in blocks if not block.is_question]
    questions = [block for block in blocks if block.is_question]

    if len(questions) > max_questions:
        for block in questions[max_questions:]:
            actions.append(
                {
                    "rule": "max_questions",
                    "dropped": block.name,
                    "reason": f"exceeds the global max_questions={max_questions}",
                }
            )
        questions = questions[:max_questions]

    return [block.name for block in (*ordered, *questions)]


def plan_conflicts(plan: ResponsePlan, templates: ResponseTemplates) -> list[str]:
    """Constraint conflicts found in a finished plan. Always empty by construction.

    An invariant check, not a repair step: the planner cannot emit a conflicting
    plan, and this exists so the tests and the E3 harness can assert that rather
    than trust it.
    """
    conflicts: list[str] = []
    blocks = [templates.block(name) for name in plan.structure if name in templates.blocks]

    unknown = [name for name in plan.structure if name not in templates.blocks]
    if unknown:
        conflicts.append(f"undeclared block(s): {', '.join(unknown)}")

    hints = [block.name for block in blocks if block.is_hint]
    solutions = [block.name for block in blocks if block.is_solution]
    if hints and solutions:
        conflicts.append(
            f"hint {hints} and solution {solutions} in the same turn "
            "(hint_first_and_full_solution_same_turn is false)"
        )
    if solutions and not plan.full_solution_allowed:
        conflicts.append(f"solution block(s) {solutions} while full_solution_allowed is false")

    questions = [block.name for block in blocks if block.is_question]
    if len(questions) > plan.max_questions:
        conflicts.append(
            f"{len(questions)} question blocks over max_questions={plan.max_questions}"
        )
    if not plan.structure:
        conflicts.append("empty structure: the generator would have no brief")
    return conflicts


def strategy_of(plan_block: str, templates: ResponseTemplates) -> StrategyId | None:
    """Which strategy contributes `plan_block` as its supporting block, if any."""
    for strategy, supporting in templates.supporting_blocks.items():
        if supporting.block == plan_block:
            return strategy
    return None


__all__ = ["PlanTrace", "PlannerError", "plan_conflicts", "plan_response", "strategy_of"]
