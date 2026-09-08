"""Pedagogy Layer — the rule tables and the deterministic teaching pipeline.

Requirements R5 (rule tables as versioned data), R7 (Strategy Selection),
R9 (Response Planning) and R11 (Output Guardrail) — pipeline steps 3, 4 and 6.
Nothing in this package calls an LLM. Teaching decisions live in
`content/rules/*.yaml`, not in prompts, so they are auditable, unit-testable and
cannot be talked out of (Architecture A1).
"""

from __future__ import annotations

from socratic_tutor.pedagogy.guardrail import (
    GuardrailAction,
    GuardrailResult,
    check_output,
)
from socratic_tutor.pedagogy.planner import PlannerError, PlanTrace, plan_response
from socratic_tutor.pedagogy.strategy import PRECEDENCE, UnmatchedLookup, select_strategy
from socratic_tutor.pedagogy.tables import (
    RuleTableError,
    RuleTables,
    get_tables,
    load_tables,
)

__all__ = [
    "PRECEDENCE",
    "GuardrailAction",
    "GuardrailResult",
    "PlanTrace",
    "PlannerError",
    "RuleTableError",
    "RuleTables",
    "UnmatchedLookup",
    "check_output",
    "get_tables",
    "load_tables",
    "plan_response",
    "select_strategy",
]
