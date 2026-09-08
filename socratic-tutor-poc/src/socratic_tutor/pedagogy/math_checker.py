"""Deterministic answer verification for Call B (Tech Spec §5.2).

"Pair the LLM comparison with a deterministic checker (exact/normalized numeric
match) whenever the domain allows; the LLM handles conceptual/partial-credit
judgments only" (§5.2). Arithmetic is checked by code, so no amount of student
persuasion can flip a numeric verdict — that is the structural half of the
anti-sycophancy guarantee.

The checker is deliberately conservative: it answers `correct` / `incorrect`
only for an attempt that ends in a **bare final answer in simplest form**, and
returns no verdict for anything else. Working-in-progress ("2x = 6"), an
unevaluated result ("x = 6/2") and prose all defer to the LLM, because those are
exactly the partial-credit judgments the spec reserves for it.

Pure arithmetic on `fractions.Fraction` — no CAS dependency. Symbolic answers
beyond a single numeric value are out of scope for the POC and defer.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from fractions import Fraction

from socratic_tutor.models.enums import Evaluation

#: `lhs = rhs`, both sides read as one token of maths-ish characters. The token
#: window is what makes this work inside Thai prose ("ผมได้ x = 4 ครับ").
_EQUATION_RE = re.compile(r"([A-Za-z0-9_.+\-/()]+)\s*=\s*([A-Za-z0-9_.+\-/()]+)")

#: A single variable letter, optionally parenthesised — the left side of a final answer.
_BARE_VARIABLE_RE = re.compile(r"^\(?([A-Za-z])\)?$")

_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_DECIMAL_RE = re.compile(r"^[+-]?\d*\.\d+$|^[+-]?\d+\.\d*$")
_FRACTION_RE = re.compile(r"^([+-]?\d+)\s*/\s*(\d+)$")

#: Exact-match markers meaning "no attempt was made". Kept tiny and matched
#: whole-string only: swallowing a real attempt would turn a gradeable answer
#: into `cannot_evaluate`, which is worse than sending prose to the LLM.
_NO_ATTEMPT_MARKERS: frozenset[str] = frozenset(
    {
        "",
        "?",
        "??",
        "idk",
        "i dont know",
        "i don't know",
        "dunno",
        "no idea",
        "not sure",
        "no clue",
        "ไม่รู้",
        "ไม่ทราบ",
        "ไม่รู้ครับ",
        "ไม่รู้ค่ะ",
        "ไม่รู้เลย",
    }
)

_PUNCTUATION = " \t\n.,;:!?—–-…()[]\"'`"


@dataclass(frozen=True)
class CheckResult:
    """Outcome of deterministic verification.

    `verdict` is None when the attempt is not a bare final answer — the caller
    must then defer to the LLM. It is never `partially_correct`: partial credit
    is a conceptual judgment the spec assigns to the model.
    """

    verdict: Evaluation | None
    reason: str
    attempt_value: Fraction | None = None
    answer_value: Fraction | None = None

    @property
    def is_decisive(self) -> bool:
        return self.verdict is not None


def has_attempt_content(text: str) -> bool:
    """False when the message carries nothing to grade ("idk", "?", empty).

    Routes to `cannot_evaluate` without an LLM call — §5.2 requires that case to
    be a real outcome, never invented feedback.
    """
    normalized = unicodedata.normalize("NFC", text).strip().strip(_PUNCTUATION).casefold()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized not in _NO_ATTEMPT_MARKERS


def parse_number(token: str) -> Fraction | None:
    """`"7"`, `"3.0"`, `"-5"`, `"1/3"` → Fraction. Anything else → None."""
    text = token.strip().replace(" ", "")
    if _INTEGER_RE.match(text):
        return Fraction(int(text))
    if _DECIMAL_RE.match(text):
        return Fraction(text)
    match = _FRACTION_RE.match(text)
    if match:
        denominator = int(match.group(2))
        if denominator == 0:
            return None
        return Fraction(int(match.group(1)), denominator)
    return None


def is_simplest_form(token: str, value: Fraction) -> bool:
    """Whether `token` is a finished value rather than arithmetic left undone.

    `"6/2"` is not — it reduces to 3, so the student stopped one step short, which
    is the partial-credit case. `"1/3"` is, because it is already lowest terms.
    Integers and decimals always are.
    """
    match = _FRACTION_RE.match(token.strip().replace(" ", ""))
    if not match:
        return True
    return (int(match.group(1)), int(match.group(2))) == (value.numerator, value.denominator)


def parse_final_answer(text: str) -> tuple[str | None, Fraction | None]:
    """The `(variable, value)` an attempt finishes on, or `(None, None)`.

    Reads the **last** equation in the text, so shown working followed by a
    conclusion ("… ได้ 2x = 6 แล้วหาร 2 ได้ x = 3") still resolves to `x = 3`.
    A bare number with no equation at all is treated as the value.
    """
    normalized = unicodedata.normalize("NFC", text)
    equations = _EQUATION_RE.findall(normalized)
    if not equations:
        value = parse_number(normalized.strip().strip(_PUNCTUATION))
        if value is not None and is_simplest_form(normalized.strip().strip(_PUNCTUATION), value):
            return None, value
        return None, None

    lhs, rhs = equations[-1]
    variable_match = _BARE_VARIABLE_RE.match(lhs.strip())
    if variable_match is None:
        return None, None
    value = parse_number(rhs)
    if value is None or not is_simplest_form(rhs, value):
        return variable_match.group(1), None
    return variable_match.group(1), value


def check_answer(attempt: str, canonical_answer: str) -> CheckResult:
    """Compare a student attempt with the answer key, deterministically.

    Returns a verdict only when both sides resolve to a bare final value and, if
    both name a variable, the same one. Everything else defers.
    """
    if not has_attempt_content(attempt):
        return CheckResult(verdict=None, reason="the message contains no attempt to grade")

    answer_variable, answer_value = parse_final_answer(canonical_answer)
    if answer_value is None:
        return CheckResult(
            verdict=None,
            reason="the answer key is not a single numeric value, so it cannot be checked here",
        )

    attempt_variable, attempt_value = parse_final_answer(attempt)
    if attempt_value is None:
        return CheckResult(
            verdict=None,
            reason="attempt does not end in a bare final answer in simplest form",
            answer_value=answer_value,
        )

    if (
        attempt_variable is not None
        and answer_variable is not None
        and attempt_variable.casefold() != answer_variable.casefold()
    ):
        return CheckResult(
            verdict=None,
            reason=(
                f"attempt solves for {attempt_variable!r} but the key solves for "
                f"{answer_variable!r}"
            ),
            attempt_value=attempt_value,
            answer_value=answer_value,
        )

    matches = attempt_value == answer_value
    return CheckResult(
        # `reason` is persisted to the session record and rendered in the
        # inspector, so it never names either value: printing the key's value
        # here would put the withheld answer in a second place the guardrail
        # does not scan (review finding M9). The values stay on the structured
        # fields below, which stop at the evaluator.
        verdict=Evaluation.CORRECT if matches else Evaluation.INCORRECT,
        reason=(
            "deterministic check: the attempt's final value "
            f"{'matches' if matches else 'does not match'} the answer key"
        ),
        attempt_value=attempt_value,
        answer_value=answer_value,
    )


__all__ = [
    "CheckResult",
    "check_answer",
    "has_attempt_content",
    "is_simplest_form",
    "parse_final_answer",
    "parse_number",
]
