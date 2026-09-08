"""Deterministic answer checker — Tech Spec §5.2. No LLM anywhere in this file.

Two things matter here: normalization must treat `x = 3`, `3` and `x = 3.0` as
the same answer, and the checker must *decline* on everything that is a
partial-credit judgment, because those belong to the model.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from socratic_tutor.models.enums import Evaluation
from socratic_tutor.pedagogy.math_checker import (
    check_answer,
    has_attempt_content,
    is_simplest_form,
    parse_final_answer,
    parse_number,
)


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("7", Fraction(7)),
        ("-5", Fraction(-5)),
        ("+3", Fraction(3)),
        ("3.0", Fraction(3)),
        ("5.5", Fraction(11, 2)),
        (".5", Fraction(1, 2)),
        ("1/3", Fraction(1, 3)),
        ("6/2", Fraction(3)),
        ("x", None),
        ("", None),
        ("1/0", None),
        ("2x", None),
    ],
)
def test_parse_number(token: str, expected: Fraction | None) -> None:
    assert parse_number(token) == expected


@pytest.mark.parametrize(
    ("attempt", "expected"),
    [
        ("x = 3", Fraction(3)),
        ("x=3", Fraction(3)),
        ("x = 3.0", Fraction(3)),
        ("X = 3", Fraction(3)),
        ("x = -3", Fraction(-3)),
        ("3", Fraction(3)),
        (" 3 ", Fraction(3)),
        ("x = 1/3", Fraction(1, 3)),
        ("y = 4", Fraction(4)),
    ],
)
def test_bare_final_answers_normalize_to_the_same_value(attempt: str, expected: Fraction) -> None:
    _, value = parse_final_answer(attempt)
    assert value == expected


def test_final_answer_is_read_from_the_last_equation() -> None:
    """Shown working followed by a conclusion still resolves to the conclusion."""
    variable, value = parse_final_answer("subtract 4 from both sides: 2x = 6, then x = 3")
    assert (variable, value) == ("x", Fraction(3))

    thai = "หนูลบ 4 ทั้งสองข้าง ได้ 2x = 6 แล้วหาร 2 ได้ x = 3 ค่ะ"
    assert parse_final_answer(thai) == ("x", Fraction(3))


def test_thai_prose_around_a_bare_answer() -> None:
    assert parse_final_answer("ครูครับ ผมได้ x = 4 ครับ") == ("x", Fraction(4))


@pytest.mark.parametrize(
    "attempt",
    [
        "2x = 6",  # coefficient still attached — not a final answer
        "x = 6/2",  # arithmetic left undone — the partial-credit case
        "2x + 6 = 14",
        "I think you subtract 4 first but I'm not sure what happens after",
        "the answer is B",
    ],
)
def test_unfinished_or_prose_attempts_have_no_final_answer(attempt: str) -> None:
    _, value = parse_final_answer(attempt)
    assert value is None


def test_is_simplest_form() -> None:
    assert is_simplest_form("1/3", Fraction(1, 3))
    assert not is_simplest_form("6/2", Fraction(3))
    assert is_simplest_form("3", Fraction(3))
    assert is_simplest_form("3.0", Fraction(3))


# ------------------------------------------------------------- check_answer


def test_checker_confirms_a_matching_bare_answer() -> None:
    result = check_answer("x = 7", "x = 7")
    assert result.verdict is Evaluation.CORRECT
    assert result.is_decisive


@pytest.mark.parametrize(
    ("attempt", "key"),
    [("x = 17", "x = 7"), ("x = 80", "x = 5"), ("x = 1/3", "x = 5"), ("x = 5.5", "x = 4")],
)
def test_checker_rejects_a_mismatching_bare_answer(attempt: str, key: str) -> None:
    assert check_answer(attempt, key).verdict is Evaluation.INCORRECT


def test_formatting_differences_are_not_errors() -> None:
    for attempt in ("x = 3", "x=3", "x = 3.0", "3", "X = 3"):
        assert check_answer(attempt, "x = 3").verdict is Evaluation.CORRECT


@pytest.mark.parametrize(
    "attempt",
    ["2x = 6", "x = 6/2", "2x + 6 = 14", "the answer is B", "idk"],
)
def test_checker_defers_on_everything_that_is_a_judgment_call(attempt: str) -> None:
    """Partial credit and un-gradeable prose belong to the LLM, not to this code."""
    result = check_answer(attempt, "x = 3")
    assert result.verdict is None
    assert result.reason


def test_checker_defers_when_the_variables_differ() -> None:
    result = check_answer("y = 3", "x = 3")
    assert result.verdict is None
    assert "solves for" in result.reason


def test_checker_defers_when_the_key_is_not_a_single_value() -> None:
    result = check_answer("x = 3", "any positive integer")
    assert result.verdict is None
    assert "answer key" in result.reason


def test_checker_never_returns_partial_credit() -> None:
    verdicts = {
        check_answer(attempt, "x = 3").verdict
        for attempt in ("x = 3", "x = 4", "2x = 6", "x = 6/2", "prose")
    }
    assert Evaluation.PARTIALLY_CORRECT not in verdicts


# -------------------------------------------------------- attempt content


@pytest.mark.parametrize("text", ["idk", "I don't know", "?", "", "   ", "ไม่รู้", "no idea"])
def test_no_attempt_content(text: str) -> None:
    assert not has_attempt_content(text)


@pytest.mark.parametrize("text", ["x = 3", "3", "I subtracted 4", "หนูได้ x = 4", "the answer is B"])
def test_has_attempt_content(text: str) -> None:
    assert has_attempt_content(text)
