"""Output Guardrail (R11, evaluation suite E2 — no LLM).

Two things matter here. That an answer written in *any* equivalent form is
caught, in English and in Thai; and that a hard leak and a guardrail save are
never conflated, because E3 counts them separately and the O3 fine-tuning
decision reads the difference (Tech Spec §9).
"""

from __future__ import annotations

import re

import pytest

from socratic_tutor.models.enums import Evaluation, GuardrailEventType, Language
from socratic_tutor.models.plan import GlobalRules, ResponsePlan
from socratic_tutor.pedagogy.guardrail import (
    NUMERIC_FORMS,
    TEXT_FORMS,
    UNDETECTED_FORMS,
    GuardrailAction,
    check_output,
    count_questions,
    count_words,
    coverage_for,
    find_leaks,
    fold_number_words,
    normalise,
)
from socratic_tutor.pedagogy.math_checker import check_answer
from socratic_tutor.pedagogy.tables import GuardrailConfig, RuleTables, load_tables

CANONICAL = "x = 7"


@pytest.fixture(scope="module")
def tables() -> RuleTables:
    return load_tables()


@pytest.fixture
def config(tables: RuleTables) -> GuardrailConfig:
    return tables.guardrail


@pytest.fixture
def rules(tables: RuleTables) -> GlobalRules:
    return tables.global_rules


@pytest.fixture
def hint_plan() -> ResponsePlan:
    """A hint turn: the solution is withheld, so the scan is armed."""
    return ResponsePlan(
        structure=["single_hint", "student_attempt_prompt"],
        max_words=50,
        max_questions=1,
        full_solution_allowed=False,
    )


def check(text: str, plan: ResponsePlan, config, rules, **kwargs):
    return check_output(text, plan, turn_id="turn-1", config=config, rules=rules, **kwargs)


# --- normalisation --------------------------------------------------------


def test_normalise_folds_case_punctuation_and_thai_digits() -> None:
    assert normalise("X = 7!") == "x = 7"
    assert normalise("๗") == "7"
    # Spacing is folded one level further down, by the compact comparison, so
    # `=` survives normalisation and formula shapes stay comparable.
    assert normalise("x=7") == "x=7"
    assert find_leaks("x=7", "X = 7", source="c")


@pytest.mark.parametrize(
    ("draft", "form"),
    [
        ("So the answer is x = 7.", "exact"),
        ("so X=7", "normalised"),
        ("Therefore x  =  7 exactly", "normalised"),
        ("x = ๗", "normalised"),
        ("It comes out to 7.", "numeric_token"),
        ("the result is 7.0", "numeric_token"),
        ("คำตอบคือ x = 7", "exact"),
        ("คำตอบคือ x = ๗", "normalised"),
    ],
)
def test_every_equivalent_form_of_the_answer_is_caught(draft: str, form: str) -> None:
    leaks = find_leaks(draft, CANONICAL, source="canonical_answer")
    assert leaks, draft
    assert leaks[0].form == form


def test_a_clean_hint_is_not_a_leak() -> None:
    assert find_leaks("What operation is attached to x?", CANONICAL, source="c") == []
    assert find_leaks("ลองดูว่ามีอะไรติดอยู่กับตัวแปร", CANONICAL, source="c") == []


def test_an_empty_secret_or_draft_is_never_a_leak() -> None:
    assert find_leaks("anything", "", source="c") == []
    assert find_leaks("", CANONICAL, source="c") == []


def test_formula_strings_are_caught_as_well_as_numbers() -> None:
    leaks = find_leaks(
        "just subtract: x = (10 - 4) / 2", "x = (10 - 4)/2", source="canonical_answer"
    )
    assert leaks and leaks[0].form == "normalised"


@pytest.mark.parametrize(
    ("draft", "secret"),
    [
        ("you get 1/3 of the total", "x = 1/3"),
        ("that reduces to 2/6", "x = 1/3"),  # the same rational value
        ("the result is 7.0", "x = 7"),
    ],
)
def test_numeric_equivalence_comes_from_the_shared_answer_parser(draft: str, secret: str) -> None:
    """`math_checker.parse_number` is the single definition of "same number".

    Sharing it with Call B's deterministic check buys exact rational comparison:
    a draft writing 2/6 leaks an answer of 1/3, which float matching would miss.
    """
    leaks = find_leaks(draft, secret, source="canonical_answer")
    assert leaks and leaks[0].form in {"numeric_token", "final_answer"}


# --- the cross-layer safety property -------------------------------------

#: Renderings of the answer to P03 (2x + 4 = 10, answer x = 3). A mix the
#: evaluator grades and forms it declines to grade, so the property below is
#: exercised on both sides rather than passing vacuously.
RENDERINGS = [
    "x = 3",
    "x=3",
    "X = 3",
    "(x) = 3",
    "3",
    "3.0",
    "+3",
    "3/1",
    "๓",
    "x = ๓",
    "6/2",
    "so after dividing, x = 3",
    "คำตอบคือ x = 3",
    "คำตอบคือ ๓",
]


@pytest.mark.parametrize("rendering", RENDERINGS)
def test_every_evaluator_equivalent_rendering_is_detected_as_a_leak(rendering: str) -> None:
    """The safety seam between the two layers that both reason about answers.

    If Call B would accept a rendering as *being* the canonical answer, the
    guardrail must recognise the same rendering as leaking it. Otherwise a draft
    can say the answer in a form one layer understands and the other does not,
    and the leak gate passes while the answer is on screen.

    One-directional by design: the guardrail may flag more than the evaluator
    grades (it does — see the Thai and unreduced-fraction cases), because a false
    positive costs a regeneration and a miss costs the guarantee.
    """
    evaluator_says_same = check_answer(rendering, "x = 3").verdict is Evaluation.CORRECT
    guardrail_detects = bool(find_leaks(rendering, "x = 3", source="canonical_answer"))
    if evaluator_says_same:
        assert guardrail_detects, (
            f"{rendering!r}: Call B grades this as the answer but the guardrail "
            "would let it through"
        )


def test_the_property_is_not_vacuous() -> None:
    """Guards the test above: the evaluator must actually accept some of them."""
    accepted = [r for r in RENDERINGS if check_answer(r, "x = 3").verdict is Evaluation.CORRECT]
    assert len(accepted) >= 8, accepted


def test_the_guardrail_catches_strictly_more_than_the_evaluator_grades() -> None:
    """Every rendering is caught, including those Call B declines to grade."""
    missed = [r for r in RENDERINGS if not find_leaks(r, "x = 3", source="canonical_answer")]
    assert missed == []


def test_a_final_answer_is_reported_as_more_than_a_stray_number() -> None:
    """`parse_final_answer` knows which variable took the value; a token does not.

    "x = 1/2" is not a substring of the key "x = 0.5" in any folding, so only
    reading the draft's final answer the way Call B does recognises it.
    """
    leaks = find_leaks("so it works out to x = 1/2", "x = 0.5", source="canonical_answer")
    assert leaks and leaks[0].form == "final_answer"
    assert leaks[0].secret_matched == "x = 1/2"


def test_a_value_assigned_to_another_variable_is_not_a_final_answer_match() -> None:
    """Reported as a loose token instead, because that is what it is."""
    leaks = find_leaks("the coefficient y = 0.5 here", "x = 0.5", source="canonical_answer")
    assert leaks and leaks[0].form == "numeric_token"


# --- telemetry must not publish the secret --------------------------------
#
# A guardrail that names the value it suppressed hands the answer to the student
# every time it fires: `GuardrailEvent.detail` reaches the browser in the trace
# frames and is persisted to the session file on disk. Each save becomes a leak
# through a channel no leak counter watches, and a student only has to rephrase
# until some draft trips it.


@pytest.mark.parametrize("rendering", RENDERINGS)
def test_no_leak_form_publishes_the_answer_in_its_event(
    rendering: str, hint_plan, config, rules
) -> None:
    """The generalising assertion, over every form the guardrail can detect.

    Serialised, because that is the shape the value would escape in — the field
    could be nested anywhere in the event and still reach the wire.
    """
    result = check(rendering, hint_plan, config, rules, canonical_answer="x = 3", problem_id="P03")
    if not result.leaks:
        pytest.skip(f"{rendering!r} is not detected as a leak")

    for event in result.events:
        serialised = event.model_dump_json()
        assert "x = 3" not in serialised, f"{rendering!r}: event names the answer"
        assert "x=3" not in serialised.replace(" ", "")
        for secret in {leak.secret_matched for leak in result.leaks}:
            # As a standalone token: a bare "3" is a substring of the problem id
            # P03 and of any timestamp, and neither of those discloses anything.
            standalone = re.compile(rf"(?<![\w.]){re.escape(secret)}(?![\w.])")
            assert not standalone.search(serialised), f"{rendering!r}: event carries {secret!r}"


def test_the_event_detail_still_tells_an_operator_what_happened() -> None:
    """Redaction must not cost diagnosability, or it gets reverted."""
    leak = find_leaks("the answer is x = 3", "x = 3", source="canonical_answer", problem_id="P03")[
        0
    ]
    detail = leak.describe()
    assert "canonical_answer" in detail  # which secret
    assert "exact" in detail  # which equivalence fired
    assert "P03" in detail  # which problem
    assert "x = 3" not in detail  # but not the value


def test_the_detail_does_not_disclose_the_length_of_the_answer() -> None:
    """A char count is information about the withheld value and buys nothing.

    `form` already separates a full-answer match from a bare numeric one, which
    is the only distinction the count would have made.
    """
    for secret in ("x = 3", "x = 12345"):
        detail = find_leaks(
            f"the answer is {secret}", secret, source="canonical_answer", problem_id="P03"
        )[0].describe()
        assert str(len(secret)) not in detail
    # ...and the two details are identical apart from nothing: length-free.
    short = find_leaks("the answer is x = 3", "x = 3", source="c", problem_id="P03")[0]
    long = find_leaks("the answer is x = 12345", "x = 12345", source="c", problem_id="P03")[0]
    assert short.describe() == long.describe()


def test_the_secret_stays_available_in_process_for_scoring() -> None:
    """E3 needs the value to score; only the wire-bound path is redacted."""
    leak = find_leaks("the answer is x = 3", "x = 3", source="canonical_answer")[0]
    assert leak.secret_matched == "x = 3"


def test_a_fallback_served_event_does_not_name_the_answer_either(hint_plan, config, rules) -> None:
    """The second-attempt path builds its own events; both are checked."""
    result = check(
        "x = 3", hint_plan, config, rules, canonical_answer="x = 3", problem_id="P03", attempt=2
    )
    assert result.action is GuardrailAction.SERVE_FALLBACK
    for event in result.events:
        assert "x = 3" not in event.model_dump_json()


def test_the_quiz_key_is_redacted_the_same_way(config, rules) -> None:
    plan = ResponsePlan(
        structure=["single_practice_question"], max_words=60, full_solution_allowed=True
    )
    result = check("Solve it. The answer is x = 3.", plan, config, rules, quiz_answer_key="x = 3")
    assert result.leaks
    for event in result.events:
        assert "x = 3" not in event.model_dump_json()


# --- the three checks -----------------------------------------------------


def test_a_clean_draft_passes_untouched(hint_plan, config, rules) -> None:
    draft = "What operation is attached to x? Try undoing it."
    result = check(draft, hint_plan, config, rules, canonical_answer=CANONICAL)
    assert result.action is GuardrailAction.PASS
    assert result.text == draft
    assert result.leaks == ()
    assert result.is_guardrail_save is False
    assert result.hard_leak is False


def test_a_leak_regenerates_once_then_serves_the_fallback(hint_plan, config, rules) -> None:
    first = check(
        "The answer is x = 7.", hint_plan, config, rules, canonical_answer=CANONICAL, attempt=1
    )
    assert first.action is GuardrailAction.REGENERATE
    assert first.regeneration_note == config.regeneration_note
    assert first.text == "The answer is x = 7."  # unchanged; the caller retries
    assert [event.type for event in first.events] == [GuardrailEventType.REGENERATED]

    second = check("Fine — x = 7.", hint_plan, config, rules, canonical_answer=CANONICAL, attempt=2)
    assert second.action is GuardrailAction.SERVE_FALLBACK
    assert second.text == config.fallback_for("en")
    assert [event.type for event in second.events] == [
        GuardrailEventType.LEAK_BLOCKED,
        GuardrailEventType.FALLBACK_SERVED,
    ]


def test_the_served_fallback_cannot_itself_leak(hint_plan, config, rules) -> None:
    served = check("x = 7", hint_plan, config, rules, canonical_answer=CANONICAL, attempt=2).text
    assert find_leaks(served, CANONICAL, source="c") == []


def test_the_thai_fallback_is_served_for_a_thai_turn(config, rules) -> None:
    plan = ResponsePlan(
        structure=["single_hint"], max_words=50, full_solution_allowed=False, language=Language.TH
    )
    result = check("x = 7", plan, config, rules, canonical_answer=CANONICAL, attempt=2)
    assert result.text == config.fallback_for("th")
    assert result.text != config.fallback_for("en")


def test_no_scan_when_the_full_solution_is_allowed(config, rules) -> None:
    """A permitted answer is not a leak; the scan is armed by the plan."""
    plan = ResponsePlan(structure=["direct_answer"], max_words=120, full_solution_allowed=True)
    result = check("The answer is x = 7.", plan, config, rules, canonical_answer=CANONICAL)
    assert result.action is GuardrailAction.PASS
    assert result.is_guardrail_save is False


def test_quiz_answer_key_is_scanned_before_the_student_attempts(config, rules) -> None:
    plan = ResponsePlan(
        structure=["single_practice_question"], max_words=60, full_solution_allowed=True
    )
    result = check("Solve 2x + 4 = 10. (It's 3.)", plan, config, rules, quiz_answer_key="x = 3")
    assert result.action is GuardrailAction.REGENERATE
    assert result.leaks[0].source == "quiz_answer_key"


def test_quiz_key_scan_respects_the_global_rule(config) -> None:
    permissive = GlobalRules(quiz_answer_before_student_attempt=True)
    plan = ResponsePlan(
        structure=["single_practice_question"], max_words=60, full_solution_allowed=True
    )
    result = check(
        "Solve it. The answer is x = 3.", plan, config, permissive, quiz_answer_key="x = 3"
    )
    assert result.action is GuardrailAction.PASS


# --- counts are logged, never blocked ------------------------------------


def test_word_count_over_tolerance_is_logged_and_served(hint_plan, config, rules) -> None:
    draft = " ".join(["word"] * 100)  # plan allows 50, tolerance 25%
    result = check(draft, hint_plan, config, rules)
    assert result.action is GuardrailAction.PASS
    assert result.text == draft
    assert [event.type for event in result.events] == [GuardrailEventType.PLAN_VIOLATION]
    assert "100 words" in result.events[0].detail


def test_word_count_inside_tolerance_is_not_logged(hint_plan, config, rules) -> None:
    result = check(" ".join(["word"] * 60), hint_plan, config, rules)  # 60 <= 50 * 1.25
    assert result.events == ()


def test_question_count_over_tolerance_is_logged(hint_plan, config, rules) -> None:
    result = check("Why? How? When? What?", hint_plan, config, rules)
    assert result.action is GuardrailAction.PASS
    assert any(event.type is GuardrailEventType.PLAN_VIOLATION for event in result.events)


def test_one_thai_question_is_not_a_plan_violation(hint_plan, config, rules) -> None:
    """Thai asks with a final particle and no "?"; two particles is still one question."""
    assert count_questions("คุณคิดว่าควรทำอะไรก่อนไหม") == 1
    result = check("คุณคิดว่าควรทำอะไรก่อนไหม", hint_plan, config, rules)
    assert result.events == ()


def test_thai_words_are_estimated_not_counted_as_one(hint_plan, config, rules) -> None:
    """Whitespace splitting would score a whole Thai paragraph as a few words."""
    thai = "มาลองแก้ไปด้วยกันทีละขั้น ขั้นแรกคุณจะลองทำอะไร"
    assert count_words(thai) > 2
    assert count_words("one two three") == 3


def test_a_count_violation_never_changes_the_action(hint_plan, config, rules) -> None:
    """Verbosity is a prompt-tuning signal, not a safety failure (§6.3)."""
    result = check(" ".join(["word"] * 400), hint_plan, config, rules, canonical_answer=CANONICAL)
    assert result.action is GuardrailAction.PASS


# --- E3 accounting --------------------------------------------------------


def test_a_save_is_counted_as_a_save_and_not_as_a_leak(hint_plan, config, rules) -> None:
    result = check("x = 7", hint_plan, config, rules, canonical_answer=CANONICAL)
    assert result.is_guardrail_save is True
    assert result.hard_leak is False


@pytest.mark.parametrize(
    "draft",
    [
        "The answer is x = 7.",
        "x=7",
        "x = ๗",
        "It's 7.",
        "คำตอบคือ x = ๗",
        "What operation is attached to x?",
        "",
    ],
)
@pytest.mark.parametrize("attempt", [1, 2, 3])
def test_a_hard_leak_is_impossible_by_construction(
    hint_plan, config, rules, draft: str, attempt: int
) -> None:
    """The invariant E3 asserts: text that leaks is never returned with PASS."""
    result = check(draft, hint_plan, config, rules, canonical_answer=CANONICAL, attempt=attempt)
    assert result.hard_leak is False
    if result.action is GuardrailAction.PASS:
        assert find_leaks(result.text, CANONICAL, source="c") == []


def test_events_carry_the_detail_a_review_needs(hint_plan, config, rules) -> None:
    result = check("x = ๗", hint_plan, config, rules, canonical_answer=CANONICAL)
    assert result.turn_id == "turn-1"
    event = result.events[0]
    assert event.type is GuardrailEventType.REGENERATED
    assert "canonical_answer" in event.detail
    assert "normalised" in event.detail
    assert event.action_taken
    assert result.events_for_trace() == list(result.events)


def test_regeneration_budget_is_configurable(hint_plan, rules, config) -> None:
    """`max_regenerations: 0` means a leak goes straight to the fallback."""
    strict = GuardrailConfig(
        count_tolerance=config.count_tolerance,
        max_regenerations=0,
        regeneration_note=config.regeneration_note,
        fallback_text=config.fallback_text,
    )
    result = check("x = 7", hint_plan, strict, rules, canonical_answer=CANONICAL)
    assert result.action is GuardrailAction.SERVE_FALLBACK


# --- spelled-out numbers ---------------------------------------------------


@pytest.mark.parametrize(
    ("draft", "secret"),
    [
        ("Fine — x is three.", "x = 3"),
        ("the answer is seven", "x = 7"),
        ("it works out to twelve", "x = 12"),
        ("ก็ได้ — x เท่ากับสาม", "x = 3"),
        ("คำตอบคือเจ็ด", "x = 7"),
        ("x เท่ากับสิบสอง", "x = 12"),
    ],
)
def test_a_number_spelled_out_is_a_leak(draft: str, secret: str) -> None:
    """A model told not to state a number writes the word instead.

    Measured on the E3 scripts: before this, a tutor caving in words scored zero
    on every leak counter across all 58 turns.
    """
    leaks = find_leaks(draft, secret, source="canonical_answer")
    assert leaks, draft
    assert leaks[0].form == "numeric_token"


def test_thai_number_words_are_folded_longest_first() -> None:
    """ "สิบสอง" (12) contains "สิบ" (10); Thai has no boundaries to separate them."""
    assert fold_number_words("สิบสอง") == "12"
    assert fold_number_words("สิบ") == "10"
    assert find_leaks("x เท่ากับสิบสอง", "x = 12", source="c")
    assert not find_leaks("x เท่ากับสิบสอง", "x = 10", source="c")


def test_english_folding_respects_word_boundaries() -> None:
    """Otherwise "money" contains "one" and every draft leaks the answer 1."""
    assert fold_number_words("money matters") == "money matters"
    assert find_leaks("money matters", "x = 1", source="c") == []
    assert fold_number_words("one side") == "1 side"


def test_the_answer_key_is_never_word_folded() -> None:
    """Asymmetric by design, and this is the case that forces it.

    The prose key P07 contains "changing one side". Folding the *key* would make
    its "one" the number 1, so every draft mentioning 1 would read as leaking a
    key that has no numeric answer at all.
    """
    prose = (
        "Because an equation says both sides are equal, so changing one side "
        "without changing the other breaks that equality."
    )
    assert find_leaks("you get 1 after dividing", prose, source="canonical_answer") == []
    assert find_leaks("the first step is one you already know", prose, source="c") == []


def test_unevaluated_arithmetic_remains_undetected() -> None:
    """A known boundary, pinned so it is not mistaken for coverage.

    Catching this needs the scanner to evaluate an expression rather than
    recognise a value; closing it is the embedding-similarity decision.
    """
    assert find_leaks("it's ten minus seven", "x = 3", source="c") == []


def test_a_reworded_prose_answer_remains_undetected() -> None:
    """The other known boundary: substring matching loses to paraphrase."""
    prose = "Because an equation says both sides are equal."
    reworded = "An equation is a claim that the two sides have the same value."
    assert find_leaks(reworded, prose, source="c") == []


# --- scan coverage ---------------------------------------------------------
#
# The trace has to be able to say "checked these forms and found nothing" rather
# than just "found nothing", and the difference between a clean scan and no scan
# at all is the single thing every disarm bug in this build hid behind.


def test_coverage_for_a_numeric_key_lists_every_numeric_form() -> None:
    coverage = coverage_for("x = 3")
    assert coverage.scanned is True
    assert set(TEXT_FORMS) <= set(coverage.forms)
    assert set(NUMERIC_FORMS) <= set(coverage.forms)
    assert set(UNDETECTED_FORMS) <= set(coverage.not_detected)


def test_coverage_for_a_prose_key_reports_only_substring_matching() -> None:
    """Against prose the scanner degrades, and the trace must say so."""
    coverage = coverage_for("Because both sides of an equation are equal.")
    assert coverage.scanned is True
    assert coverage.forms == TEXT_FORMS
    for form in NUMERIC_FORMS:
        assert form in coverage.not_detected


def test_coverage_with_no_key_reports_nothing_scanned() -> None:
    """The case that matters most: no answer resolved is not a clean result."""
    for empty in (None, "", "   "):
        coverage = coverage_for(empty)
        assert coverage.scanned is False
        assert coverage.forms == ()
        assert "nothing scanned" in coverage.describe()


def test_coverage_reaches_the_result_only_for_scans_that_ran(hint_plan, config, rules) -> None:
    scanned = check("a clean hint", hint_plan, config, rules, canonical_answer="x = 3")
    assert scanned.coverage.scanned is True

    no_key = check("a clean hint", hint_plan, config, rules)
    assert no_key.coverage.scanned is False

    permissive = ResponsePlan(structure=["direct_answer"], full_solution_allowed=True)
    gated_off = check("anything", permissive, config, rules, canonical_answer="x = 3")
    assert gated_off.coverage.scanned is False, "a gated-off check must not report coverage"


def test_coverage_merges_across_both_secrets(config, rules) -> None:
    plan = ResponsePlan(structure=["single_practice_question"], full_solution_allowed=False)
    result = check(
        "a clean question", plan, config, rules, canonical_answer="x = 3", quiz_answer_key="x = 5"
    )
    assert result.coverage.scanned is True
    assert len(result.coverage.forms) == len(set(result.coverage.forms)), "no duplicates"


def test_coverage_never_contains_a_secret() -> None:
    """It describes the detector, not the data."""
    coverage = coverage_for("x = 3")
    assert "3" not in coverage.describe()
    assert "3" not in str(coverage.as_dict())


def test_coverage_serialises_for_the_trace() -> None:
    payload = coverage_for("x = 3").as_dict()
    assert payload["scanned"] is True
    assert isinstance(payload["forms"], list)
    assert isinstance(payload["not_detected"], list)
