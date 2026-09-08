"""Output Guardrail — step 6 of the pipeline (Tech Spec §6, requirement R11).

Deterministic, post-generation, before the student sees any text. No LLM: this
is the last layer under the leak guarantee, and a layer that can be argued with
is not a guarantee.

Three checks (§6):

1. `full_solution_allowed = false` -> scan the draft for the withheld canonical
   answer, exact and normalised. A match regenerates once with an appended
   system note; a second match serves the template fallback and logs.
2. The same scan against a quiz item's answer key when a practice question is
   in flight and `quiz_answer_before_student_attempt` is false.
3. Word and question counts over the plan's limits by more than the configured
   tolerance are logged and **served anyway** — a verbose answer is a prompt-tuning
   signal, not a safety failure.

Numeric equivalence is `math_checker.parse_number`, the same parser Call B's
deterministic answer check uses (§5.2), so the two cannot disagree about whether
two numbers are the same one. The text-surface folding below (case, Thai digits,
spelled-out numbers, spacing, punctuation) has no equivalent in `math_checker`,
which reads a bare final answer rather than searching prose, so it lives here.

**Known boundary, deliberately not closed.** This is a *numeric-answer*
guardrail. It sees digits and number words in both languages; it does not
evaluate arithmetic the student would finish ("ten minus seven"), and against a
prose answer key it degrades to substring matching, which any rewording defeats.
Closing those needs embedding similarity — a real dependency, a latency cost on
the buffered path, and a probabilistic control where every other layer here is
deterministic. That trade is its own decision, not a drive-by. E3 measures the
gap on every run and labels its own scope accordingly.

**Hard leaks and guardrail saves are counted separately** (Tech Spec §9, E3).
A save means this layer worked and the layers above it did not; a hard leak means
the answer reached the student. Conflating them would hide exactly the signal the
O3 fine-tuning decision reads. `GuardrailResult.is_guardrail_save` and
`.hard_leak` are the two counters, and `hard_leak` is designed to be provably
always False — see `test_guardrail.py`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Any

from socratic_tutor.models.enums import GuardrailEventType
from socratic_tutor.models.plan import GlobalRules, ResponsePlan
from socratic_tutor.models.trace import GuardrailEvent
from socratic_tutor.pedagogy.math_checker import parse_final_answer, parse_number
from socratic_tutor.pedagogy.tables import GuardrailConfig

#: Thai digits ๐-๙ map to ASCII so "x = ๗" and "x = 7" are the same answer.
THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

#: Thai has no word delimiters. Without a segmenter (not a dependency) this
#: estimates one word per five Thai characters. It only ever feeds the >25%
#: tolerance log in check 3, never a block, so the estimate cannot cause a
#: wrong safety decision. Swap in a real segmenter if the counts start mattering.
THAI_CHARS_PER_WORD = 5

_THAI_RANGE = re.compile(r"[฀-๿]")
#: Numeric tokens, fractions first so `1/3` is read whole rather than as 1 and 3.
#: Each match is resolved by `math_checker.parse_number`, so the guardrail and
#: Call B's deterministic checker agree on what a number is.
_NUMBER = re.compile(r"-?\d+\s*/\s*\d+|-?\d+(?:\.\d+)?")
_WHITESPACE = re.compile(r"\s+")
#: Characters that carry meaning in an answer and must survive normalisation.
_KEEP = re.compile(r"[^0-9a-z฀-๿=+\-*/^().,]")
_QUESTION_MARK = re.compile(r"[?？]")
#: Sentence boundaries for the Thai question count. Thai does not end sentences
#: with a full stop; it separates them with a space or a line break.
_SENTENCE_SPLIT = re.compile(r"[\n.!?]+|\s{2,}")
#: Thai yes/no and wh-question particles, for drafts that ask without a "?".
_THAI_QUESTION_PARTICLES = ("ไหม", "หรือไม่", "อะไร", "ทำไม", "อย่างไร", "เท่าไร")

#: Spelled-out numbers a caving tutor actually writes. Bounded at twenty because
#: seed answers are small integers and every entry is a false-positive surface:
#: "one" is a common English word, so the range is kept to what the answer keys
#: can plausibly contain rather than extended for completeness.
NUMBER_WORDS_EN: dict[str, str] = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}
NUMBER_WORDS_TH: dict[str, str] = {
    "ศูนย์": "0",
    "หนึ่ง": "1",
    "เอ็ด": "1",
    "สอง": "2",
    "สาม": "3",
    "สี่": "4",
    "ห้า": "5",
    "หก": "6",
    "เจ็ด": "7",
    "แปด": "8",
    "เก้า": "9",
    "สิบ": "10",
    "สิบเอ็ด": "11",
    "สิบสอง": "12",
    "ยี่สิบ": "20",
}
#: Longest first: "สิบสอง" (12) contains "สิบ" (10), and Thai has no word
#: boundaries to disambiguate them.
_THAI_NUMBER_WORDS: tuple[tuple[str, str], ...] = tuple(
    sorted(NUMBER_WORDS_TH.items(), key=lambda item: -len(item[0]))
)
_ENGLISH_NUMBER_WORDS = re.compile(
    r"\b(?:" + "|".join(sorted(NUMBER_WORDS_EN, key=len, reverse=True)) + r")\b"
)


class GuardrailAction(str, Enum):
    """What the orchestrator must do with the draft."""

    #: Serve the draft as generated.
    PASS = "pass"
    #: Re-run Call C with `regeneration_note` appended, then check again.
    REGENERATE = "regenerate"
    #: Regeneration already failed; serve `GuardrailResult.text` instead.
    SERVE_FALLBACK = "serve_fallback"


@dataclass(frozen=True)
class LeakMatch:
    """One way the withheld answer showed up in the draft.

    `form` records *how* it matched, which is what makes a leak report
    actionable: an `exact` match is a blunt failure of the prompt, while a
    `numeric_token` match may be a false positive worth reviewing.

    **`secret_matched` holds the withheld value itself and must never cross a
    serialisation boundary.** It is here so the E3 harness can score a run in
    process; it is not telemetry. Anything that leaves this process — a
    `GuardrailEvent`, an SSE frame, a persisted `TurnRecord` — carries
    `describe()` instead.
    """

    form: str
    secret_matched: str
    source: str
    #: Answer-key id, when the caller knows it. A handle, not the value.
    problem_id: str = ""

    def describe(self) -> str:
        """A non-reversible summary, safe to publish.

        Deliberately omits the matched text. `detail` reaches the browser
        through the trace frames and the session file on disk, so a guardrail
        that named the value it suppressed would hand the answer to the student
        every time it fired — turning each save into a leak through a channel no
        leak counter watches.

        What survives is what an operator actually needs: which equivalence
        fired, which secret it came from, and a handle to look the problem up
        server-side.

        The match *length* is deliberately not included, though it was the
        obvious next field. It is information about the withheld value, and
        `form` already carries the diagnostic distinction it would have made —
        `exact` and `normalised` mean the full answer, `numeric_token` means a
        bare value. An operator with the problem id can read the length from the
        answer key; a student in the inspector should not be able to.
        """
        handle = f" ({self.problem_id})" if self.problem_id else ""
        return f"{self.source} matched as {self.form}{handle}"


#: Forms that work on any secret: substring, then the same with case, spacing,
#: Thai digits and decorative punctuation folded away.
TEXT_FORMS: tuple[str, ...] = ("exact", "normalised")
#: Forms that only apply when the secret contains a number to compare against.
NUMERIC_FORMS: tuple[str, ...] = (
    "final_answer",
    "numeric_token",
    "number_words_en",
    "number_words_th",
)
#: Known blind spots of the deterministic scanner, whatever the secret. Named so
#: a trace can say what was *not* checked rather than implying it checked
#: everything. Closing these is the embedding-similarity decision (module docs).
UNDETECTED_FORMS: tuple[str, ...] = ("unevaluated_arithmetic", "reworded_paraphrase")


@dataclass(frozen=True)
class ScanCoverage:
    """What the scanner actually looked for on one call.

    The applicable scope depends on the secret, not just on the detector: a
    numeric key is compared across digits, values and number words in both
    languages, while a prose key degrades to substring matching, and when no key
    resolved nothing was checked at all. A static list of form names could not
    tell those three apart, and the difference is exactly what a trace has to be
    honest about — "checked these forms and found nothing" is a very different
    statement from "found nothing".

    This is the single source of truth for the scanner's scope. Anything that
    describes what the guardrail covers — the turn trace, the E3 report's
    `measurement_scope` — derives from here rather than restating it, because
    two hand-maintained descriptions of one fact drift apart.
    """

    forms: tuple[str, ...] = ()
    not_detected: tuple[str, ...] = ()

    @property
    def scanned(self) -> bool:
        """False when nothing was checked, e.g. no answer key resolved."""
        return bool(self.forms)

    def describe(self) -> str:
        """One line, safe to publish: form names only, never a secret."""
        if not self.scanned:
            return "nothing scanned: no answer key was available for this turn"
        covered = ", ".join(self.forms)
        missed = ", ".join(self.not_detected)
        return (
            f"scanned for {covered}; not detected: {missed}" if missed else f"scanned for {covered}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "forms": list(self.forms),
            "not_detected": list(self.not_detected),
        }

    def merged_with(self, other: ScanCoverage) -> ScanCoverage:
        """Union, preserving declaration order, for a call scanning two secrets."""

        def ordered(*groups: tuple[str, ...]) -> tuple[str, ...]:
            seen: list[str] = []
            for group in groups:
                seen.extend(name for name in group if name not in seen)
            return tuple(seen)

        return ScanCoverage(
            forms=ordered(self.forms, other.forms),
            not_detected=ordered(self.not_detected, other.not_detected),
        )


def coverage_for(secret: str | None) -> ScanCoverage:
    """The forms `find_leaks` would apply to this secret.

    Derived by asking the same questions `find_leaks` asks, so the two cannot
    disagree: the numeric forms are reported only when the secret actually holds
    a number for them to match against.
    """
    if not secret or not secret.strip():
        return ScanCoverage()

    _, final_value = parse_final_answer(secret)
    has_numbers = bool(_numbers(secret))
    if not has_numbers:
        # A prose key: substring matching plus a whitespace fold, and nothing else.
        return ScanCoverage(
            forms=TEXT_FORMS,
            not_detected=(*NUMERIC_FORMS, *UNDETECTED_FORMS),
        )

    numeric = tuple(
        name for name in NUMERIC_FORMS if name != "final_answer" or final_value is not None
    )
    missing = tuple(name for name in NUMERIC_FORMS if name not in numeric)
    return ScanCoverage(forms=(*TEXT_FORMS, *numeric), not_detected=(*missing, *UNDETECTED_FORMS))


@dataclass(frozen=True)
class GuardrailResult:
    """The guardrail's verdict on one draft."""

    action: GuardrailAction
    text: str
    events: tuple[GuardrailEvent, ...]
    turn_id: str
    attempt: int
    leaks: tuple[LeakMatch, ...] = ()
    regeneration_note: str | None = None
    #: What was actually scanned for. Empty when no secret was supplied — which
    #: is the case a trace most needs to distinguish from a clean result.
    coverage: ScanCoverage = ScanCoverage()

    @property
    def is_guardrail_save(self) -> bool:
        """A withheld answer was caught before the student saw it.

        Counted separately from a hard leak (Tech Spec §9): this layer working
        is also the layers above it failing, and E3 reads both numbers.
        """
        return bool(self.leaks)

    @property
    def hard_leak(self) -> bool:
        """The served text still contains the withheld answer.

        Always False by construction: a leaking draft is never returned with
        `PASS`. Exposed so the eval harness can assert the invariant instead of
        assuming it.
        """
        return bool(self.leaks) and self.action is GuardrailAction.PASS

    def events_for_trace(self) -> list[GuardrailEvent]:
        """The events to append to this turn's `TurnTrace.guardrail_events`.

        The turn id lives on the `TurnTrace` that owns them; `turn_id` here is
        for standalone structured logging outside a trace.
        """
        return list(self.events)


def normalise(text: str) -> str:
    """Fold a string to the form leak comparison happens in.

    Case, Unicode composition, Thai digits, whitespace and decorative
    punctuation all differ between "the answer is x = 7" and "X=๗" without
    either being a different answer.
    """
    folded = unicodedata.normalize("NFKC", text).casefold().translate(THAI_DIGITS)
    folded = _KEEP.sub(" ", folded)
    return _WHITESPACE.sub(" ", folded).strip()


def _compact(text: str) -> str:
    """Normalised with all spaces removed, so `x = 7` and `x=7` compare equal."""
    return normalise(text).replace(" ", "")


def fold_number_words(text: str) -> str:
    """Rewrite spelled-out numbers as digits, English and Thai.

    "the answer is x = 3" and "x is three" are the same disclosure, and a model
    told not to state a number reaches for the word far more readily than for an
    elaborate paraphrase. Measured on the E3 scripts, a tutor caving in words
    scored zero on every leak counter before this existed.

    Applied to the **draft only**, never to the answer key — see `_numbers`.
    Thai is replaced longest-first because its number words nest: "สิบสอง" (12)
    contains "สิบ" (10), so the short key would win and turn twelve into ten-two.
    English uses word boundaries so "one" does not fire inside "money".
    """
    folded = text
    for word, digit in _THAI_NUMBER_WORDS:
        folded = folded.replace(word, digit)
    return _ENGLISH_NUMBER_WORDS.sub(lambda m: NUMBER_WORDS_EN[m.group(0)], folded)


def _numbers(text: str, *, fold_words: bool = False) -> list[Fraction]:
    """Numeric tokens as exact values, via Call B's parser (§5.2).

    Sharing `math_checker.parse_number` means the guardrail and the answer
    checker cannot disagree about whether two numbers are the same one, and the
    guardrail gets exact rational comparison for free: a draft writing `2/6`
    leaks an answer of `1/3`, which float comparison would have missed.

    `fold_words` is deliberately asymmetric — set for the draft, never for the
    secret. Answer keys are authored data and write numbers as digits; drafts are
    model prose and may spell them out. Folding the key too would be actively
    harmful: the prose key P07 contains the phrase "changing one side", so its
    "one" would become the number 1 and every draft mentioning 1 would read as a
    leak of a key that has no numeric answer at all.
    """
    prepared = fold_number_words(normalise(text)) if fold_words else normalise(text)
    values = []
    for raw in _NUMBER.findall(prepared):
        value = parse_number(raw)
        if value is not None:
            values.append(value)
    return values


def find_leaks(text: str, secret: str, *, source: str, problem_id: str = "") -> list[LeakMatch]:
    """Every way `secret` appears in `text`.

    Ordered most to least conclusive: an exact substring; the same string with
    spacing and script differences folded away; the draft's own *final answer*
    read by Call B's parser; and last the bare numeric value anywhere in the
    text. The numeric check is what catches a result restated in passing after a
    hint was supposed to withhold it; it is also the check most likely to fire on
    a coincidence, which is why `form` records it separately for review.

    The search only ever widens: a false positive costs one regeneration, a miss
    costs the leak guarantee.
    """
    secret = secret.strip()
    if not secret or not text.strip():
        return []

    matches: list[LeakMatch] = []
    if secret in text:
        matches.append(
            LeakMatch(form="exact", secret_matched=secret, source=source, problem_id=problem_id)
        )
        return matches

    compact_secret = _compact(secret)
    if compact_secret and compact_secret in _compact(text):
        matches.append(
            LeakMatch(
                form="normalised", secret_matched=secret, source=source, problem_id=problem_id
            )
        )
        return matches

    # The draft's final answer, read the way Call B reads a student attempt.
    # More conclusive than a loose numeric token because it knows which variable
    # the value was assigned to.
    secret_variable, secret_value = parse_final_answer(secret)
    if secret_value is not None:
        draft_variable, draft_value = parse_final_answer(text)
        same_variable = (
            secret_variable is None
            or draft_variable is None
            or secret_variable.casefold() == draft_variable.casefold()
        )
        if draft_value == secret_value and same_variable:
            matches.append(
                LeakMatch(
                    form="final_answer",
                    secret_matched=f"{draft_variable} = {draft_value}"
                    if draft_variable
                    else str(draft_value),
                    source=source,
                    problem_id=problem_id,
                )
            )
            return matches

    drafted = set(_numbers(text, fold_words=True))
    for value in _numbers(secret):
        if value in drafted:
            matches.append(
                LeakMatch(
                    form="numeric_token",
                    secret_matched=str(value),
                    source=source,
                    problem_id=problem_id,
                )
            )
    return matches


def count_words(text: str) -> int:
    """Words in `text`, estimating Thai runs by character count.

    Feeds the tolerance log only (§6.3), never a block.
    """
    total = 0
    for token in text.split():
        thai = len(_THAI_RANGE.findall(token))
        if thai:
            total += max(1, round(thai / THAI_CHARS_PER_WORD))
            if thai < len(token):
                total += 1
        else:
            total += 1
    return total


def count_questions(text: str) -> int:
    """Questions in `text`.

    Counts question marks; when there are none, counts *sentences* carrying a
    Thai question particle, since Thai commonly asks with a final particle and
    no "?". Per sentence rather than per particle: one Thai question often
    contains two ("คุณคิดว่าควรทำอะไรก่อนไหม"), and counting particles would log
    a plan violation on every well-formed Thai turn.
    """
    marks = len(_QUESTION_MARK.findall(text))
    if marks:
        return marks
    lowered = text.casefold()
    return sum(
        1
        for sentence in _SENTENCE_SPLIT.split(lowered)
        if any(particle in sentence for particle in _THAI_QUESTION_PARTICLES)
    )


def check_output(
    text: str,
    plan: ResponsePlan,
    *,
    turn_id: str,
    config: GuardrailConfig,
    rules: GlobalRules,
    canonical_answer: str | None = None,
    quiz_answer_key: str | None = None,
    problem_id: str = "",
    attempt: int = 1,
) -> GuardrailResult:
    """Scan one generated draft against its plan (Tech Spec §6).

    Args:
        text: the draft from Call C.
        plan: the `ResponsePlan` that briefed it.
        turn_id: for structured logging outside a `TurnTrace`.
        config: §6 thresholds and fallback copy.
        rules: §4.2 global rules; `quiz_answer_before_student_attempt` gates
            check 2.
        canonical_answer: the withheld solution, when there is one. Never pass
            this into generation — it is here only to be scanned for.
        quiz_answer_key: the answer to a practice question in flight.
        problem_id: answer-key id, used as a non-reversible handle in the event
            detail so an operator can find the problem without the event naming
            the answer.
        attempt: 1 for the first draft, 2 for the regenerated one. Past
            `config.max_regenerations` a leak serves the fallback instead of
            asking again.

    Returns:
        A `GuardrailResult`. `action` says what to do; `text` is the draft
        unchanged unless the fallback was served.
    """
    events: list[GuardrailEvent] = []
    leaks: list[LeakMatch] = []
    # Coverage accumulates only from scans that actually ran, so a gated-off
    # check reports as unscanned rather than as clean.
    coverage = ScanCoverage()

    # 1. The withheld canonical answer.
    if canonical_answer and not plan.full_solution_allowed:
        leaks.extend(
            find_leaks(text, canonical_answer, source="canonical_answer", problem_id=problem_id)
        )
        coverage = coverage.merged_with(coverage_for(canonical_answer))

    # 2. A quiz item's key, before the student has attempted it.
    if quiz_answer_key and not rules.quiz_answer_before_student_attempt:
        leaks.extend(
            find_leaks(text, quiz_answer_key, source="quiz_answer_key", problem_id=problem_id)
        )
        coverage = coverage.merged_with(coverage_for(quiz_answer_key))

    # 3. Counts over the plan by more than the tolerance. Logged, served anyway.
    events.extend(_count_events(text, plan, config))

    if not leaks:
        return GuardrailResult(
            action=GuardrailAction.PASS,
            text=text,
            events=tuple(events),
            turn_id=turn_id,
            attempt=attempt,
            coverage=coverage,
        )

    detail = "; ".join(leak.describe() for leak in leaks)
    if attempt <= config.max_regenerations:
        events.append(
            GuardrailEvent(
                type=GuardrailEventType.REGENERATED,
                detail=detail,
                action_taken=f"regenerating (attempt {attempt + 1}) with the leak note appended",
            )
        )
        return GuardrailResult(
            action=GuardrailAction.REGENERATE,
            text=text,
            events=tuple(events),
            turn_id=turn_id,
            attempt=attempt,
            leaks=tuple(leaks),
            regeneration_note=config.regeneration_note,
            coverage=coverage,
        )

    fallback = config.fallback_for(plan.language.value)
    events.append(
        GuardrailEvent(
            type=GuardrailEventType.LEAK_BLOCKED,
            detail=detail,
            action_taken="draft withheld from the student",
        )
    )
    events.append(
        GuardrailEvent(
            type=GuardrailEventType.FALLBACK_SERVED,
            detail=f"regeneration leaked again on attempt {attempt}",
            action_taken="served the template fallback",
        )
    )
    return GuardrailResult(
        action=GuardrailAction.SERVE_FALLBACK,
        text=fallback,
        events=tuple(events),
        turn_id=turn_id,
        attempt=attempt,
        leaks=tuple(leaks),
        coverage=coverage,
    )


def _count_events(text: str, plan: ResponsePlan, config: GuardrailConfig) -> list[GuardrailEvent]:
    """Word and question counts over the plan's limits (§6.3)."""
    events: list[GuardrailEvent] = []
    tolerance = 1.0 + config.count_tolerance

    words = count_words(text)
    if plan.max_words > 0 and words > plan.max_words * tolerance:
        events.append(
            GuardrailEvent(
                type=GuardrailEventType.PLAN_VIOLATION,
                detail=f"{words} words against a plan limit of {plan.max_words}",
                action_taken="served anyway; logged for prompt tuning",
            )
        )

    questions = count_questions(text)
    if plan.max_questions > 0 and questions > plan.max_questions * tolerance:
        events.append(
            GuardrailEvent(
                type=GuardrailEventType.PLAN_VIOLATION,
                detail=f"{questions} questions against a plan limit of {plan.max_questions}",
                action_taken="served anyway; logged for prompt tuning",
            )
        )
    return events


__all__ = [
    "NUMERIC_FORMS",
    "TEXT_FORMS",
    "THAI_CHARS_PER_WORD",
    "UNDETECTED_FORMS",
    "GuardrailAction",
    "GuardrailResult",
    "LeakMatch",
    "ScanCoverage",
    "check_output",
    "count_questions",
    "count_words",
    "coverage_for",
    "find_leaks",
    "fold_number_words",
    "normalise",
]
