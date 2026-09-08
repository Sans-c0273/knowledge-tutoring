"""Step 1 — Intent Detection, Call A (R6; Tech Spec §2.1, §5.1).

The only LLM call on the way into the pipeline. It classifies what the student
wants, whether they are confused, and whether their own words put the turn under
a policy override. Everything downstream is deterministic.

Three properties this module enforces structurally rather than by prompting:

- **No RAG content, ever** (§5.1). There is no parameter to pass evidence
  through; the call cannot see retrieved material even by mistake.
- **History is required.** "Explain again" is unclassifiable without it, so the
  last `MAX_CONTEXT_TURNS` turns are always in the prompt (§5.1).
- **Low confidence falls back to `explain`** and the fallback is recorded on the
  result, so the trace and the glass-box UI can show that it happened rather
  than presenting a guess as a classification (§2.1).

Bilingual by construction: one prompt classifies Thai and English. The student's
message is never translated first — translation would destroy exactly the
politeness and hedging cues that distinguish `clarify` from `explain`.
"""

from __future__ import annotations

from collections.abc import Sequence

from socratic_tutor.config import Settings, get_settings
from socratic_tutor.models.enums import Intent
from socratic_tutor.models.intent import IntentClassification, IntentResult
from socratic_tutor.providers import LLMProvider, get_provider
from socratic_tutor.providers.base import Msg, StructuredResult

#: §5.1 asks for the last 4–6 turns; we keep the top of that range.
MAX_CONTEXT_TURNS = 6

TOOL_TEMPERATURE = 0.0


SYSTEM_PROMPT = """\
You classify a single student message inside a one-to-one tutoring session. You \
are a classifier, not a tutor: you never answer the student, never teach, and \
never solve anything. You emit one structured classification by calling the \
provided tool.

# Languages

Messages arrive in Thai or English and may mix the two. Classify the message in \
the language it was written in. Do not translate it first — Thai politeness \
particles and hedging ("ครับ", "ค่ะ", "ยังไม่เข้าใจเลย") are evidence, and \
translation destroys them.

# intent — what the student wants (choose exactly one)

- explain — wants to learn a concept they have not been taught yet, or wants the
  reason behind a rule. "What is …?", "Why does …?"
- clarify — was already given an explanation and did not understand it. The
  giveaway is reference to a previous turn: "explain again", "I still don't get
  it", "say it another way", "that didn't make sense".
- solve — wants help completing a specific problem, or wants to continue working
  through one. "How do I solve …?", "what next", "then what".
- hint — explicitly asks for partial help only, and usually says so: "give me a
  hint", "don't tell me the answer yet", "ขอคำใบ้".
- check_answer — has produced an attempt or a claim and wants it judged. "I got
  x = 7, is that right?", "is this correct?", and also checking a generalisation
  they have just formed ("so it's basically just …?").
- practice_quiz — wants to be tested or given practice items. "Quiz me", "give me
  a harder one", "ขอแบบฝึกหัด".
- summarize_review — wants existing material consolidated. "Summarise what we
  covered", "recap this chapter".

Disambiguation rules:

- A message that both states an attempt and asks for a check is check_answer,
  not solve, even when it also asks where they went wrong.
- A message asking to continue an in-progress solution ("ok what next") is
  solve, not clarify — they are not stuck on an explanation, they are mid-problem.
- A bare arithmetic question with no request to be taught ("what is 7 minus 12")
  is solve.
- Demanding an answer outright ("just give me the answer") is solve; the demand
  is handled by policy, not by the intent label.
- Asking for something harder after a correct answer is practice_quiz.

# learner_state — normal or confused

Evidence for confused, per the design's evidence table:

- Strong: says so outright ("I don't understand", "I'm confused", "งง",
  "ยังไม่เข้าใจ"); asks for a different explanation; expresses being stuck or
  demoralised after repeated failure; exhibits a known misconception.
- Possible, not sufficient on its own: repeating the same question; getting an
  answer wrong. A wrong answer is a knowledge gap, which is not the same thing
  as confusion.
- No signal at all → normal.

Confusion is binary. There are no degrees.

# special_handling — detected from the student's own words ONLY

- homework — the student says the work is homework or an assignment: "this is
  for my homework", "การบ้านข้อ 2", "just give me the homework answer".
- assessment — the student says they are being tested right now: "I'm taking my
  test", "กำลังสอบอยู่". A test *tomorrow* is not assessment; they are revising,
  which is ordinary study.
- none — anything else.

Never infer homework or assessment from the difficulty of the problem, from the
topic, or from a bare demand for an answer. Only the student's own statement
counts. Set `detected` true when the type is homework or assessment, false when
it is none. This field is a policy switch, so a miss is a policy breach: read
the message carefully for it.

# confidence

Your own probability, 0.0–1.0, that the `intent` label is right. Be honest and
calibrated: report low confidence when the message is genuinely ambiguous or
when history you would need is missing. A low value is handled safely downstream;
an inflated one is not.

# Worked examples

message: "Can you explain what a variable actually is?"
→ explain / normal / none, high confidence.

message: "wait, that still doesn't click for me"
history: the tutor has just explained inverse operations
→ clarify / confused / none. Reference to the previous turn plus a statement of
  not understanding.

message: "ผมได้ 12 ครับ ใช่ไหมครับ"
→ check_answer / normal / none. An attempt offered for judgement.

message: "ตอนนี้กำลังสอบอยู่ ข้อนี้ตอบอะไรครับ"
→ solve / normal / assessment, detected true. The student states they are in an
  exam right now.

message: "don't give it away, just nudge me"
→ hint / normal / none.

message: "I've tried this four times and nothing works"
history: two failed attempts
→ clarify / confused / none. Repeated failure plus demoralisation.
"""


def normalize_context(context: Sequence[str | Msg]) -> list[Msg]:
    """Conversation history as `Msg` objects, newest `MAX_CONTEXT_TURNS` kept.

    Accepts the eval sets' `"user: …"` / `"assistant: …"` strings as well as
    `Msg`. An entry with no recognised prefix is treated as an assistant turn:
    the seed sets use unprefixed entries for narrative placeholders describing
    what the tutor covered.
    """
    turns: list[Msg] = []
    for entry in context:
        if isinstance(entry, Msg):
            turns.append(entry)
            continue
        text = entry.strip()
        lowered = text.casefold()
        if lowered.startswith("user:"):
            turns.append(Msg(role="user", content=text[len("user:") :].strip()))
        elif lowered.startswith("assistant:"):
            turns.append(Msg(role="assistant", content=text[len("assistant:") :].strip()))
        else:
            turns.append(Msg(role="assistant", content=text))
    return turns[-MAX_CONTEXT_TURNS:]


def build_user_message(message: str, context: Sequence[str | Msg] = ()) -> str:
    """The single user turn sent to Call A: history block, then the message.

    History travels inside one tagged block rather than as real alternating
    turns. That keeps the payload identical across providers (Anthropic rejects
    a leading assistant turn; OpenAI-compatible endpoints do not), makes the
    exact window auditable in the trace, and keeps the system prompt as a
    byte-stable cache prefix (§10).
    """
    turns = normalize_context(context)
    if turns:
        history = "\n".join(f"{turn.role}: {turn.content}" for turn in turns)
        history_block = f"<conversation_history>\n{history}\n</conversation_history>\n\n"
    else:
        history_block = (
            "<conversation_history>\n(this is the first turn)\n</conversation_history>\n\n"
        )
    return f"{history_block}<student_message>\n{message.strip()}\n</student_message>"


async def detect_intent(
    message: str,
    conversation_context: Sequence[str | Msg] = (),
    provider: LLMProvider | None = None,
    *,
    settings: Settings | None = None,
    model: str | None = None,
) -> IntentResult:
    """Classify one student message (Call A).

    `conversation_context` is the prior turns, oldest first; only the last
    `MAX_CONTEXT_TURNS` reach the model. There is deliberately no parameter for
    retrieved content — Call A never sees RAG output (§5.1).

    Raises whatever `ProviderError` the provider raises; a failed classification
    is not silently downgraded to a default label, because the caller has to be
    able to record the failure on the trace.
    """
    resolved = settings or get_settings()
    llm = provider or get_provider("intent", settings if settings is not None else None)

    result: StructuredResult = await llm.complete_structured(
        model=model or resolved.model_for("intent"),
        system=SYSTEM_PROMPT,
        messages=[Msg(role="user", content=build_user_message(message, conversation_context))],
        schema=IntentClassification,
        max_tokens=resolved.max_tokens_for("intent"),
        temperature=TOOL_TEMPERATURE,
    )
    classification = result.value
    assert isinstance(classification, IntentClassification)  # provider validated it
    return apply_confidence_floor(classification, resolved.intent_confidence_threshold)


def apply_confidence_floor(classification: IntentClassification, threshold: float) -> IntentResult:
    """Convert a raw classification to an `IntentResult`, applying the §2.1 floor.

    Below the threshold the intent becomes `explain` — the safest label, since it
    grants no answer-giving licence that the other intents would — and both the
    fallback flag and the classifier's original label are kept so the trace shows
    what really happened. `learner_state` and `special_handling` are NOT
    overridden: a low-confidence intent does not make the confusion signal or a
    stated "this is my homework" less true, and discarding the policy switch
    would turn a classifier wobble into a policy breach.
    """
    low = classification.confidence < threshold
    return IntentResult(
        # DO NOT widen this override to the whole result. Only `intent` is
        # reset. `learner_state` and `special_handling` are carried through
        # untouched, because resetting them would convert a classifier miss into
        # a policy breach: a low-confidence turn would silently lose homework or
        # assessment mode and become answerable, which is exactly what the
        # Teaching Policy layer exists to prevent — and it would be invisible,
        # since nothing downstream can tell a dropped flag from one that was
        # never set.
        intent=Intent.EXPLAIN if low else classification.intent,
        learner_state=classification.learner_state,
        special_handling=classification.special_handling,
        confidence=classification.confidence,
        low_confidence_fallback=low,
        raw_intent=classification.intent if low else None,
    )


__all__ = [
    "MAX_CONTEXT_TURNS",
    "SYSTEM_PROMPT",
    "IntentClassification",
    "apply_confidence_floor",
    "build_user_message",
    "detect_intent",
    "normalize_context",
]
