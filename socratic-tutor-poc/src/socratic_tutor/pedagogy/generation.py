"""Call C — Response Generation (R10, Tech Spec §5.3).

The system prompt is assembled in four layers, always in this order:

    ROLE    — tutor persona and tone. Byte-stable, so it is the prompt-cache
              prefix; never interpolate per-turn data into it.
    POLICY  — rendered from the `ResponsePlan`: structure sequence, word cap,
              question cap, `full_solution_allowed`, source-reference rule.
    STATE   — student level, learner state, topic, and the knowledge_guidance
              directive composed from the `Relation` enum plus the human-readable
              sentences in `KnowledgeContext.relationships`.
    CONTENT — retrieved chunks with their source references.

**Structural leak prevention.** When `full_solution_allowed` is False the
canonical answer is not in the prompt at all — not present-but-forbidden. A
model cannot leak what it never received, and that property survives prompt
injection, jailbreaks and student pressure, which an instruction does not
(Tech Spec §5.3, §6; E3's gate is zero hard leaks). `build_system_prompt`
therefore takes the canonical solution as an explicit argument and drops it
unless the plan permits it; `assert_no_canonical_leak` is the belt-and-braces
check the tests assert on.

Thai is authored natively: when the plan's language is Thai the model is
instructed in Thai to compose in Thai, rather than to translate an English
draft (Addendum §"Language & review").
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import AsyncIterator, Sequence

from socratic_tutor.config import Settings, get_settings
from socratic_tutor.models.enums import Language, LearnerState, Relation, StudentLevel
from socratic_tutor.models.knowledge import KnowledgeContext, RetrievedChunk
from socratic_tutor.models.plan import GLOBAL_RULES, ResponsePlan
from socratic_tutor.models.strategy import KnowledgeGuidance
from socratic_tutor.pedagogy.evaluation import Problem
from socratic_tutor.providers import get_provider
from socratic_tutor.providers.base import LLMProvider, Msg, StreamStats

#: Layer headings. Stable strings — the trace and the tests locate layers by them.
ROLE_HEADING = "### ROLE"
POLICY_HEADING = "### POLICY"
STATE_HEADING = "### STATE"
CONTENT_HEADING = "### CONTENT"

#: The persona. **Byte-stable**: this block is the prompt-cache prefix, so it
#: must not carry per-turn data and should not be edited casually.
ROLE_LAYER = """You are Eddi, a patient subject tutor working one-to-one with a student.

You teach by guiding, not by telling. You ask before you explain, you build on
what the student already knows, and you keep the student doing the thinking.

Tone: clear, respectful, encouraging, never patronising. Address the student as
a capable person. Do not praise effort you have not seen. Do not apologise for
the material being hard.

You never claim a fact you were not given in the CONTENT section. If the content
does not support an answer, you say what you do know and ask a question that
moves the student forward.

You follow the POLICY section exactly. It is not advice: it is the shape of the
reply you are permitted to produce this turn."""

#: Written in Thai so the model composes in Thai rather than translating.
THAI_DIRECTIVE = (
    "ตอบเป็นภาษาไทยทั้งหมด เขียนขึ้นใหม่เป็นภาษาไทยโดยตรง ไม่ใช่การแปลจากภาษาอังกฤษ "
    "ใช้ภาษาที่เป็นธรรมชาติ สุภาพ และเข้าใจง่าย คงคำศัพท์เทคนิคภาษาอังกฤษไว้ตามเดิม"
    "เมื่อคำแปลไทยทำให้เข้าใจยากกว่า"
)

#: Wording rules per declared level (Tech Spec §4.2).
LEVEL_WORDING: dict[StudentLevel, str] = {
    StudentLevel.BEGINNER: "Use simple language. Do not use a technical term without explaining it.",
    StudentLevel.INTERMEDIATE: "Use normal subject terminology; assume the basics are in place.",
    StudentLevel.ADVANCED: "Be concise and technical; do not re-explain the fundamentals.",
}

#: How each relation reads as guidance for the generator.
RELATION_PHRASING: dict[Relation, tuple[str, str]] = {
    Relation.PREREQUISITE_OF: ("is a prerequisite of", "builds on"),
    Relation.RELATED_TO: ("is related to", "is related to"),
    Relation.NEXT_TOPIC: ("comes after", "leads to"),
    Relation.PART_OF: ("is part of", "contains"),
    Relation.USES: ("uses", "is used by"),
}

#: Source refs that look like answer keys never reach Call C, whatever the plan
#: says. Ingestion already refuses them (R3); this is the second lock.
_ANSWER_KEY_MARKER = re.compile(r"answer[\s_-]?key|solution[\s_-]?key|เฉลย", re.IGNORECASE)

#: The Thai Unicode block, as an escape rather than literal Thai so the pattern
#: survives an editor that normalises the source file.
_THAI_BLOCK = re.compile("[\\u0e00-\\u0e7f]")


def detect_language(text: str) -> Language:
    """Thai when the text contains any Thai character, English otherwise.

    Deterministic on purpose: the response language is a policy input, and a
    model call to decide it would be one more thing that can go wrong before the
    student sees a reply.
    """
    return Language.TH if _THAI_BLOCK.search(text or "") else Language.EN


def build_system_prompt(
    plan: ResponsePlan,
    *,
    topic: str,
    learner_state: LearnerState = LearnerState.NORMAL,
    knowledge_context: KnowledgeContext | None = None,
    knowledge_guidance: KnowledgeGuidance | None = None,
    chunks: Sequence[RetrievedChunk] = (),
    canonical: Problem | None = None,
    quiz_item: Problem | None = None,
) -> str:
    """Assemble the four-layer Call C system prompt.

    `canonical` is included only when `plan.full_solution_allowed` is True. Pass
    it unconditionally: deciding *here* rather than at the call site is what
    makes the guarantee structural, since a caller cannot forget to withhold it.
    """
    layers = [
        f"{ROLE_HEADING}\n{ROLE_LAYER}",
        f"{POLICY_HEADING}\n{_policy_layer(plan)}",
        f"{STATE_HEADING}\n{_state_layer(plan, topic, learner_state, knowledge_context, knowledge_guidance)}",
        f"{CONTENT_HEADING}\n{_content_layer(plan, chunks, canonical, quiz_item)}",
    ]
    return "\n\n".join(layers)


def _policy_layer(plan: ResponsePlan) -> str:
    parts = describe_structure(plan.structure)
    structure = "; then ".join(parts) if parts else "answer in whatever shape fits"
    lines = [
        f"Cover these, in this order: {structure}.",
        # The raw block names are never sent. A model handed `single_hint,
        # student_attempt_prompt` treats them as an output format and prints
        # them as headings — real replies came back reading "**State check:**
        # … **Next step:** …", which is our plan's field names shown to a
        # student. Removing the tokens is the structural fix; the sentence below
        # is the belt-and-braces one, because an instruction alone was never
        # going to be enough (the same reasoning as the withheld answer).
        (
            "Write as one person talking to another: continuous prose, no headings, "
            "no bold labels, no numbered sections, and never name the parts of your "
            "reply. The order above is an internal specification, not a format."
        ),
        f"Hard limit: {plan.max_words} words. Ask at most {plan.max_questions} question(s).",
        f"At most {GLOBAL_RULES.max_examples} example and {GLOBAL_RULES.max_analogies} analogy.",
    ]

    if plan.full_solution_allowed:
        lines.append("You may give the complete solution this turn.")
    else:
        # The answer is not in this prompt; this line explains the shape of the
        # reply, it is not what stops a leak.
        lines.append(
            "You may NOT give the final answer or a complete solution this turn. "
            "The answer is deliberately not in this prompt. If the student asks "
            "for it directly, give the next step of thinking instead and invite "
            "an attempt."
        )

    if plan.wait_for_student:
        lines.append("End by handing the turn back: the student answers next, not you.")
    if GLOBAL_RULES.source_reference_required:
        lines.append(
            "Cite the source reference of any content you use, exactly as given in CONTENT."
        )
    if not GLOBAL_RULES.unsupported_claims_allowed:
        lines.append("Make no claim the CONTENT section does not support.")

    lines.append(LEVEL_WORDING[plan.language_level])
    lines.append(
        THAI_DIRECTIVE
        if plan.language is Language.TH
        else "Reply in English, matching the student's own wording where natural."
    )
    return "\n".join(f"- {line}" for line in lines)


#: Block names rendered as instructions. Derived by de-underscoring the name, so
#: a new block in `response_templates.yaml` reads sensibly without an edit here;
#: the few whose name does not read as an instruction are spelled out.
_BLOCK_PHRASING: dict[str, str] = {
    "direct_answer": "answer the question directly",
    "explanation": "explain it",
    "source_reference": "say where that comes from",
    "reframed_explanation": "explain it a different way from last time",
    "state_evaluation": "say how their attempt went",
    "explain_feedback": "explain what was right or wrong about it",
    "next_action": "say what to do next",
    "one_example_or_analogy": "give one example or analogy",
    "ordered_steps": "walk through the steps, elaborating only the first",
    "single_hint": "give exactly one hint",
    "check_understanding": "ask one question that checks they followed",
    "single_check_question": "ask one question that checks they followed",
    "next_step_prompt": "ask what they would do next",
    "student_attempt_prompt": "invite them to try it",
    "single_practice_question": "ask one practice question",
}


def describe_structure(structure: Sequence[str]) -> list[str]:
    """The plan's blocks as instructions a tutor could follow.

    Public so a test can assert the raw token never survives into the prompt.
    """
    return [_BLOCK_PHRASING.get(name, name.replace("_", " ")) for name in structure]


def _state_layer(
    plan: ResponsePlan,
    topic: str,
    learner_state: LearnerState,
    knowledge_context: KnowledgeContext | None,
    knowledge_guidance: KnowledgeGuidance | None,
) -> str:
    lines = [
        f"- Topic: {topic}",
        f"- Student's declared level: {plan.language_level.value}",
        f"- Learner state this turn: {learner_state.value}",
    ]

    if knowledge_context is not None:
        for label, items in (
            ("Prerequisites", knowledge_context.prerequisites),
            ("Related", knowledge_context.related_topics),
            ("Next", knowledge_context.next_topics),
        ):
            if items:
                rendered = ", ".join(f"{item.topic} (distance {item.distance})" for item in items)
                lines.append(f"- {label}: {rendered}")
        if knowledge_context.relationships:
            lines.append("- How these connect:")
            lines.extend(f"  - {sentence}" for sentence in knowledge_context.relationships)

    directive = _guidance_directive(knowledge_guidance)
    if directive:
        lines.append(f"- Guidance: {directive}")

    return "\n".join(lines)


def _guidance_directive(guidance: KnowledgeGuidance | None) -> str:
    """Compose the knowledge_guidance sentence from the enum, not from free text.

    A prerequisite edge is a hypothesis to check, never a diagnosis: when the
    action is check-or-scaffold, the directive says probe first in as many words,
    so the generator cannot restate graph structure as a claim about this
    student (Tech Spec §3.5 hard rule).
    """
    if guidance is None or guidance.action is None:
        return ""

    parts = [guidance.action.value.replace("_", " ")]
    if guidance.target_concept:
        parts.append(f"using “{guidance.target_concept}”")
    if guidance.relation is not None:
        forward, inverse = RELATION_PHRASING[guidance.relation]
        reading = forward if guidance.direction == "forward" else inverse
        parts.append(f"({guidance.target_concept or 'it'} {reading} this topic)")
    if guidance.distance is not None:
        parts.append(f"[distance {guidance.distance}]")

    directive = " ".join(parts)
    if guidance.action.value == "check_or_scaffold_prerequisite":
        directive += (
            ". Ask one question to check whether the student has this prerequisite. "
            "Do not tell the student they lack it — the course map says it comes "
            "first, which is not evidence about this student."
        )
    return directive


def _content_layer(
    plan: ResponsePlan,
    chunks: Sequence[RetrievedChunk],
    canonical: Problem | None,
    quiz_item: Problem | None = None,
) -> str:
    usable = [chunk for chunk in chunks if not _looks_like_answer_key(chunk)]

    blocks: list[str] = []
    if usable:
        blocks.extend(
            f"[{index}] source_ref: {chunk.source_ref}\n{chunk.text.strip()}"
            for index, chunk in enumerate(usable, start=1)
        )
    else:
        blocks.append(
            "(no retrieved content for this turn — do not state subject facts; "
            "ask the student a question that moves them forward)"
        )

    if quiz_item is not None:
        # Statement only. The tutor asks the question the system chose — which
        # is what lets the guardrail know which answer to withhold — and the
        # answer to it is never in this prompt, whatever the plan permits,
        # because §4.2 forbids a quiz answer before the student has attempted.
        blocks.append(
            "[practice item] Ask exactly this question, in the student's "
            f"language, and do not answer it:\n{quiz_item.statement}"
        )

    # The single decision that makes leak prevention structural.
    if canonical is not None and plan.full_solution_allowed:
        steps = "\n".join(
            f"  {index}. {step}" for index, step in enumerate(canonical.canonical_steps, 1)
        )
        blocks.append(
            f"[canonical solution] source_ref: answer-key#{canonical.id}\n"
            f"{canonical.statement}\nAnswer: {canonical.answer}\n{steps}".rstrip()
        )

    return "\n\n".join(blocks)


def _looks_like_answer_key(chunk: RetrievedChunk) -> bool:
    return bool(_ANSWER_KEY_MARKER.search(chunk.source_ref or ""))


def build_messages(student_message: str, conversation: Sequence[Msg] = ()) -> list[Msg]:
    """The user turn plus prior dialogue, oldest first.

    Unlike Call B, Call C *does* see the dialogue: continuity is the point of a
    tutoring turn (the isolation rule in §5.2 applies to evaluation only).
    """
    return [*conversation, Msg(role="user", content=student_message.strip())]


def assert_no_canonical_leak(prompt: str, canonical: Problem | None) -> None:
    """Raise if a withheld canonical solution is present in an assembled prompt.

    Cheap enough to run on every turn and it turns a silent policy breach into a
    loud failure. Callers run this only when the plan withholds the solution.
    """
    if canonical is None:
        return
    haystack = _normalise_for_scan(prompt)
    for token in [canonical.answer, *canonical.canonical_steps]:
        if token and _normalise_for_scan(token) in haystack:
            raise AssertionError(
                "canonical solution leaked into the generation prompt while "
                "full_solution_allowed was false"
            )


def _normalise_for_scan(text: str) -> str:
    """NFKC + casefold + whitespace-stripped, so "x = 7" and "X=7" compare equal."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or "").casefold())


async def generate_response(
    plan: ResponsePlan,
    student_message: str,
    *,
    topic: str,
    learner_state: LearnerState = LearnerState.NORMAL,
    knowledge_context: KnowledgeContext | None = None,
    knowledge_guidance: KnowledgeGuidance | None = None,
    chunks: Sequence[RetrievedChunk] = (),
    canonical: Problem | None = None,
    quiz_item: Problem | None = None,
    conversation: Sequence[Msg] = (),
    system_note: str = "",
    provider: LLMProvider | None = None,
    settings: Settings | None = None,
    stats: StreamStats | None = None,
) -> AsyncIterator[str]:
    """Stream Call C's text deltas.

    `system_note` is appended after the four layers and is how the guardrail's
    regenerate-once path tells the model what was wrong with its first draft
    (Tech Spec §6); it never carries content, only an instruction.

    Raises `AssertionError` before calling the model if a withheld canonical
    solution somehow reached the prompt — failing the turn is strictly better
    than serving a leak.
    """
    resolved = settings or get_settings()
    llm = provider or get_provider("generation", settings)

    system = build_system_prompt(
        plan,
        topic=topic,
        learner_state=learner_state,
        knowledge_context=knowledge_context,
        knowledge_guidance=knowledge_guidance,
        chunks=chunks,
        canonical=canonical,
        quiz_item=quiz_item,
    )
    if not plan.full_solution_allowed:
        assert_no_canonical_leak(system, canonical)
    # A quiz item's answer is withheld regardless of the plan's flag (§4.2
    # `quiz_answer_before_student_attempt`), so it is checked unconditionally.
    assert_no_canonical_leak(system, quiz_item)
    if system_note:
        system = f"{system}\n\n### NOTE\n{system_note.strip()}"

    async for delta in llm.stream_text(
        model=resolved.model_for("generation"),
        system=system,
        messages=build_messages(student_message, conversation),
        max_tokens=resolved.max_tokens_for("generation"),
        stats=stats,
    ):
        yield delta


__all__ = [
    "CONTENT_HEADING",
    "LEVEL_WORDING",
    "POLICY_HEADING",
    "RELATION_PHRASING",
    "ROLE_HEADING",
    "ROLE_LAYER",
    "STATE_HEADING",
    "THAI_DIRECTIVE",
    "assert_no_canonical_leak",
    "build_messages",
    "build_system_prompt",
    "describe_structure",
    "detect_language",
    "generate_response",
]
