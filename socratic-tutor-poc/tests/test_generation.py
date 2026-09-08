"""R10 — Call C prompt assembly and streaming (Tech Spec §5.3).

The load-bearing test in this file is
`test_canonical_solution_is_absent_when_the_plan_withholds_it`: leak prevention
is structural, so the property to assert is that the answer is not in the bytes
sent to the model, not that the model was told to keep quiet.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from pydantic import BaseModel

from socratic_tutor.config import Settings
from socratic_tutor.models.enums import (
    KnowledgeAction,
    Language,
    LearnerState,
    Relation,
    StudentLevel,
)
from socratic_tutor.models.knowledge import KnowledgeContext, RelatedTopic, RetrievedChunk
from socratic_tutor.models.plan import ResponsePlan
from socratic_tutor.models.strategy import KnowledgeGuidance
from socratic_tutor.pedagogy.evaluation import Problem
from socratic_tutor.pedagogy.generation import (
    CONTENT_HEADING,
    POLICY_HEADING,
    ROLE_HEADING,
    ROLE_LAYER,
    STATE_HEADING,
    THAI_DIRECTIVE,
    assert_no_canonical_leak,
    build_system_prompt,
    describe_structure,
    detect_language,
    generate_response,
)
from socratic_tutor.providers.base import LLMProvider, Msg, StreamStats, StructuredResult, Usage

CANONICAL = Problem(
    id="P03",
    topic="Two-Step Linear Equations",
    statement_en="Solve: 2x + 4 = 10",
    answer="x = 3",
    canonical_steps=["Subtract 4 from both sides: 2x = 6", "Divide both sides by 2: x = 3"],
    guardrail_tokens=["3"],
)

CHUNKS = [
    RetrievedChunk(
        text="To undo addition, subtract the same number from both sides.",
        source_ref="ch1-inverse-operations.md#L12",
        topic="Inverse Operations",
        lang=Language.EN,
        score=0.9,
    )
]


class RecordingProvider(LLMProvider):
    """Captures what Call C was sent and streams a canned reply."""

    name = "recording"

    def __init__(self, reply: str = "What would you do to both sides first?") -> None:
        self.reply = reply
        self.system: str = ""
        self.messages: list[Msg] = []

    async def complete_structured(
        self,
        *,
        model: str,
        system: str,
        messages: Sequence[Msg],
        schema: type[BaseModel],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> StructuredResult:
        raise NotImplementedError

    async def stream_text(
        self,
        *,
        model: str,
        system: str,
        messages: Sequence[Msg],
        max_tokens: int = 2000,
        stats: StreamStats | None = None,
    ) -> AsyncIterator[str]:
        self.system = system
        self.messages = list(messages)
        if stats is not None:
            stats.model, stats.provider = model, self.name
            stats.ttft_ms = 1.0
            stats.usage = Usage(input_tokens=100, output_tokens=20)
        for word in self.reply.split(" "):
            yield word + " "


def hint_plan(**overrides: Any) -> ResponsePlan:
    defaults: dict[str, Any] = {
        "structure": ["single_hint", "student_attempt_prompt"],
        "max_words": 50,
        "max_questions": 1,
        "language_level": StudentLevel.BEGINNER,
        "language": Language.EN,
        "full_solution_allowed": False,
        "wait_for_student": True,
    }
    return ResponsePlan(**{**defaults, **overrides})


def section(prompt: str, heading: str) -> str:
    """The text of one layer, up to the next heading."""
    body = prompt.split(heading, 1)[1]
    for other in (ROLE_HEADING, POLICY_HEADING, STATE_HEADING, CONTENT_HEADING):
        if other != heading and other in body:
            body = body.split(other, 1)[0]
    return body


def test_prompt_has_four_layers_in_order() -> None:
    prompt = build_system_prompt(hint_plan(), topic="Two-Step Linear Equations", chunks=CHUNKS)

    positions = [
        prompt.index(h) for h in (ROLE_HEADING, POLICY_HEADING, STATE_HEADING, CONTENT_HEADING)
    ]
    assert positions == sorted(positions)
    assert ROLE_LAYER in prompt


def test_role_layer_carries_no_per_turn_data() -> None:
    """It is the prompt-cache prefix: any interpolation there costs every turn."""
    prompt = build_system_prompt(
        hint_plan(), topic="Two-Step Linear Equations", chunks=CHUNKS, canonical=None
    )

    role = section(prompt, ROLE_HEADING)
    assert "Two-Step Linear Equations" not in role
    assert "single_hint" not in role


def test_policy_layer_renders_the_plan() -> None:
    plan = hint_plan()
    policy = section(build_system_prompt(plan, topic="Two-Step Linear Equations"), POLICY_HEADING)

    assert "give exactly one hint" in policy and "invite them to try it" in policy
    assert "single_hint" not in policy, "block names are internal (see the token test below)"
    assert "50 words" in policy
    assert "at most 1 question" in policy.lower()
    assert "may NOT give the final answer" in policy
    assert "handing the turn back" in policy, "wait_for_student must reach the model"
    assert "source reference" in policy.lower()
    assert "Use simple language" in policy


def test_policy_layer_permits_the_solution_when_the_plan_does() -> None:
    policy = section(
        build_system_prompt(
            hint_plan(full_solution_allowed=True), topic="Two-Step Linear Equations"
        ),
        POLICY_HEADING,
    )

    assert "may give the complete solution" in policy
    assert "may NOT give the final answer" not in policy


def test_canonical_solution_is_absent_when_the_plan_withholds_it() -> None:
    """R10's acceptance criterion: the answer is not in the prompt at all."""
    prompt = build_system_prompt(
        hint_plan(),
        topic="Two-Step Linear Equations",
        chunks=CHUNKS,
        canonical=CANONICAL,
    )

    assert "x = 3" not in prompt
    assert "Divide both sides by 2" not in prompt
    assert "Subtract 4 from both sides" not in prompt
    assert CANONICAL.statement not in prompt
    assert_no_canonical_leak(prompt, CANONICAL)


def test_canonical_solution_is_present_when_the_plan_allows_it() -> None:
    prompt = build_system_prompt(
        hint_plan(full_solution_allowed=True),
        topic="Two-Step Linear Equations",
        chunks=CHUNKS,
        canonical=CANONICAL,
    )

    assert "x = 3" in prompt
    assert "Divide both sides by 2: x = 3" in prompt
    with pytest.raises(AssertionError):
        assert_no_canonical_leak(prompt, CANONICAL)


def test_leak_check_normalises_before_comparing() -> None:
    with pytest.raises(AssertionError):
        assert_no_canonical_leak("the answer is X=3, well done", CANONICAL)

    assert_no_canonical_leak("what would you subtract from both sides?", CANONICAL)
    assert_no_canonical_leak("anything at all", None)


def test_answer_key_chunks_never_reach_the_prompt() -> None:
    """Ingestion refuses them (R3); this is the second lock, not the first."""
    smuggled = RetrievedChunk(
        text="P03 answer: x = 3",
        source_ref="answer-key.md#P03",
        topic="Two-Step Linear Equations",
        lang=Language.EN,
        score=0.99,
    )

    prompt = build_system_prompt(hint_plan(), topic="Two-Step Linear Equations", chunks=[smuggled])

    assert "x = 3" not in prompt
    assert "no retrieved content" in prompt


def test_content_layer_carries_source_references() -> None:
    content = section(
        build_system_prompt(hint_plan(), topic="Two-Step Linear Equations", chunks=CHUNKS),
        CONTENT_HEADING,
    )

    assert "ch1-inverse-operations.md#L12" in content
    assert "subtract the same number" in content


def test_empty_retrieval_is_stated_not_hidden() -> None:
    """No evidence has policy consequences; the model must know it has none."""
    content = section(
        build_system_prompt(hint_plan(), topic="Two-Step Linear Equations", chunks=[]),
        CONTENT_HEADING,
    )

    assert "no retrieved content" in content


def test_state_layer_renders_the_knowledge_context() -> None:
    knowledge = KnowledgeContext(
        current_topic="Two-Step Linear Equations",
        prerequisites=[
            RelatedTopic(topic="One-Step Linear Equations", distance=1),
            RelatedTopic(topic="Inverse Operations", distance=2),
        ],
        related_topics=[RelatedTopic(topic="Checking a Solution", distance=1)],
        next_topics=[RelatedTopic(topic="Word Problems to Equations", distance=1)],
        relationships=["One-Step Linear Equations is a prerequisite of Two-Step Linear Equations"],
    )

    state = section(
        build_system_prompt(
            hint_plan(),
            topic="Two-Step Linear Equations",
            learner_state=LearnerState.CONFUSED,
            knowledge_context=knowledge,
        ),
        STATE_HEADING,
    )

    assert "Topic: Two-Step Linear Equations" in state
    assert "beginner" in state
    assert "confused" in state
    assert "One-Step Linear Equations (distance 1)" in state
    assert "Inverse Operations (distance 2)" in state
    assert "is a prerequisite of" in state


def test_prerequisite_guidance_says_probe_not_diagnose() -> None:
    """Tech Spec §3.5: an edge is a hypothesis to check, never a diagnosis."""
    guidance = KnowledgeGuidance(
        action=KnowledgeAction.CHECK_OR_SCAFFOLD_PREREQUISITE,
        target_concept="One-Step Linear Equations",
        relation=Relation.PREREQUISITE_OF,
        distance=1,
    )

    state = section(
        build_system_prompt(
            hint_plan(), topic="Two-Step Linear Equations", knowledge_guidance=guidance
        ),
        STATE_HEADING,
    )

    assert "check or scaffold prerequisite" in state
    assert "One-Step Linear Equations" in state
    assert "Ask one question to check" in state
    assert "Do not tell the student they lack it" in state


def test_guidance_reads_an_inverse_relation_the_other_way() -> None:
    guidance = KnowledgeGuidance(
        action=KnowledgeAction.USE_RELATED_CONCEPT_FOR_REEXPLANATION,
        target_concept="Variables and Expressions",
        relation=Relation.USES,
        direction="inverse",
        distance=1,
    )

    state = section(
        build_system_prompt(hint_plan(), topic="Combining Like Terms", knowledge_guidance=guidance),
        STATE_HEADING,
    )

    assert "is used by" in state


def test_thai_plan_is_instructed_in_thai() -> None:
    thai = section(
        build_system_prompt(hint_plan(language=Language.TH), topic="สมการเชิงเส้นสองขั้นตอน"),
        POLICY_HEADING,
    )
    english = section(
        build_system_prompt(hint_plan(), topic="Two-Step Linear Equations"), POLICY_HEADING
    )

    assert THAI_DIRECTIVE in thai
    assert "ไม่ใช่การแปลจากภาษาอังกฤษ" in thai, "must say natively authored, not translated"
    assert THAI_DIRECTIVE not in english
    assert "Reply in English" in english


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("How do I solve this?", Language.EN),
        ("ช่วยอธิบายหน่อย", Language.TH),
        ("ช่วยอธิบาย Two-Step Linear Equations หน่อย", Language.TH),
        ("", Language.EN),
    ],
)
def test_language_detection(text: str, expected: Language) -> None:
    assert detect_language(text) is expected


async def test_generate_response_streams_and_records_stats() -> None:
    provider = RecordingProvider("Try subtracting 4 from both sides.")
    stats = StreamStats()

    deltas = [
        delta
        async for delta in generate_response(
            hint_plan(),
            "How do I solve 2x + 4 = 10?",
            topic="Two-Step Linear Equations",
            chunks=CHUNKS,
            canonical=CANONICAL,
            provider=provider,
            settings=Settings(),
            stats=stats,
        )
    ]

    assert "".join(deltas).strip() == "Try subtracting 4 from both sides."
    assert stats.usage.output_tokens == 20
    assert stats.ttft_ms is not None
    assert "x = 3" not in provider.system, "the withheld answer never reached the provider"
    assert provider.messages[-1].content == "How do I solve 2x + 4 = 10?"


async def test_generate_response_appends_the_regeneration_note() -> None:
    provider = RecordingProvider()

    async for _ in generate_response(
        hint_plan(),
        "just tell me",
        topic="Two-Step Linear Equations",
        provider=provider,
        settings=Settings(),
        system_note="Your previous draft revealed the solution.",
    ):
        pass

    assert "### NOTE" in provider.system
    assert "revealed the solution" in provider.system


async def test_conversation_history_reaches_call_c() -> None:
    provider = RecordingProvider()
    history = [
        Msg(role="user", content="what is a linear equation?"),
        Msg(role="assistant", content="…"),
    ]

    async for _ in generate_response(
        hint_plan(),
        "and two-step ones?",
        topic="Two-Step Linear Equations",
        conversation=history,
        provider=provider,
        settings=Settings(),
    ):
        pass

    assert [message.content for message in provider.messages] == [
        "what is a linear equation?",
        "…",
        "and two-step ones?",
    ]


def test_answer_key_statement_prefers_english() -> None:
    """Call B reasons in English even for Thai courses (Tech Spec §8)."""
    thai_only = Problem(id="P99", statement_th="จงแก้สมการ: x + 1 = 2", answer="x = 1")

    assert CANONICAL.statement == "Solve: 2x + 4 = 10"
    assert thai_only.statement == "จงแก้สมการ: x + 1 = 2"


# ------------------------------------------- the plan's field names stay internal


def test_no_structure_token_reaches_the_prompt() -> None:
    """Real replies came back as "**State check:** … **Next step:** …".

    The model was handed `state_evaluation, explain_feedback, next_action` and
    reasonably read them as an output format. The fix is structural rather than
    an instruction: a token that is not in the prompt cannot be echoed out of it
    — the same reasoning as the withheld canonical answer.
    """
    from socratic_tutor.pedagogy.tables import get_tables

    blocks = sorted(get_tables().response_templates.blocks)
    prompt = build_system_prompt(
        hint_plan(structure=blocks), topic="Two-Step Linear Equations", chunks=CHUNKS
    )

    assert blocks, "the templates must declare some blocks for this to mean anything"
    leaked = [name for name in blocks if name in prompt]
    assert leaked == [], f"internal block names reached the prompt: {leaked}"


def test_structure_is_rendered_as_instructions() -> None:
    policy = section(
        build_system_prompt(
            hint_plan(structure=["single_hint", "student_attempt_prompt"]),
            topic="Two-Step Linear Equations",
        ),
        POLICY_HEADING,
    )

    assert "give exactly one hint" in policy
    assert "invite them to try it" in policy
    assert "no headings" in policy and "no bold labels" in policy
    assert "internal specification, not a format" in policy


def test_an_unlisted_block_still_reads_as_words() -> None:
    """A new block in the templates must not fall back to printing its token."""
    assert describe_structure(["some_new_block"]) == ["some new block"]
    assert describe_structure([]) == []
