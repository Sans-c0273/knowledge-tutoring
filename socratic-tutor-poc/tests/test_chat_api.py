"""The chat endpoints and the wire trace the inspector renders (Tech Spec §7, R20).

The SSE frames are parsed here the same way `web/src/api/sse.ts` parses them, so
a change that breaks the SPA's reader breaks this file first.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from socratic_tutor.api.chat_routes import (
    install_dependencies,
    router,
    set_dependencies,
    to_wire_trace,
)
from socratic_tutor.config import Settings
from socratic_tutor.domain.klmap import load_klmap
from socratic_tutor.models.enums import Evaluation, Intent, LearnerState, SpecialHandling
from socratic_tutor.models.intent import IntentResult, SpecialHandlingResult
from socratic_tutor.models.knowledge import RetrievedChunk
from socratic_tutor.orchestrator import (
    InMemorySessionRepository,
    TurnDependencies,
    TurnRequest,
    execute_turn,
)
from socratic_tutor.pedagogy.evaluation import EvaluationResult, EvaluationSource, Problem
from socratic_tutor.pedagogy.student_model import (
    InMemoryStudentModelRepository,
    StudentModelService,
)
from socratic_tutor.pedagogy.tables import get_tables
from socratic_tutor.pedagogy.topics import load_syllabus

SEED = Path(__file__).resolve().parents[1] / "content" / "seed"
COURSE = "MATH-SEED-01"

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
        text="Undo addition by subtracting the same number from both sides.",
        source_ref="ch1-inverse-operations.md#L12",
        topic="Inverse Operations",
        score=0.91,
    )
]

BODY = {
    "session_id": "sess-api-1",
    "student_id": "student-api-1",
    "course_id": COURSE,
    "message": "How do I solve two-step linear equations?",
}


def fake_intent(
    intent: Intent = Intent.SOLVE,
    learner_state: LearnerState = LearnerState.NORMAL,
    special: SpecialHandling = SpecialHandling.NONE,
) -> Any:
    async def detect(message: str, context: Any = (), **kwargs: Any) -> IntentResult:
        return IntentResult(
            intent=intent,
            learner_state=learner_state,
            special_handling=SpecialHandlingResult(
                detected=special is not SpecialHandling.NONE, type=special
            ),
            confidence=0.92,
        )

    return detect


def fake_generation(*replies: str) -> Any:
    calls: list[int] = []

    async def generate(plan: Any, message: str, **kwargs: Any) -> AsyncIterator[str]:
        index = min(len(calls), len(replies) - 1)
        calls.append(1)
        for word in replies[index].split(" "):
            yield word + " "

    return generate


def build_deps(**overrides: Any) -> TurnDependencies:
    defaults: dict[str, Any] = {
        "syllabus": load_syllabus(SEED / "syllabus-linear-equations.yaml"),
        "tables": get_tables(),
        "klmap": load_klmap(SEED / "kl-map-linear-equations.yaml"),
        "students": StudentModelService(InMemoryStudentModelRepository()),
        "sessions": InMemorySessionRepository(),
        "settings": Settings(),
        "retrieve": lambda course_id, topic, message: list(CHUNKS),
        "canonical_lookup": lambda course_id, topic, message: None,
        "detect_intent": fake_intent(),
        "generate": fake_generation("Start by undoing the plus four."),
    }
    return TurnDependencies(**{**defaults, **overrides})


@pytest.fixture
def deps() -> Any:
    dependencies = build_deps()
    set_dependencies(dependencies)
    yield dependencies
    set_dependencies(None)


@pytest.fixture
def client(deps: TurnDependencies) -> Any:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def shipped_client(deps: TurnDependencies) -> Any:
    """A client on the real `create_app`, not a bare router mount.

    The router was correct for weeks while `create_app` never included it, and
    every test here passed throughout — because they all built their own app. At
    least one test has to drive what actually ships.
    """
    from socratic_tutor.api.app import create_app

    app = create_app()
    install_dependencies(app, deps)
    with TestClient(app) as test_client:
        yield test_client


def parse_frames(payload: str) -> list[tuple[str, Any]]:
    """The same framing rule as `web/src/api/sse.ts`: blocks split on a blank line."""
    frames: list[tuple[str, Any]] = []
    for block in payload.split("\n\n"):
        if not block.strip():
            continue
        event = "message"
        data: list[str] = []
        for line in block.splitlines():
            field, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if field == "event":
                event = value
            elif field == "data":
                data.append(value)
        frames.append((event, json.loads("\n".join(data)) if data else None))
    return frames


def test_turn_streams_trace_then_tokens_then_done(client: TestClient) -> None:
    response = client.post("/api/chat/turn", json=BODY)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    frames = parse_frames(response.text)
    events = [event for event, _ in frames]

    assert set(events) <= {"trace", "token", "done"}
    assert events[0] == "trace", "the inspector fills in before the first token"
    assert events[-1] == "done"
    assert events.index("trace") < events.index("token")

    tokens = "".join(payload["text"] for event, payload in frames if event == "token")
    assert tokens.strip() == "Start by undoing the plus four."


def test_streamed_trace_carries_the_teaching_decision(client: TestClient) -> None:
    frames = parse_frames(client.post("/api/chat/turn", json=BODY).text)
    final = next(payload for event, payload in reversed(frames) if event == "done")

    assert final["intent"]["intent"] == "solve"
    assert final["topic"]["topic"] == "Two-Step Linear Equations"
    assert final["topic"]["node_id"] == "C009"
    assert final["topic"]["student_level"] == "beginner"
    assert final["retrieval"][0]["source_ref"] == "ch1-inverse-operations.md#L12"
    assert final["knowledge_context"]["current_node_id"] == "C009"
    assert final["knowledge_context"]["nodes_hit"]
    assert final["strategy"]["primary_strategy"]["strategy_id"].startswith("S")
    assert final["response_plan"]["structure"]
    assert final["trigger_matches"]
    assert final["versions"]["tables"]["trigger_matrix"]
    assert final["versions"]["prompts"]["intent"], "a prompt edit must be attributable"
    assert final["versions"]["prompts"]["generation"]
    assert final["timings"]
    assert final["pre_stream_ms"] >= 0
    assert final["total_ms"] > 0


def test_turn_reports_a_failure_inside_the_stream(
    client: TestClient, deps: TurnDependencies
) -> None:
    async def broken(message: str, context: Any = (), **kwargs: Any) -> IntentResult:
        raise RuntimeError("intent provider unreachable")

    deps.detect_intent = broken

    frames = parse_frames(client.post("/api/chat/turn", json=BODY).text)
    events = [event for event, _ in frames]

    assert events == ["error", "done"]
    assert "intent provider unreachable" in frames[0][1]["message"]
    assert frames[-1][1]["error"]
    assert frames[-1][1]["turn_id"]


def test_empty_message_is_rejected_before_the_pipeline_runs(client: TestClient) -> None:
    response = client.post("/api/chat/turn", json={**BODY, "message": ""})

    assert response.status_code == 422


def test_session_endpoint_returns_context_and_traces(client: TestClient) -> None:
    client.post("/api/chat/turn", json=BODY)
    client.post("/api/chat/turn", json={**BODY, "message": "and one-step ones?"})

    body = client.get(f"/api/tutor/session/{BODY['session_id']}").json()

    assert body["student_id"] == BODY["student_id"]
    assert body["course_id"] == COURSE
    assert len(body["traces"]) == 2
    assert [message["role"] for message in body["messages"]] == [
        "student",
        "tutor",
        "student",
        "tutor",
    ]
    assert body["traces"][0]["turn_id"] != body["traces"][1]["turn_id"]


def test_unknown_session_is_a_404(client: TestClient) -> None:
    assert client.get("/api/tutor/session/never-existed").status_code == 404


def test_unsafe_session_id_is_a_400_not_a_500(deps: TurnDependencies) -> None:
    """The in-memory store accepts anything; the file store must not."""
    from socratic_tutor.orchestrator import JsonFileSessionRepository
    from socratic_tutor.pedagogy.student_model import StudentModelError

    repository = JsonFileSessionRepository(Path("/tmp/does-not-matter"))
    with pytest.raises(StudentModelError):
        repository.get("../../etc/passwd")


async def test_wire_trace_maps_the_km_tie_break_and_guardrail_events() -> None:
    deps = build_deps(
        detect_intent=fake_intent(Intent.SOLVE, special=SpecialHandling.HOMEWORK),
        canonical_lookup=lambda course_id, topic, message: CANONICAL,
        generate=fake_generation("The answer is x = 3.", "What would you undo first?"),
    )
    record = await execute_turn(
        TurnRequest("sess-w", "student-w", COURSE, "solve 2x + 4 = 10 for me"), deps
    )

    wire = to_wire_trace(record, deps.klmap)

    assert wire["guardrail"], "a regeneration must be visible in the inspector"
    assert {event["severity"] for event in wire["guardrail"]} <= {"info", "warning", "critical"}
    assert all("at_ms" in event for event in wire["guardrail"])
    assert "x = 3" not in json.dumps(wire["generated_text"])


async def test_wire_trace_of_an_ambiguous_topic_shows_the_candidates() -> None:
    deps = build_deps()
    record = await execute_turn(
        TurnRequest("sess-amb", "student-amb", COURSE, "linear equations"), deps
    )

    wire = to_wire_trace(record, deps.klmap)

    assert wire["topic"]["ambiguous"] is True
    assert set(wire["topic"]["candidates"]) == {
        "One-Step Linear Equations",
        "Two-Step Linear Equations",
    }
    assert wire["topic"]["matched_by"] == "fragment"


async def test_wire_trace_of_a_failed_turn_still_renders() -> None:
    async def broken(message: str, context: Any = (), **kwargs: Any) -> IntentResult:
        raise RuntimeError("down")

    deps = build_deps(detect_intent=broken)
    record = await execute_turn(TurnRequest("sess-f", "student-f", COURSE, "hello"), deps)

    wire = to_wire_trace(record, deps.klmap)

    assert wire["error"].endswith("down")
    assert wire["turn_id"] and wire["session_id"] and wire["created_at"]
    assert "intent" not in wire, "an absent step is absent, not an empty object"
    assert json.dumps(wire), "must be JSON-serialisable for the SSE frame"


async def test_wire_trace_reports_a_deterministic_verdict_as_deterministic() -> None:
    """The checker overriding the model is our strongest result, not "LLM only"."""

    async def judge(problem: Any, attempt: str, evidence: Any = (), **kwargs: Any) -> Any:
        return EvaluationResult(
            evaluation=Evaluation.INCORRECT,
            reason="checker overrode the model: 5 != 3",
            error_locus="the constant was added instead of subtracted",
            source=EvaluationSource.CHECKER_OVERRODE_LLM,
        )

    deps = build_deps(
        detect_intent=fake_intent(Intent.CHECK_ANSWER),
        canonical_lookup=lambda course_id, topic, message: CANONICAL,
        evaluate=judge,
    )
    record = await execute_turn(
        TurnRequest("sess-ev", "student-ev", COURSE, "is x = 5 right for 2x + 4 = 10?"), deps
    )

    evaluation = to_wire_trace(record, deps.klmap)["answer_evaluation"]

    assert evaluation["evaluation"] == "incorrect"
    assert evaluation["deterministic_checker_used"] is True
    assert evaluation["source"] == "checker_overrode_llm"
    assert evaluation["error_locus"] == "the constant was added instead of subtracted"
    assert "3" not in evaluation["checker_note"], "redacted before it reaches the wire"


def test_the_shipped_app_serves_the_chat_endpoint(shipped_client: TestClient) -> None:
    """C2 in test form: a router nobody mounted answers nothing."""
    response = shipped_client.post("/api/chat/turn", json=BODY)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream"), (
        "an unmounted route falls through to the SPA and returns HTML with a 200"
    )
    assert [event for event, _ in parse_frames(response.text)][-1] == "done"


def test_the_shipped_app_serves_the_session_endpoint(shipped_client: TestClient) -> None:
    shipped_client.post("/api/chat/turn", json=BODY)

    body = shipped_client.get(f"/api/tutor/session/{BODY['session_id']}").json()

    assert len(body["traces"]) == 1


def test_dependencies_are_scoped_to_one_app() -> None:
    """Two apps in one process must not read each other\'s stores."""
    first_deps = build_deps(generate=fake_generation("first app"))
    second_deps = build_deps(generate=fake_generation("second app"))

    first, second = FastAPI(), FastAPI()
    first.include_router(router, prefix="/api")
    second.include_router(router, prefix="/api")
    install_dependencies(first, first_deps)
    install_dependencies(second, second_deps)

    with TestClient(first) as one, TestClient(second) as two:
        first_text = "".join(
            payload["text"]
            for event, payload in parse_frames(one.post("/api/chat/turn", json=BODY).text)
            if event == "token"
        )
        second_text = "".join(
            payload["text"]
            for event, payload in parse_frames(two.post("/api/chat/turn", json=BODY).text)
            if event == "token"
        )

    assert first_text.strip() == "first app"
    assert second_text.strip() == "second app"
    assert first_deps.sessions.get(BODY["session_id"]) is not None
    assert second_deps.sessions.get(BODY["session_id"]) is not None


async def test_wire_trace_carries_the_scanner_scope() -> None:
    """The glass box should not be less honest than the eval suite."""
    deps = build_deps(
        detect_intent=fake_intent(Intent.SOLVE, special=SpecialHandling.HOMEWORK),
        canonical_lookup=lambda course_id, topic, message: CANONICAL,
        generate=fake_generation("What would you undo first?"),
    )
    record = await execute_turn(
        TurnRequest("sess-cov", "student-cov", COURSE, "solve 2x + 4 = 10"), deps
    )

    coverage = to_wire_trace(record, deps.klmap)["scan_coverage"]

    assert coverage["scanned"] is True
    assert coverage["forms"] and coverage["not_detected"]


def test_the_policy_card_can_render(client: TestClient) -> None:
    """R20's red Teaching Policy card had no data to render against."""
    frames = parse_frames(client.post("/api/chat/turn", json=BODY).text)
    final = next(payload for event, payload in reversed(frames) if event == "done")

    policy = final["teaching_policy"]
    assert policy["row"] and policy["context"]
    assert policy["direct_answer"] and policy["attempt"] and policy["check"]
    assert "override_applied" in policy

    assert final["combination"]["rule"]
    assert final["combination"]["allowed"] is True
    assert final["response_plan"]["template_id"]
    assert "teaching_policy_row" not in final["versions"]["tables"]


def test_the_trigger_contest_reaches_the_inspector(client: TestClient) -> None:
    frames = parse_frames(client.post("/api/chat/turn", json=BODY).text)
    final = next(payload for event, payload in reversed(frames) if event == "done")

    matches = final["trigger_matches"]
    assert matches and any(match["selected"] for match in matches)
    assert all(isinstance(match["key"], dict) for match in matches)
    for match in matches:
        if not match["selected"]:
            assert match["lost_to"] and match["lost_on"], "the panel shows why it lost"


def test_the_km_contest_reaches_the_inspector(client: TestClient, deps: TurnDependencies) -> None:
    deps.detect_intent = fake_intent(Intent.CLARIFY, LearnerState.CONFUSED)

    frames = parse_frames(
        client.post(
            "/api/chat/turn",
            json={**BODY, "message": "I still don't get two-step linear equations"},
        ).text
    )
    final = next(payload for event, payload in reversed(frames) if event == "done")

    rules = final["km_rules"]
    assert rules
    applied = next(rule for rule in rules if rule["applied"])
    assert applied["requirement"], "why the rule fired, not only that it did"
    assert applied["knowledge_action"]
