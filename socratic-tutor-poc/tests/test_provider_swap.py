"""R21 — swapping Call C to an OpenAI-compatible provider is a config-only change.

R21's acceptance criterion, stated as assertions. The same orchestrator runs the
same turn twice over the same seed content and the same rule tables; the only
difference between the two runs is `Settings`. One lands on Anthropic, the other
builds a well-formed OpenAI `chat/completions` request for OpenRouter.

Nothing here touches a network. Only the transport is faked — the httpx socket on
one side, the Anthropic SDK client on the other — so the orchestrator, the
strategy tables, the planner, the four-layer prompt build and both adapters are
all the real thing. That matters: a test that stubbed `get_provider` would prove
the stub swaps, not that the system does.
"""

from __future__ import annotations

import ast
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Self

import httpx
import pytest

from socratic_tutor.config import PROJECT_ROOT, Settings
from socratic_tutor.domain.klmap import load_klmap
from socratic_tutor.models.enums import Intent, Language
from socratic_tutor.models.intent import IntentResult
from socratic_tutor.models.knowledge import RetrievedChunk
from socratic_tutor.orchestrator import (
    InMemorySessionRepository,
    TurnDependencies,
    TurnRequest,
    execute_turn,
)
from socratic_tutor.pedagogy.student_model import (
    InMemoryStudentModelRepository,
    StudentModelService,
)
from socratic_tutor.pedagogy.tables import get_tables
from socratic_tutor.pedagogy.topics import load_syllabus
from socratic_tutor.providers.anthropic_provider import AnthropicProvider
from socratic_tutor.providers.claude_subscription import ClaudeSubscriptionProvider
from socratic_tutor.providers.openai_compat import OpenAICompatProvider

SEED = PROJECT_ROOT / "content" / "seed"
COURSE = "MATH-SEED-01"
MESSAGE = "How do I solve two-step linear equations?"
REPLY = "What could you subtract from both sides first?"

OPENROUTER_MODEL = "qwen/qwen3-32b-instruct"

CHUNKS = [
    RetrievedChunk(
        text="Undo addition by subtracting the same number from both sides.",
        source_ref="ch1-inverse-operations.md#L12",
        topic="Two-Step Linear Equations",
        lang=Language.EN,
        score=0.81,
    )
]


async def fake_detect_intent(message: str, conversation: Any = (), **kwargs: Any) -> IntentResult:
    """Call A stubbed — R21 is about Call C, and A must not reach a network either."""
    return IntentResult(intent=Intent.SOLVE, confidence=0.95)


def deps_for(settings: Settings) -> TurnDependencies:
    """Real tables, real seed content, real generation. Only Call A is stubbed."""
    return TurnDependencies(
        syllabus=load_syllabus(SEED / "syllabus-linear-equations.yaml"),
        tables=get_tables(),
        klmap=load_klmap(SEED / "kl-map-linear-equations.yaml"),
        students=StudentModelService(InMemoryStudentModelRepository()),
        sessions=InMemorySessionRepository(),
        settings=settings,
        retrieve=lambda course_id, topic, message: list(CHUNKS),
        canonical_lookup=lambda course_id, topic, message: None,
        detect_intent=fake_detect_intent,
        # `generate` is deliberately left at its default, the real
        # `pedagogy.generation.generate_response`, which resolves its provider
        # from config. That resolution is what this file exists to test.
    )


def turn_request() -> TurnRequest:
    return TurnRequest(
        session_id="sess-r21", student_id="student-1", course_id=COURSE, message=MESSAGE
    )


# ------------------------------------------------------------ fake transports


def sse(*frames: str) -> bytes:
    return "".join(f"data: {frame}\n\n" for frame in frames).encode()


def openai_stream_body(text: str) -> bytes:
    return sse(
        json.dumps({"choices": [{"delta": {"role": "assistant"}}]}),
        json.dumps({"choices": [{"delta": {"content": text}}]}),
        json.dumps({"choices": [], "usage": {"prompt_tokens": 900, "completion_tokens": 11}}),
        "[DONE]",
    )


@pytest.fixture
def openrouter(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Force every httpx client built inside the adapter onto a MockTransport.

    Patching the socket rather than the provider is the point: `get_provider`
    constructs a real `OpenAICompatProvider` with a real client, exactly as it
    would in production, and only the wire is fake.
    """
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=openai_stream_body(REPLY),
            headers={"content-type": "text/event-stream"},
        )

    real_client = httpx.AsyncClient

    class MockedAsyncClient(real_client):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs: Any) -> None:
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(**kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", MockedAsyncClient)
    return seen


class FakeAnthropicStream:
    def __init__(self, text: str, seen: dict[str, Any], kwargs: dict[str, Any]) -> None:
        self._text, self._seen, self._kwargs = text, seen, kwargs

    async def __aenter__(self) -> Self:
        self._seen["request"] = self._kwargs
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    @property
    async def text_stream(self) -> AsyncIterator[str]:  # pragma: no cover - see __aiter__
        yield self._text

    async def get_final_message(self) -> Any:
        class Usage:
            input_tokens, output_tokens = 850, 11

        class Message:
            usage, stop_reason, model = Usage(), "end_turn", "claude-sonnet-5"

        return Message()


class FakeAnthropicMessages:
    def __init__(self, text: str, seen: dict[str, Any]) -> None:
        self._text, self._seen = text, seen

    def stream(self, **kwargs: Any) -> FakeAnthropicStream:
        return FakeAnthropicStream(self._text, self._seen, kwargs)


class FakeAnthropicClient:
    def __init__(self, text: str, seen: dict[str, Any]) -> None:
        self.messages = FakeAnthropicMessages(text, seen)

    async def close(self) -> None:
        return None


@pytest.fixture
def anthropic_wire(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Same idea for the native path: fake the SDK client, keep the adapter real."""
    seen: dict[str, Any] = {}
    from socratic_tutor.providers import anthropic_provider

    monkeypatch.setattr(
        anthropic_provider.anthropic,
        "AsyncAnthropic",
        lambda *a, **k: FakeAnthropicClient(REPLY, seen),
    )
    return seen


# --------------------------------------------------------------- the swap


def openrouter_settings() -> Settings:
    return Settings(
        generation_provider="openai_compat",
        generation_model=OPENROUTER_MODEL,
        openai_compat_base_url="https://openrouter.ai/api/v1",
        openai_compat_api_key="sk-test",
    )


async def test_call_c_on_openrouter_is_config_only(openrouter: dict[str, Any]) -> None:
    """The headline: same turn, provider chosen by `Settings` alone."""
    record = await execute_turn(turn_request(), deps_for(openrouter_settings()))

    assert record.trace.error is None
    assert record.trace.generated_text == REPLY
    assert record.trace.model_ids["generation"] == OPENROUTER_MODEL

    assert openrouter["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert openrouter["headers"]["authorization"] == "Bearer sk-test"


async def test_the_openrouter_request_body_is_well_formed(openrouter: dict[str, Any]) -> None:
    """What an OpenAI-compatible endpoint actually has to accept."""
    await execute_turn(turn_request(), deps_for(openrouter_settings()))
    body = openrouter["body"]

    assert body["model"] == OPENROUTER_MODEL
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert isinstance(body["max_tokens"], int) and body["max_tokens"] > 0

    # OpenAI carries the system prompt as the first message; Anthropic does not.
    # The adapter, not the pipeline, is what knows the difference.
    roles = [message["role"] for message in body["messages"]]
    assert roles[0] == "system"
    assert roles[-1] == "user"
    assert set(roles) <= {"system", "user", "assistant"}
    assert all(isinstance(m["content"], str) and m["content"] for m in body["messages"])
    assert body["messages"][-1]["content"] == MESSAGE

    # The real four-layer prompt travelled, not a placeholder.
    system = body["messages"][0]["content"]
    assert "Two-Step Linear Equations" in system
    assert CHUNKS[0].source_ref in system


def anthropic_settings() -> Settings:
    return Settings(generation_provider="anthropic", generation_model="claude-sonnet-5")


async def test_the_same_turn_on_the_anthropic_config_goes_to_anthropic(
    anthropic_wire: dict[str, Any],
) -> None:
    """The control. Only `Settings` differs from the OpenRouter run above."""
    record = await execute_turn(turn_request(), deps_for(anthropic_settings()))

    assert record.trace.generated_text == REPLY
    assert record.trace.model_ids["generation"] == "claude-sonnet-5"

    request = anthropic_wire["request"]
    assert request["model"] == "claude-sonnet-5"
    # Anthropic takes the system prompt out of band, so no system-role message.
    assert [m["role"] for m in request["messages"]] == ["user"]
    assert isinstance(request["system"], str) and request["system"]


async def test_both_providers_receive_the_same_teaching_decision(
    openrouter: dict[str, Any],
) -> None:
    """Swapping the provider must not change what the pipeline decided.

    If the plan or the prompt differed between providers, "config-only" would be
    false however well-formed the request looked.
    """
    swapped = await execute_turn(turn_request(), deps_for(openrouter_settings()))
    openrouter_system = openrouter["body"]["messages"][0]["content"]

    assert swapped.trace.response_plan is not None
    assert swapped.trace.strategy_selection is not None

    # Re-run under the default config and compare the decision, not the wire.
    import socratic_tutor.providers.anthropic_provider as ap

    seen: dict[str, Any] = {}
    original = ap.anthropic.AsyncAnthropic
    ap.anthropic.AsyncAnthropic = lambda *a, **k: FakeAnthropicClient(REPLY, seen)  # type: ignore[assignment]
    try:
        native = await execute_turn(turn_request(), deps_for(anthropic_settings()))
    finally:
        ap.anthropic.AsyncAnthropic = original  # type: ignore[assignment]

    assert native.trace.response_plan == swapped.trace.response_plan
    assert native.trace.strategy_selection == swapped.trace.strategy_selection
    assert seen["request"]["system"] == openrouter_system


# ------------------------------------------------- "config-only", structurally


PROVIDER_NAMES = {"anthropic", "openai_compat"}
PROVIDER_MODULES = {
    "socratic_tutor.providers.anthropic_provider",
    "socratic_tutor.providers.openai_compat",
}


def application_modules() -> list[Path]:
    """Every module that is neither the provider package nor the settings."""
    root = PROJECT_ROOT / "src" / "socratic_tutor"
    return [
        path
        for path in sorted(root.rglob("*.py"))
        if "providers" not in path.relative_to(root).parts and path.name != "config.py"
    ]


@pytest.mark.parametrize("path", application_modules(), ids=lambda p: p.name)
def test_no_application_module_names_a_provider(path: Path) -> None:
    """R21 is only real if the pipeline cannot tell which provider it is using.

    A module that imports the Anthropic SDK, or branches on the string
    `"openai_compat"`, has hard-wired a choice that config is supposed to own —
    and the swap stops being config-only the moment one does.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value in PROVIDER_NAMES:
            pytest.fail(f"{path.name}:{node.lineno} hard-codes the provider name {node.value!r}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] != "anthropic", f"{path.name} imports the SDK"
                assert alias.name not in PROVIDER_MODULES, f"{path.name} imports a provider module"
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] != "anthropic", f"{path.name} imports the SDK"
            assert node.module not in PROVIDER_MODULES, f"{path.name} imports a provider module"


def test_the_factory_is_the_only_thing_that_picks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same role, same call, different class — decided by config and nothing else."""
    from socratic_tutor.providers import get_provider

    assert isinstance(get_provider("generation", Settings()), ClaudeSubscriptionProvider)
    assert isinstance(get_provider("generation", anthropic_settings()), AnthropicProvider)
    assert isinstance(get_provider("generation", openrouter_settings()), OpenAICompatProvider)
    # …while the roles that were not swapped stay where they were.
    assert isinstance(get_provider("intent", openrouter_settings()), ClaudeSubscriptionProvider)
