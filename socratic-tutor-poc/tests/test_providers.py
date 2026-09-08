"""Provider abstraction tests — R21. No live API calls anywhere in this file.

R21's acceptance criterion is "swapping Call C to an OpenRouter model is a
config-only change", so the factory tests drive everything from `Settings` and
the adapter tests assert on the request body an endpoint would actually receive,
served by an httpx `MockTransport`.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from socratic_tutor.config import ConfigError, Settings
from socratic_tutor.models import (
    Intent,
    IntentClassification,
    IntentResult,
    LearnerState,
    SpecialHandling,
    SpecialHandlingResult,
)
from socratic_tutor.providers import get_provider
from socratic_tutor.providers.anthropic_provider import AnthropicProvider, build_tool
from socratic_tutor.providers.base import (
    Msg,
    ProviderConfigError,
    ProviderError,
    ProviderModelNotFound,
    ProviderRateLimited,
    ProviderResponseError,
    StreamStats,
    strict_json_schema,
)
from socratic_tutor.providers.claude_subscription import ClaudeSubscriptionProvider
from socratic_tutor.providers.openai_compat import OpenAICompatProvider

MESSAGES = [Msg(role="user", content="Is x = 4 right?")]


def wire(intent: Intent, confidence: float = 0.9) -> IntentClassification:
    """A complete Call A wire payload. Every field is required in strict mode."""
    return IntentClassification(
        intent=intent,
        learner_state=LearnerState.NORMAL,
        special_handling=SpecialHandlingResult(),
        confidence=confidence,
    )


# ------------------------------------------------------------------ factory


def test_factory_defaults_every_role_to_the_subscription() -> None:
    """The zero-provisioning path: the machine's existing `claude login`, no keys."""
    settings = Settings()
    for role in ("intent", "evaluation", "generation"):
        assert isinstance(get_provider(role, settings), ClaudeSubscriptionProvider)


def test_factory_swaps_one_role_to_openai_compat() -> None:
    """R21: Call C moves to OpenRouter without touching a line of pipeline code."""
    settings = Settings(
        generation_provider="openai_compat",
        generation_model="qwen/qwen3-32b-instruct",
        openai_compat_base_url="https://openrouter.ai/api/v1",
        openai_compat_api_key="sk-test",
    )
    assert isinstance(get_provider("generation", settings), OpenAICompatProvider)
    assert isinstance(get_provider("intent", settings), ClaudeSubscriptionProvider)
    assert settings.model_for("generation") == "qwen/qwen3-32b-instruct"
    assert settings.model_for("intent") == "claude-haiku-4-5"


def test_default_models_follow_the_addendum() -> None:
    settings = Settings()
    assert settings.model_for("intent") == "claude-haiku-4-5"
    assert settings.model_for("evaluation") == "claude-haiku-4-5"
    assert settings.model_for("generation") == "claude-sonnet-5"
    assert settings.max_tokens_for("generation") > settings.max_tokens_for("intent")


def test_unknown_role_is_rejected() -> None:
    with pytest.raises(ConfigError):
        Settings().model_for("guardrail")


def test_openai_compat_without_a_key_fails_with_a_named_variable() -> None:
    settings = Settings(generation_provider="openai_compat", openai_compat_api_key=None)
    with pytest.raises(ProviderConfigError, match="POC_OPENAI_COMPAT_API_KEY"):
        get_provider("generation", settings)


def test_no_provider_construction_requires_a_credential_up_front() -> None:
    """Neither Claude path checks for a key: the subscription has none, and the
    Anthropic SDK resolves its own. Only `openai_compat` needs one, and it says so."""
    assert isinstance(get_provider("intent", Settings()), ClaudeSubscriptionProvider)
    assert isinstance(
        get_provider("intent", Settings(intent_provider="anthropic")), AnthropicProvider
    )


def test_derived_paths() -> None:
    settings = Settings()
    assert settings.chroma_dir == settings.data_dir / "chroma"
    assert settings.uploads_dir == settings.data_dir / "uploads"


def test_dump_never_reveals_the_key() -> None:
    dumped = Settings(openai_compat_api_key="sk-secret").dump()
    assert dumped["openai_compat_api_key"] == "present"
    assert "sk-secret" not in json.dumps(dumped)


# -------------------------------------------------- strict schema / tool use


def test_strict_schema_inlines_refs_and_requires_every_field() -> None:
    schema = strict_json_schema(IntentClassification)

    assert "$defs" not in json.dumps(schema)
    assert "$ref" not in json.dumps(schema)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"intent", "learner_state", "special_handling", "confidence"}

    nested = schema["properties"]["special_handling"]
    assert nested["additionalProperties"] is False
    assert set(nested["required"]) == {"detected", "type"}

    assert schema["properties"]["intent"]["enum"] == [i.value for i in Intent]
    assert nested["properties"]["type"]["enum"] == [s.value for s in SpecialHandling]


def test_strict_schema_drops_defaults_and_constraints() -> None:
    dumped = json.dumps(strict_json_schema(IntentClassification))
    for keyword in ("default", "minimum", "maximum", "pattern"):
        assert f'"{keyword}"' not in dumped


def test_strict_schema_refuses_a_pipeline_model() -> None:
    """Every field becomes required, so a system-computed field would be fabricated.

    `IntentResult` carries `low_confidence_fallback` and `raw_intent`, both decided
    by the pipeline *after* reading the model's output. Handing them to the model
    would have it self-report a decision it cannot know, and the glass-box UI would
    render the invention as a system fact. The builder refuses outright.
    """
    with pytest.raises(ProviderError, match="system-computed"):
        strict_json_schema(IntentResult)

    with pytest.raises(ProviderError, match="system-computed"):
        build_tool(IntentResult)


def test_strict_schema_rejects_free_form_objects() -> None:
    from pydantic import BaseModel

    class HasFreeDict(BaseModel):
        context: dict[str, Any]

    with pytest.raises(ProviderError, match="free-form object"):
        strict_json_schema(HasFreeDict)


def test_build_tool_forces_strict_single_tool() -> None:
    tool = build_tool(IntentClassification)
    assert tool["name"] == "emit_intent_classification"
    assert tool["strict"] is True
    assert tool["description"]
    assert tool["input_schema"] == strict_json_schema(IntentClassification)


# ------------------------------------------------- openai_compat: structured


def compat_provider(handler: Any) -> OpenAICompatProvider:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://openrouter.ai/api/v1",
        headers={"Authorization": "Bearer sk-test"},
    )
    return OpenAICompatProvider(client=client)


def chat_completion(content: str) -> dict[str, Any]:
    return {
        "model": "qwen/qwen3-8b-instruct",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 18},
    }


async def test_openai_compat_builds_a_correct_structured_request() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        payload = wire(Intent.HINT, 0.8).model_dump(mode="json")
        return httpx.Response(200, json=chat_completion(json.dumps(payload)))

    provider = compat_provider(handler)
    result = await provider.complete_structured(
        model="qwen/qwen3-8b-instruct",
        system="You classify student intent.",
        messages=MESSAGES,
        schema=IntentClassification,
        max_tokens=512,
        temperature=0.0,
    )
    await provider.aclose()

    assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"

    body = seen["body"]
    assert body["model"] == "qwen/qwen3-8b-instruct"
    assert body["max_tokens"] == 512
    assert body["temperature"] == 0.0
    assert body["messages"] == [
        {"role": "system", "content": "You classify student intent."},
        {"role": "user", "content": "Is x = 4 right?"},
    ]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["name"] == "IntentClassification"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["schema"] == strict_json_schema(
        IntentClassification
    )

    assert isinstance(result.value, IntentClassification)
    assert result.value.intent is Intent.HINT
    assert result.usage.input_tokens == 120
    assert result.usage.output_tokens == 18
    assert result.latency_ms >= 0.0
    assert result.provider == "openai_compat"


async def test_openai_compat_rejects_content_that_is_not_the_schema() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=chat_completion('{"intent": "not_an_intent"}'))

    provider = compat_provider(handler)
    with pytest.raises(ProviderResponseError, match="does not match IntentClassification"):
        await provider.complete_structured(
            model="m", system="s", messages=MESSAGES, schema=IntentClassification
        )
    await provider.aclose()


async def test_openai_compat_rejects_non_json_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=chat_completion("I think it is correct!"))

    provider = compat_provider(handler)
    with pytest.raises(ProviderResponseError, match="unparseable JSON"):
        await provider.complete_structured(
            model="m", system="s", messages=MESSAGES, schema=IntentClassification
        )
    await provider.aclose()


@pytest.mark.parametrize(
    ("status", "expected"),
    [(404, ProviderModelNotFound), (429, ProviderRateLimited), (500, ProviderError)],
)
async def test_openai_compat_maps_http_errors(status: int, expected: type[Exception]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "nope"}})

    provider = compat_provider(handler)
    with pytest.raises(expected):
        await provider.complete_structured(
            model="m", system="s", messages=MESSAGES, schema=IntentClassification
        )
    await provider.aclose()


# ---------------------------------------------------- openai_compat: stream


def sse(*frames: str) -> bytes:
    return "".join(f"data: {frame}\n\n" for frame in frames).encode()


async def test_openai_compat_streams_text_deltas_and_records_usage() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        body = sse(
            json.dumps({"choices": [{"delta": {"role": "assistant"}}]}),
            json.dumps({"choices": [{"delta": {"content": "Let's "}}]}),
            json.dumps({"choices": [{"delta": {"content": "work through it."}}]}),
            json.dumps({"choices": [], "usage": {"prompt_tokens": 900, "completion_tokens": 12}}),
            "[DONE]",
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    provider = compat_provider(handler)
    stats = StreamStats()
    chunks = [
        delta
        async for delta in provider.stream_text(
            model="qwen/qwen3-32b-instruct",
            system="You are a tutor.",
            messages=MESSAGES,
            max_tokens=2000,
            stats=stats,
        )
    ]
    await provider.aclose()

    assert "".join(chunks) == "Let's work through it."
    assert seen["body"]["stream"] is True
    assert seen["body"]["stream_options"] == {"include_usage": True}
    assert stats.usage.input_tokens == 900
    assert stats.usage.output_tokens == 12
    assert stats.ttft_ms is not None and stats.ttft_ms >= 0.0
    assert stats.total_ms is not None and stats.total_ms >= 0.0
    assert stats.provider == "openai_compat"


async def test_openai_compat_stream_maps_http_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "slow down"})

    provider = compat_provider(handler)
    with pytest.raises(ProviderRateLimited):
        async for _ in provider.stream_text(model="m", system="s", messages=MESSAGES):
            pass
    await provider.aclose()


# ------------------------------------------------------- anthropic: parsing


class FakeBlock:
    def __init__(self, type: str, name: str = "", input: Any = None) -> None:
        self.type, self.name, self.input = type, name, input


class FakeUsage:
    input_tokens = 210
    output_tokens = 24


class FakeResponse:
    def __init__(self, content: list[FakeBlock], stop_reason: str = "tool_use") -> None:
        self.content, self.stop_reason = content, stop_reason
        self.model = "claude-haiku-4-5"
        self.usage = FakeUsage()


class FakeMessages:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.last_request: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.last_request = kwargs
        return self._response


class FakeClient:
    def __init__(self, response: Any) -> None:
        self.messages = FakeMessages(response)

    async def close(self) -> None:
        return None


async def test_anthropic_forces_the_tool_and_parses_its_input() -> None:
    payload = wire(Intent.SOLVE, 0.88).model_dump(mode="json")
    client = FakeClient(
        FakeResponse([FakeBlock("tool_use", "emit_intent_classification", payload)])
    )
    provider = AnthropicProvider(client=client)

    result = await provider.complete_structured(
        model="claude-haiku-4-5",
        system="You classify student intent.",
        messages=MESSAGES,
        schema=IntentClassification,
    )

    request = client.messages.last_request
    assert request["tool_choice"] == {"type": "tool", "name": "emit_intent_classification"}
    assert len(request["tools"]) == 1
    assert request["tools"][0]["strict"] is True
    # No sampling parameter: the current models reject them and the SDK has
    # removed them (see test_request_kwargs_bind_against_the_installed_sdk).
    assert "temperature" not in request
    assert request["system"] == "You classify student intent."
    assert request["messages"] == [{"role": "user", "content": "Is x = 4 right?"}]

    assert isinstance(result.value, IntentClassification)
    assert result.value.intent is Intent.SOLVE
    assert result.usage.input_tokens == 210
    assert result.model == "claude-haiku-4-5"
    assert result.provider == "anthropic"


async def test_anthropic_parses_tool_input_delivered_as_a_json_string() -> None:
    payload = json.dumps(wire(Intent.CLARIFY).model_dump(mode="json"))
    provider = AnthropicProvider(
        client=FakeClient(
            FakeResponse([FakeBlock("tool_use", "emit_intent_classification", payload)])
        )
    )
    result = await provider.complete_structured(
        model="claude-haiku-4-5", system="s", messages=MESSAGES, schema=IntentClassification
    )
    assert result.value.intent is Intent.CLARIFY


async def test_anthropic_errors_when_no_tool_block_came_back() -> None:
    provider = AnthropicProvider(
        client=FakeClient(FakeResponse([FakeBlock("text")], stop_reason="end_turn"))
    )
    with pytest.raises(ProviderResponseError, match="no tool_use block"):
        await provider.complete_structured(
            model="claude-haiku-4-5", system="s", messages=MESSAGES, schema=IntentClassification
        )


# ---------------------------------- anthropic: the request the SDK will accept


def test_request_kwargs_bind_against_the_installed_sdk() -> None:
    """An argument the SDK has removed must fail here, not on first live contact.

    `anthropic` 1.2.0 dropped `temperature`/`top_p` from `messages.create()`, and
    the current Claude models reject sampling parameters outright. Binding the
    real kwargs against the real signature catches that at CI time.
    """
    import inspect

    from anthropic.resources.messages import AsyncMessages

    from socratic_tutor.providers.anthropic_provider import SAMPLING_PARAMS, request_kwargs

    kwargs = request_kwargs(
        model="claude-haiku-4-5",
        max_tokens=512,
        system="classify this",
        messages=MESSAGES,
        tool=build_tool(IntentClassification),
    )
    # Raises TypeError if any key is not a parameter of the installed method.
    inspect.signature(AsyncMessages.create).bind(None, **kwargs)

    streaming = request_kwargs(
        model="claude-sonnet-5", max_tokens=2000, system="teach", messages=MESSAGES
    )
    inspect.signature(AsyncMessages.stream).bind(None, **streaming)

    assert not SAMPLING_PARAMS & set(kwargs)
    assert not SAMPLING_PARAMS & set(streaming)


def test_the_sdk_has_indeed_removed_sampling_parameters() -> None:
    """Pins the reason the rule exists, so a future SDK bump makes it re-decidable."""
    import inspect

    from anthropic.resources.messages import AsyncMessages

    from socratic_tutor.providers.anthropic_provider import SAMPLING_PARAMS

    accepted = set(inspect.signature(AsyncMessages.create).parameters)
    assert not SAMPLING_PARAMS & accepted


async def test_structured_call_sends_no_sampling_parameter() -> None:
    """Even though `complete_structured` accepts `temperature` for interface parity."""
    payload = wire(Intent.SOLVE).model_dump(mode="json")
    client = FakeClient(
        FakeResponse([FakeBlock("tool_use", "emit_intent_classification", payload)])
    )
    provider = AnthropicProvider(client=client)

    await provider.complete_structured(
        model="claude-haiku-4-5",
        system="s",
        messages=MESSAGES,
        schema=IntentClassification,
        temperature=0.0,
    )
    assert "temperature" not in client.messages.last_request
