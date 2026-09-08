"""WU2 test doubles for the LLM boundary (DESIGN §7, §17).

Everything here is offline and deterministic: no subprocess, no socket, no clock.
The fakes are *transport-level* — they stand in for `claude_agent_sdk.query`, the
`openai` client and the `anthropic` client — so the real adapters are exercised
against them. They never import a kg.llm module at import time, so this file is
importable before the boundary exists.

Naming used by the WU2 tests (DESIGN §7/§20 names where given; the rest chosen
here and listed in the test-engineer report):

  kg.llm                          call_stage(stage, system, user_text, schema, images=None, *,
                                             cfg, ledger, adapter=None, prompt_version=None) -> StageResult
                                  StageResult(.data, .usage, .provider, .model_requested,
                                              .model_served, .attempts, .prompt_version)
                                  LLMError, StageOutputInvalid(.stage, .attempts), StageTruncated,
                                  BudgetExceeded(.spent_usd, .soft_budget_usd),
                                  RateLimited(.action, .wait_seconds, .resets_at, .rate_limit_type,
                                              .utilization), ProviderError
  kg.llm.ledger                   UsageLedger(soft_budget_usd=None) .record(row) .rows .spent_usd
                                  .check_budget() .per_stage() .total() .to_dict()
                                  UsageRow(...), cost_display(provider, cost_usd, cost_estimate_usd=None)
  kg.llm.adapters.base            ImageInput(media_type, data_b64) / .from_bytes(data, media_type)
                                  Msg(role, text, images=())
                                  Usage(input_tokens, output_tokens, cached_tokens=0, reasoning_tokens=0)
                                  RawCompletion(text_or_obj, usage, model_requested, model_served=None,
                                                cost_usd=None, stop_reason=None, provider_meta={}, events=[])
                                  Adapter protocol: .provider, .max_repairs (int | None),
                                                    .complete(system, messages, schema_json, model, max_tokens)
  kg.llm.adapters.claude_subscription
                                  ClaudeSubscriptionAdapter(cfg, *, query_fn=None, clock=None)
                                  rate_limit_policy(info, *, now, wait_minutes) -> RateLimitDecision
                                  RateLimitDecision(.action, .wait_seconds, .resets_at, .rate_limit_type, .utilization)
  kg.llm.adapters.openrouter      OpenRouterAdapter(cfg, *, client=None)  .client
  kg.llm.adapters.anthropic_api   AnthropicApiAdapter(cfg, *, client=None)
                                  (anthropic 1.0.0 `output_config` route: `parsed_output` is
                                  always None, the adapter returns the text block; DESIGN §7.5/D30)
  kg.config                       adapter_factory(cfg) -> Adapter
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from kg.config import Config, load_config

FAKE_OR_KEY = "sk-or-v1-TESTONLY-not-a-real-key-0123456789"
FAKE_ANTHROPIC_KEY = "sk-ant-TESTONLY-not-a-real-key-0123456789"

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + bytes(range(16))
PNG_B64 = base64.b64encode(PNG_BYTES).decode("ascii")
JPEG_BYTES = b"\xff\xd8\xff\xe0" + bytes(range(12))
JPEG_B64 = base64.b64encode(JPEG_BYTES).decode("ascii")

SYSTEM = "You are the describe stage. Return JSON only."
USER_TEXT = "Describe the informational content of this image."

# A DescribeOutput-compatible object (DESIGN §8.0 shape; the WU0 model is a subset
# of it — tests only ever assert on `description`, which both shapes carry).
GOOD_DESCRIBE_MIN = {"description": "A flow chart with three boxes."}


def describe_payload(schema: type) -> dict[str, Any]:
    """A valid instance dict for whichever DescribeOutput shape is currently built."""
    fields = set(schema.model_fields)
    payload: dict[str, Any] = {"description": "A flow chart with three boxes."}
    if "kind" in fields:
        payload["kind"] = "diagram"
    if "title" in fields:
        payload["title"] = "Process flow"
    for list_field in ("elements", "text_visible", "relationships"):
        if list_field in fields:
            payload[list_field] = []
    return payload


# ------------------------------------------------------------------- configs


def load_cfg(make_project: Callable[..., Path], provider: str = "claude_subscription", **overrides: Any) -> Config:
    """Load a Config for `provider` from a temp kg.yaml; fake credentials in .env where required."""
    env_text: str | None = None
    if provider == "openrouter":
        env_text = f"OPENROUTER_API_KEY={FAKE_OR_KEY}\n"
    elif provider == "anthropic_api":
        env_text = f"ANTHROPIC_API_KEY={FAKE_ANTHROPIC_KEY}\n"
    over = {"llm.provider": provider, **overrides}
    return load_config(make_project(over, env_text=env_text, subdir=f"proj-{provider}"))


# ---------------------------------------------------------- RawCompletion


def raw(text_or_obj: Any, **over: Any):
    """Build a kg.llm.adapters.base.RawCompletion with sane defaults (lazy import)."""
    from kg.llm.adapters.base import RawCompletion, Usage

    kw: dict[str, Any] = {
        "usage": Usage(input_tokens=100, output_tokens=20),
        "model_requested": "fake-model",
        "model_served": "fake-model-served",
        "cost_usd": 0.001,
        "stop_reason": "end_turn",
        "provider_meta": {},
        "events": [],
    }
    kw.update(over)
    return RawCompletion(text_or_obj=text_or_obj, **kw)


# -------------------------------------------------- Claude Agent SDK fakes


def sdk_result(**over: Any):
    """A real claude_agent_sdk.ResultMessage (plain dataclass; importing types spawns nothing)."""
    from claude_agent_sdk import ResultMessage

    kw: dict[str, Any] = {
        "subtype": "success",
        "duration_ms": 1200,
        "duration_api_ms": 1100,
        "is_error": False,
        "num_turns": 1,
        "session_id": "sess-test",
        "stop_reason": "end_turn",
        "total_cost_usd": 0.0123,
        "usage": {
            "input_tokens": 1500,
            "output_tokens": 300,
            "cache_read_input_tokens": 400,
            "cache_creation_input_tokens": 0,
        },
        "result": None,
        "structured_output": None,
        "model_usage": None,
    }
    kw.update(over)
    return ResultMessage(**kw)


def sdk_rate_limit_event(status: str, resets_at: int | None, rate_limit_type: str = "five_hour", utilization: float = 1.0):
    from claude_agent_sdk import RateLimitEvent, RateLimitInfo

    info = RateLimitInfo(status=status, resets_at=resets_at, rate_limit_type=rate_limit_type, utilization=utilization)
    return RateLimitEvent(rate_limit_info=info, uuid="evt-1", session_id="sess-test")


class FakeQuery:
    """Scripted stand-in for `claude_agent_sdk.query(*, prompt, options, ...)`.

    `scripts` is one entry per expected call: a list of messages to yield, optionally
    ending in an Exception instance to raise after yielding the preceding messages.
    Records the options object and the (drained) prompt messages of every call.
    """

    def __init__(self, scripts: list[list[Any]]) -> None:
        self.scripts = [list(s) for s in scripts]
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *, prompt: Any, options: Any = None, **kw: Any) -> AsyncIterator[Any]:
        if not self.scripts:
            raise AssertionError("FakeQuery exhausted: adapter issued more queries than scripted")
        script = self.scripts.pop(0)
        call: dict[str, Any] = {"options": options, "prompt_raw": prompt, "prompt_messages": [], "extra_kwargs": kw}
        self.calls.append(call)
        return self._stream(prompt, script, call)

    async def _stream(self, prompt: Any, script: list[Any], call: dict[str, Any]) -> AsyncIterator[Any]:
        # DESIGN §7.3: prompt must be an async iterable (streaming input), never a str.
        call["prompt_is_str"] = isinstance(prompt, str)
        if hasattr(prompt, "__aiter__"):
            async for msg in prompt:
                call["prompt_messages"].append(msg)
        for item in script:
            if isinstance(item, BaseException):
                raise item
            yield item


# ------------------------------------------------------------ openai fakes


def openai_response(
    content: str | None,
    *,
    model: str = "anthropic/claude-sonnet-5",
    provider: str = "Anthropic",
    finish_reason: str = "stop",
    prompt_tokens: int = 1200,
    completion_tokens: int = 250,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
    cost: float | None = 0.0042,
) -> SimpleNamespace:
    """Shape of a ChatCompletion as returned by OpenRouter (DESIGN §7.4)."""
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
        cost=cost,
        cost_details=SimpleNamespace(upstream_inference_cost=cost),
    )
    message = SimpleNamespace(role="assistant", content=content, refusal=None)
    choice = SimpleNamespace(index=0, message=message, finish_reason=finish_reason)
    return SimpleNamespace(id="gen-test", model=model, provider=provider, choices=[choice], usage=usage)


class FakeOpenAIClient:
    """Duck-types `openai.OpenAI` far enough for the adapter: `.chat.completions.create(**kw)`.

    `scripts` entries are either a response object to return or an Exception to raise.
    """

    def __init__(self, scripts: list[Any], *, base_url: str = "https://openrouter.ai/api/v1/") -> None:
        self.scripts = list(scripts)
        self.calls: list[dict[str, Any]] = []
        self.base_url = base_url
        self.max_retries = 3
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.scripts:
            raise AssertionError("FakeOpenAIClient exhausted: adapter issued more requests than scripted")
        item = self.scripts.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def openai_status_error(status: int, message: str = "upstream error"):
    """Construct the real `openai` exception the client raises for `status` (no network)."""
    import httpx2  # openai 3.3.1's HTTP layer (DESIGN §1)
    import openai

    req = httpx2.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    resp = httpx2.Response(status, request=req)
    if status == 429:
        return openai.RateLimitError(message, response=resp, body=None)
    if status >= 500:
        return openai.InternalServerError(message, response=resp, body=None)
    return openai.APIStatusError(message, response=resp, body=None)


# --------------------------------------------------------- anthropic fakes


def anthropic_response(
    parsed: dict[str, Any] | None,
    *,
    model: str = "claude-sonnet-5",
    stop_reason: str = "end_turn",
    input_tokens: int = 1000,
    output_tokens: int = 500,
    cache_read: int = 200,
    cache_write: int = 100,
    parsed_output: Any | None = None,
) -> SimpleNamespace:
    """A `messages.parse(...)` response whose text block is `json.dumps(parsed)`.

    `parsed_output` defaults to None: with the real anthropic 1.0.0 `output_config`
    route the SDK does NOT hand back a Pydantic instance, so the adapter must read
    the text block and let `call_stage` parse it (DESIGN §7.5 / D30). Pass
    `parsed_output=...` explicitly to exercise the parsed-object route.
    """
    text = json.dumps(parsed) if parsed is not None else ""
    return SimpleNamespace(
        id="msg-test",
        model=model,
        stop_reason=stop_reason,
        parsed_output=parsed_output,
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_write,
        ),
    )


class FakeAnthropicClient:
    """Duck-types `anthropic.Anthropic`: `.messages.parse(**kw)` and `.messages.create(**kw)`.

    Records which method was used and its kwargs; returns scripted responses.
    """

    def __init__(self, scripts: list[Any]) -> None:
        self.scripts = list(scripts)
        self.calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(parse=self._parse, create=self._create)

    def _handle(self, method: str, kwargs: dict[str, Any]) -> Any:
        self.calls.append({"method": method, **kwargs})
        if not self.scripts:
            raise AssertionError("FakeAnthropicClient exhausted")
        item = self.scripts.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def _parse(self, **kwargs: Any) -> Any:
        return self._handle("parse", kwargs)

    def _create(self, **kwargs: Any) -> Any:
        return self._handle("create", kwargs)


# ------------------------------------------------------------------ misc


def run_async(fn: Callable[..., Awaitable[Any]], *a: Any, **kw: Any) -> Any:
    import asyncio

    return asyncio.run(fn(*a, **kw))
