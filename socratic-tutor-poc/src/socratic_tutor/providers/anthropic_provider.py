"""Anthropic native provider — the POC's default for all three roles (Addendum §1).

Auth: the client is constructed with **no arguments**. The SDK resolves
`ANTHROPIC_API_KEY` when it is set and otherwise the OAuth profile written by
`ant auth login`, which is the owner's chosen path (no pay-per-token billing).
Nothing here may require or check for an API key — doing so would break the
subscription path.

Structured output (Calls A and B) uses strict tool use: one tool built from the
Pydantic model's JSON schema, `tool_choice` forcing it, and the returned
`tool_use.input` parsed into the model. Assistant prefill is deliberately not
used — it 400s on these models.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

import anthropic
from pydantic import BaseModel, ValidationError

from socratic_tutor.providers.base import (
    LLMProvider,
    Msg,
    ProviderConnectionError,
    ProviderError,
    ProviderModelNotFound,
    ProviderRateLimited,
    ProviderResponseError,
    StreamStats,
    StructuredResult,
    Usage,
    strict_json_schema,
    to_wire,
)

PROVIDER = "anthropic"

#: Sampling parameters this provider must never send. The current Claude models
#: reject them outright — Sonnet 5 and the Opus 4.7/4.8/5 family 400 on them —
#: and `anthropic` 1.2.0 has removed them from `messages.create()` altogether, so
#: passing one raises `TypeError` before a request is even built. Older models
#: such as Haiku 4.5 still accept them, but we run a mix, so the rule is simply
#: not to send any rather than to branch per model.
#:
#: Nothing is lost. `temperature=0` was a determinism lever for Calls A and B;
#: determinism now comes from the strict tool schema constraining the output
#: shape, plus the deterministic checker outranking the model on Call B. If a
#: determinism knob is ever wanted back, the current equivalent is
#: `output_config={"effort": "low"}` — but check per-model support first, since
#: an unsupported `effort` is the same class of live-path 400 this rule prevents.
SAMPLING_PARAMS: frozenset[str] = frozenset({"temperature", "top_p", "top_k"})


def request_kwargs(
    *,
    model: str,
    max_tokens: int,
    system: str,
    messages: Sequence[Msg],
    tool: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The exact kwargs handed to the SDK, built in one place so a test can check them.

    `tests/test_providers.py` binds the result against the installed SDK
    signature, so an argument the SDK has removed fails in CI rather than on
    first live contact.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": to_wire(messages),
    }
    if tool is not None:
        kwargs["tools"] = [tool]
        kwargs["tool_choice"] = {"type": "tool", "name": tool["name"]}
    return kwargs


def build_tool(schema: type[BaseModel], *, name: str | None = None) -> dict[str, Any]:
    """Strict tool definition for `schema` (Anthropic strict tool use).

    The input schema is the model's JSON schema put through
    `strict_json_schema`: refs inlined, `additionalProperties: false`, every
    field required. `strict: true` makes the API enforce it rather than treat it
    as a hint.
    """
    input_schema = strict_json_schema(schema)
    doc = (schema.__doc__ or "").strip().splitlines()
    return {
        "name": name or _tool_name(schema),
        "description": doc[0] if doc else f"Return a {schema.__name__} object.",
        "input_schema": input_schema,
        "strict": True,
    }


def _tool_name(schema: type[BaseModel]) -> str:
    """`IntentResult` → `emit_intent_result`. Tool names must be snake_case."""
    chars: list[str] = []
    for i, ch in enumerate(schema.__name__):
        if ch.isupper() and i > 0:
            chars.append("_")
        chars.append(ch.lower())
    return "emit_" + "".join(chars)


class AnthropicProvider(LLMProvider):
    """Calls Claude through the official SDK."""

    name = PROVIDER

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    @property
    def client(self) -> Any:
        """The async SDK client, built on first use so import never touches credentials."""
        if self._client is None:
            self._client = anthropic.AsyncAnthropic()
        return self._client

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
        tool = build_tool(schema)
        started = time.perf_counter()
        try:
            # `temperature` is accepted for interface parity with the
            # OpenAI-compatible adapter but is deliberately NOT sent — see
            # `SAMPLING_PARAMS` above.
            response = await self.client.messages.create(
                **request_kwargs(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=messages,
                    tool=tool,
                )
            )
        except anthropic.NotFoundError as exc:
            raise ProviderModelNotFound(f"anthropic: model {model!r} not found") from exc
        except anthropic.RateLimitError as exc:
            raise ProviderRateLimited(f"anthropic: rate limited on {model!r}") from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"anthropic: HTTP {exc.status_code} on {model!r}: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderConnectionError(f"anthropic: cannot reach the API: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0

        value = _parse_tool_use(response, tool["name"], schema)
        return StructuredResult(
            value=value,
            usage=_usage_of(response),
            latency_ms=latency_ms,
            model=getattr(response, "model", model) or model,
            provider=PROVIDER,
        )

    async def stream_text(
        self,
        *,
        model: str,
        system: str,
        messages: Sequence[Msg],
        max_tokens: int = 2000,
        stats: StreamStats | None = None,
    ) -> AsyncIterator[str]:
        if stats is not None:
            stats.model, stats.provider = model, PROVIDER
        started = time.perf_counter()
        try:
            async with self.client.messages.stream(
                **request_kwargs(
                    model=model, max_tokens=max_tokens, system=system, messages=messages
                )
            ) as stream:
                async for delta in stream.text_stream:
                    if stats is not None and stats.ttft_ms is None:
                        stats.ttft_ms = (time.perf_counter() - started) * 1000.0
                    yield delta
                if stats is not None:
                    final = await stream.get_final_message()
                    stats.usage = _usage_of(final)
                    stats.stop_reason = getattr(final, "stop_reason", None)
        except anthropic.NotFoundError as exc:
            raise ProviderModelNotFound(f"anthropic: model {model!r} not found") from exc
        except anthropic.RateLimitError as exc:
            raise ProviderRateLimited(f"anthropic: rate limited on {model!r}") from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"anthropic: HTTP {exc.status_code} on {model!r}: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderConnectionError(f"anthropic: cannot reach the API: {exc}") from exc
        finally:
            if stats is not None:
                stats.total_ms = (time.perf_counter() - started) * 1000.0

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()


def _parse_tool_use(response: Any, tool_name: str, schema: type[BaseModel]) -> BaseModel:
    """Pull the forced tool call out of the response and validate it into `schema`."""
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) != "tool_use" or getattr(block, "name", None) != tool_name:
            continue
        payload = getattr(block, "input", None)
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ProviderResponseError(
                    f"anthropic: tool {tool_name!r} returned unparseable JSON: {exc}"
                ) from exc
        if not isinstance(payload, dict):
            raise ProviderResponseError(
                f"anthropic: tool {tool_name!r} input was {type(payload).__name__}, expected an object"
            )
        try:
            return schema.model_validate(payload)
        except ValidationError as exc:
            raise ProviderResponseError(
                f"anthropic: tool {tool_name!r} output does not match {schema.__name__}: {exc}"
            ) from exc
    raise ProviderResponseError(
        f"anthropic: response contained no tool_use block named {tool_name!r} "
        f"(stop_reason={getattr(response, 'stop_reason', None)!r})"
    )


def _usage_of(response: Any) -> Usage:
    usage = getattr(response, "usage", None)
    return Usage(
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
    )


__all__ = ["PROVIDER", "AnthropicProvider", "build_tool"]
