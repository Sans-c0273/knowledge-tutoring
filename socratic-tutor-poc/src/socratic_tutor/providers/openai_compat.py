"""OpenAI-compatible provider — the config-swap alternative (Addendum §1, R21).

Any endpoint that serves `POST /chat/completions` in OpenAI's shape works:
OpenRouter and Alibaba Model Studio are the two the Addendum names. Structured
output uses `response_format: {"type": "json_schema", ...}` with the same strict
schema the Anthropic path sends as a tool; streaming parses the standard SSE
frames.

Talks raw HTTP through httpx rather than the `openai` SDK — one fewer
dependency, and the request body is then something a test can assert on
directly.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from socratic_tutor.providers.base import (
    LLMProvider,
    Msg,
    ProviderConfigError,
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

PROVIDER = "openai_compat"
ENDPOINT = "/chat/completions"
DONE_SENTINEL = "[DONE]"


class OpenAICompatProvider(LLMProvider):
    """Calls any OpenAI-compatible `/chat/completions` endpoint."""

    name = PROVIDER

    def __init__(
        self,
        *,
        base_url: str = "",
        api_key: str | None = None,
        timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if client is None:
            if not base_url:
                raise ProviderConfigError(
                    "openai_compat is selected but POC_OPENAI_COMPAT_BASE_URL is empty"
                )
            if not api_key:
                raise ProviderConfigError(
                    "openai_compat is selected but POC_OPENAI_COMPAT_API_KEY is not set; "
                    "put it in .env at the repo root (see .env.example)"
                )
            client = httpx.AsyncClient(
                base_url=base_url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=timeout_s,
            )
        self.client = client

    def __repr__(self) -> str:
        return f"OpenAICompatProvider(base_url={self.client.base_url!s}, api_key=<hidden>)"

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
        body = {
            "model": model,
            "messages": _wire_with_system(system, messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": strict_json_schema(schema),
                },
            },
        }
        started = time.perf_counter()
        try:
            response = await self.client.post(ENDPOINT, json=body)
        except httpx.TimeoutException as exc:
            raise ProviderConnectionError(f"openai_compat: request to {model!r} timed out") from exc
        except httpx.TransportError as exc:
            raise ProviderConnectionError(
                f"openai_compat: cannot reach the endpoint: {exc}"
            ) from exc
        _raise_for_status(response, model)
        latency_ms = (time.perf_counter() - started) * 1000.0

        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderResponseError(f"openai_compat: response was not JSON: {exc}") from exc

        content = _first_message_content(data)
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError(
                f"openai_compat: {model!r} returned unparseable JSON content: {exc}"
            ) from exc
        try:
            value = schema.model_validate(payload)
        except ValidationError as exc:
            raise ProviderResponseError(
                f"openai_compat: output does not match {schema.__name__}: {exc}"
            ) from exc

        return StructuredResult(
            value=value,
            usage=_usage_of(data),
            latency_ms=latency_ms,
            model=str(data.get("model") or model),
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
        body = {
            "model": model,
            "messages": _wire_with_system(system, messages),
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if stats is not None:
            stats.model, stats.provider = model, PROVIDER
        started = time.perf_counter()
        try:
            async with self.client.stream("POST", ENDPOINT, json=body) as response:
                if response.status_code >= 400:
                    await response.aread()
                    _raise_for_status(response, model)
                async for line in response.aiter_lines():
                    frame = _sse_payload(line)
                    if frame is None:
                        continue
                    if frame == DONE_SENTINEL:
                        break
                    try:
                        chunk = json.loads(frame)
                    except json.JSONDecodeError as exc:
                        raise ProviderResponseError(
                            f"openai_compat: malformed SSE frame from {model!r}: {exc}"
                        ) from exc
                    if stats is not None and chunk.get("usage"):
                        stats.usage = _usage_of(chunk)
                    delta = _delta_text(chunk)
                    if not delta:
                        continue
                    if stats is not None and stats.ttft_ms is None:
                        stats.ttft_ms = (time.perf_counter() - started) * 1000.0
                    yield delta
        except httpx.TimeoutException as exc:
            raise ProviderConnectionError(
                f"openai_compat: stream from {model!r} timed out"
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderConnectionError(
                f"openai_compat: cannot reach the endpoint: {exc}"
            ) from exc
        finally:
            if stats is not None:
                stats.total_ms = (time.perf_counter() - started) * 1000.0

    async def aclose(self) -> None:
        await self.client.aclose()


def _wire_with_system(system: str, messages: Sequence[Msg]) -> list[dict[str, str]]:
    """OpenAI carries the system prompt as the first message; Anthropic does not."""
    wire = to_wire(messages)
    return [{"role": "system", "content": system}, *wire] if system else wire


def _sse_payload(line: str) -> str | None:
    """The payload of one `data:` frame, or None for keep-alives and blank lines."""
    stripped = line.strip()
    if not stripped or not stripped.startswith("data:"):
        return None
    return stripped[len("data:") :].strip()


def _first_message_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderResponseError("openai_compat: response carried no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ProviderResponseError("openai_compat: first choice carried no message content")
    return content


def _delta_text(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
    content = delta.get("content") if isinstance(delta, dict) else None
    return content if isinstance(content, str) else ""


def _usage_of(data: dict[str, Any]) -> Usage:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return Usage()
    return Usage(
        input_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or 0),
    )


def _raise_for_status(response: httpx.Response, model: str) -> None:
    if response.status_code < 400:
        return
    detail = response.text[:500]
    if response.status_code == 404:
        raise ProviderModelNotFound(f"openai_compat: model {model!r} not found: {detail}")
    if response.status_code == 429:
        raise ProviderRateLimited(f"openai_compat: rate limited on {model!r}: {detail}")
    raise ProviderError(f"openai_compat: HTTP {response.status_code} on {model!r}: {detail}")


__all__ = ["ENDPOINT", "PROVIDER", "OpenAICompatProvider"]
