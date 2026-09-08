"""Azure AI Foundry provider for `socratic-tutor-poc`.

Modeled directly on `providers/openai_compat.py` (same raw-`httpx` choice, same
docstring reasoning: "one fewer dependency, and the request body is then
something a test can assert on directly") with the two contract differences
Azure requires:

- `api-key` header instead of `Authorization: Bearer`
- mandatory `api-version` query parameter, and the deployment name is part of
  the URL path (`/openai/deployments/{deployment}/chat/completions`), not the
  JSON body's `"model"` field — `model` is still accepted and sent for
  compatibility/logging but Azure ignores it in favor of the URL's deployment.
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

PROVIDER = "azure_foundry"
DEFAULT_API_VERSION = "2024-10-21"
DONE_SENTINEL = "[DONE]"


class AzureFoundryProvider(LLMProvider):
    """Calls an Azure AI Foundry / Azure OpenAI deployment's chat-completions endpoint."""

    name = PROVIDER

    def __init__(
        self,
        *,
        endpoint: str = "",
        deployment: str = "",
        api_key: str | None = None,
        api_version: str = DEFAULT_API_VERSION,
        timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._deployment = deployment
        self._api_version = api_version
        if client is None:
            if not endpoint:
                raise ProviderConfigError(
                    "azure_foundry is selected but POC_AZURE_FOUNDRY_ENDPOINT is empty"
                )
            if not deployment:
                raise ProviderConfigError(
                    "azure_foundry is selected but POC_AZURE_FOUNDRY_DEPLOYMENT is empty"
                )
            if not api_key:
                raise ProviderConfigError(
                    "azure_foundry is selected but POC_AZURE_FOUNDRY_API_KEY is not set; "
                    "put it in .env at the repo root (see .env.example)"
                )
            client = httpx.AsyncClient(
                base_url=f"{endpoint.rstrip('/')}/openai/deployments/{deployment}",
                headers={"api-key": api_key},
                params={"api-version": api_version},
                timeout=timeout_s,
            )
        self.client = client

    def __repr__(self) -> str:
        return f"AzureFoundryProvider(base_url={self.client.base_url!s}, deployment={self._deployment!r}, api_key=<hidden>)"

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
            response = await self.client.post("/chat/completions", json=body)
        except httpx.TimeoutException as exc:
            raise ProviderConnectionError(f"azure_foundry: request to {model!r} timed out") from exc
        except httpx.TransportError as exc:
            raise ProviderConnectionError(
                f"azure_foundry: cannot reach the endpoint: {exc}"
            ) from exc
        _raise_for_status(response, model, self._deployment)
        latency_ms = (time.perf_counter() - started) * 1000.0

        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderResponseError(f"azure_foundry: response was not JSON: {exc}") from exc

        content = _first_message_content(data)
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError(
                f"azure_foundry: {model!r} returned unparseable JSON content: {exc}"
            ) from exc
        try:
            value = schema.model_validate(payload)
        except ValidationError as exc:
            raise ProviderResponseError(
                f"azure_foundry: output does not match {schema.__name__}: {exc}"
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
            async with self.client.stream("POST", "/chat/completions", json=body) as response:
                if response.status_code >= 400:
                    await response.aread()
                    _raise_for_status(response, model, self._deployment)
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
                            f"azure_foundry: malformed SSE frame from {model!r}: {exc}"
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
                f"azure_foundry: stream from {model!r} timed out"
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderConnectionError(
                f"azure_foundry: cannot reach the endpoint: {exc}"
            ) from exc
        finally:
            if stats is not None:
                stats.total_ms = (time.perf_counter() - started) * 1000.0

    async def aclose(self) -> None:
        await self.client.aclose()


def _wire_with_system(system: str, messages: Sequence[Msg]) -> list[dict[str, str]]:
    wire = to_wire(messages)
    return [{"role": "system", "content": system}, *wire] if system else wire


def _sse_payload(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or not stripped.startswith("data:"):
        return None
    return stripped[len("data:") :].strip()


def _first_message_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderResponseError("azure_foundry: response carried no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ProviderResponseError("azure_foundry: first choice carried no message content")
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


def _raise_for_status(response: httpx.Response, model: str, deployment: str) -> None:
    if response.status_code < 400:
        return
    detail = response.text[:500]
    if response.status_code == 404:
        raise ProviderModelNotFound(
            f"azure_foundry: deployment {deployment!r} not found: {detail}"
        )
    if response.status_code == 429:
        raise ProviderRateLimited(f"azure_foundry: rate limited on {deployment!r}: {detail}")
    raise ProviderError(f"azure_foundry: HTTP {response.status_code} on {deployment!r}: {detail}")


__all__ = ["DEFAULT_API_VERSION", "PROVIDER", "AzureFoundryProvider"]
