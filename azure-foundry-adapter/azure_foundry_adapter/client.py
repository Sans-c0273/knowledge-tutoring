"""Shared low-level Azure AI Foundry / Azure OpenAI chat-completions client.

Azure's REST contract differs from plain OpenAI/OpenRouter in three ways every
adapter in this package needs to account for:

1. Auth is an `api-key` header by default (not `Authorization: Bearer`) when
   using a resource API key. Microsoft Entra ID (AAD) bearer-token auth is also
   supported — pass `entra_token_provider` instead of `api_key`.
2. Every request carries a mandatory `api-version` query parameter.
3. The model is addressed by **deployment name**, not the base model id — and
   the URL shape depends on how the resource was provisioned:
   - `"azure_openai"` (default): classic Azure OpenAI resource,
     `POST {endpoint}/openai/deployments/{deployment}/chat/completions?api-version=...`
   - `"foundry_models"`: an Azure AI Foundry resource using the unified
     "Foundry Models" inference endpoint,
     `POST {endpoint}/models/chat/completions?api-version=...` with the
     deployment/model name sent as `"model"` in the JSON body instead of the URL.
   Pick whichever matches the resource you provisioned (see this package's
   README) via `AzureFoundrySettings.api_style`.

This module is intentionally dependency-light (`httpx` only, no `openai`
package) so it can be dropped into `kg_reasoner` (which declares zero
dependencies today) as well as used standalone. `for_knowledge_graph_poc/`
instead uses the `openai` SDK's dedicated `AzureOpenAI`/`AsyncAzureOpenAI`
classes, since that project already depends on `openai` and its existing
adapters follow that convention.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from azure_foundry_adapter.errors import (
    AzureFoundryConfigError,
    AzureFoundryConnectionError,
    AzureFoundryModelNotFound,
    AzureFoundryRateLimited,
    AzureFoundryResponseError,
    AzureFoundryStatusError,
)

ApiStyle = Literal["azure_openai", "foundry_models"]

DEFAULT_API_VERSION = "2024-10-21"


@dataclass(frozen=True)
class AzureFoundrySettings:
    """Connection settings for one Azure AI Foundry / Azure OpenAI deployment.

    `endpoint` is the resource's base URL, e.g.
    `https://<resource-name>.openai.azure.com` (azure_openai style) or
    `https://<resource-name>.services.ai.azure.com` (foundry_models style) —
    no trailing path, no query string.
    """

    endpoint: str
    deployment: str
    api_key: str | None = field(default=None, repr=False)
    api_version: str = DEFAULT_API_VERSION
    api_style: ApiStyle = "azure_openai"
    timeout_s: float = 120.0
    #: Alternative to `api_key`: a zero-arg callable returning a fresh Entra ID
    #: (AAD) bearer token, e.g. `azure.identity.DefaultAzureCredential().get_token(...).token`.
    #: When set, it takes priority over `api_key`.
    entra_token_provider: Callable[[], str] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.endpoint:
            raise AzureFoundryConfigError("AzureFoundrySettings.endpoint is required")
        if not self.deployment:
            raise AzureFoundryConfigError("AzureFoundrySettings.deployment is required")
        if self.entra_token_provider is None and not self.api_key:
            raise AzureFoundryConfigError(
                "AzureFoundrySettings needs either api_key or entra_token_provider"
            )

    @classmethod
    def from_env(cls, *, prefix: str = "AZURE_FOUNDRY_") -> AzureFoundrySettings:
        """Build settings from `{prefix}ENDPOINT`, `{prefix}API_KEY`, `{prefix}DEPLOYMENT`,
        `{prefix}API_VERSION` (optional) and `{prefix}API_STYLE` (optional). See `.env.example`."""
        endpoint = os.environ.get(f"{prefix}ENDPOINT", "")
        deployment = os.environ.get(f"{prefix}DEPLOYMENT", "")
        api_key = os.environ.get(f"{prefix}API_KEY") or None
        api_version = os.environ.get(f"{prefix}API_VERSION", DEFAULT_API_VERSION)
        api_style = os.environ.get(f"{prefix}API_STYLE", "azure_openai")
        if api_style not in ("azure_openai", "foundry_models"):
            raise AzureFoundryConfigError(
                f"{prefix}API_STYLE must be 'azure_openai' or 'foundry_models', got {api_style!r}"
            )
        return cls(
            endpoint=endpoint,
            deployment=deployment,
            api_key=api_key,
            api_version=api_version,
            api_style=api_style,  # type: ignore[arg-type]
        )


def _url(settings: AzureFoundrySettings) -> str:
    base = settings.endpoint.rstrip("/")
    if settings.api_style == "foundry_models":
        return f"{base}/models/chat/completions"
    return f"{base}/openai/deployments/{settings.deployment}/chat/completions"


def _headers(settings: AzureFoundrySettings) -> dict[str, str]:
    if settings.entra_token_provider is not None:
        return {"Authorization": f"Bearer {settings.entra_token_provider()}"}
    assert settings.api_key  # enforced by __post_init__
    return {"api-key": settings.api_key}


def _params(settings: AzureFoundrySettings) -> dict[str, str]:
    return {"api-version": settings.api_version}


def build_request_body(
    *,
    settings: AzureFoundrySettings,
    system: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float = 0.0,
    schema_json: dict[str, Any] | None = None,
    schema_name: str = "response",
    stream: bool = False,
) -> dict[str, Any]:
    """Chat-completions request body shared by every adapter.

    `schema_json` (a plain JSON Schema, `additionalProperties: false` + full
    `required`) turns on strict structured output the same way OpenAI/OpenRouter
    do — Azure OpenAI's `response_format: json_schema` is wire-compatible.
    Omit it for the plain streaming/generation call.
    """
    body: dict[str, Any] = {
        "messages": [{"role": "system", "content": system}, *messages],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if settings.api_style == "foundry_models":
        body["model"] = settings.deployment
    if schema_json is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema_json},
        }
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def _raise_for_status(response: httpx.Response, deployment: str) -> None:
    if response.status_code < 400:
        return
    detail = response.text[:500]
    if response.status_code == 404:
        raise AzureFoundryModelNotFound(
            f"azure_foundry: deployment {deployment!r} not found (HTTP 404): {detail}. "
            f"Azure addresses a *deployment name*, not the base model id — double-check it."
        )
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        raise AzureFoundryRateLimited(
            f"azure_foundry: rate limited on {deployment!r} (HTTP 429): {detail}",
            retry_after=float(retry_after) if retry_after else None,
        )
    raise AzureFoundryStatusError(
        f"azure_foundry: HTTP {response.status_code} on {deployment!r}: {detail}",
        status_code=response.status_code,
    )


def first_message_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AzureFoundryResponseError("azure_foundry: response carried no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise AzureFoundryResponseError("azure_foundry: first choice carried no message content")
    return content


def delta_text(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
    content = delta.get("content") if isinstance(delta, dict) else None
    return content if isinstance(content, str) else ""


def _embeddings_url(settings: AzureFoundrySettings) -> str:
    base = settings.endpoint.rstrip("/")
    if settings.api_style == "foundry_models":
        return f"{base}/models/embeddings"
    return f"{base}/openai/deployments/{settings.deployment}/embeddings"


def build_embeddings_body(settings: AzureFoundrySettings, inputs: list[str]) -> dict[str, Any]:
    body: dict[str, Any] = {"input": inputs}
    if settings.api_style == "foundry_models":
        body["model"] = settings.deployment
    return body


def embed_sync(
    settings: AzureFoundrySettings,
    inputs: list[str],
    *,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """One embeddings call. Returns the parsed JSON body: `{"data": [{"embedding": [...], "index": 0}, ...], "usage": {...}}`."""
    body = build_embeddings_body(settings, inputs)
    owns_client = client is None
    http_client = client or httpx.Client(timeout=settings.timeout_s)
    try:
        response = http_client.post(
            _embeddings_url(settings), json=body, headers=_headers(settings), params=_params(settings)
        )
    except httpx.TimeoutException as exc:
        raise AzureFoundryConnectionError("azure_foundry: embeddings request timed out") from exc
    except httpx.TransportError as exc:
        raise AzureFoundryConnectionError(f"azure_foundry: cannot reach the endpoint: {exc}") from exc
    finally:
        if owns_client:
            http_client.close()
    _raise_for_status(response, settings.deployment)
    try:
        return response.json()
    except ValueError as exc:
        raise AzureFoundryResponseError(f"azure_foundry: embeddings response was not JSON: {exc}") from exc


async def embed_async(
    settings: AzureFoundrySettings,
    inputs: list[str],
    *,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    body = build_embeddings_body(settings, inputs)
    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=settings.timeout_s)
    try:
        response = await http_client.post(
            _embeddings_url(settings), json=body, headers=_headers(settings), params=_params(settings)
        )
    except httpx.TimeoutException as exc:
        raise AzureFoundryConnectionError("azure_foundry: embeddings request timed out") from exc
    except httpx.TransportError as exc:
        raise AzureFoundryConnectionError(f"azure_foundry: cannot reach the endpoint: {exc}") from exc
    finally:
        if owns_client:
            await http_client.aclose()
    _raise_for_status(response, settings.deployment)
    try:
        return response.json()
    except ValueError as exc:
        raise AzureFoundryResponseError(f"azure_foundry: embeddings response was not JSON: {exc}") from exc


def usage_of(data: dict[str, Any]) -> dict[str, int]:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
    }


def sse_payload(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or not stripped.startswith("data:"):
        return None
    return stripped[len("data:") :].strip()


# --------------------------------------------------------------------- sync


def complete_sync(
    settings: AzureFoundrySettings,
    *,
    system: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float = 0.0,
    schema_json: dict[str, Any] | None = None,
    schema_name: str = "response",
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """One non-streaming chat-completions call. Returns the parsed JSON response body."""
    body = build_request_body(
        settings=settings,
        system=system,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        schema_json=schema_json,
        schema_name=schema_name,
    )
    owns_client = client is None
    http_client = client or httpx.Client(timeout=settings.timeout_s)
    try:
        response = http_client.post(
            _url(settings), json=body, headers=_headers(settings), params=_params(settings)
        )
    except httpx.TimeoutException as exc:
        raise AzureFoundryConnectionError(
            f"azure_foundry: request to {settings.deployment!r} timed out"
        ) from exc
    except httpx.TransportError as exc:
        raise AzureFoundryConnectionError(f"azure_foundry: cannot reach endpoint: {exc}") from exc
    finally:
        if owns_client:
            http_client.close()
    _raise_for_status(response, settings.deployment)
    try:
        return response.json()
    except ValueError as exc:
        raise AzureFoundryResponseError(f"azure_foundry: response was not JSON: {exc}") from exc


def stream_sync(
    settings: AzureFoundrySettings,
    *,
    system: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float = 0.0,
    client: httpx.Client | None = None,
) -> Iterator[str]:
    """Yields text deltas from a streaming chat-completions call."""
    body = build_request_body(
        settings=settings,
        system=system,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    owns_client = client is None
    http_client = client or httpx.Client(timeout=settings.timeout_s)
    try:
        with http_client.stream(
            "POST", _url(settings), json=body, headers=_headers(settings), params=_params(settings)
        ) as response:
            if response.status_code >= 400:
                response.read()
                _raise_for_status(response, settings.deployment)
            for line in response.iter_lines():
                frame = sse_payload(line)
                if frame is None or frame == "[DONE]":
                    continue
                try:
                    chunk = json.loads(frame)
                except json.JSONDecodeError as exc:
                    raise AzureFoundryResponseError(
                        f"azure_foundry: malformed SSE frame: {exc}"
                    ) from exc
                text = delta_text(chunk)
                if text:
                    yield text
    except httpx.TimeoutException as exc:
        raise AzureFoundryConnectionError(
            f"azure_foundry: stream from {settings.deployment!r} timed out"
        ) from exc
    except httpx.TransportError as exc:
        raise AzureFoundryConnectionError(f"azure_foundry: cannot reach endpoint: {exc}") from exc
    finally:
        if owns_client:
            http_client.close()


# -------------------------------------------------------------------- async


async def complete_async(
    settings: AzureFoundrySettings,
    *,
    system: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float = 0.0,
    schema_json: dict[str, Any] | None = None,
    schema_name: str = "response",
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    body = build_request_body(
        settings=settings,
        system=system,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        schema_json=schema_json,
        schema_name=schema_name,
    )
    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=settings.timeout_s)
    try:
        response = await http_client.post(
            _url(settings), json=body, headers=_headers(settings), params=_params(settings)
        )
    except httpx.TimeoutException as exc:
        raise AzureFoundryConnectionError(
            f"azure_foundry: request to {settings.deployment!r} timed out"
        ) from exc
    except httpx.TransportError as exc:
        raise AzureFoundryConnectionError(f"azure_foundry: cannot reach endpoint: {exc}") from exc
    finally:
        if owns_client:
            await http_client.aclose()
    _raise_for_status(response, settings.deployment)
    try:
        return response.json()
    except ValueError as exc:
        raise AzureFoundryResponseError(f"azure_foundry: response was not JSON: {exc}") from exc


async def stream_async(
    settings: AzureFoundrySettings,
    *,
    system: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float = 0.0,
    client: httpx.AsyncClient | None = None,
) -> AsyncIterator[str]:
    body = build_request_body(
        settings=settings,
        system=system,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=settings.timeout_s)
    try:
        async with http_client.stream(
            "POST", _url(settings), json=body, headers=_headers(settings), params=_params(settings)
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                _raise_for_status(response, settings.deployment)
            async for line in response.aiter_lines():
                frame = sse_payload(line)
                if frame is None or frame == "[DONE]":
                    continue
                try:
                    chunk = json.loads(frame)
                except json.JSONDecodeError as exc:
                    raise AzureFoundryResponseError(
                        f"azure_foundry: malformed SSE frame: {exc}"
                    ) from exc
                text = delta_text(chunk)
                if text:
                    yield text
    except httpx.TimeoutException as exc:
        raise AzureFoundryConnectionError(
            f"azure_foundry: stream from {settings.deployment!r} timed out"
        ) from exc
    except httpx.TransportError as exc:
        raise AzureFoundryConnectionError(f"azure_foundry: cannot reach endpoint: {exc}") from exc
    finally:
        if owns_client:
            await http_client.aclose()


__all__ = [
    "DEFAULT_API_VERSION",
    "AzureFoundrySettings",
    "build_embeddings_body",
    "build_request_body",
    "complete_async",
    "complete_sync",
    "delta_text",
    "embed_async",
    "embed_sync",
    "first_message_content",
    "sse_payload",
    "stream_async",
    "stream_sync",
    "usage_of",
]
