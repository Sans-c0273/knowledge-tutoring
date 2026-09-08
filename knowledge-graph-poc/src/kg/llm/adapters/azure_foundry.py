"""Azure AI Foundry adapter for `knowledge-graph-poc` (kg-mapper-poc).

Mirrors `kg.llm.adapters.openrouter.OpenRouterAdapter` (same DESIGN §7.1
`Adapter` protocol, same `openai` package — Azure OpenAI is reached through the
SDK's dedicated `AzureOpenAI` client class rather than plain `OpenAI(base_url=...)`,
since Azure's auth (`api-key` header) and required `api-version` query param
are exactly what that class exists to handle).

Azure addresses a *deployment*, not a base model id: the `model` argument
`complete()` receives (resolved from `kg.yaml`'s `llm.azure_foundry.models`
block, same shape as every other provider) must be the deployment name you
created in Azure AI Foundry / Azure OpenAI Studio, not e.g. `"gpt-4o"`.
"""

from __future__ import annotations

import re
from typing import Any

import openai

from kg.config import AzureFoundrySettings, Config, ConfigError
from kg.llm.adapters.base import Msg, RawCompletion, Usage, openai_image_block, redact, require_provider, wire_messages
from kg.llm.errors import ProviderError, RateLimited

PROVIDER = "azure_foundry"
MAX_RETRIES = 3
_SCHEMA_NAME_BAD = re.compile(r"[^A-Za-z0-9_-]")


def _schema_name(schema_json: dict[str, Any]) -> str:
    raw = str(schema_json.get("title") or "stage_output")
    return (_SCHEMA_NAME_BAD.sub("_", raw) or "stage_output")[:64]


def _wire_messages(system: str, messages: list[Msg]) -> list[dict[str, Any]]:
    return [{"role": "system", "content": system}, *wire_messages(messages, openai_image_block)]


class AzureFoundryAdapter:
    provider = PROVIDER
    max_repairs: int | None = None

    def __init__(self, cfg: Config, *, client: Any | None = None) -> None:
        require_provider(cfg, PROVIDER, "AzureFoundryAdapter")
        self._settings: AzureFoundrySettings = cfg.llm.settings  # type: ignore[assignment]
        self._api_key = cfg.llm.api_key
        if client is None:
            if not self._api_key:
                raise ConfigError("llm.provider is 'azure_foundry' but AZURE_FOUNDRY_API_KEY is not set")
            client = openai.AzureOpenAI(
                azure_endpoint=self._settings.endpoint,
                api_version=self._settings.api_version,
                api_key=self._api_key,
                max_retries=MAX_RETRIES,
                timeout=float(cfg.llm.stage_timeout_s),
            )
        self.client = client

    def __repr__(self) -> str:
        return f"AzureFoundryAdapter(endpoint={self._settings.endpoint!r}, api_key=<set>)"

    def _redact(self, text: str) -> str:
        return redact(text, self._api_key)

    def complete(self, system: str, messages: list[Msg], schema_json: dict[str, Any], model: str, max_tokens: int) -> RawCompletion:
        request: dict[str, Any] = {
            "model": model,  # the Azure *deployment name* for this stage
            "messages": _wire_messages(system, messages),
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": _schema_name(schema_json), "strict": True, "schema": schema_json},
            },
        }
        try:
            response = self.client.chat.completions.create(**request)
        except openai.RateLimitError as exc:
            raise RateLimited(
                f"azure_foundry: HTTP 429 persisted after {MAX_RETRIES} client retries: {self._redact(exc.message)}",
                action="defer",
            ) from exc
        except openai.APIStatusError as exc:
            raise ProviderError(f"azure_foundry: HTTP {exc.status_code}: {self._redact(exc.message)}") from exc
        except openai.APIError as exc:
            raise ProviderError(f"azure_foundry: {exc.__class__.__name__}: {self._redact(str(exc))}") from exc

        choice = response.choices[0] if response.choices else None
        content = getattr(getattr(choice, "message", None), "content", None) if choice is not None else None
        usage = getattr(response, "usage", None)
        return RawCompletion(
            text_or_obj=content if content is not None else "",
            usage=Usage(
                input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            ),
            model_requested=model,
            model_served=getattr(response, "model", None),
            cost_usd=None,  # Azure does not report per-call cost; price from your Azure rate card if needed
            stop_reason=getattr(choice, "finish_reason", None) if choice is not None else None,
            provider_meta={"response_id": getattr(response, "id", None)},
        )


__all__ = ["AzureFoundryAdapter", "AzureFoundrySettings"]
