"""Provider 2 — OpenRouter through the official `openai` client (DESIGN §7.4, §7.6; D23–D25).

`chat.completions.create` with `response_format` json_schema strict (the exact
profile-filtered schema, D24) and the provider-routing block in `extra_body`.
Validation and repair live in `kg.llm.call_stage`; this adapter issues exactly
one request per `complete()` (the client already retries HTTP 429/5xx three times).
"""

from __future__ import annotations

import re
from typing import Any

import openai

from kg.config import Config, ConfigError
from kg.llm.adapters.base import Msg, RawCompletion, Usage, openai_image_block, redact, require_provider, wire_messages
from kg.llm.errors import ProviderError, RateLimited

PROVIDER = "openrouter"
MAX_RETRIES = 3
_SCHEMA_NAME_BAD = re.compile(r"[^A-Za-z0-9_-]")


def _schema_name(schema_json: dict[str, Any]) -> str:
    raw = str(schema_json.get("title") or "stage_output")
    return (_SCHEMA_NAME_BAD.sub("_", raw) or "stage_output")[:64]


def _wire_messages(system: str, messages: list[Msg]) -> list[dict[str, Any]]:
    return [{"role": "system", "content": system}, *wire_messages(messages, openai_image_block)]


class OpenRouterAdapter:
    provider = PROVIDER
    max_repairs: int | None = None

    def __init__(self, cfg: Config, *, client: Any | None = None) -> None:
        require_provider(cfg, PROVIDER, "OpenRouterAdapter")
        self._settings = cfg.llm.settings
        self._api_key = cfg.llm.api_key
        if client is None:
            if not self._api_key:
                raise ConfigError("llm.provider is 'openrouter' but OPENROUTER_API_KEY is not set")
            client = openai.OpenAI(
                base_url=self._settings.base_url,
                api_key=self._api_key,
                max_retries=MAX_RETRIES,
                timeout=float(cfg.llm.stage_timeout_s),
                default_headers={"X-Title": self._settings.app_title},
            )
        self.client = client

    def __repr__(self) -> str:
        return f"OpenRouterAdapter(base_url={self._settings.base_url!r}, api_key=<set>)"

    def _redact(self, text: str) -> str:
        return redact(text, self._api_key)

    def complete(self, system: str, messages: list[Msg], schema_json: dict[str, Any], model: str, max_tokens: int) -> RawCompletion:
        routing = self._settings.provider
        request: dict[str, Any] = {
            "model": model,
            "messages": _wire_messages(system, messages),
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": _schema_name(schema_json), "strict": True, "schema": schema_json},
            },
            "extra_body": {
                "provider": {
                    "require_parameters": routing.require_parameters,
                    "data_collection": routing.data_collection,
                    "order": list(routing.order),
                    "allow_fallbacks": routing.allow_fallbacks,
                }
            },
        }
        try:
            response = self.client.chat.completions.create(**request)
        except openai.RateLimitError as exc:
            raise RateLimited(
                f"openrouter: HTTP 429 persisted after {MAX_RETRIES} client retries: {self._redact(exc.message)}", action="defer"
            ) from exc
        except openai.APIStatusError as exc:
            raise ProviderError(f"openrouter: HTTP {exc.status_code}: {self._redact(exc.message)}") from exc
        except openai.APIError as exc:
            raise ProviderError(f"openrouter: {exc.__class__.__name__}: {self._redact(str(exc))}") from exc

        choice = response.choices[0] if response.choices else None
        content = getattr(getattr(choice, "message", None), "content", None) if choice is not None else None
        usage = getattr(response, "usage", None)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        cost = getattr(usage, "cost", None)
        cost_details = getattr(usage, "cost_details", None)
        return RawCompletion(
            text_or_obj=content if content is not None else "",
            usage=Usage(
                input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                cached_tokens=int(getattr(prompt_details, "cached_tokens", 0) or 0),
                reasoning_tokens=int(getattr(completion_details, "reasoning_tokens", 0) or 0),
            ),
            model_requested=model,
            model_served=getattr(response, "model", None),
            cost_usd=float(cost) if cost is not None else None,
            stop_reason=getattr(choice, "finish_reason", None) if choice is not None else None,
            provider_meta={
                "upstream_provider": getattr(response, "provider", None),
                "response_id": getattr(response, "id", None),
                "upstream_inference_cost": getattr(cost_details, "upstream_inference_cost", None),
            },
        )


__all__ = ["OpenRouterAdapter"]
