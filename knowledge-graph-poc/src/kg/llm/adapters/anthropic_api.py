"""Provider 3 (optional) — Anthropic Console API via the `anthropic` SDK (DESIGN §7.5; D4, D26).

The package is an optional extra (`kg[anthropic-api]`), so it is imported only
when a client has to be built; this module imports without it. The GA
structured-output surface in `anthropic` 1.0.0 is
`messages.parse(..., output_config={"format": {"type": "json_schema", "schema": ...}})`
(passing a schema dict as `output_format` is rejected by the SDK), which sends
exactly the profile-filtered schema and needs no beta header. `parsed_output`
is used when present, otherwise the text is handed to the shared repair loop.
Cost is computed from `llm.anthropic_api.pricing_usd_per_mtok`.
"""

from __future__ import annotations

from typing import Any

from kg.config import Config, ConfigError
from kg.llm.adapters.base import Msg, RawCompletion, Usage, anthropic_image_block, redact, require_provider, wire_messages
from kg.llm.errors import ProviderError, RateLimited

PROVIDER = "anthropic_api"
MAX_RETRIES = 3
EXTRA_NAME = "anthropic-api"
DEFAULT_CACHE_READ_MULTIPLIER = 0.10
DEFAULT_CACHE_WRITE_MULTIPLIER = 1.25


def _wire_messages(messages: list[Msg]) -> list[dict[str, Any]]:
    return wire_messages(messages, anthropic_image_block)


def compute_cost(pricing: dict[str, Any], model: str, usage: Usage, cache_write_tokens: int) -> float | None:
    """USD from the config price table; None for a model that is not priced there.

    The table's numerics are validated when kg.yaml is loaded
    (`kg.config._parse_pricing`), so a malformed entry fails before any paid call.
    """
    row = pricing.get(model)
    if not isinstance(row, dict) or "input" not in row or "output" not in row:
        return None
    read_mult = float(pricing.get("cache_read_multiplier", DEFAULT_CACHE_READ_MULTIPLIER))
    write_mult = float(pricing.get("cache_write_multiplier", DEFAULT_CACHE_WRITE_MULTIPLIER))
    in_price = float(row["input"])
    out_price = float(row["output"])
    total = (
        usage.input_tokens * in_price
        + usage.output_tokens * out_price
        + usage.cached_tokens * in_price * read_mult
        + cache_write_tokens * in_price * write_mult
    )
    return total / 1_000_000


class AnthropicApiAdapter:
    provider = PROVIDER
    max_repairs: int | None = None

    def __init__(self, cfg: Config, *, client: Any | None = None) -> None:
        require_provider(cfg, PROVIDER, "AnthropicApiAdapter")
        self._settings = cfg.llm.settings
        self._api_key = cfg.llm.api_key
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    f"llm.provider is '{PROVIDER}' but the 'anthropic' package is not installed. "
                    f"Install the optional extra: uv sync --extra {EXTRA_NAME}  (kg-mapper-poc[{EXTRA_NAME}])"
                ) from exc
            if not self._api_key:
                raise ConfigError(f"llm.provider is '{PROVIDER}' but ANTHROPIC_API_KEY is not set")
            client = anthropic.Anthropic(api_key=self._api_key, max_retries=MAX_RETRIES, timeout=float(cfg.llm.stage_timeout_s))
        self.client = client

    def __repr__(self) -> str:
        return "AnthropicApiAdapter(api_key=<set>)"

    def _redact(self, text: str) -> str:
        return redact(text, self._api_key)

    def complete(self, system: str, messages: list[Msg], schema_json: dict[str, Any], model: str, max_tokens: int) -> RawCompletion:
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": _wire_messages(messages),
            "output_config": {"format": {"type": "json_schema", "schema": schema_json}},
        }
        try:
            response = self.client.messages.parse(**request)
        except Exception as exc:
            if exc.__class__.__name__.endswith("RateLimitError"):
                raise RateLimited(f"anthropic_api: HTTP 429 persisted after {MAX_RETRIES} client retries", action="defer") from exc
            raise ProviderError(f"anthropic_api: {exc.__class__.__name__}: {self._redact(str(exc))}") from exc

        parsed = getattr(response, "parsed_output", None)
        if parsed is not None:
            text_or_obj: Any = parsed
        else:
            text_or_obj = "".join(getattr(b, "text", "") for b in (getattr(response, "content", None) or []) if getattr(b, "type", None) == "text")

        raw_usage = getattr(response, "usage", None)
        cache_write = int(getattr(raw_usage, "cache_creation_input_tokens", 0) or 0)
        usage = Usage(
            input_tokens=int(getattr(raw_usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(raw_usage, "output_tokens", 0) or 0),
            cached_tokens=int(getattr(raw_usage, "cache_read_input_tokens", 0) or 0),
        )
        return RawCompletion(
            text_or_obj=text_or_obj,
            usage=usage,
            model_requested=model,
            model_served=getattr(response, "model", None),
            cost_usd=compute_cost(self._settings.pricing_usd_per_mtok, model, usage, cache_write),
            stop_reason=getattr(response, "stop_reason", None),
            provider_meta={"response_id": getattr(response, "id", None), "cache_creation_input_tokens": cache_write},
        )


__all__ = ["AnthropicApiAdapter", "compute_cost"]
