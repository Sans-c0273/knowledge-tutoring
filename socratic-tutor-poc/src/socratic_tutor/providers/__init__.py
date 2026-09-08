"""Provider factory — resolves a pipeline role to an LLM backend (R21).

R21's acceptance criterion is that swapping Call C to an OpenRouter model is a
config-only change. That holds because nothing outside this package names a
provider: the pipeline asks for `get_provider("generation")` and gets whatever
`POC_GENERATION_PROVIDER` selects.

Providers are cached per provider name, so the httpx connection pool and the
Anthropic client are shared across turns. Tests pass explicit settings, which
bypasses the cache.
"""

from __future__ import annotations

from socratic_tutor.config import PROVIDERS, ConfigError, Settings, get_settings
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
)

_CACHE: dict[str, LLMProvider] = {}


def _build(provider_name: str, settings: Settings) -> LLMProvider:
    if provider_name == "claude_subscription":
        from socratic_tutor.providers.claude_subscription import ClaudeSubscriptionProvider

        return ClaudeSubscriptionProvider(
            data_dir=settings.data_dir, effort=settings.claude_subscription_effort
        )
    if provider_name == "anthropic":
        from socratic_tutor.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider()
    if provider_name == "openai_compat":
        from socratic_tutor.providers.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(
            base_url=settings.openai_compat_base_url,
            api_key=settings.openai_compat_api_key,
            timeout_s=settings.openai_compat_timeout_s,
        )
    if provider_name == "azure_foundry":
        from socratic_tutor.providers.azure_foundry import AzureFoundryProvider

        return AzureFoundryProvider(
            endpoint=settings.azure_foundry_endpoint,
            deployment=settings.azure_foundry_deployment,
            api_key=settings.azure_foundry_api_key,
            api_version=settings.azure_foundry_api_version,
            timeout_s=settings.azure_foundry_timeout_s,
        )
    raise ConfigError(f"unknown provider {provider_name!r}; expected one of {', '.join(PROVIDERS)}")


def get_provider(role: str, settings: Settings | None = None) -> LLMProvider:
    """Provider for `intent`, `evaluation` or `generation`.

    With `settings` given, a fresh provider is built and not cached — the caller
    owns it and should `await provider.aclose()`.
    """
    if settings is not None:
        return _build(settings.provider_for(role), settings)

    resolved = get_settings()
    provider_name = resolved.provider_for(role)
    if provider_name not in _CACHE:
        _CACHE[provider_name] = _build(provider_name, resolved)
    return _CACHE[provider_name]


async def reset_providers() -> None:
    """Close and drop every cached provider. For shutdown and for tests."""
    while _CACHE:
        _, provider = _CACHE.popitem()
        await provider.aclose()


__all__ = [
    "LLMProvider",
    "Msg",
    "ProviderConfigError",
    "ProviderConnectionError",
    "ProviderError",
    "ProviderModelNotFound",
    "ProviderRateLimited",
    "ProviderResponseError",
    "StreamStats",
    "StructuredResult",
    "Usage",
    "get_provider",
    "reset_providers",
    "strict_json_schema",
]
