"""Runtime settings — per-role model and provider assignment (Addendum §1, R21).

Everything is read from the environment with the `POC_` prefix, falling back to
a `.env` file at the repo root (see `.env.example`).

The Anthropic path deliberately has **no API-key setting**. The owner's chosen
auth is the `ant auth login` OAuth profile, which the SDK's zero-arg client
resolves by itself; requiring a key here would break that. Only the
OpenAI-compatible adapter (OpenRouter / Alibaba Model Studio) needs a
credential, and it is checked when that adapter is actually selected.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, get_args

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Repo root: .../socratic-tutor-poc/src/socratic_tutor/config.py → three parents up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

Role = Literal["intent", "evaluation", "generation"]
ProviderName = Literal["claude_subscription", "anthropic", "openai_compat", "azure_foundry"]

ROLES: tuple[str, ...] = get_args(Role)
PROVIDERS: tuple[str, ...] = get_args(ProviderName)


class ConfigError(Exception):
    """Settings are missing, contradictory, or select an unknown role/provider."""


class Settings(BaseSettings):
    """POC configuration. Instantiate via `get_settings()`; construct directly in tests."""

    model_config = SettingsConfigDict(
        env_prefix="POC_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- per-role model assignment (Addendum §1: Haiku classifies, Sonnet teaches)
    intent_model: str = "claude-haiku-4-5"
    evaluation_model: str = "claude-haiku-4-5"
    generation_model: str = "claude-sonnet-5"

    # --- per-role provider selection (R21: swapping Call C is config-only)
    intent_provider: ProviderName = "claude_subscription"
    evaluation_provider: ProviderName = "claude_subscription"
    generation_provider: ProviderName = "claude_subscription"

    # --- Claude Agent SDK on the machine's existing `claude login`
    #: Reasoning effort for the structured calls. `low` suits classification.
    claude_subscription_effort: str = "low"

    # --- OpenAI-compatible adapter (OpenRouter, Alibaba Model Studio)
    openai_compat_base_url: str = "https://openrouter.ai/api/v1"
    openai_compat_api_key: str | None = Field(default=None, repr=False)
    openai_compat_timeout_s: float = 120.0

    # --- Azure AI Foundry / Azure OpenAI adapter
    azure_foundry_endpoint: str = ""
    azure_foundry_api_key: str | None = Field(default=None, repr=False)
    azure_foundry_deployment: str = ""
    azure_foundry_api_version: str = "2024-10-21"
    azure_foundry_timeout_s: float = 120.0

    # --- generation / classification budgets
    structured_max_tokens: int = 512
    generation_max_tokens: int = 2000
    #: Below this, Call A's result falls back to `explain` and logs (Tech Spec §2.1).
    intent_confidence_threshold: float = 0.6

    # --- retrieval
    embedding_model: str = "BAAI/bge-m3"

    # --- embeddings backend: "local" runs BGE-M3 on this machine; "azure_foundry"
    # calls out to an Azure AI Foundry / Azure OpenAI embedding deployment instead,
    # for a machine that can't run the local model.
    embedding_provider: Literal["local", "azure_foundry"] = "local"
    embedding_azure_endpoint: str = ""
    embedding_azure_api_key: str | None = Field(default=None, repr=False)
    embedding_azure_deployment: str = ""
    embedding_azure_api_version: str = "2024-10-21"
    embedding_azure_timeout_s: float = 60.0
    #: Texts per HTTP request; keeps well under Azure's per-request token/array limits.
    embedding_azure_batch_size: int = 96

    # --- paths (chroma_dir / uploads_dir derive from data_dir when unset)
    content_dir: Path = PROJECT_ROOT / "content"
    data_dir: Path = PROJECT_ROOT / "data"
    chroma_dir: Path | None = None
    uploads_dir: Path | None = None

    @model_validator(mode="after")
    def _derive_paths(self) -> Settings:
        if self.chroma_dir is None:
            self.chroma_dir = self.data_dir / "chroma"
        if self.uploads_dir is None:
            self.uploads_dir = self.data_dir / "uploads"
        return self

    def model_for(self, role: str) -> str:
        """Model ID for `intent`, `evaluation` or `generation`."""
        self._check_role(role)
        return str(getattr(self, f"{role}_model"))

    def provider_for(self, role: str) -> str:
        """Provider name for `intent`, `evaluation` or `generation`."""
        self._check_role(role)
        return str(getattr(self, f"{role}_provider"))

    def max_tokens_for(self, role: str) -> int:
        """Token budget per role: classification is short, generation is capped by the plan."""
        self._check_role(role)
        return self.generation_max_tokens if role == "generation" else self.structured_max_tokens

    @staticmethod
    def _check_role(role: str) -> None:
        if role not in ROLES:
            raise ConfigError(f"unknown role {role!r}; expected one of {', '.join(ROLES)}")

    def ensure_dirs(self) -> None:
        """Create the runtime directories. Called at app startup, never at import."""
        for path in (self.data_dir, self.chroma_dir, self.uploads_dir):
            if path is not None:
                path.mkdir(parents=True, exist_ok=True)

    def dump(self) -> dict[str, object]:
        """Redacted view for traces and the glass-box UI. The key appears only as set/unset."""
        data = self.model_dump(
            mode="json",
            exclude={"openai_compat_api_key", "azure_foundry_api_key", "embedding_azure_api_key"},
        )
        data["openai_compat_api_key"] = "present" if self.openai_compat_api_key else "absent"
        data["azure_foundry_api_key"] = "present" if self.azure_foundry_api_key else "absent"
        data["embedding_azure_api_key"] = "present" if self.embedding_azure_api_key else "absent"
        return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. `get_settings.cache_clear()` resets it in tests."""
    return Settings()


__all__ = [
    "PROJECT_ROOT",
    "PROVIDERS",
    "ROLES",
    "ConfigError",
    "ProviderName",
    "Role",
    "Settings",
    "get_settings",
]
