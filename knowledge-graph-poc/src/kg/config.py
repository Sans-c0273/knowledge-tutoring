"""kg.yaml + .env loading, provider selection and credential guards (DESIGN §3, D22).

Everything but credentials comes from kg.yaml (R18). Credentials come from the
process environment or a `.env` file next to kg.yaml, read with python-dotenv
without mutating `os.environ`. An empty value counts as missing. Error messages
name variables, never values, and `api_key` is excluded from `repr()`, `str()`
and `dump()`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

from kg.paths import KINDS, Sandbox, SandboxViolation

STAGES: tuple[str, ...] = ("describe", "atomize", "edges", "dedup")
PROVIDERS: tuple[str, ...] = ("claude_subscription", "openrouter", "anthropic_api", "azure_foundry")
SCHEMA_NAMES: tuple[str, ...] = ("education", "general")
SUPPORTED_VERSION = 2

CREDENTIAL_VARS: tuple[str, ...] = ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "AZURE_FOUNDRY_API_KEY")
_CREDENTIAL_FOR_PROVIDER: dict[str, str | None] = {
    "claude_subscription": None,
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic_api": "ANTHROPIC_API_KEY",
    "azure_foundry": "AZURE_FOUNDRY_API_KEY",
}


class ConfigError(Exception):
    """kg.yaml / .env is missing, malformed, or fails a guard."""


# ------------------------------------------------------------- typed sections


@dataclass(frozen=True)
class CorpusConfig:
    slug: str
    name: str
    schema: str


@dataclass(frozen=True)
class PathsConfig:
    inbox: str
    converted: str
    graph: str
    processed: str
    runs: str

    def as_dict(self) -> dict[str, str]:
        return {k: getattr(self, k) for k in KINDS}


@dataclass(frozen=True)
class ClaudeSubscriptionSettings:
    models: dict[str, str]
    effort: str
    rate_limit_wait_minutes: int


@dataclass(frozen=True)
class OpenRouterRouting:
    require_parameters: bool
    data_collection: str
    order: list[str]
    allow_fallbacks: bool


@dataclass(frozen=True)
class OpenRouterSettings:
    base_url: str
    models: dict[str, str]
    provider: OpenRouterRouting
    app_title: str


@dataclass(frozen=True)
class AnthropicApiSettings:
    models: dict[str, str]
    pricing_usd_per_mtok: dict[str, Any]


@dataclass(frozen=True)
class AzureFoundrySettings:
    """Provider 4 — Azure AI Foundry / Azure OpenAI via `openai.AzureOpenAI` (see
    `kg.llm.adapters.azure_foundry`). `models` maps stage -> Azure *deployment name*,
    same convention as `OpenRouterSettings.models`."""

    endpoint: str
    api_version: str
    models: dict[str, str]


ProviderSettings = ClaudeSubscriptionSettings | OpenRouterSettings | AnthropicApiSettings | AzureFoundrySettings


@dataclass(frozen=True, repr=False)
class LLMConfig:
    provider: str
    repair_retries: int
    max_output_tokens: int
    stage_timeout_s: int
    settings: ProviderSettings
    #: Raw, unvalidated blocks for the providers that are *not* selected (kept for `dump()`).
    other_blocks: dict[str, Any] = field(default_factory=dict)
    #: The selected provider's credential, or None for claude_subscription. Never printed.
    api_key: str | None = field(default=None, repr=False, compare=False)
    #: Names (never values) of the credential variables found in env or .env, for `dump()`.
    credentials_present: frozenset[str] = field(default_factory=frozenset, compare=False)

    @property
    def models(self) -> dict[str, str]:
        return dict(self.settings.models)

    def model_for(self, stage: str) -> str:
        if stage not in STAGES:
            raise ConfigError(f"unknown stage {stage!r}; expected one of {', '.join(STAGES)}")
        return self.settings.models[stage]

    @property
    def credential_var(self) -> str | None:
        return _CREDENTIAL_FOR_PROVIDER[self.provider]

    def __repr__(self) -> str:
        return (
            f"LLMConfig(provider={self.provider!r}, models={self.models!r}, "
            f"repair_retries={self.repair_retries}, api_key={'<set>' if self.api_key else None})"
        )

    __str__ = __repr__


@dataclass(frozen=True)
class ChunkingConfig:
    target_tokens: int
    max_tokens: int
    min_tokens: int


@dataclass(frozen=True)
class LimitsConfig:
    max_nodes_per_chunk: int
    max_dedup_pairs: int
    roster_cap: int
    soft_budget_usd: float


@dataclass(frozen=True)
class DedupConfig:
    threshold: float
    write_same_as: bool
    #: Pairs adjudicated per dedup call (DESIGN §8.0, D29). Optional in kg.yaml; default 10.
    pairs_per_call: int = 10


@dataclass(frozen=True)
class ImageConfig:
    max_long_edge_px: int


@dataclass(frozen=True)
class WebConfig:
    timeout_s: int
    user_agent: str


@dataclass(frozen=True)
class RelevanceSampleConfig:
    n: int
    bands: list[list[int]]
    seed: int


@dataclass(frozen=True)
class Config:
    config_path: Path
    project_root: Path
    version: int
    corpus: CorpusConfig
    paths: PathsConfig
    llm: LLMConfig
    chunking: ChunkingConfig
    limits: LimitsConfig
    dedup: DedupConfig
    image: ImageConfig
    web: WebConfig
    relevance_sample: RelevanceSampleConfig
    sandbox: Sandbox = field(compare=False)

    def dump(self) -> dict[str, Any]:
        """Redacted, JSON-serialisable view for reports. Credentials appear only as set/unset."""
        out: dict[str, Any] = {
            "config_path": str(self.config_path),
            "project_root": str(self.project_root),
            "version": self.version,
            "corpus": _plain(self.corpus),
            "paths": self.paths.as_dict(),
            "llm": {
                "provider": self.llm.provider,
                "repair_retries": self.llm.repair_retries,
                "max_output_tokens": self.llm.max_output_tokens,
                "stage_timeout_s": self.llm.stage_timeout_s,
                "models": self.llm.models,
                self.llm.provider: _plain(self.llm.settings),
                **{k: _plain(v) for k, v in self.llm.other_blocks.items()},
            },
            "chunking": _plain(self.chunking),
            "limits": _plain(self.limits),
            "dedup": _plain(self.dedup),
            "image": _plain(self.image),
            "web": _plain(self.web),
            "relevance_sample": _plain(self.relevance_sample),
            "credentials": {var: ("present" if var in self.llm.credentials_present else "absent") for var in CREDENTIAL_VARS},
        }
        return out


def _plain(value: Any) -> Any:
    """Recursively convert dataclasses/paths to JSON-friendly builtins."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value) if f.repr}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


# ------------------------------------------------------------ typed readers


def _get(doc: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = doc
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise ConfigError(f"kg.yaml: missing required key '{dotted}'")
        cur = cur[part]
    return cur


def _mapping(doc: Mapping[str, Any], dotted: str) -> Mapping[str, Any]:
    value = _get(doc, dotted)
    if not isinstance(value, Mapping):
        raise ConfigError(f"kg.yaml: '{dotted}' must be a mapping")
    return value


def _str(doc: Mapping[str, Any], dotted: str, *, choices: tuple[str, ...] | None = None) -> str:
    value = _get(doc, dotted)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"kg.yaml: '{dotted}' must be a non-empty string")
    if choices is not None and value not in choices:
        raise ConfigError(f"kg.yaml: '{dotted}' must be one of {', '.join(choices)} (got {value!r})")
    return value


def _int(doc: Mapping[str, Any], dotted: str, *, minimum: int | None = None) -> int:
    value = _get(doc, dotted)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"kg.yaml: '{dotted}' must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"kg.yaml: '{dotted}' must be >= {minimum}")
    return value


def _optional_int(doc: Mapping[str, Any], dotted: str, *, default: int, minimum: int | None = None) -> int:
    """Like `_int` but the key may be absent, in which case `default` is used."""
    try:
        _get(doc, dotted)
    except ConfigError:
        return default
    return _int(doc, dotted, minimum=minimum)


def _float(doc: Mapping[str, Any], dotted: str, *, minimum: float | None = None) -> float:
    value = _get(doc, dotted)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"kg.yaml: '{dotted}' must be a number")
    if minimum is not None and value < minimum:
        raise ConfigError(f"kg.yaml: '{dotted}' must be >= {minimum}")
    return float(value)


def _bool(doc: Mapping[str, Any], dotted: str) -> bool:
    value = _get(doc, dotted)
    if not isinstance(value, bool):
        raise ConfigError(f"kg.yaml: '{dotted}' must be true or false")
    return value


def _str_list(doc: Mapping[str, Any], dotted: str) -> list[str]:
    value = _get(doc, dotted)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"kg.yaml: '{dotted}' must be a list of strings")
    return list(value)


def _models(doc: Mapping[str, Any], dotted: str) -> dict[str, str]:
    """A complete per-stage model map: every stage present, non-empty string."""
    block = _mapping(doc, dotted)
    missing = [s for s in STAGES if s not in block]
    if missing:
        raise ConfigError(f"kg.yaml: '{dotted}' is missing stage(s): {', '.join(missing)}")
    unknown = [s for s in block if s not in STAGES]
    if unknown:
        raise ConfigError(f"kg.yaml: '{dotted}' has unknown stage(s): {', '.join(unknown)}")
    return {s: _str(doc, f"{dotted}.{s}") for s in STAGES}


PRICING_MULTIPLIER_KEYS: tuple[str, ...] = ("cache_read_multiplier", "cache_write_multiplier")


def _parse_pricing(doc: Mapping[str, Any], dotted: str) -> dict[str, Any]:
    """`llm.anthropic_api.pricing_usd_per_mtok`: model rows `{input, output}` plus optional cache multipliers.

    Validated at load time so a malformed price table fails before any paid call
    (`compute_cost` would otherwise raise after the tokens were already spent).
    """
    block = _mapping(doc, dotted)
    out: dict[str, Any] = {}
    for key, value in block.items():
        if not isinstance(key, str) or not key.strip():
            raise ConfigError(f"kg.yaml: '{dotted}' keys must be non-empty model IDs or {', '.join(PRICING_MULTIPLIER_KEYS)}")
        if key in PRICING_MULTIPLIER_KEYS:
            out[key] = _float(doc, f"{dotted}.{key}", minimum=0)
            continue
        row = _mapping(doc, f"{dotted}.{key}")
        unknown = [k for k in row if k not in ("input", "output")]
        if unknown:
            raise ConfigError(f"kg.yaml: '{dotted}.{key}' has unknown key(s): {', '.join(str(k) for k in unknown)} (expected input, output)")
        out[key] = {"input": _float(doc, f"{dotted}.{key}.input", minimum=0), "output": _float(doc, f"{dotted}.{key}.output", minimum=0)}
    return out


# ---------------------------------------------------------- section parsers


def _parse_provider_settings(doc: Mapping[str, Any], provider: str) -> ProviderSettings:
    base = f"llm.{provider}"
    if provider == "claude_subscription":
        return ClaudeSubscriptionSettings(
            models=_models(doc, f"{base}.models"),
            effort=_str(doc, f"{base}.effort", choices=("low", "medium", "high")),
            rate_limit_wait_minutes=_int(doc, f"{base}.rate_limit_wait_minutes", minimum=0),
        )
    if provider == "openrouter":
        return OpenRouterSettings(
            base_url=_str(doc, f"{base}.base_url"),
            models=_models(doc, f"{base}.models"),
            provider=OpenRouterRouting(
                require_parameters=_bool(doc, f"{base}.provider.require_parameters"),
                data_collection=_str(doc, f"{base}.provider.data_collection", choices=("allow", "deny")),
                order=_str_list(doc, f"{base}.provider.order"),
                allow_fallbacks=_bool(doc, f"{base}.provider.allow_fallbacks"),
            ),
            app_title=_str(doc, f"{base}.app_title"),
        )
    if provider == "anthropic_api":
        return AnthropicApiSettings(
            models=_models(doc, f"{base}.models"),
            pricing_usd_per_mtok=_parse_pricing(doc, f"{base}.pricing_usd_per_mtok"),
        )
    if provider == "azure_foundry":
        return AzureFoundrySettings(
            endpoint=_str(doc, f"{base}.endpoint"),
            api_version=_str(doc, f"{base}.api_version"),
            models=_models(doc, f"{base}.models"),
        )
    raise ConfigError(f"kg.yaml: 'llm.provider' must be one of {', '.join(PROVIDERS)} (got {provider!r})")


def _read_credentials(project_root: Path) -> dict[str, str]:
    """Non-empty credential values from the process env, then `.env` next to kg.yaml.

    Never writes to `os.environ`. Values are returned only so the selected
    provider's key can be attached to the config; they are not logged.
    """
    dotenv_path = project_root / ".env"
    file_values: Mapping[str, str | None] = {}
    if dotenv_path.is_file():
        try:
            file_values = dotenv_values(dotenv_path, interpolate=False)
        except OSError as exc:
            raise ConfigError(f"cannot read {dotenv_path.name} next to kg.yaml: {exc.strerror or exc.__class__.__name__}") from exc

    found: dict[str, str] = {}
    for var in CREDENTIAL_VARS:
        raw = os.environ.get(var)
        if raw is None or not raw.strip():
            raw = file_values.get(var)
        if raw is not None and raw.strip():
            found[var] = raw.strip()
    return found


def _parse_llm(doc: Mapping[str, Any], project_root: Path, provider_override: str | None = None) -> LLMConfig:
    provider_value = provider_override if provider_override is not None else _get(doc, "llm.provider")
    if not isinstance(provider_value, str) or provider_value not in PROVIDERS:
        raise ConfigError(f"kg.yaml: 'llm.provider' must be one of {', '.join(PROVIDERS)} (got {provider_value!r})")
    provider = provider_value

    settings = _parse_provider_settings(doc, provider)
    llm_block = _mapping(doc, "llm")
    other_blocks = {p: llm_block[p] for p in PROVIDERS if p != provider and p in llm_block}

    credentials = _read_credentials(project_root)
    api_key: str | None = None
    if provider == "claude_subscription":
        # D22: the Claude Code CLI prefers an API key over the subscription login.
        if "ANTHROPIC_API_KEY" in credentials:
            raise ConfigError(
                "llm.provider is 'claude_subscription' but ANTHROPIC_API_KEY is set "
                "(environment or .env). The Claude Code CLI would bill the Console account "
                "instead of the subscription; unset ANTHROPIC_API_KEY or switch to provider anthropic_api."
            )
    else:
        var = _CREDENTIAL_FOR_PROVIDER[provider]
        assert var is not None
        if var not in credentials:
            raise ConfigError(
                f"llm.provider is '{provider}' but {var} is not set. "
                f"Put it in .env next to kg.yaml (see .env.example) or export it in the environment."
            )
        api_key = credentials[var]

    return LLMConfig(
        provider=provider,
        repair_retries=_int(doc, "llm.repair_retries", minimum=0),
        max_output_tokens=_int(doc, "llm.max_output_tokens", minimum=1),
        stage_timeout_s=_int(doc, "llm.stage_timeout_s", minimum=1),
        settings=settings,
        other_blocks=other_blocks,
        api_key=api_key,
        credentials_present=frozenset(credentials),
    )


def _parse_paths(doc: Mapping[str, Any], project_root: Path) -> tuple[PathsConfig, Sandbox]:
    block = _mapping(doc, "paths")
    roots: dict[str, str] = {}
    for kind in KINDS:
        if kind not in block:
            raise ConfigError(f"kg.yaml: missing required key 'paths.{kind}'")
        value = block[kind]
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"kg.yaml: 'paths.{kind}' must be a non-empty relative path")
        roots[kind] = value
    unknown = [k for k in block if k not in KINDS]
    if unknown:
        raise ConfigError(f"kg.yaml: 'paths' has unknown folder kind(s): {', '.join(unknown)}")
    try:
        sandbox = Sandbox(project_root=project_root, roots=roots)
    except SandboxViolation as exc:
        raise ConfigError(f"kg.yaml: {exc}") from exc
    return PathsConfig(**roots), sandbox


def _parse_bands(doc: Mapping[str, Any], dotted: str) -> list[list[int]]:
    value = _get(doc, dotted)
    ok = isinstance(value, list) and all(
        isinstance(b, list) and len(b) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in b) for b in value
    )
    if not ok:
        raise ConfigError(f"kg.yaml: '{dotted}' must be a list of [low, high] integer pairs")
    return [list(b) for b in value]


# -------------------------------------------------------------------- loader


def load_config(path: str | os.PathLike[str], *, provider: str | None = None) -> Config:
    """Load and validate kg.yaml at `path`; read `.env` from the same directory.

    `provider` overrides `llm.provider` for this load (`kg ingest --provider`, DESIGN
    §3.2); the credential guards then apply to the overriding provider.
    """
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    project_root = config_path.parent

    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {config_path}: {exc.strerror or exc.__class__.__name__}") from exc
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{config_path.name}: malformed YAML: {exc}") from exc
    if not isinstance(doc, Mapping):
        raise ConfigError(f"{config_path.name}: top level must be a mapping")

    version = _int(doc, "version")
    if version != SUPPORTED_VERSION:
        raise ConfigError(f"kg.yaml: unsupported 'version' {version}; this build reads version {SUPPORTED_VERSION}")

    corpus = CorpusConfig(
        slug=_str(doc, "corpus.slug"),
        name=_str(doc, "corpus.name"),
        schema=_str(doc, "corpus.schema", choices=SCHEMA_NAMES),
    )
    paths, sandbox = _parse_paths(doc, project_root)
    llm = _parse_llm(doc, project_root, provider)

    chunking = ChunkingConfig(
        target_tokens=_int(doc, "chunking.target_tokens", minimum=1),
        max_tokens=_int(doc, "chunking.max_tokens", minimum=1),
        min_tokens=_int(doc, "chunking.min_tokens", minimum=0),
    )
    if not chunking.min_tokens <= chunking.target_tokens <= chunking.max_tokens:
        raise ConfigError("kg.yaml: chunking must satisfy min_tokens <= target_tokens <= max_tokens")

    limits = LimitsConfig(
        max_nodes_per_chunk=_int(doc, "limits.max_nodes_per_chunk", minimum=1),
        max_dedup_pairs=_int(doc, "limits.max_dedup_pairs", minimum=0),
        roster_cap=_int(doc, "limits.roster_cap", minimum=1),
        soft_budget_usd=_float(doc, "limits.soft_budget_usd", minimum=0),
    )
    dedup = DedupConfig(
        threshold=_float(doc, "dedup.threshold", minimum=0),
        write_same_as=_bool(doc, "dedup.write_same_as"),
        pairs_per_call=_optional_int(doc, "dedup.pairs_per_call", default=10, minimum=1),
    )
    image = ImageConfig(max_long_edge_px=_int(doc, "image.max_long_edge_px", minimum=1))
    web = WebConfig(timeout_s=_int(doc, "web.timeout_s", minimum=1), user_agent=_str(doc, "web.user_agent"))
    relevance_sample = RelevanceSampleConfig(
        n=_int(doc, "relevance_sample.n", minimum=1),
        bands=_parse_bands(doc, "relevance_sample.bands"),
        seed=_int(doc, "relevance_sample.seed"),
    )

    return Config(
        config_path=config_path,
        project_root=project_root,
        version=version,
        corpus=corpus,
        paths=paths,
        llm=llm,
        chunking=chunking,
        limits=limits,
        dedup=dedup,
        image=image,
        web=web,
        relevance_sample=relevance_sample,
        sandbox=sandbox,
    )


# ------------------------------------------------------------ adapter factory


def adapter_factory(cfg: Config) -> Any:
    """Build the adapter for `cfg.llm.provider` (DESIGN §7.2, §17 "Config guards").

    Constructs only: no subprocess, no socket. Adapter modules are imported here,
    lazily, so `kg.config` itself never imports a provider SDK. Returns an object
    satisfying `kg.llm.adapters.base.Adapter`.
    """
    provider = cfg.llm.provider
    if provider == "claude_subscription":
        from kg.llm.adapters.claude_subscription import ClaudeSubscriptionAdapter

        return ClaudeSubscriptionAdapter(cfg)
    if provider == "openrouter":
        from kg.llm.adapters.openrouter import OpenRouterAdapter

        return OpenRouterAdapter(cfg)
    if provider == "anthropic_api":
        from kg.llm.adapters.anthropic_api import AnthropicApiAdapter

        return AnthropicApiAdapter(cfg)
    if provider == "azure_foundry":
        from kg.llm.adapters.azure_foundry import AzureFoundryAdapter

        return AzureFoundryAdapter(cfg)
    raise ConfigError(f"llm.provider {provider!r} is not a known provider; expected one of {', '.join(PROVIDERS)}")
