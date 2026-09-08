"""Provider protocol, wire types, and the strict-JSON-schema builder (R21).

Two call shapes cover the whole pipeline:

- `complete_structured` — Calls A (intent) and B (answer evaluation): temperature 0,
  short output, schema-enforced (Tech Spec §5.1, §5.2).
- `stream_text` — Call C (generation): streamed so the student sees tokens before
  the turn completes (Tech Spec §5.3, §10).

Both record token usage and latency: `complete_structured` returns them on
`StructuredResult`, `stream_text` fills a caller-supplied `StreamStats` (an
async generator cannot also return a value).

No provider SDK is imported here.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

ROLES_TO_STREAM: tuple[str, ...] = ("generation",)

#: JSON Schema keywords that strict structured-output modes reject. Ranges and
#: formats are validated by the pedagogy layer, never on the wire.
_UNSUPPORTED_KEYWORDS: frozenset[str] = frozenset(
    {
        "default",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "format",
        "maxItems",
        "maxLength",
        "maximum",
        "minItems",
        "minLength",
        "minimum",
        "multipleOf",
        "pattern",
        "patternProperties",
        "uniqueItems",
    }
)


# ------------------------------------------------------------------- errors


class ProviderError(Exception):
    """A provider call failed. Base class for every error raised from this package."""


class ProviderConfigError(ProviderError):
    """The provider is selected but not usable — missing credential, unknown name."""


class ProviderModelNotFound(ProviderError):
    """The requested model ID does not exist for this provider (HTTP 404)."""


class ProviderRateLimited(ProviderError):
    """The provider is rate-limiting (HTTP 429). Expected under subscription auth."""


class ProviderConnectionError(ProviderError):
    """The request never reached the provider — DNS, TLS, timeout."""


class ProviderResponseError(ProviderError):
    """The provider answered, but the payload was not what the contract requires."""


# --------------------------------------------------------------- wire types


@dataclass(frozen=True)
class Msg:
    """One conversation turn. The system prompt is passed separately, never as a Msg."""

    role: Literal["user", "assistant"]
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("user", "assistant"):
            raise ValueError(f"Msg.role must be 'user' or 'assistant'; got {self.role!r}")


@dataclass(frozen=True)
class Usage:
    """Token counts for one call."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class StructuredResult:
    """Parsed structured output plus what the trace records (Tech Spec §7)."""

    value: BaseModel
    usage: Usage
    latency_ms: float
    model: str
    provider: str


@dataclass
class StreamStats:
    """Mutable sink for a streaming call's measurements.

    `ttft_ms` is time-to-first-token — the Tech Spec §10 target the pre-stream
    latency budget exists to protect.
    """

    model: str = ""
    provider: str = ""
    ttft_ms: float | None = None
    total_ms: float | None = None
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None


def prompt_version(prompt: str) -> str:
    """Short content hash identifying a system prompt (Tech Spec §7).

    Recorded per role on `TurnTrace.prompt_versions`, so a shift in behaviour can
    be attributed to a prompt edit after the fact. A hash rather than a hand-kept
    number: nobody remembers to bump a number, and a stale one is worse than none.
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def to_wire(messages: Sequence[Msg]) -> list[dict[str, str]]:
    """Message list as both APIs accept it: `{"role": ..., "content": ...}`."""
    return [{"role": m.role, "content": m.content} for m in messages]


# ------------------------------------------------------- strict JSON schema


def strict_json_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Pydantic model → a JSON schema both strict modes accept.

    Anthropic strict tool use and OpenAI `json_schema` strict mode agree on the
    same restrictions, so one builder serves both:

    - `$ref`/`$defs` are fully inlined (nested models and enums become literal
      subschemas), so no resolver is needed on the provider side;
    - every object gets `additionalProperties: false` and `required` listing
      *all* its properties — strict mode has no notion of an optional field, so
      optionality must be expressed as a nullable type in the model;
    - validation keywords strict mode rejects (`minimum`, `pattern`, `default`, …)
      are dropped.

    **Every field becomes required, which makes every field a model obligation.**
    That is correct for strict tool use and it is exactly why a *wire* model and
    a *pipeline* model must never be the same class: a field the pipeline
    computes about the model (a fallback flag, a latency, a decision made after
    reading the output) would be handed back to the model to invent, and an
    invented value then reaches the trace and the glass-box UI looking like a
    system fact. Mark such fields with
    `Field(json_schema_extra=models.intent.SYSTEM_COMPUTED)` and this function
    will refuse the model outright.

    Raises `ProviderError` for a model that cannot be made strict: one carrying a
    system-computed field, a free-form object (`dict[str, Any]`), or recursion.
    """
    raw = schema.model_json_schema(ref_template="#/$defs/{model}")
    defs = raw.pop("$defs", {})
    return _strictify(raw, defs, (), schema.__name__)


def _strictify(node: Any, defs: dict[str, Any], seen: tuple[str, ...], path: str) -> Any:
    if isinstance(node, list):
        return [_strictify(item, defs, seen, f"{path}[{i}]") for i, item in enumerate(node)]
    if not isinstance(node, dict):
        return node

    node = dict(node)

    if node.pop("system_computed", False):
        raise ProviderError(
            f"{path}: field is system-computed and must never be a model obligation. "
            f"This model is a pipeline model, not a wire model — build a narrow wire "
            f"model containing only what the model decides, and construct the pipeline "
            f"model from its output."
        )

    ref = node.pop("$ref", None)
    if ref is not None:
        name = ref.rsplit("/", 1)[-1]
        if name in seen:
            raise ProviderError(
                f"{path}: recursive model {name!r} cannot be expressed as a strict schema"
            )
        target = defs.get(name)
        if target is None:
            raise ProviderError(f"{path}: unresolved schema reference {ref!r}")
        # Siblings of a $ref (title, description) override the target's.
        return _strictify({**target, **node}, defs, (*seen, name), path)

    all_of = node.pop("allOf", None)
    if all_of is not None:
        if len(all_of) != 1:
            raise ProviderError(
                f"{path}: allOf with {len(all_of)} branches is not supported in strict mode"
            )
        return _strictify({**all_of[0], **node}, defs, seen, path)

    for keyword in _UNSUPPORTED_KEYWORDS:
        node.pop(keyword, None)

    for key in ("anyOf", "oneOf", "items", "prefixItems"):
        if key in node:
            node[key] = _strictify(node[key], defs, seen, f"{path}.{key}")

    properties = node.get("properties")
    if properties is not None:
        node["properties"] = {
            name: _strictify(sub, defs, seen, f"{path}.{name}") for name, sub in properties.items()
        }
        node["additionalProperties"] = False
        node["required"] = list(node["properties"])
    elif node.get("type") == "object":
        raise ProviderError(
            f"{path}: free-form object fields have no strict-schema equivalent; "
            f"declare an explicit model instead of dict[str, Any]"
        )

    return node


# ------------------------------------------------------------- the protocol


class LLMProvider(ABC):
    """One LLM backend, addressed per role (R21).

    Implementations must not hold per-turn state: a single instance is shared
    across concurrent turns.
    """

    name: str = "base"

    @abstractmethod
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
        """Call the model and parse its output into `schema` (Calls A and B).

        Raises a `ProviderError` subclass on transport, status, or parse failure —
        never returns a partially populated result.
        """
        raise NotImplementedError

    @abstractmethod
    def stream_text(
        self,
        *,
        model: str,
        system: str,
        messages: Sequence[Msg],
        max_tokens: int = 2000,
        stats: StreamStats | None = None,
    ) -> AsyncIterator[str]:
        """Stream the model's text deltas (Call C).

        Implementations are async generators, so the return value is used
        directly: `async for delta in provider.stream_text(...)`. When `stats`
        is given it is filled with usage, TTFT and total latency as the stream
        proceeds and completes.
        """
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release transport resources. Safe to call more than once."""
        return


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
    "prompt_version",
    "strict_json_schema",
    "to_wire",
]
