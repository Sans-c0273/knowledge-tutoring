"""Adapter protocol and wire types shared by every provider (DESIGN §7.1, §4.3, §20).

No provider SDK is imported here (asserted by the selftest). An adapter is any
object with a `provider` name and a
`complete(system, messages, schema_json, model, max_tokens) -> RawCompletion`
method; it may also expose `max_repairs` to cap the shared repair loop (D20).
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only; kg.config is imported lazily in require_provider
    from kg.config import Config

ROLES: tuple[str, ...] = ("user", "assistant")


@dataclass(frozen=True)
class ImageInput:
    """Base64 image payload handed to the provider boundary (DESIGN §4.3)."""

    media_type: str
    data_b64: str

    @classmethod
    def from_bytes(cls, data: bytes, media_type: str) -> ImageInput:
        if not media_type.startswith("image/") or len(media_type) <= len("image/"):
            raise ValueError(f"media_type must be an image type, got {media_type!r}")
        if not data:
            raise ValueError("image payload is empty")
        return cls(media_type=media_type, data_b64=base64.b64encode(data).decode("ascii"))


@dataclass(frozen=True)
class Msg:
    """One conversation turn. The system prompt is passed separately, never as a Msg."""

    role: str
    text: str
    images: tuple[ImageInput, ...] = ()

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"Msg.role must be one of {', '.join(ROLES)}; got {self.role!r}")
        object.__setattr__(self, "images", tuple(self.images))


@dataclass(frozen=True)
class Usage:
    """Token counts for one attempt; `cached_tokens` are cache-read input tokens."""

    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "cached_tokens", "reasoning_tokens"):
            if getattr(self, name) < 0:
                raise ValueError(f"Usage.{name} must be >= 0")


@dataclass(frozen=True)
class RawCompletion:
    """What an adapter returns: the raw output plus everything the ledger records.

    `text_or_obj` is a dict when the provider produced native structured output,
    otherwise the text to `json.loads`. `cost_usd` is None when unknown or not
    billed per token (D28) — never 0.0 by default.
    """

    text_or_obj: Any
    usage: Usage
    model_requested: str
    model_served: str | None = None
    cost_usd: float | None = None
    stop_reason: str | None = None
    provider_meta: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)


@runtime_checkable
class Adapter(Protocol):
    """Structural type for a provider adapter (DESIGN §7.1 step 1).

    Adapters may also define `max_repairs: int | None`; `call_stage` reads it
    with `getattr(adapter, "max_repairs", None)` so minimal fakes without the
    attribute still satisfy the protocol.
    """

    provider: str

    def complete(self, system: str, messages: list[Msg], schema_json: dict[str, Any], model: str, max_tokens: int) -> RawCompletion: ...


# ------------------------------------------------------------ shared helpers


def require_provider(cfg: Config, provider: str, adapter_name: str) -> None:
    """Refuse to build `adapter_name` for a config that selects another provider (D22)."""
    from kg.config import ConfigError  # lazy: keeps this module free of config-time imports

    if cfg.llm.provider != provider:
        raise ConfigError(f"{adapter_name} needs llm.provider '{provider}', config selects {cfg.llm.provider!r}")


def redact(text: str, secret: str | None) -> str:
    """Replace every occurrence of `secret` in `text` (error messages must never carry a key)."""
    return text.replace(secret, "<redacted>") if secret else text


def anthropic_image_block(img: ImageInput) -> dict[str, Any]:
    """Anthropic-style base64 image content block (Agent SDK and Console API)."""
    return {"type": "image", "source": {"type": "base64", "media_type": img.media_type, "data": img.data_b64}}


def openai_image_block(img: ImageInput) -> dict[str, Any]:
    """OpenAI-style data-URL image part (OpenRouter)."""
    return {"type": "image_url", "image_url": {"url": f"data:{img.media_type};base64,{img.data_b64}"}}


def wire_messages(messages: list[Msg], image_block: Callable[[ImageInput], dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn the Msg list into provider wire dicts: image blocks first, then the text block.

    Text-only turns are sent as a plain string `content`, as every provider accepts.
    """
    wire: list[dict[str, Any]] = []
    for m in messages:
        if m.images:
            content: list[dict[str, Any]] = [image_block(img) for img in m.images]
            content.append({"type": "text", "text": m.text})
            wire.append({"role": m.role, "content": content})
        else:
            wire.append({"role": m.role, "content": m.text})
    return wire


__all__ = [
    "ROLES",
    "Adapter",
    "ImageInput",
    "Msg",
    "RawCompletion",
    "Usage",
    "anthropic_image_block",
    "openai_image_block",
    "redact",
    "require_provider",
    "wire_messages",
]
