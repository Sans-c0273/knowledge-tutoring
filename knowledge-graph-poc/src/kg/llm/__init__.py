"""LLM boundary: `call_stage` and the validate-or-repair loop (DESIGN §7.1; D20).

Provider-agnostic. Steps per call:

1. resolve the stage's model from config and pick the adapter (`adapter_factory`
   unless one is injected);
2. export the stage schema through the profile filter (§7.7);
3. check the soft budget, call `adapter.complete()`, record a ledger row for
   every attempt (ok / invalid / truncated / error);
4. on a JSON or Pydantic failure append the bad output as an assistant turn plus
   a repair user turn and try again, at most
   `min(llm.repair_retries, adapter.max_repairs)` times; then raise
   `StageOutputInvalid` — nothing partial is ever returned;
5. `stop_reason` max_tokens / length raises `StageTruncated` with no repair.

Nothing in this package imports a provider SDK except modules under
`kg.llm.adapters` (asserted by the selftest).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from kg.config import Config, ConfigError
from kg.llm.adapters.base import Adapter, ImageInput, Msg, RawCompletion, Usage
from kg.llm.errors import BudgetExceeded, LLMError, ProviderError, RateLimited, StageOutputInvalid, StageTruncated
from kg.llm.ledger import UsageLedger, UsageRow
from kg.llm.schema_profile import export

T = TypeVar("T", bound=BaseModel)

TRUNCATION_STOP_REASONS: frozenset[str] = frozenset({"max_tokens", "length"})
SYSTEM_SHA_LEN = 16
#: Exactly one Markdown code fence wrapping the whole output (``` or ```json), nothing outside it.
_FENCE_RE = re.compile(r"\A\s*```[ \t]*(?:json|JSON)?[ \t]*\r?\n(.*?)\r?\n?[ \t]*```\s*\Z", re.DOTALL)


@dataclass(frozen=True)
class StageResult(Generic[T]):
    data: T
    usage: Usage
    provider: str
    model_requested: str
    model_served: str | None
    attempts: int
    prompt_version: str | None = None


def system_sha(system: str) -> str:
    """Prefix of sha256(system) recorded on every ledger row (prompt identity, DESIGN §7.8)."""
    return hashlib.sha256(system.encode("utf-8")).hexdigest()[:SYSTEM_SHA_LEN]


def _echo_text(text_or_obj: Any) -> str:
    """The model's output as text, for the assistant turn of a repair round."""
    if text_or_obj is None:
        return ""
    if isinstance(text_or_obj, str):
        return text_or_obj
    return json.dumps(text_or_obj, ensure_ascii=False)


def _validation_summary(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ())) or "$"
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")
    return "; ".join(parts)


def _strip_fence(text: str) -> str:
    """Remove ONE ```json / ``` fence that wraps the whole text; anything else is returned unchanged."""
    m = _FENCE_RE.match(text)
    return m.group(1) if m else text


def _parse(text_or_obj: Any, schema: type[T]) -> tuple[T | None, str | None]:
    """(instance, None) on success; (None, reason) on any JSON or schema failure."""
    obj = text_or_obj
    if not isinstance(obj, dict):
        if not isinstance(obj, str):
            return None, f"output was {type(obj).__name__}, expected a JSON object"
        try:
            obj = json.loads(_strip_fence(obj))
        except json.JSONDecodeError as exc:
            return None, f"output is not valid JSON ({exc.msg} at position {exc.pos})"
        if not isinstance(obj, dict):
            return None, f"top-level JSON value must be an object, got {type(obj).__name__}"
    try:
        return schema.model_validate(obj), None
    except ValidationError as exc:
        return None, _validation_summary(exc)


def _repair_msg(reason: str) -> Msg:
    return Msg(role="user", text=f"The JSON failed validation: {reason}. Return only corrected JSON matching the schema.")


def _retry_msg(reason: str) -> Msg:
    """Repair turn when the adapter itself reported failure and there is no model output to echo."""
    return Msg(
        role="user",
        text=f"The previous attempt produced no output matching the schema ({reason}). Try again and return only JSON matching the schema.",
    )


def _row(
    *,
    stage: str,
    provider: str,
    model: str,
    attempt: int,
    outcome: str,
    duration_s: float,
    prompt_version: str | None,
    sha: str,
    rc: RawCompletion | None = None,
    exc: Exception | None = None,
) -> UsageRow:
    """One ledger row; token/cost fields come from `rc`, or from what an adapter-raised `exc` carried.

    Provider events come from `rc`, or from the exception's `events` (RateLimited /
    StageOutputInvalid raised by the adapter), so a rejected call still leaves its
    rate-limit events in the ledger.
    """
    invalid = exc if isinstance(exc, StageOutputInvalid) else None
    usage = rc.usage if rc is not None else (invalid.usage if invalid is not None and isinstance(invalid.usage, Usage) else Usage(0, 0))
    if rc is not None:
        model_served, cost_estimate = rc.model_served, rc.provider_meta.get("cost_estimate_usd")
    elif invalid is not None:
        model_served, cost_estimate = invalid.model_served, invalid.cost_estimate_usd
    else:
        model_served, cost_estimate = None, None
    events = tuple(rc.events) if rc is not None else tuple(getattr(exc, "events", ()) or ())
    return UsageRow(
        stage=stage,
        provider=provider,
        model_requested=rc.model_requested if rc is not None else model,
        model_served=model_served,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_tokens=usage.cached_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        cost_usd=rc.cost_usd if rc is not None else None,
        cost_estimate_usd=cost_estimate,
        attempt_no=attempt,
        outcome=outcome,
        duration_s=duration_s,
        prompt_version=prompt_version,
        system_sha=sha,
        events=events,
    )


def call_stage(
    stage: str,
    system: str,
    user_text: str,
    schema: type[T],
    images: list[ImageInput] | None = None,
    *,
    cfg: Config,
    ledger: UsageLedger,
    adapter: Adapter | None = None,
    prompt_version: str | None = None,
) -> StageResult[T]:
    """Run one model stage and return a validated `schema` instance or raise an `LLMError`."""
    model = cfg.llm.model_for(stage)  # ConfigError on an unknown stage, before anything else
    if adapter is None:
        from kg.config import adapter_factory

        adapter = adapter_factory(cfg)
    schema_json = export(schema)

    repairs = cfg.llm.repair_retries
    cap = getattr(adapter, "max_repairs", None)
    if cap is not None:
        repairs = min(repairs, cap)
    max_attempts = 1 + repairs

    sha = system_sha(system)
    messages: list[Msg] = [Msg(role="user", text=user_text, images=tuple(images or ()))]
    common = {"stage": stage, "provider": adapter.provider, "model": model, "prompt_version": prompt_version, "sha": sha}

    for attempt in range(1, max_attempts + 1):
        ledger.check_budget()
        started = time.perf_counter()
        try:
            rc = adapter.complete(system, list(messages), schema_json, model, cfg.llm.max_output_tokens)
        except StageOutputInvalid as exc:
            # The adapter itself detected exhaustion of native structured-output retries (Agent SDK).
            # No model output exists to echo, so the repair turn is a plain retry instruction (no assistant turn).
            ledger.record(_row(attempt=attempt, outcome="invalid", duration_s=time.perf_counter() - started, exc=exc, **common))
            if attempt == max_attempts:
                raise StageOutputInvalid(
                    f"{stage}: {exc}",
                    stage=stage,
                    attempts=attempt,
                    usage=exc.usage,
                    cost_estimate_usd=exc.cost_estimate_usd,
                    model_served=exc.model_served,
                    events=exc.events,
                ) from exc
            messages.append(_retry_msg(str(exc)))
            continue
        except Exception as exc:
            # A RateLimited (or other provider failure) still gets its row; any events it carries land in the ledger.
            ledger.record(_row(attempt=attempt, outcome="error", duration_s=time.perf_counter() - started, exc=exc, **common))
            raise
        duration_s = time.perf_counter() - started

        if rc.stop_reason in TRUNCATION_STOP_REASONS:
            ledger.record(_row(attempt=attempt, outcome="truncated", duration_s=duration_s, rc=rc, **common))
            raise StageTruncated(
                f"{stage}: output truncated by the provider (stop_reason={rc.stop_reason!r}) after {rc.usage.output_tokens} output tokens",
                stage=stage,
                attempts=attempt,
            )

        data, reason = _parse(rc.text_or_obj, schema)
        if data is not None:
            ledger.record(_row(attempt=attempt, outcome="ok", duration_s=duration_s, rc=rc, **common))
            return StageResult(
                data=data,
                usage=rc.usage,
                provider=adapter.provider,
                model_requested=rc.model_requested,
                model_served=rc.model_served,
                attempts=attempt,
                prompt_version=prompt_version,
            )

        ledger.record(_row(attempt=attempt, outcome="invalid", duration_s=duration_s, rc=rc, **common))
        if attempt == max_attempts:
            raise StageOutputInvalid(
                f"{stage}: output failed validation after {attempt} attempt(s): {reason}", stage=stage, attempts=attempt
            )
        messages.append(Msg(role="assistant", text=_echo_text(rc.text_or_obj)))
        messages.append(_repair_msg(reason or "unknown error"))

    raise AssertionError("unreachable: repair loop exited without returning or raising")  # pragma: no cover


__all__ = [
    "BudgetExceeded",
    "ConfigError",
    "LLMError",
    "ProviderError",
    "RateLimited",
    "StageOutputInvalid",
    "StageResult",
    "StageTruncated",
    "call_stage",
    "system_sha",
]
