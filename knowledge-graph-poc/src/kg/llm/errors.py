"""Exceptions raised across the LLM boundary (DESIGN §7.1, §7.6; D20).

Kept in their own module so `kg.llm.ledger` and the adapters can import them
without importing `kg.llm` itself. Everything roots at `LLMError`, which is
deliberately *not* a `ConfigError`: a bad config is caught before any call; an
`LLMError` happens during one.
"""

from __future__ import annotations

from typing import Any


class LLMError(Exception):
    """Base class for every failure raised by `call_stage` or an adapter."""


class StageOutputInvalid(LLMError):
    """The model's output never validated against the stage schema (after repairs).

    `stage` and `attempts` are filled in by `call_stage`; an adapter that
    detects the condition itself (e.g. the Agent SDK's own structured-output
    retries being exhausted) raises it with both left as None and may attach
    what the provider still reported for the failed attempt — `usage`
    (a `kg.llm.adapters.base.Usage`), `cost_estimate_usd`, `model_served` — so
    the ledger row for that attempt is not recorded as zero tokens — and the
    provider `events` (rate-limit notices) seen during the attempt, so
    `UsageLedger.events()` carries them even though no completion was returned.
    """

    def __init__(
        self,
        message: str,
        *,
        stage: str | None = None,
        attempts: int | None = None,
        usage: Any | None = None,
        cost_estimate_usd: float | None = None,
        model_served: str | None = None,
        events: tuple[dict[str, Any], ...] = (),
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.attempts = attempts
        self.usage = usage
        self.cost_estimate_usd = cost_estimate_usd
        self.model_served = model_served
        self.events = tuple(events)


class StageTruncated(LLMError):
    """The provider stopped on its output-token limit; the caller may split and retry.

    `stage` and `attempts` are filled in by `call_stage`, as for `StageOutputInvalid`.
    """

    def __init__(self, message: str, *, stage: str | None = None, attempts: int | None = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.attempts = attempts


class BudgetExceeded(LLMError):
    """Priced spend in the ledger passed `limits.soft_budget_usd` (checked before every call)."""

    def __init__(self, spent_usd: float, soft_budget_usd: float) -> None:
        super().__init__(f"soft budget exceeded: spent ${spent_usd:.4f} of ${soft_budget_usd:.2f}")
        self.spent_usd = spent_usd
        self.soft_budget_usd = soft_budget_usd


class RateLimited(LLMError):
    """The provider refused the call for rate-limit reasons (DESIGN §7.6).

    `action` is "wait" (reset is within `rate_limit_wait_minutes`; sleep
    `wait_seconds` then continue) or "defer" (stop model stages, defer the
    remaining files). Adapters never sleep themselves; the pipeline decides.
    `events` are the provider events (warnings and the rejection itself) seen
    during the failed attempt, recorded on its ledger row by `call_stage`.
    """

    def __init__(
        self,
        message: str,
        *,
        action: str,
        wait_seconds: float = 0.0,
        resets_at: int | None = None,
        rate_limit_type: str | None = None,
        utilization: float | None = None,
        events: tuple[dict[str, Any], ...] = (),
    ) -> None:
        super().__init__(message)
        self.action = action
        self.wait_seconds = wait_seconds
        self.resets_at = resets_at
        self.rate_limit_type = rate_limit_type
        self.utilization = utilization
        self.events = tuple(events)


class ProviderError(LLMError):
    """Transport, authentication or server failure that is not a rate limit."""
