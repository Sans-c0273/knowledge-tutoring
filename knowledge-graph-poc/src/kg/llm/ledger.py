"""Usage ledger: one row per attempt, aggregated per stage and in total (DESIGN §7.1 step 4, §14 item 6; D28).

Cost semantics per provider: OpenRouter rows carry the server-reported USD,
the Anthropic API adapter computes from the config price table, and the
subscription adapter leaves `cost_usd` None and puts the SDK's list-price
estimate in `cost_estimate_usd`. Unknown is never rendered as "$0.00".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from kg.llm.errors import BudgetExceeded

OUTCOMES: tuple[str, ...] = ("ok", "invalid", "truncated", "error")
SUBSCRIPTION_PROVIDER = "claude_subscription"


@dataclass(frozen=True)
class UsageRow:
    stage: str
    provider: str
    model_requested: str
    model_served: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    attempt_no: int
    outcome: str
    duration_s: float
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    cost_estimate_usd: float | None = None
    prompt_version: str | None = None
    system_sha: str | None = None
    #: Provider events seen during this attempt (rate-limit warnings etc., DESIGN §7.6, §14 item 7).
    events: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"UsageRow.outcome must be one of {', '.join(OUTCOMES)}; got {self.outcome!r}")
        if self.attempt_no < 1:
            raise ValueError("UsageRow.attempt_no starts at 1")
        for name in ("input_tokens", "output_tokens", "cached_tokens", "reasoning_tokens"):
            if getattr(self, name) < 0:
                raise ValueError(f"UsageRow.{name} must be >= 0")
        object.__setattr__(self, "events", tuple(dict(e) for e in self.events))


def _usd(value: float) -> str:
    return f"${value:.4f}"


def cost_display(provider: str, cost_usd: float | None, cost_estimate_usd: float | None = None) -> str:
    """Human-readable cost for a stage or a run (DESIGN §14 item 6, D28)."""
    if provider == SUBSCRIPTION_PROVIDER:
        text = "n/a (subscription)"
        if cost_estimate_usd is not None:
            text += f"; equivalent API list price (SDK estimate): {_usd(cost_estimate_usd)}"
        return text
    if cost_usd is None:
        return "unknown (no cost reported)"
    return _usd(cost_usd)


def _sum_optional(values: list[float | None]) -> float | None:
    priced = [v for v in values if v is not None]
    return sum(priced) if priced else None


def _aggregate(rows: list[UsageRow]) -> dict[str, Any]:
    providers = {r.provider for r in rows}
    cost_usd = _sum_optional([r.cost_usd for r in rows])
    cost_estimate_usd = _sum_optional([r.cost_estimate_usd for r in rows])
    if providers == {SUBSCRIPTION_PROVIDER}:
        display_provider = SUBSCRIPTION_PROVIDER
    elif len(providers) == 1:
        display_provider = next(iter(providers))
    else:
        display_provider = "mixed"
    last = rows[-1] if rows else None
    return {
        "provider": last.provider if last else None,
        "model_requested": last.model_requested if last else None,
        "model_served": last.model_served if last else None,
        "calls": sum(1 for r in rows if r.attempt_no == 1),
        "attempts": len(rows),
        "repair_attempts": sum(1 for r in rows if r.attempt_no > 1),
        "input_tokens": sum(r.input_tokens for r in rows),
        "output_tokens": sum(r.output_tokens for r in rows),
        "cached_tokens": sum(r.cached_tokens for r in rows),
        "reasoning_tokens": sum(r.reasoning_tokens for r in rows),
        "cost_usd": cost_usd,
        "cost_estimate_usd": cost_estimate_usd,
        "cost_display": cost_display(display_provider, cost_usd, cost_estimate_usd),
    }


class UsageLedger:
    """Append-only record of every model attempt in a run, with a soft budget."""

    def __init__(self, soft_budget_usd: float | None = None) -> None:
        self.soft_budget_usd = soft_budget_usd
        self._rows: list[UsageRow] = []

    def record(self, row: UsageRow) -> None:
        if not isinstance(row, UsageRow):
            raise TypeError(f"UsageLedger.record expects a UsageRow, got {type(row).__name__}")
        self._rows.append(row)

    @property
    def rows(self) -> tuple[UsageRow, ...]:
        return tuple(self._rows)

    @property
    def spent_usd(self) -> float | None:
        """Sum of priced rows only; None when nothing priced has been recorded."""
        return _sum_optional([r.cost_usd for r in self._rows])

    def check_budget(self) -> None:
        """Raise `BudgetExceeded` if priced spend is above the soft budget. Unpriced rows never count."""
        if self.soft_budget_usd is None:
            return
        spent = self.spent_usd
        if spent is not None and spent > self.soft_budget_usd:
            raise BudgetExceeded(spent, self.soft_budget_usd)

    def per_stage(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for stage in dict.fromkeys(r.stage for r in self._rows):
            out[stage] = _aggregate([r for r in self._rows if r.stage == stage])
        return out

    def events(self) -> list[dict[str, Any]]:
        """Every provider event in the run, in order, tagged with the stage and attempt it came from (§14 item 7)."""
        return [{**event, "stage": r.stage, "attempt_no": r.attempt_no} for r in self._rows for event in r.events]

    def total(self) -> dict[str, Any]:
        return {**_aggregate(list(self._rows)), "events": self.events()}

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view for the run report (DESIGN §14 items 6–7)."""
        return {
            "soft_budget_usd": self.soft_budget_usd,
            "spent_usd": self.spent_usd,
            "per_stage": self.per_stage(),
            "total": self.total(),
            "rows": [{**asdict(r), "events": list(r.events)} for r in self._rows],
        }


__all__ = ["OUTCOMES", "UsageLedger", "UsageRow", "cost_display"]
