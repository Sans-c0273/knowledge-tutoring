"""WU2 — UsageLedger (DESIGN §7.1 step 4, §7.6, §14 item 6, §19; D28; PRD R15, R19, R21).

Rows per attempt; aggregation per stage and total; cost semantics per provider
(subscription -> "n/a (subscription)", never a price of $0.00); soft budget raises
BudgetExceeded; serialisable to the report shape.
"""

from __future__ import annotations

import json
import re

import pytest

from kg.llm import BudgetExceeded
from kg.llm.ledger import UsageLedger, UsageRow, cost_display


def row(**over):
    kw = {
        "stage": "atomize",
        "provider": "openrouter",
        "model_requested": "anthropic/claude-sonnet-5",
        "model_served": "anthropic/claude-sonnet-5",
        "input_tokens": 1000,
        "output_tokens": 200,
        "cost_usd": 0.01,
        "attempt_no": 1,
        "outcome": "ok",
        "duration_s": 1.5,
    }
    kw.update(over)
    return UsageRow(**kw)


# ------------------------------------------------------------------- rows


def test_R15_record_appends_rows_in_order():
    ledger = UsageLedger()
    r1, r2 = row(), row(stage="edges")
    ledger.record(r1)
    ledger.record(r2)
    assert list(ledger.rows) == [r1, r2]


def test_R15_row_carries_every_report_field():
    r = row(cached_tokens=50, reasoning_tokens=7, prompt_version="atomize@1", system_sha="abc123def456", cost_estimate_usd=None)
    for name in (
        "stage",
        "provider",
        "model_requested",
        "model_served",
        "input_tokens",
        "output_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "cost_usd",
        "cost_estimate_usd",
        "attempt_no",
        "outcome",
        "duration_s",
        "prompt_version",
        "system_sha",
    ):
        assert hasattr(r, name), name
    assert r.cached_tokens == 50 and r.reasoning_tokens == 7


def test_R15_row_optional_fields_default_sensibly():
    r = row()
    assert r.cached_tokens == 0
    assert r.reasoning_tokens == 0
    assert r.cost_estimate_usd is None
    assert r.prompt_version is None
    assert r.system_sha is None


@pytest.mark.parametrize("bad", ["done", "", "OK"])
def test_R15_row_outcome_is_a_closed_vocabulary(bad):
    with pytest.raises(ValueError):
        row(outcome=bad)


def test_R15_row_attempt_no_starts_at_one():
    with pytest.raises(ValueError):
        row(attempt_no=0)


# -------------------------------------------------------------- aggregation


def test_R15_per_stage_aggregates_tokens_calls_attempts_and_cost():
    ledger = UsageLedger()
    ledger.record(row(stage="atomize", attempt_no=1, outcome="invalid", input_tokens=1000, output_tokens=100, cost_usd=0.010))
    ledger.record(row(stage="atomize", attempt_no=2, outcome="ok", input_tokens=1200, output_tokens=150, cost_usd=0.012))
    ledger.record(row(stage="atomize", attempt_no=1, outcome="ok", input_tokens=900, output_tokens=90, cost_usd=0.009, cached_tokens=300))
    ledger.record(row(stage="edges", attempt_no=1, outcome="ok", input_tokens=500, output_tokens=50, cost_usd=0.005, reasoning_tokens=20))

    per = ledger.per_stage()
    a = per["atomize"]
    assert a["calls"] == 2  # two logical call_stage invocations (attempt_no == 1 rows)
    assert a["attempts"] == 3  # every row
    assert a["repair_attempts"] == 1
    assert a["input_tokens"] == 3100
    assert a["output_tokens"] == 340
    assert a["cached_tokens"] == 300
    assert a["reasoning_tokens"] == 0
    assert a["cost_usd"] == pytest.approx(0.031)

    e = per["edges"]
    assert (e["calls"], e["attempts"], e["repair_attempts"]) == (1, 1, 0)
    assert e["reasoning_tokens"] == 20
    assert e["cost_usd"] == pytest.approx(0.005)


def test_R15_total_sums_across_stages():
    ledger = UsageLedger()
    ledger.record(row(stage="atomize", input_tokens=1000, output_tokens=100, cost_usd=0.01))
    ledger.record(row(stage="edges", input_tokens=500, output_tokens=50, cost_usd=0.005))
    t = ledger.total()
    assert t["calls"] == 2
    assert t["attempts"] == 2
    assert t["input_tokens"] == 1500
    assert t["output_tokens"] == 150
    assert t["cost_usd"] == pytest.approx(0.015)


def test_R15_empty_ledger_aggregates_to_zero_and_unknown_cost():
    ledger = UsageLedger()
    assert ledger.per_stage() == {}
    t = ledger.total()
    assert t["calls"] == 0 and t["attempts"] == 0 and t["input_tokens"] == 0 and t["output_tokens"] == 0
    assert t["cost_usd"] is None
    assert ledger.spent_usd is None


def test_R15_per_stage_records_provider_and_models_for_the_report_table():
    """DESIGN §14 item 1: per-stage model_requested / model_served."""
    ledger = UsageLedger()
    ledger.record(row(stage="describe", provider="openrouter", model_requested="google/gemini-2.5-flash", model_served="google/gemini-2.5-flash"))
    d = ledger.to_dict()["per_stage"]["describe"]
    assert d["provider"] == "openrouter"
    assert d["model_requested"] == "google/gemini-2.5-flash"
    assert d["model_served"] == "google/gemini-2.5-flash"


# ---------------------------------------------------------- cost semantics


def test_D28_subscription_rows_have_no_price_and_total_cost_is_none():
    ledger = UsageLedger()
    ledger.record(row(provider="claude_subscription", model_requested="claude-sonnet-5", cost_usd=None, cost_estimate_usd=0.31))
    ledger.record(row(provider="claude_subscription", model_requested="claude-sonnet-5", cost_usd=None, cost_estimate_usd=0.20))
    t = ledger.total()
    assert t["cost_usd"] is None
    assert t["cost_estimate_usd"] == pytest.approx(0.51)
    assert ledger.spent_usd is None


def test_D28_cost_display_subscription_reads_n_a_never_zero_dollars():
    s = cost_display("claude_subscription", None)
    assert "n/a (subscription)" in s
    assert "$0.00" not in s


def test_D28_cost_display_subscription_appends_sdk_list_price_estimate():
    s = cost_display("claude_subscription", None, cost_estimate_usd=0.3149)
    assert "n/a (subscription)" in s
    assert "equivalent API list price (SDK estimate)" in s
    m = re.search(r"\$([0-9]+\.[0-9]+)", s)
    assert m and float(m.group(1)) == pytest.approx(0.31, abs=0.005)


def test_R15_cost_display_openrouter_shows_server_reported_dollars():
    s = cost_display("openrouter", 0.3)
    m = re.search(r"\$([0-9]+\.[0-9]+)", s)
    assert m, s
    assert float(m.group(1)) == pytest.approx(0.30, abs=0.005)
    assert "n/a" not in s


def test_R15_cost_display_anthropic_api_shows_computed_dollars():
    s = cost_display("anthropic_api", 0.0073)
    m = re.search(r"\$([0-9]+\.[0-9]+)", s)
    assert m, s
    assert float(m.group(1)) == pytest.approx(0.0073, abs=0.005)


def test_R15_cost_display_unknown_cost_on_priced_provider_is_not_zero_dollars():
    s = cost_display("openrouter", None)
    assert "$0.00" not in s
    assert s.strip() != ""


def test_R15_to_dict_total_and_per_stage_include_cost_display():
    ledger = UsageLedger()
    ledger.record(row(provider="claude_subscription", model_requested="claude-sonnet-5", cost_usd=None, cost_estimate_usd=0.31))
    d = ledger.to_dict()
    assert "n/a (subscription)" in d["total"]["cost_display"]
    assert "n/a (subscription)" in d["per_stage"]["atomize"]["cost_display"]


# ------------------------------------------------------------- soft budget


def test_R19_check_budget_raises_when_spent_exceeds_soft_budget():
    ledger = UsageLedger(soft_budget_usd=1.00)
    ledger.record(row(cost_usd=0.60))
    ledger.record(row(cost_usd=0.50))
    assert ledger.spent_usd == pytest.approx(1.10)
    with pytest.raises(BudgetExceeded) as ei:
        ledger.check_budget()
    assert ei.value.spent_usd == pytest.approx(1.10)
    assert ei.value.soft_budget_usd == pytest.approx(1.00)


def test_R19_check_budget_passes_at_or_below_budget():
    ledger = UsageLedger(soft_budget_usd=1.00)
    ledger.record(row(cost_usd=1.00))
    ledger.check_budget()  # exactly at budget is not "exceeded"


def test_R19_check_budget_ignores_subscription_rows_with_no_price():
    """DESIGN §3.2: soft_budget_usd 'ignored on subscription'."""
    ledger = UsageLedger(soft_budget_usd=0.01)
    for _ in range(50):
        ledger.record(row(provider="claude_subscription", cost_usd=None, cost_estimate_usd=5.0))
    ledger.check_budget()


def test_R19_check_budget_with_no_budget_configured_never_raises():
    ledger = UsageLedger()
    ledger.record(row(cost_usd=999.0))
    ledger.check_budget()


def test_R19_mixed_priced_and_unpriced_rows_sum_only_the_priced_ones():
    ledger = UsageLedger(soft_budget_usd=1.00)
    ledger.record(row(cost_usd=0.40))
    ledger.record(row(cost_usd=None))
    assert ledger.spent_usd == pytest.approx(0.40)
    ledger.check_budget()


# ------------------------------------------------------------ serialising


def test_R15_to_dict_is_json_serialisable_and_shaped_for_the_report():
    ledger = UsageLedger(soft_budget_usd=2.0)
    ledger.record(row(stage="describe", input_tokens=10, output_tokens=5, cost_usd=0.001))
    ledger.record(row(stage="atomize", attempt_no=1, outcome="invalid"))
    ledger.record(row(stage="atomize", attempt_no=2, outcome="ok"))
    d = ledger.to_dict()
    json.dumps(d)  # must not raise
    assert set(d) >= {"per_stage", "total"}
    for stage_block in d["per_stage"].values():
        assert {"calls", "attempts", "repair_attempts", "input_tokens", "output_tokens", "cached_tokens", "reasoning_tokens", "cost_usd", "cost_display"} <= set(stage_block)
    assert d["per_stage"]["atomize"]["repair_attempts"] == 1
    assert d["total"]["attempts"] == 3


def test_R18_to_dict_never_contains_a_credential_looking_value():
    ledger = UsageLedger()
    ledger.record(row())
    text = json.dumps(ledger.to_dict())
    assert "sk-or-" not in text and "sk-ant-" not in text
