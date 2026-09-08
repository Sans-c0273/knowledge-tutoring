"""WU2 — `kg.llm.call_stage`: the single provider boundary (DESIGN §7.1, §17 "Provider boundary";
PRD R21, R15, R19, R6).

Uses conftest's FakeAdapter (duck-typed `complete(system, messages, schema_json, model, max_tokens)`)
so the loop is tested provider-agnostically. Adapter-specific request shapes are in the
per-adapter WU2 files.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from kg.config import ConfigError
from kg.llm import BudgetExceeded, LLMError, StageOutputInvalid, StageResult, StageTruncated, call_stage
from kg.llm.adapters.base import ImageInput, Msg
from kg.llm.ledger import UsageLedger
from kg.llm.schema_profile import export
from kg.schemas import AtomizeOutput, DescribeOutput

from wu2_fakes import PNG_B64, SYSTEM, USER_TEXT, describe_payload, load_cfg, raw


@pytest.fixture
def cfg(make_project):
    return load_cfg(make_project)  # claude_subscription defaults, repair_retries=2


@pytest.fixture
def ledger():
    return UsageLedger()


@pytest.fixture
def good():
    return describe_payload(DescribeOutput)


# ---------------------------------------------------------------- happy path


def test_R21_call_stage_returns_validated_instance_and_metadata(cfg, ledger, fake_adapter, good):
    fake_adapter.canned.append(raw(good, model_requested="claude-sonnet-5", model_served="claude-sonnet-5-20260101"))
    res = call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=fake_adapter)
    assert isinstance(res, StageResult)
    assert isinstance(res.data, DescribeOutput)
    assert res.data.description == good["description"]
    assert res.attempts == 1
    assert res.provider == "fake"
    assert res.model_requested == "claude-sonnet-5"
    assert res.model_served == "claude-sonnet-5-20260101"
    assert res.usage.input_tokens == 100 and res.usage.output_tokens == 20


def test_R21_call_stage_parses_json_text_output(cfg, ledger, fake_adapter, good):
    fake_adapter.canned.append(raw(json.dumps(good)))
    res = call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=fake_adapter)
    assert isinstance(res.data, DescribeOutput)


def test_R21_call_stage_passes_stage_model_schema_and_max_tokens_to_adapter(cfg, ledger, fake_adapter, good):
    fake_adapter.canned.append(raw(good))
    call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=fake_adapter)
    call = fake_adapter.calls[0]
    assert call["system"] == SYSTEM
    assert call["model"] == cfg.llm.model_for("describe")
    assert call["max_tokens"] == cfg.llm.max_output_tokens
    assert call["schema_json"] == export(DescribeOutput)


def test_R21_call_stage_sends_one_user_msg_with_the_prompt_text(cfg, ledger, fake_adapter, good):
    fake_adapter.canned.append(raw(good))
    call_stage("atomize", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=fake_adapter)
    msgs = fake_adapter.calls[0]["messages"]
    assert len(msgs) == 1
    assert isinstance(msgs[0], Msg)
    assert msgs[0].role == "user"
    assert msgs[0].text == USER_TEXT
    assert list(msgs[0].images) == []


def test_R6_call_stage_forwards_images_on_the_user_msg(cfg, ledger, fake_adapter, good):
    img = ImageInput(media_type="image/png", data_b64=PNG_B64)
    fake_adapter.canned.append(raw(good))
    call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, images=[img], cfg=cfg, ledger=ledger, adapter=fake_adapter)
    msgs = fake_adapter.calls[0]["messages"]
    assert list(msgs[0].images) == [img]


def test_R21_call_stage_uses_per_stage_model_from_config(make_project, ledger, fake_adapter, good):
    cfg = load_cfg(make_project, **{"llm.claude_subscription.models.edges": "claude-opus-5"})
    fake_adapter.canned.append(raw(good))
    call_stage("edges", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=fake_adapter)
    assert fake_adapter.calls[0]["model"] == "claude-opus-5"


# ------------------------------------------------------------------- ledger


def test_R15_every_attempt_is_recorded_in_the_ledger(cfg, ledger, fake_adapter, good):
    fake_adapter.canned.append(raw(good, model_requested="claude-sonnet-5", model_served="claude-sonnet-5", cost_usd=None))
    call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=fake_adapter, prompt_version="describe@1")
    assert len(ledger.rows) == 1
    r = ledger.rows[0]
    assert r.stage == "describe"
    assert r.provider == "fake"
    assert r.model_requested == "claude-sonnet-5"
    assert r.model_served == "claude-sonnet-5"
    assert r.input_tokens == 100 and r.output_tokens == 20
    assert r.cost_usd is None
    assert r.attempt_no == 1
    assert r.outcome == "ok"
    assert isinstance(r.duration_s, float) and r.duration_s >= 0.0
    assert r.prompt_version == "describe@1"


def test_R15_ledger_row_records_system_prompt_sha(cfg, ledger, fake_adapter, good):
    fake_adapter.canned.append(raw(good))
    call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=fake_adapter)
    sha = ledger.rows[0].system_sha
    assert isinstance(sha, str) and len(sha) >= 8
    assert hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest().startswith(sha)


def test_R15_stage_result_carries_prompt_version(cfg, ledger, fake_adapter, good):
    fake_adapter.canned.append(raw(good))
    res = call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=fake_adapter, prompt_version="describe@1")
    assert res.prompt_version == "describe@1"


# --------------------------------------------------------------- repair loop


def test_R21_repair_loop_feeds_validation_error_back_and_succeeds_on_third_attempt(cfg, ledger, fake_adapter, good):
    fake_adapter.canned.extend(
        [
            raw("not json at all"),  # (1) unparseable
            raw({"title": "no description key"}),  # (2) schema-invalid
            raw(good),  # (3) valid
        ]
    )
    res = call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=fake_adapter)
    assert isinstance(res.data, DescribeOutput)
    assert res.attempts == 3
    assert len(fake_adapter.calls) == 3
    assert [r.attempt_no for r in ledger.rows] == [1, 2, 3]
    assert [r.outcome for r in ledger.rows] == ["invalid", "invalid", "ok"]

    # Second request carries the bad output as an assistant turn + a repair user turn (DESIGN §7.1 step 3).
    msgs2 = fake_adapter.calls[1]["messages"]
    assert [m.role for m in msgs2] == ["user", "assistant", "user"]
    assert msgs2[0].text == USER_TEXT
    assert msgs2[1].text == "not json at all"
    assert "failed validation" in msgs2[2].text.lower()

    msgs3 = fake_adapter.calls[2]["messages"]
    assert [m.role for m in msgs3] == ["user", "assistant", "user", "assistant", "user"]
    assert json.loads(msgs3[3].text) == {"title": "no description key"}
    assert "description" in msgs3[4].text  # the pydantic error summary names the missing field


def test_R21_repair_exhaustion_raises_stage_output_invalid_never_partial(cfg, ledger, fake_adapter):
    fake_adapter.canned.extend([raw("x"), raw("y"), raw("z")])
    with pytest.raises(StageOutputInvalid) as ei:
        call_stage("atomize", SYSTEM, USER_TEXT, AtomizeOutput, cfg=cfg, ledger=ledger, adapter=fake_adapter)
    assert len(fake_adapter.calls) == 3  # 1 + repair_retries(2)
    assert ei.value.stage == "atomize"
    assert ei.value.attempts == 3
    assert isinstance(ei.value, LLMError)
    assert [r.outcome for r in ledger.rows] == ["invalid", "invalid", "invalid"]


def test_R21_repair_retries_zero_fails_on_first_bad_output(make_project, ledger, fake_adapter):
    cfg = load_cfg(make_project, **{"llm.repair_retries": 0})
    fake_adapter.canned.extend([raw("bad"), raw("would-be-second")])
    with pytest.raises(StageOutputInvalid):
        call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=fake_adapter)
    assert len(fake_adapter.calls) == 1
    assert len(fake_adapter.canned) == 1  # second canned response untouched


def test_R21_extra_keys_are_schema_invalid_and_trigger_repair(cfg, ledger, fake_adapter, good):
    """WireModel is extra='forbid' (DESIGN §7.7); a stray key must not be silently accepted."""
    fake_adapter.canned.extend([raw({**good, "hallucinated": 1}), raw(good)])
    res = call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=fake_adapter)
    assert res.attempts == 2


def test_R21_adapter_max_repairs_caps_the_loop(cfg, ledger, good):
    """D20: the Claude-subscription adapter re-issues the whole query and is capped at 1 repair."""

    class CappedFake:
        provider = "capped"
        max_repairs = 1

        def __init__(self):
            self.calls = 0

        def complete(self, system, messages, schema_json, model, max_tokens):
            self.calls += 1
            return raw("never valid")

    fake = CappedFake()
    with pytest.raises(StageOutputInvalid):
        call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=fake)
    assert fake.calls == 2  # 1 + min(repair_retries=2, max_repairs=1)


def test_R21_json_array_at_top_level_is_invalid_not_a_crash(cfg, ledger, fake_adapter, good):
    fake_adapter.canned.extend([raw("[1, 2, 3]"), raw(good)])
    res = call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=fake_adapter)
    assert res.attempts == 2


# --------------------------------------------------------------- truncation


@pytest.mark.parametrize("stop", ["max_tokens", "length"])
def test_R21_truncated_output_raises_stage_truncated(cfg, ledger, fake_adapter, stop):
    fake_adapter.canned.append(raw('{"description": "cut off mid', stop_reason=stop))
    with pytest.raises(StageTruncated):
        call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=fake_adapter)
    assert len(fake_adapter.calls) == 1  # no repair attempt on truncation
    assert ledger.rows[0].outcome == "truncated"


# ------------------------------------------------------------------- budget


def test_R19_budget_exceeded_is_raised_before_the_next_model_call(make_project, fake_adapter, good):
    from kg.llm.ledger import UsageRow

    cfg = load_cfg(make_project, "openrouter", **{"limits.soft_budget_usd": 1.0})
    ledger = UsageLedger(soft_budget_usd=cfg.limits.soft_budget_usd)
    ledger.record(
        UsageRow(
            stage="atomize",
            provider="openrouter",
            model_requested="anthropic/claude-sonnet-5",
            model_served="anthropic/claude-sonnet-5",
            input_tokens=1,
            output_tokens=1,
            cost_usd=1.50,
            attempt_no=1,
            outcome="ok",
            duration_s=0.1,
        )
    )
    fake_adapter.canned.append(raw(good))
    with pytest.raises(BudgetExceeded):
        call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=fake_adapter)
    assert fake_adapter.calls == []  # the adapter was never invoked


def test_R19_budget_not_enforced_on_subscription_rows(make_project, fake_adapter, good):
    from kg.llm.ledger import UsageRow

    cfg = load_cfg(make_project, **{"limits.soft_budget_usd": 0.01})
    ledger = UsageLedger(soft_budget_usd=cfg.limits.soft_budget_usd)
    ledger.record(
        UsageRow(
            stage="atomize",
            provider="claude_subscription",
            model_requested="claude-sonnet-5",
            model_served="claude-sonnet-5",
            input_tokens=1,
            output_tokens=1,
            cost_usd=None,
            cost_estimate_usd=9.99,
            attempt_no=1,
            outcome="ok",
            duration_s=0.1,
        )
    )
    fake_adapter.canned.append(raw(good, cost_usd=None))
    res = call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=fake_adapter)
    assert isinstance(res.data, DescribeOutput)


# ------------------------------------------------------------ unhappy input


def test_R21_unknown_stage_is_refused_before_any_model_call(cfg, ledger, fake_adapter, good):
    fake_adapter.canned.append(raw(good))
    with pytest.raises((ConfigError, ValueError)):
        call_stage("summarise", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=fake_adapter)
    assert fake_adapter.calls == []


def test_R21_adapter_exception_propagates_and_is_recorded_as_error(cfg, ledger):
    class Exploding:
        provider = "boom"
        max_repairs = None

        def complete(self, *a, **k):
            raise RuntimeError("socket closed")

    with pytest.raises(RuntimeError):
        call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=Exploding())
    assert len(ledger.rows) == 1
    assert ledger.rows[0].outcome == "error"
    assert ledger.rows[0].input_tokens == 0 and ledger.rows[0].output_tokens == 0


def test_R21_exception_hierarchy_roots_at_llm_error():
    for exc in (StageOutputInvalid, StageTruncated, BudgetExceeded):
        assert issubclass(exc, LLMError)
    assert not issubclass(LLMError, ConfigError)
