"""WU2 — ClaudeSubscriptionAdapter over claude-agent-sdk (DESIGN §7.3, §7.6, §17 rows
"Repair-retry — Claude subscription path" and "Rate-limit handling"; D19, D20, D21, D22, D28;
PRD R21(1), R6, §9 sandbox).

`claude_agent_sdk.query` is INJECTED as `query_fn` (a FakeQuery). conftest's `no_real_llm`
poisons the real `claude_agent_sdk.query`; the last test proves the default wiring is
covered by that poison (zero-token guarantee).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kg.config import ConfigError
from kg.llm import LLMError, ProviderError, RateLimited, StageOutputInvalid, call_stage
from kg.llm.adapters.base import ImageInput, Msg, RawCompletion
from kg.llm.adapters.claude_subscription import ClaudeSubscriptionAdapter, RateLimitDecision, rate_limit_policy
from kg.llm.ledger import UsageLedger
from kg.llm.schema_profile import export
from kg.schemas import DescribeOutput

from wu2_fakes import PNG_B64, SYSTEM, USER_TEXT, FakeQuery, describe_payload, load_cfg, sdk_rate_limit_event, sdk_result

NOW = 1_800_000_000.0  # fixed epoch seconds; no datetime.now() anywhere in these tests


@pytest.fixture
def cfg(make_project):
    return load_cfg(make_project)  # provider claude_subscription, effort medium, wait 30 min


@pytest.fixture
def good():
    return describe_payload(DescribeOutput)


def make_adapter(cfg, scripts, *, now: float = NOW):
    fq = FakeQuery(scripts)
    adapter = ClaudeSubscriptionAdapter(cfg, query_fn=fq, clock=lambda: now)
    return adapter, fq


def one_user_msg(text: str = USER_TEXT, images=()):
    return [Msg(role="user", text=text, images=list(images))]


def agent_cwd_root(cfg) -> Path:
    return (cfg.project_root / cfg.paths.runs / ".agent-cwd").resolve()


# --------------------------------------------------------------- identity


def test_R21_adapter_identifies_its_provider(cfg):
    adapter, _ = make_adapter(cfg, [])
    assert adapter.provider == "claude_subscription"


def test_D20_adapter_caps_client_side_repairs_at_one(cfg):
    adapter, _ = make_adapter(cfg, [])
    assert adapter.max_repairs == 1


# ------------------------------------------------------- options (D21, §7.3)


def test_D21_options_isolate_the_agent_runtime(cfg, good):
    adapter, fq = make_adapter(cfg, [[sdk_result(structured_output=good)]])
    adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), "claude-sonnet-5", 8000)
    opts = fq.calls[0]["options"]
    assert opts.tools == []
    assert opts.allowed_tools == []
    assert opts.setting_sources == []
    assert opts.mcp_servers == {} or opts.mcp_servers is None
    # The adapter may BLANK other providers' credential variables for the CLI subprocess, but it
    # must never inject anything else (so both `{}` and `{"OPENAI_API_KEY": "", ...}` pass).
    blankable = {"OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"}
    assert set(opts.env) <= blankable, f"unexpected env keys injected: {set(opts.env) - blankable}"
    assert all(v == "" for v in opts.env.values()), f"env values must be blank: {opts.env}"
    assert opts.permission_mode is None  # nothing interactive; no tools to permit (DESIGN §7.3)
    assert opts.max_turns is not None and 1 <= opts.max_turns <= 3
    assert opts.continue_conversation is False and opts.resume is None


def test_D21_options_cwd_is_an_empty_dir_under_runs_agent_cwd(cfg, good):
    adapter, fq = make_adapter(cfg, [[sdk_result(structured_output=good)]])
    adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), "claude-sonnet-5", 8000)
    cwd = Path(fq.calls[0]["options"].cwd).resolve()
    root = agent_cwd_root(cfg)
    assert cwd == root or root in cwd.parents, f"{cwd} not under {root}"
    assert cwd.is_dir()
    assert not any(cwd.iterdir()), "agent cwd must be empty (no CLAUDE.md, no settings)"


def test_D21_never_bare_mode(cfg, good):
    """Headless docs: --bare 'doesn't use your subscription login'."""
    adapter, fq = make_adapter(cfg, [[sdk_result(structured_output=good)]])
    adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), "claude-sonnet-5", 8000)
    opts = fq.calls[0]["options"]
    assert "bare" not in {k.lstrip("-") for k in opts.extra_args}
    assert getattr(opts, "bare", False) in (False, None)


def test_R21_options_carry_system_prompt_model_effort_and_native_json_schema(cfg, good):
    adapter, fq = make_adapter(cfg, [[sdk_result(structured_output=good)]])
    adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), "claude-sonnet-5", 8000)
    opts = fq.calls[0]["options"]
    assert opts.system_prompt == SYSTEM
    assert opts.model == "claude-sonnet-5"
    assert opts.effort == "medium"
    assert opts.output_format == {"type": "json_schema", "schema": export(DescribeOutput)}


def test_R21_effort_follows_config(make_project, good):
    cfg = load_cfg(make_project, **{"llm.claude_subscription.effort": "high"})
    adapter, fq = make_adapter(cfg, [[sdk_result(structured_output=good)]])
    adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), "claude-sonnet-5", 8000)
    assert fq.calls[0]["options"].effort == "high"


# ------------------------------------------- streaming-input prompt (§7.3)


def test_R21_prompt_is_a_one_message_async_generator_not_a_string(cfg, good):
    adapter, fq = make_adapter(cfg, [[sdk_result(structured_output=good)]])
    adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), "claude-sonnet-5", 8000)
    call = fq.calls[0]
    assert call["prompt_is_str"] is False
    assert hasattr(call["prompt_raw"], "__aiter__")
    assert len(call["prompt_messages"]) == 1
    msg = call["prompt_messages"][0]
    assert msg["type"] == "user"
    assert msg["message"]["role"] == "user"
    content = msg["message"]["content"]
    assert isinstance(content, list)
    assert content[-1] == {"type": "text", "text": USER_TEXT}


def test_R21_text_only_prompt_has_no_image_block(cfg, good):
    adapter, fq = make_adapter(cfg, [[sdk_result(structured_output=good)]])
    adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), "claude-sonnet-5", 8000)
    content = fq.calls[0]["prompt_messages"][0]["message"]["content"]
    assert [b["type"] for b in content] == ["text"]


def test_R6_image_is_embedded_as_base64_content_block_before_the_text(cfg, good):
    img = ImageInput(media_type="image/png", data_b64=PNG_B64)
    adapter, fq = make_adapter(cfg, [[sdk_result(structured_output=good)]])
    adapter.complete(SYSTEM, one_user_msg(images=[img]), export(DescribeOutput), "claude-sonnet-5", 8000)
    content = fq.calls[0]["prompt_messages"][0]["message"]["content"]
    assert [b["type"] for b in content] == ["image", "text"]
    assert content[0] == {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64}}


def test_R6_two_images_keep_order_and_media_types(cfg, good):
    a = ImageInput(media_type="image/png", data_b64=PNG_B64)
    b = ImageInput(media_type="image/jpeg", data_b64=PNG_B64)
    adapter, fq = make_adapter(cfg, [[sdk_result(structured_output=good)]])
    adapter.complete(SYSTEM, one_user_msg(images=[a, b]), export(DescribeOutput), "claude-sonnet-5", 8000)
    content = fq.calls[0]["prompt_messages"][0]["message"]["content"]
    assert [b_["type"] for b_ in content] == ["image", "image", "text"]
    assert [b_["source"]["media_type"] for b_ in content[:2]] == ["image/png", "image/jpeg"]


# -------------------------------------------------------- result handling


def test_R21_structured_output_dict_is_returned_with_usage_and_no_price(cfg, good):
    result = sdk_result(
        structured_output=good,
        usage={"input_tokens": 1500, "output_tokens": 300, "cache_read_input_tokens": 400, "cache_creation_input_tokens": 25},
        total_cost_usd=0.0123,
        model_usage={"claude-sonnet-5-20260101": {"inputTokens": 1500, "outputTokens": 300, "cacheReadInputTokens": 400, "cacheCreationInputTokens": 25, "webSearchRequests": 0, "costUSD": 0.0123}},
    )
    adapter, _ = make_adapter(cfg, [[result]])
    rc = adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), "claude-sonnet-5", 8000)
    assert isinstance(rc, RawCompletion)
    assert rc.text_or_obj == good
    assert rc.usage.input_tokens == 1500
    assert rc.usage.output_tokens == 300
    assert rc.usage.cached_tokens == 400
    assert rc.cost_usd is None  # subscription — n/a (D28)
    assert rc.provider_meta.get("cost_estimate_usd") == pytest.approx(0.0123)
    assert rc.model_requested == "claude-sonnet-5"
    assert rc.model_served == "claude-sonnet-5-20260101"


def test_R21_model_served_is_none_when_sdk_reports_no_model_usage(cfg, good):
    adapter, _ = make_adapter(cfg, [[sdk_result(structured_output=good, model_usage=None)]])
    rc = adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), "claude-sonnet-5", 8000)
    assert rc.model_served is None


def test_R21_missing_usage_yields_zero_tokens_not_a_crash(cfg, good):
    adapter, _ = make_adapter(cfg, [[sdk_result(structured_output=good, usage=None, total_cost_usd=None)]])
    rc = adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), "claude-sonnet-5", 8000)
    assert (rc.usage.input_tokens, rc.usage.output_tokens) == (0, 0)
    assert rc.cost_usd is None


def test_R21_max_structured_output_retries_result_raises_stage_output_invalid(cfg):
    bad = sdk_result(subtype="error_max_structured_output_retries", is_error=True, structured_output=None)
    adapter, _ = make_adapter(cfg, [[bad]])
    with pytest.raises(StageOutputInvalid):
        adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), "claude-sonnet-5", 8000)


def test_R21_result_error_from_sdk_for_structured_output_exhaustion_is_stage_output_invalid(cfg):
    from claude_agent_sdk import ResultError

    err = ResultError("max structured output retries", data={"subtype": "error_max_structured_output_retries"}, exit_code=1)
    adapter, _ = make_adapter(cfg, [[err]])
    with pytest.raises(StageOutputInvalid):
        adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), "claude-sonnet-5", 8000)


def test_R21_success_without_structured_output_is_stage_output_invalid(cfg):
    adapter, _ = make_adapter(cfg, [[sdk_result(structured_output=None, result="plain prose, no JSON")]])
    with pytest.raises(StageOutputInvalid):
        adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), "claude-sonnet-5", 8000)


def test_R21_stream_that_ends_without_a_result_message_is_a_provider_error(cfg):
    adapter, _ = make_adapter(cfg, [[]])
    with pytest.raises(LLMError):
        adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), "claude-sonnet-5", 8000)


def test_R21_auth_or_api_error_result_is_a_provider_error_not_invalid_output(cfg):
    failed = sdk_result(subtype="error_during_execution", is_error=True, api_error_status=401, errors=["Not logged in"])
    adapter, _ = make_adapter(cfg, [[failed]])
    with pytest.raises(ProviderError) as ei:
        adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), "claude-sonnet-5", 8000)
    assert not isinstance(ei.value, StageOutputInvalid)
    assert "401" in str(ei.value) or "logged in" in str(ei.value).lower()


# --------------------------------------------- through call_stage (D20 cap)


def test_D20_pydantic_failure_reissues_whole_query_once_then_fails(cfg):
    """SDK-validated JSON can still fail Pydantic (e.g. extra key); one re-query, then StageOutputInvalid."""
    bad_obj = {"description": "ok", "hallucinated": True}
    adapter, fq = make_adapter(cfg, [[sdk_result(structured_output=bad_obj)], [sdk_result(structured_output=bad_obj)], [sdk_result(structured_output=bad_obj)]])
    ledger = UsageLedger()
    with pytest.raises(StageOutputInvalid):
        call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=adapter)
    assert len(fq.calls) == 2  # repair_retries=2 in config, but this adapter caps at 1 (D20)
    assert len(ledger.rows) == 2
    # Each re-query is still a single-message streaming prompt (no continuation).
    for call in fq.calls:
        assert len(call["prompt_messages"]) == 1
        assert call["prompt_messages"][0]["message"]["content"][-1]["type"] == "text"


def test_D20_pydantic_failure_then_good_object_succeeds_with_two_attempts(cfg, good):
    adapter, fq = make_adapter(cfg, [[sdk_result(structured_output={"description": "x", "extra": 1})], [sdk_result(structured_output=good)]])
    ledger = UsageLedger()
    res = call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=adapter)
    assert res.attempts == 2 and len(fq.calls) == 2
    assert res.provider == "claude_subscription"
    assert [r.cost_usd for r in ledger.rows] == [None, None]


# ------------------------------------------------------ rate limits (§7.6)


def _info(status, resets_at, rl_type="five_hour", util=1.0):
    return sdk_rate_limit_event(status, resets_at, rl_type, util).rate_limit_info


def test_R21_policy_rejected_and_reset_within_wait_window_says_wait():
    d = rate_limit_policy(_info("rejected", int(NOW) + 600), now=NOW, wait_minutes=30)
    assert isinstance(d, RateLimitDecision)
    assert d.action == "wait"
    assert d.wait_seconds == pytest.approx(600, abs=1)
    assert d.resets_at == int(NOW) + 600
    assert d.rate_limit_type == "five_hour"
    assert d.utilization == pytest.approx(1.0)


def test_R21_policy_rejected_and_reset_beyond_wait_window_says_defer():
    d = rate_limit_policy(_info("rejected", int(NOW) + 7200), now=NOW, wait_minutes=30)
    assert d.action == "defer"
    assert d.wait_seconds == 0


def test_R21_policy_boundary_exactly_at_wait_window_waits():
    d = rate_limit_policy(_info("rejected", int(NOW) + 30 * 60), now=NOW, wait_minutes=30)
    assert d.action == "wait"


def test_R21_policy_rejected_with_unknown_reset_time_defers():
    d = rate_limit_policy(_info("rejected", None), now=NOW, wait_minutes=30)
    assert d.action == "defer"


def test_R21_policy_rejected_with_reset_in_the_past_waits_zero():
    d = rate_limit_policy(_info("rejected", int(NOW) - 5), now=NOW, wait_minutes=30)
    assert d.action == "wait"
    assert d.wait_seconds == 0


def test_R21_policy_wait_minutes_zero_never_waits():
    d = rate_limit_policy(_info("rejected", int(NOW) + 1), now=NOW, wait_minutes=0)
    assert d.action == "defer"


def test_R21_policy_allowed_warning_is_warn_and_allowed_is_proceed():
    assert rate_limit_policy(_info("allowed_warning", int(NOW) + 900, util=0.9), now=NOW, wait_minutes=30).action == "warn"
    assert rate_limit_policy(_info("allowed", None, util=0.2), now=NOW, wait_minutes=30).action == "proceed"


def test_R21_rejected_event_with_near_reset_raises_rate_limited_wait(cfg):
    """Adapter never sleeps itself; it signals the pipeline (DESIGN §20 puts the wait in pipeline.py)."""
    ev = sdk_rate_limit_event("rejected", int(NOW) + 600, "five_hour", 1.0)
    failed = sdk_result(subtype="success", is_error=True, api_error_status=429)
    adapter, fq = make_adapter(cfg, [[ev, failed]])
    with pytest.raises(RateLimited) as ei:
        adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), "claude-sonnet-5", 8000)
    e = ei.value
    assert e.action == "wait"
    assert e.wait_seconds == pytest.approx(600, abs=1)
    assert e.resets_at == int(NOW) + 600
    assert e.rate_limit_type == "five_hour"
    assert e.utilization == pytest.approx(1.0)
    assert len(fq.calls) == 1  # no silent re-query inside the adapter


def test_R21_rejected_event_with_far_reset_raises_rate_limited_defer(cfg):
    ev = sdk_rate_limit_event("rejected", int(NOW) + 7200, "seven_day", 1.0)
    failed = sdk_result(subtype="success", is_error=True, api_error_status=429)
    adapter, _ = make_adapter(cfg, [[ev, failed]])
    with pytest.raises(RateLimited) as ei:
        adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), "claude-sonnet-5", 8000)
    assert ei.value.action == "defer"
    assert ei.value.rate_limit_type == "seven_day"


def test_R21_wait_window_follows_config(make_project):
    cfg = load_cfg(make_project, **{"llm.claude_subscription.rate_limit_wait_minutes": 5})
    ev = sdk_rate_limit_event("rejected", int(NOW) + 600)
    failed = sdk_result(subtype="success", is_error=True, api_error_status=429)
    adapter, _ = make_adapter(cfg, [[ev, failed]])
    with pytest.raises(RateLimited) as ei:
        adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), "claude-sonnet-5", 8000)
    assert ei.value.action == "defer"  # 10 min reset > 5 min window


def test_R21_http_429_result_without_event_still_signals_rate_limited(cfg):
    failed = sdk_result(subtype="success", is_error=True, api_error_status=429)
    adapter, _ = make_adapter(cfg, [[failed]])
    with pytest.raises(RateLimited) as ei:
        adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), "claude-sonnet-5", 8000)
    assert ei.value.action == "defer"  # no resets_at known → defer
    assert ei.value.resets_at is None


def test_R21_allowed_warning_is_recorded_as_an_event_and_call_succeeds(cfg, good):
    ev = sdk_rate_limit_event("allowed_warning", int(NOW) + 900, "five_hour", 0.92)
    adapter, _ = make_adapter(cfg, [[ev, sdk_result(structured_output=good)]])
    rc = adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), "claude-sonnet-5", 8000)
    assert rc.text_or_obj == good
    assert len(rc.events) == 1
    ev_dict = rc.events[0]
    assert ev_dict["status"] == "allowed_warning"
    assert ev_dict["rate_limit_type"] == "five_hour"
    assert ev_dict["utilization"] == pytest.approx(0.92)
    assert ev_dict["resets_at"] == int(NOW) + 900
    json.dumps(rc.events)  # report-serialisable


def test_R21_rate_limited_is_an_llm_error_and_not_invalid_output():
    assert issubclass(RateLimited, LLMError)
    assert not issubclass(RateLimited, StageOutputInvalid)


# ------------------------------------------------------------------- D22


def test_D22_adapter_refuses_to_construct_when_anthropic_api_key_is_in_environment(cfg, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-TESTONLY-000")
    with pytest.raises(ConfigError) as ei:
        ClaudeSubscriptionAdapter(cfg, query_fn=FakeQuery([]), clock=lambda: NOW)
    assert "ANTHROPIC_API_KEY" in str(ei.value)
    assert "sk-ant-TESTONLY-000" not in str(ei.value)


def test_D22_adapter_refuses_a_config_for_another_provider(make_project):
    cfg = load_cfg(make_project, "openrouter")
    with pytest.raises(ConfigError):
        ClaudeSubscriptionAdapter(cfg, query_fn=FakeQuery([]), clock=lambda: NOW)


# ----------------------------------------------------- zero-token guarantee


def test_R20_default_wiring_is_the_real_sdk_query_and_is_covered_by_the_selftest_poison(cfg):
    """No query_fn injected → the adapter must reach `claude_agent_sdk.query`, which conftest poisons.

    This proves the guard actually protects this adapter (the lookup must happen at call time).
    """
    adapter = ClaudeSubscriptionAdapter(cfg, clock=lambda: NOW)
    with pytest.raises(AssertionError, match="real LLM provider invoked"):
        adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), "claude-sonnet-5", 8000)
