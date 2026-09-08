"""WU2 — AnthropicApiAdapter, optional provider 3 (DESIGN §7.5, §17 "Config guards"; D4, D26;
PRD R21 fallback route, R6, R15).

The `anthropic` package is an optional extra (`kg[anthropic-api]`) and is NOT installed in the
default environment. The module must therefore import lazily: `kg.llm.adapters.anthropic_api`
is importable without `anthropic`, constructing without an injected client raises a clear
ImportError naming the extra, and an injected fake client exercises the request/response path.
"""

from __future__ import annotations

import json
import sys

import pytest

from kg.config import ConfigError
from kg.llm import ProviderError, StageOutputInvalid, StageTruncated, call_stage
from kg.llm.adapters.anthropic_api import AnthropicApiAdapter
from kg.llm.adapters.base import ImageInput, Msg, RawCompletion
from kg.llm.ledger import UsageLedger
from kg.llm.schema_profile import export
from kg.schemas import DescribeOutput

from wu2_fakes import FAKE_ANTHROPIC_KEY, PNG_B64, SYSTEM, USER_TEXT, FakeAnthropicClient, anthropic_response, describe_payload, load_cfg

MODEL = "claude-sonnet-5"


@pytest.fixture
def cfg(make_project):
    return load_cfg(make_project, "anthropic_api")


@pytest.fixture
def good():
    return describe_payload(DescribeOutput)


@pytest.fixture
def no_anthropic(monkeypatch):
    """Deterministically simulate the extra being absent, whether or not it is installed."""
    monkeypatch.setitem(sys.modules, "anthropic", None)


def make_adapter(cfg, scripts):
    client = FakeAnthropicClient(scripts)
    return AnthropicApiAdapter(cfg, client=client), client


def one_user_msg(text: str = USER_TEXT, images=()):
    return [Msg(role="user", text=text, images=list(images))]


def _output_schema(kw: dict) -> dict | None:
    """The JSON schema the adapter asked for, whichever GA spelling it used (DESIGN §7.5 / D4)."""
    of = kw.get("output_format")
    if isinstance(of, dict):
        return of.get("schema", of)
    oc = kw.get("output_config")
    if isinstance(oc, dict) and isinstance(oc.get("format"), dict):
        return oc["format"].get("schema")
    return None


# ---------------------------------------------------------- optional extra


def test_D26_constructing_without_the_extra_raises_import_error_naming_it(cfg, no_anthropic):
    with pytest.raises(ImportError) as ei:
        AnthropicApiAdapter(cfg)
    msg = str(ei.value)
    assert "anthropic-api" in msg
    assert FAKE_ANTHROPIC_KEY not in msg


def test_D26_module_is_importable_without_the_extra(no_anthropic):
    import importlib

    mod = importlib.import_module("kg.llm.adapters.anthropic_api")
    assert hasattr(mod, "AnthropicApiAdapter")


def test_D26_injected_client_bypasses_the_import(cfg, no_anthropic, good):
    adapter, _ = make_adapter(cfg, [anthropic_response(good)])
    rc = adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 8000)
    assert isinstance(rc, RawCompletion)


# --------------------------------------------------------------- identity


def test_R21_adapter_identifies_its_provider(cfg):
    adapter, _ = make_adapter(cfg, [])
    assert adapter.provider == "anthropic_api"


def test_R21_adapter_refuses_a_config_for_another_provider(make_project):
    cfg = load_cfg(make_project, "openrouter")
    with pytest.raises(ConfigError):
        AnthropicApiAdapter(cfg, client=FakeAnthropicClient([]))


def test_R18_api_key_never_appears_in_repr(cfg):
    adapter, _ = make_adapter(cfg, [])
    assert FAKE_ANTHROPIC_KEY not in repr(adapter)


# ---------------------------------------------------------- request shape


def test_D4_request_uses_messages_parse_with_json_schema_output(cfg, good):
    adapter, client = make_adapter(cfg, [anthropic_response(good)])
    adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 4321)
    kw = client.calls[0]
    assert kw["method"] in ("parse", "create")
    assert kw["model"] == MODEL
    assert kw["max_tokens"] == 4321
    assert _output_schema(kw) == export(DescribeOutput)
    assert "betas" not in kw and "extra_headers" not in kw  # GA surface, no beta header (D4)


def test_R21_system_prompt_is_a_cached_text_block(cfg, good):
    adapter, client = make_adapter(cfg, [anthropic_response(good)])
    adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 8000)
    system = client.calls[0]["system"]
    assert system == [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}]


def test_R21_user_message_carries_the_prompt_text(cfg, good):
    adapter, client = make_adapter(cfg, [anthropic_response(good)])
    adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 8000)
    msgs = client.calls[0]["messages"]
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    content = msgs[0]["content"]
    if isinstance(content, list):
        assert content[-1] == {"type": "text", "text": USER_TEXT}
    else:
        assert content == USER_TEXT


def test_R6_image_is_a_base64_image_block_before_the_text(cfg, good):
    img = ImageInput(media_type="image/png", data_b64=PNG_B64)
    adapter, client = make_adapter(cfg, [anthropic_response(good)])
    adapter.complete(SYSTEM, one_user_msg(images=[img]), export(DescribeOutput), MODEL, 8000)
    content = client.calls[0]["messages"][0]["content"]
    assert [b["type"] for b in content] == ["image", "text"]
    assert content[0] == {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64}}


def test_R21_repair_turns_are_rendered_as_alternating_messages(cfg, good):
    adapter, client = make_adapter(cfg, [anthropic_response(good)])
    msgs = [Msg(role="user", text=USER_TEXT), Msg(role="assistant", text="garbage"), Msg(role="user", text="The JSON failed validation")]
    adapter.complete(SYSTEM, msgs, export(DescribeOutput), MODEL, 8000)
    wire = client.calls[0]["messages"]
    assert [m["role"] for m in wire] == ["user", "assistant", "user"]


# ------------------------------------------------------- response parsing


def test_R21_parsed_output_is_returned_and_usage_captured(cfg, good):
    adapter, _ = make_adapter(cfg, [anthropic_response(good, model="claude-sonnet-5-20260101", input_tokens=1000, output_tokens=500, cache_read=200, cache_write=100)])
    rc = adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 8000)
    obj = rc.text_or_obj if isinstance(rc.text_or_obj, dict) else json.loads(rc.text_or_obj)
    assert obj == good
    assert rc.usage.input_tokens == 1000
    assert rc.usage.output_tokens == 500
    assert rc.usage.cached_tokens == 200
    assert rc.model_requested == MODEL
    assert rc.model_served == "claude-sonnet-5-20260101"
    assert rc.stop_reason == "end_turn"


def test_R15_cost_is_computed_from_the_config_price_table(cfg, good):
    """claude-sonnet-5: $2 in / $10 out per MTok; cache read ×0.10, cache write ×1.25 (DESIGN §3.2)."""
    adapter, _ = make_adapter(cfg, [anthropic_response(good, input_tokens=1000, output_tokens=500, cache_read=200, cache_write=100)])
    rc = adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 8000)
    expected = (1000 * 2 + 500 * 10 + 200 * 2 * 0.10 + 100 * 2 * 1.25) / 1_000_000
    assert rc.cost_usd == pytest.approx(expected, rel=1e-6)  # 0.00729


def test_R15_cost_for_unpriced_model_is_none_not_zero(cfg, good):
    adapter, _ = make_adapter(cfg, [anthropic_response(good, model="claude-future-9")])
    rc = adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), "claude-future-9", 8000)
    assert rc.cost_usd is None


def test_R15_cost_with_no_cache_tokens_is_plain_in_out(cfg, good):
    adapter, _ = make_adapter(cfg, [anthropic_response(good, input_tokens=100_000, output_tokens=10_000, cache_read=0, cache_write=0)])
    rc = adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 8000)
    assert rc.cost_usd == pytest.approx(0.2 + 0.1)


def test_R21_no_parsed_output_falls_back_to_text_so_the_repair_loop_can_act(cfg):
    resp = anthropic_response(None)
    resp.content = [type(resp.content[0])(type="text", text="not json")]
    adapter, _ = make_adapter(cfg, [resp])
    rc = adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 8000)
    assert rc.text_or_obj == "not json"


def test_R21_max_tokens_stop_reason_truncates_via_call_stage(cfg):
    resp = anthropic_response(None, stop_reason="max_tokens")
    resp.content = [type(resp.content[0])(type="text", text='{"description": "cut')]
    adapter, _ = make_adapter(cfg, [resp])
    with pytest.raises(StageTruncated):
        call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=UsageLedger(), adapter=adapter)


def test_R21_repair_loop_runs_through_call_stage(cfg, good):
    adapter, client = make_adapter(cfg, [anthropic_response({"title": "only"}), anthropic_response(good)])
    ledger = UsageLedger()
    res = call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=adapter)
    assert res.attempts == 2 and len(client.calls) == 2
    assert res.provider == "anthropic_api"
    assert ledger.total()["cost_usd"] is not None and ledger.total()["cost_usd"] > 0


def test_R21_three_bad_outputs_fail_loudly(cfg):
    adapter, _ = make_adapter(cfg, [anthropic_response({"x": 1}), anthropic_response({"x": 2}), anthropic_response({"x": 3})])
    with pytest.raises(StageOutputInvalid):
        call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=UsageLedger(), adapter=adapter)


def test_R21_client_exception_surfaces_as_provider_error(cfg):
    adapter, _ = make_adapter(cfg, [RuntimeError("connection reset")])
    with pytest.raises((ProviderError, RuntimeError)):
        adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 8000)
