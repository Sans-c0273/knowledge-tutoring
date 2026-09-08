"""WU2 — OpenRouterAdapter over the `openai` client (DESIGN §7.4, §7.6, §17 row
"Repair-retry — OpenRouter path"; D23, D24; PRD R21(2), R6, R15).

The `openai` client is INJECTED as a FakeOpenAIClient. conftest poisons
`openai.OpenAI.chat`; the last test proves the default wiring is covered by that poison.
"""

from __future__ import annotations

import base64
import json
import re

import pytest

from kg.config import ConfigError
from kg.llm import LLMError, ProviderError, RateLimited, StageOutputInvalid, StageTruncated, call_stage
from kg.llm.adapters.base import ImageInput, Msg, RawCompletion
from kg.llm.adapters.openrouter import OpenRouterAdapter
from kg.llm.ledger import UsageLedger
from kg.llm.schema_profile import export
from kg.schemas import DescribeOutput

from wu2_fakes import FAKE_OR_KEY, JPEG_B64, PNG_B64, SYSTEM, USER_TEXT, FakeOpenAIClient, describe_payload, load_cfg, openai_response, openai_status_error

MODEL = "anthropic/claude-sonnet-5"


@pytest.fixture
def cfg(make_project):
    return load_cfg(make_project, "openrouter")


@pytest.fixture
def good():
    return describe_payload(DescribeOutput)


def make_adapter(cfg, scripts):
    client = FakeOpenAIClient(scripts)
    return OpenRouterAdapter(cfg, client=client), client


def one_user_msg(text: str = USER_TEXT, images=()):
    return [Msg(role="user", text=text, images=list(images))]


# --------------------------------------------------------------- identity


def test_R21_adapter_identifies_its_provider(cfg):
    adapter, _ = make_adapter(cfg, [])
    assert adapter.provider == "openrouter"


def test_R21_adapter_does_not_cap_repairs_below_config(cfg):
    adapter, _ = make_adapter(cfg, [])
    assert adapter.max_repairs is None or adapter.max_repairs >= 2


# --------------------------------------------------------- client wiring


def test_D23_default_client_points_at_openrouter_with_retries_and_title(cfg):
    """No client injected → adapter builds openai.OpenAI(base_url, api_key, max_retries=3, timeout, X-Title)."""
    adapter = OpenRouterAdapter(cfg)
    client = adapter.client
    assert str(client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"
    assert client.max_retries == 3
    assert client.api_key == FAKE_OR_KEY
    assert client.default_headers.get("X-Title") == "kg-mapper-poc"
    assert client.timeout is not None  # cfg.llm.stage_timeout_s


def test_D23_base_url_follows_config(make_project):
    cfg = load_cfg(make_project, "openrouter", **{"llm.openrouter.base_url": "https://openrouter.ai/api/v1"})
    adapter = OpenRouterAdapter(cfg)
    assert str(adapter.client.base_url).startswith("https://openrouter.ai/api/v1")


def test_R21_adapter_refuses_a_config_for_another_provider(make_project):
    cfg = load_cfg(make_project)  # claude_subscription
    with pytest.raises(ConfigError):
        OpenRouterAdapter(cfg, client=FakeOpenAIClient([]))


# ---------------------------------------------------------- request shape


def test_R21_request_uses_native_json_schema_strict_response_format(cfg, good):
    adapter, client = make_adapter(cfg, [openai_response(json.dumps(good))])
    adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 8000)
    kw = client.calls[0]
    rf = kw["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"] == export(DescribeOutput)
    name = rf["json_schema"]["name"]
    assert isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name), name


def test_R21_request_carries_provider_routing_block_in_extra_body(cfg, good):
    adapter, client = make_adapter(cfg, [openai_response(json.dumps(good))])
    adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 8000)
    prov = client.calls[0]["extra_body"]["provider"]
    assert prov["require_parameters"] is True
    assert prov["data_collection"] == "deny"
    assert prov["order"] == []
    assert prov["allow_fallbacks"] is True


def test_R21_provider_routing_block_follows_config(make_project, good):
    cfg = load_cfg(
        make_project,
        "openrouter",
        **{"llm.openrouter.provider.order": ["anthropic", "google-vertex"], "llm.openrouter.provider.allow_fallbacks": False},
    )
    adapter, client = make_adapter(cfg, [openai_response(json.dumps(good))])
    adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 8000)
    prov = client.calls[0]["extra_body"]["provider"]
    assert prov["order"] == ["anthropic", "google-vertex"]
    assert prov["allow_fallbacks"] is False
    assert prov["require_parameters"] is True


def test_R21_request_carries_model_max_tokens_and_messages(cfg, good):
    adapter, client = make_adapter(cfg, [openai_response(json.dumps(good))])
    adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 4321)
    kw = client.calls[0]
    assert kw["model"] == MODEL
    assert kw["max_tokens"] == 4321
    msgs = kw["messages"]
    assert msgs[0] == {"role": "system", "content": SYSTEM}
    assert msgs[1]["role"] == "user"
    content = msgs[1]["content"]
    if isinstance(content, list):
        assert content[-1] == {"type": "text", "text": USER_TEXT}
    else:
        assert content == USER_TEXT
    assert kw.get("stream") in (None, False)


def test_R6_image_is_sent_as_image_url_data_uri_with_correct_media_type(cfg, good):
    img = ImageInput(media_type="image/jpeg", data_b64=JPEG_B64)
    adapter, client = make_adapter(cfg, [openai_response(json.dumps(good))])
    adapter.complete(SYSTEM, one_user_msg(images=[img]), export(DescribeOutput), "google/gemini-2.5-flash", 8000)
    content = client.calls[0]["messages"][1]["content"]
    assert isinstance(content, list)
    assert [p["type"] for p in content] == ["image_url", "text"]
    url = content[0]["image_url"]["url"]
    assert url == f"data:image/jpeg;base64,{JPEG_B64}"
    assert base64.b64decode(url.split(",", 1)[1])  # decodes


def test_R6_png_image_uses_png_media_type_in_data_uri(cfg, good):
    img = ImageInput(media_type="image/png", data_b64=PNG_B64)
    adapter, client = make_adapter(cfg, [openai_response(json.dumps(good))])
    adapter.complete(SYSTEM, one_user_msg(images=[img]), export(DescribeOutput), MODEL, 8000)
    url = client.calls[0]["messages"][1]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")


def test_R21_repair_turns_are_rendered_as_assistant_and_user_messages(cfg, good):
    adapter, client = make_adapter(cfg, [openai_response(json.dumps(good))])
    msgs = [
        Msg(role="user", text=USER_TEXT),
        Msg(role="assistant", text="garbage"),
        Msg(role="user", text="The JSON failed validation: ... Return only corrected JSON."),
    ]
    adapter.complete(SYSTEM, msgs, export(DescribeOutput), MODEL, 8000)
    wire = client.calls[0]["messages"]
    assert [m["role"] for m in wire] == ["system", "user", "assistant", "user"]
    assert wire[2]["content"] == "garbage"


# ------------------------------------------------------- response parsing


def test_R15_response_usage_cost_and_served_model_are_captured(cfg, good):
    resp = openai_response(
        json.dumps(good),
        model="anthropic/claude-sonnet-5-20260101",
        provider="Google Vertex",
        prompt_tokens=1200,
        completion_tokens=250,
        cached_tokens=300,
        reasoning_tokens=40,
        cost=0.0042,
    )
    adapter, _ = make_adapter(cfg, [resp])
    rc = adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 8000)
    assert isinstance(rc, RawCompletion)
    assert rc.text_or_obj == json.dumps(good)
    assert rc.usage.input_tokens == 1200
    assert rc.usage.output_tokens == 250
    assert rc.usage.cached_tokens == 300
    assert rc.usage.reasoning_tokens == 40
    assert rc.cost_usd == pytest.approx(0.0042)
    assert rc.model_requested == MODEL
    assert rc.model_served == "anthropic/claude-sonnet-5-20260101"
    assert rc.provider_meta.get("upstream_provider") == "Google Vertex"
    assert rc.stop_reason == "stop"


def test_R15_missing_cost_is_none_not_zero(cfg, good):
    adapter, _ = make_adapter(cfg, [openai_response(json.dumps(good), cost=None)])
    rc = adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 8000)
    assert rc.cost_usd is None


def test_R15_missing_token_details_default_to_zero(cfg, good):
    resp = openai_response(json.dumps(good))
    resp.usage.prompt_tokens_details = None
    resp.usage.completion_tokens_details = None
    adapter, _ = make_adapter(cfg, [resp])
    rc = adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 8000)
    assert rc.usage.cached_tokens == 0 and rc.usage.reasoning_tokens == 0


def test_R21_empty_content_is_returned_as_empty_text_for_the_repair_loop(cfg):
    adapter, _ = make_adapter(cfg, [openai_response(None)])
    rc = adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 8000)
    assert rc.text_or_obj in ("", None)


def test_R21_finish_reason_length_surfaces_as_stop_reason_and_truncates_via_call_stage(cfg):
    adapter, _ = make_adapter(cfg, [openai_response('{"description": "cut', finish_reason="length")])
    rc = adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 8000)
    assert rc.stop_reason == "length"

    adapter2, _ = make_adapter(cfg, [openai_response('{"description": "cut', finish_reason="length")])
    with pytest.raises(StageTruncated):
        call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=UsageLedger(), adapter=adapter2)


# ------------------------------------------------ §17 repair-retry, end-to-end


def test_R21_openrouter_repair_path_non_json_then_missing_key_then_valid(cfg, good):
    adapter, client = make_adapter(
        cfg,
        [
            openai_response("Sure! Here is the JSON you asked for."),
            openai_response(json.dumps({"kind": "diagram"})),
            openai_response(json.dumps(good)),
        ],
    )
    ledger = UsageLedger()
    res = call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=adapter)
    assert isinstance(res.data, DescribeOutput)
    assert res.attempts == 3
    assert len(client.calls) == 3
    assert len(ledger.rows) == 3
    for kw in client.calls:  # every attempt kept the strict schema + routing guard
        assert kw["response_format"]["type"] == "json_schema"
        assert kw["response_format"]["json_schema"]["strict"] is True
        assert kw["extra_body"]["provider"]["require_parameters"] is True
    assert [m["role"] for m in client.calls[2]["messages"]] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert res.provider == "openrouter"
    assert ledger.total()["cost_usd"] == pytest.approx(3 * 0.0042)


def test_R21_openrouter_three_bad_outputs_fail_loudly(cfg):
    adapter, client = make_adapter(cfg, [openai_response("a"), openai_response("b"), openai_response("c"), openai_response("never")])
    with pytest.raises(StageOutputInvalid):
        call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=UsageLedger(), adapter=adapter)
    assert len(client.calls) == 3


# ------------------------------------------------------ errors and 429 (§7.6)


def test_R21_persisting_http_429_becomes_rate_limited_defer_without_adapter_level_retry(cfg):
    """The openai client already retries ×3 (max_retries=3); the adapter must not multiply that."""
    adapter, client = make_adapter(cfg, [openai_status_error(429), openai_status_error(429)])
    with pytest.raises(RateLimited) as ei:
        adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 8000)
    assert ei.value.action == "defer"
    assert len(client.calls) == 1


def test_R21_http_5xx_becomes_provider_error(cfg):
    adapter, client = make_adapter(cfg, [openai_status_error(502, "bad gateway")])
    with pytest.raises(ProviderError) as ei:
        adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 8000)
    assert isinstance(ei.value, LLMError)
    assert "502" in str(ei.value)
    assert len(client.calls) == 1


def test_R21_http_4xx_other_than_429_becomes_provider_error(cfg):
    adapter, _ = make_adapter(cfg, [openai_status_error(400, "structured outputs not supported by any provider")])
    with pytest.raises(ProviderError) as ei:
        adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 8000)
    assert "400" in str(ei.value)


def test_R18_provider_error_message_never_contains_the_api_key(cfg):
    adapter, _ = make_adapter(cfg, [openai_status_error(401, "Unauthorized")])
    with pytest.raises(LLMError) as ei:
        adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 8000)
    assert FAKE_OR_KEY not in str(ei.value)
    assert FAKE_OR_KEY not in repr(adapter)


# ----------------------------------------------------- zero-token guarantee


def test_R20_default_wiring_reaches_openai_chat_and_is_covered_by_the_selftest_poison(cfg):
    adapter = OpenRouterAdapter(cfg)  # real openai.OpenAI object; no network at construction
    with pytest.raises(AssertionError, match="real LLM provider invoked"):
        adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 8000)
