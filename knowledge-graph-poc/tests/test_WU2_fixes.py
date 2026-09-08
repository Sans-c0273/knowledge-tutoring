"""WU2 fix cycle 1 — regression pins for commits 53b6f97, 5dbad3b, 7cd1956, 14be7bf, ff23b04.

PRD R21 (LLM boundary), R6 (image describe), R15 (cost ledger), R18 (config guards).
Each test pins one claim the implementer made in that cycle; these are characterization
tests of the fixed behaviour, so they must keep passing on any refactor of src/kg/llm.

Offline and deterministic: adapters are exercised against the wu2_fakes transports; the
noisy image is built from a seeded `random.Random`; no clock, no network.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import random
import re
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from kg.config import ConfigError, load_config
from kg.convert._common import ConversionError
from kg.convert.image import MAX_IMAGE_BYTES, prepare_image
from kg.llm import ProviderError, StageOutputInvalid, StageTruncated, call_stage
from kg.llm.adapters.anthropic_api import AnthropicApiAdapter
from kg.llm.adapters.base import Msg, RawCompletion, Usage
from kg.llm.adapters.claude_subscription import ClaudeSubscriptionAdapter, agent_cwd, cleanup_transcripts, project_slug
from kg.llm.ledger import UsageLedger, cost_display
from kg.llm.schema_profile import export
from kg.schemas import DescribeOutput

from wu2_fakes import SYSTEM, USER_TEXT, FakeAnthropicClient, FakeQuery, anthropic_response, describe_payload, load_cfg, raw, sdk_rate_limit_event, sdk_result

NOW = 1_800_000_000.0
MODEL = "claude-sonnet-5"


@pytest.fixture
def cfg(make_project):
    return load_cfg(make_project)  # claude_subscription, repair_retries=2


@pytest.fixture
def good():
    return describe_payload(DescribeOutput)


class ScriptedAdapter:
    """Adapter-protocol stand-in: each script entry is a RawCompletion to return or an exception to raise.

    Records the `messages` list of every call so tests can inspect the repair turns.
    """

    provider = "scripted"
    max_repairs: int | None = None

    def __init__(self, scripts: list[Any]) -> None:
        self.scripts = list(scripts)
        self.calls: list[list[Msg]] = []

    def complete(self, system: str, messages: list[Msg], schema_json: dict[str, Any], model: str, max_tokens: int) -> RawCompletion:
        self.calls.append(list(messages))
        if not self.scripts:
            raise AssertionError("ScriptedAdapter exhausted")
        item = self.scripts.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def one_user_msg(text: str = USER_TEXT):
    return [Msg(role="user", text=text)]


def sub_adapter(cfg, scripts):
    fq = FakeQuery(scripts)
    return ClaudeSubscriptionAdapter(cfg, query_fn=fq, clock=lambda: NOW), fq


# ------------------------------------------------- (a) provider events reach the ledger (53b6f97)


def test_R21_fix_events_on_completion_are_copied_to_the_ledger_row(cfg, good):
    ev = {"status": "allowed_warning", "rate_limit_type": "five_hour", "utilization": 0.9, "resets_at": 1}
    adapter = ScriptedAdapter([raw(good, events=[ev])])
    ledger = UsageLedger()
    call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=adapter)
    assert len(ledger.rows) == 1
    assert list(ledger.rows[0].events) == [ev]


def test_R21_fix_ledger_total_events_are_tagged_with_stage_and_attempt_no(cfg, good):
    ev1 = {"status": "allowed_warning", "rate_limit_type": "five_hour", "utilization": 0.8, "resets_at": 10}
    ev2 = {"status": "allowed_warning", "rate_limit_type": "seven_day", "utilization": 0.95, "resets_at": 20}
    # attempt 1 invalid (carries ev1), attempt 2 ok (carries ev2)
    adapter = ScriptedAdapter([raw({"description": "x", "extra": 1}, events=[ev1]), raw(good, events=[ev2])])
    ledger = UsageLedger()
    call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=adapter)
    total_events = ledger.to_dict()["total"]["events"]
    assert total_events == [{**ev1, "stage": "describe", "attempt_no": 1}, {**ev2, "stage": "describe", "attempt_no": 2}]
    assert ledger.total()["events"] == total_events
    rows = ledger.to_dict()["rows"]
    assert rows[0]["events"] == [ev1] and rows[1]["events"] == [ev2]
    json.dumps(ledger.to_dict())  # report-serialisable


def test_R21_fix_rate_limit_event_from_the_sdk_lands_in_the_ledger_via_call_stage(cfg, good):
    ev = sdk_rate_limit_event("allowed_warning", int(NOW) + 900, "five_hour", 0.92)
    adapter, _ = sub_adapter(cfg, [[ev, sdk_result(structured_output=good)]])
    ledger = UsageLedger()
    call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=adapter)
    events = ledger.events()
    assert len(events) == 1
    assert events[0]["status"] == "allowed_warning" and events[0]["stage"] == "describe" and events[0]["attempt_no"] == 1


def test_R21_fix_no_events_means_empty_lists_not_missing_keys(cfg, good):
    adapter = ScriptedAdapter([raw(good)])
    ledger = UsageLedger()
    call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=adapter)
    d = ledger.to_dict()
    assert d["total"]["events"] == []
    assert d["rows"][0]["events"] == []


# ---------------------------------------------- (b) encoded image payload cap (ff23b04, R6)


def _noise_rgba_png(path: Path, side: int = 1600, seed: int = 20260825) -> Path:
    """Incompressible RGBA noise: as PNG this is ~10.8 MB, far above MAX_IMAGE_BYTES."""
    rnd = random.Random(seed)
    img = Image.frombytes("RGBA", (side, side), rnd.randbytes(side * side * 4))
    img.save(path, format="PNG", compress_level=1)
    assert path.stat().st_size > MAX_IMAGE_BYTES, "fixture must exceed the cap as a PNG"
    return path


def test_R6_fix_max_image_bytes_is_4_5_mib_under_anthropics_5_mb_limit():
    assert MAX_IMAGE_BYTES == int(4.5 * 1024 * 1024)
    assert MAX_IMAGE_BYTES < 5_000_000


def test_R6_fix_oversize_noisy_rgba_png_is_reencoded_as_jpeg_under_the_cap(tmp_path):
    src = _noise_rgba_png(tmp_path / "noise.png")
    payload = prepare_image(src, max_long_edge_px=1568)
    assert payload.media_type == "image/jpeg"
    data = base64.b64decode(payload.data_b64)
    assert len(data) <= MAX_IMAGE_BYTES
    with Image.open(io.BytesIO(data)) as decoded:
        decoded.load()
        assert decoded.format == "JPEG"
        assert max(decoded.size) <= 1568


def test_R6_fix_media_type_names_the_encoding_actually_sent(tmp_path):
    """A PNG that fits stays PNG; only the fallback flips the media type."""
    src = tmp_path / "small.png"
    Image.new("RGBA", (64, 64), (10, 20, 30, 255)).save(src, format="PNG")
    payload = prepare_image(src, max_long_edge_px=1568)
    assert payload.media_type == "image/png"
    assert base64.b64decode(payload.data_b64)[:8] == b"\x89PNG\r\n\x1a\n"


def test_R6_fix_original_file_is_left_untouched_by_the_fallback(tmp_path):
    src = _noise_rgba_png(tmp_path / "noise.png")
    before = src.read_bytes()
    prepare_image(src, max_long_edge_px=1568)
    assert src.read_bytes() == before


def test_R6_fix_unreadable_image_still_fails_before_any_encoding(tmp_path):
    src = tmp_path / "broken.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    with pytest.raises(ConversionError):
        prepare_image(src, max_long_edge_px=1568)


# ------------------------------------------------- (c) one JSON fence tolerated (53b6f97)


def _cfg_no_repairs(make_project):
    return load_cfg(make_project, **{"llm.repair_retries": 0})


def test_R21_fix_single_json_fence_around_the_whole_output_is_accepted(make_project, good):
    cfg = _cfg_no_repairs(make_project)
    adapter = ScriptedAdapter([raw("```json\n" + json.dumps(good) + "\n```")])
    res = call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=UsageLedger(), adapter=adapter)
    assert res.data.description == good["description"]
    assert res.attempts == 1


def test_R21_fix_bare_fence_without_language_tag_is_accepted(make_project, good):
    cfg = _cfg_no_repairs(make_project)
    adapter = ScriptedAdapter([raw("```\n" + json.dumps(good) + "\n```\n")])
    res = call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=UsageLedger(), adapter=adapter)
    assert res.attempts == 1


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("Here is the JSON:\n```json\n{payload}\n```", id="prose-before-fence"),
        pytest.param("```json\n{payload}\n```\nHope this helps!", id="prose-after-fence"),
        pytest.param("```json\n{payload}\n```\n```json\n{payload}\n```", id="double-fence"),
        pytest.param("```json\n```json\n{payload}\n```\n```", id="nested-fence"),
    ],
)
def test_R21_fix_text_outside_the_fence_or_a_double_fence_is_still_invalid(make_project, good, text):
    cfg = _cfg_no_repairs(make_project)
    adapter = ScriptedAdapter([raw(text.format(payload=json.dumps(good)))])
    ledger = UsageLedger()
    with pytest.raises(StageOutputInvalid) as ei:
        call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=adapter)
    assert ei.value.stage == "describe" and ei.value.attempts == 1
    assert ledger.rows[0].outcome == "invalid"


def test_R21_fix_unfenced_json_still_parses_as_before(make_project, good):
    cfg = _cfg_no_repairs(make_project)
    adapter = ScriptedAdapter([raw(json.dumps(good))])
    res = call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=UsageLedger(), adapter=adapter)
    assert res.attempts == 1


def test_R21_fix_bare_dict_from_a_structured_output_adapter_is_accepted_without_parsing(make_project, good):
    """Structured-output adapters hand back the object itself; no JSON round-trip, no fence handling."""
    cfg = _cfg_no_repairs(make_project)
    adapter = ScriptedAdapter([raw(dict(good))])
    res = call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=UsageLedger(), adapter=adapter)
    assert res.attempts == 1 and res.data.description == good["description"]


def test_R21_fix_fenced_non_object_json_is_still_invalid(make_project, good):
    cfg = _cfg_no_repairs(make_project)
    adapter = ScriptedAdapter([raw("```json\n" + json.dumps([good]) + "\n```")])
    with pytest.raises(StageOutputInvalid) as ei:
        call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=UsageLedger(), adapter=adapter)
    assert "object" in str(ei.value)


# --------------------------------------------------- (d) agent cwd must be empty (7cd1956, D21)


def test_D21_fix_agent_cwd_with_a_planted_file_is_a_config_error(cfg):
    path = cfg.sandbox.resolve("runs", ".agent-cwd")
    path.mkdir(parents=True, exist_ok=True)
    (path / "CLAUDE.md").write_text("# injected project context\n", encoding="utf-8")
    with pytest.raises(ConfigError) as ei:
        agent_cwd(cfg)
    assert "empty" in str(ei.value).lower()


def test_D21_fix_adapter_construction_refuses_a_non_empty_agent_cwd(cfg):
    path = cfg.sandbox.resolve("runs", ".agent-cwd")
    path.mkdir(parents=True, exist_ok=True)
    (path / "settings.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ConfigError):
        ClaudeSubscriptionAdapter(cfg, query_fn=FakeQuery([]), clock=lambda: NOW)


def test_D21_fix_agent_cwd_is_created_empty_when_absent(cfg):
    path = agent_cwd(cfg)
    assert path.is_dir() and not any(path.iterdir())
    assert path == cfg.sandbox.resolve("runs", ".agent-cwd")


# ------------------------- (e) adapter-raised StageOutputInvalid keeps its usage (53b6f97)


def test_R15_fix_adapter_raised_invalid_carries_tokens_into_the_ledger_row(cfg, good):
    exc = StageOutputInvalid("adapter: exhausted", usage=Usage(input_tokens=1234, output_tokens=56, cached_tokens=7), cost_estimate_usd=0.02, model_served="claude-sonnet-5-20260101")
    adapter = ScriptedAdapter([exc, raw(good)])
    ledger = UsageLedger()
    res = call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=adapter)
    assert res.attempts == 2
    row = ledger.rows[0]
    assert row.outcome == "invalid"
    assert (row.input_tokens, row.output_tokens, row.cached_tokens) == (1234, 56, 7)
    assert row.cost_estimate_usd == pytest.approx(0.02)
    assert row.model_served == "claude-sonnet-5-20260101"
    assert row.cost_usd is None


def test_R21_fix_adapter_raised_invalid_retry_has_no_empty_assistant_turn(cfg, good):
    adapter = ScriptedAdapter([StageOutputInvalid("adapter: exhausted", usage=Usage(10, 1)), raw(good)])
    call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=UsageLedger(), adapter=adapter)
    second_call = adapter.calls[1]
    assert [m.role for m in second_call] == ["user", "user"], "no assistant turn should be fabricated"
    assert not any(m.role == "assistant" and m.text == "" for m in second_call)
    assert second_call[0].text == USER_TEXT
    assert "adapter: exhausted" in second_call[1].text


def test_R21_fix_adapter_raised_invalid_without_usage_records_zero_tokens_not_a_crash(cfg, good):
    adapter = ScriptedAdapter([StageOutputInvalid("adapter: exhausted"), raw(good)])
    ledger = UsageLedger()
    call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=adapter)
    assert (ledger.rows[0].input_tokens, ledger.rows[0].output_tokens) == (0, 0)


def test_R21_fix_final_stage_output_invalid_propagates_the_adapter_usage(cfg):
    adapter = ScriptedAdapter([StageOutputInvalid("a", usage=Usage(5, 1)), StageOutputInvalid("b", usage=Usage(6, 2)), StageOutputInvalid("c", usage=Usage(7, 3))])
    ledger = UsageLedger()
    with pytest.raises(StageOutputInvalid) as ei:
        call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=adapter)
    assert ei.value.stage == "describe" and ei.value.attempts == 3
    assert isinstance(ei.value.usage, Usage) and ei.value.usage.input_tokens == 7
    assert [r.input_tokens for r in ledger.rows] == [5, 6, 7]


def test_R15_fix_sdk_structured_output_exhaustion_is_not_a_zero_token_row(cfg, good):
    """End to end through the real subscription adapter: the SDK's ResultMessage still reports usage."""
    exhausted = sdk_result(
        subtype="error_max_structured_output_retries",
        is_error=True,
        structured_output=None,
        usage={"input_tokens": 4321, "output_tokens": 987, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 0},
        total_cost_usd=0.0456,
        model_usage={"claude-sonnet-5-20260101": {"inputTokens": 4321}},
    )
    adapter, fq = sub_adapter(cfg, [[exhausted], [sdk_result(structured_output=good)]])
    ledger = UsageLedger()
    res = call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=adapter)
    assert res.attempts == 2
    row = ledger.rows[0]
    assert (row.input_tokens, row.output_tokens, row.cached_tokens) == (4321, 987, 100)
    assert row.cost_estimate_usd == pytest.approx(0.0456)
    assert row.model_served == "claude-sonnet-5-20260101"
    # The re-query quotes nothing: there was no model output to echo.
    retry_text = fq.calls[1]["prompt_messages"][0]["message"]["content"][-1]["text"]
    assert retry_text.startswith(USER_TEXT)
    assert "quoted below" not in retry_text and "```" not in retry_text


def test_R21_fix_repair_after_a_real_bad_output_quotes_it_as_fenced_data(cfg, good):
    adapter, fq = sub_adapter(cfg, [[sdk_result(structured_output={"description": "x", "extra": 1})], [sdk_result(structured_output=good)]])
    call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=UsageLedger(), adapter=adapter)
    retry_text = fq.calls[1]["prompt_messages"][0]["message"]["content"][-1]["text"]
    assert "not an instruction" in retry_text
    assert "```text" in retry_text and '"extra": 1' in retry_text


# -------------------------------------------- (f) pricing table validated at load (14be7bf)


def _pricing_cfg_error(make_project, **overrides):
    with pytest.raises(ConfigError) as ei:
        load_cfg(make_project, "anthropic_api", **overrides)
    return str(ei.value)


def test_R18_fix_pricing_input_given_as_a_string_is_a_config_error(make_project):
    msg = _pricing_cfg_error(make_project, **{"llm.anthropic_api.pricing_usd_per_mtok.claude-sonnet-5.input": "two"})
    assert "pricing_usd_per_mtok" in msg and "claude-sonnet-5" in msg


def test_R18_fix_negative_cache_multiplier_is_a_config_error(make_project):
    msg = _pricing_cfg_error(make_project, **{"llm.anthropic_api.pricing_usd_per_mtok.cache_read_multiplier": -0.1})
    assert "cache_read_multiplier" in msg


def test_R18_fix_negative_output_price_is_a_config_error(make_project):
    _pricing_cfg_error(make_project, **{"llm.anthropic_api.pricing_usd_per_mtok.claude-opus-5.output": -25})


def test_R18_fix_boolean_price_is_a_config_error(make_project):
    _pricing_cfg_error(make_project, **{"llm.anthropic_api.pricing_usd_per_mtok.claude-opus-5.input": True})


def test_R18_fix_unknown_key_in_a_model_row_is_a_config_error(make_project):
    msg = _pricing_cfg_error(make_project, **{"llm.anthropic_api.pricing_usd_per_mtok.claude-opus-5.cached": 1})
    assert "cached" in msg


def test_R18_fix_valid_pricing_loads_as_floats(make_project):
    cfg = load_cfg(make_project, "anthropic_api")
    table = cfg.llm.settings.pricing_usd_per_mtok
    assert table["claude-sonnet-5"] == {"input": 2.0, "output": 10.0}
    assert table["cache_read_multiplier"] == pytest.approx(0.10)


def test_R18_fix_malformed_pricing_is_ignored_when_another_provider_is_selected(make_project):
    """Only the selected provider's block is validated (current behaviour: a subscription run must not
    be blocked by a typo in the unused anthropic_api table)."""
    cfg = load_cfg(make_project, "claude_subscription", **{"llm.anthropic_api.pricing_usd_per_mtok.claude-sonnet-5.input": "two"})
    assert cfg.llm.provider == "claude_subscription"


# ----------------------------------------------- (g) StageTruncated carries stage/attempts


def test_R21_fix_stage_truncated_carries_stage_and_attempts(cfg, good):
    adapter = ScriptedAdapter([raw('{"description": "cut', stop_reason="max_tokens")])
    ledger = UsageLedger()
    with pytest.raises(StageTruncated) as ei:
        call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=ledger, adapter=adapter)
    assert ei.value.stage == "describe"
    assert ei.value.attempts == 1
    assert ledger.rows[0].outcome == "truncated"


def test_R21_fix_stage_truncated_on_the_second_attempt_reports_attempts_2(cfg, good):
    adapter = ScriptedAdapter([raw({"description": "x", "extra": 1}), raw('{"description": "cut', stop_reason="length")])
    with pytest.raises(StageTruncated) as ei:
        call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=cfg, ledger=UsageLedger(), adapter=adapter)
    assert (ei.value.stage, ei.value.attempts) == ("describe", 2)


def test_R21_fix_stage_truncated_raised_bare_has_none_fields():
    exc = StageTruncated("bare")
    assert exc.stage is None and exc.attempts is None


# ------------------------------------------------ (h) cost_display sub-cent estimates (D28)


def test_R15_fix_subscription_estimate_below_a_cent_is_not_rendered_as_zero():
    text = cost_display("claude_subscription", None, 0.0012)
    assert "$0.0012" in text
    assert not re.search(r"\$0\.00(?!\d)", text), text
    assert text.startswith("n/a (subscription)")


def test_R15_fix_subscription_without_estimate_shows_only_na():
    assert cost_display("claude_subscription", None, None) == "n/a (subscription)"


def test_R15_fix_paid_provider_cost_uses_the_same_four_decimals():
    assert cost_display("openrouter", 0.0012) == "$0.0012"
    assert cost_display("openrouter", None) == "unknown (no cost reported)"


# --------------------------------------- (i) complete() inside a running event loop (7cd1956)


def test_R21_fix_complete_inside_a_running_event_loop_is_a_provider_error(cfg, good):
    adapter, fq = sub_adapter(cfg, [[sdk_result(structured_output=good)]])

    async def inside_loop():
        return adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 8000)

    with pytest.raises(ProviderError) as ei:
        asyncio.run(inside_loop())
    assert "event loop" in str(ei.value)
    assert fq.calls == [], "no query may be issued when the guard fires"


def test_R21_fix_complete_outside_a_loop_still_works_afterwards(cfg, good):
    adapter, fq = sub_adapter(cfg, [[sdk_result(structured_output=good)]])
    rc = adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 8000)
    assert rc.text_or_obj == good and len(fq.calls) == 1


# ---------------------------------------------------------- (j) skills=[] (7cd1956, D21)


def test_D21_fix_options_disable_skills_explicitly(cfg, good):
    adapter, fq = sub_adapter(cfg, [[sdk_result(structured_output=good)]])
    adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 8000)
    opts = fq.calls[0]["options"]
    assert opts.skills == []  # None would leave the CLI's own skill defaults in force


# ------------------------------------------ (k) transcript cleanup is scoped to the slug (7cd1956)


def test_R21_fix_project_slug_replaces_every_non_alphanumeric_with_a_dash():
    assert project_slug("/Users/me/proj dir.v2/.agent-cwd") == "-Users-me-proj-dir-v2--agent-cwd"
    assert project_slug("C:\\work\\kg_run") == "C--work-kg-run"
    assert project_slug("abc123") == "abc123"


def test_R21_fix_project_slug_over_200_chars_is_cut_and_hash_suffixed():
    long = "/" + "a" * 250
    slug = project_slug(long)
    assert slug.startswith("-" + "a" * 199 + "-")
    assert len(slug) > 200 and re.fullmatch(r"-a{199}-[0-9a-z]+", slug)


@pytest.fixture
def transcript_root(tmp_path, monkeypatch):
    root = tmp_path / "claude-config"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(root))
    return root


def _plant(projects: Path, slug: str) -> Path:
    d = projects / slug
    d.mkdir(parents=True)
    (d / "session.jsonl").write_text('{"type":"user","content":"secret prompt"}\n', encoding="utf-8")
    return d


def test_R21_fix_cleanup_removes_only_the_slug_dir_for_the_sandbox_cwd(cfg, transcript_root):
    cwd = agent_cwd(cfg)
    projects = transcript_root / "projects"
    target = _plant(projects, project_slug(str(cwd)))
    sibling = _plant(projects, "-Users-someone-else-real-project")
    other_cwd = _plant(projects, project_slug(str(cwd.parent / "not-the-agent-cwd")))
    stray_file = projects / "notes.txt"
    stray_file.write_text("keep me", encoding="utf-8")

    removed = cleanup_transcripts(cwd)

    assert removed == [target]
    assert not target.exists()
    assert sibling.is_dir() and (sibling / "session.jsonl").exists()
    assert other_cwd.is_dir()
    assert stray_file.read_text(encoding="utf-8") == "keep me"
    assert projects.is_dir() and transcript_root.is_dir()


def test_R21_fix_cleanup_with_no_transcript_dir_is_a_silent_noop(cfg, transcript_root):
    cwd = agent_cwd(cfg)
    assert cleanup_transcripts(cwd) == []
    assert not (transcript_root / "projects").exists() or not any((transcript_root / "projects").iterdir())


def test_R21_fix_adapter_cleans_its_transcript_after_a_successful_call(cfg, good, transcript_root):
    adapter, _ = sub_adapter(cfg, [[sdk_result(structured_output=good)]])
    projects = transcript_root / "projects"
    target = _plant(projects, project_slug(str(agent_cwd(cfg))))
    sibling = _plant(projects, "-Users-someone-else-real-project")
    adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 8000)
    assert not target.exists()
    assert sibling.is_dir()


def test_R21_fix_adapter_cleans_its_transcript_even_when_the_call_fails(cfg, transcript_root):
    failed = sdk_result(subtype="error_during_execution", is_error=True, api_error_status=401, errors=["Not logged in"])
    adapter, _ = sub_adapter(cfg, [[failed]])
    target = _plant(transcript_root / "projects", project_slug(str(agent_cwd(cfg))))
    with pytest.raises(ProviderError):
        adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 8000)
    assert not target.exists()


def test_R21_fix_adapter_cleanup_transcripts_method_returns_what_it_removed(cfg, transcript_root):
    adapter, _ = sub_adapter(cfg, [])
    target = _plant(transcript_root / "projects", project_slug(str(agent_cwd(cfg))))
    assert adapter.cleanup_transcripts() == [target]
    assert adapter.cleanup_transcripts() == []


# ------------------------------ anthropic_api: text-block route is the default (D30, wu2_fakes)


@pytest.fixture
def api_cfg(make_project):
    return load_cfg(make_project, "anthropic_api")


def test_D30_anthropic_api_default_route_returns_the_text_block_for_call_stage_to_parse(api_cfg, good):
    client = FakeAnthropicClient([anthropic_response(good)])  # parsed_output=None by default
    adapter = AnthropicApiAdapter(api_cfg, client=client)
    rc = adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 8000)
    assert isinstance(rc.text_or_obj, str)
    assert json.loads(rc.text_or_obj) == good
    res = call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, cfg=api_cfg, ledger=UsageLedger(), adapter=AnthropicApiAdapter(api_cfg, client=FakeAnthropicClient([anthropic_response(good)])))
    assert res.attempts == 1 and res.data.description == good["description"]


def test_D30_anthropic_api_parsed_output_route_is_still_honoured_when_the_sdk_provides_it(api_cfg, good):
    client = FakeAnthropicClient([anthropic_response(good, parsed_output=good)])
    adapter = AnthropicApiAdapter(api_cfg, client=client)
    rc = adapter.complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 8000)
    assert rc.text_or_obj == good and isinstance(rc.text_or_obj, dict)


def test_R21_fix_anthropic_api_request_uses_output_config_json_schema(api_cfg, good):
    client = FakeAnthropicClient([anthropic_response(good)])
    AnthropicApiAdapter(api_cfg, client=client).complete(SYSTEM, one_user_msg(), export(DescribeOutput), MODEL, 8000)
    kw = client.calls[0]
    assert kw["method"] == "parse"
    assert kw["output_config"] == {"format": {"type": "json_schema", "schema": export(DescribeOutput)}}
