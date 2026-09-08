"""WU3 — `kg.report`: run report `.md` + `.json` (DESIGN §14; D28; PRD R15, R18, R21).

Interface (chosen):
  build(cfg, *, run_id, started, finished, mode, inputs, nodes, edges_by_type, rejected_edges,
        review_queues, ledger, prompt_versions, stage_models=None, warnings=(), events=(), moves=None,
        provider_overridden=False) -> RunReport
  write(rep, cfg) -> (md_path, json_path)   under <graph>/_reports/run-<run_id>.{md,json}
  render_markdown(rep) -> str ; to_dict(rep) -> dict ; ReportRedactionError (raised by write(); nothing written)
Plain dicts are used for rows so the report module owns no extra types.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kg.config import load_config
from wu2_fakes import FAKE_OR_KEY
from wu3_fixtures import RUN_ID

STARTED = "2026-08-25T10:15:02Z"
FINISHED = "2026-08-25T10:18:40Z"


def UsageLedger(*a, **kw):  # lazy: kg.llm.ledger is WU2, built concurrently
    from kg.llm.ledger import UsageLedger as _L

    return _L(*a, **kw)


def row(**over):
    from kg.llm.ledger import UsageRow

    kw = {
        "stage": "atomize",
        "provider": "claude_subscription",
        "model_requested": "claude-sonnet-5",
        "model_served": "claude-sonnet-5",
        "input_tokens": 1000,
        "output_tokens": 200,
        "cost_usd": None,
        "cost_estimate_usd": 0.31,
        "attempt_no": 1,
        "outcome": "ok",
        "duration_s": 1.5,
        "prompt_version": "atomize@1",
    }
    kw.update(over)
    return UsageRow(**kw)


def fields(**over):
    base = {
        "run_id": RUN_ID,
        "started": STARTED,
        "finished": FINISHED,
        "mode": "ingest",
        "inputs": [
            {"name": "a.md", "status": "ingested", "reason": None, "units": 3, "chunks": 1},
            {"name": "legacy.xls", "status": "unsupported", "reason": "unsupported extension .xls", "units": 0, "chunks": 0},
            {"name": "b.md", "status": "deferred", "reason": "atomize: StageOutputInvalid after 3 attempts", "units": 2, "chunks": 1},
        ],
        "nodes": {"created": 3, "updated": 0, "total": 3},
        "edges_by_type": {"related_to": 1, "part_of": 1},
        "rejected_edges": [{"gate": "relevance", "reason": "related_to without relevance", "type": "related_to", "source_id": "sample-0001", "target_id": "sample-0002"}],
        "review_queues": {"duplicates": 1, "grounding": 0, "not_adjudicated": 2},
        "prompt_versions": {"describe": "describe@1", "atomize": "atomize@1", "edges": "edges@1", "dedup": "dedup@1"},
        "warnings": ["roster truncated to 400 of 412 nodes"],
        "events": [],
        "moves": {"before": {"inbox": 3, "processed": 0}, "after": {"inbox": 2, "processed": 1}},
    }
    base.update(over)
    return base


@pytest.fixture
def cfg(make_project):
    return load_config(make_project())


@pytest.fixture
def sub_ledger():
    l = UsageLedger()
    l.record(row())
    l.record(row(stage="edges", prompt_version="edges@1", cost_estimate_usd=0.10))
    return l


def build_and_write(cfg, ledger, **over):
    from kg import report

    rep = report.build(cfg, ledger=ledger, **fields(**over))
    md, js = report.write(rep, cfg)
    return rep, Path(md), Path(js)


# -------------------------------------------------------------- location


def test_R15_report_files_are_written_under_graph_reports_named_by_run_id(cfg, sub_ledger):
    _, md, js = build_and_write(cfg, sub_ledger)
    assert md == cfg.sandbox.root("graph") / "_reports" / f"run-{RUN_ID}.md"
    assert js == cfg.sandbox.root("graph") / "_reports" / f"run-{RUN_ID}.json"
    assert md.is_file() and js.is_file()
    json.loads(js.read_text(encoding="utf-8"))  # valid JSON


def test_R15_to_dict_and_json_file_agree(cfg, sub_ledger):
    from kg import report

    rep, _, js = build_and_write(cfg, sub_ledger)
    assert json.loads(js.read_text(encoding="utf-8")) == report.to_dict(rep)


# ---------------------------------------------------------------- header


def test_R15_R21_header_carries_run_schema_provider_and_prompt_versions(cfg, sub_ledger):
    from kg import report

    rep, md, js = build_and_write(cfg, sub_ledger)
    d = json.loads(js.read_text(encoding="utf-8"))
    assert d["run_id"] == RUN_ID
    assert d["started"] == STARTED and d["finished"] == FINISHED
    assert d["mode"] == "ingest" and d["dry_run"] is False
    assert d["corpus"] == "sample" and d["schema"] == "general"
    assert d["provider"] == "claude_subscription" and d["provider_overridden"] is False
    assert d["prompt_versions"] == fields()["prompt_versions"]
    assert isinstance(d["prompt_set_hash"], str) and d["prompt_set_hash"]
    text = report.render_markdown(rep)
    assert text == md.read_text(encoding="utf-8")
    for needle in (RUN_ID, "general", "claude_subscription", "atomize@1", STARTED):
        assert needle in text


def test_R21_per_stage_table_lists_requested_and_served_models_calls_and_repairs(cfg):
    ledger = UsageLedger()
    ledger.record(row(stage="atomize", attempt_no=1, outcome="invalid"))
    ledger.record(row(stage="atomize", attempt_no=2, outcome="ok", model_served="claude-sonnet-5-20260101"))
    ledger.record(row(stage="edges", prompt_version="edges@1"))
    _, md, js = build_and_write(cfg, ledger)
    d = json.loads(js.read_text(encoding="utf-8"))
    st = d["stages"]
    assert set(st) == {"describe", "atomize", "edges", "dedup"}
    assert st["atomize"]["model_requested"] == "claude-sonnet-5"
    assert st["atomize"]["model_served"] == "claude-sonnet-5-20260101"
    assert st["atomize"]["calls"] == 1 and st["atomize"]["repair_attempts"] == 1
    assert st["describe"]["model_requested"] == cfg.llm.model_for("describe") and st["describe"]["calls"] == 0
    assert st["describe"]["model_served"] is None
    text = md.read_text(encoding="utf-8")
    assert "claude-sonnet-5-20260101" in text and "describe" in text and "dedup" in text


def test_R21_provider_override_is_recorded(cfg, sub_ledger):
    _, _, js = build_and_write(cfg, sub_ledger, provider_overridden=True)
    assert json.loads(js.read_text(encoding="utf-8"))["provider_overridden"] is True


# ---------------------------------------------------------------- inputs


def test_R1_inputs_table_lists_every_input_with_outcome_and_reason(cfg, sub_ledger):
    _, md, js = build_and_write(cfg, sub_ledger)
    d = json.loads(js.read_text(encoding="utf-8"))
    assert [(i["name"], i["status"]) for i in d["inputs"]] == [("a.md", "ingested"), ("legacy.xls", "unsupported"), ("b.md", "deferred")]
    assert d["inputs"][2]["reason"] == "atomize: StageOutputInvalid after 3 attempts"
    text = md.read_text(encoding="utf-8")
    assert "legacy.xls" in text and "unsupported" in text and "deferred" in text and "StageOutputInvalid" in text


# ------------------------------------------------------- counts / rejects


def test_R15_counts_rejections_and_review_queues_are_reported(cfg, sub_ledger):
    _, md, js = build_and_write(cfg, sub_ledger)
    d = json.loads(js.read_text(encoding="utf-8"))
    assert d["counts"]["nodes"] == {"created": 3, "updated": 0, "total": 3}
    assert d["counts"]["edges_by_type"] == {"related_to": 1, "part_of": 1}
    assert d["rejected_edges"] == fields()["rejected_edges"]
    assert d["review_queues"] == {"duplicates": 1, "grounding": 0, "not_adjudicated": 2}
    text = md.read_text(encoding="utf-8")
    assert "related_to without relevance" in text
    assert "roster truncated" in text


# -------------------------------------------------------------- cost (D28)


def test_D28_subscription_cost_reads_n_a_with_estimate_never_zero_dollars(cfg, sub_ledger):
    _, md, js = build_and_write(cfg, sub_ledger)
    d = json.loads(js.read_text(encoding="utf-8"))
    total = d["usage"]["total"]
    assert total["cost_usd"] is None
    assert "n/a (subscription)" in total["cost_display"]
    assert total["input_tokens"] == 2000 and total["output_tokens"] == 400 and total["calls"] == 2
    text = md.read_text(encoding="utf-8")
    assert "n/a (subscription)" in text
    assert "equivalent API list price (SDK estimate)" in text and "$0.41" in text
    assert "$0.00" not in text


def test_R21_openrouter_cost_is_the_server_reported_sum(make_project):
    cfg = load_config(make_project({"llm.provider": "openrouter"}, env_text=f"OPENROUTER_API_KEY={FAKE_OR_KEY}\n", subdir="or"))
    ledger = UsageLedger()
    ledger.record(row(provider="openrouter", model_requested="anthropic/claude-sonnet-5", model_served="anthropic/claude-sonnet-5", cost_usd=0.30, cost_estimate_usd=None))
    ledger.record(row(stage="edges", provider="openrouter", model_requested="anthropic/claude-sonnet-5", model_served="anthropic/claude-sonnet-5", cost_usd=0.12, cost_estimate_usd=None))
    _, md, js = build_and_write(cfg, ledger)
    d = json.loads(js.read_text(encoding="utf-8"))
    assert d["usage"]["total"]["cost_usd"] == pytest.approx(0.42)
    text = md.read_text(encoding="utf-8")
    assert "$0.42" in text and "n/a" not in d["usage"]["total"]["cost_display"]
    assert "soft_budget" in json.dumps(d["usage"]).lower() or "budget" in text.lower()


def test_R15_empty_ledger_reports_zero_usage_and_no_price_not_zero_dollars(cfg):
    _, md, js = build_and_write(cfg, UsageLedger(), mode="dry-run")
    d = json.loads(js.read_text(encoding="utf-8"))
    assert d["dry_run"] is True
    assert d["usage"]["total"]["calls"] == 0 and d["usage"]["total"]["cost_usd"] is None
    assert "$0.00" not in md.read_text(encoding="utf-8")


# -------------------------------------------------------------- redaction


def test_R18_write_refuses_to_emit_a_credential_and_writes_nothing(make_project):
    from kg import report

    cfg = load_config(make_project({"llm.provider": "openrouter"}, env_text=f"OPENROUTER_API_KEY={FAKE_OR_KEY}\n", subdir="or2"))
    rep = report.build(cfg, ledger=UsageLedger(), **fields(warnings=[f"debug: authorization bearer {FAKE_OR_KEY}"]))
    with pytest.raises(report.ReportRedactionError):
        report.write(rep, cfg)
    assert not (cfg.sandbox.root("graph") / "_reports").exists() or list((cfg.sandbox.root("graph") / "_reports").iterdir()) == []


def test_R18_clean_report_on_a_keyed_provider_contains_no_credential(make_project):
    cfg = load_config(make_project({"llm.provider": "openrouter"}, env_text=f"OPENROUTER_API_KEY={FAKE_OR_KEY}\n", subdir="or3"))
    ledger = UsageLedger()
    ledger.record(row(provider="openrouter", model_requested="anthropic/claude-sonnet-5", cost_usd=0.01, cost_estimate_usd=None))
    _, md, js = build_and_write(cfg, ledger)
    assert FAKE_OR_KEY not in md.read_text(encoding="utf-8")
    assert FAKE_OR_KEY not in js.read_text(encoding="utf-8")
    assert "TESTONLY" not in js.read_text(encoding="utf-8")


# ------------------------------------------------------------ determinism


def test_R15_report_is_deterministic_for_identical_inputs(cfg, sub_ledger):
    from kg import report

    r1 = report.build(cfg, ledger=sub_ledger, **fields())
    r2 = report.build(cfg, ledger=sub_ledger, **fields())
    assert report.render_markdown(r1) == report.render_markdown(r2)
    assert report.to_dict(r1) == report.to_dict(r2)


def test_R15_write_overwrites_the_same_run_id_rather_than_appending(cfg, sub_ledger):
    _, md, _ = build_and_write(cfg, sub_ledger)
    first = md.read_text(encoding="utf-8")
    _, md2, _ = build_and_write(cfg, sub_ledger)
    assert md2 == md and md.read_text(encoding="utf-8") == first
    assert len(list(md.parent.iterdir())) == 2
