"""WU5 — `kg.stability`: two clean ingests of the same inbox, node-set agreement (PRD R16, §8(b); DESIGN §15, D11, R21-l).

Interface (chosen, see wu45_fixtures docstring):
  STABILITY_THRESHOLD = 0.70                               # §8(b); no `stability:` key exists in kg.yaml §3.2
  run(cfg, *, call_stage, now, keep=False, threshold=STABILITY_THRESHOLD, fetch=None) -> StabilityResult
  leg_config(cfg, run_id, leg) -> Config                    # paths under <runs>/stability-<run_id>/<leg>/…
  jaccard(a, b) ; overlaps(a, b) -> (|A∩B|/|A|, |A∩B|/|B|) ; match_nodes(a_titles, b_titles) -> NodeMatch
  relevance_delta_histogram(deltas) -> {"0-5","6-10","11-20","21-40","41-100": int}
Legs run sequentially, `a` then `b`, through `kg.pipeline.run` with the SAME scripted call_stage;
leg identity is observable through the `cfg=` the pipeline passes to call_stage (§7.1).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from kg.config import STAGES, load_config
from wu3_fixtures import A_MD, ScriptedCallStage, default_handlers, general_edges_handler, ids_in, smart_atomize, write_inbox
from wu45_fixtures import DELTA_BUCKETS, NOW, RUN_ID, snapshot_tree


@pytest.fixture
def cfg(make_project):
    return load_config(make_project())


def leg_of(call: dict) -> str:
    """Which leg a scripted call belongs to, from the derived cfg the pipeline passed."""
    c = call["extra"].get("cfg")
    assert c is not None, "pipeline must pass cfg= to call_stage (DESIGN §7.1)"
    parts = Path(c.sandbox.root("graph")).parts
    return "b" if "b" in parts[-2:] else "a"


def drift_atomize(rename: dict[str, str]):
    """smart_atomize, but on leg b some titles are renamed (simulated model drift)."""

    def _h(call):
        out = smart_atomize(call)
        if leg_of(call) == "b":
            for n in out["nodes"]:
                n["title"] = rename.get(n["title"], n["title"])
        return out

    return _h


def relevance_by_leg(slug: str, rel_a: int, rel_b: int):
    def _h(call):
        rel = rel_b if leg_of(call) == "b" else rel_a
        return general_edges_handler(slug, relevance=rel)(call)

    return _h


def run_stability(cfg, fake, **kw):
    from kg.stability import run

    return run(cfg, call_stage=fake, now=kw.pop("now", NOW), **kw)


def report(res) -> dict:
    return json.loads(Path(res.report_json).read_text(encoding="utf-8"))


# ------------------------------------------------------------- pure maths


def test_R20_jaccard_of_identical_sets_is_one_and_of_disjoint_sets_is_zero():
    from kg.stability import jaccard

    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert jaccard({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(2 / 4)
    assert jaccard(set(), set()) == 1.0, "two empty node sets agree perfectly (guarded elsewhere as invalid)"
    assert jaccard(set(), {"a"}) == 0.0


def test_R20_overlaps_are_per_run_fractions():
    from kg.stability import overlaps

    fa, fb = overlaps({"a", "b", "c", "d"}, {"c", "d"})
    assert (fa, fb) == (0.5, 1.0)
    assert overlaps(set(), {"x"}) == (0.0, 0.0)


def test_R16_match_nodes_compares_normalised_titles_and_lists_the_diff():
    from kg.stability import match_nodes

    m = match_nodes(["The Sample Space", "Events", "Probability measure"], ["sample space", "Event", "Random Variable"])
    assert m.common == ["event", "sample space"]
    assert m.only_in_a == ["probability measure"]
    assert m.only_in_b == ["random variable"]


def test_R16_relevance_delta_histogram_buckets():
    from kg.stability import relevance_delta_histogram

    h = relevance_delta_histogram([0, 5, 6, 10, 11, 20, 21, 40, 41, 100, -12])
    assert tuple(h) == DELTA_BUCKETS
    assert h == {"0-5": 2, "6-10": 2, "11-20": 3, "21-40": 2, "41-100": 2}
    assert relevance_delta_histogram([]) == {b: 0 for b in DELTA_BUCKETS}


def test_R20_threshold_default_is_the_prd_bar():
    from kg.stability import STABILITY_THRESHOLD

    assert STABILITY_THRESHOLD == 0.70


# ------------------------------------------------------------ leg configs


def test_R21_leg_configs_share_provider_and_models_and_differ_only_in_paths(cfg):
    from kg.stability import leg_config

    a, b = leg_config(cfg, RUN_ID, "a"), leg_config(cfg, RUN_ID, "b")
    assert a.llm == cfg.llm == b.llm, "identical provider + per-stage model map (§15)"
    assert a.corpus == cfg.corpus and a.chunking == cfg.chunking and a.dedup == cfg.dedup
    root = cfg.sandbox.root("runs") / f"stability-{RUN_ID}"
    for leg, c in (("a", a), ("b", b)):
        for kind in ("inbox", "converted", "graph", "processed", "runs"):
            assert c.sandbox.root(kind) == root / leg / kind, (leg, kind)
    assert a.sandbox.root("graph") != b.sandbox.root("graph")
    assert a.project_root == cfg.project_root


def test_R21_leg_config_rejects_unknown_leg(cfg):
    from kg.stability import leg_config

    with pytest.raises(ValueError):
        leg_config(cfg, RUN_ID, "c")


# -------------------------------------------------------- orchestration


def test_R16_same_script_twice_gives_agreement_one_and_passes(cfg):
    write_inbox(cfg, {"a.md": A_MD})
    fake = ScriptedCallStage(default_handlers(cfg, edges=general_edges_handler(cfg.corpus.slug)), cfg=cfg)
    res = run_stability(cfg, fake, keep=True)

    assert res.run_id == RUN_ID
    assert res.jaccard == 1.0 and res.overlap_a == 1.0 and res.overlap_b == 1.0
    assert res.passed is True and res.valid is True and res.invalid_reasons == []
    assert res.only_in_a == [] and res.only_in_b == []
    assert sorted(res.common) == ["event", "probability measure", "sample space"]
    assert res.threshold == 0.70
    # two full ingests through the boundary
    assert len(fake.calls_for("atomize")) == 2 and len(fake.calls_for("edges")) == 2
    assert {leg_of(c) for c in fake.calls} == {"a", "b"}
    legs = [leg_of(c) for c in fake.calls]
    assert legs == sorted(legs), "leg a runs to completion before leg b starts"


def test_R16_legs_live_under_runs_stability_ts_and_each_has_its_own_graph(cfg):
    write_inbox(cfg, {"a.md": A_MD})
    res = run_stability(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg), keep=True)
    root = cfg.sandbox.root("runs") / f"stability-{RUN_ID}"
    assert Path(res.root) == root
    for leg in ("a", "b"):
        nodes = sorted(p.name for p in (root / leg / "graph" / "nodes").glob("*.md"))
        assert nodes == ["sample-0001-sample-space.md", "sample-0002-event.md", "sample-0003-probability-measure.md"], leg
        assert (root / leg / "graph" / "_registry.yaml").is_file()
        assert (root / leg / "processed" / "a.md").is_file(), "the COPY was ingested and moved inside the leg"
        assert (root / leg / "graph" / "_reports").is_dir(), "each leg writes its own run report (R15)"
    assert Path(res.report_md) == root / "stability-report.md"
    assert Path(res.report_json) == root / "stability-report.json"


def test_R5_original_inbox_and_project_graph_are_untouched(cfg):
    write_inbox(cfg, {"a.md": A_MD, "urls.txt": "# no urls in this fixture\n"})
    before_inbox = snapshot_tree(cfg.sandbox.root("inbox"))
    before_graph = snapshot_tree(cfg.sandbox.root("graph"))
    res = run_stability(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg), keep=True)
    assert snapshot_tree(cfg.sandbox.root("inbox")) == before_inbox, "originals are copied, never moved"
    assert snapshot_tree(cfg.sandbox.root("graph")) == before_graph, "project graph is not written by a stability run"
    assert list(cfg.sandbox.root("processed").iterdir()) == []
    root = Path(res.root)
    leg_a_files = {p.name for p in (root / "a" / "inbox").iterdir()} | {p.name for p in (root / "a" / "processed").iterdir()}
    assert "urls.txt" in leg_a_files, "urls.txt is copied into each leg"


def test_R16_legs_never_reuse_converted_output(cfg):
    """Each leg converts from its own inbox copy: nothing pre-existing in <converted> is consulted."""
    write_inbox(cfg, {"a.md": A_MD})
    stale = cfg.sandbox.root("converted") / "a.md"
    stale.write_text("# STALE\n\nstale converted text must not be ingested.\n", encoding="utf-8")
    res = run_stability(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg), keep=True)
    assert stale.read_text(encoding="utf-8").startswith("# STALE"), "project converted/ untouched"
    root = Path(res.root)
    for leg in ("a", "b"):
        assert (root / leg / "converted" / "a.md").is_file()
        assert "STALE" not in (root / leg / "converted" / "a.md").read_text(encoding="utf-8")
    assert sorted(res.common) == ["event", "probability measure", "sample space"]


def test_R16_scripted_drift_yields_the_exact_jaccard_and_diff_lists(cfg):
    write_inbox(cfg, {"a.md": A_MD})
    handlers = default_handlers(cfg, atomize=drift_atomize({"Probability Measure": "Probability Function"}), edges=general_edges_handler(cfg.corpus.slug))
    res = run_stability(cfg, ScriptedCallStage(handlers, cfg=cfg), keep=True)

    # A = {sample space, event, probability measure}; B = {sample space, event, probability function}
    assert res.jaccard == pytest.approx(2 / 4)
    assert res.overlap_a == pytest.approx(2 / 3) and res.overlap_b == pytest.approx(2 / 3)
    assert res.passed is False, "0.5 < 0.70"
    assert res.valid is True, "a low score is a result, not an invalid comparison"
    assert [d["title"] for d in res.only_in_a] == ["probability measure"]
    assert [d["title"] for d in res.only_in_b] == ["probability function"]
    assert res.only_in_a[0]["locators"] and res.only_in_a[0]["locators"][0].startswith("a.md#"), "diff rows carry locators (§15)"
    assert 0.0 <= res.fuzzy_jaccard <= 1.0 and res.fuzzy_jaccard >= res.jaccard


def test_R16_threshold_is_a_parameter(cfg):
    write_inbox(cfg, {"a.md": A_MD})
    handlers = default_handlers(cfg, atomize=drift_atomize({"Probability Measure": "Probability Function"}))
    res = run_stability(cfg, ScriptedCallStage(handlers, cfg=cfg), threshold=0.5)
    assert res.jaccard == pytest.approx(0.5) and res.passed is True and res.threshold == 0.5


# -------------------------------------------------- relevance deltas (R16)


def test_R16_relevance_delta_distribution_is_reported_for_matched_related_to_edges(cfg):
    write_inbox(cfg, {"a.md": A_MD})
    handlers = default_handlers(cfg, edges=relevance_by_leg(cfg.corpus.slug, 72, 60))
    res = run_stability(cfg, ScriptedCallStage(handlers, cfg=cfg))
    d = res.relevance_deltas
    assert d.n == 1
    assert d.histogram == {"0-5": 0, "6-10": 0, "11-20": 1, "21-40": 0, "41-100": 0}
    assert d.mean_abs == 12 and d.median_abs == 12
    assert res.passed is True, "relevance variance is observed, never gated (PRD §8(b))"


def test_R16_relevance_deltas_are_zero_when_scores_agree(cfg):
    write_inbox(cfg, {"a.md": A_MD})
    res = run_stability(cfg, ScriptedCallStage(default_handlers(cfg, edges=general_edges_handler(cfg.corpus.slug, relevance=72)), cfg=cfg))
    assert res.relevance_deltas.n == 1 and res.relevance_deltas.histogram["0-5"] == 1
    assert res.relevance_deltas.mean_abs == 0


def test_R16_relevance_deltas_are_empty_when_no_related_to_edges_match(cfg):
    write_inbox(cfg, {"a.md": A_MD})
    res = run_stability(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg))  # no_edges
    assert res.relevance_deltas.n == 0
    assert res.relevance_deltas.mean_abs is None and res.relevance_deltas.median_abs is None
    assert sum(res.relevance_deltas.histogram.values()) == 0


# ---------------------------------------------------------- validity guard


def test_R21_served_model_mismatch_between_legs_marks_the_comparison_invalid(cfg):
    write_inbox(cfg, {"a.md": A_MD})
    inner = ScriptedCallStage(default_handlers(cfg), cfg=cfg)

    def call_stage(stage, system, user_text, schema, images=None, **kw):
        # Record the ledger row ourselves so the leg report's per-stage table (§14) sees the drift too.
        ledger = kw.pop("ledger", None)
        r = inner(stage, system, user_text, schema, images, **kw)
        if leg_of(inner.calls[-1]) == "b":
            r.model_served = f"{r.model_requested}-DIFFERENT-BUILD"
        if ledger is not None:
            ScriptedCallStage._record(ledger, stage, r.provider, r.model_requested, r.model_served, kw.get("prompt_version"))
        return r

    res = run_stability(cfg, call_stage)
    assert res.jaccard == 1.0, "scores are still computed and printed"
    assert res.valid is False
    assert any("model" in r.lower() for r in res.invalid_reasons)
    rj = report(res)
    assert rj["valid"] is False
    assert rj["legs"]["a"]["models"]["atomize"]["served"] != rj["legs"]["b"]["models"]["atomize"]["served"]
    assert "INVALID" in Path(res.report_md).read_text(encoding="utf-8")


def test_R21_deferred_file_in_one_leg_marks_the_comparison_invalid_and_names_the_file(cfg):
    def edges_fail_on_b(call):
        if leg_of(call) == "b":
            raise RuntimeError("provider unavailable")
        return {"edges": []}

    write_inbox(cfg, {"a.md": A_MD})
    res = run_stability(cfg, ScriptedCallStage(default_handlers(cfg, edges=edges_fail_on_b), cfg=cfg))
    assert res.valid is False
    assert res.legs["b"].deferred == ["a.md"] and res.legs["a"].deferred == []
    assert any("a.md" in r for r in res.invalid_reasons)
    assert Path(res.report_md).is_file(), "report still written"


def test_R16_empty_inbox_is_invalid_not_a_vacuous_pass(cfg):
    res = run_stability(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg))
    assert res.valid is False
    assert res.legs["a"].nodes == 0 and res.legs["b"].nodes == 0
    assert any("no nodes" in r.lower() or "empty" in r.lower() for r in res.invalid_reasons)


# ------------------------------------------------------------- report


def test_R16_R21_report_json_records_scores_diff_and_provider_models_per_leg(cfg):
    write_inbox(cfg, {"a.md": A_MD})
    handlers = default_handlers(cfg, atomize=drift_atomize({"Event": "Outcome Subset"}), edges=relevance_by_leg(cfg.corpus.slug, 80, 50))
    res = run_stability(cfg, ScriptedCallStage(handlers, cfg=cfg))
    rj = report(res)
    assert rj["run_id"] == RUN_ID
    assert rj["threshold"] == 0.70
    assert rj["jaccard"] == pytest.approx(2 / 4) and rj["passed"] is False and rj["valid"] is True
    assert rj["overlap_a"] == pytest.approx(2 / 3) and rj["overlap_b"] == pytest.approx(2 / 3)
    assert rj["counts"] == {"a": 3, "b": 3, "common": 2}
    assert [d["title"] for d in rj["only_in_a"]] == ["event"]
    assert [d["title"] for d in rj["only_in_b"]] == ["outcome subset"]
    # the matched related_to edge is sample space <-> event in leg a but the event node was renamed → no match
    assert rj["relevance_deltas"]["n"] == 0
    for leg in ("a", "b"):
        L = rj["legs"][leg]
        assert L["provider"] == cfg.llm.provider
        assert set(L["models"]) == set(STAGES)
        for stage in ("atomize", "edges"):
            assert L["models"][stage]["requested"] == cfg.llm.model_for(stage)
            assert L["models"][stage]["served"] == f"{cfg.llm.model_for(stage)}-served"
        assert L["nodes"] == 3 and L["deferred"] == []
        assert Path(L["report_json"]).is_file()


def test_R16_report_markdown_prints_the_score_the_diff_and_the_provider(cfg):
    write_inbox(cfg, {"a.md": A_MD})
    handlers = default_handlers(cfg, atomize=drift_atomize({"Probability Measure": "Probability Function"}))
    res = run_stability(cfg, ScriptedCallStage(handlers, cfg=cfg))
    md = Path(res.report_md).read_text(encoding="utf-8")
    assert "0.50" in md
    assert "probability measure" in md and "probability function" in md
    assert cfg.llm.provider in md and cfg.llm.model_for("atomize") in md
    assert "0-5" in md and "41-100" in md, "relevance-delta histogram buckets"
    assert "INVALID" not in md


def test_R16_report_json_and_result_agree(cfg):
    write_inbox(cfg, {"a.md": A_MD})
    res = run_stability(cfg, ScriptedCallStage(default_handlers(cfg, edges=general_edges_handler(cfg.corpus.slug)), cfg=cfg))
    rj = report(res)
    assert rj["jaccard"] == res.jaccard and rj["passed"] == res.passed and rj["valid"] == res.valid
    assert rj["relevance_deltas"]["histogram"] == res.relevance_deltas.histogram


# -------------------------------------------------------------- --keep


def test_S18_legs_are_removed_by_default_and_kept_with_keep(cfg):
    write_inbox(cfg, {"a.md": A_MD})
    res = run_stability(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg))
    root = Path(res.root)
    assert Path(res.report_md).is_file() and Path(res.report_json).is_file()
    assert not (root / "a").exists() and not (root / "b").exists()
    assert (cfg.sandbox.root("inbox") / "a.md").is_file(), "removing leg copies never touches originals"

    res2 = run_stability(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg), keep=True, now=NOW.replace(day=26))
    root2 = Path(res2.root)
    assert (root2 / "a" / "graph").is_dir() and (root2 / "b" / "graph").is_dir()
    assert root2 != root


def test_R16_deterministic_across_two_stability_runs_with_the_same_script(cfg):
    write_inbox(cfg, {"a.md": A_MD})
    r1 = run_stability(cfg, ScriptedCallStage(default_handlers(cfg, edges=general_edges_handler(cfg.corpus.slug)), cfg=cfg))
    r2 = run_stability(cfg, ScriptedCallStage(default_handlers(cfg, edges=general_edges_handler(cfg.corpus.slug)), cfg=cfg), now=NOW.replace(hour=11))
    a, b = report(r1), report(r2)
    for k in ("jaccard", "overlap_a", "overlap_b", "passed", "valid", "counts", "only_in_a", "only_in_b", "relevance_deltas"):
        assert a[k] == b[k], k
    assert not math.isnan(a["jaccard"])
