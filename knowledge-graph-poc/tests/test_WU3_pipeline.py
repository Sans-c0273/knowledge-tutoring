"""WU3 — `kg.pipeline.run` end-to-end on a tiny inbox with a scripted call_stage (S0–S8).

Spec: PRD R1 (every input reported), R3/R4 (source + quote per node), R5 (sources moved, never
deleted; counts constant), R7 (permanent ids), R8 (schema per run), R9 (`origin` education only),
R10, R11 (idempotent wikilinks), R12 (duplicates queued or same_as), R13 (DAG), R15 (report every
run incl. dry-run), R21 (malformed output exhausting repair → file deferred, never written);
DESIGN §6 ("unit of model work is one source file", file-level deferral), §8.5–8.6, §11, §14, §23.

Interface (chosen): run(cfg, *, call_stage, now: datetime, dry_run=False, write_same_as=None, fetch=None)
-> RunResult(run_id, status, inputs: list[.name .status .reason], nodes_written, report_md, report_json);
run_id_for(now) == "YYYY-MM-DDTHH-MM-SSZ". Timestamps are injected — no wall clock inside the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from kg.config import load_config
from kg.schemas import EducationEdgesOutput, GeneralEdgesOutput
from wu3_fixtures import (
    A_MD,
    B_MD,
    DUP_MD,
    EDU_MD,
    NOW,
    RUN_ID,
    ScriptedCallStage,
    default_handlers,
    education_cycle_handler,
    general_edges_handler,
    load_nodes,
    node_files,
    smart_atomize,
    snapshot_tree,
    write_inbox,
    ws,
)


@pytest.fixture
def cfg(make_project):
    return load_config(make_project())


@pytest.fixture
def cfg_edu(make_project):
    return load_config(make_project({"corpus.schema": "education"}, subdir="edu"))


def run_pipeline(cfg, fake, **kw):
    from kg.pipeline import run

    return run(cfg, call_stage=fake, now=kw.pop("now", NOW), **kw)


def graph(cfg) -> Path:
    return cfg.sandbox.root("graph")


def report_json(res) -> dict:
    return json.loads(Path(res.report_json).read_text(encoding="utf-8"))


# ------------------------------------------------------------- happy path


def test_R1_R5_R7_end_to_end_general_run(cfg):
    write_inbox(cfg, {"a.md": A_MD, "b.md": B_MD})
    fake = ScriptedCallStage(default_handlers(cfg, edges=general_edges_handler(cfg.corpus.slug)), cfg=cfg)
    res = run_pipeline(cfg, fake)

    from kg.pipeline import run_id_for

    assert run_id_for(NOW) == RUN_ID and res.run_id == RUN_ID
    # S3/S5 once per chunk; no describe (no images); every model stage went through the boundary
    assert len(fake.calls_for("atomize")) == 2
    assert len(fake.calls_for("edges")) == 2
    assert fake.calls_for("describe") == []

    # S8: converted/ inspectable (R2), nodes written, registry + index present
    assert (cfg.sandbox.root("converted") / "a.md").is_file() and (cfg.sandbox.root("converted") / "a.units.json").is_file()
    files = node_files(cfg)
    assert len(files) == 5, [p.name for p in files]
    assert sorted(p.name for p in files) == [
        "sample-0001-sample-space.md",
        "sample-0002-event.md",
        "sample-0003-probability-measure.md",
        "sample-0004-random-variable.md",
        "sample-0005-expectation.md",
    ]
    reg = yaml.safe_load((graph(cfg) / "_registry.yaml").read_text(encoding="utf-8"))
    assert reg["corpus"] == "sample" and reg["schema"] == "general" and reg["next_seq"] == 6
    assert [n["id"] for n in reg["nodes"]] == [f"sample-000{i}" for i in range(1, 6)]
    assert all(n["status"] == "active" for n in reg["nodes"])

    # R5: moved, not deleted
    inbox, processed = cfg.sandbox.root("inbox"), cfg.sandbox.root("processed")
    assert sorted(p.name for p in inbox.iterdir()) == []
    assert sorted(p.name for p in processed.iterdir()) == ["a.md", "b.md"]
    assert (processed / "a.md").read_text(encoding="utf-8") == A_MD

    # report for the run (R15)
    assert Path(res.report_md).is_file() and Path(res.report_json).is_file()
    assert Path(res.report_md).parent == graph(cfg) / "_reports"
    assert Path(res.report_md).name == f"run-{RUN_ID}.md" and Path(res.report_json).name == f"run-{RUN_ID}.json"
    rj = report_json(res)
    assert rj["run_id"] == RUN_ID and rj["schema"] == "general" and rj["dry_run"] is False
    assert {i["name"]: i["status"] for i in rj["inputs"]} == {"a.md": "ingested", "b.md": "ingested"}
    assert [i.status for i in res.inputs] == ["ingested", "ingested"]
    assert res.nodes_written == 5
    assert res.status == "ok" and rj["status"] == "ok"
    assert rj["counts"]["nodes"]["total"] == 5


def test_R5_file_counts_are_constant_across_inbox_plus_processed(cfg):
    write_inbox(cfg, {"a.md": A_MD, "b.md": B_MD, "archive.zip": "PK\x05\x06" + "\0" * 18})
    inbox, processed = cfg.sandbox.root("inbox"), cfg.sandbox.root("processed")
    before = len(list(inbox.iterdir())) + len(list(processed.iterdir()))
    res = run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg))
    after = len(list(inbox.iterdir())) + len(list(processed.iterdir()))
    assert before == after == 3
    assert (inbox / "archive.zip").exists(), "unsupported stays in the inbox (R1)"
    rj = report_json(res)
    assert rj["moves"]["before"] == {"inbox": 3, "processed": 0}
    assert rj["moves"]["after"] == {"inbox": 1, "processed": 2}


def test_R1_unsupported_input_is_listed_in_the_report_with_a_reason(cfg):
    write_inbox(cfg, {"a.md": A_MD, "legacy.xls": "\xd0\xcf\x11\xe0"})
    res = run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg))
    rows = {i["name"]: i for i in report_json(res)["inputs"]}
    assert rows["legacy.xls"]["status"] == "unsupported"
    assert rows["legacy.xls"]["reason"] and ".xls" in rows["legacy.xls"]["reason"]
    assert rows["a.md"]["status"] == "ingested"


def test_R3_R4_every_node_has_a_locator_and_a_quote_found_in_the_source(cfg):
    write_inbox(cfg, {"a.md": A_MD})
    run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg))
    processed_text = ws((cfg.sandbox.root("processed") / "a.md").read_text(encoding="utf-8"))
    nodes = load_nodes(cfg)
    assert len(nodes) == 3
    for n in nodes:
        assert n.sources, n.id
        for s in n.sources:
            assert s.locator.startswith("a.md#"), s.locator
            assert s.file == "a.md"
            assert s.quote and ws(s.quote) in processed_text, (n.id, s.quote)
        assert n.review == []


def test_R7_nodes_carry_run_metadata_and_prompt_version(cfg):
    write_inbox(cfg, {"a.md": A_MD})
    run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg))
    for n in load_nodes(cfg):
        assert n.run == RUN_ID
        assert n.created == "2026-08-25" and n.updated == "2026-08-25"
        assert n.prompt_version.startswith("atomize@")
        assert n.provider == cfg.llm.provider
        assert n.model == f"{cfg.llm.model_for('atomize')}-served", "model_served when known (§8.2)"
        assert n.schema == "general"


# ----------------------------------------------------------------- origin


def test_R9_general_run_nodes_have_no_origin_field(cfg):
    write_inbox(cfg, {"a.md": A_MD})
    run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg))
    for p in node_files(cfg):
        fm = yaml.safe_load(p.read_text(encoding="utf-8").split("---")[1])
        assert "origin" not in fm, p.name


def test_R9_education_run_nodes_all_carry_origin_course_material(cfg_edu):
    write_inbox(cfg_edu, {"e.md": EDU_MD})
    run_pipeline(cfg_edu, ScriptedCallStage(default_handlers(cfg_edu), cfg=cfg_edu))
    files = node_files(cfg_edu)
    assert len(files) == 3
    for p in files:
        fm = yaml.safe_load(p.read_text(encoding="utf-8").split("---")[1])
        assert fm["origin"] == "course_material", p.name
        assert fm["schema"] == "education"


# ----------------------------------------------------------------- schema


def test_R8_schema_from_config_selects_the_edges_wire_model(cfg, cfg_edu):
    write_inbox(cfg, {"a.md": A_MD})
    fake = ScriptedCallStage(default_handlers(cfg), cfg=cfg)
    res = run_pipeline(cfg, fake)
    assert {c["schema"] for c in fake.calls_for("edges")} == {GeneralEdgesOutput}
    assert report_json(res)["schema"] == "general"

    write_inbox(cfg_edu, {"e.md": EDU_MD})
    fake_e = ScriptedCallStage(default_handlers(cfg_edu), cfg=cfg_edu)
    res_e = run_pipeline(cfg_edu, fake_e)
    assert {c["schema"] for c in fake_e.calls_for("edges")} == {EducationEdgesOutput}
    assert report_json(res_e)["schema"] == "education"


# ---------------------------------------------------------- edges on disk


def test_R10_related_to_edges_carry_relevance_and_are_mirrored_on_both_nodes(cfg):
    write_inbox(cfg, {"a.md": A_MD})
    run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg, edges=general_edges_handler(cfg.corpus.slug, relevance=72)), cfg=cfg))
    by_id = {n.id: n for n in load_nodes(cfg)}
    a, b, c = by_id["sample-0001"], by_id["sample-0002"], by_id["sample-0003"]
    assert [(e.type, e.target, e.relevance) for e in a.edges if e.type == "related_to"] == [("related_to", "sample-0002", 72)]
    assert [(e.type, e.target, e.relevance) for e in b.edges if e.type == "related_to"] == [("related_to", "sample-0001", 72)]
    assert [(e.type, e.target, e.relevance) for e in c.edges] == [("part_of", "sample-0001", None)], "directed edges are stored on the source only"
    assert all(e.relevance is None for n in by_id.values() for e in n.edges if e.type != "related_to")
    text = (graph(cfg) / "nodes" / "sample-0001-sample-space.md").read_text(encoding="utf-8")
    assert "- [[sample-0002-event|Event]] — relevance 72" in text


def test_R11_definition_prose_is_woven_with_wikilinks_to_referenced_nodes(cfg):
    write_inbox(cfg, {"a.md": A_MD})
    run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg))
    event = (graph(cfg) / "nodes" / "sample-0002-event.md").read_text(encoding="utf-8")
    definition = event.split("## Definition")[1].split("## Relations")[0]
    assert definition.count("[[sample-0001-sample-space|sample space]]") == 1
    assert "[[[[" not in event


def test_R11_running_ingest_again_over_the_existing_graph_is_byte_stable(cfg):
    write_inbox(cfg, {"a.md": A_MD})
    run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg, edges=general_edges_handler(cfg.corpus.slug)), cfg=cfg))
    first = snapshot_tree(graph(cfg) / "nodes")
    assert first
    run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg))  # empty inbox now
    second = snapshot_tree(graph(cfg) / "nodes")
    assert second == first
    assert not any(b"[[[[" in body for body in second.values())


def test_R11_new_file_referencing_existing_titles_weaves_links_and_keeps_old_nodes_stable(cfg):
    write_inbox(cfg, {"a.md": A_MD})
    run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg))
    before = snapshot_tree(graph(cfg) / "nodes")
    write_inbox(cfg, {"b.md": B_MD})  # "Random Variable" definition mentions "outcome"; "Expectation" mentions "random variable"
    run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg))
    after = snapshot_tree(graph(cfg) / "nodes")
    for name, body in before.items():
        assert after[name] == body, f"{name} changed although its file was not re-ingested"
    exp = (graph(cfg) / "nodes" / "sample-0005-expectation.md").read_text(encoding="utf-8")
    assert "[[sample-0004-random-variable|random variable]]" in exp
    assert "[[[[" not in exp


# ------------------------------------------------------------ duplicates


def test_R12_planted_near_duplicate_is_queued_by_default(cfg):
    write_inbox(cfg, {"a.md": A_MD, "dup.md": DUP_MD})
    fake = ScriptedCallStage(default_handlers(cfg), cfg=cfg)
    res = run_pipeline(cfg, fake)
    assert len(fake.calls_for("dedup")) == 1
    queue = graph(cfg) / "_review" / "duplicates.md"
    assert queue.is_file()
    text = queue.read_text(encoding="utf-8")
    assert "sample-0001" in text and "sample-0004" in text
    assert "same concept, different wording" in text
    assert all(e.type != "same_as" for n in load_nodes(cfg) for e in n.edges)
    assert report_json(res)["review_queues"]["duplicates"] == 1


def test_R12_write_same_as_flag_writes_mirrored_same_as_edges_instead(cfg):
    write_inbox(cfg, {"a.md": A_MD, "dup.md": DUP_MD})
    run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg), write_same_as=True)
    by_id = {n.id: n for n in load_nodes(cfg)}
    assert [(e.type, e.target) for e in by_id["sample-0001"].edges if e.type == "same_as"] == [("same_as", "sample-0004")]
    assert [(e.type, e.target) for e in by_id["sample-0004"].edges if e.type == "same_as"] == [("same_as", "sample-0001")]
    queue = graph(cfg) / "_review" / "duplicates.md"
    assert not queue.exists() or "sample-0004" not in queue.read_text(encoding="utf-8"), "a written pair is not an open review item"


def test_R12_no_file_is_merged_or_deleted_by_dedup(cfg):
    write_inbox(cfg, {"a.md": A_MD, "dup.md": DUP_MD})
    run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg), write_same_as=True)
    assert len(node_files(cfg)) == 4
    assert sorted(p.name for p in cfg.sandbox.root("processed").iterdir()) == ["a.md", "dup.md"]


# ------------------------------------------------------------------- DAG


def test_R13_planted_prerequisite_cycle_is_rejected_logged_and_the_disk_graph_is_acyclic(cfg_edu):
    write_inbox(cfg_edu, {"e.md": EDU_MD})
    res = run_pipeline(cfg_edu, ScriptedCallStage(default_handlers(cfg_edu, edges=education_cycle_handler(cfg_edu.corpus.slug)), cfg=cfg_edu))

    log = graph(cfg_edu) / "_review" / "rejected-edges.jsonl"
    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["gate"] == "dag" and rows[0]["type"] == "prerequisite_of" and rows[0]["reason"]

    nodes = load_nodes(cfg_edu)
    prereq = {n.id: [e.target for e in n.edges if e.type == "prerequisite_of"] for n in nodes}
    assert sum(len(v) for v in prereq.values()) == 2

    def acyclic(g):
        seen, stack = set(), set()

        def visit(v):
            if v in stack:
                return False
            if v in seen:
                return True
            seen.add(v)
            stack.add(v)
            ok = all(visit(w) for w in g.get(v, []))
            stack.discard(v)
            return ok

        return all(visit(v) for v in g)

    assert acyclic(prereq)
    rj = report_json(res)
    assert len(rj["rejected_edges"]) == 1 and rj["rejected_edges"][0]["gate"] == "dag"
    assert rj["counts"]["edges_by_type"]["prerequisite_of"] == 2


# ------------------------------------------------------- grounding queue


def test_R4_grounding_failure_is_queued_and_the_node_flagged_not_repaired(cfg):
    def bad_quote_atomize(call):
        out = smart_atomize(call)
        out["nodes"][0]["quotes"][0]["text"] = "this sentence appears nowhere in the source"
        return out

    write_inbox(cfg, {"a.md": A_MD})
    fake = ScriptedCallStage(default_handlers(cfg, atomize=bad_quote_atomize), cfg=cfg)
    res = run_pipeline(cfg, fake)
    assert len(fake.calls_for("atomize")) == 1, "grounding is never a repair trigger"
    nodes = load_nodes(cfg)
    flagged = [n for n in nodes if n.review == ["quote_not_found"]]
    assert len(flagged) == 1 and flagged[0].title == "Sample Space"
    grounding = graph(cfg) / "_review" / "grounding.md"
    assert grounding.is_file()
    text = grounding.read_text(encoding="utf-8")
    assert flagged[0].id in text and "quote_not_found" in text
    assert report_json(res)["review_queues"]["grounding"] == 1


# ------------------------------------------------------------- deferral


def test_R21_R5_stage_failure_defers_that_file_only_nothing_partial_written(cfg):
    def atomize_b_fails(call):
        if "b.md#" in call["user_text"]:
            raise RuntimeError("provider unavailable")
        return smart_atomize(call)

    write_inbox(cfg, {"a.md": A_MD, "b.md": B_MD})
    res = run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg, atomize=atomize_b_fails), cfg=cfg))

    inbox, processed = cfg.sandbox.root("inbox"), cfg.sandbox.root("processed")
    assert sorted(p.name for p in inbox.iterdir()) == ["b.md"]
    assert sorted(p.name for p in processed.iterdir()) == ["a.md"]
    nodes = load_nodes(cfg)
    assert len(nodes) == 3
    assert all(s.file == "a.md" for n in nodes for s in n.sources)
    reg = yaml.safe_load((graph(cfg) / "_registry.yaml").read_text(encoding="utf-8"))
    assert all(r["title"] not in ("Random Variable", "Expectation") for r in reg["nodes"])

    rows = {i["name"]: i for i in report_json(res)["inputs"]}
    assert rows["a.md"]["status"] == "ingested"
    assert rows["b.md"]["status"] == "deferred"
    assert "provider unavailable" in rows["b.md"]["reason"]
    assert res.status == "partial" and report_json(res)["status"] == "partial"


def test_R21_malformed_output_that_exhausts_repair_is_never_written(cfg):
    from kg.llm import StageOutputInvalid

    def make_invalid():
        try:
            return StageOutputInvalid(stage="atomize", attempts=3)
        except TypeError:
            return StageOutputInvalid("atomize", 3)

    def atomize_invalid(call):
        if "b.md#" in call["user_text"]:
            return make_invalid()
        return smart_atomize(call)

    write_inbox(cfg, {"a.md": A_MD, "b.md": B_MD})
    res = run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg, atomize=atomize_invalid), cfg=cfg))
    assert (cfg.sandbox.root("inbox") / "b.md").exists()
    assert not any("random-variable" in p.name or "expectation" in p.name for p in node_files(cfg))
    rows = {i["name"]: i for i in report_json(res)["inputs"]}
    assert rows["b.md"]["status"] == "deferred"


def test_R21_edges_stage_failure_also_defers_the_whole_file(cfg):
    def edges_fail(call):
        raise RuntimeError("edges exploded")

    write_inbox(cfg, {"a.md": A_MD})
    res = run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg, edges=edges_fail), cfg=cfg))
    assert node_files(cfg) == []
    assert (cfg.sandbox.root("inbox") / "a.md").exists()
    assert not (graph(cfg) / "_registry.yaml").exists() or yaml.safe_load((graph(cfg) / "_registry.yaml").read_text())["nodes"] == []
    assert report_json(res)["inputs"][0]["status"] == "deferred"
    assert Path(res.report_md).is_file(), "report written even when everything is deferred (R15)"


def test_R21_deferred_file_is_picked_up_by_the_next_run(cfg):
    calls = {"n": 0}

    def flaky(call):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return smart_atomize(call)

    write_inbox(cfg, {"a.md": A_MD})
    run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg, atomize=flaky), cfg=cfg))
    assert node_files(cfg) == []
    from wu3_fixtures import NOW_2

    run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg, atomize=flaky), cfg=cfg), now=NOW_2)
    assert len(node_files(cfg)) == 3
    assert (cfg.sandbox.root("processed") / "a.md").exists()


# -------------------------------------------------------------- dry run


def test_R15_dry_run_calls_no_model_writes_no_nodes_moves_nothing_but_writes_a_report(cfg):
    write_inbox(cfg, {"a.md": A_MD, "b.md": B_MD})
    inbox_before = snapshot_tree(cfg.sandbox.root("inbox"))
    fake = ScriptedCallStage(default_handlers(cfg), cfg=cfg)
    res = run_pipeline(cfg, fake, dry_run=True)
    assert fake.calls == []
    assert node_files(cfg) == []
    assert not (graph(cfg) / "_registry.yaml").exists()
    assert snapshot_tree(cfg.sandbox.root("inbox")) == inbox_before
    assert list(cfg.sandbox.root("processed").iterdir()) == []
    assert (cfg.sandbox.root("converted") / "a.md").is_file(), "dry-run still converts (D16)"
    assert Path(res.report_md).is_file() and Path(res.report_json).is_file()
    rj = report_json(res)
    assert rj["dry_run"] is True
    assert {i["name"]: i["status"] for i in rj["inputs"]} == {"a.md": "dry-run", "b.md": "dry-run"}
    assert rj["inputs"][0]["units"] >= 1 and rj["inputs"][0]["chunks"] >= 1


def test_R15_dry_run_with_an_empty_inbox_still_writes_a_report(cfg):
    res = run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg), dry_run=True)
    assert Path(res.report_json).is_file()
    assert report_json(res)["inputs"] == []


# ---------------------------------------------------------- index / report


def test_S8_index_json_mirrors_the_notes_on_disk(cfg):
    write_inbox(cfg, {"a.md": A_MD})
    run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg, edges=general_edges_handler(cfg.corpus.slug)), cfg=cfg))
    idx = json.loads((graph(cfg) / "_index.json").read_text(encoding="utf-8"))
    assert {n["id"] for n in idx["nodes"]} == {"sample-0001", "sample-0002", "sample-0003"}
    assert {(e["source"], e["target"], e["type"]) for e in idx["edges"]} == {("sample-0001", "sample-0002", "related_to"), ("sample-0003", "sample-0001", "part_of")}
    rel = next(e for e in idx["edges"] if e["type"] == "related_to")
    assert rel["relevance"] == 72
    assert idx["meta"]["schema"] == "general" and idx["meta"]["corpus"] == "sample" and idx["meta"]["run"] == RUN_ID
    node = next(n for n in idx["nodes"] if n["id"] == "sample-0002")
    assert "[[" not in node["definition_plain"]
    assert node["sources"][0]["locator"].startswith("a.md#")


def test_R15_R21_report_names_provider_and_per_stage_models_and_usage(cfg):
    write_inbox(cfg, {"a.md": A_MD})
    res = run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg))
    rj = report_json(res)
    assert rj["provider"] == "claude_subscription"
    for stage in ("describe", "atomize", "edges", "dedup"):
        assert rj["stages"][stage]["model_requested"] == cfg.llm.model_for(stage)
    assert rj["stages"]["atomize"]["model_served"] == f"{cfg.llm.model_for('atomize')}-served"
    assert rj["stages"]["atomize"]["calls"] == 1 and rj["stages"]["edges"]["calls"] == 1
    assert rj["stages"]["describe"]["calls"] == 0
    assert rj["usage"]["total"]["input_tokens"] == 200 and rj["usage"]["total"]["output_tokens"] == 40
    assert "n/a (subscription)" in rj["usage"]["total"]["cost_display"]
    md = Path(res.report_md).read_text(encoding="utf-8")
    assert "$0.00" not in md
    assert "claude_subscription" in md and cfg.llm.model_for("atomize") in md
    assert rj["prompt_versions"]["atomize"].startswith("atomize@")


def test_R18_report_never_contains_a_credential(make_project):
    from wu2_fakes import FAKE_OR_KEY

    cfg = load_config(make_project({"llm.provider": "openrouter"}, env_text=f"OPENROUTER_API_KEY={FAKE_OR_KEY}\n", subdir="or"))
    write_inbox(cfg, {"a.md": A_MD})
    res = run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg))
    for p in (Path(res.report_md), Path(res.report_json), *node_files(cfg)):
        assert FAKE_OR_KEY not in p.read_text(encoding="utf-8"), p
    rj = report_json(res)
    assert rj["provider"] == "openrouter"
    assert rj["stages"]["atomize"]["model_requested"] == "anthropic/claude-sonnet-5"
    assert rj["stages"]["describe"]["model_requested"] == "google/gemini-2.5-flash"
    assert "TESTONLY" not in json.dumps(rj)


# ---------------------------------------------------------------- sandbox


def test_sandbox_pipeline_writes_only_under_the_configured_roots(cfg):
    write_inbox(cfg, {"a.md": A_MD})
    root = cfg.project_root
    before = snapshot_tree(root)
    run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg))
    after = snapshot_tree(root)
    roots = [cfg.sandbox.root(k).relative_to(root).as_posix() + "/" for k in ("inbox", "converted", "graph", "processed", "runs")]
    changed = {p for p in after if p not in before or after[p] != before[p]}
    assert changed
    outside = sorted(p for p in changed if not any(p.startswith(r) for r in roots))
    assert outside == [], f"written outside the sandbox roots: {outside}"


def test_sandbox_source_name_that_escapes_is_refused_not_followed(cfg):
    # a symlink in the inbox pointing outside the project must not be read, converted or moved
    outside = cfg.project_root.parent / "outside-secret.md"
    outside.write_text("# Secret\n\nnever ingest me\n", encoding="utf-8")
    (cfg.sandbox.root("inbox") / "link.md").symlink_to(outside)
    res = run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg))
    assert outside.exists()
    assert node_files(cfg) == []
    assert (cfg.sandbox.root("inbox") / "link.md").is_symlink()
    assert report_json(res)["inputs"][0]["status"] in ("failed", "unsupported")
