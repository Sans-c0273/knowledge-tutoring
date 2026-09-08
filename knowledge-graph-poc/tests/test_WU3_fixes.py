"""WU3 fix cycle 1 — regression tests pinning commits 7985cd7, 791e869, 0d79f9e, c4f36e4.

Spec: PRD R3 (no node without a locator), R4/R11 (definition can never forge a `## Source`
section; render/parse byte-stable), R5 (sources move only after nodes are on disk; nothing
partial), R7 (permanent ids: registry consulted before in-file aliases, alias<->alias never
merges, `norm_title` recomputed on load), R12 (duplicates queue escaped and kept per run),
R15/R18 (report on every run, never carrying a credential), R21 (bounded one-line deferral
reasons; a rate-limit wait is tried once, not N times).

Each test names the fix it pins in its docstring; the setups are the ones the implementer
verified. Scripted model, injected clock and sleep, zero tokens.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from kg.config import load_config
from kg.normalise import norm_title
from kg.notes import parse
from wu3_fixtures import (
    A_MD,
    B_MD,
    DUP_MD,
    NOW,
    NOW_2,
    RUN_ID,
    RUN_ID_2,
    ScriptedCallStage,
    default_handlers,
    ids_in,
    load_nodes,
    make_chunk,
    make_unit,
    node_files,
    smart_atomize,
    snapshot_tree,
    write_inbox,
)

U1_TEXT = "The sample space is the set of all possible outcomes of a random experiment."


@pytest.fixture
def cfg(make_project):
    return load_config(make_project())


@pytest.fixture
def reg():
    from kg.registry import Registry

    return Registry(corpus="kc", schema="general")


@pytest.fixture
def chunk():
    return make_chunk("a.md#c1", [make_unit("U1", U1_TEXT, heading=["Sample Space"])])


def C(title, *, definition="Def.", aliases=(), source="a.md", chunk_id="a.md#c1", quotes=(("a.md#heading=X", "quote"),)):
    from kg.notes import NodeSource
    from kg.stages.atomize import Candidate

    return Candidate(
        title=title,
        definition=definition,
        aliases=list(aliases),
        unit_ids=["U1"],
        quotes=[],
        source=source,
        chunk_id=chunk_id,
        review=[],
        sources=[NodeSource(locator=loc, file=source, quote=q) for loc, q in quotes],
    )


def wire_node(title, *, unit_ids=("U1",), quotes=(("U1", U1_TEXT),), definition="A definition.", aliases=()):
    return {"title": title, "definition": definition, "aliases": list(aliases), "unit_ids": list(unit_ids), "quotes": [{"unit_id": u, "text": t} for u, t in quotes]}


def run_pipeline(cfg, fake, **kw):
    from kg.pipeline import run

    return run(cfg, call_stage=fake, now=kw.pop("now", NOW), **kw)


def graph(cfg) -> Path:
    return cfg.sandbox.root("graph")


def report_json(res) -> dict:
    return json.loads(Path(res.report_json).read_text(encoding="utf-8"))


def registry_yaml(cfg) -> dict:
    return yaml.safe_load((graph(cfg) / "_registry.yaml").read_text(encoding="utf-8"))


# ------------------------------------------------------- 1, 2: S4 registry-first (791e869)


def test_R7_candidate_whose_alias_is_a_registered_title_reuses_that_id_and_adds_its_title_as_alias(reg):
    """Fix 791e869: registry lookup happens before in-file grouping; no new id for a known concept."""
    from kg.stages.consolidate import consolidate

    reg.allocate("PDF", run=RUN_ID)
    assert reg.next_seq == 2
    (draft,) = consolidate([C("Portable Document Format", aliases=["PDF"])], reg, run=RUN_ID)
    assert draft.id == "kc-0001"
    assert draft.is_new is False
    assert reg.next_seq == 2, "no id allocated for a title that names an active row through its alias"
    row = next(r for r in reg.rows if r.id == "kc-0001")
    assert row.title == "PDF"
    assert "Portable Document Format" in row.aliases, "the differently-worded title becomes an alias of the registered node"


def test_R7_two_candidates_sharing_only_an_alias_stay_two_drafts(reg):
    """Fix 791e869: alias<->alias never merges (PDF = Portable Document Format and Probability Density Function)."""
    from kg.stages.consolidate import consolidate

    drafts = consolidate([C("Portable Document Format", aliases=["PDF"]), C("Probability Density Function", aliases=["PDF"])], reg, run=RUN_ID)
    assert [d.title for d in drafts] == ["Portable Document Format", "Probability Density Function"]
    assert [d.id for d in drafts] == ["kc-0001", "kc-0002"]
    assert all(d.is_new for d in drafts)


def test_R7_registered_alias_does_not_capture_a_different_title_that_shares_it(reg):
    """Fix 791e869: a registry row's alias matches a candidate *title* only; a candidate alias equal to it is not a match."""
    from kg.stages.consolidate import consolidate

    reg.allocate("Portable Document Format", run=RUN_ID, aliases=["PDF"])
    drafts = consolidate([C("Portable Document Format", aliases=["PDF"]), C("Probability Density Function", aliases=["PDF"])], reg, run=RUN_ID)
    by_title = {d.title: d for d in drafts}
    assert by_title["Portable Document Format"].id == "kc-0001" and by_title["Portable Document Format"].is_new is False
    assert by_title["Probability Density Function"].id == "kc-0002" and by_title["Probability Density Function"].is_new is True
    assert reg.next_seq == 3


# ---------------------------------------------------- 3: locator-less candidate dropped (7985cd7)


def test_R3_candidate_citing_no_real_unit_is_dropped_listed_in_grounding_queue_and_warned(cfg):
    """Fix 7985cd7: a node is never written without a locator; the drop is visible, not silent."""

    def atomize_with_phantom_unit(call):
        out = smart_atomize(call)
        out["nodes"][0]["unit_ids"] = ["U99"]
        out["nodes"][0]["quotes"] = [{"unit_id": "U99", "text": "anything"}]
        return out

    write_inbox(cfg, {"a.md": A_MD})
    res = run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg, atomize=atomize_with_phantom_unit), cfg=cfg))

    nodes = load_nodes(cfg)
    assert sorted(n.title for n in nodes) == ["Event", "Probability Measure"], "the phantom-cited candidate (Sample Space) was not written"
    assert all(n.sources and all(s.locator for s in n.sources) for n in nodes)
    grounding = (graph(cfg) / "_review" / "grounding.md").read_text(encoding="utf-8")
    assert "Dropped candidates" in grounding and "Sample Space" in grounding
    rj = report_json(res)
    assert rj["review_queues"]["grounding"] == 1
    assert any("Sample Space" in w and "dropped" in w for w in rj["warnings"]), rj["warnings"]
    assert rj["inputs"][0]["status"] == "ingested", "one dropped candidate does not defer the file"
    assert (cfg.sandbox.root("processed") / "a.md").exists()


# ------------------------------------------------- 4: per-run duplicates queue (c4f36e4)


def test_R12_duplicates_queue_is_written_per_run_and_the_latest_is_left_alone_by_a_run_without_pairs(cfg):
    """Fix c4f36e4: `_review/duplicates-<run>.md` per run; an empty run does not blank `duplicates.md`."""
    write_inbox(cfg, {"a.md": A_MD, "dup.md": DUP_MD})
    run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg))
    review = graph(cfg) / "_review"
    assert (review / f"duplicates-{RUN_ID}.md").is_file()
    assert (review / "duplicates.md").is_file()
    latest_before = (review / "duplicates.md").read_bytes()
    assert (review / f"duplicates-{RUN_ID}.md").read_bytes() == latest_before

    run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg), now=NOW_2)  # empty inbox
    assert (review / "duplicates.md").read_bytes() == latest_before
    assert not (review / f"duplicates-{RUN_ID_2}.md").exists()


# --------------------------------------------- 5: atomic S8 batch, registry last (c4f36e4)


def test_R5_R7_stale_registry_tmp_aborts_the_write_leaving_graph_and_inbox_untouched(cfg):
    """Fix c4f36e4: notes + registry go in one atomic batch; a failure leaves no partial note, no moved source."""
    from kg.pipeline import PipelineError

    write_inbox(cfg, {"a.md": A_MD})
    run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg))
    nodes_before = snapshot_tree(graph(cfg) / "nodes")
    registry_before = (graph(cfg) / "_registry.yaml").read_bytes()
    assert nodes_before and registry_before

    (graph(cfg) / "_registry.yaml.tmp").write_text("stale\n", encoding="utf-8")
    write_inbox(cfg, {"b.md": B_MD})
    with pytest.raises(PipelineError):  # cycle 2 (61d3e35): a bare OSError no longer escapes; the report is written first
        run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg), now=NOW_2)
    assert (graph(cfg) / "_reports" / f"run-{RUN_ID_2}.json").is_file(), "R15: a report exists even when S8 could not write"
    rj = json.loads((graph(cfg) / "_reports" / f"run-{RUN_ID_2}.json").read_text(encoding="utf-8"))
    assert rj["status"] == "failed", "nothing landed, so the run is failed, not partial"
    (b_row,) = [i for i in rj["inputs"] if i["name"] == "b.md"]
    assert b_row["status"] == "deferred" and b_row["reason"].startswith("write: FileExistsError"), b_row
    assert [e for e in rj["events"] if e["event"] == "write_failed"], rj["events"]

    assert snapshot_tree(graph(cfg) / "nodes") == nodes_before, "no note from b.md and no rewritten a.md note"
    assert (graph(cfg) / "_registry.yaml").read_bytes() == registry_before
    assert [p.name for p in (graph(cfg) / "nodes").glob("*.tmp")] == []
    assert (cfg.sandbox.root("inbox") / "b.md").is_file(), "the source stays in the inbox when its nodes did not land"
    assert sorted(p.name for p in cfg.sandbox.root("processed").iterdir()) == ["a.md"]


# ------------------------------------------------- 6: unreadable-note guard (c4f36e4)


def test_R5_R21_unreadable_existing_note_is_never_overwritten_and_edges_to_it_are_rejected(cfg):
    """Fix c4f36e4: a note this run could not parse is not the node the registry says it is; leave it alone."""
    write_inbox(cfg, {"a.md": A_MD})
    run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg))
    broken = graph(cfg) / "nodes" / "sample-0001-sample-space.md"
    garbage = b"this is not a note\n\x00\x01"
    broken.write_bytes(garbage)

    def edge_to_broken(call):
        ids = ids_in(call["user_text"], cfg.corpus.slug)
        return {"edges": [{"type": "related_to", "source_id": max(ids), "target_id": "sample-0001", "relevance": 50}]}

    c_md = "# Sample Space\n\nThe sample space appears again in this file.\n\n# Zeta Thing\n\nA zeta thing is a made-up concept.\n"
    write_inbox(cfg, {"c.md": c_md})
    res = run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg, edges=edge_to_broken), cfg=cfg), now=NOW_2)

    assert broken.read_bytes() == garbage, "the unreadable note was not written over"
    rj = report_json(res)
    assert any("unreadable" in w for w in rj["warnings"]), rj["warnings"]
    assert {i["name"]: i["status"] for i in rj["inputs"]} == {"c.md": "ingested"}, "the new file is still processed"
    assert (cfg.sandbox.root("processed") / "c.md").is_file()
    assert (graph(cfg) / "nodes" / "sample-0004-zeta-thing.md").is_file()
    gates = [(r["gate"], r["target_id"]) for r in rj["rejected_edges"]]
    assert ("unreadable_note", "sample-0001") in gates, gates
    readable = [parse(p.read_text(encoding="utf-8")) for p in node_files(cfg) if p != broken]
    assert len(readable) == 3
    assert all(e.target != "sample-0001" for n in readable for e in n.edges), "the rejected edge was not attached anywhere"


# ------------------------------------------ 7: one-line escaped definition, locator (7985cd7)


def test_R4_R11_definition_cannot_smuggle_a_source_section_and_the_note_round_trips(cfg, chunk):
    """Fix 7985cd7: model prose is collapsed to one line and a structural prefix is escaped."""
    from kg.notes import LinkTarget, Node, parse, render
    from kg.stages.atomize import atomize_chunk

    forged = "## Source\n### evil.pdf#page=9\n> FORGED quote\n\nReal def."
    fake = ScriptedCallStage({"atomize": [{"nodes": [wire_node("Sample Space", definition=forged)]}]}, cfg=cfg)
    (cand,) = atomize_chunk(chunk, cfg, call_stage=fake)
    assert "\n" not in cand.definition
    assert cand.definition.startswith("\\## Source")
    assert cand.definition.endswith("Real def.")

    node = Node(
        id="kc-0001", title=cand.title, aliases=cand.aliases, schema="general", origin=None, created="2026-08-25", updated="2026-08-25",
        run=RUN_ID, prompt_version="atomize@1", provider="claude_subscription", model="claude-sonnet-5", review=cand.review,
        sources=cand.sources, edges=[], definition=cand.definition,
    )
    targets = {"kc-0001": LinkTarget(stem="kc-0001-sample-space", title="Sample Space")}
    text = render(node, targets)
    back = parse(text)
    assert render(back, targets) == text
    assert [(s.locator, s.quote) for s in back.sources] == [("a.md#heading=Sample Space", U1_TEXT)]
    assert "FORGED" not in "".join(s.quote for s in back.sources)
    assert "evil.pdf" not in "".join(s.locator for s in back.sources)
    assert back.definition == cand.definition


def test_R4_locator_with_a_line_break_is_flattened_to_one_line():
    """Fix 7985cd7: a locator is one `### ` heading line; a break inside it would read back as a forged quote block."""
    from kg.notes import NodeSource

    s = NodeSource(locator="a#X\n### forged", file="a.md", quote="q")
    assert "\n" not in s.locator and "\r" not in s.locator
    assert s.locator.startswith("a#X")


# -------------------------------------------- 8, 9: bounded, redacted reasons (0d79f9e, c4f36e4)


def test_R21_R15_deferral_reason_is_one_bounded_line_naming_stage_and_exception_type(cfg):
    """Fix c4f36e4: reasons are `<stage>: <Type>: <first line>`, capped, never a raw SDK body."""

    def atomize_explodes(call):
        raise RuntimeError("provider said no\n" + "x" * 5000)

    write_inbox(cfg, {"a.md": A_MD})
    res = run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg, atomize=atomize_explodes), cfg=cfg))
    row = report_json(res)["inputs"][0]
    assert row["status"] == "deferred"
    assert row["reason"].startswith("atomize: RuntimeError: provider said no")
    assert len(row["reason"]) <= 300
    assert "\n" not in row["reason"] and "\r" not in row["reason"]
    md = Path(res.report_md).read_text(encoding="utf-8")
    assert "x" * 400 not in md


def test_R15_R18_report_is_still_written_when_a_failure_message_carries_the_api_key(make_project):
    """Fix 0d79f9e: redaction refusal falls back to a redacted report marked failed, never no report."""
    from wu2_fakes import FAKE_OR_KEY

    cfg = load_config(make_project({"llm.provider": "openrouter"}, env_text=f"OPENROUTER_API_KEY={FAKE_OR_KEY}\n", subdir="or"))

    def atomize_leaks_key(call):
        raise RuntimeError(f"401 unauthorized: key {FAKE_OR_KEY} was rejected")

    write_inbox(cfg, {"a.md": A_MD})
    res = run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg, atomize=atomize_leaks_key), cfg=cfg))
    md, js = Path(res.report_md), Path(res.report_json)
    assert md.is_file() and js.is_file()
    md_text, js_text = md.read_text(encoding="utf-8"), js.read_text(encoding="utf-8")
    assert FAKE_OR_KEY not in md_text and FAKE_OR_KEY not in js_text
    assert "TESTONLY" not in md_text and "TESTONLY" not in js_text
    assert "<redacted>" in md_text and "<redacted>" in js_text
    assert res.status == "failed" and json.loads(js_text)["status"] == "failed"


# ---------------------------------------------------- 10: name caps (7985cd7)


def test_R11_titles_and_aliases_lose_control_characters_and_are_capped(cfg, chunk):
    """Fix 7985cd7: names are names, not prose; C0 controls dropped, length capped at 200."""
    from kg.stages.atomize import MAX_NAME_LEN, atomize_chunk

    title = "Sam\x00ple\x07 " + "S" * 300
    fake = ScriptedCallStage({"atomize": [{"nodes": [wire_node(title, aliases=["a\x1bb"])]}]}, cfg=cfg)
    (cand,) = atomize_chunk(chunk, cfg, call_stage=fake)
    assert len(cand.title) == 200 == MAX_NAME_LEN
    assert cand.title.startswith("Sample S")
    assert not any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in cand.title)
    assert cand.aliases == ["ab"]


# ------------------------------------------------ 11: escaped queue cells (0d79f9e)


def test_R12_duplicates_queue_cells_escape_pipes_and_flatten_line_breaks():
    """Fix 0d79f9e: a model reason or title with `|` or a newline cannot break the Markdown table."""
    from kg.stages.dedup import DedupResult, Judged, render_queue

    result = DedupResult(unsure=[Judged("kc-0001", "kc-0002", 0.71, "unsure", "a | b\nc")], titles={"kc-0001": "T|1", "kc-0002": "Other"})
    text = render_queue(result)
    assert "a \\| b c" in text
    assert "T\\|1" in text
    (row,) = [ln for ln in text.splitlines() if ln.startswith("| kc-0001")]
    unescaped_cells = re.split(r"(?<!\\)\|", row)[1:-1]
    assert len(unescaped_cells) == 7, "seven cells; the escaped pipes do not split a cell"


# ------------------------------------------ 12: one rate-limit wait, then defer (c4f36e4)


def test_R21_rate_limit_wait_is_tried_once_then_the_rest_is_deferred(cfg):
    """Fix c4f36e4: a second consecutive `wait` means waiting is not working; no N x wait, no N x call."""
    from kg.llm.errors import RateLimited

    sleeps: list[float] = []

    def always_limited(call):
        return RateLimited("rate limited", action="wait", wait_seconds=5)

    write_inbox(cfg, {"a.md": A_MD, "b.md": B_MD})
    fake = ScriptedCallStage(default_handlers(cfg, atomize=always_limited), cfg=cfg)
    res = run_pipeline(cfg, fake, sleep=sleeps.append)

    assert sleeps == [5]
    assert len(fake.calls_for("atomize")) == 2, "first call, one wait, retry — then stop"
    assert {i["name"]: i["status"] for i in report_json(res)["inputs"]} == {"a.md": "deferred", "b.md": "deferred"}
    assert sorted(p.name for p in cfg.sandbox.root("inbox").iterdir()) == ["a.md", "b.md"]
    assert node_files(cfg) == []


# ------------------------------------------------- 13: finished_at is read (c4f36e4)


def test_R15_report_finished_comes_from_the_injected_finished_at_clock(cfg):
    """Fix c4f36e4: `finished` is no longer a copy of `started`."""
    write_inbox(cfg, {"a.md": A_MD})
    res = run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg), finished_at=lambda: NOW_2)
    rj = report_json(res)
    assert rj["started"] == "2026-08-25T10:15:02Z"
    assert rj["finished"] == "2026-08-26T09:00:00Z"
    assert rj["finished"] != rj["started"]


# -------------------------------------------- 14: norm_title recomputed on load (791e869)


def test_R7_registry_load_recomputes_norm_title_from_the_stored_title(tmp_path):
    """Fix 791e869: the stored `norm_title` is informational; lookups use the current normaliser."""
    from kg.registry import Registry

    doc = {
        "corpus": "kc",
        "schema": "general",
        "next_seq": 2,
        "nodes": [{"id": "kc-0001", "title": "Sample Space", "norm_title": "WRONG", "file": "nodes/kc-0001-sample-space.md", "status": "active", "created_run": RUN_ID, "aliases": []}],
    }
    path = tmp_path / "_registry.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    reg = Registry.load(path)
    (row,) = reg.rows
    assert row.norm_title == norm_title("Sample Space")
    assert row.norm_title != "WRONG"
    assert reg.find("sample space") == "kc-0001"


# =====================================================================================
# WU3 fix cycle 2 — commits 6f23cbd, 61d3e35, d0e81f3, 0af492e, 67e2760.
#
# R7 (consolidate routes by registry id, merges bridged groups, drops foreign aliases),
# R5/R7 (a note this run did not read is never written over; the id stays burnt),
# R5/R21 (drops survive a rate-limit retry; the second wait defers with a warning),
# R15/R18 (only credential-shaped .env values are redacted; a key past the reason cap
# never survives; a write failure still yields a report), R12/R18 (review files are never
# written through a planted symlink), R4/R11 (HTML comments and `<` in model prose).
# =====================================================================================

from datetime import UTC, datetime  # noqa: E402

from wu3_fixtures import parse_rendered_chunk  # noqa: E402

RESETS_AT = int(datetime(2026, 8, 25, 10, 20, 2, tzinfo=UTC).timestamp())
RESETS_AT_ISO = "2026-08-25T10:20:02Z"


def _outside(cfg) -> Path:
    """A file outside the project directory (the sandbox must refuse a symlink that lands here)."""
    target = cfg.project_root.parent / "outside-target.md"
    target.write_text("KEEP\n", encoding="utf-8")
    return target


# --------------------------------------------- 1: foreign alias dropped, not shared (6f23cbd)


def test_R7_alias_naming_another_registered_node_is_dropped_and_that_node_keeps_its_id(reg):
    """Fix 6f23cbd: `Foo (alias Qux)` cannot claim `Qux` when Qux is already a registered node."""
    from kg.stages.consolidate import consolidate

    foo = reg.allocate("Foo", run=RUN_ID).id
    qux = reg.allocate("Qux", run=RUN_ID).id
    warnings: list[str] = []
    drafts = consolidate([C("Foo", aliases=["Qux"]), C("Qux")], reg, run=RUN_ID, warnings=warnings)
    by_title = {d.title: d for d in drafts}
    assert set(by_title) == {"Foo", "Qux"}
    assert by_title["Qux"].id == qux and by_title["Qux"].is_new is False
    assert by_title["Foo"].id == foo and by_title["Foo"].is_new is False
    assert "Qux" not in by_title["Foo"].aliases, "the alias was dropped from the draft"
    assert "Qux" not in next(r for r in reg.rows if r.id == foo).aliases, "and never reached Foo's registry row"
    assert any("alias 'Qux' of 'Foo' dropped: it already names active node" in w for w in warnings), warnings
    assert reg.next_seq == 3, "no id allocated for either"


# ----------------------------------------- 2: a bridging candidate merges two groups (6f23cbd)


def test_R7_candidate_bridging_two_in_file_groups_yields_one_draft_with_the_registered_id(reg):
    """Fix 6f23cbd: `Alpha`, `Beta`, `Alpha (alias Beta)` is one concept once Beta is registered; no duplicate ids."""
    from kg.stages.consolidate import consolidate

    beta = reg.allocate("Beta", run=RUN_ID).id
    seq_before = reg.next_seq
    cands = [
        C("Alpha", quotes=(("a.md#heading=A", "alpha quote"),)),
        C("Beta", quotes=(("a.md#heading=B", "beta quote"),)),
        C("Alpha", aliases=["Beta"], quotes=(("a.md#heading=AB", "bridge quote"),)),
    ]
    (draft,) = consolidate(cands, reg, run=RUN_ID)
    assert draft.id == beta and draft.is_new is False
    assert {(s.locator, s.quote) for s in draft.sources} == {("a.md#heading=A", "alpha quote"), ("a.md#heading=B", "beta quote"), ("a.md#heading=AB", "bridge quote")}
    assert len(draft.sources) == 3
    assert reg.next_seq == seq_before, "the bridged group attaches to Beta; nothing is allocated"


def test_R7_bridging_candidate_without_a_registry_row_allocates_exactly_one_id(reg):
    """Fix 6f23cbd: the same three candidates on an empty registry are one draft, one allocation."""
    from kg.stages.consolidate import consolidate

    drafts = consolidate([C("Alpha"), C("Beta"), C("Alpha", aliases=["Beta"])], reg, run=RUN_ID)
    assert len(drafts) == 1 and drafts[0].is_new is True
    assert reg.next_seq == 2, "exactly one id allocated"
    assert len({d.id for d in drafts}) == len(drafts)
    assert sorted({drafts[0].title, *drafts[0].aliases}) == ["Alpha", "Beta"]


# -------------------------------------- 3: note-name collision refused before S8 (61d3e35)


def test_R5_R7_note_file_this_run_did_not_read_is_never_written_over_and_its_id_stays_burnt(cfg):
    """Fix 61d3e35: a planted `nodes/sample-0001-sample-space.md` is left alone; no row, no edges, no reuse of the id."""
    foreign = graph(cfg) / "nodes" / "sample-0001-sample-space.md"
    foreign.parent.mkdir(parents=True, exist_ok=True)
    foreign_bytes = b"# Not a note this run wrote\n"
    foreign.write_bytes(foreign_bytes)

    def edge_to_collided(call):
        ids = ids_in(call["user_text"], cfg.corpus.slug)
        return {"edges": [{"type": "related_to", "source_id": ids[1], "target_id": ids[0], "relevance": 50}]} if len(ids) >= 2 else {"edges": []}

    write_inbox(cfg, {"a.md": A_MD})
    res = run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg, edges=edge_to_collided), cfg=cfg))

    assert foreign.read_bytes() == foreign_bytes, "the foreign file was not written over"
    rj = report_json(res)
    assert rj["counts"]["nodes"]["created"] == 2, "Sample Space was refused; Event and Probability Measure landed"
    assert {i["name"]: i["status"] for i in rj["inputs"]} == {"a.md": "ingested"}
    assert any("was not read this run" in w and "sample-0001" in w for w in rj["warnings"]), rj["warnings"]
    assert ("collision", "sample-0001") in [(r["gate"], r["target_id"]) for r in rj["rejected_edges"]], rj["rejected_edges"]

    reg = registry_yaml(cfg)
    assert "sample-0001" not in {r["id"] for r in reg["nodes"]}, "no row without its note"
    assert reg["next_seq"] == 4, "the burnt id is skipped, never handed out again"
    notes = [parse(p.read_text(encoding="utf-8")) for p in node_files(cfg) if p != foreign]
    assert sorted(n.id for n in notes) == ["sample-0002", "sample-0003"]
    assert all(e.target != "sample-0001" for n in notes for e in n.edges)
    assert all("sample-0001" not in p.read_text(encoding="utf-8") for p in node_files(cfg) if p != foreign), "no note links to the refused id"

    write_inbox(cfg, {"b.md": B_MD})  # a later run must not reuse sample-0001 either
    run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg), now=NOW_2)
    assert sorted(r["id"] for r in registry_yaml(cfg)["nodes"]) == ["sample-0002", "sample-0003", "sample-0004", "sample-0005"]
    assert foreign.read_bytes() == foreign_bytes


# ------------------------------------------ 4: dropped candidates survive a retry (61d3e35)


def test_R3_R21_dropped_candidates_from_before_a_rate_limit_wait_are_still_in_the_grounding_queue(make_project):
    """Fix 61d3e35: a.md's drop and the waiting file's cached-chunk drop both survive the one retry."""
    from kg.llm.errors import RateLimited

    cfg = load_config(make_project({"chunking.target_tokens": 8, "chunking.max_tokens": 60, "chunking.min_tokens": 0}))
    state = {"limited": False}
    sleeps: list[float] = []

    def atomize(call):
        paths = [u["path"] for u in parse_rendered_chunk(call["user_text"])]
        if "Expectation" in paths and not state["limited"]:
            state["limited"] = True
            return RateLimited("rate limited", action="wait", wait_seconds=5)
        out = smart_atomize(call)
        for node in out["nodes"]:
            if node["title"] in ("Sample Space", "Random Variable"):
                node["unit_ids"], node["quotes"] = ["U99"], [{"unit_id": "U99", "text": "anything"}]
        return out

    write_inbox(cfg, {"a.md": A_MD, "b.md": B_MD})
    fake = ScriptedCallStage(default_handlers(cfg, atomize=atomize), cfg=cfg)
    res = run_pipeline(cfg, fake, sleep=sleeps.append)

    assert sleeps == [5]
    rv_calls = [c for c in fake.calls_for("atomize") if [u["path"] for u in parse_rendered_chunk(c["user_text"])] == ["Random Variable"]]
    assert len(rv_calls) == 1, "the finished chunk was served from the cache on the retry"
    assert {i["name"]: i["status"] for i in report_json(res)["inputs"]} == {"a.md": "ingested", "b.md": "ingested"}
    grounding = (cfg.sandbox.root("graph") / "_review" / "grounding.md").read_text(encoding="utf-8")
    dropped_section = grounding.split("## Dropped candidates", 1)[1]
    assert "Sample Space" in dropped_section, "a.md's drop is still listed after b.md's retry"
    assert "Random Variable" in dropped_section, "the waiting file's cached drop is listed too"
    assert report_json(res)["review_queues"]["grounding"] == 2
    assert sorted(n.title for n in load_nodes(cfg)) == ["Event", "Expectation", "Probability Measure"]


# ------------------------------ 5: only credential-shaped .env values are secrets (d0e81f3)


def test_R15_R18_short_and_non_credential_env_values_do_not_redact_a_healthy_report(make_project):
    """Fix d0e81f3: `X=x`, `KG_SCHEMA=general`, `MY_TOKEN=abc` must not turn `schema: general` into <redacted>."""
    from wu2_fakes import FAKE_OR_KEY

    env = f"OPENROUTER_API_KEY={FAKE_OR_KEY}\nX=x\nKG_SCHEMA=general\nMY_TOKEN=abc\n"
    cfg = load_config(make_project({"llm.provider": "openrouter"}, env_text=env, subdir="or"))
    write_inbox(cfg, {"a.md": A_MD})
    res = run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg))
    md = Path(res.report_md).read_text(encoding="utf-8")
    rj = report_json(res)
    assert res.status == "ok" and rj["status"] == "ok"
    assert rj["schema"] == "general" and "- schema: general" in md
    assert "<redacted>" not in md and "<redacted>" not in json.dumps(rj)


def test_R15_R18_credential_shaped_env_values_are_redacted_from_a_failure_message(make_project):
    """Fix d0e81f3: `DB_PASSWORD` / `SOME_SECRET` (8+ chars) are scanned for, even though no stage uses them."""
    from wu2_fakes import FAKE_OR_KEY

    env = f"OPENROUTER_API_KEY={FAKE_OR_KEY}\nDB_PASSWORD=supersecret123\nSOME_SECRET=abcdefgh\n"
    cfg = load_config(make_project({"llm.provider": "openrouter"}, env_text=env, subdir="or"))

    def atomize_leaks_env(call):
        raise RuntimeError("db said: password supersecret123 rejected, secret abcdefgh expired")

    write_inbox(cfg, {"a.md": A_MD})
    res = run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg, atomize=atomize_leaks_env), cfg=cfg))
    md, js = Path(res.report_md).read_text(encoding="utf-8"), Path(res.report_json).read_text(encoding="utf-8")
    for leaked in ("supersecret123", "abcdefgh"):
        assert leaked not in md and leaked not in js, leaked
    assert "<redacted>" in md and "<redacted>" in js
    assert res.status == "failed" and json.loads(js)["status"] == "failed"
    assert json.loads(js)["inputs"][0]["reason"].startswith("atomize: RuntimeError: db said: password <redacted>")


# ------------------------------------- 6: second wait is a visible deferral (61d3e35)


def test_R21_R15_second_consecutive_wait_is_reported_as_a_deferral_with_the_remaining_count(cfg):
    """Fix 61d3e35: the warning names what was waited for and how many files are deferred; the event carries the count."""
    from kg.llm.errors import RateLimited

    def always_limited(call):
        return RateLimited("rate limited", action="wait", wait_seconds=5, resets_at=RESETS_AT, rate_limit_type="five_hour")

    write_inbox(cfg, {"a.md": A_MD, "b.md": B_MD})
    res = run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg, atomize=always_limited), cfg=cfg), sleep=lambda s: None)
    rj = report_json(res)
    pattern = re.compile(r"^rate limit: waited once until \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z; second limit → 2 remaining file\(s\) deferred$")
    matching = [w for w in rj["warnings"] if pattern.match(w)]
    assert len(matching) == 1, rj["warnings"]
    assert RESETS_AT_ISO in matching[0]
    (event,) = [e for e in rj["events"] if e["event"] == "rate_limit_deferral"]
    assert event["remaining_files"] == 2 and event["file"] == "a.md" and event["waited_until"] == RESETS_AT_ISO
    assert {i["name"]: i["status"] for i in rj["inputs"]} == {"a.md": "deferred", "b.md": "deferred"}
    assert all(i["reason"].startswith("model: RateLimited") for i in rj["inputs"]), rj["inputs"]


# ------------------------------ 7: review files never written through a symlink (0af492e)


def test_R12_R18_duplicates_queue_symlink_pointing_outside_the_project_is_refused_with_a_report(cfg):
    """Fix 0af492e: a planted `_review/duplicates.md -> <outside>` stops the write; the report still lands, nothing moves."""
    from kg.pipeline import PipelineError

    target = _outside(cfg)
    review = graph(cfg) / "_review"
    review.mkdir(parents=True, exist_ok=True)
    (review / "duplicates.md").symlink_to(target)

    write_inbox(cfg, {"a.md": A_MD, "dup.md": DUP_MD})
    with pytest.raises(PipelineError):
        run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg))

    assert target.read_text(encoding="utf-8") == "KEEP\n", "the target outside the project was not written"
    assert (review / "duplicates.md").is_symlink(), "the planted link was not followed or replaced"
    assert sorted(p.name for p in cfg.sandbox.root("inbox").iterdir()) == ["a.md", "dup.md"], "inbox untouched"
    assert not any(cfg.sandbox.root("processed").iterdir())
    rj = json.loads((graph(cfg) / "_reports" / f"run-{RUN_ID}.json").read_text(encoding="utf-8"))
    (event,) = [e for e in rj["events"] if e["event"] == "write_failed"]
    assert event["reason"].startswith("write: SandboxViolation"), event
    assert {i["status"] for i in rj["inputs"]} == {"deferred"}


def test_R12_R18_duplicates_queue_symlink_inside_the_project_is_replaced_by_a_regular_file(cfg):
    """Fix 0af492e: `.tmp` + replace (O_NOFOLLOW) swaps the link for a real file; the link's target is untouched."""
    review = graph(cfg) / "_review"
    review.mkdir(parents=True, exist_ok=True)
    keep = review / "keep.md"
    keep.write_text("KEEP\n", encoding="utf-8")
    (review / "duplicates.md").symlink_to(keep)

    write_inbox(cfg, {"a.md": A_MD, "dup.md": DUP_MD})
    res = run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg))

    assert res.status == "ok"
    assert keep.read_text(encoding="utf-8") == "KEEP\n", "not written through the link"
    queue = review / "duplicates.md"
    assert queue.is_file() and not queue.is_symlink(), "the link is gone; a regular file stands in its place"
    assert queue.read_bytes() == (review / f"duplicates-{RUN_ID}.md").read_bytes()
    assert "Duplicate review queue" in queue.read_text(encoding="utf-8")


def test_R18_rejected_edges_log_symlink_is_not_written_through_and_the_report_says_so(cfg):
    """Fix 0af492e: `_review/rejected-edges.jsonl -> <file>` is refused; rejections stay in the report."""
    review = graph(cfg) / "_review"
    review.mkdir(parents=True, exist_ok=True)
    keep = review / "keep.jsonl"
    keep.write_bytes(b"")
    (review / "rejected-edges.jsonl").symlink_to(keep)

    def same_edge_twice(call):  # S5 pre-filters self/dangling/out-of-schema; a duplicate reaches the gates
        ids = ids_in(call["user_text"], cfg.corpus.slug)
        edge = {"type": "related_to", "source_id": ids[0], "target_id": ids[1], "relevance": 50}
        return {"edges": [edge, dict(edge)]}

    write_inbox(cfg, {"a.md": A_MD})
    res = run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg, edges=same_edge_twice), cfg=cfg))

    assert keep.read_bytes() == b"", "nothing appended through the link"
    assert (review / "rejected-edges.jsonl").is_symlink()
    rj = report_json(res)
    assert any("rejected-edges.jsonl not written" in w for w in rj["warnings"]), rj["warnings"]
    assert [r["gate"] for r in rj["rejected_edges"]] == ["duplicate_edge"], "the rejection is still reported"
    assert rj["counts"]["edges_by_type"] == {"related_to": 1}
    assert res.status == "ok" and {i["status"] for i in rj["inputs"]} == {"ingested"}


# ---------------------------- 8: a key past the reason cap never survives as a prefix (61d3e35)


def test_R15_R18_R21_key_beyond_the_reason_cap_is_redacted_before_truncation(make_project):
    """Fix 61d3e35: redaction happens before the 300-char cap, so no prefix of the key can survive the cut."""
    from wu2_fakes import FAKE_OR_KEY

    cfg = load_config(make_project({"llm.provider": "openrouter"}, env_text=f"OPENROUTER_API_KEY={FAKE_OR_KEY}\n", subdir="or"))

    def atomize_leaks_late(call):
        raise RuntimeError("provider said no " + "x" * 300 + f" key={FAKE_OR_KEY}")

    write_inbox(cfg, {"a.md": A_MD})
    res = run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg, atomize=atomize_leaks_late), cfg=cfg))
    md, js = Path(res.report_md).read_text(encoding="utf-8"), Path(res.report_json).read_text(encoding="utf-8")
    row = json.loads(js)["inputs"][0]
    assert row["status"] == "deferred" and len(row["reason"]) <= 300
    for text in (md, js):
        assert FAKE_OR_KEY not in text
        assert FAKE_OR_KEY[:12] not in text, "no 12+ character prefix of the key either"
        assert "TESTONLY" not in text
    assert res.status == "failed" and json.loads(js)["status"] == "failed"
    assert any("<redacted>" in w for w in json.loads(js)["warnings"]), json.loads(js)["warnings"]


# ------------------------------------ 10: hidden text and `<` in model prose (67e2760)


def test_R4_R11_safe_prose_removes_html_comments_and_escapes_angle_brackets():
    """Fix 67e2760: a renderer-hidden comment cannot persist into a note or the dedup prompt."""
    from kg.notes import safe_prose

    assert safe_prose("Real def. <!-- hidden --> more") == "Real def. more"
    assert safe_prose("Real def. <!-- hidden to the end") == "Real def.", "an unterminated comment hides the rest; all of it goes"
    assert safe_prose("a <b> c") == "a \\<b> c"
    assert "hidden" not in safe_prose("<!-- hidden -->## Source")


def test_R4_R11_escaped_prose_round_trips_byte_identically(cfg, chunk):
    """Fix 67e2760: the atomize path applies safe_prose; render -> parse -> render is stable."""
    from kg.notes import LinkTarget, Node, parse, render, safe_prose
    from kg.stages.atomize import atomize_chunk

    raw = "Real def. <!-- hidden --> a <b> c <!-- open"
    fake = ScriptedCallStage({"atomize": [{"nodes": [wire_node("Sample Space", definition=raw)]}]}, cfg=cfg)
    (cand,) = atomize_chunk(chunk, cfg, call_stage=fake)
    assert cand.definition == safe_prose(raw) == "Real def. a \\<b> c"
    assert "hidden" not in cand.definition and "open" not in cand.definition

    node = Node(
        id="kc-0001", title=cand.title, aliases=[], schema="general", origin=None, created="2026-08-25", updated="2026-08-25",
        run=RUN_ID, prompt_version="atomize@1", provider="claude_subscription", model="claude-sonnet-5", review=[],
        sources=cand.sources, edges=[], definition=cand.definition,
    )
    targets = {"kc-0001": LinkTarget(stem="kc-0001-sample-space", title="Sample Space")}
    text = render(node, targets)
    back = parse(text)
    assert back.definition == cand.definition
    assert render(back, targets) == text
