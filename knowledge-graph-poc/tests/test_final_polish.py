"""Final polish and DESIGN §14 follow-up — regression tests pinning commits a295bb6 and a3b56a5.

Spec: PRD R5 (sources move only after their nodes are on disk; a failed move is a failed run with
a report), R6 (image payloads: transparency composited on white, never black), R14 (`_index.json`
is written atomically and never through a symlink), R15 (a report on every run, including S8
failures and sandbox refusals), R17 (relevance summary: a typo'd verdict is visible, not a blank),
R21 (rate-limit events reach the ledger even when the call is rejected; the wait budget is one per
run; every deferral is an explicit event).

Each test names the fix it pins in its docstring. Scripted model, injected clock and sleep, zero
tokens, no network.
"""

from __future__ import annotations

import base64
import io
import json
import random
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image

from kg.config import load_config
from kg.notes import parse
from kg.paths import SandboxViolation
from wu3_fixtures import (
    A_MD,
    B_MD,
    NOW,
    NOW_2,
    RUN_ID,
    ScriptedCallStage,
    default_handlers,
    ids_in,
    load_nodes,
    make_chunk,
    make_unit,
    node_files,
    parse_rendered_chunk,
    simple_node,
    smart_atomize,
    write_inbox,
)

RESETS_AT = int(datetime(2026, 8, 25, 10, 20, 2, tzinfo=UTC).timestamp())
RESETS_AT_ISO = "2026-08-25T10:20:02Z"
SDK_NOW = 1_800_000_000.0  # fixed epoch seconds for the adapter clock
U1_TEXT = "The sample space is the set of all possible outcomes of a random experiment."


@pytest.fixture
def cfg(make_project):
    return load_config(make_project())


@pytest.fixture
def chunk():
    return make_chunk("a.md#c1", [make_unit("U1", U1_TEXT, heading=["Sample Space"])])


def wire_node(title, *, aliases=(), definition="A definition."):
    return {"title": title, "definition": definition, "aliases": list(aliases), "unit_ids": ["U1"], "quotes": [{"unit_id": "U1", "text": U1_TEXT}]}


def run_pipeline(cfg, fake, **kw):
    from kg.pipeline import run

    return run(cfg, call_stage=fake, now=kw.pop("now", NOW), **kw)


def graph(cfg) -> Path:
    return cfg.sandbox.root("graph")


def report_json(res) -> dict:
    return json.loads(Path(res.report_json).read_text(encoding="utf-8"))


def report_json_for(cfg, run_id: str = RUN_ID) -> dict:
    return json.loads((graph(cfg) / "_reports" / f"run-{run_id}.json").read_text(encoding="utf-8"))


def _outside_file(cfg) -> Path:
    """A file outside the project directory (the sandbox must refuse a symlink that lands here)."""
    target = cfg.project_root.parent / "outside-target.md"
    target.write_text("KEEP\n", encoding="utf-8")
    return target


def _outside_dir(cfg) -> Path:
    target = cfg.project_root.parent / "outside-review"
    target.mkdir(exist_ok=True)
    return target


# =====================================================================================
# a295bb6 — polish
# =====================================================================================


# ------------------------------------------ (a) symlinked _index.json is refused (R14, §23)


def test_R14_write_index_refuses_a_symlinked_target_and_leaves_the_victim_and_no_tmp_behind(tmp_path):
    """Fix a295bb6: `write_index` never follows a planted `_index.json` link; the write is refused outright."""
    from kg.graph_io import load_graph, write_index
    from wu45_fixtures import write_general_graph

    graph_dir = tmp_path / "data" / "graph"
    graph_dir.mkdir(parents=True)
    write_general_graph(graph_dir)
    victim = graph_dir / "victim.json"
    victim.write_text("KEEP\n", encoding="utf-8")
    link = graph_dir / "_index.json"
    link.symlink_to(victim)

    with pytest.raises((OSError, SandboxViolation)):
        write_index(load_graph(graph_dir), link)

    assert victim.read_text(encoding="utf-8") == "KEEP\n", "not written through the link"
    assert link.is_symlink(), "the planted link was neither followed nor replaced"
    assert [p.name for p in graph_dir.rglob("*.tmp")] == [], "no temporary left behind"


def test_R14_write_index_over_a_regular_file_leaves_no_tmp_and_reloads(tmp_path):
    """Happy path of the same fix: the atomic write (tmp + replace) cleans up after itself."""
    from kg.graph_io import load_graph, write_index
    from wu45_fixtures import write_general_graph

    graph_dir = tmp_path / "data" / "graph"
    graph_dir.mkdir(parents=True)
    write_general_graph(graph_dir)
    (graph_dir / "_index.json").write_text("stale\n", encoding="utf-8")
    g = load_graph(graph_dir)
    write_index(g, graph_dir / "_index.json")
    assert json.loads((graph_dir / "_index.json").read_text(encoding="utf-8")) == g.to_index()
    assert [p.name for p in graph_dir.rglob("*.tmp")] == []


# ------------------------------------- (b) unrecognised verdicts are counted separately (R17)


def test_R17_summarise_counts_a_typo_verdict_as_unrecognised_separately_from_a_blank(tmp_path):
    """Fix a295bb6: `agre` is `unfilled` (excluded from rates) AND `unrecognised`; a blank is only `unfilled`."""
    from kg.relevance_sample import sample, summarise
    from wu45_fixtures import NOW as SAMPLE_NOW
    from wu45_fixtures import fill_verdicts, write_sampler_graph

    graph_dir = tmp_path / "wide" / "graph"
    graph_dir.mkdir(parents=True)
    write_sampler_graph(graph_dir)
    res = sample(graph_dir, n_per_band=8, seed=1, now=SAMPLE_NOW)

    seen = {"0-39": 0}

    def verdict(rel, band):
        if band == "0-39":
            seen[band] += 1
            if seen[band] == 1:
                return "agre"  # a typo, not a blank
            if seen[band] == 2:
                return ""  # a blank
        return "agree"

    fill_verdicts(res.path, verdict)
    s = summarise(res.path)

    assert s.unrecognised == 1
    assert s.unfilled == 2, "the typo and the blank are both unfilled"
    assert s.filled == 22
    low = s.bands["0-39"]
    assert (low.agree, low.disagree, low.unfilled, low.unrecognised) == (6, 0, 2, 1)
    assert s.bands["40-70"].unrecognised == 0 and s.bands["71-100"].unrecognised == 0
    assert s.malformed == 0


def test_R17_summarise_of_a_fully_judged_file_reports_zero_unrecognised(tmp_path):
    """Unhappy-path guard for the new counter: it must not fire on `Agree.` / `DISAGREE` first-word matches."""
    from kg.relevance_sample import sample, summarise
    from wu45_fixtures import NOW as SAMPLE_NOW
    from wu45_fixtures import fill_verdicts, write_sampler_graph

    graph_dir = tmp_path / "wide" / "graph"
    graph_dir.mkdir(parents=True)
    write_sampler_graph(graph_dir)
    res = sample(graph_dir, n_per_band=8, seed=1, now=SAMPLE_NOW)
    fill_verdicts(res.path, lambda rel, band: "Agree." if band == "71-100" else "DISAGREE - obviously")
    s = summarise(res.path)
    assert s.unrecognised == 0 and s.unfilled == 0 and s.filled == 24


# ------------------------------------------ (c) transparency composited on white, not black (R6)


def _transparent_noise_png(path: Path, side: int = 1568, seed: int = 20260826) -> Path:
    """Incompressible RGB noise under a fully transparent alpha channel: as a PNG it exceeds the byte cap,
    so `prepare_image` must fall back to JPEG — and JPEG has no alpha, so the flattening colour shows.

    `side` equals the long-edge cap on purpose: a resize would premultiply the (zero) alpha into the RGB
    noise, leaving an all-zero image that compresses under the cap and never reaches the JPEG path."""
    from kg.convert.image import MAX_IMAGE_BYTES

    rnd = random.Random(seed)
    img = Image.frombytes("RGBA", (side, side), rnd.randbytes(side * side * 4))
    img.putalpha(0)
    img.save(path, format="PNG", compress_level=1)
    assert path.stat().st_size > MAX_IMAGE_BYTES, "fixture must exceed the cap as a PNG"
    return path


def test_R6_fully_transparent_rgba_png_becomes_a_white_jpeg_not_a_black_one(tmp_path):
    """Fix a295bb6: alpha is composited on white before any JPEG encoding (a bare RGB convert paints it black)."""
    from kg.convert.image import MAX_IMAGE_BYTES, prepare_image

    src = _transparent_noise_png(tmp_path / "transparent.png")
    payload = prepare_image(src, max_long_edge_px=1568)

    assert payload.media_type == "image/jpeg"
    data = base64.b64decode(payload.data_b64)
    assert len(data) <= MAX_IMAGE_BYTES
    with Image.open(io.BytesIO(data)) as decoded:
        decoded.load()
        assert decoded.format == "JPEG"
        rgb = decoded.convert("RGB")
        w, h = rgb.size
        for xy in ((0, 0), (w // 2, h // 2), (w - 1, h - 1)):
            assert rgb.getpixel(xy) == (255, 255, 255), f"pixel {xy} is {rgb.getpixel(xy)}, expected white"


# --------------------------- (d) rate-limit events reach the ledger on a rejected call (R21, §14)


def test_R21_rejected_sdk_stream_defers_and_its_warning_and_rejection_events_are_in_the_ledger(make_project):
    """Fix a295bb6: `RateLimited` carries the events seen during the attempt; `call_stage` records them on the
    error row, so `ledger.events()` shows the warning and the rejection tagged with the stage."""
    from kg.llm import RateLimited, call_stage
    from kg.llm.adapters.claude_subscription import ClaudeSubscriptionAdapter
    from kg.llm.ledger import UsageLedger
    from kg.schemas import DescribeOutput
    from wu2_fakes import SYSTEM, USER_TEXT, FakeQuery, load_cfg, sdk_rate_limit_event, sdk_result

    cfg = load_cfg(make_project)  # claude_subscription, wait window 30 min
    warning = sdk_rate_limit_event("allowed_warning", int(SDK_NOW) + 900, "five_hour", 0.92)
    rejected = sdk_rate_limit_event("rejected", int(SDK_NOW) + 7200, "seven_day", 1.0)  # reset beyond the window
    failed = sdk_result(subtype="success", is_error=True, api_error_status=429)
    fq = FakeQuery([[warning, rejected, failed]])
    adapter = ClaudeSubscriptionAdapter(cfg, query_fn=fq, clock=lambda: SDK_NOW)
    ledger = UsageLedger()

    with pytest.raises(RateLimited) as ei:
        call_stage("describe", SYSTEM, USER_TEXT, DescribeOutput, None, cfg=cfg, ledger=ledger, adapter=adapter)

    assert ei.value.action == "defer"
    assert ei.value.rate_limit_type == "seven_day"
    assert len(fq.calls) == 1, "a rate limit is never retried inside call_stage"
    assert len(ledger.rows) == 1 and ledger.rows[0].outcome == "error"
    events = ledger.events()
    assert [e["status"] for e in events] == ["allowed_warning", "rejected"]
    assert all(e["stage"] == "describe" and e["attempt_no"] == 1 for e in events)
    assert events[0]["utilization"] == pytest.approx(0.92) and events[1]["rate_limit_type"] == "seven_day"
    json.dumps(events)  # report-serialisable


# =====================================================================================
# a3b56a5 — §14 follow-up
# =====================================================================================


# ----------------------------------------- (e) a failed source move is a failed run with a report (R5)


def test_R5_R15_move_failure_on_the_second_file_is_reported_as_failed_after_the_notes_landed(cfg):
    """Fix a3b56a5: S8 moves get their own try — the first source still moves, the second stays, the row says
    `failed`, the event is `move_failed`, the report exists, and only then does the run raise."""
    from kg.pipeline import PipelineError

    target = _outside_file(cfg)
    processed = cfg.sandbox.root("processed")
    (processed / "b.md").symlink_to(target)  # the destination for b.md resolves outside the project

    write_inbox(cfg, {"a.md": A_MD, "b.md": B_MD})
    with pytest.raises(PipelineError) as ei:
        run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg))
    assert "SandboxViolation" in str(ei.value)

    rj = report_json_for(cfg)
    assert rj["status"] == "failed"
    assert {i["name"]: i["status"] for i in rj["inputs"]} == {"a.md": "ingested", "b.md": "failed"}
    (b_row,) = [i for i in rj["inputs"] if i["name"] == "b.md"]
    assert b_row["reason"].startswith("move: SandboxViolation"), b_row
    (event,) = [e for e in rj["events"] if e["event"] == "move_failed"]
    assert event["file"] == "b.md"
    assert not [e for e in rj["events"] if e["event"] == "write_failed"], "the notes did land; this is a move failure"

    assert (processed / "a.md").is_file(), "the first source moved"
    assert sorted(p.name for p in cfg.sandbox.root("inbox").iterdir()) == ["b.md"], "the second stays in the inbox"
    assert (processed / "b.md").is_symlink() and target.read_text(encoding="utf-8") == "KEEP\n"
    assert sorted(n.title for n in load_nodes(cfg)) == ["Event", "Expectation", "Probability Measure", "Random Variable", "Sample Space"], "notes from both files are on disk"
    assert rj["counts"]["nodes"]["created"] == 5


# --------------------------------- (f) write_index failure after the notes landed (R5, R15)


def test_R5_R15_index_write_failure_after_the_notes_landed_is_failed_with_write_failed_and_nothing_moved(cfg):
    """Fix a3b56a5: any S8 write failure is `status: failed` (not only when nothing landed); the rows say the
    notes are on disk but index/queues failed; no source moves."""
    from kg.pipeline import PipelineError

    keep = graph(cfg) / "keep.json"
    keep.write_text("KEEP\n", encoding="utf-8")
    (graph(cfg) / "_index.json").symlink_to(keep)  # inside the project, so the sandbox allows it; write_index refuses it

    write_inbox(cfg, {"a.md": A_MD})
    with pytest.raises(PipelineError):
        run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg))

    rj = report_json_for(cfg)
    assert rj["status"] == "failed"
    (event,) = [e for e in rj["events"] if e["event"] == "write_failed"]
    assert event["reason"].startswith("write: OSError"), event
    (row,) = rj["inputs"]
    assert row["status"] == "deferred" and "notes landed" in row["reason"], row
    assert len(node_files(cfg)) == 3, "the atomic batch landed before the index failed"
    assert (cfg.sandbox.root("inbox") / "a.md").is_file(), "nothing moved"
    assert not any(cfg.sandbox.root("processed").iterdir())
    assert keep.read_text(encoding="utf-8") == "KEEP\n" and (graph(cfg) / "_index.json").is_symlink()


# ----------------------------- (g) the wait budget is one per run, not one per streak (R21, D34)


def test_R21_two_separate_rate_limit_streaks_in_one_run_sleep_once_and_defer_the_second(cfg):
    """Fix a3b56a5: a.md waits once and succeeds; b.md's own first limit is not a fresh budget — it defers,
    with a `rate_limit_deferral` naming what the one wait waited for."""
    from kg.llm.errors import RateLimited

    sleeps: list[float] = []
    limited = {"a": False, "b": False}

    def atomize(call):
        paths = [u["path"] for u in parse_rendered_chunk(call["user_text"])]
        key = "a" if "Sample Space" in paths else "b"
        if not limited[key]:
            limited[key] = True
            return RateLimited("rate limited", action="wait", wait_seconds=5, resets_at=RESETS_AT, rate_limit_type="five_hour")
        return smart_atomize(call)

    write_inbox(cfg, {"a.md": A_MD, "b.md": B_MD})
    fake = ScriptedCallStage(default_handlers(cfg, atomize=atomize), cfg=cfg)
    res = run_pipeline(cfg, fake, sleep=sleeps.append)

    assert sleeps == [5], "exactly one sleep in the whole run"
    assert len(fake.calls_for("atomize")) == 3, "a.md: limit + retry; b.md: limit, then no retry"
    rj = report_json(res)
    assert {i["name"]: i["status"] for i in rj["inputs"]} == {"a.md": "ingested", "b.md": "deferred"}
    assert [e["file"] for e in rj["events"] if e["event"] == "rate_limit"] == ["a.md", "b.md"]
    (deferral,) = [e for e in rj["events"] if e["event"] == "rate_limit_deferral"]
    assert deferral["file"] == "b.md" and deferral["remaining_files"] == 1
    assert deferral["waited_until"] == RESETS_AT_ISO, "the second streak's deferral names what the one wait waited for"
    assert any(w == f"rate limit: waited once until {RESETS_AT_ISO}; second limit → 1 remaining file(s) deferred" for w in rj["warnings"]), rj["warnings"]
    assert res.status == "partial"
    assert (cfg.sandbox.root("processed") / "a.md").is_file() and (cfg.sandbox.root("inbox") / "b.md").is_file()
    assert sorted(n.title for n in load_nodes(cfg)) == ["Event", "Probability Measure", "Sample Space"]


# ------------------------------------- (h) a first-hit `defer` is an explicit deferral event (R21)


def test_R21_first_hit_defer_records_a_rate_limit_deferral_with_no_waited_until_and_no_sleep(cfg):
    """Fix a3b56a5: a reset beyond the wait cap on the very first limit is still a `rate_limit_deferral`
    (`waited_until: null`) with a warning; nothing sleeps."""
    from kg.llm.errors import RateLimited

    sleeps: list[float] = []

    def defer_now(call):
        return RateLimited("rate limited", action="defer", wait_seconds=0, resets_at=None, rate_limit_type="seven_day")

    write_inbox(cfg, {"a.md": A_MD, "b.md": B_MD})
    fake = ScriptedCallStage(default_handlers(cfg, atomize=defer_now), cfg=cfg)
    res = run_pipeline(cfg, fake, sleep=sleeps.append)

    assert sleeps == []
    assert len(fake.calls_for("atomize")) == 1, "stop after the first rejection; b.md is never called"
    rj = report_json(res)
    (deferral,) = [e for e in rj["events"] if e["event"] == "rate_limit_deferral"]
    assert deferral == {"event": "rate_limit_deferral", "file": "a.md", "waited_until": None, "remaining_files": 2}
    assert "rate limit: reset beyond the wait cap; 2 remaining file(s) deferred" in rj["warnings"], rj["warnings"]
    assert {i["name"]: i["status"] for i in rj["inputs"]} == {"a.md": "deferred", "b.md": "deferred"}
    assert all(i["reason"].startswith("model: RateLimited") for i in rj["inputs"])
    assert node_files(cfg) == [] and sorted(p.name for p in cfg.sandbox.root("inbox").iterdir()) == ["a.md", "b.md"]


# ------------------------- (i) `_review` symlinked outside the project is a write failure with a report (R15, R18)


def test_R15_R18_review_dir_symlinked_outside_the_project_fails_the_run_with_a_report_and_writes_nothing_through(cfg):
    """Fix a3b56a5: SandboxViolation no longer escapes `run()`; the refusal is a `write_failed` with a report,
    the PipelineError names it, and the outside directory stays empty."""
    from kg.pipeline import PipelineError

    outside = _outside_dir(cfg)
    (graph(cfg) / "_review").symlink_to(outside, target_is_directory=True)

    write_inbox(cfg, {"a.md": A_MD})
    with pytest.raises(PipelineError) as ei:
        run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg))
    assert "SandboxViolation" in str(ei.value)

    rj = report_json_for(cfg)
    assert rj["status"] == "failed"
    (event,) = [e for e in rj["events"] if e["event"] == "write_failed"]
    assert event["reason"].startswith("write: SandboxViolation"), event
    assert sorted(p.name for p in outside.iterdir()) == [], "nothing written through the link"
    assert (graph(cfg) / "_review").is_symlink(), "the planted link was not replaced"
    assert (cfg.sandbox.root("inbox") / "a.md").is_file() and not any(cfg.sandbox.root("processed").iterdir()), "nothing moved"
    assert {i["status"] for i in rj["inputs"]} == {"deferred"}


# ------------------------- (j) an existing note replaced by an outside symlink is `unreadable` (R5, R15, R18)


def test_R15_R18_existing_note_replaced_by_an_outside_symlink_is_unreadable_and_the_other_file_still_ingests(cfg):
    """Fix a3b56a5: the sandbox refusal on an existing note is an unreadable-note warning, not a crash; the
    target behind the link is never read or written; the new file is still processed."""
    write_inbox(cfg, {"a.md": A_MD})
    run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg))
    note = graph(cfg) / "nodes" / "sample-0001-sample-space.md"
    assert note.is_file()
    target = _outside_file(cfg)
    note.unlink()
    note.symlink_to(target)

    def edge_to_linked(call):
        ids = ids_in(call["user_text"], cfg.corpus.slug)
        return {"edges": [{"type": "related_to", "source_id": max(ids), "target_id": "sample-0001", "relevance": 50}]}

    c_md = "# Sample Space\n\nThe sample space appears again in this file.\n\n# Zeta Thing\n\nA zeta thing is a made-up concept.\n"
    write_inbox(cfg, {"c.md": c_md})
    res = run_pipeline(cfg, ScriptedCallStage(default_handlers(cfg, edges=edge_to_linked), cfg=cfg), now=NOW_2)

    assert res.status == "ok"
    rj = report_json(res)
    assert any("unreadable" in w and "sample-0001-sample-space.md" in w for w in rj["warnings"]), rj["warnings"]
    assert {i["name"]: i["status"] for i in rj["inputs"]} == {"c.md": "ingested"}
    assert (cfg.sandbox.root("processed") / "c.md").is_file()
    assert (graph(cfg) / "nodes" / "sample-0004-zeta-thing.md").is_file()
    assert target.read_text(encoding="utf-8") == "KEEP\n", "the outside target was not written"
    assert note.is_symlink(), "the planted link was not replaced"
    assert ("unreadable_note", "sample-0001") in [(r["gate"], r["target_id"]) for r in rj["rejected_edges"]], rj["rejected_edges"]
    readable = [parse(p.read_text(encoding="utf-8")) for p in node_files(cfg) if p != note]
    assert all(e.target != "sample-0001" for n in readable for e in n.edges), "no edge attached to the unreadable id"


# ------------------------------------ (k) names lose HTML comments and angle brackets (R4, R11)


def test_R4_R11_title_and_alias_lose_html_comments_and_angle_brackets(cfg, chunk):
    """Fix a3b56a5: `_name()` strips comments and drops `<`/`>` — a name becomes an H1, a frontmatter value
    and a dedup-prompt line, where a backslash escape would show."""
    from kg.stages.atomize import atomize_chunk

    title = "<!-- hidden -->Sample <b>Space</b>"
    fake = ScriptedCallStage({"atomize": [{"nodes": [wire_node(title, aliases=["<i>SS</i><!-- secret -->", "<!-- only a comment -->"])]}]}, cfg=cfg)
    (cand,) = atomize_chunk(chunk, cfg, call_stage=fake)

    assert "hidden" not in cand.title and "<!--" not in cand.title
    assert "<" not in cand.title and ">" not in cand.title
    assert cand.title.startswith("Sample") and "Space" in cand.title
    assert len(cand.aliases) == 1, "an alias that was only a comment is dropped"
    (alias,) = cand.aliases
    assert "secret" not in alias and "<" not in alias and ">" not in alias and "SS" in alias


# ---------------------- (l) an existing node's on-disk definition is sanitised before it enters the prompt (R4)


def test_R4_dedup_prompt_sanitises_an_existing_nodes_definition_containing_an_html_comment(cfg):
    """Fix a3b56a5: `_pair_text` passes on-disk definitions through `safe_prose`, so a hand-edited or
    pre-`safe_prose` note cannot smuggle hidden text into the dedup prompt."""
    from kg.stages import dedup as s6
    from kg.stages.consolidate import NodeDraft
    from kg.notes import NodeSource

    existing = simple_node("sample-0001", "Sample Space", definition="Real def. <!-- hide --> tail <b>bold</b>")
    draft = NodeDraft(
        id="sample-0002", title="The Sample Space Definition", aliases=[], definition="Formally the sample space collects every result.",
        sources=[NodeSource(locator="dup.md#heading=X", file="dup.md", quote="q")], review=[], chunk_ids=["dup.md#c1"], source_files=["dup.md"], is_new=True,
    )
    fake = ScriptedCallStage({"dedup": lambda call: {"judgements": []}}, cfg=cfg)
    s6.run([existing, draft], cfg, call_stage=fake, focus={"sample-0002"})

    (call,) = fake.calls_for("dedup")
    text = call["user_text"]
    assert "sample-0001" in text and "sample-0002" in text, "the pair was adjudicated"
    assert "<!--" not in text and "hide" not in text
    assert "Real def." in text and "tail" in text
    assert "\\<b>bold" in text and " <b>" not in text, "a bare `<` from the on-disk definition is escaped, as `safe_prose` does for model prose"
