"""WU4/WU5 fix cycle 1 — regression pins for commits 854eaf2..84cb01d (PRD R14, R16, R17).

Each test names the behaviour the fix introduced and would fail on 3b2f9a7:

  R16 stability   (a) same file failing conversion in BOTH legs → valid, warning names it, legs.a.failed == legs.b.failed
                  (b) a file failing conversion in ONE leg → invalid, reason mentions "conversion failed for different files"
                  (c) a model-stage `deferred` file in EITHER leg → invalid
                  (i) two runs with the same `now` → the second raises FileExistsError; the first report is intact
                  (j) served-model mismatch → invalid by default; `allow_served_drift=True` → valid + warning + `served_drift`
                  (n) `kg stability` on the subscription provider prints an estimated CALL count = legs × items × stages
  R17 sampler     (d) a configured band with 0 judged rows → "insufficient data" even when the other bands are monotonic
                  (e) a column-aligned header (`|  edge   | … |  verdict  |`) parses
                  (f) a row with an extra `|` is parsed, counted in `malformed`, and its verdict still counts
                  (g) `Agree.` / `agree - obviously` → agree; `maybe` → unfilled
  R14 render      (h) `kg render` twice → byte-identical graph.html AND _index.json, no `generated`; `--stamp` → `generated`
                  (k) `render(..., sandbox=cfg.sandbox)` with `out_path` outside the graph root → SandboxViolation
                  (l) a corpus name containing `__KG_GRAPH_JSON__` appears literally in <title> (single-pass substitution)
                  (m) the page carries a CSP meta with `default-src 'none'`

Offline, deterministic, zero tokens: model calls go through `ScriptedCallStage`, web fetches through an injected `fetch`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from kg.cli import main
from kg.config import STAGES, load_config
from kg.schemas import GENERAL
from wu3_fixtures import A_MD, ScriptedCallStage, default_handlers, general_edges_handler, write_inbox
from wu45_fixtures import BAND_LABELS, NOW, RUN_ID, fill_verdicts, graph_payload, note_spec, write_general_graph, write_graph, write_sampler_graph


@pytest.fixture
def cfg_path(make_project) -> Path:
    return make_project()


@pytest.fixture
def cfg(cfg_path):
    return load_config(cfg_path)


def leg_of(call: dict) -> str:
    c = call["extra"].get("cfg")
    assert c is not None, "pipeline must pass cfg= to call_stage (DESIGN §7.1)"
    return "b" if "b" in Path(c.sandbox.root("graph")).parts[-2:] else "a"


def run_stability(cfg, fake, **kw):
    from kg.stability import run

    return run(cfg, call_stage=fake, now=kw.pop("now", NOW), **kw)


BROKEN_PDF = b"%PDF-1.4\nnot a pdf\n%%EOF\n"  # deterministic conversion failure (same fixture as WU1)


# ------------------------------------------------------------ R16 stability


def test_fix_R16_same_conversion_failure_in_both_legs_is_a_warning_not_an_invalidation(cfg):
    write_inbox(cfg, {"a.md": A_MD})
    (cfg.sandbox.root("inbox") / "broken.pdf").write_bytes(BROKEN_PDF)
    res = run_stability(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg))

    assert res.legs["a"].failed == ["broken.pdf"] == res.legs["b"].failed
    assert res.legs["a"].deferred == [] and res.legs["b"].deferred == []
    assert res.valid is True, "excluded identically on both sides — the comparison stays valid"
    assert res.invalid_reasons == []
    assert any("broken.pdf" in w for w in res.warnings), "the warning names the file"
    md = Path(res.report_md).read_text(encoding="utf-8")
    assert "INVALID" not in md and "broken.pdf" in md
    rj = json.loads(Path(res.report_json).read_text(encoding="utf-8"))
    assert rj["legs"]["a"]["failed"] == ["broken.pdf"] and rj["valid"] is True


def test_fix_R16_conversion_failure_in_one_leg_only_invalidates_and_says_so(cfg):
    write_inbox(cfg, {"a.md": A_MD, "urls.txt": "https://example.org/web-concept\n"})
    page = "<html><body><h1>Web Concept</h1><p>A web concept is a concept fetched from the web.</p></body></html>"
    seen: list[str] = []

    def fetch(url: str):
        seen.append(url)
        return page if len(seen) == 1 else None  # leg a fetches fine; leg b gets nothing back → conversion fails there only

    res = run_stability(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg), fetch=fetch)
    assert len(seen) == 2, "one fetch per leg"
    assert res.legs["a"].failed == [] and res.legs["b"].failed == ["https://example.org/web-concept"]
    assert res.valid is False
    assert any("conversion failed for different files" in r for r in res.invalid_reasons), res.invalid_reasons
    assert "INVALID" in Path(res.report_md).read_text(encoding="utf-8")


@pytest.mark.parametrize("bad_leg", ["a", "b"])
def test_fix_R16_deferred_file_in_either_leg_invalidates(cfg, bad_leg):
    def edges_fail(call):
        if leg_of(call) == bad_leg:
            raise RuntimeError("provider unavailable")
        return {"edges": []}

    write_inbox(cfg, {"a.md": A_MD})
    res = run_stability(cfg, ScriptedCallStage(default_handlers(cfg, edges=edges_fail), cfg=cfg))
    other = "b" if bad_leg == "a" else "a"
    assert res.legs[bad_leg].deferred == ["a.md"] and res.legs[other].deferred == []
    assert res.legs[bad_leg].failed == [] and res.legs[other].failed == [], "deferred is a model-stage outcome, not a conversion failure"
    assert res.valid is False
    assert any(f"leg {bad_leg}" in r and "a.md" in r for r in res.invalid_reasons), res.invalid_reasons


def test_fix_R16_second_run_in_the_same_second_refuses_and_leaves_the_first_report_intact(cfg):
    write_inbox(cfg, {"a.md": A_MD})
    first = run_stability(cfg, ScriptedCallStage(default_handlers(cfg, edges=general_edges_handler(cfg.corpus.slug)), cfg=cfg))
    md_before, json_before = Path(first.report_md).read_bytes(), Path(first.report_json).read_bytes()

    with pytest.raises(FileExistsError) as exc:
        run_stability(cfg, ScriptedCallStage(default_handlers(cfg), cfg=cfg))  # same NOW → same run_id
    assert RUN_ID in str(exc.value)
    assert Path(first.report_md).read_bytes() == md_before
    assert Path(first.report_json).read_bytes() == json_before
    assert (cfg.sandbox.root("inbox") / "a.md").is_file(), "the refused run touched nothing"


def _drifting_call_stage(cfg):
    """Scripted call_stage whose leg-b responses report a different served model (router drift)."""
    inner = ScriptedCallStage(default_handlers(cfg), cfg=cfg)

    def call_stage(stage, system, user_text, schema, images=None, **kw):
        ledger = kw.pop("ledger", None)
        r = inner(stage, system, user_text, schema, images, **kw)
        if leg_of(inner.calls[-1]) == "b":
            r.model_served = f"{r.model_requested}-DIFFERENT-BUILD"
        if ledger is not None:
            ScriptedCallStage._record(ledger, stage, r.provider, r.model_requested, r.model_served, kw.get("prompt_version"))
        return r

    return call_stage


def test_fix_R16_served_model_mismatch_is_invalid_by_default(cfg):
    write_inbox(cfg, {"a.md": A_MD})
    res = run_stability(cfg, _drifting_call_stage(cfg))
    assert res.valid is False
    assert any("served" in r for r in res.invalid_reasons)
    assert res.served_drift and set(res.served_drift) <= set(STAGES)


def test_fix_R16_allow_served_drift_turns_the_mismatch_into_a_warning_and_records_the_stages(cfg):
    write_inbox(cfg, {"a.md": A_MD})
    res = run_stability(cfg, _drifting_call_stage(cfg), allow_served_drift=True)
    assert res.valid is True and res.invalid_reasons == []
    assert "atomize" in res.served_drift, "every model stage that ran in both legs drifted"
    assert set(res.served_drift) <= set(STAGES)
    assert any("served" in w and "allow" in w for w in res.warnings), res.warnings
    rj = json.loads(Path(res.report_json).read_text(encoding="utf-8"))
    assert rj["valid"] is True and rj["served_drift"] == res.served_drift
    md = Path(res.report_md).read_text(encoding="utf-8")
    assert "INVALID" not in md and "drift" in md.lower()


def test_fix_R16_cli_subscription_warning_estimates_calls_as_legs_times_items_times_stages(cfg_path, cfg, monkeypatch, capsys):
    import kg.llm
    import kg.stability

    write_inbox(cfg, {"a.md": A_MD, "b.md": "# B\n\nB text.\n", "c.md": "# C\n\nC text.\n"})
    assert cfg.llm.provider == "claude_subscription"
    fake_result = SimpleNamespace(
        run_id=RUN_ID,
        jaccard=1.0,
        overlap_a=1.0,
        overlap_b=1.0,
        fuzzy_jaccard=1.0,
        passed=True,
        threshold=0.70,
        valid=True,
        invalid_reasons=[],
        only_in_a=[],
        only_in_b=[],
        common=["x"],
        relevance_deltas=SimpleNamespace(n=0),
        report_md=cfg.sandbox.root("runs") / f"stability-{RUN_ID}" / "stability-report.md",
    )
    monkeypatch.setattr(kg.llm, "call_stage", object())
    monkeypatch.setattr(kg.stability, "run", lambda *a, **kw: fake_result)

    assert main(["stability", "--config", str(cfg_path)]) == 0
    out = capsys.readouterr().out
    expected = len(kg.stability.LEGS) * 3 * len(STAGES)
    assert expected == 24
    assert re.search(rf"\b{expected}\b\s+model call", out), out
    assert "2 legs" in out and "3 inbox item(s)" in out and f"{len(STAGES)} stage(s)" in out


def test_fix_R16_cli_forwards_allow_served_drift(cfg_path, cfg, monkeypatch):
    import kg.llm
    import kg.stability

    calls: list[dict] = []

    def fake_run(*a, **kw):
        calls.append(kw)
        return SimpleNamespace(
            run_id=RUN_ID, jaccard=1.0, overlap_a=1.0, overlap_b=1.0, fuzzy_jaccard=1.0, passed=True, threshold=0.7, valid=True,
            invalid_reasons=[], only_in_a=[], only_in_b=[], common=[], relevance_deltas=SimpleNamespace(n=0), report_md=Path("r.md"),
        )

    monkeypatch.setattr(kg.llm, "call_stage", object())
    monkeypatch.setattr(kg.stability, "run", fake_run)
    assert main(["stability", "--config", str(cfg_path)]) == 0
    assert calls[-1]["allow_served_drift"] is False
    assert main(["stability", "--allow-served-drift", "--config", str(cfg_path)]) == 0
    assert calls[-1]["allow_served_drift"] is True


# -------------------------------------------------------------- R17 sampler


@pytest.fixture
def wide_graph(tmp_path: Path) -> Path:
    d = tmp_path / "wide" / "graph"
    d.mkdir(parents=True)
    write_sampler_graph(d)  # 36 related_to edges: 12 per band
    return d


@pytest.fixture
def small_graph(tmp_path: Path) -> Path:
    d = tmp_path / "data" / "graph"
    d.mkdir(parents=True)
    write_general_graph(d)  # 6 related_to edges: 2 per band
    return d


def summarise(path: Path):
    from kg.relevance_sample import summarise as _s

    return _s(path)


def sample(graph_dir: Path, n_per_band: int):
    from kg.relevance_sample import sample as _sample

    return _sample(graph_dir, n_per_band=n_per_band, seed=1, now=NOW)


def review_file(tmp_path: Path, table_lines: list[str], *, bands_header: str = "- bands: 0-39, 40-70, 71-100") -> Path:
    p = tmp_path / "relevance-sample-test.md"
    p.write_text("\n".join(["# Relevance judgment sample — test", "", bands_header, "", "## Sample", "", *table_lines, ""]), encoding="utf-8")
    return p


def test_fix_R17_configured_band_with_zero_judged_rows_is_insufficient_even_if_the_rest_is_monotonic(wide_graph):
    res = sample(wide_graph, n_per_band=5)
    counters = {label: 0 for label in BAND_LABELS}
    want = {"0-39": 0, "40-70": 3, "71-100": 5}  # 40-70 → 0.6, 71-100 → 1.0: monotonic on their own

    def verdict(rel, band):
        counters[band] += 1
        return "agree" if counters[band] <= want[band] else "disagree"

    fill_verdicts(res.path, verdict)
    # The reviewer deletes every low-band row from the table (the header still says the band was configured).
    lines = res.path.read_text(encoding="utf-8").splitlines()
    kept = [l for l in lines if not (l.startswith("|") and len(l.strip("|").split("|")) == 7 and l.split("|")[5].strip() == "0-39")]
    assert len(kept) == len(lines) - 5
    res.path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    s = summarise(res.path)
    assert set(s.bands) == set(BAND_LABELS), "an emptied band never vanishes from the summary"
    assert s.bands["0-39"].agree + s.bands["0-39"].disagree == 0 and s.bands["0-39"].rate is None
    assert s.bands["40-70"].rate == pytest.approx(0.6) and s.bands["71-100"].rate == pytest.approx(1.0)
    assert s.verdict == "insufficient data"
    assert s.monotonic is False


def aligned_rows() -> list[str]:
    rows = ["|  edge   |  source  |  target  |  relevance  |  band   |  verdict  |  notes  |", "|---------|----------|----------|-------------|---------|-----------|---------|"]
    rel_by_band = {"0-39": 10, "40-70": 50, "71-100": 90}
    for band, rel in rel_by_band.items():
        for i in range(5):
            rows.append(f"|  x-{i:04d}~y-{i:04d}  |  x-{i:04d} X  |  y-{i:04d} Y  |  {rel:<9}  |  {band:<6} |  agree    |         |")
    return rows


def test_fix_R17_column_aligned_table_parses(tmp_path):
    p = review_file(tmp_path, aligned_rows())
    s = summarise(p)
    assert s.filled == 15 and s.unfilled == 0 and s.malformed == 0
    assert all(b.agree == 5 for b in s.bands.values())
    assert s.verdict == "indistinguishable", "1.0 / 1.0 / 1.0 with 5 judged per band"


def test_fix_R17_row_with_an_extra_pipe_is_parsed_and_counted_as_malformed(tmp_path):
    rows = aligned_rows()
    rows.append("| e-0001~f-0001 | e-0001 E | f-0001 F | 55 | 40-70 | disagree | note with a | stray pipe |")
    s = summarise(review_file(tmp_path, rows))
    assert s.malformed == 1
    assert s.bands["40-70"].disagree == 1 and s.bands["40-70"].agree == 5, "the row's verdict still counts; the tail folds into notes"
    assert s.filled == 16 and s.unfilled == 0


def test_fix_R17_verdict_matches_on_the_first_word(small_graph):
    from kg.relevance_sample import classify_verdict

    assert classify_verdict("Agree.") == "agree"
    assert classify_verdict("agree - obviously") == "agree"
    assert classify_verdict("DISAGREE, too high") == "disagree"
    assert classify_verdict("maybe") is None
    assert classify_verdict("") is None

    res = sample(small_graph, n_per_band=2)
    fill_verdicts(res.path, lambda rel, band: {"0-39": "Agree.", "40-70": "agree - obviously", "71-100": "maybe"}[band])
    s = summarise(res.path)
    assert s.bands["0-39"].agree == 2 and s.bands["40-70"].agree == 2
    assert s.bands["71-100"].unfilled == 2 and s.bands["71-100"].agree == 0
    assert s.filled == 4 and s.unfilled == 2


def test_fix_R17_review_file_header_declares_the_configured_bands(small_graph):
    from kg.relevance_sample import BANDS_HEADER_PREFIX

    res = sample(small_graph, n_per_band=1)
    text = res.path.read_text(encoding="utf-8")
    assert re.search(rf"^{re.escape(BANDS_HEADER_PREFIX)}\s*0-39, 40-70, 71-100\s*$", text, re.MULTILINE), "summarise reads the configured bands from here"


# ---------------------------------------------------------------- R14 render


def test_fix_R14_kg_render_is_byte_identical_by_default_and_stamps_only_on_request(cfg_path, cfg, capsys):
    graph_dir = cfg.sandbox.root("graph")
    write_general_graph(graph_dir)
    html_path, index_path = graph_dir / "graph.html", graph_dir / "_index.json"

    assert main(["render", "--config", str(cfg_path)]) == 0
    html1, idx1 = html_path.read_bytes(), index_path.read_bytes()
    assert main(["render", "--config", str(cfg_path)]) == 0
    assert html_path.read_bytes() == html1 and index_path.read_bytes() == idx1
    assert "generated" not in graph_payload(html1.decode("utf-8"))["meta"]
    assert "generated" not in json.loads(idx1)["meta"]

    assert main(["render", "--stamp", "--config", str(cfg_path)]) == 0
    stamped = graph_payload(html_path.read_text(encoding="utf-8"))["meta"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", stamped["generated"])
    assert "generated" in json.loads(index_path.read_text(encoding="utf-8"))["meta"]
    assert str(html_path) in capsys.readouterr().out


def test_fix_R14_render_with_a_sandbox_refuses_an_out_path_outside_the_graph_root(cfg, tmp_path):
    from kg.paths import SandboxViolation
    from kg.render.html import render

    graph_dir = cfg.sandbox.root("graph")
    write_general_graph(graph_dir)
    for out in (cfg.sandbox.root("runs") / "graph.html", tmp_path / "elsewhere" / "graph.html", cfg.project_root / "graph.html"):
        with pytest.raises(SandboxViolation):
            render(graph_dir, out, schema=GENERAL, sandbox=cfg.sandbox)
        assert not out.exists()
    inside = graph_dir / "graph.html"
    assert render(graph_dir, inside, schema=GENERAL, sandbox=cfg.sandbox) == inside and inside.is_file()


def test_fix_R14_corpus_name_containing_a_placeholder_is_inserted_verbatim(tmp_path):
    from kg.render.html import render

    d = tmp_path / "hostile" / "graph"
    d.mkdir(parents=True)
    write_graph(d, [note_spec("h-0001", "Harmless", schema="general")], corpus="__KG_GRAPH_JSON__", schema="general")
    out = d / "graph.html"
    html = render(d, out, schema=GENERAL).read_text(encoding="utf-8")
    title = re.search(r"<title>(.*?)</title>", html, re.DOTALL)
    assert title and title.group(1) == "__KG_GRAPH_JSON__ — knowledge graph", "no second-pass expansion into the payload"
    payload = graph_payload(html)  # exactly one well-formed data block
    assert payload["meta"]["corpus"] == "__KG_GRAPH_JSON__" and [n["id"] for n in payload["nodes"]] == ["h-0001"]
    assert html.count('id="graph-data"') == 1


def test_fix_R14_page_carries_a_csp_that_denies_everything_by_default(tmp_path):
    from kg.render.html import render

    d = tmp_path / "data" / "graph"
    d.mkdir(parents=True)
    write_general_graph(d)
    html = render(d, d / "graph.html", schema=GENERAL).read_text(encoding="utf-8")
    meta = re.search(r"<meta\s+http-equiv=\"Content-Security-Policy\"\s+content=\"([^\"]*)\"", html, re.IGNORECASE)
    assert meta, "CSP meta tag missing"
    assert "default-src 'none'" in meta.group(1)
    assert html.index(meta.group(0)) < html.index("<script"), "the policy is declared before any script runs"
