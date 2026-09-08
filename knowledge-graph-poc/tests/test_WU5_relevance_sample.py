"""WU5 — `kg.relevance_sample`: stratified `related_to` sample for human judgment + summary (PRD R17, §8(c); DESIGN §16).

Interface (chosen, see wu45_fixtures docstring):
  DEFAULT_BANDS = ((0, 39), (40, 70), (71, 100)) ; band_label((lo, hi)) -> "lo-hi"
  INDISTINGUISHABLE_TOLERANCE = 0.10        # bands whose agreement rates all lie within this spread carry no information
  sample(graph_dir, *, n_per_band, bands=DEFAULT_BANDS, seed, now) -> SampleResult(rows, path, shortfall, bands)
  summarise(path) -> Summary(bands: {label: BandStat(agree, disagree, unfilled, rate)}, monotonic, verdict, filled, unfilled, malformed)
  verdict ∈ {"monotonic", "not monotonic", "indistinguishable", "insufficient data"}
  MIN_JUDGED_PER_BAND = 5                   # fewer judged rows than this in ANY configured band → "insufficient data"
                                            # (fix cycle 1, d38688b); the verdict tests below judge 8 rows per band
Review file: <graph>/_review/relevance-sample-<run_id>.md with a table
  | edge | source | target | relevance | band | verdict | notes |   (verdict blank; human writes agree/disagree)
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from wu45_fixtures import (
    BAND_LABELS,
    BANDS,
    GENERAL_RELEVANCES_BY_BAND,
    NOW,
    NOW_2,
    RUN_ID,
    RUN_ID_2,
    THAI_TITLE,
    band_of,
    fill_verdicts,
    sampler_pool_by_band,
    write_general_graph,
    write_sampler_graph,
)


@pytest.fixture
def small_graph(tmp_path: Path) -> Path:
    d = tmp_path / "data" / "graph"
    d.mkdir(parents=True)
    write_general_graph(d)  # 6 related_to edges: 2 per band
    return d


@pytest.fixture
def wide_graph(tmp_path: Path) -> Path:
    d = tmp_path / "wide" / "graph"
    d.mkdir(parents=True)
    write_sampler_graph(d)  # 36 related_to edges: 12 per band
    return d


def do_sample(graph_dir: Path, **kw):
    from kg.relevance_sample import sample

    kw.setdefault("n_per_band", 2)
    kw.setdefault("seed", 1)
    kw.setdefault("now", NOW)
    return sample(graph_dir, **kw)


def table_rows(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith("|") and "| verdict |" in l)
    cols = [c.strip() for c in lines[hdr].strip("|").split("|")]
    rows = []
    for line in lines[hdr + 2 :]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == len(cols):
            rows.append(dict(zip(cols, cells)))
    return rows


# ---------------------------------------------------------------- bands


def test_R17_default_bands_and_labels():
    from kg.relevance_sample import DEFAULT_BANDS, band_label

    assert tuple(tuple(b) for b in DEFAULT_BANDS) == BANDS
    assert [band_label(b) for b in DEFAULT_BANDS] == list(BAND_LABELS)


def test_R17_bands_must_be_contiguous_and_cover_0_to_100(small_graph):
    from kg.relevance_sample import sample

    with pytest.raises(ValueError):
        sample(small_graph, n_per_band=1, bands=((0, 30), (40, 100)), seed=1, now=NOW)  # gap 31–39
    with pytest.raises(ValueError):
        sample(small_graph, n_per_band=1, bands=((0, 50), (40, 100)), seed=1, now=NOW)  # overlap


# --------------------------------------------------------------- sampling


def test_R17_sample_is_stratified_with_n_per_band(wide_graph):
    res = do_sample(wide_graph, n_per_band=3)
    assert len(res.rows) == 9
    by_band = {}
    for r in res.rows:
        by_band.setdefault(r.band, []).append(r)
    assert set(by_band) == set(BAND_LABELS)
    for label, rows in by_band.items():
        assert len(rows) == 3
        assert all(band_of(r.relevance) == label for r in rows)
    assert res.shortfall == {label: 0 for label in BAND_LABELS}


def test_R17_rows_carry_ids_titles_relevance_and_both_definitions(small_graph):
    res = do_sample(small_graph, n_per_band=2)
    assert len(res.rows) == 6
    row = next(r for r in res.rows if r.relevance == 95)
    assert {row.source_id, row.target_id} == {"kb-0004", "kb-0006"}
    assert row.edge_id == "kb-0004~kb-0006", "canonical: min id ~ max id"
    assert {row.source_title, row.target_title} == {"Data Residency", THAI_TITLE}
    assert row.source_definition and row.target_definition
    assert "[[" not in row.source_definition + row.target_definition
    assert row.band == "71-100"


def test_R17_only_related_to_edges_are_sampled(small_graph):
    res = do_sample(small_graph, n_per_band=10)
    ids = {r.edge_id for r in res.rows}
    assert "kb-0001~kb-0005" not in ids, "part_of is never sampled"
    assert "kb-0005~kb-0007" not in ids, "same_as is never sampled"
    assert {r.relevance for r in res.rows} == {15, 33, 41, 65, 82, 95}


def test_R17_each_symmetric_edge_is_sampled_at_most_once(small_graph):
    res = do_sample(small_graph, n_per_band=10)
    ids = [r.edge_id for r in res.rows]
    assert len(ids) == len(set(ids)) == 6


def test_R17_band_with_fewer_edges_than_n_takes_all_and_reports_the_shortfall(small_graph):
    res = do_sample(small_graph, n_per_band=5)
    per_band = {label: sum(1 for r in res.rows if r.band == label) for label in BAND_LABELS}
    assert per_band == {"0-39": 2, "40-70": 2, "71-100": 2}
    assert res.shortfall == {"0-39": 3, "40-70": 3, "71-100": 3}
    text = res.path.read_text(encoding="utf-8")
    assert "shortfall" in text.lower()


def test_R17_same_seed_is_reproducible_and_different_seed_differs(wide_graph):
    r1 = do_sample(wide_graph, n_per_band=3, seed=7)
    r2 = do_sample(wide_graph, n_per_band=3, seed=7, now=NOW_2)
    assert [r.edge_id for r in r1.rows] == [r.edge_id for r in r2.rows]
    r3 = do_sample(wide_graph, n_per_band=3, seed=8, now=NOW_2.replace(hour=10))
    assert {r.edge_id for r in r3.rows} != {r.edge_id for r in r1.rows}
    assert r1.path != r2.path, "path carries the injected timestamp"


def test_R17_sampling_order_is_by_band_then_deterministic(wide_graph):
    res = do_sample(wide_graph, n_per_band=2)
    bands_seen = [r.band for r in res.rows]
    assert bands_seen == ["0-39", "0-39", "40-70", "40-70", "71-100", "71-100"]


def test_R17_pool_sizes_match_fixture(wide_graph):
    res = do_sample(wide_graph, n_per_band=100)
    counts = {label: sum(1 for r in res.rows if r.band == label) for label in BAND_LABELS}
    assert counts == sampler_pool_by_band() == {"0-39": 12, "40-70": 12, "71-100": 12}


# ------------------------------------------------------------ review file


def test_R17_review_file_is_written_under_review_with_the_injected_timestamp(small_graph):
    res = do_sample(small_graph)
    assert res.path == small_graph / "_review" / f"relevance-sample-{RUN_ID}.md"
    assert res.path.is_file()
    res2 = do_sample(small_graph, now=NOW_2)
    assert res2.path.name == f"relevance-sample-{RUN_ID_2}.md"


def test_R17_review_file_table_has_the_documented_columns_and_blank_verdicts(small_graph):
    res = do_sample(small_graph)
    text = res.path.read_text(encoding="utf-8")
    assert re.search(r"^\|\s*edge\s*\|\s*source\s*\|\s*target\s*\|\s*relevance\s*\|\s*band\s*\|\s*verdict\s*\|\s*notes\s*\|\s*$", text, re.MULTILINE)
    rows = table_rows(res.path)
    assert len(rows) == 6
    assert all(r["verdict"] == "" for r in rows), "human fills agree/disagree"
    assert {r["edge"] for r in rows} == {r.edge_id for r in res.rows}
    assert {int(r["relevance"]) for r in rows} == {15, 33, 41, 65, 82, 95}
    assert THAI_TITLE in text


def test_R17_review_file_includes_both_definitions_for_context_and_instructions(small_graph):
    res = do_sample(small_graph)
    text = res.path.read_text(encoding="utf-8")
    for row in res.rows:
        assert row.source_definition in text and row.target_definition in text
    assert "agree" in text and "disagree" in text, "instructions name the accepted verdict values"


def test_R17_review_file_header_records_provider_and_edge_stage_model_of_the_graph(small_graph):
    text = do_sample(small_graph).path.read_text(encoding="utf-8")
    assert "openrouter" in text and "anthropic/claude-sonnet-5" in text
    assert "seed: 1" in text.replace("**", "") or "seed=1" in text


def test_R17_graph_without_related_to_edges_yields_an_empty_sample_and_still_writes_the_file(tmp_path):
    from wu45_fixtures import write_education_graph

    d = tmp_path / "edu" / "graph"
    d.mkdir(parents=True)
    write_education_graph(d)
    res = do_sample(d)
    assert res.rows == []
    assert res.path.is_file()
    assert res.shortfall == {"0-39": 2, "40-70": 2, "71-100": 2}


def test_R17_n_per_band_must_be_positive(small_graph):
    with pytest.raises(ValueError):
        do_sample(small_graph, n_per_band=0)


# --------------------------------------------------------------- summary


def summarise(path: Path):
    from kg.relevance_sample import summarise as _s

    return _s(path)


def test_R17_summary_of_a_filled_file_computes_agreement_per_band(wide_graph):
    res = do_sample(wide_graph, n_per_band=8)  # ≥ MIN_JUDGED_PER_BAND so a verdict is reachable
    # low band: 2/8 agree; mid: 4/8; high: 8/8  → 0.25 / 0.50 / 1.00, monotonic
    counters = {label: 0 for label in BAND_LABELS}
    want = {"0-39": 2, "40-70": 4, "71-100": 8}

    def verdict(rel, band):
        counters[band] += 1
        return "agree" if counters[band] <= want[band] else "disagree"

    fill_verdicts(res.path, verdict)
    s = summarise(res.path)
    assert set(s.bands) == set(BAND_LABELS)
    assert (s.bands["0-39"].agree, s.bands["0-39"].disagree) == (2, 6) and s.bands["0-39"].rate == pytest.approx(0.25)
    assert s.bands["40-70"].rate == pytest.approx(0.5)
    assert s.bands["71-100"].rate == pytest.approx(1.0)
    assert s.filled == 24 and s.unfilled == 0
    assert s.malformed == 0
    assert s.monotonic is True
    assert s.verdict == "monotonic"


def test_R17_non_monotonic_rates_are_reported_as_such(wide_graph):
    res = do_sample(wide_graph, n_per_band=8)
    counters = {label: 0 for label in BAND_LABELS}
    want = {"0-39": 8, "40-70": 2, "71-100": 6}  # 1.0, 0.25, 0.75

    def verdict(rel, band):
        counters[band] += 1
        return "agree" if counters[band] <= want[band] else "disagree"

    fill_verdicts(res.path, verdict)
    s = summarise(res.path)
    assert s.monotonic is False
    assert s.verdict == "not monotonic"


def test_R17_indistinguishable_bands_carry_no_information(wide_graph):
    from kg.relevance_sample import INDISTINGUISHABLE_TOLERANCE

    assert 0 < INDISTINGUISHABLE_TOLERANCE < 0.5
    res = do_sample(wide_graph, n_per_band=8)
    fill_verdicts(res.path, lambda rel, band: "agree")  # 1.0 / 1.0 / 1.0
    s = summarise(res.path)
    assert all(b.rate == 1.0 for b in s.bands.values())
    assert s.verdict == "indistinguishable", "flat rates are non-decreasing but the score carries no information (PRD §8(c))"


def test_R17_fewer_than_min_judged_rows_in_a_band_is_insufficient_data(wide_graph):
    """Fix cycle 1: a monotonic pattern on 4 judged rows per band is not evidence; the verdict needs MIN_JUDGED_PER_BAND."""
    from kg.relevance_sample import MIN_JUDGED_PER_BAND

    assert MIN_JUDGED_PER_BAND == 5
    res = do_sample(wide_graph, n_per_band=MIN_JUDGED_PER_BAND - 1)
    counters = {label: 0 for label in BAND_LABELS}
    want = {"0-39": 1, "40-70": 2, "71-100": 4}  # 0.25 / 0.50 / 1.00 — would be monotonic with enough rows

    def verdict(rel, band):
        counters[band] += 1
        return "agree" if counters[band] <= want[band] else "disagree"

    fill_verdicts(res.path, verdict)
    s = summarise(res.path)
    assert s.filled == 12 and s.unfilled == 0
    assert all(b.agree + b.disagree == 4 for b in s.bands.values())
    assert s.bands["0-39"].rate == pytest.approx(0.25) and s.bands["71-100"].rate == pytest.approx(1.0), "rates are still reported"
    assert s.verdict == "insufficient data"
    assert s.monotonic is False


def test_R17_equal_rates_within_tolerance_are_indistinguishable_even_if_ordered(wide_graph):
    from kg.relevance_sample import INDISTINGUISHABLE_TOLERANCE

    res = do_sample(wide_graph, n_per_band=10)
    counters = {label: 0 for label in BAND_LABELS}
    # 7/10, 7/10, 8/10 → spread 0.10 == tolerance → indistinguishable
    want = {"0-39": 7, "40-70": 7, "71-100": 8}

    def verdict(rel, band):
        counters[band] += 1
        return "agree" if counters[band] <= want[band] else "disagree"

    fill_verdicts(res.path, verdict)
    s = summarise(res.path)
    rates = [s.bands[l].rate for l in BAND_LABELS]
    assert max(rates) - min(rates) <= INDISTINGUISHABLE_TOLERANCE + 1e-9
    assert s.verdict == "indistinguishable"


def test_R17_unfilled_rows_are_counted_separately_and_excluded_from_rates(wide_graph):
    res = do_sample(wide_graph, n_per_band=4)
    counters = {label: 0 for label in BAND_LABELS}

    def verdict(rel, band):
        counters[band] += 1
        if counters[band] == 1:
            return ""  # left blank by the reviewer
        return "agree" if band == "71-100" else "disagree"

    fill_verdicts(res.path, verdict)
    s = summarise(res.path)
    assert s.unfilled == 3 and s.filled == 9
    assert all(b.unfilled == 1 for b in s.bands.values())
    assert s.bands["0-39"].rate == 0.0 and s.bands["71-100"].rate == 1.0


def test_R17_verdict_parsing_is_case_and_whitespace_insensitive(wide_graph):
    res = do_sample(wide_graph, n_per_band=2)
    fill_verdicts(res.path, lambda rel, band: " Agree " if band == "71-100" else "DISAGREE")
    s = summarise(res.path)
    assert s.bands["71-100"].agree == 2 and s.bands["0-39"].disagree == 2 and s.unfilled == 0


def test_R17_unrecognised_verdict_text_counts_as_unfilled_not_as_agreement(wide_graph):
    res = do_sample(wide_graph, n_per_band=2)
    fill_verdicts(res.path, lambda rel, band: "maybe")
    s = summarise(res.path)
    assert s.filled == 0 and s.unfilled == 6
    assert all(b.rate is None for b in s.bands.values())
    assert s.monotonic is False
    assert s.verdict == "insufficient data"


def test_R17_summary_of_an_untouched_file_is_insufficient_data(small_graph):
    res = do_sample(small_graph)
    s = summarise(res.path)
    assert s.filled == 0 and s.unfilled == 6
    assert s.verdict == "insufficient data"


def test_R17_summary_rejects_a_file_without_the_review_table(tmp_path):
    bad = tmp_path / "notes.md"
    bad.write_text("# just notes\n\nno table here\n", encoding="utf-8")
    with pytest.raises(ValueError):
        summarise(bad)


def test_R17_summary_rejects_missing_file(tmp_path):
    with pytest.raises((FileNotFoundError, OSError)):
        summarise(tmp_path / "nope.md")
