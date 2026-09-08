"""WU3 — `kg.stages.atomize` (S3) + S3b quote grounding (DESIGN §6, §8.0; D8; PRD R2, R4).

One model call per chunk; output candidates carry unit_ids + quotes; every quote must be a
whitespace-normalised substring of the unit it cites, else the node is flagged
`review: quote_not_found` — grounding is never a repair trigger (§8.0).

Interface (chosen): atomize_chunk(chunk, cfg, *, call_stage, ledger=None) -> list[Candidate];
atomize(chunks, cfg, *, call_stage, ledger=None) -> list[Candidate];
Candidate(title, definition, aliases, unit_ids, quotes, source, chunk_id, review, sources: list[NodeSource]).
"""

from __future__ import annotations

import pytest

from kg.config import load_config
from kg.schemas import AtomizeOutput
from wu3_fixtures import ScriptedCallStage, make_chunk, make_unit, smart_atomize

U1_TEXT = "The sample space is the set of all possible outcomes\nof a random experiment."
U2_TEXT = "An event is a subset of the sample space."
U3_TEXT = "A random variable maps each outcome to a real number."


@pytest.fixture
def cfg(make_project):
    return load_config(make_project())


@pytest.fixture
def chunk():
    return make_chunk("a.md#c1", [make_unit("U1", U1_TEXT, heading=["Sample Space"]), make_unit("U2", U2_TEXT, heading=["Event"])])


@pytest.fixture
def chunk_b():
    return make_chunk("b.md#c1", [make_unit("U3", U3_TEXT, source="b.md", heading=["Random Variable"])])


def cand(title, unit_ids, quotes, definition="A definition.", aliases=()):
    return {"title": title, "definition": definition, "aliases": list(aliases), "unit_ids": list(unit_ids), "quotes": [{"unit_id": u, "text": t} for u, t in quotes]}


# ------------------------------------------------------------- one call/chunk


def test_S3_one_call_per_chunk_with_the_atomize_schema_and_rendered_units(cfg, chunk, chunk_b):
    from kg.stages.atomize import atomize

    fake = ScriptedCallStage({"atomize": smart_atomize}, cfg=cfg)
    out = atomize([chunk, chunk_b], cfg, call_stage=fake)
    assert len(fake.calls) == 2
    for call in fake.calls:
        assert call["stage"] == "atomize"
        assert call["schema"] is AtomizeOutput
        assert call["images"] in (None, [])
    assert "<<U1 | a.md#heading=Sample Space | Sample Space>>" in fake.calls[0]["user_text"]
    assert "<<U2 | a.md#heading=Event | Event>>" in fake.calls[0]["user_text"]
    assert U1_TEXT.splitlines()[0] in fake.calls[0]["user_text"]
    assert "<<U3 | b.md#heading=Random Variable | Random Variable>>" in fake.calls[1]["user_text"]
    assert [c.source for c in out] == ["a.md", "a.md", "b.md"]
    assert [c.chunk_id for c in out] == ["a.md#c1", "a.md#c1", "b.md#c1"]


def test_S3_call_passes_cfg_and_prompt_version_through_the_boundary(cfg, chunk):
    from kg.stages.atomize import atomize_chunk

    fake = ScriptedCallStage({"atomize": smart_atomize}, cfg=cfg)
    atomize_chunk(chunk, cfg, call_stage=fake)
    extra = fake.calls[0]["extra"]
    assert extra.get("cfg") is cfg
    assert "ledger" in extra, "the ledger is forwarded (None allowed at stage level; the pipeline passes a UsageLedger)"
    assert str(extra.get("prompt_version", "")).startswith("atomize@")


def test_S3_empty_chunk_list_makes_no_calls(cfg):
    from kg.stages.atomize import atomize

    fake = ScriptedCallStage({"atomize": smart_atomize}, cfg=cfg)
    assert atomize([], cfg, call_stage=fake) == []
    assert fake.calls == []


def test_S3_model_returning_no_nodes_is_valid_and_yields_no_candidates(cfg, chunk):
    from kg.stages.atomize import atomize_chunk

    fake = ScriptedCallStage({"atomize": [{"nodes": []}]}, cfg=cfg)
    assert atomize_chunk(chunk, cfg, call_stage=fake) == []


# ------------------------------------------------------------ candidates


def test_S3_candidates_carry_unit_ids_quotes_and_derived_sources(cfg, chunk):
    from kg.stages.atomize import atomize_chunk

    fake = ScriptedCallStage(
        {"atomize": [{"nodes": [cand("Sample Space", ["U1"], [("U1", "the set of all possible outcomes")], aliases=["Outcome space"])]}]},
        cfg=cfg,
    )
    (c,) = atomize_chunk(chunk, cfg, call_stage=fake)
    assert c.title == "Sample Space"
    assert c.definition == "A definition."
    assert list(c.aliases) == ["Outcome space"]
    assert list(c.unit_ids) == ["U1"]
    assert [(q.unit_id, q.text) for q in c.quotes] == [("U1", "the set of all possible outcomes")]
    assert c.review == []
    # D8: locator derived deterministically from the cited unit, never model-authored
    assert [(s.locator, s.file, s.quote) for s in c.sources] == [("a.md#heading=Sample Space", "a.md", "the set of all possible outcomes")]


def test_S3_node_drawing_on_two_units_gets_one_source_per_quote(cfg, chunk):
    from kg.stages.atomize import atomize_chunk

    fake = ScriptedCallStage(
        {"atomize": [{"nodes": [cand("Sample Space", ["U1", "U2"], [("U1", "all possible outcomes"), ("U2", "subset of the sample space")])]}]},
        cfg=cfg,
    )
    (c,) = atomize_chunk(chunk, cfg, call_stage=fake)
    assert [s.locator for s in c.sources] == ["a.md#heading=Sample Space", "a.md#heading=Event"]
    assert c.review == []


# ----------------------------------------------------------- S3b grounding


def test_R4_quote_with_different_whitespace_still_grounds(cfg, chunk):
    from kg.stages.atomize import atomize_chunk

    quote = "set  of all\npossible   outcomes of a random"  # line break + runs of spaces vs the unit text
    fake = ScriptedCallStage({"atomize": [{"nodes": [cand("Sample Space", ["U1"], [("U1", quote)])]}]}, cfg=cfg)
    (c,) = atomize_chunk(chunk, cfg, call_stage=fake)
    assert c.review == []
    assert len(c.sources) == 1


def test_R4_quote_not_in_unit_flags_quote_not_found_and_never_triggers_a_repair(cfg, chunk):
    from kg.stages.atomize import atomize_chunk

    fake = ScriptedCallStage({"atomize": [{"nodes": [cand("Sample Space", ["U1"], [("U1", "this sentence is not in the source")])]}]}, cfg=cfg)
    (c,) = atomize_chunk(chunk, cfg, call_stage=fake)
    assert c.review == ["quote_not_found"]
    assert len(fake.calls) == 1, "grounding failure is a review flag, not a repair-retry"
    assert c.sources == [] or all(s.quote == "" for s in c.sources)


def test_R4_paraphrased_quote_does_not_ground(cfg, chunk):
    from kg.stages.atomize import atomize_chunk

    fake = ScriptedCallStage({"atomize": [{"nodes": [cand("Sample Space", ["U1"], [("U1", "the sample space is every possible outcome")])]}]}, cfg=cfg)
    (c,) = atomize_chunk(chunk, cfg, call_stage=fake)
    assert c.review == ["quote_not_found"]


def test_R4_one_bad_quote_among_good_ones_is_dropped_without_flagging(cfg, chunk):
    from kg.stages.atomize import atomize_chunk

    fake = ScriptedCallStage(
        {"atomize": [{"nodes": [cand("Sample Space", ["U1", "U2"], [("U1", "all possible outcomes"), ("U2", "not present anywhere")])]}]},
        cfg=cfg,
    )
    (c,) = atomize_chunk(chunk, cfg, call_stage=fake)
    assert [q.text for q in c.quotes] == ["all possible outcomes"]
    assert [s.locator for s in c.sources] == ["a.md#heading=Sample Space"]
    assert c.review == []


def test_R4_quote_citing_a_unit_outside_the_chunk_is_dropped(cfg, chunk):
    from kg.stages.atomize import atomize_chunk

    fake = ScriptedCallStage(
        {"atomize": [{"nodes": [cand("Sample Space", ["U1", "U9"], [("U9", "all possible outcomes"), ("U1", "all possible outcomes")])]}]},
        cfg=cfg,
    )
    (c,) = atomize_chunk(chunk, cfg, call_stage=fake)
    assert [q.unit_id for q in c.quotes] == ["U1"]
    assert "U9" not in c.unit_ids, "unknown unit ids are not kept as sources"
    assert c.review == []


def test_R4_quote_citing_a_unit_not_in_the_nodes_unit_ids_is_dropped(cfg, chunk):
    from kg.stages.atomize import atomize_chunk

    # §8.0: every quote.unit_id must be in the node's unit_ids AND in the chunk.
    fake = ScriptedCallStage({"atomize": [{"nodes": [cand("Event", ["U2"], [("U1", "all possible outcomes"), ("U2", "subset of the sample space")])]}]}, cfg=cfg)
    (c,) = atomize_chunk(chunk, cfg, call_stage=fake)
    assert [q.unit_id for q in c.quotes] == ["U2"]
    assert c.review == []


def test_R4_node_with_no_quotes_at_all_is_flagged(cfg, chunk):
    from kg.stages.atomize import atomize_chunk

    fake = ScriptedCallStage({"atomize": [{"nodes": [cand("Sample Space", ["U1"], [])]}]}, cfg=cfg)
    (c,) = atomize_chunk(chunk, cfg, call_stage=fake)
    assert c.review == ["quote_not_found"]


def test_R4_empty_quote_text_is_flagged_not_grounded(cfg, chunk):
    from kg.stages.atomize import atomize_chunk

    fake = ScriptedCallStage({"atomize": [{"nodes": [cand("Sample Space", ["U1"], [("U1", "   ")])]}]}, cfg=cfg)
    (c,) = atomize_chunk(chunk, cfg, call_stage=fake)
    assert c.review == ["quote_not_found"]


def test_R4_thai_quote_grounds_against_thai_unit_text(cfg):
    from kg.stages.atomize import atomize_chunk

    thai = "การเรียนรู้เชิงลึก คือสาขาหนึ่งของการเรียนรู้ของเครื่อง"
    chunk = make_chunk("t.md#c1", [make_unit("U1", thai, source="t.md", heading=["บทนำ"])])
    fake = ScriptedCallStage({"atomize": [{"nodes": [cand("การเรียนรู้เชิงลึก", ["U1"], [("U1", "คือสาขาหนึ่งของการเรียนรู้ของเครื่อง")])]}]}, cfg=cfg)
    (c,) = atomize_chunk(chunk, cfg, call_stage=fake)
    assert c.review == []


# --------------------------------------------------------- definitions clean


def test_R11_wikilinks_in_a_model_definition_are_stripped_on_read(cfg, chunk):
    from kg.stages.atomize import atomize_chunk

    fake = ScriptedCallStage(
        {"atomize": [{"nodes": [cand("Event", ["U2"], [("U2", "subset of the sample space")], definition="An [[event]] is a subset of the [[sample-space|sample space]].")]}]},
        cfg=cfg,
    )
    (c,) = atomize_chunk(chunk, cfg, call_stage=fake)
    assert c.definition == "An event is a subset of the sample space."


# ---------------------------------------------------------- failure passes up


def test_R21_a_boundary_exception_propagates_so_the_pipeline_can_defer_the_file(cfg, chunk):
    from kg.stages.atomize import atomize_chunk

    fake = ScriptedCallStage({"atomize": [RuntimeError("provider unavailable")]}, cfg=cfg)
    with pytest.raises(RuntimeError):
        atomize_chunk(chunk, cfg, call_stage=fake)
