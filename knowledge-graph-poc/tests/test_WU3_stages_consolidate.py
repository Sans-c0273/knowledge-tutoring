"""WU3 — `kg.stages.consolidate` (S4): merge candidates across chunks by normalised title/alias;
registry match reuses the permanent id; new titles get ids in deterministic order (DESIGN §6, §8.5, §9; R7, R16).

Interface (chosen): consolidate(candidates, registry, *, run) -> list[NodeDraft];
NodeDraft(id, title, aliases, definition, sources, review, chunk_ids, source_files, is_new).
Candidates are the objects `kg.stages.atomize` returns; this file builds them directly.
"""

from __future__ import annotations

import pytest

from wu3_fixtures import RUN_ID


def C(title, *, definition="Def.", aliases=(), source="a.md", chunk_id="a.md#c1", quotes=(("a.md#heading=X", "quote"),), review=()):
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
        review=list(review),
        sources=[NodeSource(locator=loc, file=source, quote=q) for loc, q in quotes],
    )


@pytest.fixture
def reg():
    from kg.registry import Registry

    return Registry(corpus="kc", schema="general")


def test_S4_distinct_titles_become_distinct_drafts_with_monotonic_ids_in_input_order(reg):
    from kg.stages.consolidate import consolidate

    drafts = consolidate([C("Sample Space"), C("Event"), C("Probability Measure", source="b.md", chunk_id="b.md#c1")], reg, run=RUN_ID)
    assert [d.id for d in drafts] == ["kc-0001", "kc-0002", "kc-0003"]
    assert [d.title for d in drafts] == ["Sample Space", "Event", "Probability Measure"]
    assert all(d.is_new for d in drafts)
    assert reg.next_seq == 4
    assert reg.find("Event") == "kc-0002"


def test_S4_exact_normalised_title_collapse_across_chunks(reg):
    from kg.stages.consolidate import consolidate

    a = C("Sample Space", definition="First definition.", quotes=(("a.md#heading=A", "q1"),))
    b = C("the sample spaces", definition="Second definition.", source="b.md", chunk_id="b.md#c1", quotes=(("b.md#heading=B", "q2"),))
    (d,) = consolidate([a, b], reg, run=RUN_ID)
    assert d.id == "kc-0001"
    assert d.title == "Sample Space", "first occurrence wins the surface title"
    assert d.definition == "First definition."
    assert [(s.locator, s.quote) for s in d.sources] == [("a.md#heading=A", "q1"), ("b.md#heading=B", "q2")]
    assert sorted(d.chunk_ids) == ["a.md#c1", "b.md#c1"]
    assert sorted(d.source_files) == ["a.md", "b.md"]
    assert d.aliases == [], "a surface form that normalises to the title is not an alias"


def test_S4_title_matching_another_candidates_alias_merges(reg):
    from kg.stages.consolidate import consolidate

    a = C("Sample Space", aliases=["Outcome space"])
    b = C("Outcome Space", source="b.md", chunk_id="b.md#c1")
    drafts = consolidate([a, b], reg, run=RUN_ID)
    assert len(drafts) == 1
    assert drafts[0].title == "Sample Space"
    assert "Outcome space" in drafts[0].aliases


def test_S4_alias_union_is_deduplicated_and_never_contains_the_title(reg):
    from kg.stages.consolidate import consolidate

    a = C("Sample Space", aliases=["Outcome space", "S"])
    b = C("Sample Space", aliases=["S", "Sample space", "Omega"], source="b.md")
    (d,) = consolidate([a, b], reg, run=RUN_ID)
    assert d.aliases == ["Outcome space", "S", "Omega"]


def test_S4_identical_sources_are_not_duplicated(reg):
    from kg.stages.consolidate import consolidate

    a = C("Sample Space", quotes=(("a.md#heading=A", "q1"),))
    b = C("Sample Space", quotes=(("a.md#heading=A", "q1"), ("a.md#heading=A", "q2")))
    (d,) = consolidate([a, b], reg, run=RUN_ID)
    assert [(s.locator, s.quote) for s in d.sources] == [("a.md#heading=A", "q1"), ("a.md#heading=A", "q2")]


def test_S4_registry_match_reuses_the_permanent_id_and_allocates_nothing(reg):
    from kg.stages.consolidate import consolidate

    reg.allocate("Sample Space", run="2026-08-01T00-00-00Z", aliases=("Outcome space",))
    before = reg.next_seq
    drafts = consolidate([C("sample spaces"), C("outcome space")], reg, run=RUN_ID)
    assert [d.id for d in drafts] == ["kc-0001"]
    assert drafts[0].is_new is False
    assert reg.next_seq == before


def test_S4_mix_of_existing_and_new_keeps_input_order_and_allocates_only_the_new(reg):
    from kg.stages.consolidate import consolidate

    reg.allocate("Event", run="2026-08-01T00-00-00Z")  # kc-0001
    drafts = consolidate([C("Sample Space"), C("Event"), C("Random Variable")], reg, run=RUN_ID)
    assert [(d.title, d.id, d.is_new) for d in drafts] == [("Sample Space", "kc-0002", True), ("Event", "kc-0001", False), ("Random Variable", "kc-0003", True)]


def test_S4_review_flags_are_unioned(reg):
    from kg.stages.consolidate import consolidate

    a = C("Sample Space", review=["quote_not_found"])
    b = C("Sample Space", source="b.md")
    (d,) = consolidate([a, b], reg, run=RUN_ID)
    assert d.review == ["quote_not_found"]


def test_S4_is_deterministic_for_the_same_input():
    from kg.registry import Registry
    from kg.stages.consolidate import consolidate

    cands = [C("B Concept"), C("A Concept"), C("b concept", source="b.md"), C("C Concept")]
    r1 = consolidate(list(cands), Registry(corpus="kc", schema="general"), run=RUN_ID)
    r2 = consolidate(list(cands), Registry(corpus="kc", schema="general"), run=RUN_ID)
    assert [(d.id, d.title) for d in r1] == [(d.id, d.title) for d in r2] == [("kc-0001", "B Concept"), ("kc-0002", "A Concept"), ("kc-0003", "C Concept")]


def test_S4_thai_titles_collapse_on_whitespace_and_nfkc_only(reg):
    from kg.stages.consolidate import consolidate

    drafts = consolidate([C("การเรียนรู้เชิงลึก"), C(" การเรียนรู้เชิงลึก "), C("การเรียนรู้ของเครื่อง")], reg, run=RUN_ID)
    assert [d.title for d in drafts] == ["การเรียนรู้เชิงลึก", "การเรียนรู้ของเครื่อง"]


def test_S4_empty_input_yields_no_drafts_and_touches_the_registry_not_at_all(reg):
    from kg.stages.consolidate import consolidate

    assert consolidate([], reg, run=RUN_ID) == []
    assert reg.next_seq == 1 and list(reg.rows) == []


def test_S4_candidate_with_blank_title_is_skipped_not_fatal(reg):
    from kg.stages.consolidate import consolidate

    drafts = consolidate([C("   "), C("Event")], reg, run=RUN_ID)
    assert [d.title for d in drafts] == ["Event"]
