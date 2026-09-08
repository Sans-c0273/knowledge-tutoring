"""WU1 — `kg.chunk` (S2; DESIGN §5, D8): greedy grouping of whole Units.

Spec (DESIGN §5, verbatim): "Greedy grouping of whole units to `target_tokens`;
a unit is split only if it alone exceeds `max_tokens`, sub-units inherit the
locator; trailing chunk < `min_tokens` merges backwards; chunks never cross
source files; each unit is prefixed in the prompt with
`<<U3 | file.pptx#slide=7 | Heading > Path>>` and the model returns `unit_ids`.
Locators live on units, never chunks."

Interfaces under test (module named in DESIGN §20; identifiers chosen):
  kg.chunk.estimate_tokens(text: str) -> int              deterministic, offline
  kg.chunk.build_chunks(units: list[Unit], cfg: ChunkingConfig) -> list[Chunk]
  kg.chunk.Chunk: .cid: str, .source: str, .units: list[Unit], .unit_ids: list[str], .locators: list[str]
  kg.chunk.render_chunk(chunk: Chunk) -> str              prompt text with the <<uid | locator | path>> prefixes

All sizes are expressed through `estimate_tokens` itself so the tests do not
depend on the estimator's constant.
"""

from __future__ import annotations

import re

import pytest

from kg.config import ChunkingConfig


def words(n: int, tag: str = "w") -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def make_unit(i: int, text: str, source: str = "a.md", heading: list[str] | None = None):
    from kg.convert import Unit

    heading = heading if heading is not None else ["Intro", f"Sec{i}"]
    return Unit(
        uid=f"U{i}",
        locator=f"{source}#heading={' > '.join(heading)}",
        heading_path=heading,
        text=text,
        kind="text",
        source=source,
    )


def tokens_of(chunk) -> int:
    from kg.chunk import estimate_tokens

    return sum(estimate_tokens(u.text) for u in chunk.units)


def norm(s: str) -> str:
    return " ".join(s.split())


# ------------------------------------------------------------ estimate_tokens


def test_S2_estimate_tokens_is_zero_for_empty_and_monotonic_in_length():
    from kg.chunk import estimate_tokens

    assert estimate_tokens("") == 0
    a, b, c = estimate_tokens(words(10)), estimate_tokens(words(100)), estimate_tokens(words(1000))
    assert 0 < a < b < c


def test_S2_estimate_tokens_counts_thai_text():
    from kg.chunk import estimate_tokens

    assert estimate_tokens("การเรียนรู้เชิงลึกคือสาขาหนึ่งของการเรียนรู้ของเครื่อง") > 0


def test_S2_estimate_tokens_is_deterministic():
    from kg.chunk import estimate_tokens

    t = words(500)
    assert estimate_tokens(t) == estimate_tokens(t)


# --------------------------------------------------------------- build_chunks


@pytest.fixture
def unit_tokens() -> int:
    from kg.chunk import estimate_tokens

    return estimate_tokens(words(60))


@pytest.fixture
def cfg3(unit_tokens: int) -> ChunkingConfig:
    """Fits three 60-word units per chunk, never four; nothing is oversize; no back-merge."""
    return ChunkingConfig(target_tokens=unit_tokens * 3 + unit_tokens // 2, max_tokens=unit_tokens * 6, min_tokens=0)


def test_S2_every_unit_appears_exactly_once_in_document_order(cfg3: ChunkingConfig):
    from kg.chunk import build_chunks

    units = [make_unit(i, words(60, f"u{i}_")) for i in range(1, 11)]
    chunks = build_chunks(units, cfg3)
    flat = [uid for c in chunks for uid in c.unit_ids]
    assert flat == [u.uid for u in units]
    for c in chunks:
        assert c.unit_ids == [u.uid for u in c.units]


def test_S2_multi_unit_chunks_stay_within_target_tokens(cfg3: ChunkingConfig):
    from kg.chunk import build_chunks

    units = [make_unit(i, words(60, f"u{i}_")) for i in range(1, 11)]
    for c in build_chunks(units, cfg3):
        if len(c.units) > 1:
            assert tokens_of(c) <= cfg3.target_tokens, f"{c.cid}: {tokens_of(c)} > target {cfg3.target_tokens}"


def test_S2_grouping_is_greedy_so_chunks_are_maximal(cfg3: ChunkingConfig):
    from kg.chunk import build_chunks

    units = [make_unit(i, words(60, f"u{i}_")) for i in range(1, 11)]
    chunks = build_chunks(units, cfg3)
    assert [len(c.units) for c in chunks] == [3, 3, 3, 1]
    from kg.chunk import estimate_tokens

    for cur, nxt in zip(chunks, chunks[1:]):
        # adding the next unit would have overshot the target (allow a small separator margin)
        assert tokens_of(cur) + estimate_tokens(nxt.units[0].text) > cfg3.target_tokens * 0.9


def test_S2_locators_live_on_units_and_survive_into_every_chunk(cfg3: ChunkingConfig):
    from kg.chunk import build_chunks

    units = [make_unit(i, words(60, f"u{i}_")) for i in range(1, 8)]
    chunks = build_chunks(units, cfg3)
    for c in chunks:
        assert c.locators == [u.locator for u in c.units]
        assert all(loc.startswith("a.md#heading=") for loc in c.locators)
    assert {loc for c in chunks for loc in c.locators} == {u.locator for u in units}
    assert not hasattr(chunks[0], "locator"), "D8: a Chunk carries no locator of its own"


def test_S2_chunks_never_cross_source_files(cfg3: ChunkingConfig):
    from kg.chunk import build_chunks

    # tiny units: everything from one file would fit in one chunk, but two files must stay apart
    units = [make_unit(i, words(5, f"a{i}_"), source="a.md") for i in range(1, 4)]
    units += [make_unit(i, words(5, f"b{i}_"), source="deck.pptx", heading=[f"Slide {i}"]) for i in range(4, 7)]
    units += [make_unit(i, words(5, f"c{i}_"), source="a.md") for i in range(7, 9)]
    chunks = build_chunks(units, cfg3)
    for c in chunks:
        sources = {u.source for u in c.units}
        assert len(sources) == 1, f"{c.cid} mixes sources {sources}"
        assert c.source in sources
    assert len(chunks) >= 3, "a.md, deck.pptx, a.md again → at least three chunks in order"
    assert [c.source for c in chunks][:3] == ["a.md", "deck.pptx", "a.md"]


def test_S2_unit_over_max_tokens_is_split_and_sub_units_inherit_the_locator(unit_tokens: int):
    from kg.chunk import build_chunks, estimate_tokens

    cfg = ChunkingConfig(target_tokens=unit_tokens * 3, max_tokens=unit_tokens * 4, min_tokens=0)
    big_text = "\n\n".join(words(60, f"p{p}_") for p in range(12))  # ≈ 12 × unit_tokens » max
    big = make_unit(1, big_text, heading=["Big"])
    after = make_unit(2, words(20, "z"), heading=["After"])
    chunks = build_chunks([big, after], cfg)

    subs = [u for c in chunks for u in c.units if u.locator == big.locator]
    assert len(subs) >= 3, "an oversize unit must be split into several sub-units"
    for s in subs:
        assert estimate_tokens(s.text) <= cfg.max_tokens
        assert s.locator == "a.md#heading=Big", "sub-units inherit the parent's locator (§5, D8)"
        assert s.heading_path == ["Big"]
        assert s.source == "a.md"
    sub_ids = [s.uid for s in subs]
    assert len(set(sub_ids)) == len(sub_ids), "sub-units need distinct ids so the model can cite each"
    assert norm(" ".join(s.text for s in subs)) == norm(big_text), "no text lost or duplicated by the split"
    assert any(u.uid == "U2" for c in chunks for u in c.units), "the following unit still present, unsplit"


def test_S2_unit_between_target_and_max_is_kept_whole(unit_tokens: int):
    from kg.chunk import build_chunks

    cfg = ChunkingConfig(target_tokens=unit_tokens, max_tokens=unit_tokens * 3, min_tokens=0)
    medium = make_unit(1, words(120, "m"))  # ≈ 2 × unit_tokens: over target, under max
    chunks = build_chunks([medium], cfg)
    assert len(chunks) == 1
    assert chunks[0].unit_ids == ["U1"]
    assert chunks[0].units[0].text == medium.text


def test_S2_trailing_chunk_below_min_tokens_merges_backwards(unit_tokens: int):
    from kg.chunk import build_chunks, estimate_tokens

    tiny_text = words(4, "t")
    tiny_tokens = estimate_tokens(tiny_text)
    cfg = ChunkingConfig(
        target_tokens=unit_tokens + tiny_tokens // 2,  # A fills the chunk; A+tiny overshoots
        max_tokens=unit_tokens * 4,
        min_tokens=tiny_tokens + 1,  # a chunk of just `tiny` is below the floor
    )
    a = make_unit(1, words(60, "a"))
    tiny = make_unit(2, tiny_text)
    chunks = build_chunks([a, tiny], cfg)
    assert len(chunks) == 1, "the under-sized trailing chunk merges into its predecessor"
    assert chunks[0].unit_ids == ["U1", "U2"]


def test_S2_trailing_chunk_does_not_merge_across_source_files(unit_tokens: int):
    from kg.chunk import build_chunks, estimate_tokens

    tiny_text = words(4, "t")
    tiny_tokens = estimate_tokens(tiny_text)
    cfg = ChunkingConfig(target_tokens=unit_tokens + tiny_tokens // 2, max_tokens=unit_tokens * 4, min_tokens=tiny_tokens + 1)
    a = make_unit(1, words(60, "a"), source="a.md")
    tiny = make_unit(2, tiny_text, source="b.md")
    chunks = build_chunks([a, tiny], cfg)
    assert len(chunks) == 2
    assert [c.source for c in chunks] == ["a.md", "b.md"]


def test_S2_output_is_deterministic(cfg3: ChunkingConfig):
    from kg.chunk import build_chunks

    units = [make_unit(i, words(60, f"u{i}_")) for i in range(1, 11)]
    first = build_chunks(list(units), cfg3)
    second = build_chunks(list(units), cfg3)
    assert [c.cid for c in first] == [c.cid for c in second]
    assert [c.unit_ids for c in first] == [c.unit_ids for c in second]
    assert [[u.text for u in c.units] for c in first] == [[u.text for u in c.units] for c in second]


def test_S2_chunk_ids_are_unique_non_empty_strings(cfg3: ChunkingConfig):
    from kg.chunk import build_chunks

    units = [make_unit(i, words(60, f"u{i}_")) for i in range(1, 11)]
    cids = [c.cid for c in build_chunks(units, cfg3)]
    assert all(isinstance(c, str) and c for c in cids)
    assert len(set(cids)) == len(cids)


def test_S2_empty_input_gives_no_chunks(cfg3: ChunkingConfig):
    from kg.chunk import build_chunks

    assert build_chunks([], cfg3) == []


def test_S2_empty_text_units_do_not_produce_empty_chunks(cfg3: ChunkingConfig):
    # e.g. `empty_page` PDF units (DESIGN §4.2) carry no text; they must not become a chunk of nothing.
    from kg.chunk import build_chunks

    units = [make_unit(1, "", heading=["p1"]), make_unit(2, words(30, "x"), heading=["p2"]), make_unit(3, "   ", heading=["p3"])]
    chunks = build_chunks(units, cfg3)
    assert chunks, "the non-empty unit must still be chunked"
    for c in chunks:
        assert any(u.text.strip() for u in c.units), f"{c.cid} contains only empty units"


# ---------------------------------------------------------------- render_chunk


def test_S2_render_chunk_prefixes_each_unit_with_uid_locator_and_heading_path(cfg3: ChunkingConfig):
    from kg.chunk import build_chunks, render_chunk

    units = [
        make_unit(3, "Slide seven body.", source="file.pptx", heading=["Heading", "Path"]),
        make_unit(4, "Slide eight body.", source="file.pptx", heading=["Heading", "Other"]),
    ]
    units[0].locator = "file.pptx#slide=7"
    units[1].locator = "file.pptx#slide=8"
    chunk = build_chunks(units, cfg3)[0]
    text = render_chunk(chunk)
    assert "<<U3 | file.pptx#slide=7 | Heading > Path>>" in text
    assert "<<U4 | file.pptx#slide=8 | Heading > Other>>" in text
    assert text.index("<<U3 |") < text.index("Slide seven body.") < text.index("<<U4 |") < text.index("Slide eight body.")
    markers = re.findall(r"<<(U\d+) \| ", text)
    assert markers == ["U3", "U4"], "exactly one marker per unit, in order"


def test_S2_render_chunk_with_empty_heading_path_still_has_three_fields(cfg3: ChunkingConfig):
    from kg.chunk import build_chunks, render_chunk

    u = make_unit(1, "Page text.", source="deck.pdf", heading=[])
    u.locator = "deck.pdf#page=4"
    text = render_chunk(build_chunks([u], cfg3)[0])
    assert re.search(r"<<U1 \| deck\.pdf#page=4 \| ?>>", text), text
