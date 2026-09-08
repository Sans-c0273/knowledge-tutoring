"""Chunker tests (R3): script purity, provenance, and boundary handling."""

from __future__ import annotations

import re
from pathlib import Path

from socratic_tutor.domain.rag.chunker import (
    MAX_TOKENS,
    Chunk,
    chunk_segments,
    estimate_tokens,
)
from socratic_tutor.domain.rag.parsers import ParsedSegment, parse_path

SEED_CORPUS = Path(__file__).resolve().parents[1] / "content" / "seed" / "rag-corpus"

_THAI = re.compile(r"[฀-๿]")
#: Same threshold the chunker uses: 3+ Latin letters is prose, not algebra.
_ENGLISH_WORD = re.compile(r"[A-Za-z]{3,}")


def seed_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(SEED_CORPUS.glob("*.md")):
        chunks.extend(chunk_segments(parse_path(path)))
    return chunks


def is_script_mixed(text: str) -> bool:
    """Thai script alongside English prose — the condition the chunker forbids."""
    return bool(_THAI.search(text)) and bool(_ENGLISH_WORD.search(text))


def test_seed_corpus_chunks_never_mix_thai_and_english():
    chunks = seed_chunks()

    assert chunks
    for chunk in chunks:
        assert not is_script_mixed(chunk.text), chunk.text[:80]
        assert chunk.lang == ("th" if _THAI.search(chunk.text) else "en")


def test_english_chunks_contain_no_thai_and_thai_chunks_no_english_prose():
    for chunk in seed_chunks():
        if chunk.lang == "en":
            assert not _THAI.search(chunk.text), chunk.text[:80]
        else:
            # Thai chunks legitimately embed algebra ("2(x + 3)"), but never
            # English prose words.
            assert not _ENGLISH_WORD.search(chunk.text), chunk.text[:80]


def test_every_chunk_carries_source_ref_and_topic():
    for chunk in seed_chunks():
        assert chunk.source_ref.strip()
        assert chunk.topic


def test_authored_chunk_boundaries_are_preserved():
    chunks = chunk_segments(parse_path(SEED_CORPUS / "ch1-inverse-operations.md"))

    assert [chunk.chunk_id for chunk in chunks] == ["c1-01", "c1-02", "c1-03", "c1-04", "c1-05"]
    assert chunks[2].lang == "th"
    assert chunks[2].source_ref == "Algebra Basics Ch.1 p.13"
    assert chunks[4].topic == "Equation Balance"


def test_mixed_language_paragraph_splits_at_the_script_boundary():
    segment = ParsedSegment(
        text=(
            "An inverse operation undoes another operation. "
            "การดำเนินการผกผันคือการดำเนินการที่ย้อนกลับผลของอีกการดำเนินการหนึ่ง "
            "Multiplication and division are inverses too."
        ),
        source_ref="Algebra Basics Ch.1 p.12",
        page_or_slide_or_timestamp="12",
        lang_hint="en",
        topic="Inverse Operations",
    )

    chunks = chunk_segments([segment], doc_id="mixed")

    assert [chunk.lang for chunk in chunks] == ["en", "th", "en"]
    assert all(chunk.source_ref == "Algebra Basics Ch.1 p.12" for chunk in chunks)
    assert all(chunk.topic == "Inverse Operations" for chunk in chunks)
    assert not _THAI.search(chunks[0].text)
    assert not _ENGLISH_WORD.search(chunks[1].text)


def test_long_english_segment_splits_with_overlap_and_respects_the_ceiling():
    body = " ".join(
        f"Step {index}: subtract the constant from both sides and then divide by the coefficient."
        for index in range(1, 60)
    )
    segment = ParsedSegment(
        text=body, source_ref="Algebra Basics Ch.2 p.23", lang_hint="en", topic="Two-Step"
    )

    chunks = chunk_segments([segment], doc_id="long-en")

    assert len(chunks) > 1
    assert all(estimate_tokens(chunk.text) <= MAX_TOKENS for chunk in chunks)
    assert all(chunk.lang == "en" for chunk in chunks)
    # Consecutive chunks share text: the tail of one opens the next.
    assert chunks[1].text.split(".")[0] in chunks[0].text


def test_unspaced_thai_segment_is_split_by_codepoint_budget():
    body = "การแก้สมการเชิงเส้นต้องใช้การดำเนินการผกผันกับทั้งสองข้างของสมการเสมอ" * 40
    segment = ParsedSegment(
        text=body, source_ref="Algebra Basics Ch.2 p.21", lang_hint="th", topic="Linear Equations"
    )

    chunks = chunk_segments([segment], doc_id="long-th")

    assert len(chunks) > 1
    assert all(chunk.lang == "th" for chunk in chunks)
    assert all(estimate_tokens(chunk.text) <= MAX_TOKENS for chunk in chunks)


def test_generated_chunk_ids_are_unique_per_document():
    segments = [
        ParsedSegment(text="First page text.", source_ref="Deck slide 1", lang_hint="en"),
        ParsedSegment(text="Second page text.", source_ref="Deck slide 2", lang_hint="en"),
    ]

    chunks = chunk_segments(segments, doc_id="deck")

    ids = [chunk.chunk_id for chunk in chunks]
    assert ids == ["deck-000-00", "deck-001-00"]
    assert len(set(ids)) == len(ids)


def test_empty_input_produces_no_chunks():
    assert chunk_segments([]) == []
