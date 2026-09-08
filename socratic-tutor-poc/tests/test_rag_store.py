"""Store and ingest-pipeline tests (R3).

Everything runs offline on `FakeEmbedder` and a throwaway Chroma directory — no
model download, no network.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from socratic_tutor.domain.rag import store
from socratic_tutor.domain.rag.chunker import Chunk, chunk_segments
from socratic_tutor.domain.rag.embeddings import (
    FAKE_ENV_FLAG,
    FakeEmbedder,
    get_embedder,
    reset_embedder_cache,
)
from socratic_tutor.domain.rag.ingest import ingest_file, ingest_paths
from socratic_tutor.domain.rag.parsers import parse_path
from socratic_tutor.domain.rag.store import StoreIntegrityError

SEED_CORPUS = Path(__file__).resolve().parents[1] / "content" / "seed" / "rag-corpus"
SEED_FILES = sorted(SEED_CORPUS.glob("*.md"))
COURSE = "ALG101"


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def chroma_dir(tmp_path: Path) -> Iterator[Path]:
    store.reset_client_cache()
    yield tmp_path / "chroma"
    store.reset_client_cache()


@pytest.fixture
def seeded(chroma_dir: Path, embedder: FakeEmbedder) -> Path:
    """The whole seed corpus, ingested into a throwaway store."""
    for path in SEED_FILES:
        segments = parse_path(path)
        chunks = chunk_segments(segments, doc_id=path.stem)
        store.upsert_chunks(COURSE, chunks, embedder=embedder, persist_dir=chroma_dir)
    return chroma_dir


def test_round_trip_returns_provenance(seeded: Path, embedder: FakeEmbedder):
    hits = store.query(
        COURSE,
        "distributive property multiplies every term",
        top_k=3,
        embedder=embedder,
        persist_dir=seeded,
    )

    assert hits
    assert all(hit.source_ref.strip() for hit in hits)
    assert "Distributive Property Video" in hits[0].source_ref
    assert hits[0].score > 0


def test_every_hit_carries_a_source_ref_for_any_query(seeded: Path, embedder: FakeEmbedder):
    for question in ["inverse operation", "สมการสองขั้นตอน", "like terms", "checking a solution"]:
        hits = store.query(COURSE, question, top_k=5, embedder=embedder, persist_dir=seeded)
        assert hits, question
        assert all(hit.source_ref.strip() for hit in hits), question


def test_language_filter_returns_only_that_language(seeded: Path, embedder: FakeEmbedder):
    thai = store.query(
        COURSE, "การดำเนินการผกผัน", top_k=5, lang="th", embedder=embedder, persist_dir=seeded
    )
    english = store.query(
        COURSE, "inverse operation", top_k=5, lang="en", embedder=embedder, persist_dir=seeded
    )

    assert thai and english
    assert {hit.lang.value for hit in thai} == {"th"}
    assert {hit.lang.value for hit in english} == {"en"}


def test_topic_filter_narrows_the_candidate_set(seeded: Path, embedder: FakeEmbedder):
    hits = store.query(
        COURSE,
        "solve the equation",
        top_k=5,
        topic="Two-Step Linear Equations",
        embedder=embedder,
        persist_dir=seeded,
    )

    assert hits
    assert {hit.topic for hit in hits} == {"Two-Step Linear Equations"}


def test_query_on_an_unknown_course_returns_empty(chroma_dir: Path, embedder: FakeEmbedder):
    assert store.query("NOPE404", "anything", embedder=embedder, persist_dir=chroma_dir) == []


def test_upsert_rejects_a_chunk_without_provenance(chroma_dir: Path, embedder: FakeEmbedder):
    chunks = [
        Chunk(text="Balance both sides.", source_ref="Algebra Basics p.15", lang="en", ordinal=0),
        Chunk(text="Orphan text.", source_ref="", lang="en", chunk_id="orphan", ordinal=1),
    ]

    with pytest.raises(StoreIntegrityError, match="source_ref"):
        store.upsert_chunks(COURSE, chunks, embedder=embedder, persist_dir=chroma_dir)

    # The whole batch is rejected — no half-written document is left behind.
    assert store.count(COURSE, persist_dir=chroma_dir) == 0


def test_upsert_rejects_mismatched_embedding_count(chroma_dir: Path, embedder: FakeEmbedder):
    chunks = [Chunk(text="One.", source_ref="Ref p.1", lang="en", chunk_id="a", ordinal=0)]

    with pytest.raises(StoreIntegrityError, match="does not match"):
        store.upsert_chunks(COURSE, chunks, embeddings=[], persist_dir=chroma_dir)


def test_reingesting_the_same_document_does_not_duplicate(chroma_dir: Path, embedder: FakeEmbedder):
    chunks = chunk_segments(parse_path(SEED_FILES[0]), doc_id=SEED_FILES[0].stem)
    store.upsert_chunks(COURSE, chunks, embedder=embedder, persist_dir=chroma_dir)
    first = store.count(COURSE, persist_dir=chroma_dir)
    store.upsert_chunks(COURSE, chunks, embedder=embedder, persist_dir=chroma_dir)

    assert store.count(COURSE, persist_dir=chroma_dir) == first == len(chunks)


def test_collection_name_is_chroma_legal():
    assert store.collection_name("ALG 101/section#2") == "course-alg-101-section-2"
    assert store.collection_name("") == "course-default"


async def test_ingest_file_reports_every_stage(chroma_dir: Path, embedder: FakeEmbedder):
    events: list[tuple[str, float, str]] = []

    result = await ingest_file(
        SEED_FILES[0],
        COURSE,
        on_progress=lambda stage, percent, message: events.append((stage, percent, message)),
        embedder=embedder,
        persist_dir=chroma_dir,
    )

    assert result.ok
    assert result.chunks == 5
    assert result.errors == []
    assert [stage for stage, _, _ in events][:1] == ["parse"]
    assert {stage for stage, _, _ in events} == {"parse", "chunk", "embed", "store"}
    percents = [percent for _, percent, _ in events]
    assert percents == sorted(percents)
    assert percents[-1] == 100.0


async def test_ingest_batch_survives_a_bad_file(
    tmp_path: Path, chroma_dir: Path, embedder: FakeEmbedder
):
    broken = tmp_path / "diagram.png"
    broken.write_bytes(b"\x89PNG")

    results = await ingest_paths(
        [SEED_FILES[0], broken, SEED_FILES[1]],
        COURSE,
        embedder=embedder,
        persist_dir=chroma_dir,
    )

    assert [result.ok for result in results] == [True, False, True]
    assert "unsupported format" in results[1].errors[0]
    assert store.count(COURSE, persist_dir=chroma_dir) > 0


async def test_ingest_refuses_answer_keys(tmp_path: Path, chroma_dir: Path, embedder: FakeEmbedder):
    answer_key = tmp_path / "answer-keys.md"
    answer_key.write_text("q1: x = 7\nq2: x = 3\n", encoding="utf-8")

    result = await ingest_file(answer_key, COURSE, embedder=embedder, persist_dir=chroma_dir)

    assert not result.ok
    assert "answer key" in result.errors[0]
    assert store.count(COURSE, persist_dir=chroma_dir) == 0


async def test_ingest_refuses_content_declaring_canonical_answers(
    tmp_path: Path, chroma_dir: Path, embedder: FakeEmbedder
):
    disguised = tmp_path / "chapter-two-extras.md"
    disguised.write_text("canonical_answer: x = 4\ntopic: Two-Step\n", encoding="utf-8")

    result = await ingest_file(disguised, COURSE, embedder=embedder, persist_dir=chroma_dir)

    assert not result.ok
    assert store.count(COURSE, persist_dir=chroma_dir) == 0


async def test_a_renamed_solutions_pdf_is_still_refused(
    tmp_path: Path, chroma_dir: Path, embedder: FakeEmbedder
):
    """H6: the filename check is defeated by renaming, so the text must be checked.

    Binary formats used to skip the content check entirely, so this file — a
    solutions booklet called `notes.pdf` — walked into the corpus that feeds the
    generation prompt.
    """
    from test_rag_parsers import _minimal_pdf

    disguised = tmp_path / "notes.pdf"
    disguised.write_bytes(
        _minimal_pdf(
            [
                "Q1. Solve 2x + 4 = 10. Answer: x = 3",
                "Q2. Solve x + 5 = 12. Answer: x = 7",
                "Q3. Solve 4x = 20. Solution: x = 5",
            ],
            title="Chapter Notes",
        )
    )

    result = await ingest_file(disguised, COURSE, embedder=embedder, persist_dir=chroma_dir)

    assert not result.ok
    assert "answer key" in result.errors[0].lower() or "answers or solutions" in result.errors[0]
    assert store.count(COURSE, persist_dir=chroma_dir) == 0


async def test_thai_answer_labels_are_detected(
    tmp_path: Path, chroma_dir: Path, embedder: FakeEmbedder
):
    """Thai is the obvious authoring language for this product's answer keys."""
    thai_key = tmp_path / "บทที่-2.md"
    thai_key.write_text(
        "ข้อ 1 จงแก้สมการ 2x + 4 = 10\nเฉลย: x = 3\n\n"
        "ข้อ 2 จงแก้สมการ x + 5 = 12\nเฉลย: x = 7\n\n"
        "ข้อ 3 จงแก้สมการ 4x = 20\nคำตอบ: x = 5\n",
        encoding="utf-8",
    )

    result = await ingest_file(thai_key, COURSE, embedder=embedder, persist_dir=chroma_dir)

    assert not result.ok
    assert store.count(COURSE, persist_dir=chroma_dir) == 0


async def test_an_answer_key_refusal_is_logged_not_only_returned(
    tmp_path: Path,
    chroma_dir: Path,
    embedder: FakeEmbedder,
    caplog: pytest.LogCaptureFixture,
):
    """The uploader sees one refused file; the operator needs to see the pattern."""
    key = tmp_path / "worked-answers.md"
    key.write_text(
        "Q1 2x + 4 = 10\nAnswer: x = 3\n\nQ2 x + 5 = 12\nAnswer: x = 7\n\n"
        "Q3 4x = 20\nAnswer: x = 5\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="socratic_tutor.domain.rag.ingest"):
        result = await ingest_file(key, COURSE, embedder=embedder, persist_dir=chroma_dir)

    assert not result.ok
    assert any("answer-key refusal" in record.message for record in caplog.records)


async def test_teaching_material_that_merely_mentions_answers_is_still_ingested(
    chroma_dir: Path, embedder: FakeEmbedder
):
    """The check must not swallow real course content — the seed corpus explains
    how to check a solution, and says "the answer" in prose."""
    results = await ingest_paths(SEED_FILES, COURSE, embedder=embedder, persist_dir=chroma_dir)

    assert all(result.ok for result in results), [r.errors for r in results if not r.ok]
    assert store.count(COURSE, persist_dir=chroma_dir) == 14


async def test_a_failing_progress_callback_does_not_break_ingestion(
    chroma_dir: Path, embedder: FakeEmbedder
):
    def explode(stage: str, percent: float, message: str) -> None:
        raise RuntimeError("the SSE client went away")

    result = await ingest_file(
        SEED_FILES[0], COURSE, on_progress=explode, embedder=embedder, persist_dir=chroma_dir
    )

    assert result.ok


async def test_full_seed_corpus_ingests_and_is_retrievable(
    chroma_dir: Path, embedder: FakeEmbedder
):
    results = await ingest_paths(SEED_FILES, COURSE, embedder=embedder, persist_dir=chroma_dir)

    assert all(result.ok for result in results)
    total = sum(result.chunks for result in results)
    assert total == store.count(COURSE, persist_dir=chroma_dir) == 14

    hits = store.query(
        COURSE, "two-step equation", top_k=3, embedder=embedder, persist_dir=chroma_dir
    )
    assert hits and all(hit.source_ref for hit in hits)


def test_fake_embeddings_env_flag_selects_the_offline_embedder(monkeypatch: pytest.MonkeyPatch):
    reset_embedder_cache()
    monkeypatch.setenv(FAKE_ENV_FLAG, "1")
    try:
        assert isinstance(get_embedder(), FakeEmbedder)
    finally:
        reset_embedder_cache()


def test_fake_embeddings_are_deterministic_and_unit_length(embedder: FakeEmbedder):
    first = embedder.embed_query("การดำเนินการผกผัน")
    second = embedder.embed_query("การดำเนินการผกผัน")

    assert first == second
    assert len(first) == embedder.dimension
    assert abs(sum(value * value for value in first) - 1.0) < 1e-9
