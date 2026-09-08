"""Ingestion pipeline: parse → chunk → embed → store (Tech Spec §7 `POST /content/rag`).

Progress is reported through a callback with a named stage and a percentage, so
the SSE layer can stream "parse 25% · chunk 45% · embed 80% · store 100%" per
file to the ingestion UI (R17: progress visibly advances per stage; failures
surface with actionable messages).

A bad file never breaks a batch: `ingest_file` and `ingest_url` return an
`IngestResult` carrying the error instead of raising.

**Answer keys are refused, not indexed** (Tech Spec §5.2/§5.3). Canonical
solutions belong to Call B only and must never reach Call C's retrieval context,
so a source that identifies itself as an answer key is rejected at the door
rather than relying on the output guardrail to catch a leak later. The name is
checked first and the *extracted text* second, because a filename is controlled
by whoever uploads the file: `worked-solutions.pdf` renamed `notes.pdf` has to
fail on its contents.

**Known limit of that refusal.** A teacher whose legitimate material trips the
density test is blocked outright, with no override. That is the right trade for
a POC — failing closed, with a message saying what to do — but it does not
survive cohort scale, where the answer is an *override carrying an audit
record*, not a looser threshold. Loosening the threshold to buy back false
positives would silently reopen the leak this exists to prevent; whoever builds
the human-review queue the security review asked for should inherit that
reasoning rather than retune the constants. Every refusal is logged at WARNING,
so repeated hits are visible to whoever runs the pilot as a signal about the
corpus.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from socratic_tutor.domain.rag import parsers, store
from socratic_tutor.domain.rag.chunker import Chunk, chunk_segments
from socratic_tutor.domain.rag.embeddings import Embedder, get_embedder
from socratic_tutor.domain.rag.parsers import ParsedSegment, ParseError

logger = logging.getLogger(__name__)

#: `(stage, percent, message)` — stage is one of `STAGES`, percent is 0–100.
ProgressCallback = Callable[[str, float, str], None]

STAGES = ("parse", "chunk", "embed", "store")
#: Percentage reached when each stage completes.
_STAGE_DONE = {"parse": 25.0, "chunk": 45.0, "embed": 80.0, "store": 100.0}

#: A source *named* like any of these is an answer key, not teaching content.
#: Names are the weakest signal — renaming defeats them — so this is only the
#: cheap first pass; `_extracted_text_looks_like_answer_key` is the real check.
ANSWER_KEY_PATTERNS = (
    re.compile(r"answer[-_\s]?keys?", re.IGNORECASE),
    re.compile(r"canonical[-_\s]?(answer|solution)", re.IGNORECASE),
    re.compile(r"marking[-_\s]?scheme", re.IGNORECASE),
)

#: A labelled answer, bilingual: "Answer:", "Solution:", "เฉลย:", "คำตอบ:".
#: The colon is what distinguishes them — teaching prose says "the answer is 7";
#: an answer key *labels* it. Deliberately not anchored to line starts: PDF text
#: extraction returns a page as one long line, which is precisely the case this
#: has to catch.
ANSWER_LABEL = re.compile(
    r"(?:\b(?:answers?|solutions?)\s*[:：]|(?:เฉลย|คำตอบ)\s*[:：])",
    re.IGNORECASE,
)
#: An answer key labels answers *densely*. Both thresholds must hold: at least
#: this many labels, and roughly one per this many characters — so a chapter
#: with one worked "Solution:" passes and a solutions booklet does not.
ANSWER_KEY_MIN_LABELS = 3
ANSWER_KEY_CHARS_PER_LABEL = 800


@dataclass
class IngestResult:
    """Outcome of ingesting one source."""

    source: str
    course_id: str
    ok: bool = False
    segments: int = 0
    chunks: int = 0
    errors: list[str] = field(default_factory=list)


def _emit(on_progress: ProgressCallback | None, stage: str, percent: float, message: str) -> None:
    """Call the progress callback, never letting a UI-side failure abort ingestion."""
    if on_progress is None:
        return
    try:
        on_progress(stage, percent, message)
    except Exception:
        logger.exception("ingestion progress callback failed at stage %s", stage)


def _looks_like_answer_key(source: str) -> bool:
    return any(pattern.search(source) for pattern in ANSWER_KEY_PATTERNS)


def _extracted_text_looks_like_answer_key(segments: list[ParsedSegment]) -> str | None:
    """Answer-key detection on *extracted text*, so it works for every format.

    The filename check alone was a paper lock: `_content_declares_answer_key`
    only reads text files, so `worked-solutions.pdf` renamed `notes.pdf` walked
    past both gates and into the corpus that feeds Call C. Parsing already gives
    us the text of PDF, DOCX and PPTX, so the same structural check runs on all
    of them here.

    Returns a human-readable reason, or `None` when the source looks like
    teaching material.
    """
    if not segments:
        return None

    text = "\n".join(segment.text for segment in segments)
    labels = len(ANSWER_LABEL.findall(text))
    if labels >= ANSWER_KEY_MIN_LABELS and labels >= len(text) / ANSWER_KEY_CHARS_PER_LABEL:
        return (
            f"it labels {labels} answers or solutions in {len(text)} characters, "
            "which is the shape of an answer key rather than teaching material"
        )

    named = [
        segment.source_ref
        for segment in segments
        if segment.source_ref and _looks_like_answer_key(segment.source_ref)
    ]
    if named:
        return f"its own content identifies it as an answer key ({named[0]})"
    return None


def _content_declares_answer_key(path: Path) -> bool:
    """Detect answer-key structure in a text source (`answer_key:`, `canonical_answer:`)."""
    if path.suffix.lower() not in parsers.TEXT_SUFFIXES:
        return False
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:4000]
    except OSError:
        return False
    return (
        re.search(r"^\s*(answer_key|canonical_answer|answer_keys)\s*:", head, re.MULTILINE)
        is not None
    )


def _doc_id(source: str) -> str:
    """Stable, readable chunk-id prefix derived from a filename or URL."""
    stem = Path(source).stem or source
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", stem.lower()).strip("-")
    return slug[:48] or "doc"


async def _run_pipeline(
    source: str,
    course_id: str,
    parse: Callable[[], list[ParsedSegment]],
    on_progress: ProgressCallback | None,
    embedder: Embedder | None,
    persist_dir: Path | None,
    doc_id: str | None,
) -> IngestResult:
    """Shared parse → chunk → embed → store body for files and URLs."""
    result = IngestResult(source=source, course_id=course_id)

    _emit(on_progress, "parse", 0.0, f"Reading {source}")
    try:
        segments = await asyncio.to_thread(parse)
    except ParseError as exc:
        result.errors.append(str(exc))
        _emit(on_progress, "parse", 0.0, f"Failed: {exc}")
        return result
    except Exception as exc:
        logger.exception("unexpected parse failure for %s", source)
        result.errors.append(f"parse failed for {source}: {exc}")
        _emit(on_progress, "parse", 0.0, f"Failed: {exc}")
        return result

    result.segments = len(segments)
    _emit(on_progress, "parse", _STAGE_DONE["parse"], f"{len(segments)} segment(s)")

    # Answer-key check on the extracted text, before anything is chunked or
    # indexed. This is the gate the filename check cannot be: a renamed
    # solutions PDF reaches here with its content readable (Tech Spec §5.3).
    reason = _extracted_text_looks_like_answer_key(segments)
    if reason is not None:
        message = (
            f"refused {source}: {reason}. Answer keys stay server-side for answer "
            "evaluation and must never enter the retrieval corpus (Tech Spec §5.3). "
            "If this really is teaching material, split the worked answers out and "
            "upload the explanatory text on its own."
        )
        # Logged, not just returned: repeated refusals are a signal about the
        # corpus (or about a mis-tuned threshold) that the pilot operator needs
        # to see, and the uploader only ever sees their own one file.
        logger.warning("answer-key refusal for course %s: %s", course_id, message)
        result.errors.append(message)
        _emit(on_progress, "parse", _STAGE_DONE["parse"], message)
        return result

    if not segments:
        result.errors.append(f"no readable text found in {source}")
        return result

    _emit(on_progress, "chunk", _STAGE_DONE["parse"], "Splitting into chunks")
    chunks: list[Chunk] = chunk_segments(segments, doc_id=doc_id or _doc_id(source))
    result.chunks = len(chunks)
    _emit(on_progress, "chunk", _STAGE_DONE["chunk"], f"{len(chunks)} chunk(s)")

    if not chunks:
        result.errors.append(f"no chunks produced from {source}")
        return result

    _emit(on_progress, "embed", _STAGE_DONE["chunk"], f"Embedding {len(chunks)} chunk(s)")
    active = embedder or get_embedder()
    try:
        vectors = await asyncio.to_thread(active.embed_texts, [chunk.text for chunk in chunks])
    except Exception as exc:
        logger.exception("embedding failed for %s", source)
        result.errors.append(f"embedding failed for {source}: {exc}")
        _emit(on_progress, "embed", _STAGE_DONE["chunk"], f"Failed: {exc}")
        return result
    _emit(on_progress, "embed", _STAGE_DONE["embed"], "Embedded")

    _emit(on_progress, "store", _STAGE_DONE["embed"], "Indexing")
    try:
        await asyncio.to_thread(
            store.upsert_chunks,
            course_id,
            chunks,
            vectors,
            active,
            persist_dir,
        )
    except Exception as exc:
        logger.exception("indexing failed for %s", source)
        result.errors.append(f"indexing failed for {source}: {exc}")
        _emit(on_progress, "store", _STAGE_DONE["embed"], f"Failed: {exc}")
        return result

    result.ok = True
    _emit(on_progress, "store", _STAGE_DONE["store"], f"Indexed {len(chunks)} chunk(s)")
    return result


async def ingest_file(
    path: str | Path,
    course_id: str,
    on_progress: ProgressCallback | None = None,
    doc_id: str | None = None,
    embedder: Embedder | None = None,
    persist_dir: Path | None = None,
) -> IngestResult:
    """Ingest one file (PDF/DOCX/PPTX/Markdown/text/VTT/SRT transcript).

    Never raises: parse, embed and index failures come back in
    `IngestResult.errors` with `ok=False`.
    """
    path = Path(path)
    source = path.name
    result = IngestResult(source=source, course_id=course_id)

    if _looks_like_answer_key(source) or _content_declares_answer_key(path):
        message = (
            f"refused {source}: answer keys stay server-side for answer evaluation "
            "and must never enter the retrieval corpus (Tech Spec §5.3)"
        )
        logger.warning("answer-key refusal for course %s: %s", course_id, message)
        result.errors.append(message)
        _emit(on_progress, "parse", 0.0, message)
        return result

    return await _run_pipeline(
        source=source,
        course_id=course_id,
        parse=lambda: parsers.parse_path(path),
        on_progress=on_progress,
        embedder=embedder,
        persist_dir=persist_dir,
        doc_id=doc_id,
    )


async def ingest_url(
    url: str,
    course_id: str,
    on_progress: ProgressCallback | None = None,
    doc_id: str | None = None,
    embedder: Embedder | None = None,
    persist_dir: Path | None = None,
) -> IngestResult:
    """Fetch and ingest a web page. Same contract as `ingest_file`."""
    if _looks_like_answer_key(url):
        result = IngestResult(source=url, course_id=course_id)
        message = f"refused {url}: looks like an answer key (Tech Spec §5.3)"
        logger.warning("answer-key refusal for course %s: %s", course_id, message)
        result.errors.append(message)
        _emit(on_progress, "parse", 0.0, message)
        return result

    return await _run_pipeline(
        source=url,
        course_id=course_id,
        parse=lambda: parsers.parse_url(url),
        on_progress=on_progress,
        embedder=embedder,
        persist_dir=persist_dir,
        doc_id=doc_id,
    )


async def ingest_paths(
    paths: Iterable[str | Path],
    course_id: str,
    on_progress: ProgressCallback | None = None,
    embedder: Embedder | None = None,
    persist_dir: Path | None = None,
) -> list[IngestResult]:
    """Ingest a batch sequentially; one failure does not stop the rest.

    Sequential by design — the local embedding model is the bottleneck and
    parallel calls would only contend for it.
    """
    items: Sequence[str | Path] = list(paths)
    results: list[IngestResult] = []
    for index, path in enumerate(items, start=1):

        def scoped(stage: str, percent: float, message: str, position: int = index) -> None:
            _emit(on_progress, stage, percent, f"[{position}/{len(items)}] {message}")

        results.append(
            await ingest_file(
                path,
                course_id,
                on_progress=scoped if on_progress else None,
                embedder=embedder,
                persist_dir=persist_dir,
            )
        )
    return results


__all__ = [
    "ANSWER_KEY_PATTERNS",
    "STAGES",
    "IngestResult",
    "ProgressCallback",
    "ingest_file",
    "ingest_paths",
    "ingest_url",
]
