"""Chroma vector store — one collection per course (Tech Spec §1 step 2a).

Retrieval is hybrid-ish: a metadata pre-filter (topic, language) narrows the
candidate set, cosine vector search ranks it, and a lexical overlap score
re-ranks the top candidates. Pure vector search on a small POC corpus misses
exact-term questions ("what is the distributive property"); pure keyword search
misses paraphrase.

**Provenance is not optional.** Tech Spec §4.2 requires a source reference on any
factual response, so a chunk without a `source_ref` is rejected at write time and
never returned at read time. Losing a hit is recoverable; an unsourced claim
reaching a student is not.

**Canonical answers never live here** (Tech Spec §5.2/§5.3): this collection is
Call C's context, and answer keys must stay server-side in Call B. `ingest`
enforces that at the door.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from socratic_tutor.config import get_settings
from socratic_tutor.domain.rag.chunker import Chunk
from socratic_tutor.domain.rag.embeddings import Embedder, get_embedder, text_features
from socratic_tutor.models.knowledge import RetrievedChunk

logger = logging.getLogger(__name__)

#: Blend of cosine similarity and lexical overlap in the final ranking.
VECTOR_WEIGHT = 0.75
KEYWORD_WEIGHT = 0.25
#: Vector candidates fetched per requested hit before lexical re-ranking.
CANDIDATE_MULTIPLIER = 4

_INVALID_NAME_CHARS = re.compile(r"[^a-zA-Z0-9._-]+")
_clients: dict[str, Any] = {}


class StoreIntegrityError(Exception):
    """A chunk violates a storage contract — currently, missing provenance."""


def _default_persist_dir() -> Path:
    settings = get_settings()
    if settings.chroma_dir is None:  # pragma: no cover — the validator always derives it
        return settings.data_dir / "chroma"
    return Path(settings.chroma_dir)


def collection_name(course_id: str) -> str:
    """Chroma-legal collection name for a course id.

    Chroma requires 3–63 chars of `[a-zA-Z0-9._-]` starting and ending
    alphanumeric, so the id is slugged rather than passed through.
    """
    slug = _INVALID_NAME_CHARS.sub("-", course_id.strip().lower()).strip("-._")
    return f"course-{slug or 'default'}"[:63].rstrip("-._")


def get_client(persist_dir: Path | None = None) -> Any:
    """Persistent Chroma client, cached per directory.

    `allow_reset` stays off: `client.reset()` erases every course's corpus in one
    call, and re-ingesting means re-uploading material the POC has no other copy
    of. No route needs it, so the capability should not exist at runtime. Tests
    that want a clean store point `persist_dir` at a temp directory instead.
    """
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    directory = Path(persist_dir or _default_persist_dir())
    directory.mkdir(parents=True, exist_ok=True)
    key = str(directory.resolve())
    if key not in _clients:
        _clients[key] = chromadb.PersistentClient(
            path=key,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
        )
    return _clients[key]


def get_collection(course_id: str, persist_dir: Path | None = None) -> Any:
    return get_client(persist_dir).get_or_create_collection(
        name=collection_name(course_id),
        metadata={"hnsw:space": "cosine"},
    )


def reset_client_cache() -> None:
    """Drop cached clients — tests point the store at a fresh tmp directory."""
    _clients.clear()


def _metadata(chunk: Chunk, course_id: str) -> dict[str, Any]:
    return {
        "course_id": course_id,
        "source_ref": chunk.source_ref,
        "lang": chunk.lang,
        "topic": chunk.topic or "",
        "ordinal": chunk.ordinal,
    }


def upsert_chunks(
    course_id: str,
    chunks: list[Chunk],
    embeddings: list[list[float]] | None = None,
    embedder: Embedder | None = None,
    persist_dir: Path | None = None,
) -> int:
    """Index chunks for a course; returns the number written.

    Pass `embeddings` when the caller has already embedded (the ingest pipeline
    does, so it can report an `embed` stage separately); otherwise they are
    computed here.

    Raises `StoreIntegrityError` if any chunk lacks a `source_ref` — the whole
    batch is rejected so a half-written document is never left behind.
    """
    if not chunks:
        return 0

    unsourced = [chunk.chunk_id or f"#{chunk.ordinal}" for chunk in chunks if not chunk.source_ref]
    if unsourced:
        raise StoreIntegrityError(
            f"{len(unsourced)} chunk(s) have no source_ref and cannot be indexed: "
            f"{', '.join(unsourced[:5])}"
        )

    if embeddings is None:
        embeddings = (embedder or get_embedder()).embed_texts([chunk.text for chunk in chunks])
    if len(embeddings) != len(chunks):
        raise StoreIntegrityError(
            f"embedding count {len(embeddings)} does not match chunk count {len(chunks)}"
        )

    collection = get_collection(course_id, persist_dir)
    collection.upsert(
        ids=[chunk.chunk_id or f"{collection_name(course_id)}-{chunk.ordinal}" for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        embeddings=embeddings,
        metadatas=[_metadata(chunk, course_id) for chunk in chunks],
    )
    return len(chunks)


def _where(lang: str | None, topic: str | None) -> dict[str, Any] | None:
    """Metadata pre-filter: the keyword half of the hybrid search."""
    clauses: list[dict[str, Any]] = []
    if lang:
        clauses.append({"lang": str(getattr(lang, "value", lang))})
    if topic:
        clauses.append({"topic": topic})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _keyword_score(query_text: str, document: str) -> float:
    """Fraction of the query's lexical features present in the document."""
    query_features = set(text_features(query_text))
    if not query_features:
        return 0.0
    document_features = set(text_features(document))
    return len(query_features & document_features) / len(query_features)


def query(
    course_id: str,
    query_text: str,
    top_k: int = 5,
    lang: str | None = None,
    topic: str | None = None,
    embedder: Embedder | None = None,
    persist_dir: Path | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the best chunks for a query, each carrying its source reference.

    Returns `[]` when the course has no matching content — an empty retrieval set
    is a meaningful signal (Teaching Policy row 4: no evidence → no direct
    factual answer), not an error.
    """
    if top_k <= 0 or not query_text.strip():
        return []

    collection = get_collection(course_id, persist_dir)
    if collection.count() == 0:
        return []

    embedder = embedder or get_embedder()
    result = collection.query(
        query_embeddings=[embedder.embed_query(query_text)],
        n_results=min(max(top_k * CANDIDATE_MULTIPLIER, top_k), collection.count()),
        where=_where(lang, topic),
        include=["documents", "metadatas", "distances"],
    )

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    hits: list[RetrievedChunk] = []
    for document, metadata, distance in zip(documents, metadatas, distances, strict=False):
        metadata = metadata or {}
        source_ref = str(metadata.get("source_ref") or "")
        if not source_ref:
            # Should be unreachable: upsert_chunks rejects these. If it happens,
            # the chunk is dropped rather than served without provenance.
            logger.error(
                "dropping retrieved chunk with no source_ref from course %s: %.60s",
                course_id,
                document,
            )
            continue
        similarity = 1.0 - float(distance)
        score = VECTOR_WEIGHT * similarity + KEYWORD_WEIGHT * _keyword_score(query_text, document)
        hits.append(
            RetrievedChunk(
                text=document,
                source_ref=source_ref,
                topic=str(metadata.get("topic") or ""),
                lang=str(metadata.get("lang") or "en"),
                score=round(score, 6),
            )
        )

    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits[:top_k]


def count(course_id: str, persist_dir: Path | None = None) -> int:
    return get_collection(course_id, persist_dir).count()


def all_chunks(course_id: str, persist_dir: Path | None = None) -> list[Chunk]:
    """Every indexed chunk for a course, in stored order.

    KL Map extraction reads the whole course rather than one upload, because a
    concept graph built from a single file would miss every relation that
    crosses documents.
    """
    collection = get_collection(course_id, persist_dir)
    if collection.count() == 0:
        return []

    result = collection.get(include=["documents", "metadatas"])
    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []

    chunks: list[Chunk] = []
    for chunk_id, document, metadata in zip(ids, documents, metadatas, strict=False):
        metadata = metadata or {}
        chunks.append(
            Chunk(
                text=document or "",
                source_ref=str(metadata.get("source_ref") or ""),
                lang=str(metadata.get("lang") or "en"),
                topic=str(metadata.get("topic") or "") or None,
                chunk_id=str(chunk_id),
                ordinal=int(metadata.get("ordinal") or 0),
            )
        )
    chunks.sort(key=lambda chunk: chunk.ordinal)
    return chunks


def delete_course(course_id: str, persist_dir: Path | None = None) -> None:
    """Drop a course collection; a course that was never indexed is a no-op."""
    client = get_client(persist_dir)
    try:
        client.delete_collection(collection_name(course_id))
    except Exception as exc:  # noqa: BLE001 — Chroma raises ValueError/NotFoundError by version
        logger.info("no collection to delete for course %s (%s)", course_id, exc)


__all__ = [
    "CANDIDATE_MULTIPLIER",
    "KEYWORD_WEIGHT",
    "VECTOR_WEIGHT",
    "StoreIntegrityError",
    "all_chunks",
    "collection_name",
    "count",
    "delete_course",
    "get_client",
    "get_collection",
    "query",
    "reset_client_cache",
    "upsert_chunks",
]
