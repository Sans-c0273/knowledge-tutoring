"""RAG ingestion and retrieval (Tech Spec §1 step 2a, §7 `POST /content/rag`).

Pipeline: `parsers` (bytes → text + provenance) → `chunker` (text → retrieval
chunks, never mixing scripts) → `embeddings` (local BGE-M3) → `store` (Chroma,
one collection per course). `ingest` wires the four together and reports progress.

**Canonical-answer isolation (Tech Spec §5.2, §5.3).** This corpus feeds Call C's
context. Canonical solutions and answer keys live server-side and are retrieved
only into Call B, so they must never be ingested here — `ingest` refuses sources
that look like answer keys rather than indexing them (see `ANSWER_KEY_PATTERNS`).
"""

from __future__ import annotations

from socratic_tutor.domain.rag.chunker import Chunk, chunk_segments
from socratic_tutor.domain.rag.embeddings import (
    AzureEmbedder,
    BGEM3Embedder,
    Embedder,
    EmbedderError,
    FakeEmbedder,
    get_embedder,
)
from socratic_tutor.domain.rag.ingest import (
    ANSWER_KEY_PATTERNS,
    IngestResult,
    ingest_file,
    ingest_paths,
    ingest_url,
)
from socratic_tutor.domain.rag.parsers import (
    ParsedSegment,
    ParseError,
    detect_lang,
    parse_path,
    parse_url,
)
from socratic_tutor.domain.rag.store import StoreIntegrityError, query, upsert_chunks

__all__ = [
    "ANSWER_KEY_PATTERNS",
    "AzureEmbedder",
    "BGEM3Embedder",
    "Chunk",
    "Embedder",
    "EmbedderError",
    "FakeEmbedder",
    "IngestResult",
    "ParseError",
    "ParsedSegment",
    "StoreIntegrityError",
    "chunk_segments",
    "detect_lang",
    "get_embedder",
    "ingest_file",
    "ingest_paths",
    "ingest_url",
    "parse_path",
    "parse_url",
    "query",
    "upsert_chunks",
]
