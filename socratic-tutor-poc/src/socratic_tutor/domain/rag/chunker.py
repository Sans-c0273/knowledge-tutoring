"""Segments → retrieval chunks.

Three rules, in priority order:

1. **Never mix scripts.** A chunk is Thai or English, never both. Thai has no
   inter-word spaces, so language is detected from the Unicode block, never from
   whitespace tokenization. A retrieval hit that mixes scripts poisons both the
   embedding and the generated answer's language (Addendum §"Language & review").
2. **Respect authored and natural boundaries.** Chunking never crosses a segment
   boundary, so `<!-- chunk: ... -->` metadata in markdown, slide breaks, page
   breaks and headings all survive as-is. Inside a segment, paragraphs and
   sentences are the split points.
3. **Carry provenance.** Every chunk keeps its segment's `source_ref` and topic
   (Tech Spec §4.2 `source_reference_required`).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from socratic_tutor.domain.rag.parsers import ParsedSegment

#: Target window, in BGE-M3-ish tokens. `MAX` is a hard ceiling per chunk;
#: `OVERLAP` is re-emitted at the head of the next chunk of the same run.
TARGET_TOKENS = 400
MAX_TOKENS = 500
OVERLAP_TOKENS = 60

_THAI_CHARS = re.compile(r"[฀-๿]")
#: Three letters or more = English prose. Shorter Latin runs are algebra
#: ("x", "2x") and stay with the sentence they sit in, whatever its script.
_ENGLISH_WORD = re.compile(r"[A-Za-z]{3,}")
_WORD = re.compile(r"\S+")
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
#: Sentence break after Latin terminal punctuation only. Thai writes without
#: sentence periods, so a Thai paragraph stays one unit and is split by budget.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """One indexed unit of course content."""

    text: str
    source_ref: str
    lang: str
    topic: str | None = None
    chunk_id: str = ""
    ordinal: int = 0


def estimate_tokens(text: str) -> int:
    """Approximate BGE-M3 token count for mixed Thai/English text.

    Thai is counted per codepoint (~2.5 chars/token for the SentencePiece
    vocabulary); everything else per whitespace word (~1.3 tokens/word). Exact
    counts would need the tokenizer, which means downloading the model — the
    chunker must work offline.
    """
    thai_chars = len(_THAI_CHARS.findall(text))
    non_thai = _THAI_CHARS.sub(" ", text)
    words = len(_WORD.findall(non_thai))
    return max(1, round(thai_chars / 2.5 + words * 1.3))


def _units(text: str) -> list[str]:
    """Split a segment into the smallest units the packer will not break apart."""
    units: list[str] = []
    for paragraph in _PARAGRAPH_SPLIT.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in _SENTENCE_SPLIT.split(paragraph):
            sentence = sentence.strip()
            if sentence:
                units.append(sentence)
    return units


def _split_oversized(unit: str) -> list[str]:
    """Break a single over-budget unit on whitespace, staying under `MAX_TOKENS`.

    Pieces inherit the language of the unit they came from: splitting a Thai
    sentence can yield a piece of bare algebra ("2(x + 3)"), which belongs to the
    Thai explanation around it and must not be relabelled English.
    """
    pieces: list[str] = []
    current: list[str] = []
    for word in _WORD.findall(unit):
        candidate = [*current, word]
        if current and estimate_tokens(" ".join(candidate)) > MAX_TOKENS:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current = candidate
    if current:
        pieces.append(" ".join(current))

    # Unbroken Thai runs have no whitespace to split on, so oversized pieces get
    # a codepoint-budget cut. Approximate, but bounded — never a 3000-char chunk.
    bounded: list[str] = []
    for piece in pieces or [unit]:
        if estimate_tokens(piece) <= MAX_TOKENS:
            bounded.append(piece)
            continue
        width = max(1, int(len(piece) * MAX_TOKENS / estimate_tokens(piece)))
        bounded.extend(piece[start : start + width] for start in range(0, len(piece), width))
    return bounded


def _word_language(word: str) -> str | None:
    """`"th"`, `"en"`, or `None` for a word that carries no script signal.

    Bare algebra ("2(x", "=", "x") is `None` on purpose: it belongs to whichever
    explanation surrounds it, and treating a variable name as English would cut
    Thai sentences in half.
    """
    if _THAI_CHARS.search(word):
        return "th"
    if _ENGLISH_WORD.search(word):
        return "en"
    return None


def _script_pieces(unit: str, default_lang: str) -> list[tuple[str, str]]:
    """Split one unit at script transitions into `(lang, text)` pieces.

    Sentence boundaries alone are not enough: Thai does not end sentences with a
    period, so "…ผกผัน Multiplication and division are inverses." arrives as a
    single unit and must still be cut before the English clause.
    """
    pieces: list[tuple[str, str]] = []
    buffer: list[str] = []
    current: str | None = None

    for word in _WORD.findall(unit):
        lang = _word_language(word)
        if lang is not None and current is not None and lang != current:
            pieces.append((current, " ".join(buffer)))
            buffer = []
        if lang is not None:
            current = lang
        buffer.append(word)

    if buffer:
        pieces.append((current or default_lang, " ".join(buffer)))
    return pieces


def _language_runs(units: list[str], default_lang: str) -> list[tuple[str, list[str]]]:
    """Group unit fragments into runs of a single script."""
    runs: list[tuple[str, list[str]]] = []
    for unit in units:
        for lang, text in _script_pieces(unit, default_lang):
            if runs and runs[-1][0] == lang:
                runs[-1][1].append(text)
            else:
                runs.append((lang, [text]))
    return runs


def _pack(units: list[str]) -> list[str]:
    """Pack units of one language into ~`TARGET_TOKENS` chunks with overlap."""
    expanded: list[str] = []
    for unit in units:
        if estimate_tokens(unit) > MAX_TOKENS:
            expanded.extend(_split_oversized(unit))
        else:
            expanded.append(unit)

    chunks: list[str] = []
    current: list[str] = []
    #: True while `current` holds nothing but text already emitted as overlap —
    #: flushing it again would duplicate a chunk rather than add content.
    overlap_only = False

    def flush() -> list[str]:
        """Emit `current` as a chunk and return the tail to carry as overlap."""
        chunks.append(" ".join(current).strip())
        carried: list[str] = []
        for unit in reversed(current):
            if estimate_tokens(" ".join([unit, *carried])) > OVERLAP_TOKENS:
                break
            carried.insert(0, unit)
        return carried if len(carried) < len(current) else []

    for unit in expanded:
        if current and estimate_tokens(" ".join([*current, unit])) > MAX_TOKENS:
            current = flush()
        current = [*current, unit]
        overlap_only = False
        if estimate_tokens(" ".join(current)) >= TARGET_TOKENS:
            current = flush()
            overlap_only = True

    if current and not overlap_only:
        chunks.append(" ".join(current).strip())
    return [c for c in chunks if c]


def _doc_key(segments: list[ParsedSegment]) -> str:
    seed = segments[0].source_ref if segments else "empty"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]


def chunk_segments(segments: list[ParsedSegment], doc_id: str | None = None) -> list[Chunk]:
    """Turn parsed segments into retrieval chunks.

    `doc_id` prefixes generated chunk ids so ids stay unique across documents in
    a course collection. Segments that declared their own `chunk_id` keep it.
    """
    if not segments:
        return []
    prefix = doc_id or _doc_key(segments)

    chunks: list[Chunk] = []
    for seg_index, segment in enumerate(segments):
        runs = _language_runs(_units(segment.text), segment.lang_hint)
        part = 0
        for lang, units in runs:
            for body in _pack(units):
                base = segment.chunk_id or f"{prefix}-{seg_index:03d}"
                chunk_id = base if segment.chunk_id and part == 0 else f"{base}-{part:02d}"
                chunks.append(
                    Chunk(
                        text=body,
                        source_ref=segment.source_ref,
                        lang=lang,
                        topic=segment.topic,
                        chunk_id=chunk_id,
                        ordinal=len(chunks),
                    )
                )
                part += 1
    return chunks


__all__ = [
    "MAX_TOKENS",
    "OVERLAP_TOKENS",
    "TARGET_TOKENS",
    "Chunk",
    "chunk_segments",
    "estimate_tokens",
]
