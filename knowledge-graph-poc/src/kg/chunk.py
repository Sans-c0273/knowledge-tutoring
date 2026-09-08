"""S2 chunking (DESIGN §5, D8): greedy grouping of whole Units into model-sized chunks.

Rules, in order: units with no text are dropped (an ``empty_page`` carries
nothing to extract); a unit whose estimate exceeds ``max_tokens`` is split at
blank lines (then line breaks, whitespace, and finally characters for
space-less Thai) into sub-units that inherit its locator and heading path and
get ids ``U3.1, U3.2, ...``; units are then packed greedily up to
``target_tokens`` without ever crossing a source file; a trailing chunk of a
source smaller than ``min_tokens`` merges backwards into its predecessor when
the merge stays within ``max_tokens``. Locators live on units, never on chunks.

``estimate_tokens`` is a deterministic offline heuristic: one token per
whitespace-separated ASCII word (plus one per further 8 characters) and one per
non-ASCII character — Thai tokenises at roughly one token per character on
current models, and words carry no spaces, so anything cheaper under-counts and
lets chunks overshoot ``target_tokens``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from kg.config import ChunkingConfig
from kg.convert import HEADING_SEP, Unit


def estimate_tokens(text: str) -> int:
    total = 0
    for piece in text.split():
        ascii_n = sum(1 for ch in piece if ord(ch) < 128)
        if ascii_n:
            total += 1 + ascii_n // 8
        total += len(piece) - ascii_n
    return total


@dataclass
class Chunk:
    cid: str
    source: str
    units: list[Unit] = field(default_factory=list)

    @property
    def unit_ids(self) -> list[str]:
        return [u.uid for u in self.units]

    @property
    def locators(self) -> list[str]:
        return [u.locator for u in self.units]

    @property
    def tokens(self) -> int:
        return _tokens(self.units)


# ------------------------------------------------------------- oversize split


def _pack(pieces: list[str], sep: str, limit: int, finer: Callable[[str], list[str]] | None) -> list[str]:
    """Greedily join `pieces` with `sep` up to `limit` tokens; a piece over the limit is broken by `finer`."""
    out: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for piece in pieces:
        if not piece.strip():
            continue
        t = estimate_tokens(piece)
        if t > limit:
            if current:
                out.append(sep.join(current))
                current, current_tokens = [], 0
            out.extend(finer(piece) if finer is not None else [piece])
            continue
        if current and current_tokens + t > limit:
            out.append(sep.join(current))
            current, current_tokens = [], 0
        current.append(piece)
        current_tokens += t
    if current:
        out.append(sep.join(current))
    return out


def _split_text(text: str, limit: int) -> list[str]:
    def by_chars(s: str) -> list[str]:
        # No whitespace at all (typical for Thai): cut by characters so each part fits.
        step = max(1, limit)  # non-ASCII counts 1 token per char, ASCII 1 per word (+1/8 chars): `limit` chars <= limit tokens
        parts = [s[i : i + step] for i in range(0, len(s), step)]
        return [p for p in parts if p.strip()]

    def by_words(s: str) -> list[str]:
        return _pack(s.split(), " ", limit, by_chars)

    def by_lines(s: str) -> list[str]:
        return _pack(s.split("\n"), "\n", limit, by_words)

    return _pack(text.split("\n\n"), "\n\n", limit, by_lines)


def split_unit(unit: Unit, limit: int) -> list[Unit]:
    parts = _split_text(unit.text, limit)
    if len(parts) <= 1:
        return [unit]
    return [
        Unit(
            uid=f"{unit.uid}.{i}",
            locator=unit.locator,
            heading_path=list(unit.heading_path),
            text=part,
            kind=unit.kind,
            source=unit.source,
        )
        for i, part in enumerate(parts, start=1)
    ]


# ---------------------------------------------------------------- build_chunks


def build_chunks(units: list[Unit], cfg: ChunkingConfig) -> list[Chunk]:
    prepared: list[Unit] = []
    for unit in units:
        if not unit.text.strip():
            continue
        if estimate_tokens(unit.text) > cfg.max_tokens:
            prepared.extend(split_unit(unit, cfg.target_tokens))
        else:
            prepared.append(unit)

    groups: list[list[Unit]] = []
    current: list[Unit] = []
    current_tokens = 0
    for unit in prepared:
        t = estimate_tokens(unit.text)
        if current and (unit.source != current[0].source or current_tokens + t > cfg.target_tokens):
            groups.append(current)
            current, current_tokens = [], 0
        current.append(unit)
        current_tokens += t
    if current:
        groups.append(current)

    merged: list[list[Unit]] = []
    for i, group in enumerate(groups):
        last_of_source = i + 1 == len(groups) or groups[i + 1][0].source != group[0].source
        if (
            last_of_source
            and merged
            and merged[-1][0].source == group[0].source
            and _tokens(group) < cfg.min_tokens
            and _tokens(merged[-1]) + _tokens(group) <= cfg.max_tokens
        ):
            merged[-1].extend(group)
        else:
            merged.append(group)

    return [Chunk(cid=f"C{n}", source=group[0].source, units=group) for n, group in enumerate(merged, start=1)]


def _tokens(group: list[Unit]) -> int:
    return sum(estimate_tokens(u.text) for u in group)


# ---------------------------------------------------------------- render_chunk


def unit_marker(unit: Unit) -> str:
    return f"<<{unit.uid} | {unit.locator} | {HEADING_SEP.join(unit.heading_path)}>>"


def render_chunk(chunk: Chunk) -> str:
    """Prompt text: each unit prefixed with ``<<uid | locator | Heading > Path>>`` (D8)."""
    return "\n\n".join(f"{unit_marker(u)}\n{u.text.strip()}" for u in chunk.units) + "\n"


__all__ = ["Chunk", "HEADING_SEP", "build_chunks", "estimate_tokens", "render_chunk", "split_unit", "unit_marker"]
