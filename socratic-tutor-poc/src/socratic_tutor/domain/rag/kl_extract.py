"""LLM-assisted KL Map extraction from ingested course content.

The Addendum settles the authoring model: the map is **LLM-extracted from
uploaded content, then human-reviewed before it goes live**. So everything here
produces a *draft*. It is written to `content/drafts/klmap-<course>.yaml` and
returned with its `ValidationReport`; nothing in this module makes a map live.

Two passes, following `~/src/kg-mapper-poc` (the owner's earlier extraction POC):

1. **Concepts** — per batch of chunks, extract the concepts the material
   actually teaches. A roster of what earlier batches found is carried forward
   so the same concept is not re-invented under a new id, and so a Thai passage
   can attach `name_th` to a concept first seen in English.
2. **Edges** — per batch, propose typed relations, given the full canonical
   roster plus that batch's passage. Asking one call to do both jobs at once
   produced worse edges in the reference POC.

Ids the model returns are never trusted as final: after the concept passes,
concepts are consolidated and renumbered to canonical `C001`-style ids, and the
edge passes only ever see those.

**Nothing is silently repaired.** An edge naming an unknown concept or pointing
a node at itself is dropped with a `KX1xx` warning naming it; a graph that
violates a Tech Spec §2.3 rule (cycle, both-direction relation) is returned as
it was extracted, with the loader's errors attached, because a human has to see
what the model actually proposed in order to fix it.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from socratic_tutor.config import get_settings
from socratic_tutor.domain.klmap import (
    KLEdge,
    KLMap,
    KLNode,
    ValidationReport,
    normalise_text,
    save_klmap,
    validate_klmap,
)
from socratic_tutor.domain.rag.chunker import Chunk, estimate_tokens
from socratic_tutor.models.enums import Relation
from socratic_tutor.providers import LLMProvider, Msg, ProviderError, get_provider

logger = logging.getLogger(__name__)

CONCEPTS_PROMPT_VERSION = "klmap-concepts@1"
EDGES_PROMPT_VERSION = "klmap-edges@1"

#: Chunks per model call, by estimated tokens. Small enough that the passage,
#: the roster and the answer fit comfortably; large enough that a chapter's
#: concepts are seen together, which is where the edges come from.
BATCH_TOKEN_BUDGET = 2500

#: Progress callback for the ingestion UI's `kl-extract` stage: `(percent, message)`
#: with percent 0–100 *within extraction*, mirroring `ingest`'s convention.
ExtractProgress = Callable[[float, str], None]

#: Directed meaning of each relation, as Tech Spec §2.3 defines it. Keyed by the
#: closed vocabulary so the prompt cannot drift from `Relation`.
RELATION_MEANING: dict[Relation, str] = {
    Relation.PREREQUISITE_OF: (
        "source must be understood before target can be learned "
        "(source is the prerequisite, target is the concept that needs it)"
    ),
    Relation.RELATED_TO: "source and target are meaningfully connected; read symmetrically",
    Relation.NEXT_TOPIC: "target is what the course teaches after source",
    Relation.PART_OF: "source is a component of the larger topic target",
    Relation.USES: "source applies or depends on target as a tool or technique",
}


# ----------------------------------------------------------------- wire models


class ExtractedConcept(BaseModel):
    """One concept as the model reports it, before consolidation.

    `id` is the model's own handle, used only to connect this concept to the
    roster of the next batch; `extract_klmap` reassigns canonical ids.
    """

    id: str
    name: str
    name_th: str | None = None


class ExtractedConcepts(BaseModel):
    """Concept-pass output. An empty list is a valid answer."""

    concepts: list[ExtractedConcept] = Field(default_factory=list)


class ProposedEdge(BaseModel):
    """One typed relation between two roster ids, read as `from_id -> to_id`."""

    from_id: str
    to_id: str
    relation: Relation


class ProposedEdges(BaseModel):
    """Edge-pass output. An empty list is a valid answer."""

    edges: list[ProposedEdge] = Field(default_factory=list)


# ---------------------------------------------------------------------- prompts


def relation_vocabulary() -> str:
    """The relation block for the edge prompt, generated from `Relation`.

    Generating it means a new relation in the closed vocabulary cannot be added
    to the enum and forgotten in the prompt — this raises instead.
    """
    missing = [relation.value for relation in Relation if relation not in RELATION_MEANING]
    if missing:
        raise ValueError(
            f"RELATION_MEANING is missing an entry for {', '.join(missing)}; "
            "the edge prompt must describe every relation in the closed vocabulary"
        )
    return "\n".join(
        f"- {relation.value} (source -> target): {RELATION_MEANING[relation]}."
        for relation in Relation
    )


CONCEPTS_SYSTEM = """You extract the concepts a course teaches, from a passage of its material.

The passage is a sequence of units. Each unit starts with a marker line of the form
<<UNIT_ID | source reference | language>> followed by that unit's text. The marker is metadata;
it is not part of the content.

Return one entry for every distinct concept the passage actually defines, explains or teaches:
- id: a new identifier of the form C001, C002, ... for a concept not already in the roster below.
  If the concept IS already in the roster, reuse that roster id exactly and do not invent a new one.
- name: the concept's name in English, as the material names it. One concept per entry; do not merge
  two ideas into one name and do not split one idea across several entries.
- name_th: the concept's name in Thai when this passage gives it, otherwise null. When a Thai unit
  describes a concept already in the roster under an English name, reuse that roster id and supply
  name_th so the two names attach to one concept.

Extract teachable concepts, not phrases: "Two-Step Linear Equations" is a concept, "both sides" is not.
Skip headings without content, worked examples that teach nothing new, and anything the passage merely
mentions without explaining. An empty list is a valid answer."""

EDGES_SYSTEM = """You propose typed relationships between the concepts of a course.

You receive:
1. The roster of every concept in the map, as "id | name | Thai name". Ids are opaque: copy them
   verbatim and never invent, shorten or renumber one.
2. A passage of the course material, as units with <<UNIT_ID | source reference | language>> markers.

Propose an edge only where the passage gives clear support for it. Prefer few well-supported edges over
many speculative ones. Never propose an edge from a concept to itself, never repeat the same pair with
the same relation, and only use ids that appear in the roster. An empty list is a valid answer.

Relation vocabulary — use these and nothing else:
"""


def edges_system() -> str:
    return EDGES_SYSTEM + relation_vocabulary() + "\n"


# ------------------------------------------------------------------- rendering


def render_passage(chunks: Sequence[Chunk]) -> str:
    """Chunks as markered units the model can read and cite."""
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        unit_id = chunk.chunk_id or f"u{index}"
        blocks.append(f"<<{unit_id} | {chunk.source_ref} | {chunk.lang}>>\n{chunk.text}")
    return "\n\n".join(blocks)


def batch_chunks(chunks: Sequence[Chunk], budget: int = BATCH_TOKEN_BUDGET) -> list[list[Chunk]]:
    """Group chunks into passages under a token budget, preserving corpus order."""
    batches: list[list[Chunk]] = []
    current: list[Chunk] = []
    used = 0
    for chunk in chunks:
        cost = estimate_tokens(chunk.text)
        if current and used + cost > budget:
            batches.append(current)
            current = []
            used = 0
        current.append(chunk)
        used += cost
    if current:
        batches.append(current)
    return batches


def _render_roster(nodes: Sequence[KLNode]) -> str:
    if not nodes:
        return "(empty — this is the first passage)"
    return "\n".join(f"{node.id} | {node.name} | {node.name_th or '-'}" for node in nodes)


# --------------------------------------------------------------- consolidation


class _Consolidator:
    """Merges concepts across batches and assigns canonical `C001`-style ids.

    Two concepts are the same when their normalised English names match, or when
    one's Thai name matches the other's — that is how a Thai passage's concept
    attaches to the English concept it re-describes, which is the whole point of
    carrying the roster forward.
    """

    def __init__(self) -> None:
        self.nodes: list[KLNode] = []
        self._by_key: dict[str, str] = {}
        self._aliases: dict[str, str] = {}

    def _key(self, text: str | None) -> str | None:
        if not text or not text.strip():
            return None
        return normalise_text(text) or None

    def add(self, raw: ExtractedConcept) -> str | None:
        """Merge one extracted concept; returns its canonical id, or None if unusable."""
        if not raw.name or not raw.name.strip():
            return None

        keys = [key for key in (self._key(raw.name), self._key(raw.name_th)) if key]
        canonical = next((self._by_key[key] for key in keys if key in self._by_key), None)
        if canonical is None and raw.id in self._aliases:
            canonical = self._aliases[raw.id]

        if canonical is None:
            canonical = f"C{len(self.nodes) + 1:03d}"
            self.nodes.append(
                KLNode(id=canonical, name=raw.name.strip(), name_th=(raw.name_th or None))
            )
        else:
            existing = next(node for node in self.nodes if node.id == canonical)
            if existing.name_th is None and raw.name_th:
                existing.name_th = raw.name_th.strip()

        for key in keys:
            self._by_key.setdefault(key, canonical)
        if raw.id:
            self._aliases.setdefault(raw.id, canonical)
        return canonical

    def resolve(self, node_id: str) -> str | None:
        """Canonical id for whatever the edge pass named, or None if unknown."""
        if any(node.id == node_id for node in self.nodes):
            return node_id
        if node_id in self._aliases:
            return self._aliases[node_id]
        key = self._key(node_id)
        return self._by_key.get(key) if key else None


# ------------------------------------------------------------------ extraction


def draft_path(course_id: str, drafts_dir: Path | None = None) -> Path:
    """Where a course's extracted draft lives. Never the live map."""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", course_id.strip()).strip("-") or "course"
    base = drafts_dir or (get_settings().content_dir / "drafts")
    return Path(base) / f"klmap-{slug}.yaml"


def draft_metadata_path(course_id: str, drafts_dir: Path | None = None) -> Path:
    """Sidecar for the draft's extraction metadata and validation report."""
    return draft_path(course_id, drafts_dir).with_suffix(".report.json")


def _save_draft_metadata(
    path: Path,
    course_id: str,
    course_name: str,
    chunks: Sequence[Chunk],
    klmap: KLMap,
    report: ValidationReport,
) -> Path:
    """Persist what the R18 review UI needs but the map YAML cannot hold.

    The KL Map file format is the teacher-editable contract (`domain.klmap`), so
    extraction provenance and the report live beside it rather than inside it.
    Without this the review endpoint would have to re-run extraction — real
    model calls — just to show a reviewer why a draft is flagged.
    """
    payload = {
        "course_id": course_id,
        "course_name": course_name,
        "path": str(path),
        "extracted_at": datetime.now(UTC).isoformat(),
        "prompt_versions": {
            "concepts": CONCEPTS_PROMPT_VERSION,
            "edges": EDGES_PROMPT_VERSION,
        },
        "sources": sorted({chunk.source_ref for chunk in chunks if chunk.source_ref}),
        "node_count": len(klmap.nodes),
        "edge_count": len(klmap.edges),
        "report": report.model_dump(mode="json"),
    }
    metadata = path.with_suffix(".report.json")
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def _label(node_id: str, nodes: Sequence[KLNode]) -> str:
    node = next((n for n in nodes if n.id == node_id), None)
    return f"{node_id} ({node.name})" if node else repr(node_id)


async def extract_klmap(
    chunks: Sequence[Chunk],
    course_id: str,
    provider: LLMProvider | None = None,
    course_name: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    drafts_dir: Path | None = None,
    on_progress: ExtractProgress | None = None,
) -> tuple[KLMap, ValidationReport]:
    """Extract a draft KL Map from a course's ingested chunks.

    Always returns a map and a report, even when the map is invalid: the R18
    review UI renders both, and a human fixes the draft. The draft is written to
    `content/drafts/klmap-<course>.yaml` regardless of validity, so the fix can
    happen in the file.

    Extraction problems are added to the report under `KX` codes: `KX0xx` errors
    (a model call failed, nothing extracted) and `KX1xx` warnings (a proposal was
    dropped, and why). Loader codes (`KL0xx`) come from `validate_klmap`.
    """
    settings = get_settings()
    provider = provider or get_provider("generation")
    model = model or settings.model_for("generation")
    max_tokens = max_tokens or settings.max_tokens_for("generation")
    path = draft_path(course_id, drafts_dir)

    report = ValidationReport(source=str(path))
    batches = batch_chunks(chunks)
    if not batches:
        report.error("KX001", "No content to extract from: the course has no indexed chunks.")
        klmap = KLMap(course_id=course_id, course_name=course_name or course_id)
        save_klmap(klmap, path)
        _save_draft_metadata(path, course_id, course_name or course_id, chunks, klmap, report)
        return klmap, report

    def progress(fraction: float, message: str) -> None:
        if on_progress is None:
            return
        try:
            on_progress(round(min(max(fraction, 0.0), 1.0) * 100, 1), message)
        except Exception:
            logger.exception("KL extraction progress callback failed")

    # --- pass 1: concepts, roster carried forward across batches
    consolidator = _Consolidator()
    total_calls = len(batches) * 2
    for index, batch in enumerate(batches):
        progress(index / total_calls, f"Extracting concepts ({index + 1}/{len(batches)})")
        user = (
            f"## Roster of concepts already extracted (id | name | Thai name)\n"
            f"{_render_roster(consolidator.nodes)}\n\n"
            f"## Passage\n{render_passage(batch)}"
        )
        try:
            result = await provider.complete_structured(
                model=model,
                system=CONCEPTS_SYSTEM,
                messages=[Msg(role="user", content=user)],
                schema=ExtractedConcepts,
                max_tokens=max_tokens,
                temperature=0.0,
            )
        except ProviderError as exc:
            logger.warning("concept extraction failed for batch %d: %s", index + 1, exc)
            report.error(
                "KX002",
                f"Concept extraction failed on passage {index + 1} of {len(batches)}: {exc}. "
                "The draft below is missing whatever that passage would have contributed.",
            )
            continue

        extracted = result.value
        assert isinstance(extracted, ExtractedConcepts)
        for raw in extracted.concepts:
            if consolidator.add(raw) is None:
                report.warn(
                    "KX101",
                    f"Dropped an extracted concept with an empty name (model id {raw.id!r}).",
                )

    if not consolidator.nodes:
        report.error(
            "KX003",
            "No concepts were extracted from this course's content. Check that the uploaded "
            "material teaches named concepts, or review the model's output.",
        )

    # --- pass 2: edges, against the canonical roster
    seen: set[tuple[str, str, Relation]] = set()
    edges: list[KLEdge] = []
    roster = _render_roster(consolidator.nodes)

    for index, batch in enumerate(batches):
        if not consolidator.nodes:
            break
        progress(
            (len(batches) + index) / total_calls,
            f"Proposing relationships ({index + 1}/{len(batches)})",
        )
        user = f"## Roster (id | name | Thai name)\n{roster}\n\n## Passage\n{render_passage(batch)}"
        try:
            result = await provider.complete_structured(
                model=model,
                system=edges_system(),
                messages=[Msg(role="user", content=user)],
                schema=ProposedEdges,
                max_tokens=max_tokens,
                temperature=0.0,
            )
        except ProviderError as exc:
            logger.warning("edge extraction failed for batch %d: %s", index + 1, exc)
            report.error(
                "KX004",
                f"Relationship extraction failed on passage {index + 1} of {len(batches)}: {exc}. "
                "Concepts from that passage are present but may have no edges.",
            )
            continue

        proposed = result.value
        assert isinstance(proposed, ProposedEdges)
        for edge in proposed.edges:
            source = consolidator.resolve(edge.from_id)
            target = consolidator.resolve(edge.to_id)
            if source is None or target is None:
                unknown = [
                    raw
                    for raw, resolved in ((edge.from_id, source), (edge.to_id, target))
                    if resolved is None
                ]
                report.warn(
                    "KX102",
                    f"Dropped proposed edge {edge.from_id!r} -{edge.relation.value}-> "
                    f"{edge.to_id!r}: no concept has id {', '.join(repr(u) for u in unknown)}.",
                )
                continue
            if source == target:
                report.warn(
                    "KX103",
                    f"Dropped proposed edge: {_label(source, consolidator.nodes)} "
                    f"points at itself via '{edge.relation.value}'.",
                )
                continue
            triple = (source, target, edge.relation)
            if triple in seen:
                report.warn(
                    "KX104",
                    f"Dropped duplicate proposed edge {_label(source, consolidator.nodes)} "
                    f"-{edge.relation.value}-> {_label(target, consolidator.nodes)}.",
                )
                continue
            seen.add(triple)
            edges.append(KLEdge(from_=source, to=target, relation=edge.relation))

    progress(1.0, f"{len(consolidator.nodes)} concept(s), {len(edges)} relationship(s)")

    # --- the draft, exactly as extracted, plus everything the loader thinks of it
    klmap = KLMap(
        course_id=course_id,
        course_name=course_name or course_id,
        nodes=consolidator.nodes,
        edges=edges,
    )
    save_klmap(klmap, path)

    _, validation = validate_klmap(klmap.model_dump(by_alias=True, mode="json"), source=str(path))
    report.errors.extend(validation.errors)
    report.warnings.extend(validation.warnings)
    _save_draft_metadata(path, course_id, course_name or course_id, chunks, klmap, report)
    return klmap, report


__all__ = [
    "BATCH_TOKEN_BUDGET",
    "CONCEPTS_PROMPT_VERSION",
    "CONCEPTS_SYSTEM",
    "EDGES_PROMPT_VERSION",
    "RELATION_MEANING",
    "ExtractProgress",
    "ExtractedConcept",
    "ExtractedConcepts",
    "ProposedEdge",
    "ProposedEdges",
    "batch_chunks",
    "draft_metadata_path",
    "draft_path",
    "edges_system",
    "extract_klmap",
    "relation_vocabulary",
    "render_passage",
]
