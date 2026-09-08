"""S5 edges (DESIGN §6, §8.0; D6; R8).

One model call per chunk that produced at least one draft: the call carries the
compact roster of every linkable node (capped at ``limits.roster_cap``, with a
warning when truncated), the chunk's own drafts in full, and the passage. The
run schema's edges-output model is the wire schema; proposals become internal
``Edge``s. Unknown ids, out-of-schema types and self edges are dropped here with
a reason; everything else is decided by the gates (S7).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from kg.chunk import render_chunk
from kg.config import Config
from kg.gates import Edge, Rejection
from kg.prompts import edges as prompt
from kg.schemas import SCHEMAS
from kg.stages.consolidate import NodeDraft

CallStage = Callable[..., Any]


@dataclass(frozen=True)
class RosterEntry:
    id: str
    title: str
    aliases: tuple[str, ...] = ()


@dataclass
class Roster:
    entries: list[RosterEntry]
    truncated: bool
    total: int

    def render(self) -> str:
        return "\n".join(f"{e.id} | {e.title} | {', '.join(e.aliases)}" for e in self.entries)


@dataclass
class EdgesResult:
    edges: list[Edge] = field(default_factory=list)
    dropped: list[Rejection] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def build_roster(drafts: list[Any], cap: int) -> Roster:
    entries = [RosterEntry(id=d.id, title=d.title, aliases=tuple(d.aliases)) for d in drafts]
    total = len(entries)
    truncated = total > cap
    return Roster(entries=entries[:cap], truncated=truncated, total=total)


def _draft_block(d: NodeDraft) -> str:
    lines = [f"### {d.id} — {d.title}"]
    if d.aliases:
        lines.append(f"aliases: {', '.join(d.aliases)}")
    lines.append(d.definition)
    for s in d.sources:
        if s.quote:
            lines.append(f"> {s.quote}  ({s.locator})")
    return "\n".join(lines)


def user_text(roster: Roster, drafts: list[NodeDraft], chunk: Any) -> str:
    parts = [
        "## Roster (id | title | aliases)",
        roster.render(),
        "",
        "## Notes extracted from this passage",
        "\n\n".join(_draft_block(d) for d in drafts),
        "",
        "## Passage",
        render_chunk(chunk),
    ]
    return "\n".join(parts)


def propose(
    drafts: list[NodeDraft],
    chunks: list[Any],
    cfg: Config,
    *,
    call_stage: CallStage,
    ledger: Any = None,
    roster_drafts: list[Any] | None = None,
) -> EdgesResult:
    """Edges for the drafts of these chunks. `roster_drafts` (default: `drafts`) is every node that may be linked."""
    result = EdgesResult()
    if not drafts:
        return result
    schema = SCHEMAS[cfg.corpus.schema]
    linkable = roster_drafts if roster_drafts is not None else drafts
    roster = build_roster(linkable, cfg.limits.roster_cap)
    if roster.truncated:
        result.warnings.append(f"roster truncated to {cfg.limits.roster_cap} of {roster.total} nodes (limits.roster_cap)")
    known = {d.id for d in linkable} | {d.id for d in drafts}
    system = prompt.system_for(schema)

    for chunk in chunks:
        mine = [d for d in drafts if chunk.cid in d.chunk_ids]
        if not mine:
            continue
        res = call_stage("edges", system, user_text(roster, mine, chunk), schema.edges_output_model, None, cfg=cfg, ledger=ledger, prompt_version=prompt.VERSION)
        for p in res.data.edges:
            edge = Edge(type=p.type, source_id=p.source_id, target_id=p.target_id, relevance=getattr(p, "relevance", None))
            if edge.type not in schema.edges:
                result.dropped.append(Rejection("out_of_schema", f"edge type {edge.type!r} is not in the {schema.name} schema", edge))
                continue
            unknown = [i for i in (edge.source_id, edge.target_id) if i not in known]
            if unknown:
                result.dropped.append(Rejection("dangling", f"unknown id(s) {', '.join(repr(u) for u in unknown)} in {edge.type} {edge.source_id} -> {edge.target_id}", edge))
                continue
            if edge.source_id == edge.target_id:
                result.dropped.append(Rejection("self_edge", f"{edge.type} from {edge.source_id} to itself", edge))
                continue
            result.edges.append(edge)
    return result


__all__ = ["EdgesResult", "Roster", "RosterEntry", "build_roster", "propose", "user_text"]
