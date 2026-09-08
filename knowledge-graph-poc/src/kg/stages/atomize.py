"""S3 atomize + S3b grounding (DESIGN §6, §8.0; D8; R2, R4, R11).

One model call per chunk. Every returned ``unit_id`` and every ``quote.unit_id`` is
checked against the chunk's real unit set (source text can spoof ``<<...>>`` markers,
so nothing model-authored is trusted as a locator); locators are derived from the
cited unit, never from the model. A quote grounds when its whitespace-normalised text
is a substring of the cited unit's whitespace-normalised text; a node with no grounded
quote is flagged ``review: quote_not_found`` — a review flag, never a repair trigger.
A candidate none of whose ``unit_ids`` exist is dropped (``DroppedCandidate``) rather
than written without a locator (R3); the definition is collapsed to one line of prose
so it can never carry a forged ``## Source`` section (R4, R11).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from kg.chunk import render_chunk
from kg.config import Config
from kg.notes import _HTML_COMMENT, NodeSource, safe_prose
from kg.prompts import atomize as prompt
from kg.schemas import AtomizeOutput, Quote

QUOTE_NOT_FOUND = "quote_not_found"
NO_VALID_UNIT = "no valid unit_ids"
#: Titles and aliases are names, not prose; anything longer is truncated (model output is untrusted).
MAX_NAME_LEN = 200
CallStage = Callable[..., Any]
# \t \n \v \f \r are whitespace and are collapsed by ``split``; everything else in C0 (and DEL) is dropped.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0e-\x1f\x7f]")


@dataclass
class Candidate:
    title: str
    definition: str
    aliases: list[str]
    unit_ids: list[str]
    quotes: list[Quote]
    source: str
    chunk_id: str
    review: list[str]
    sources: list[NodeSource]
    #: Provenance of the call that produced this candidate (filled by the stage).
    provider: str | None = field(default=None, compare=False)
    model: str | None = field(default=None, compare=False)
    prompt_version: str | None = field(default=None, compare=False)


@dataclass(frozen=True)
class DroppedCandidate:
    """A wire node that could not become a Candidate; listed in the grounding review queue."""

    title: str
    source: str
    chunk_id: str
    reason: str


def ws(text: str) -> str:
    """Whitespace-normalised, control characters removed."""
    return " ".join(_CONTROL_CHARS.sub("", text or "").split())


def _name(text: str) -> str:
    """A title or alias: HTML comments removed and `<`/`>` dropped (a name becomes an H1, a frontmatter
    value and a dedup-prompt line, where a backslash escape would show), then whitespace-normalised and capped."""
    return ws(_HTML_COMMENT.sub(" ", text or "").replace("<", "").replace(">", ""))[:MAX_NAME_LEN].rstrip()


def _clean_list(items: list[str]) -> list[str]:
    return [x for x in dict.fromkeys(_name(i) for i in items) if x]


def ground(node: Any, chunk: Any) -> Candidate | None:
    """Apply the §8.0 S3b rules to one wire ``CandidateNode`` against its chunk.

    Returns ``None`` when no cited ``unit_id`` exists in the chunk: there is nothing to
    derive a locator from, and a node is never written without one (R3).
    """
    units = {u.uid: u for u in chunk.units}
    source = chunk.source or (chunk.units[0].source if chunk.units else "")
    unit_ids = [uid for uid in dict.fromkeys(node.unit_ids) if uid in units]
    if not unit_ids:
        return None
    quotes: list[Quote] = []
    sources: list[NodeSource] = []
    for q in node.quotes:
        if q.unit_id not in unit_ids:
            continue
        text = ws(q.text)
        if not text or text not in ws(units[q.unit_id].text):
            continue
        quotes.append(Quote(unit_id=q.unit_id, text=text))
        sources.append(NodeSource(locator=units[q.unit_id].locator, file=source, quote=text))
    review: list[str] = []
    if not quotes:
        review.append(QUOTE_NOT_FOUND)
        sources = [NodeSource(locator=units[uid].locator, file=source, quote="") for uid in unit_ids]
    return Candidate(
        title=_name(node.title),
        definition=safe_prose(node.definition),
        aliases=_clean_list(node.aliases),
        unit_ids=unit_ids,
        quotes=quotes,
        source=source,
        chunk_id=chunk.cid,
        review=review,
        sources=sources,
    )


def atomize_chunk(chunk: Any, cfg: Config, *, call_stage: CallStage, ledger: Any = None, dropped: list[DroppedCandidate] | None = None) -> list[Candidate]:
    """One call for `chunk`; exceptions from the boundary propagate so the pipeline can defer the file.

    Wire nodes with no valid ``unit_ids`` are appended to `dropped` (when given) instead of returned.
    """
    result = call_stage("atomize", prompt.SYSTEM, render_chunk(chunk), AtomizeOutput, None, cfg=cfg, ledger=ledger, prompt_version=prompt.VERSION)
    data = result.data
    if not isinstance(data, AtomizeOutput):
        raise TypeError(f"atomize: call_stage returned {type(data).__name__}, expected AtomizeOutput")
    nodes = data.nodes[: cfg.limits.max_nodes_per_chunk]
    out: list[Candidate] = []
    for node in nodes:
        cand = ground(node, chunk)
        if cand is None:
            if dropped is not None:
                dropped.append(DroppedCandidate(title=_name(node.title), source=chunk.source or "", chunk_id=chunk.cid, reason=NO_VALID_UNIT))
            continue
        cand.provider = getattr(result, "provider", None)
        cand.model = getattr(result, "model_served", None) or getattr(result, "model_requested", None)
        cand.prompt_version = getattr(result, "prompt_version", None) or prompt.VERSION
        out.append(cand)
    return out


def atomize(chunks: list[Any], cfg: Config, *, call_stage: CallStage, ledger: Any = None, dropped: list[DroppedCandidate] | None = None) -> list[Candidate]:
    out: list[Candidate] = []
    for chunk in chunks:
        out.extend(atomize_chunk(chunk, cfg, call_stage=call_stage, ledger=ledger, dropped=dropped))
    return out


__all__ = ["Candidate", "DroppedCandidate", "MAX_NAME_LEN", "NO_VALID_UNIT", "QUOTE_NOT_FOUND", "atomize", "atomize_chunk", "ground", "ws"]
