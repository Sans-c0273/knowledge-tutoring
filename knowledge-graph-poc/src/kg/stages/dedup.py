"""S6 dedup (DESIGN §6, §8.0, §11; D10, D12, D29; R12).

Candidate pairs are drafts whose titles/aliases score ≥ ``dedup.threshold`` under
``kg.normalise.similarity`` (exact normalised-title matches are consolidate's job and
are suppressed). At most ``limits.max_dedup_pairs`` are adjudicated, batched
``dedup.pairs_per_call`` per model call; the rest is ``overflow``. Verdicts are matched
order-insensitively; a requested pair without a judgement is ``unsure``; a judgement
for an unrequested pair is ignored and counted. ``same`` becomes a ``same_as`` edge
only when ``write_same_as`` is set (CLI flag or config); otherwise it is queued in
``_review/duplicates.md``. No merge, no delete.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kg.config import Config
from kg.convert._common import write_files_atomically
from kg.gates import Edge
from kg.normalise import norm_title, similarity
from kg.notes import safe_prose
from kg.prompts import dedup as prompt
from kg.report import cell
from kg.schemas import DedupOutput

CallStage = Callable[..., Any]
QUEUE_FILE = "duplicates.md"
NO_JUDGEMENT = "no judgement returned for this pair"


@dataclass(frozen=True)
class Pair:
    a_id: str
    b_id: str
    score: float


@dataclass(frozen=True)
class Judged:
    a_id: str
    b_id: str
    score: float
    verdict: str
    reason: str


@dataclass
class DedupResult:
    same: list[Judged] = field(default_factory=list)
    unsure: list[Judged] = field(default_factory=list)
    different: list[Judged] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    unrequested: int = 0
    overflow: list[Pair] = field(default_factory=list)
    titles: dict[str, str] = field(default_factory=dict, compare=False)

    def open_same(self) -> list[Judged]:
        """`same` verdicts that were not written as edges (still open for review)."""
        written = {(e.source_id, e.target_id) for e in self.edges}
        return [j for j in self.same if (j.a_id, j.b_id) not in written]


def _names(draft: Any) -> list[str]:
    return [draft.title, *getattr(draft, "aliases", ())]


def candidate_pairs(drafts: list[Any], threshold: float, *, max_pairs: int | None = None, focus: set[str] | None = None) -> list[Pair]:
    """Similar pairs, best first. With `focus`, only pairs touching at least one focused id are kept
    (the pipeline focuses on this run's drafts so existing<->existing pairs are not re-adjudicated every run)."""
    pairs: list[Pair] = []
    for i, a in enumerate(drafts):
        for b in drafts[i + 1 :]:
            if a.id == b.id or norm_title(a.title) == norm_title(b.title):
                continue
            if focus is not None and a.id not in focus and b.id not in focus:
                continue
            score = max(similarity(x, y) for x in _names(a) for y in _names(b))
            if score >= threshold:
                lo, hi = sorted((a.id, b.id))
                pairs.append(Pair(lo, hi, score))
    pairs.sort(key=lambda p: (-p.score, p.a_id, p.b_id))
    return pairs[:max_pairs] if max_pairs is not None else pairs


def _pair_text(n: int, pair: Pair, by_id: dict[str, Any]) -> str:
    def block(node_id: str) -> str:
        d = by_id[node_id]
        aliases = f" (aliases: {', '.join(d.aliases)})" if getattr(d, "aliases", None) else ""
        # Existing notes may predate `safe_prose` or be hand-edited: their definitions are made safe here too.
        return f"  {node_id}: {d.title}{aliases}\n    {safe_prose(d.definition)}"

    return f"{n}. pair {pair.a_id} / {pair.b_id}\n{block(pair.a_id)}\n{block(pair.b_id)}"


def run(drafts: list[Any], cfg: Config, *, call_stage: CallStage, ledger: Any = None, write_same_as: bool | None = None, focus: set[str] | None = None) -> DedupResult:
    write = cfg.dedup.write_same_as if write_same_as is None else bool(write_same_as)
    by_id = {d.id: d for d in drafts}
    result = DedupResult(titles={d.id: d.title for d in drafts})
    pairs = candidate_pairs(drafts, cfg.dedup.threshold, focus=focus)
    adjudicate, result.overflow = pairs[: cfg.limits.max_dedup_pairs], pairs[cfg.limits.max_dedup_pairs :]
    if not adjudicate:
        return result

    verdicts: dict[tuple[str, str], tuple[str, str]] = {}
    size = cfg.dedup.pairs_per_call
    for start in range(0, len(adjudicate), size):
        batch = adjudicate[start : start + size]
        text = "Pairs to judge:\n\n" + "\n\n".join(_pair_text(i + 1, p, by_id) for i, p in enumerate(batch))
        res = call_stage("dedup", prompt.SYSTEM, text, DedupOutput, None, cfg=cfg, ledger=ledger, prompt_version=prompt.VERSION)
        requested = {(p.a_id, p.b_id) for p in batch}
        for j in res.data.judgements:
            key = (min(j.a_id, j.b_id), max(j.a_id, j.b_id))
            if key not in requested or key in verdicts:
                result.unrequested += 1
                continue
            verdicts[key] = (j.verdict, j.reason)

    for p in adjudicate:
        verdict, reason = verdicts.get((p.a_id, p.b_id), ("unsure", NO_JUDGEMENT))
        judged = Judged(p.a_id, p.b_id, p.score, verdict, reason)
        if verdict == "same":
            result.same.append(judged)
            if write:
                result.edges.append(Edge("same_as", p.a_id, p.b_id))
        elif verdict == "different":
            result.different.append(judged)
        else:
            result.unsure.append(judged)
    return result


def render_queue(result: DedupResult) -> str:
    def title(node_id: str) -> str:
        return result.titles.get(node_id, "")

    def row(j: Judged) -> str:
        return f"| {cell(j.a_id)} | {cell(title(j.a_id))} | {cell(j.b_id)} | {cell(title(j.b_id))} | {j.score:.2f} | {cell(j.verdict)} | {cell(j.reason)} |"

    header = "| a | title | b | title | score | verdict | reason |\n|---|---|---|---|---|---|---|"
    lines = ["# Duplicate review queue", "", "Pairs the model judged `same` (not written as edges) or `unsure`. Nothing has been merged or deleted.", ""]
    open_rows = [row(j) for j in (*result.open_same(), *result.unsure)]
    if open_rows:
        lines += ["## Open", "", header, *open_rows, ""]
    else:
        lines += ["## Open", "", "(none)", ""]
    if result.edges:
        lines += [f"{len(result.edges)} pair(s) judged `same` were written as same_as edges and are not open items (see the run report).", ""]
    if result.overflow:
        lines += ["## Not adjudicated (over limits.max_dedup_pairs)", "", "| a | title | b | title | score |", "|---|---|---|---|---|"]
        lines += [f"| {cell(p.a_id)} | {cell(title(p.a_id))} | {cell(p.b_id)} | {cell(title(p.b_id))} | {p.score:.2f} |" for p in result.overflow]
        lines.append("")
    return "\n".join(lines)


def write_queue(result: DedupResult, review_dir: Path, *, name: str = QUEUE_FILE) -> Path:
    """Write the queue atomically (`.tmp` + replace, O_NOFOLLOW): a planted symlink is replaced, never written through."""
    review_dir = Path(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    path = review_dir / name
    write_files_atomically({path: render_queue(result)})
    return path


__all__ = ["DedupResult", "Judged", "NO_JUDGEMENT", "Pair", "QUEUE_FILE", "candidate_pairs", "render_queue", "run", "write_queue"]
