"""Relevance judgment sampler (PRD R17, §8(c); DESIGN §16).

`sample()` draws a stratified sample of `related_to` edges — `n_per_band` from each
relevance band, every symmetric edge at most once — and writes a review file
``<graph>/_review/relevance-sample-<run_id>.md`` whose table has a blank `verdict`
column for the human (`agree` / `disagree`). Both definitions are listed under the
table for context. `summarise()` reads a filled file back and reports agreement per
band plus the §8(c) verdict:

  monotonic         rates non-decreasing from the low band to the high band
  not monotonic     some higher band agrees less often than a lower one
  indistinguishable every band lies within INDISTINGUISHABLE_TOLERANCE of the others —
                    the score carries no information, whatever the ordering
  insufficient data some CONFIGURED band has fewer than MIN_JUDGED_PER_BAND judged rows
                    (a band with no rows at all counts as zero — it never vanishes)

The verdict runs over the bands written into the review-file header (`- bands: …`),
not over whatever bands happen to appear in the rows, so an emptied band cannot make
`monotonic` pass on the remainder. Files without that header line fall back to the
bands present in the rows. Verdict cells match on their first word (`Agree.`,
`agree - obviously` count); a column-aligned table (Obsidian "format table") parses;
rows with the wrong number of cells are re-joined into `notes` and counted as
`malformed`. A non-blank cell that is neither word is `unfilled` (excluded from
rates) and additionally counted as `unrecognised`, so a typo is visible rather
than silently indistinguishable from a blank.

No model call, no provider adapter; only the graph folder is read and written.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kg import graph_io
from kg.notes import strip_wikilinks
from kg.paths import SandboxViolation, is_denied

DEFAULT_BANDS: tuple[tuple[int, int], ...] = ((0, 39), (40, 70), (71, 100))
INDISTINGUISHABLE_TOLERANCE = 0.10
#: Fewer judged (agree/disagree) rows than this in any configured band → `insufficient data`.
MIN_JUDGED_PER_BAND = 5
BANDS_HEADER_PREFIX = "- bands:"
REVIEW_DIR = "_review"
REPORTS_DIR = "_reports"
AGREE = "agree"
DISAGREE = "disagree"
VERDICT_MONOTONIC = "monotonic"
VERDICT_NOT_MONOTONIC = "not monotonic"
VERDICT_INDISTINGUISHABLE = "indistinguishable"
VERDICT_INSUFFICIENT = "insufficient data"
TABLE_COLUMNS: tuple[str, ...] = ("edge", "source", "target", "relevance", "band", "verdict", "notes")

Band = tuple[int, int]


# -------------------------------------------------------------------- bands


def band_label(band: Band) -> str:
    lo, hi = band
    return f"{lo}-{hi}"


def validate_bands(bands: Any) -> tuple[Band, ...]:
    """Bands must be integer [lo, hi] pairs that are contiguous and cover 0–100 exactly."""
    try:
        parsed = tuple((int(lo), int(hi)) for lo, hi in bands)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"relevance bands must be [low, high] integer pairs: {bands!r}") from exc
    if not parsed:
        raise ValueError("at least one relevance band is required")
    ordered = sorted(parsed)
    if ordered[0][0] != 0 or ordered[-1][1] != 100:
        raise ValueError(f"relevance bands must cover 0-100: {parsed!r}")
    for (lo, hi), (nlo, _nhi) in zip(ordered, ordered[1:] + [(101, 101)], strict=True):
        if lo > hi:
            raise ValueError(f"relevance band {lo}-{hi} is inverted")
        if nlo != hi + 1:
            raise ValueError(f"relevance bands must be contiguous without overlap: {lo}-{hi} is followed by {nlo}")
    return tuple(ordered)


def band_for(relevance: int, bands: tuple[Band, ...]) -> str:
    for band in bands:
        if band[0] <= relevance <= band[1]:
            return band_label(band)
    raise ValueError(f"relevance {relevance} lies outside every band")


# ------------------------------------------------------------------- sample


@dataclass(frozen=True)
class SampleRow:
    edge_id: str
    source_id: str
    source_title: str
    target_id: str
    target_title: str
    relevance: int
    band: str
    source_definition: str
    target_definition: str


@dataclass
class SampleResult:
    rows: list[SampleRow]
    path: Path
    shortfall: dict[str, int]
    bands: tuple[Band, ...]


def _run_id(now: datetime) -> str:
    return now.astimezone(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


def _cell(text: Any) -> str:
    """One Markdown table cell: no pipes, no line breaks."""
    return " ".join(str(text if text is not None else "").replace("|", "/").split())


def _edge_stage_model(graph_dir: Path, graph: graph_io.Graph) -> str | None:
    """The edges-stage model from the newest run report if one exists, else the notes' majority model."""
    reports = graph_dir / REPORTS_DIR
    if reports.is_dir():
        for path in sorted(reports.glob("run-*.json"), reverse=True):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            model = ((doc.get("stages") or {}).get("edges") or {}).get("model_requested")
            if model:
                return str(model)
    models = Counter(n.model for n in graph.nodes.values() if n.model)
    return models.most_common(1)[0][0] if models else None


def sample(graph_dir: Path, *, n_per_band: int, bands: Any = DEFAULT_BANDS, seed: int, now: datetime) -> SampleResult:
    if n_per_band < 1:
        raise ValueError(f"n_per_band must be a positive integer (got {n_per_band})")
    bands_t = validate_bands(bands)
    labels = [band_label(b) for b in bands_t]
    graph_dir = Path(graph_dir)
    graph = graph_io.load_graph(graph_dir)  # refuses a forbidden graph_dir

    pool: dict[str, list[SampleRow]] = {label: [] for label in labels}
    for e in graph.edges:
        if e.type != "related_to" or e.relevance is None:
            continue
        s, t = graph.nodes[e.source], graph.nodes[e.target]
        a, b = (s, t) if s.id <= t.id else (t, s)
        pool[band_for(e.relevance, bands_t)].append(
            SampleRow(
                edge_id=f"{a.id}~{b.id}",
                source_id=a.id,
                source_title=a.title,
                target_id=b.id,
                target_title=b.title,
                relevance=e.relevance,
                band=band_for(e.relevance, bands_t),
                source_definition=strip_wikilinks(a.definition_plain),
                target_definition=strip_wikilinks(b.definition_plain),
            )
        )

    rng = random.Random(seed)
    rows: list[SampleRow] = []
    shortfall: dict[str, int] = {}
    pool_sizes: dict[str, int] = {}
    for label in labels:
        candidates = sorted(pool[label], key=lambda r: r.edge_id)
        pool_sizes[label] = len(candidates)
        k = min(n_per_band, len(candidates))
        picked = rng.sample(candidates, k) if k else []
        rows.extend(sorted(picked, key=lambda r: r.edge_id))
        shortfall[label] = n_per_band - k

    run_id = _run_id(now)
    path = graph_dir / REVIEW_DIR / f"relevance-sample-{run_id}.md"
    if is_denied(path):
        raise SandboxViolation(f"review file would land in a forbidden location: {path}")
    # `_review` could be a symlink planted inside the graph folder; the resolved file must stay under graph_dir.
    # Checked before mkdir so nothing is ever created behind the link.
    if not path.resolve().is_relative_to(graph_dir.resolve()):
        raise SandboxViolation(f"review file would resolve outside the graph folder: {path} -> {path.resolve()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _review_text(rows, graph, graph_dir, run_id=run_id, now=now, seed=seed, n_per_band=n_per_band, labels=labels, shortfall=shortfall, pool_sizes=pool_sizes),
        encoding="utf-8",
    )
    return SampleResult(rows=rows, path=path, shortfall=shortfall, bands=bands_t)


def _review_text(
    rows: list[SampleRow],
    graph: graph_io.Graph,
    graph_dir: Path,
    *,
    run_id: str,
    now: datetime,
    seed: int,
    n_per_band: int,
    labels: list[str],
    shortfall: dict[str, int],
    pool_sizes: dict[str, int],
) -> str:
    meta = graph.meta
    edge_model = _edge_stage_model(graph_dir, graph)
    lines: list[str] = [
        f"# Relevance judgment sample — {run_id}",
        "",
        f"- corpus: {meta.get('corpus') or '?'} · schema: {meta.get('schema') or '?'}",
        f"- provider: {meta.get('provider') or '?'} · edge-stage model: {edge_model or '?'}",
        f"- sampled: {now.astimezone(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')} · seed: {seed} · n per band: {n_per_band}",
        f"{BANDS_HEADER_PREFIX} {', '.join(labels)}",
        f"- edges sampled: {len(rows)} of {sum(pool_sizes.values())} related_to edge(s)",
        "",
        "## How to fill in",
        "",
        f"Read both definitions, then write `{AGREE}` or `{DISAGREE}` in the **verdict** column of each row:",
        f"`{AGREE}` = the relevance score is a fair reflection of how related the two concepts are;",
        f"`{DISAGREE}` = it is not. Leave the cell blank if you cannot judge (counted as unfilled, excluded from rates).",
        "Anything in **notes** is kept for the record. Then run `kg summarise-relevance <this file>`.",
        "",
        "## Coverage and shortfall",
        "",
        "| band | pool | requested | sampled | shortfall |",
        "|---|---|---|---|---|",
    ]
    for label in labels:
        got = n_per_band - shortfall[label]
        lines.append(f"| {label} | {pool_sizes[label]} | {n_per_band} | {got} | {shortfall[label]} |")
    lines += [
        "",
        "## Sample",
        "",
        "| " + " | ".join(TABLE_COLUMNS) + " |",
        "|" + "---|" * len(TABLE_COLUMNS),
    ]
    for r in rows:
        lines.append(
            f"| {_cell(r.edge_id)} | {_cell(f'{r.source_id} {r.source_title}')} | {_cell(f'{r.target_id} {r.target_title}')} | {r.relevance} | {r.band} |  |  |"
        )
    if not rows:
        lines.append("| (no related_to edges in this graph) | — | — | — | — |  |  |")
    lines += ["", "## Definitions", ""]
    if not rows:
        lines.append("(none)")
    for r in rows:
        lines += [
            f"### {r.edge_id} — relevance {r.relevance} ({r.band})",
            "",
            f"- **{r.source_id} — {r.source_title}**: {r.source_definition}",
            f"- **{r.target_id} — {r.target_title}**: {r.target_definition}",
            "",
        ]
    return "\n".join(lines).rstrip("\n") + "\n"


# ---------------------------------------------------------------- summarise


@dataclass
class BandStat:
    agree: int = 0
    disagree: int = 0
    #: Rows not judged: blank cells plus unrecognised text (the latter also counted in `unrecognised`).
    unfilled: int = 0
    #: Non-blank cells whose first word is neither `agree` nor `disagree` (a subset of `unfilled`).
    unrecognised: int = 0

    @property
    def rate(self) -> float | None:
        judged = self.agree + self.disagree
        return self.agree / judged if judged else None


@dataclass
class Summary:
    bands: dict[str, BandStat]
    monotonic: bool
    verdict: str
    filled: int
    unfilled: int
    malformed: int = 0
    #: Non-blank verdict cells that were not `agree`/`disagree` (already inside `unfilled`; e.g. a typo such as `agre`).
    unrecognised: int = 0


def _band_sort_key(label: str) -> tuple[int, str]:
    head = label.split("-", 1)[0].strip()
    return (int(head) if head.lstrip("-").isdigit() else 10**9, label)


def _split_cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _parse_table(text: str) -> tuple[list[dict[str, str]], int]:
    """(rows, malformed). Header match is on stripped cell names, so a column-aligned table parses."""
    lines = text.splitlines()
    header_idx = next((i for i, line in enumerate(lines) if line.lstrip().startswith("|") and "verdict" in _split_cells(line)), None)
    if header_idx is None:
        raise ValueError("no review table found (expected a header row with a `verdict` column)")
    cols = _split_cells(lines[header_idx])
    for needed in ("edge", "relevance", "band", "verdict"):
        if needed not in cols:
            raise ValueError(f"review table lacks the `{needed}` column")
    rows: list[dict[str, str]] = []
    malformed = 0
    for line in lines[header_idx + 2 :]:
        if not line.lstrip().startswith("|"):
            break
        cells = _split_cells(line)
        if len(cells) != len(cols):
            # A stray pipe or a dropped cell: keep the leading columns, fold the remainder into the last one.
            malformed += 1
            head, tail = cells[: len(cols) - 1], cells[len(cols) - 1 :]
            cells = [*head, *[""] * (len(cols) - 1 - len(head)), " ".join(c for c in tail if c)]
        row = dict(zip(cols, cells, strict=True))
        if not row["relevance"].lstrip("-").isdigit():
            continue  # placeholder row of an empty sample
        rows.append(row)
    return rows, malformed


def _configured_bands(text: str) -> list[str] | None:
    """Band labels from the `- bands:` header line the sampler wrote; None if the file has none."""
    for line in text.splitlines():
        if line.strip().casefold().startswith(BANDS_HEADER_PREFIX):
            labels = [b.strip() for b in line.split(":", 1)[1].split(",") if b.strip()]
            return labels or None
    return None


def classify_verdict(cell: str) -> str | None:
    """`agree` / `disagree` from the first word of the cell (case-insensitive, trailing punctuation ignored), else None."""
    tokens = cell.split()
    if not tokens:
        return None
    word = tokens[0].casefold().rstrip(".,;:!?-–—")
    return word if word in (AGREE, DISAGREE) else None


def summarise(path: Path) -> Summary:
    path = Path(path)
    text = path.read_text(encoding="utf-8")  # FileNotFoundError / OSError propagate
    rows, malformed = _parse_table(text)
    configured = _configured_bands(text)
    stats: dict[str, BandStat] = {label: BandStat() for label in configured or []}
    for row in rows:
        stat = stats.setdefault(row["band"], BandStat())
        verdict = classify_verdict(row["verdict"])
        if verdict == AGREE:
            stat.agree += 1
        elif verdict == DISAGREE:
            stat.disagree += 1
        else:
            stat.unfilled += 1
            if row["verdict"].strip():
                stat.unrecognised += 1
    bands = {label: stats[label] for label in sorted(stats, key=_band_sort_key)}
    filled = sum(s.agree + s.disagree for s in bands.values())
    unfilled = sum(s.unfilled for s in bands.values())
    unrecognised = sum(s.unrecognised for s in bands.values())

    rates = [s.rate for s in bands.values()]
    if not rates or any(r is None for r in rates) or any(s.agree + s.disagree < MIN_JUDGED_PER_BAND for s in bands.values()):
        verdict_word = VERDICT_INSUFFICIENT
    elif max(rates) - min(rates) <= INDISTINGUISHABLE_TOLERANCE + 1e-9:  # type: ignore[type-var]
        verdict_word = VERDICT_INDISTINGUISHABLE
    elif all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1)):  # type: ignore[operator]
        verdict_word = VERDICT_MONOTONIC
    else:
        verdict_word = VERDICT_NOT_MONOTONIC
    return Summary(
        bands=bands,
        monotonic=verdict_word == VERDICT_MONOTONIC,
        verdict=verdict_word,
        filled=filled,
        unfilled=unfilled,
        malformed=malformed,
        unrecognised=unrecognised,
    )


__all__ = [
    "AGREE",
    "BANDS_HEADER_PREFIX",
    "BandStat",
    "DEFAULT_BANDS",
    "DISAGREE",
    "INDISTINGUISHABLE_TOLERANCE",
    "MIN_JUDGED_PER_BAND",
    "REVIEW_DIR",
    "SampleResult",
    "SampleRow",
    "Summary",
    "TABLE_COLUMNS",
    "VERDICT_INDISTINGUISHABLE",
    "VERDICT_INSUFFICIENT",
    "VERDICT_MONOTONIC",
    "VERDICT_NOT_MONOTONIC",
    "band_for",
    "band_label",
    "classify_verdict",
    "sample",
    "summarise",
    "validate_bands",
]
