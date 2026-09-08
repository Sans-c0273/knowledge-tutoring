"""`kg stability`: two clean ingests of the same inbox, node-set agreement (PRD R16, §8(b); DESIGN §15; D11).

Legs `a` and `b` live under ``<runs>/stability-<run_id>/{a,b}/`` with their own
inbox / converted / graph / processed / runs folders. The inbox is COPIED into each
leg (originals never move) and `kg.pipeline.run` is executed twice, sequentially,
with a `Config` cloned from the one parsed `kg.yaml` and differing only in paths —
so both legs share `llm.provider` and the per-stage model map by construction.

Scores (all on `norm_title`): headline Jaccard, per-run overlaps, fuzzy Jaccard at
similarity ≥ 0.80, `only_in_a` / `only_in_b` with locators, and the |Δ relevance|
histogram of `related_to` edges matched by their endpoint titles (observed, never
gated). `passed` is the Jaccard against `threshold`; `valid` is the equivalence
guard — a `deferred` (model-stage) file in either leg, a `failed` (conversion-stage,
deterministic) file set that differs between legs, a provider or requested-model
mismatch, or an empty leg marks the comparison INVALID for §8(b) while the scores
are still printed. The same file failing conversion in BOTH legs is a warning, not
an invalidation: it was excluded identically on both sides. A served-model mismatch
invalidates too unless `allow_served_drift` (then a warning; the drift is recorded).
The run folder must not pre-exist — two runs started in the same second collide
on `run_id`, and the second refuses rather than overwriting. Leg folders are removed
afterwards unless `keep`; the two leg run reports are copied next to the stability
report first so the evidence survives.
"""

from __future__ import annotations

import json
import shutil
import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from kg import graph_io, pipeline
from kg.config import STAGES, Config, PathsConfig
from kg.normalise import norm_title, similarity
from kg.paths import KINDS, Sandbox

#: PRD §8(b): node-set agreement ≥ 0.70. No `stability:` key exists in kg.yaml (§3.2).
STABILITY_THRESHOLD = 0.70
#: DESIGN §15: fuzzy Jaccard counts title pairs at similarity ≥ 0.80 as matched.
FUZZY_SIMILARITY = 0.80
LEGS: tuple[str, ...] = ("a", "b")
REPORT_MD = "stability-report.md"
REPORT_JSON = "stability-report.json"
DELTA_BUCKETS: tuple[tuple[str, int, int], ...] = (("0-5", 0, 5), ("6-10", 6, 10), ("11-20", 11, 20), ("21-40", 21, 40), ("41-100", 41, 100))

CallStage = Callable[..., Any]


# ------------------------------------------------------------------ results


@dataclass
class RelevanceDeltas:
    n: int
    histogram: dict[str, int]
    mean_abs: float | None
    median_abs: float | None

    def to_dict(self) -> dict[str, Any]:
        return {"n": self.n, "histogram": dict(self.histogram), "mean_abs": self.mean_abs, "median_abs": self.median_abs}


@dataclass
class LegInfo:
    provider: str
    models: dict[str, dict[str, str | None]]
    nodes: int
    deferred: list[str]
    report_json: Path
    report_md: Path | None = None
    failed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "models": {s: dict(m) for s, m in self.models.items()},
            "nodes": self.nodes,
            "deferred": list(self.deferred),
            "failed": list(self.failed),
            "report_json": str(self.report_json),
            "report_md": str(self.report_md) if self.report_md else None,
        }


@dataclass
class NodeMatch:
    common: list[str]
    only_in_a: list[str]
    only_in_b: list[str]


@dataclass
class StabilityResult:
    run_id: str
    root: Path
    jaccard: float
    overlap_a: float
    overlap_b: float
    fuzzy_jaccard: float
    passed: bool
    threshold: float
    valid: bool
    invalid_reasons: list[str]
    only_in_a: list[dict[str, Any]]
    only_in_b: list[dict[str, Any]]
    common: list[str]
    relevance_deltas: RelevanceDeltas
    legs: dict[str, LegInfo]
    report_md: Path
    report_json: Path
    warnings: list[str] = field(default_factory=list)
    served_drift: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "root": str(self.root),
            "threshold": self.threshold,
            "jaccard": self.jaccard,
            "overlap_a": self.overlap_a,
            "overlap_b": self.overlap_b,
            "fuzzy_jaccard": self.fuzzy_jaccard,
            "passed": self.passed,
            "valid": self.valid,
            "invalid_reasons": list(self.invalid_reasons),
            "counts": {"a": self.legs["a"].nodes, "b": self.legs["b"].nodes, "common": len(self.common)},
            "common": list(self.common),
            "only_in_a": [dict(d) for d in self.only_in_a],
            "only_in_b": [dict(d) for d in self.only_in_b],
            "relevance_deltas": self.relevance_deltas.to_dict(),
            "legs": {leg: info.to_dict() for leg, info in self.legs.items()},
            "warnings": list(self.warnings),
            "served_drift": list(self.served_drift),
            "report_md": str(self.report_md),
            "report_json": str(self.report_json),
        }


# --------------------------------------------------------------- pure maths


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """|A∩B| / |A∪B|; two empty sets agree perfectly (the empty-leg case is guarded by `valid`)."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def overlaps(a: Iterable[str], b: Iterable[str]) -> tuple[float, float]:
    """(|A∩B|/|A|, |A∩B|/|B|); an empty side scores 0."""
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    return (inter / len(sa) if sa else 0.0, inter / len(sb) if sb else 0.0)


def _norm_set(titles: Iterable[str]) -> set[str]:
    return {t for t in (norm_title(x) for x in titles) if t}


def match_nodes(a_titles: Iterable[str], b_titles: Iterable[str]) -> NodeMatch:
    """Exact match on normalised titles; every list sorted."""
    na, nb = _norm_set(a_titles), _norm_set(b_titles)
    return NodeMatch(common=sorted(na & nb), only_in_a=sorted(na - nb), only_in_b=sorted(nb - na))


def fuzzy_jaccard(a_titles: Iterable[str], b_titles: Iterable[str], *, min_similarity: float = FUZZY_SIMILARITY) -> float:
    """Jaccard where the unmatched remainder is paired greedily by `similarity` ≥ `min_similarity`.

    Exact matches are taken first, so the result is never below the exact Jaccard.
    """
    na, nb = _norm_set(a_titles), _norm_set(b_titles)
    if not na and not nb:
        return 1.0
    matched = len(na & nb)
    rest_a, rest_b = sorted(na - nb), sorted(nb - na)
    scored = ((similarity(x, y), x, y) for x in rest_a for y in rest_b)  # one similarity call per pair
    candidates = sorted((c for c in scored if c[0] >= min_similarity), key=lambda t: (-t[0], t[1], t[2]))
    used_a: set[str] = set()
    used_b: set[str] = set()
    for _score, x, y in candidates:
        if x in used_a or y in used_b:
            continue
        used_a.add(x)
        used_b.add(y)
        matched += 1
    return matched / (len(na) + len(nb) - matched)


def relevance_delta_histogram(deltas: Iterable[int | float]) -> dict[str, int]:
    """|Δ| buckets 0–5, 6–10, 11–20, 21–40, 41–100 (DESIGN §15); anything above 100 lands in the last bucket."""
    out = {label: 0 for label, _lo, _hi in DELTA_BUCKETS}
    for d in deltas:
        mag = abs(d)
        for label, lo, hi in DELTA_BUCKETS:
            if lo <= mag <= hi:
                out[label] += 1
                break
        else:
            out[DELTA_BUCKETS[-1][0]] += 1
    return out


def _relevance_deltas(deltas: list[int]) -> RelevanceDeltas:
    mags = [abs(d) for d in deltas]
    return RelevanceDeltas(
        n=len(deltas),
        histogram=relevance_delta_histogram(deltas),
        mean_abs=statistics.mean(mags) if mags else None,
        median_abs=statistics.median(mags) if mags else None,
    )


# -------------------------------------------------------------- leg configs


def leg_root_rel(cfg: Config, run_id: str, leg: str) -> str:
    """Project-relative folder of one leg: ``<paths.runs>/stability-<run_id>/<leg>``."""
    if leg not in LEGS:
        raise ValueError(f"unknown stability leg {leg!r}; expected one of {', '.join(LEGS)}")
    return str(PurePosixPath(cfg.paths.runs) / f"stability-{run_id}" / leg)


def leg_config(cfg: Config, run_id: str, leg: str) -> Config:
    """`cfg` with every folder moved under the leg root; provider, models and every other section untouched."""
    base = leg_root_rel(cfg, run_id, leg)
    roots = {kind: str(PurePosixPath(base) / kind) for kind in KINDS}
    sandbox = Sandbox(project_root=cfg.project_root, roots=roots)
    return replace(cfg, paths=PathsConfig(**roots), sandbox=sandbox)


# ------------------------------------------------------------- orchestration


def _copy_inbox(cfg: Config, leg_cfg: Config, warnings: list[str]) -> int:
    """Copy every regular file of the project inbox (urls.txt included) into the leg inbox. Returns the count."""
    src_root = cfg.sandbox.root("inbox")
    copied = 0
    if not src_root.is_dir():
        return 0
    for path in sorted(src_root.rglob("*")):
        rel = path.relative_to(src_root)
        if path.is_symlink():
            warnings.append(f"inbox/{rel.as_posix()}: symlink not copied into the legs")
            continue
        if not path.is_file():
            continue
        dst = leg_cfg.sandbox.resolve("inbox", rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        copied += 1
    return copied


def _load_leg_graph(leg_cfg: Config) -> graph_io.Graph:
    graph_dir = leg_cfg.sandbox.root("graph")
    return graph_io.load_graph(graph_dir) if graph_dir.is_dir() else graph_io.Graph()


def _leg_info(leg_cfg: Config, res: pipeline.RunResult, graph: graph_io.Graph, *, copy_to: Path, leg: str) -> LegInfo:
    rep = json.loads(Path(res.report_json).read_text(encoding="utf-8"))
    md_copy = copy_to / f"leg-{leg}-report.md"
    json_copy = copy_to / f"leg-{leg}-report.json"
    shutil.copy2(res.report_md, md_copy)
    shutil.copy2(res.report_json, json_copy)
    stages = rep.get("stages", {})
    return LegInfo(
        provider=str(rep.get("provider", leg_cfg.llm.provider)),
        models={
            s: {"requested": (stages.get(s) or {}).get("model_requested") or leg_cfg.llm.model_for(s), "served": (stages.get(s) or {}).get("model_served")}
            for s in STAGES
        },
        nodes=len(graph.nodes),
        deferred=[str(i.get("name")) for i in rep.get("inputs", []) if i.get("status") == "deferred"],
        failed=[str(i.get("name")) for i in rep.get("inputs", []) if i.get("status") == "failed"],
        report_json=json_copy,
        report_md=md_copy,
    )


def _diff_rows(norm_titles: list[str], graph: graph_io.Graph) -> list[dict[str, Any]]:
    by_norm: dict[str, list[graph_io.GraphNode]] = {}
    for n in graph.nodes.values():
        by_norm.setdefault(norm_title(n.title), []).append(n)
    rows: list[dict[str, Any]] = []
    for t in norm_titles:
        for n in sorted(by_norm.get(t, []), key=lambda n: n.id):
            rows.append({"title": t, "id": n.id, "original_title": n.title, "file": n.file, "locators": [s["locator"] for s in n.sources]})
    return rows


def _related_to_by_titles(graph: graph_io.Graph) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for e in graph.edges:
        if e.type != "related_to" or e.relevance is None:
            continue
        key = tuple(sorted((norm_title(graph.nodes[e.source].title), norm_title(graph.nodes[e.target].title))))
        out.setdefault((key[0], key[1]), e.relevance)
    return out


def _validity(legs: dict[str, LegInfo], *, allow_served_drift: bool = False) -> tuple[list[str], list[str], list[str]]:
    """(invalid_reasons, warnings, served_drift_stages) for the equivalence guard.

    `deferred` is a model-stage outcome (rate limit, invalid output) and always breaks
    equivalence. `failed` is a deterministic conversion outcome: the same corrupt file
    fails in both legs and is excluded on both sides, so only a DIFFERENT failed set
    invalidates; an equal non-empty set is a warning naming the files.
    """
    reasons: list[str] = []
    warnings: list[str] = []
    drift: list[str] = []
    a, b = legs["a"], legs["b"]
    for leg, info in legs.items():
        if info.deferred:
            reasons.append(f"leg {leg} deferred file(s): {', '.join(info.deferred)} — legs not equivalent")
    failed_a, failed_b = set(a.failed), set(b.failed)
    if failed_a != failed_b:
        reasons.append(
            f"conversion failed for different files per leg: a = {', '.join(sorted(failed_a)) or 'none'}; b = {', '.join(sorted(failed_b)) or 'none'} — legs not equivalent"
        )
    elif failed_a:
        warnings.append(f"conversion failed identically in both legs for: {', '.join(sorted(failed_a))} (excluded from both; comparison remains valid)")
    for leg, info in legs.items():
        if info.nodes == 0:
            reasons.append(f"leg {leg} produced no nodes (empty inbox or every file deferred/failed)")
    if a.provider != b.provider:
        reasons.append(f"provider differs between legs: {a.provider} vs {b.provider}")
    for stage in STAGES:
        ma, mb = a.models.get(stage, {}), b.models.get(stage, {})
        if ma.get("requested") != mb.get("requested"):
            reasons.append(f"model requested for stage {stage} differs between legs: {ma.get('requested')} vs {mb.get('requested')}")
        if ma.get("served") is not None and mb.get("served") is not None and ma["served"] != mb["served"]:
            drift.append(stage)
            text = f"model served for stage {stage} differs between legs: {ma['served']} vs {mb['served']}"
            (warnings if allow_served_drift else reasons).append(text + (" (allowed by --allow-served-drift)" if allow_served_drift else ""))
    return reasons, warnings, drift


def run(
    cfg: Config,
    *,
    call_stage: CallStage,
    now: datetime,
    keep: bool = False,
    threshold: float = STABILITY_THRESHOLD,
    fetch: Callable[[str], str | None] | None = None,
    allow_served_drift: bool = False,
) -> StabilityResult:
    run_id = pipeline.run_id_for(now)
    root = cfg.sandbox.resolve("runs", f"stability-{run_id}")
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        root.mkdir(exist_ok=False)  # run_id is second-granular: never overwrite a run started in the same second
    except FileExistsError as exc:
        raise FileExistsError(f"stability run folder already exists: {root} — a run with the same run_id started in this second; retry") from exc
    warnings: list[str] = []

    graphs: dict[str, graph_io.Graph] = {}
    legs: dict[str, LegInfo] = {}
    for leg in LEGS:  # a runs to completion before b starts
        leg_cfg = leg_config(cfg, run_id, leg)
        for kind in KINDS:
            leg_cfg.sandbox.root(kind).mkdir(parents=True, exist_ok=True)
        _copy_inbox(cfg, leg_cfg, warnings)
        res = pipeline.run(leg_cfg, call_stage=call_stage, now=now, fetch=fetch)
        graphs[leg] = _load_leg_graph(leg_cfg)
        legs[leg] = _leg_info(leg_cfg, res, graphs[leg], copy_to=root, leg=leg)

    titles_a = [n.title for n in graphs["a"].nodes.values()]
    titles_b = [n.title for n in graphs["b"].nodes.values()]
    match = match_nodes(titles_a, titles_b)
    na, nb = _norm_set(titles_a), _norm_set(titles_b)
    score = jaccard(na, nb)
    overlap_a, overlap_b = overlaps(na, nb)

    rel_a, rel_b = _related_to_by_titles(graphs["a"]), _related_to_by_titles(graphs["b"])
    deltas = [rel_a[k] - rel_b[k] for k in sorted(set(rel_a) & set(rel_b))]

    reasons, guard_warnings, served_drift = _validity(legs, allow_served_drift=allow_served_drift)
    warnings.extend(guard_warnings)
    result = StabilityResult(
        run_id=run_id,
        root=root,
        jaccard=score,
        overlap_a=overlap_a,
        overlap_b=overlap_b,
        fuzzy_jaccard=fuzzy_jaccard(titles_a, titles_b),
        passed=score >= threshold,
        threshold=threshold,
        valid=not reasons,
        invalid_reasons=reasons,
        only_in_a=_diff_rows(match.only_in_a, graphs["a"]),
        only_in_b=_diff_rows(match.only_in_b, graphs["b"]),
        common=match.common,
        relevance_deltas=_relevance_deltas(deltas),
        legs=legs,
        report_md=root / REPORT_MD,
        report_json=root / REPORT_JSON,
        warnings=warnings,
        served_drift=served_drift,
    )
    cfg.sandbox.resolve("runs", f"stability-{run_id}/{REPORT_JSON}").write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cfg.sandbox.resolve("runs", f"stability-{run_id}/{REPORT_MD}").write_text(render_markdown(result), encoding="utf-8")

    if not keep:
        for leg in LEGS:
            leg_dir = cfg.sandbox.resolve("runs", f"stability-{run_id}/{leg}")
            if leg_dir.is_dir():
                shutil.rmtree(leg_dir)
    return result


# ------------------------------------------------------------------ report


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers), *("| " + " | ".join(str(c) for c in r) + " |" for r in rows)]


def render_markdown(res: StabilityResult) -> str:
    a, b = res.legs["a"], res.legs["b"]
    lines: list[str] = [f"# Stability report — {res.run_id}", ""]
    if not res.valid:
        lines += ["**INVALID for §8(b) — legs not equivalent.** Scores below are printed for information only.", ""]
        lines += [f"- {r}" for r in res.invalid_reasons]
        lines.append("")
    verdict = "PASS" if res.passed else "FAIL"
    lines += [
        f"- Jaccard (normalised titles): **{res.jaccard:.2f}** — threshold {res.threshold:.2f} → **{verdict}**",
        f"- Overlap: {res.overlap_a:.2f} of leg a, {res.overlap_b:.2f} of leg b",
        f"- Fuzzy Jaccard (similarity ≥ {FUZZY_SIMILARITY:.2f}): {res.fuzzy_jaccard:.2f}",
        f"- Nodes: a = {a.nodes}, b = {b.nodes}, common = {len(res.common)}",
        f"- Folder: `{res.root}`" + ("" if any((res.root / leg).is_dir() for leg in LEGS) else " (leg folders removed; run with --keep to retain them)"),
        "",
        "## Provider and models",
        "",
        f"Provider: `{a.provider}`" + ("" if a.provider == b.provider else f" (leg a) / `{b.provider}` (leg b)"),
        *([f"Served-model drift between legs (allowed): {', '.join(res.served_drift)}"] if res.served_drift and res.valid else []),
        "",
        *_table(
            ["stage", "requested (a)", "served (a)", "requested (b)", "served (b)"],
            [[s, a.models[s]["requested"], a.models[s]["served"] or "—", b.models[s]["requested"], b.models[s]["served"] or "—"] for s in STAGES],
        ),
        "",
        "## Only in leg a",
        "",
    ]
    lines += [f"- {d['title']} ({d['id']}) — {'; '.join(d['locators']) or 'no locator'}" for d in res.only_in_a] or ["(none)"]
    lines += ["", "## Only in leg b", ""]
    lines += [f"- {d['title']} ({d['id']}) — {'; '.join(d['locators']) or 'no locator'}" for d in res.only_in_b] or ["(none)"]
    lines += ["", "## Common", ""]
    lines += [f"- {t}" for t in res.common] or ["(none)"]
    d = res.relevance_deltas
    lines += [
        "",
        "## related_to relevance deltas (observed, not gated)",
        "",
        f"Matched edges: {d.n}; mean |Δ| = {'—' if d.mean_abs is None else f'{d.mean_abs:.1f}'}; median |Δ| = {'—' if d.median_abs is None else f'{d.median_abs:.1f}'}",
        "",
        *_table(["|Δ| bucket", "edges"], [[label, d.histogram.get(label, 0)] for label, _lo, _hi in DELTA_BUCKETS]),
        "",
        "## Legs",
        "",
    ]
    for leg, info in res.legs.items():
        lines.append(
            f"- leg {leg}: {info.nodes} node(s); deferred: {', '.join(info.deferred) or 'none'}; failed (conversion): {', '.join(info.failed) or 'none'}; "
            f"report `{info.report_md or info.report_json}`"
        )
    if res.warnings:
        lines += ["", "## Warnings", "", *(f"- {w}" for w in res.warnings)]
    return "\n".join(lines) + "\n"


__all__ = [
    "DELTA_BUCKETS",
    "FUZZY_SIMILARITY",
    "LEGS",
    "LegInfo",
    "NodeMatch",
    "REPORT_JSON",
    "REPORT_MD",
    "RelevanceDeltas",
    "STABILITY_THRESHOLD",
    "StabilityResult",
    "fuzzy_jaccard",
    "jaccard",
    "leg_config",
    "leg_root_rel",
    "match_nodes",
    "overlaps",
    "relevance_delta_histogram",
    "render_markdown",
    "run",
]
