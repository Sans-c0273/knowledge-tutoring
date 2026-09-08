"""WU4/WU5 test data and doubles: graph_io, HTML render (R14), stability (R16),
relevance sampler (R17) and the remaining CLI commands (DESIGN §18).

Offline, deterministic, zero tokens. Node files are written here as LITERAL §8.2
text (not through `kg.notes.render`) so the R14/R17 tests depend only on the
on-disk note format, not on WU3's renderer being finished.

Identifiers this work unit fixes (DESIGN §13, §15, §16, §18, §20 name the modules;
the rest is chosen here and listed in the test-engineer report):

  kg.graph_io       load_graph(graph_dir: Path) -> Graph
                    Graph(.nodes: dict[id, GraphNode], .edges: list[GraphEdge], .meta: dict,
                          .problems: list[Problem])            .to_index() -> dict  (§8.6 shape)
                    GraphNode(id, title, aliases, file, definition_plain, origin, provider, model,
                              sources: list[dict(locator, quote, file)], degree)
                    GraphEdge(source, target, type, relevance=None)
                        -- symmetric types (related_to, same_as) appear ONCE, source=min(id), target=max(id)
                    Problem(file, reason)                      -- missing/corrupt note: reported, not fatal
                    write_index(graph, path) -> Path           -- `_index.json`
  kg.render.html    render(graph_dir, out_path, *, schema: EdgeSchema, now: datetime | None = None) -> Path
                    EDGE_STYLES: dict[edge_type, dict]         -- covers every edge type of BOTH schemas
                    TEMPLATE_PATH -> src/kg/render/template.html ; VENDOR_JS_PATH -> vendor/force-graph.min.js
                    graph.html DOM:   <script id="graph-data" type="application/json">…</script>
                                      #search  #min-relevance (type=range)  #edge-filters  #inspector
                    payload = {nodes:[…§8.6…], edges:[{source,target,type,relevance?}],
                               meta:{schema,corpus,…}, styles:{edge_type: {…}}}
  kg.stability      STABILITY_THRESHOLD = 0.70
                    run(cfg, *, call_stage, now, keep=False, threshold=STABILITY_THRESHOLD, fetch=None)
                        -> StabilityResult
                    StabilityResult(run_id, root, jaccard, overlap_a, overlap_b, fuzzy_jaccard, passed,
                                    threshold, valid, invalid_reasons, only_in_a, only_in_b, common,
                                    relevance_deltas: RelevanceDeltas(n, histogram, mean_abs, median_abs),
                                    legs: {"a": LegInfo, "b": LegInfo}, report_md, report_json)
                    LegInfo(provider, models: {stage: {"requested": str, "served": str|None}},
                            nodes: int, deferred: list[str], report_json)
                    leg_config(cfg, run_id, leg) -> Config     -- paths under <runs>/stability-<run_id>/<leg>/
                    jaccard(a, b) -> float ; overlaps(a, b) -> (frac_of_a, frac_of_b)
                    match_nodes(a_titles, b_titles) -> NodeMatch(common, only_in_a, only_in_b)  (norm_title'd, sorted)
                    relevance_delta_histogram(deltas) -> {"0-5","6-10","11-20","21-40","41-100": int}
                    files: <runs>/stability-<run_id>/{a,b}/ + stability-report.md + stability-report.json
  kg.relevance_sample
                    DEFAULT_BANDS = ((0, 39), (40, 70), (71, 100)) ; band_label((lo, hi)) -> "lo-hi"
                    INDISTINGUISHABLE_TOLERANCE = 0.10
                    VERDICTS: "monotonic" | "not monotonic" | "indistinguishable" | "insufficient data"
                    sample(graph_dir, *, n_per_band, bands=DEFAULT_BANDS, seed, now) -> SampleResult
                    SampleResult(rows: list[SampleRow], path, shortfall: dict[label, int], bands)
                    SampleRow(edge_id, source_id, source_title, target_id, target_title, relevance, band,
                              source_definition, target_definition)     edge_id = "<min id>~<max id>"
                    review file: _review/relevance-sample-<run_id>.md, table header
                        | edge | source | target | relevance | band | verdict | notes |
                    summarise(path) -> Summary(bands: {label: BandStat(agree, disagree, unfilled, rate|None)},
                                               monotonic: bool, verdict: str, filled: int, unfilled: int)
  kg.cli            main(["render"|"gates"|"stability"|"sample-relevance"|"summarise-relevance", …]) -> int
                    sample-relevance --n N  →  n_per_band = ceil(N / len(bands))
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_JS = REPO_ROOT / "vendor" / "force-graph.min.js"
TEMPLATE = REPO_ROOT / "src" / "kg" / "render" / "template.html"

NOW = datetime(2026, 8, 25, 10, 15, 2, tzinfo=UTC)
RUN_ID = "2026-08-25T10-15-02Z"
TODAY = "2026-08-25"
NOW_2 = datetime(2026, 8, 26, 9, 0, 0, tzinfo=UTC)
RUN_ID_2 = "2026-08-26T09-00-00Z"

THAI_TITLE = "การเรียนรู้เชิงลึก"  # "deep learning"

BANDS = ((0, 39), (40, 70), (71, 100))
BAND_LABELS = ("0-39", "40-70", "71-100")
DELTA_BUCKETS = ("0-5", "6-10", "11-20", "21-40", "41-100")

SYMMETRIC = {"related_to", "same_as"}


# ------------------------------------------------------------ literal notes


def slugify(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return s


def stem_for(node_id: str, title: str) -> str:
    slug = slugify(title)
    return f"{node_id}-{slug}" if slug else node_id


def note_spec(
    node_id: str,
    title: str,
    *,
    schema: str,
    definition: str | None = None,
    aliases: list[str] | None = None,
    sources: list[tuple[str, str, str]] | None = None,
    edges: list[tuple[str, str, int | None]] | None = None,
    provider: str = "claude_subscription",
    model: str = "claude-sonnet-5",
    run: str = RUN_ID,
) -> dict[str, Any]:
    """edges: list of (type, target_id, relevance|None); sources: list of (locator, file, quote)."""
    src_file = "a.md"
    return {
        "id": node_id,
        "title": title,
        "aliases": list(aliases or []),
        "schema": schema,
        "origin": "course_material" if schema == "education" else None,
        "definition": definition if definition is not None else f"{title} is a concept in this corpus.",
        "sources": list(sources or [(f"{src_file}#heading={title}", src_file, f"{title} is defined here.")]),
        "edges": list(edges or []),
        "provider": provider,
        "model": model,
        "run": run,
    }


def _yaml_flow_list(items: list[str]) -> str:
    return "[" + ", ".join(items) + "]"


def note_text(spec: dict[str, Any], titles: dict[str, str]) -> str:
    """Render one node file exactly in the DESIGN §8.2 / §8.3 / §8.4 shape."""
    fm: list[str] = [
        "---",
        f"id: {spec['id']}",
        f"title: {spec['title']}",
        f"aliases: {_yaml_flow_list(spec['aliases'])}",
        f"schema: {spec['schema']}",
    ]
    if spec["origin"]:
        fm.append(f"origin: {spec['origin']}")
    fm += [
        f"created: {TODAY}",
        f"updated: {TODAY}",
        f"run: {spec['run']}",
        "prompt_version: atomize@1",
        f"provider: {spec['provider']}",
        f"model: {spec['model']}",
        "review: []",
        "sources:",
    ]
    for locator, file, _quote in spec["sources"]:
        fm.append(f"  - locator: {locator}")
        fm.append(f"    file: {file}")
    by_type: dict[str, list[tuple[str, int | None]]] = {}
    for etype, target, rel in spec["edges"]:
        by_type.setdefault(etype, []).append((target, rel))
    if by_type:
        fm.append("edges:")
        for etype, targets in by_type.items():
            if etype == "related_to":
                fm.append("  related_to:")
                for target, rel in targets:
                    fm.append(f"    - {{id: {target}, relevance: {rel}}}")
            else:
                fm.append(f"  {etype}: {_yaml_flow_list([t for t, _ in targets])}")
    else:
        fm.append("edges: {}")
    fm.append("---")

    body = [f"# {spec['title']}", "", "## Definition", spec["definition"], "", "## Relations"]
    for etype, targets in by_type.items():
        body.append(f"### {etype}")
        for target, rel in targets:
            t_title = titles[target]
            link = f"- [[{stem_for(target, t_title)}|{t_title}]]"
            body.append(f"{link} — relevance {rel}" if etype == "related_to" else link)
    body += ["", "## Source"]
    for locator, _file, quote in spec["sources"]:
        body.append(f"### {locator}")
        body.append(f"> {quote}")
    return "\n".join(fm + body) + "\n"


def norm_title_local(s: str) -> str:
    """Just enough normalisation for registry rows in fixtures (lower + collapse ws)."""
    return " ".join(s.lower().split())


def write_graph(graph_dir: Path, specs: list[dict[str, Any]], *, corpus: str, schema: str) -> dict[str, Path]:
    """Write nodes/*.md + _registry.yaml (§8.5) the way S8 would. Returns id -> file path."""
    nodes_dir = graph_dir / "nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)
    titles = {s["id"]: s["title"] for s in specs}
    paths: dict[str, Path] = {}
    rows = []
    for spec in sorted(specs, key=lambda s: s["id"]):
        stem = stem_for(spec["id"], spec["title"])
        p = nodes_dir / f"{stem}.md"
        p.write_text(note_text(spec, titles), encoding="utf-8")
        paths[spec["id"]] = p
        rows.append(
            {
                "id": spec["id"],
                "title": spec["title"],
                "norm_title": norm_title_local(spec["title"]),
                "file": f"nodes/{stem}.md",
                "status": "active",
                "created_run": spec["run"],
            }
        )
    seq = max(int(s["id"].rsplit("-", 1)[1]) for s in specs) + 1 if specs else 1
    reg = {"corpus": corpus, "schema": schema, "next_seq": seq, "nodes": rows}
    (graph_dir / "_registry.yaml").write_text(yaml.safe_dump(reg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return paths


# ----------------------------------------------------------- fixed corpora


def general_specs() -> list[dict[str, Any]]:
    """7 nodes, corpus `kb`, schema general. Canonical edges (see GENERAL_EDGES)."""
    return [
        note_spec(
            "kb-0001",
            "Cloud First Policy",
            schema="general",
            definition="Cloud First requires agencies to justify any non-cloud deployment; it constrains which workloads may leave a [[kb-0002-sovereign-cloud|sovereign cloud]].",
            aliases=["Cloud-first"],
            sources=[
                ("https://example.go.th/policy/cloud-first#heading=2. Scope > 2.1 Data classes", "https://example.go.th/policy/cloud-first", "Agencies shall adopt cloud services as the default option."),
                ("sovereignty-notes.docx#heading=Cloud First > Data residency", "sovereignty-notes.docx", "Highly Protected data may only be processed in a sovereign cloud."),
            ],
            edges=[("related_to", "kb-0002", 82), ("related_to", "kb-0003", 41), ("part_of", "kb-0005", None)],
            provider="openrouter",
            model="anthropic/claude-sonnet-5",
        ),
        note_spec("kb-0002", "Sovereign Cloud", schema="general", edges=[("related_to", "kb-0001", 82), ("related_to", "kb-0004", 15), ("related_to", "kb-0006", 33)], provider="openrouter", model="anthropic/claude-sonnet-5"),
        note_spec("kb-0003", "In-Country Region", schema="general", edges=[("related_to", "kb-0001", 41), ("related_to", "kb-0004", 65)], provider="openrouter", model="anthropic/claude-sonnet-5"),
        note_spec("kb-0004", "Data Residency", schema="general", edges=[("related_to", "kb-0002", 15), ("related_to", "kb-0003", 65), ("related_to", "kb-0006", 95)], provider="openrouter", model="anthropic/claude-sonnet-5"),
        note_spec("kb-0005", "Thai Public Sector Cloud", schema="general", edges=[("same_as", "kb-0007", None)], provider="openrouter", model="anthropic/claude-sonnet-5"),
        note_spec("kb-0006", THAI_TITLE, schema="general", definition="เทคนิคการเรียนรู้ของเครื่องที่ใช้โครงข่ายประสาทหลายชั้น", edges=[("related_to", "kb-0004", 95), ("related_to", "kb-0002", 33)], provider="openrouter", model="anthropic/claude-sonnet-5"),
        note_spec("kb-0007", "Public Cloud (Thailand)", schema="general", edges=[("same_as", "kb-0005", None)], provider="openrouter", model="anthropic/claude-sonnet-5"),
    ]


# Canonical (source, target, type, relevance) — symmetric pairs once, min id first.
GENERAL_EDGES: set[tuple[str, str, str, int | None]] = {
    ("kb-0001", "kb-0002", "related_to", 82),
    ("kb-0001", "kb-0003", "related_to", 41),
    ("kb-0002", "kb-0004", "related_to", 15),
    ("kb-0003", "kb-0004", "related_to", 65),
    ("kb-0004", "kb-0006", "related_to", 95),
    ("kb-0002", "kb-0006", "related_to", 33),
    ("kb-0001", "kb-0005", "part_of", None),
    ("kb-0005", "kb-0007", "same_as", None),
}
GENERAL_IDS = {f"kb-000{i}" for i in range(1, 8)}
GENERAL_RELEVANCES_BY_BAND = {"0-39": {15, 33}, "40-70": {41, 65}, "71-100": {82, 95}}


def education_specs() -> list[dict[str, Any]]:
    """7 nodes, corpus `stat101`, schema education, every education edge type used once."""
    return [
        note_spec("stat101-0001", "Foundations of Probability", schema="education"),
        note_spec("stat101-0002", "Sample Space", schema="education", aliases=["Outcome space"], edges=[("prerequisite_of", "stat101-0003", None), ("part_of", "stat101-0001", None)]),
        note_spec("stat101-0003", "Event", schema="education"),
        note_spec("stat101-0004", "Coin Toss Outcome Set", schema="education", edges=[("example_of", "stat101-0002", None)]),
        note_spec("stat101-0005", "Measurable Event", schema="education", edges=[("refines", "stat101-0003", None)]),
        note_spec("stat101-0006", "Sigma-Algebra Event", schema="education", edges=[("supersedes", "stat101-0005", None), ("same_as", "stat101-0007", None)]),
        note_spec("stat101-0007", "Event (Measure Theory)", schema="education", edges=[("same_as", "stat101-0006", None)]),
    ]


EDUCATION_EDGES: set[tuple[str, str, str, int | None]] = {
    ("stat101-0002", "stat101-0003", "prerequisite_of", None),
    ("stat101-0002", "stat101-0001", "part_of", None),
    ("stat101-0004", "stat101-0002", "example_of", None),
    ("stat101-0005", "stat101-0003", "refines", None),
    ("stat101-0006", "stat101-0005", "supersedes", None),
    ("stat101-0006", "stat101-0007", "same_as", None),
}
EDUCATION_IDS = {f"stat101-000{i}" for i in range(1, 8)}


def write_general_graph(graph_dir: Path) -> dict[str, Path]:
    return write_graph(graph_dir, general_specs(), corpus="kb", schema="general")


def write_education_graph(graph_dir: Path) -> dict[str, Path]:
    return write_graph(graph_dir, education_specs(), corpus="stat101", schema="education")


# A wide chain graph for the sampler: many related_to edges per band.
SAMPLER_RELEVANCES = [3, 8, 12, 17, 21, 26, 30, 35, 38, 39, 0, 11,  # 12 in 0-39
                      40, 44, 48, 52, 55, 58, 61, 64, 67, 70, 45, 69,  # 12 in 40-70
                      71, 74, 78, 81, 85, 88, 91, 94, 97, 100, 73, 99]  # 12 in 71-100


def sampler_specs() -> list[dict[str, Any]]:
    """Chain s-0001 — s-0002 — … with one related_to edge per link (36 edges) + one part_of."""
    n = len(SAMPLER_RELEVANCES) + 1
    edges_for: dict[int, list[tuple[str, str, int | None]]] = {i: [] for i in range(1, n + 1)}
    for i, rel in enumerate(SAMPLER_RELEVANCES, start=1):
        a, b = f"s-{i:04d}", f"s-{i + 1:04d}"
        edges_for[i].append(("related_to", b, rel))
        edges_for[i + 1].append(("related_to", a, rel))
    edges_for[n].append(("part_of", "s-0001", None))
    return [note_spec(f"s-{i:04d}", f"Concept {i:02d}", schema="general", definition=f"Definition of concept {i:02d}.", edges=edges_for[i]) for i in range(1, n + 1)]


def write_sampler_graph(graph_dir: Path) -> dict[str, Path]:
    return write_graph(graph_dir, sampler_specs(), corpus="s", schema="general")


def band_of(rel: int) -> str:
    for (lo, hi), label in zip(BANDS, BAND_LABELS):
        if lo <= rel <= hi:
            return label
    raise ValueError(rel)


def sampler_pool_by_band() -> dict[str, int]:
    out = {label: 0 for label in BAND_LABELS}
    for rel in SAMPLER_RELEVANCES:
        out[band_of(rel)] += 1
    return out


# --------------------------------------------------------------- html tools

SCRIPT_RE = re.compile(r"<script\b([^>]*)>(.*?)</script>", re.DOTALL | re.IGNORECASE)


def inline_scripts(html: str) -> list[tuple[str, str]]:
    """[(attributes, body)] for every <script> element."""
    return [(m.group(1), m.group(2)) for m in SCRIPT_RE.finditer(html)]


def graph_payload(html: str) -> dict[str, Any]:
    import json

    for attrs, body in inline_scripts(html):
        if 'id="graph-data"' in attrs:
            return json.loads(body)
    raise AssertionError('no <script id="graph-data" type="application/json"> in graph.html')


def js_scripts(html: str) -> list[str]:
    """Bodies of executable inline scripts (everything that is not the JSON payload)."""
    return [body for attrs, body in inline_scripts(html) if "application/json" not in attrs and body.strip()]


def canonical_edges(edges: list[dict[str, Any]]) -> set[tuple[str, str, str, int | None]]:
    out = set()
    for e in edges:
        s, t = e["source"], e["target"]
        if e["type"] in SYMMETRIC:
            s, t = sorted((s, t))
        out.add((s, t, e["type"], e.get("relevance")))
    return out


# ------------------------------------------------------------- review files


def fill_verdicts(path: Path, verdict_for: Any) -> None:
    """Fill the blank `verdict` column of a review file. verdict_for(relevance:int, band:str) -> str."""
    lines = path.read_text(encoding="utf-8").splitlines()
    header_idx = next(i for i, l in enumerate(lines) if l.startswith("|") and "| verdict |" in l)
    cols = [c.strip() for c in lines[header_idx].strip("|").split("|")]
    v_i, rel_i, band_i = cols.index("verdict"), cols.index("relevance"), cols.index("band")
    for i in range(header_idx + 2, len(lines)):
        line = lines[i]
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != len(cols):
            continue
        verdict = verdict_for(int(cells[rel_i]), cells[band_i])
        cells[v_i] = verdict
        lines[i] = "| " + " | ".join(cells) + " |"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def snapshot_tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
