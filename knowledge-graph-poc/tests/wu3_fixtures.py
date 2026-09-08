"""WU3 test doubles and worked-example data (graph core: notes, registry, normalise,
gates, stages, pipeline, report). Offline, deterministic, zero tokens.

Names this work unit fixes (DESIGN §8, §9, §10, §11, §12, §14, §20 name the modules;
identifiers below are chosen here and listed in the test-engineer report):

  kg.notes        Node(id, title, aliases, schema, origin, created, updated, run, prompt_version,
                       provider, model, review, sources: list[NodeSource], edges: list[NodeEdge],
                       definition)                      # definition is PLAIN prose (links stripped)
                  NodeSource(locator, file, quote)      # quote lives in `## Source`, not frontmatter
                  NodeEdge(type, target, relevance=None)
                  LinkTarget(stem, title, aliases=())
                  render(node, targets: Mapping[id, LinkTarget]) -> str
                  parse(text) -> Node
                  weave(text, targets, *, exclude=None) -> str
                  strip_wikilinks(text) -> str
  kg.registry     Registry(corpus, schema) .allocate(title, *, run, aliases=()) -> RegistryRow
                  .retire(id) .find(title_or_alias) -> id | None .rows .active() .next_seq
                  .file_for(id) .save(path) / Registry.load(path)   (path = _registry.yaml)
                  RegistryRow(id, title, norm_title, file, status, created_run, aliases)
                  slugify(title) -> str, stem_for(node_id, title) -> str
  kg.normalise    norm_title(s) -> str, similarity(a, b) -> float, uses_trigrams(s) -> bool
  kg.gates        Edge(type, source_id, target_id, relevance=None)
                  Rejection(gate, reason, edge)
                  out_of_schema(edge, schema), relevance(edge, schema), dag(edge, existing, schema)
                      -> Rejection | None
                  mirror(nodes, schema) -> list[Rejection]
                  gate_edges(proposals, *, existing, schema, review_dir=None) -> GateResult(accepted, rejected)
                  validate_graph(graph_dir, schema) -> list[Rejection]        (`kg gates`, no API)
                  REJECTED_EDGES_FILE = "_review/rejected-edges.jsonl"
  kg.stages.atomize     atomize_chunk(chunk, cfg, *, call_stage, ledger=None) -> list[Candidate]
                        atomize(chunks, cfg, *, call_stage, ledger=None) -> list[Candidate]
                        Candidate(title, definition, aliases, unit_ids, quotes, source, chunk_id,
                                  review, sources: list[NodeSource])
  kg.stages.consolidate consolidate(candidates, registry, *, run) -> list[NodeDraft]
                        NodeDraft(id, title, aliases, definition, sources, review, chunk_ids,
                                  source_files, is_new)
  kg.stages.edges       build_roster(drafts, cap) -> Roster(entries: list[RosterEntry(id, title, aliases)],
                                                            truncated, total)
                        propose(drafts, chunks, cfg, *, call_stage, ledger=None)
                            -> EdgesResult(edges: list[Edge], dropped: list[Rejection], warnings: list[str])
  kg.stages.dedup       candidate_pairs(drafts, threshold, *, max_pairs=None) -> list[Pair(a_id, b_id, score)]
                        run(drafts, cfg, *, call_stage, ledger=None, write_same_as=None)
                            -> DedupResult(same, unsure, different: list[Judged], edges: list[Edge],
                                           unrequested: int, overflow: list[Pair])
                        Judged(a_id, b_id, score, verdict, reason)
                        write_queue(result, review_dir) -> Path          (_review/duplicates.md)
  kg.pipeline     run(cfg, *, call_stage, now: datetime, dry_run=False, write_same_as=None, fetch=None)
                      -> RunResult(run_id, status, inputs, nodes_written, report_md, report_json)
                  run_id_for(now) -> "YYYY-MM-DDTHH-MM-SSZ"
  kg.report       build(cfg, **fields) -> RunReport ; write(rep, cfg) -> (md_path, json_path)
                  render_markdown(rep) -> str ; to_dict(rep) -> dict ; ReportRedactionError
  kg.prompts.<stage>    VERSION ("<stage>@N"), SYSTEM, TEXT_SHA ; kg.prompts.edges.system_for(schema)
  kg.config       DedupConfig.pairs_per_call (new key `dedup.pairs_per_call`, default 10 — §8.0)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

# --------------------------------------------------------------------- clock

NOW = datetime(2026, 8, 25, 10, 15, 2, tzinfo=UTC)
RUN_ID = "2026-08-25T10-15-02Z"
TODAY = "2026-08-25"

NOW_2 = datetime(2026, 8, 26, 9, 0, 0, tzinfo=UTC)
RUN_ID_2 = "2026-08-26T09-00-00Z"

THAI_TITLE = "การเรียนรู้เชิงลึก"  # "deep learning"
THAI_TITLE_2 = "การเรียนรู้ของเครื่อง"  # "machine learning" — shares many trigrams with the first


# ------------------------------------------------------------ units / chunks


try:  # WU1 lands before WU3 is implemented; until then a duck-typed stand-in
    from kg.convert import Unit  # type: ignore
except Exception:  # pragma: no cover - only before WU1 exists

    @dataclass
    class Unit:  # type: ignore[no-redef]
        uid: str
        locator: str
        heading_path: list[str]
        text: str
        kind: str
        source: str


def make_unit(uid: str, text: str, *, source: str = "a.md", heading: list[str] | None = None, locator: str | None = None, kind: str = "text"):
    heading = heading if heading is not None else [f"Sec {uid}"]
    if locator is None:
        locator = f"{source}#heading={' > '.join(heading)}" if heading else f"{source}#lines=1-1"
    return Unit(uid=uid, locator=locator, heading_path=list(heading), text=text, kind=kind, source=source)


@dataclass
class FakeChunk:
    """Duck-types kg.chunk.Chunk (WU1 surface: .cid .source .units .unit_ids .locators).

    Stages must work on these attributes only — they must not isinstance-check Chunk.
    """

    cid: str
    units: list[Any]
    source: str = ""
    unit_ids: list[str] = field(default_factory=list)
    locators: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.source:
            self.source = self.units[0].source if self.units else ""
        self.unit_ids = [u.uid for u in self.units]
        self.locators = [u.locator for u in self.units]


def make_chunk(cid: str, units: list[Any]) -> FakeChunk:
    return FakeChunk(cid=cid, units=list(units))


MARKER_RE = re.compile(r"<<(?P<uid>[^|>]+?) \| (?P<locator>[^|>]*?) \| (?P<path>[^>]*?)>>")


def parse_rendered_chunk(user_text: str) -> list[dict[str, str]]:
    """Split a rendered chunk prompt back into [{uid, locator, path, text}] using the §5 markers."""
    out: list[dict[str, str]] = []
    matches = list(MARKER_RE.finditer(user_text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(user_text)
        out.append(
            {
                "uid": m.group("uid").strip(),
                "locator": m.group("locator").strip(),
                "path": m.group("path").strip(),
                "text": user_text[m.end() : end],
            }
        )
    return out


def ws(s: str) -> str:
    return " ".join(s.split())


def first_line(text: str) -> str:
    """First non-empty, non-heading line (unit text may or may not start with its `# Heading`)."""
    for line in text.splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            return line.strip()
    return ""


# ------------------------------------------------------------- id helpers

ID_RE = re.compile(r"\b[a-z][a-z0-9]*-\d{4}\b")


def ids_in(text: str, slug: str | None = None) -> list[str]:
    """Permanent-shaped ids (`<slug>-NNNN`) found in `text`, de-duplicated and sorted.

    The edge/dedup prompts must show the roster ids verbatim (the model copies them);
    sorting makes the scripted handlers independent of the prompt's listing order.
    """
    pat = re.compile(rf"\b{re.escape(slug)}-\d{{4}}\b") if slug else ID_RE
    return sorted({m.group(0) for m in pat.finditer(text)})


# -------------------------------------------------------- scripted call_stage


class ScriptedCallStage:
    """Stand-in for `kg.llm.call_stage` with per-stage handlers. Never touches a provider.

    handlers: {stage: callable(call) -> data | dict | Exception}  or  {stage: [data, ...]}.
    A dict result is validated through the requested `schema`; an Exception is raised.
    If a `ledger` kwarg is passed, a UsageRow is recorded (lazy import) so report tests
    can observe usage through the real ledger shape.
    """

    def __init__(self, handlers: dict[str, Any] | None = None, *, cfg: Any = None) -> None:
        self.handlers: dict[str, Any] = dict(handlers or {})
        self.cfg = cfg
        self.calls: list[dict[str, Any]] = []

    def calls_for(self, stage: str) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["stage"] == stage]

    def __call__(self, stage: str, system: str, user_text: str, schema: type, images: list | None = None, **kw: Any):
        call = {"stage": stage, "system": system, "user_text": user_text, "schema": schema, "images": images, "extra": kw}
        self.calls.append(call)
        handler = self.handlers.get(stage)
        if handler is None:
            raise AssertionError(f"ScriptedCallStage: unexpected call to stage {stage!r}")
        out = handler(call) if callable(handler) else handler.pop(0)
        if isinstance(out, BaseException):
            raise out
        if isinstance(out, dict):
            out = schema.model_validate(out)  # type: ignore[attr-defined]
        model = self.cfg.llm.model_for(stage) if self.cfg is not None else "fake-model"
        served = f"{model}-served"
        provider = self.cfg.llm.provider if self.cfg is not None else "fake"
        ledger = kw.get("ledger")
        if ledger is not None:
            self._record(ledger, stage, provider, model, served, kw.get("prompt_version"))
        return SimpleNamespace(
            data=out,
            usage=SimpleNamespace(input_tokens=100, output_tokens=20, cached_tokens=0, reasoning_tokens=0),
            provider=provider,
            model_requested=model,
            model_served=served,
            attempts=1,
            prompt_version=kw.get("prompt_version"),
        )

    @staticmethod
    def _record(ledger: Any, stage: str, provider: str, model: str, served: str, prompt_version: str | None) -> None:
        try:
            from kg.llm.ledger import UsageRow
        except Exception:  # pragma: no cover - ledger not built yet
            return
        subscription = provider == "claude_subscription"
        ledger.record(
            UsageRow(
                stage=stage,
                provider=provider,
                model_requested=model,
                model_served=served,
                input_tokens=100,
                output_tokens=20,
                cost_usd=None if subscription else 0.001,
                cost_estimate_usd=0.01 if subscription else None,
                attempt_no=1,
                outcome="ok",
                duration_s=0.01,
                prompt_version=prompt_version,
            )
        )


# ------------------------------------------------------ smart stage handlers


def smart_atomize(call: dict[str, Any]) -> dict[str, Any]:
    """One CandidateNode per unit in the rendered chunk; title = last heading segment,
    definition = quote = first non-empty line of the unit text (so grounding always passes)."""
    nodes = []
    for u in parse_rendered_chunk(call["user_text"]):
        line = first_line(u["text"])
        if not line:
            continue
        title = u["path"].split(" > ")[-1].strip() if u["path"] else " ".join(line.split()[:3])
        title = title.lstrip("#").strip() or u["uid"]
        nodes.append({"title": title, "definition": line.lstrip("#").strip(), "aliases": [], "unit_ids": [u["uid"]], "quotes": [{"unit_id": u["uid"], "text": line}]})
    return {"nodes": nodes}


def no_edges(call: dict[str, Any]) -> dict[str, Any]:
    return {"edges": []}


def general_edges_handler(slug: str, relevance: int = 72) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """related_to ids[0]<->ids[1] (scored) and part_of ids[2]->ids[0] when a third id exists."""

    def _h(call: dict[str, Any]) -> dict[str, Any]:
        ids = ids_in(call["user_text"], slug)
        edges: list[dict[str, Any]] = []
        if len(ids) >= 2:
            edges.append({"type": "related_to", "source_id": ids[0], "target_id": ids[1], "relevance": relevance})
        if len(ids) >= 3:
            edges.append({"type": "part_of", "source_id": ids[2], "target_id": ids[0], "relevance": None})
        return {"edges": edges}

    return _h


def education_cycle_handler(slug: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """prerequisite_of ids[0]->ids[1]->ids[2]->ids[0]: the third closes a cycle (R13)."""

    def _h(call: dict[str, Any]) -> dict[str, Any]:
        ids = ids_in(call["user_text"], slug)
        if len(ids) < 3:
            return {"edges": []}
        return {
            "edges": [
                {"type": "prerequisite_of", "source_id": ids[0], "target_id": ids[1]},
                {"type": "prerequisite_of", "source_id": ids[1], "target_id": ids[2]},
                {"type": "prerequisite_of", "source_id": ids[2], "target_id": ids[0]},
            ]
        }

    return _h


def dedup_same_handler(slug: str, reason: str = "same concept, different wording") -> Callable[[dict[str, Any]], dict[str, Any]]:
    def _h(call: dict[str, Any]) -> dict[str, Any]:
        ids = ids_in(call["user_text"], slug)
        if len(ids) < 2:
            return {"judgements": []}
        return {"judgements": [{"a_id": ids[0], "b_id": ids[1], "verdict": "same", "reason": reason}]}

    return _h


def default_handlers(cfg: Any, *, edges: Callable | None = None, dedup: Callable | None = None, atomize: Callable | None = None) -> dict[str, Any]:
    return {
        "describe": lambda call: {"kind": "diagram", "title": "Diagram", "description": "A box.", "elements": [], "text_visible": [], "relationships": []},
        "atomize": atomize or smart_atomize,
        "edges": edges or no_edges,
        "dedup": dedup or dedup_same_handler(cfg.corpus.slug),
    }


# --------------------------------------------------------- pipeline corpora

A_MD = """# Sample Space

The sample space is the set of all possible outcomes of a random experiment.

# Event

An event is a subset of the sample space.

# Probability Measure

A probability measure assigns a number between zero and one to each event.
"""

B_MD = """# Random Variable

A random variable maps each outcome to a real number.

# Expectation

The expectation is the probability-weighted average of a random variable.
"""

DUP_MD = """# The Sample Space Definition

Formally the sample space collects every result an experiment can produce.
"""

EDU_MD = """# Sets

A set is an unordered collection of distinct objects.

# Functions

A function assigns to each element of one set exactly one element of another.

# Limits

A limit describes the value a function approaches as its input approaches a point.
"""


def write_inbox(cfg: Any, files: dict[str, str]) -> Path:
    inbox = cfg.sandbox.root("inbox")
    for name, text in files.items():
        (inbox / name).write_text(text, encoding="utf-8")
    return inbox


def snapshot_tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def node_files(cfg: Any) -> list[Path]:
    return sorted((cfg.sandbox.root("graph") / "nodes").glob("*.md"))


def load_nodes(cfg: Any) -> list[Any]:
    from kg.notes import parse

    return [parse(p.read_text(encoding="utf-8")) for p in node_files(cfg)]


# -------------------------------------------------- worked examples (§8.3/8.4)


def edu_example_node():
    """DESIGN §8.3 `stat101-0004-sample-space.md` as a Node."""
    from kg.notes import Node, NodeEdge, NodeSource

    return Node(
        id="stat101-0004",
        title="Sample Space",
        aliases=["Outcome space"],
        schema="education",
        origin="course_material",
        created="2026-08-25",
        updated="2026-08-25",
        run="2026-08-25T10-15-02Z",
        prompt_version="atomize@1",
        provider="claude_subscription",
        model="claude-sonnet-5",
        review=[],
        sources=[
            NodeSource(locator="week01-probability.pptx#slide=4", file="week01-probability.pptx", quote="The sample space S is the set of all possible outcomes of an experiment."),
            NodeSource(locator="week01-handout.pdf#page=2", file="week01-handout.pdf", quote="We write S (sometimes Ω) for the collection of every outcome that could occur; an event is any subset of S."),
        ],
        edges=[
            NodeEdge(type="prerequisite_of", target="stat101-0005"),
            NodeEdge(type="prerequisite_of", target="stat101-0009"),
            NodeEdge(type="part_of", target="stat101-0001"),
        ],
        definition=(
            "The sample space is the set of all possible outcomes of a random experiment. "
            "Every event is a subset of the sample space, and a probability measure assigns a number to each such subset."
        ),
    )


def edu_example_targets():
    from kg.notes import LinkTarget

    return {
        "stat101-0004": LinkTarget(stem="stat101-0004-sample-space", title="Sample Space", aliases=("Outcome space",)),
        "stat101-0005": LinkTarget(stem="stat101-0005-event", title="Event"),
        "stat101-0009": LinkTarget(stem="stat101-0009-probability-measure", title="Probability Measure"),
        "stat101-0001": LinkTarget(stem="stat101-0001-foundations-of-probability", title="Foundations of Probability"),
    }


EDU_EXPECTED = """---
id: stat101-0004
title: Sample Space
aliases: [Outcome space]
schema: education
origin: course_material
created: 2026-08-25
updated: 2026-08-25
run: 2026-08-25T10-15-02Z
prompt_version: atomize@1
provider: claude_subscription
model: claude-sonnet-5
review: []
sources:
  - locator: week01-probability.pptx#slide=4
    file: week01-probability.pptx
  - locator: week01-handout.pdf#page=2
    file: week01-handout.pdf
edges:
  prerequisite_of: [stat101-0005, stat101-0009]
  part_of: [stat101-0001]
---
# Sample Space

## Definition
The sample space is the set of all possible outcomes of a random experiment. Every [[stat101-0005-event|event]] is a subset of the sample space, and a [[stat101-0009-probability-measure|probability measure]] assigns a number to each such subset.

## Relations
### prerequisite_of
- [[stat101-0005-event|Event]]
- [[stat101-0009-probability-measure|Probability Measure]]
### part_of
- [[stat101-0001-foundations-of-probability|Foundations of Probability]]

## Source
### week01-probability.pptx#slide=4
> The sample space S is the set of all possible outcomes of an experiment.
### week01-handout.pdf#page=2
> We write S (sometimes Ω) for the collection of every outcome that could occur; an event is any subset of S.
"""


def gen_example_node():
    """DESIGN §8.4 `kb-0017-cloud-first-policy.md` as a Node."""
    from kg.notes import Node, NodeEdge, NodeSource

    return Node(
        id="kb-0017",
        title="Cloud First Policy",
        aliases=[],
        schema="general",
        origin=None,
        created="2026-08-25",
        updated="2026-08-25",
        run="2026-08-25T14-02-11Z",
        prompt_version="atomize@1",
        provider="openrouter",
        model="anthropic/claude-sonnet-5",
        review=[],
        sources=[
            NodeSource(
                locator="https://example.go.th/policy/cloud-first#heading=2. Scope > 2.1 Data classes",
                file="https://example.go.th/policy/cloud-first",
                quote="Agencies shall adopt cloud services as the default option and document the reasons for any exception.",
            ),
            NodeSource(
                locator="sovereignty-notes.docx#heading=Cloud First > Data residency",
                file="sovereignty-notes.docx",
                quote="Highly Protected data may only be processed in a sovereign cloud, not merely an in-country region.",
            ),
        ],
        edges=[
            NodeEdge(type="related_to", target="kb-0021", relevance=82),
            NodeEdge(type="related_to", target="kb-0030", relevance=41),
            NodeEdge(type="part_of", target="kb-0002"),
        ],
        definition=(
            "Cloud First is a government procurement stance that requires agencies to justify any non-cloud deployment; "
            "it is paired with data-classification rules that determine which workloads may leave a sovereign cloud."
        ),
    )


def gen_example_targets():
    from kg.notes import LinkTarget

    return {
        "kb-0017": LinkTarget(stem="kb-0017-cloud-first-policy", title="Cloud First Policy"),
        "kb-0021": LinkTarget(stem="kb-0021-sovereign-cloud", title="Sovereign Cloud"),
        "kb-0030": LinkTarget(stem="kb-0030-in-country-region", title="In-Country Region"),
        "kb-0002": LinkTarget(stem="kb-0002-thai-public-sector-cloud", title="Thai Public Sector Cloud"),
    }


GEN_EXPECTED = """---
id: kb-0017
title: Cloud First Policy
aliases: []
schema: general
created: 2026-08-25
updated: 2026-08-25
run: 2026-08-25T14-02-11Z
prompt_version: atomize@1
provider: openrouter
model: anthropic/claude-sonnet-5
review: []
sources:
  - locator: https://example.go.th/policy/cloud-first#heading=2. Scope > 2.1 Data classes
    file: https://example.go.th/policy/cloud-first
  - locator: sovereignty-notes.docx#heading=Cloud First > Data residency
    file: sovereignty-notes.docx
edges:
  related_to:
    - {id: kb-0021, relevance: 82}
    - {id: kb-0030, relevance: 41}
  part_of: [kb-0002]
---
# Cloud First Policy

## Definition
Cloud First is a government procurement stance that requires agencies to justify any non-cloud deployment; it is paired with data-classification rules that determine which workloads may leave a [[kb-0021-sovereign-cloud|sovereign cloud]].

## Relations
### related_to
- [[kb-0021-sovereign-cloud|Sovereign Cloud]] — relevance 82
- [[kb-0030-in-country-region|In-Country Region]] — relevance 41
### part_of
- [[kb-0002-thai-public-sector-cloud|Thai Public Sector Cloud]]

## Source
### https://example.go.th/policy/cloud-first#heading=2. Scope > 2.1 Data classes
> Agencies shall adopt cloud services as the default option and document the reasons for any exception.
### sovereignty-notes.docx#heading=Cloud First > Data residency
> Highly Protected data may only be processed in a sovereign cloud, not merely an in-country region.
"""


# ------------------------------------------------------ simple node builder


def simple_node(node_id: str, title: str, *, schema: str = "general", edges: list | None = None, definition: str | None = None, aliases: list[str] | None = None, review: list[str] | None = None, source: str = "a.md"):
    """A minimal well-formed Node for gate / graph tests."""
    from kg.notes import Node, NodeSource

    return Node(
        id=node_id,
        title=title,
        aliases=list(aliases or []),
        schema=schema,
        origin="course_material" if schema == "education" else None,
        created=TODAY,
        updated=TODAY,
        run=RUN_ID,
        prompt_version="atomize@1",
        provider="claude_subscription",
        model="claude-sonnet-5",
        review=list(review or []),
        sources=[NodeSource(locator=f"{source}#heading={title}", file=source, quote=f"{title} is defined here.")],
        edges=list(edges or []),
        definition=definition if definition is not None else f"{title} is a concept.",
    )


def write_graph(graph_dir: Path, nodes: list[Any], *, corpus: str, schema: str) -> dict[str, Any]:
    """Write nodes + a matching _registry.yaml the way S8 would; returns id -> LinkTarget."""
    from kg.notes import LinkTarget, render
    from kg.registry import Registry, stem_for

    (graph_dir / "nodes").mkdir(parents=True, exist_ok=True)
    reg = Registry(corpus=corpus, schema=schema)
    targets = {n.id: LinkTarget(stem=stem_for(n.id, n.title), title=n.title, aliases=tuple(n.aliases)) for n in nodes}
    for n in sorted(nodes, key=lambda n: n.id):
        reg.allocate(n.title, run=RUN_ID, aliases=tuple(n.aliases))
        (graph_dir / "nodes" / f"{targets[n.id].stem}.md").write_text(render(n, targets), encoding="utf-8")
    reg.save(graph_dir / "_registry.yaml")
    return targets
