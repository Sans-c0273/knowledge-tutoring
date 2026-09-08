"""`kg ingest`: S0–S8 in order with file-level deferral (DESIGN §6, §8.5–8.6, §11, §14, §23; D16, D20).

The unit of model work is one source file. For each converted file: S3 atomize →
S4 consolidate (ids allocated in a registry snapshot) → S5 edges. If any model
call for the file raises — provider error, `StageOutputInvalid` after repairs,
rate limit that cannot be waited out — the snapshot is discarded, the file is
marked `deferred(reason)`, stays in the inbox and nothing from it is written.
Then, once for the run: S6 dedup → S7 gates → S8 write (registry, notes, index,
queues), move the ingested sources to `processed/`, and write the report. The
report is written on every run, including dry runs and total failure.

Timestamps are injected (`now`); the pipeline reads no wall clock, so two runs over
the same inbox with the same `now` and the same scripted model are byte-identical.
"""

from __future__ import annotations

import copy
import re
import shutil
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kg import graph_io, report
from kg.chunk import build_chunks
from kg.config import Config
from kg.convert import ConversionOutcome, convert_inbox
from kg.convert._common import write_files_atomically
from kg.gates import REJECTED_EDGES_FILE, Edge, GateResult, Rejection, gate_edges
from kg.llm.errors import BudgetExceeded, RateLimited
from kg.llm.ledger import SUBSCRIPTION_PROVIDER, UsageLedger
from kg.notes import LinkTarget, Node, NodeEdge, NoteError, parse, render
from kg.paths import SandboxViolation
from kg.prompts import versions as prompt_versions
from kg.prompts import atomize as atomize_prompt
from kg.prompts import describe as describe_prompt
from kg.registry import Registry, RegistryError
from kg.report import REDACTED, cell, secret_values
from kg.schemas import SCHEMAS, EdgeSchema
from kg.stages import atomize as s3
from kg.stages import consolidate as s4
from kg.stages import dedup as s6
from kg.stages import edges as s5
from kg.stages.atomize import Candidate, DroppedCandidate
from kg.stages.consolidate import NodeDraft

CallStage = Callable[..., Any]
REGISTRY_FILE = "_registry.yaml"
INDEX_FILE = "_index.json"
REVIEW_DIR = "_review"
GROUNDING_FILE = "grounding.md"
DRY_RUN_IMAGE_TOKENS = 2500
#: Reasons built from an exception are `Type: first line`, capped here — SDK errors can carry whole response bodies.
MAX_REASON_LEN = 300
#: The pipeline waits out a rate limit at most this many times per run (D34); any later limit defers the remaining files.
MAX_WAITS_PER_RUN = 1
UNREADABLE_GATE = "unreadable_note"
#: A new node whose file name is already taken by a file this run did not read (never written over).
COLLISION_GATE = "collision"
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


class PipelineError(Exception):
    """The run cannot start (e.g. the registry on disk belongs to another corpus/schema)."""


class DryRunSkip(Exception):
    """Raised by the dry-run stand-in for `call_stage`: no model is ever called (D16)."""


@dataclass
class InputRow:
    name: str
    status: str
    reason: str | None = None
    units: int = 0
    chunks: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunResult:
    run_id: str
    status: str
    inputs: list[InputRow]
    nodes_written: int
    report_md: Path
    report_json: Path


@dataclass
class _FileWork:
    """Everything a successfully modelled file contributes before S7/S8."""

    outcome: ConversionOutcome
    drafts: list[NodeDraft] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    dropped: list[Rejection] = field(default_factory=list)
    dropped_candidates: list[DroppedCandidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: Stage in progress; read by the caller when `_model_file` raises.
    stage: str = "atomize"
    #: S3 results per chunk id — (candidates, dropped) — kept across a rate-limit retry so a finished
    #: chunk is never re-run and its dropped candidates are not lost from the grounding queue.
    atomized: dict[str, tuple[list[Candidate], list[DroppedCandidate]]] = field(default_factory=dict)


def run_id_for(now: datetime) -> str:
    return now.astimezone(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


def _stamp(now: datetime) -> str:
    return now.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _count_entries(root: Path) -> int:
    return sum(1 for p in root.iterdir() if not p.name.startswith(".")) if root.is_dir() else 0


class _Redactor:
    """Replaces every known credential value in free text and remembers whether it ever had to.

    Applied to exception text *before* it is truncated, so a key sitting past the cap
    cannot survive as a partial prefix. A hit marks the run ``failed`` (a credential in
    an error message is something a human must look at), as `report.redact` does.
    """

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        self.secrets = sorted({s for s in secrets if s}, key=len, reverse=True)
        self.hits = 0

    def __call__(self, text: str) -> str:
        for secret in self.secrets:
            if secret in text:
                text = text.replace(secret, REDACTED)
                self.hits += 1
        return text


def _cap(text: str) -> str:
    return text if len(text) <= MAX_REASON_LEN else text[: MAX_REASON_LEN - 3] + "..."


def _exc_text(exc: BaseException, redactor: _Redactor | None = None) -> str:
    """``<Type>: <first line of the message>``, redacted, control characters stripped, capped (never a raw SDK body)."""
    text = str(exc)
    if redactor is not None:
        text = redactor(text)
    first = _CONTROL_CHARS.sub("", text.splitlines()[0]).strip() if text.strip() else ""
    reason = f"{type(exc).__name__}: {first}" if first else type(exc).__name__
    return _cap(reason)


def _reason(stage: str, exc: BaseException, redactor: _Redactor | None = None) -> str:
    """One bounded line ``<stage>: <Type>: <message>``; the cap applies to the whole line, prefix included."""
    return _cap(f"{stage}: {_exc_text(exc, redactor)}")


# ------------------------------------------------------------ existing graph


def _load_existing(cfg: Config, warnings: list[str], redactor: _Redactor | None = None) -> tuple[Registry, dict[str, Node], set[str]]:
    """Registry, parsed active notes by id, and the ids of active rows whose note could not be read.

    An unreadable note is never overwritten: S8 skips drafts and edges that touch those ids.
    """
    reg_path = cfg.sandbox.resolve("graph", REGISTRY_FILE)
    if reg_path.is_file():
        registry = Registry.load(reg_path)
        if registry.corpus != cfg.corpus.slug or registry.schema != cfg.corpus.schema:
            raise PipelineError(
                f"{REGISTRY_FILE} belongs to corpus {registry.corpus!r} / schema {registry.schema!r}; "
                f"kg.yaml says {cfg.corpus.slug!r} / {cfg.corpus.schema!r}. Point paths.graph at a different folder."
            )
    else:
        registry = Registry(corpus=cfg.corpus.slug, schema=cfg.corpus.schema)
    nodes: dict[str, Node] = {}
    unreadable: set[str] = set()
    for row in registry.active():
        try:
            path = cfg.sandbox.resolve("graph", row.file)  # a note path escaping the sandbox is an unreadable note, not a crash (R15)
            if not path.is_file():
                warnings.append(f"registry lists {row.id} at {row.file} but the note is missing")
                continue
            nodes[row.id] = parse(path.read_text(encoding="utf-8"))
        except (NoteError, OSError, UnicodeDecodeError, SandboxViolation) as exc:
            unreadable.add(row.id)
            warnings.append(f"{row.file}: unreadable note ({_exc_text(exc, redactor)}); it will not be linked or updated this run")
    return registry, nodes, unreadable


def _existing_edges(nodes: dict[str, Node]) -> list[Edge]:
    return [Edge(e.type, n.id, e.target, e.relevance) for n in nodes.values() for e in n.edges]


# ------------------------------------------------------------------ per file


def _model_file(
    work: _FileWork,
    chunks: list[Any],
    cfg: Config,
    registry: Registry,
    linkable: dict[str, Any],
    *,
    call_stage: CallStage,
    ledger: UsageLedger,
    run_id: str,
) -> _FileWork:
    """S3 → S4 → S5 for one file. Raises on any model-boundary failure; the caller rolls back.

    `work.stage` names the stage in progress when an exception escapes. Chunks already
    atomized on a previous attempt of the same file (`work.atomized`) are not called again.
    """
    work.stage = "atomize"
    work.dropped_candidates = []
    candidates: list[Candidate] = []
    for chunk in chunks:
        if chunk.cid not in work.atomized:
            dropped: list[DroppedCandidate] = []
            found = s3.atomize_chunk(chunk, cfg, call_stage=call_stage, ledger=ledger, dropped=dropped)
            work.atomized[chunk.cid] = (found, dropped)
        found, dropped = work.atomized[chunk.cid]
        candidates.extend(found)
        work.dropped_candidates.extend(dropped)
    work.stage = "consolidate"
    s4_warnings: list[str] = []
    work.drafts = s4.consolidate(candidates, registry, run=run_id, warnings=s4_warnings)
    work.stage = "edges"
    roster = {**linkable, **{d.id: d for d in work.drafts}}
    res = s5.propose(work.drafts, chunks, cfg, call_stage=call_stage, ledger=ledger, roster_drafts=list(roster.values()))
    work.edges, work.dropped = res.edges, res.dropped
    work.warnings = [*(f"{work.outcome.item.name}: {w}" for w in s4_warnings), *res.warnings]
    return work


# -------------------------------------------------------------------- S8


def _new_node(draft: NodeDraft, cfg: Config, schema: EdgeSchema, *, today: str, run_id: str) -> Node:
    return Node(
        id=draft.id,
        title=draft.title,
        aliases=list(draft.aliases),
        schema=schema.name,
        origin=schema.origin_values[0] if schema.origin_values else None,
        created=today,
        updated=today,
        run=run_id,
        prompt_version=draft.prompt_version or atomize_prompt.VERSION,
        provider=draft.provider or cfg.llm.provider,
        model=draft.model or cfg.llm.model_for("atomize"),
        review=list(draft.review),
        sources=list(draft.sources),
        edges=[],
        definition=draft.definition,
    )


def _merge_into(node: Node, draft: NodeDraft, *, today: str, run_id: str) -> Node:
    node = copy.deepcopy(node)
    known = {a.casefold() for a in node.aliases} | {node.title.casefold()}
    for a in draft.aliases:
        if a.casefold() not in known:
            node.aliases.append(a)
            known.add(a.casefold())
    seen = {(s.locator, s.quote) for s in node.sources}
    for s in draft.sources:
        if (s.locator, s.quote) not in seen:
            node.sources.append(s)
            seen.add((s.locator, s.quote))
    for flag in draft.review:
        if flag not in node.review:
            node.review.append(flag)
    if not node.definition and draft.definition:
        node.definition = draft.definition
    node.updated, node.run = today, run_id
    return node


def _attach_edges(nodes: dict[str, Node], existing: dict[str, Node], accepted: list[Edge], schema: EdgeSchema, *, today: str, run_id: str) -> None:
    """Add accepted edges to the nodes to be written; symmetric types are mirrored (DESIGN §8.2)."""

    def target_node(node_id: str) -> Node:
        if node_id not in nodes:
            node = copy.deepcopy(existing[node_id])
            node.updated, node.run = today, run_id
            nodes[node_id] = node
        return nodes[node_id]

    def add(node: Node, edge: NodeEdge) -> None:
        if not any(e.type == edge.type and e.target == edge.target for e in node.edges):
            node.edges.append(edge)

    for e in accepted:
        add(target_node(e.source_id), NodeEdge(e.type, e.target_id, e.relevance))
        if schema.is_symmetric(e.type):
            add(target_node(e.target_id), NodeEdge(e.type, e.source_id, e.relevance))


def _grounding_queue(nodes: dict[str, Node], dropped: list[DroppedCandidate] = ()) -> tuple[str, int]:
    flagged = sorted((n for n in nodes.values() if n.review), key=lambda n: n.id)
    lines = ["# Grounding review queue", "", "Nodes whose quotes could not be found verbatim in the cited unit (`quote_not_found`) or that carry another review flag.", ""]
    if not flagged:
        lines.append("(none)")
    else:
        lines += ["| node | title | flags | locators |", "|---|---|---|---|"]
        lines += [f"| {cell(n.id)} | {cell(n.title)} | {cell(', '.join(n.review))} | {cell('; '.join(s.locator for s in n.sources))} |" for n in flagged]
    if dropped:
        lines += ["", "## Dropped candidates", "", "Model output that cited no unit of its chunk; nothing was written for these (a node is never written without a locator).", ""]
        lines += ["| title | source | chunk | reason |", "|---|---|---|---|"]
        lines += [f"| {cell(d.title)} | {cell(d.source)} | {cell(d.chunk_id)} | {cell(d.reason)} |" for d in dropped]
    return "\n".join(lines) + "\n", len(flagged) + len(dropped)


def _move_to_processed(cfg: Config, name: str, run_id: str) -> None:
    src = cfg.sandbox.resolve("inbox", name)
    dst = cfg.sandbox.resolve("processed", name)
    if dst.exists() or dst.is_symlink():
        p = Path(name)
        dst = cfg.sandbox.resolve("processed", f"{p.stem}-{run_id}{p.suffix}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


# -------------------------------------------------------------------- run


def run(
    cfg: Config,
    *,
    call_stage: CallStage,
    now: datetime,
    dry_run: bool = False,
    write_same_as: bool | None = None,
    fetch: Callable[[str], str | None] | None = None,
    provider_overridden: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    finished_at: Callable[[], datetime] | None = None,
) -> RunResult:
    """`finished_at` (default: `now`) is read once when the report is built, so real runs record
    their duration while scripted runs stay byte-stable."""
    run_id = run_id_for(now)
    today = now.astimezone(UTC).date().isoformat()
    schema = SCHEMAS[cfg.corpus.schema]
    graph_dir = cfg.sandbox.root("graph")
    inbox, processed = cfg.sandbox.root("inbox"), cfg.sandbox.root("processed")
    ledger = UsageLedger(soft_budget_usd=None if cfg.llm.provider == SUBSCRIPTION_PROVIDER else cfg.limits.soft_budget_usd)
    warnings: list[str] = []
    events: list[dict[str, Any]] = []
    redactor = _Redactor(secret_values(cfg).values())
    moves_before = {"inbox": _count_entries(inbox), "processed": _count_entries(processed)}

    def bound_call_stage(stage: str, system: str, user_text: str, wire: type, images: list | None = None, **kw: Any) -> Any:
        """`call_stage` with this run's cfg/ledger; the converters (describe) pass neither, the stages pass both."""
        kw.setdefault("cfg", cfg)
        kw.setdefault("ledger", ledger)
        if kw.get("ledger") is None:
            kw["ledger"] = ledger
        kw.setdefault("prompt_version", describe_prompt.VERSION if stage == "describe" else None)
        return call_stage(stage, system, user_text, wire, images, **kw)

    def dry_run_call_stage(*_a: Any, **_k: Any) -> Any:
        raise DryRunSkip(f"model call skipped in dry-run (estimated {DRY_RUN_IMAGE_TOKENS} tokens)")

    # S0 + S1 (+ S2)
    outcomes = convert_inbox(cfg, call_stage=dry_run_call_stage if dry_run else bound_call_stage, fetch=fetch)
    rows: dict[str, InputRow] = {}
    chunked: list[tuple[ConversionOutcome, list[Any]]] = []
    for oc in outcomes:
        row = InputRow(name=oc.item.name, status=oc.status, reason=oc.reason, units=len(oc.units))
        if oc.status == "converted":
            chunks = build_chunks(oc.units, cfg.chunking)
            row.chunks = len(chunks)
            chunked.append((oc, chunks))
        if dry_run and oc.status in ("converted", "deferred"):
            row.status = "dry-run"
            if oc.status == "deferred":
                row.reason = f"image not described in dry-run; estimated {DRY_RUN_IMAGE_TOKENS} tokens"
        rows[oc.item.name] = row

    if dry_run:
        return _finish(cfg, run_id=run_id, now=now, finished_at=finished_at, mode=report.DRY_RUN_MODE, rows=list(rows.values()), ledger=ledger, warnings=warnings, events=events, moves_before=moves_before, nodes_written=0, node_counts=None, edges_by_type={}, rejected=[], queues={"duplicates": 0, "grounding": 0, "not_adjudicated": 0}, provider_overridden=provider_overridden, redactor=redactor)

    registry, existing, unreadable = _load_existing(cfg, warnings, redactor)
    linkable: dict[str, Any] = dict(existing)
    works: list[_FileWork] = []
    stop: str | None = None  # set when model stages must not continue (budget, long rate limit)
    waits = 0  # rate-limit waits taken this run; the budget is per run, not per streak (D34)
    waited_until: str | None = None  # what the one rate-limit wait waited for, for the deferral warning

    for index, (oc, chunks) in enumerate(chunked):
        row = rows[oc.item.name]
        if stop is not None:
            row.status, row.reason = "deferred", stop
            continue
        work = _FileWork(outcome=oc)
        while True:
            snapshot = copy.deepcopy(registry)
            try:
                _model_file(work, chunks, cfg, registry, linkable, call_stage=bound_call_stage, ledger=ledger, run_id=run_id)
            except RateLimited as exc:
                registry = snapshot
                events.append({"event": "rate_limit", "file": oc.item.name, "action": exc.action, "rate_limit_type": exc.rate_limit_type, "resets_at": exc.resets_at, "utilization": exc.utilization, "wait_seconds": exc.wait_seconds})
                if exc.action == "wait" and waits < MAX_WAITS_PER_RUN:
                    waits += 1
                    waited_until = _stamp(datetime.fromtimestamp(exc.resets_at, UTC)) if exc.resets_at else f"+{exc.wait_seconds:.0f}s"
                    sleep(exc.wait_seconds)
                    continue  # finished chunks are cached on `work`; only the rest is called again
                # The reset is beyond the cap, or this run has already waited once: defer this and every
                # remaining file, and say so (DESIGN §7.6, D34).
                remaining = len(chunked) - index
                if waited_until is None:
                    warnings.append(f"rate limit: reset beyond the wait cap; {remaining} remaining file(s) deferred")
                else:
                    warnings.append(f"rate limit: waited once until {waited_until}; second limit → {remaining} remaining file(s) deferred")
                events.append({"event": "rate_limit_deferral", "file": oc.item.name, "waited_until": waited_until, "remaining_files": remaining})
                stop = _reason("model", exc, redactor)
                row.status, row.reason = "deferred", stop
                break
            except BudgetExceeded as exc:
                registry = snapshot
                stop = _reason("budget", exc, redactor)
                events.append({"event": "budget_exceeded", "file": oc.item.name, "spent_usd": exc.spent_usd, "soft_budget_usd": exc.soft_budget_usd})
                row.status, row.reason = "deferred", stop
                break
            except Exception as exc:  # any model-boundary failure defers this file only (DESIGN §6)
                registry = snapshot
                row.status, row.reason = "deferred", _reason(work.stage, exc, redactor)
                events.append({"event": "deferred", "file": oc.item.name, "reason": row.reason})
                break
            works.append(work)
            linkable.update({d.id: d for d in work.drafts})
            warnings.extend(work.warnings)
            row.status = "ingested"
            break

    # Drafts and edges that touch an active row whose note could not be parsed are skipped, never written over it.
    # Likewise a new node whose file name is already taken by a file this run did not read: no note, no row, no edges.
    drafts: list[NodeDraft] = []
    collided: set[str] = set()
    for w in works:
        for d in w.drafts:
            where = f"{d.id} ({d.title!r} from {', '.join(d.source_files)})"
            if d.id in unreadable:
                warnings.append(f"{where}: existing note is unreadable; not updated this run")
            elif d.id in collided:
                continue  # already refused for an earlier file's draft of the same node
            elif d.id not in existing and _note_path_taken(cfg, registry, d.id):
                warnings.append(f"{registry.file_for(d.id)}: a file already exists there that was not read this run; {where} not written")
                collided.add(d.id)
                if d.is_new:
                    registry.discard(d.id, run=run_id)  # the id stays burnt; no row without its note
            else:
                drafts.append(d)
    proposals: list[Edge] = []
    dropped = [r for w in works for r in w.dropped]
    for e in (e for w in works for e in w.edges):
        if e.source_id in unreadable or e.target_id in unreadable:
            dropped.append(Rejection(UNREADABLE_GATE, f"{e.type} {e.source_id} -> {e.target_id}: the existing note is unreadable and is not touched this run", e))
        elif e.source_id in collided or e.target_id in collided:
            dropped.append(Rejection(COLLISION_GATE, f"{e.type} {e.source_id} -> {e.target_id}: the node's file name is taken by a file this run did not read; the node was not written", e))
        else:
            proposals.append(e)
    dropped_candidates = [c for w in works for c in w.dropped_candidates]
    for c in dropped_candidates:
        warnings.append(f"{c.source} ({c.chunk_id}): candidate {c.title!r} dropped: {c.reason}")

    # S6 dedup: this run's drafts against everything linkable; existing<->existing pairs are not re-adjudicated
    dedup_pool = list({**existing, **{d.id: d for d in drafts}}.values())
    focus = {d.id for d in drafts}
    try:
        dd = s6.run(dedup_pool, cfg, call_stage=bound_call_stage, ledger=ledger, write_same_as=write_same_as, focus=focus) if drafts else s6.DedupResult()
    except Exception as exc:  # dedup is cross-file: a failure queues every pair instead of deferring files
        warnings.append(f"dedup skipped: {_exc_text(exc, redactor)}; candidate pairs listed as not adjudicated")
        dd = s6.DedupResult(overflow=s6.candidate_pairs(dedup_pool, cfg.dedup.threshold, focus=focus), titles={d.id: d.title for d in dedup_pool})

    # S7 gates
    known_ids = set(existing) | {d.id for d in drafts}
    gate_args: dict[str, Any] = {"existing": _existing_edges(existing), "schema": schema, "known_ids": known_ids}
    try:
        # A planted symlink escaping the sandbox (`_review` itself or the log inside it) is refused here, not raised.
        review_dir = cfg.sandbox.resolve("graph", REVIEW_DIR)
        cfg.sandbox.resolve("graph", REJECTED_EDGES_FILE)
        gated: GateResult = gate_edges([*proposals, *dd.edges], review_dir=review_dir if works else None, **gate_args)
    except (OSError, SandboxViolation) as exc:  # the log is a convenience; the gates themselves are pure
        warnings.append(f"{REJECTED_EDGES_FILE} not written: {_exc_text(exc, redactor)}; rejections are listed in the report only")
        gated = gate_edges([*proposals, *dd.edges], review_dir=None, **gate_args)
    rejected = [*dropped, *gated.rejected]

    # S8 write
    to_write: dict[str, Node] = {}
    created = updated = 0
    for d in drafts:
        if d.id in to_write:  # a second file's draft of a node this run already built: union, count once
            to_write[d.id] = _merge_into(to_write[d.id], d, today=today, run_id=run_id)
        elif d.is_new or d.id not in existing:
            to_write[d.id] = _new_node(d, cfg, schema, today=today, run_id=run_id)
            created += 1
        else:
            to_write[d.id] = _merge_into(existing[d.id], d, today=today, run_id=run_id)
            updated += 1
    before_edges = set(to_write)
    _attach_edges(to_write, existing, gated.accepted, schema, today=today, run_id=run_id)
    updated += len(set(to_write) - before_edges)

    all_nodes = {**existing, **to_write}
    targets = {n.id: LinkTarget(stem=Path(registry.file_for(n.id)).stem, title=n.title, aliases=tuple(n.aliases)) for n in all_nodes.values()}
    grounding_text, grounding_n = _grounding_queue(all_nodes, dropped_candidates)
    nodes_written = 0
    edges_by_type = Counter(e.type for e in gated.accepted)
    queues = {"duplicates": len(dd.open_same()) + len(dd.unsure), "grounding": grounding_n, "not_adjudicated": len(dd.overflow)}
    if dd.unrequested:
        warnings.append(f"dedup returned {dd.unrequested} judgement(s) for pairs that were not requested; ignored")
    write_error: str | None = None
    try:
        if works:
            files: dict[Path, str] = {}
            for node in to_write.values():
                path = cfg.sandbox.resolve("graph", registry.file_for(node.id))
                path.parent.mkdir(parents=True, exist_ok=True)
                files[path] = render(node, targets)
            # One atomic batch, registry last: a crash can leave a note without its row, never a row without its note.
            files[cfg.sandbox.resolve("graph", REGISTRY_FILE)] = registry.dump()
            write_files_atomically(files)
            nodes_written = len(files) - 1
        if works or existing:
            graph_io.write_index(graph_io.load_graph(graph_dir), cfg.sandbox.resolve("graph", INDEX_FILE))
            review_dir = cfg.sandbox.resolve("graph", REVIEW_DIR)  # `_review` escaping the sandbox is a write failure with a report, not a crash
            if dd.same or dd.unsure or dd.different or dd.overflow:  # a run without pairs leaves the latest queue alone
                for name in (f"duplicates-{run_id}.md", s6.QUEUE_FILE):
                    cfg.sandbox.resolve("graph", f"{REVIEW_DIR}/{name}")  # a planted symlink escaping the sandbox is refused here
                    s6.write_queue(dd, review_dir, name=name)
            review_dir.mkdir(parents=True, exist_ok=True)
            write_files_atomically({cfg.sandbox.resolve("graph", f"{REVIEW_DIR}/{GROUNDING_FILE}"): grounding_text})
    except (OSError, SandboxViolation) as exc:
        # Nothing landed (the batch is atomic) or the queues did not: the report must still be written (R15),
        # every modelled file stays in the inbox, and the failure is raised once the report exists.
        write_error = _reason("write", exc, redactor)
        events.append({"event": "write_failed", "reason": write_error})
        warnings.append(f"{write_error}; {len(works)} modelled file(s) left in the inbox, nothing moved")
        # Past the atomic batch the notes are on disk (index or queues failed): the row says so, so the reader
        # does not take `deferred` to mean nothing was written.
        row_reason = write_error if nodes_written == 0 else _cap(f"{write_error} (notes landed; index/queues failed)")
        for w in works:
            rows[w.outcome.item.name].status, rows[w.outcome.item.name].reason = "deferred", row_reason
        if nodes_written == 0:
            created = updated = 0
            edges_by_type = Counter()
    else:
        for w in works:  # sources move only after their nodes are on disk (R5)
            if w.outcome.item.path is None:
                continue
            name = w.outcome.item.name
            try:
                _move_to_processed(cfg, name, run_id)
            except (OSError, SandboxViolation) as exc:
                # The notes are on disk but this source is not in `processed/`: the report must still be written
                # (DESIGN §14) naming the file, and the run fails once it exists. The other sources still move.
                reason = _reason("move", exc, redactor)
                write_error = write_error or reason
                events.append({"event": "move_failed", "file": name, "reason": reason})
                warnings.append(f"{name}: {reason}; its nodes are written but the source stays in the inbox")
                rows[name].status, rows[name].reason = "failed", reason

    result = _finish(
        cfg,
        run_id=run_id,
        now=now,
        finished_at=finished_at,
        mode="ingest",
        rows=list(rows.values()),
        ledger=ledger,
        warnings=warnings,
        events=events,
        moves_before=moves_before,
        nodes_written=nodes_written,
        node_counts={"created": created, "updated": updated, "total": len(registry.active())},
        edges_by_type=dict(edges_by_type),
        rejected=[r.to_dict() for r in rejected],
        queues=queues,
        provider_overridden=provider_overridden,
        redactor=redactor,
        status="failed" if write_error is not None else None,  # any S8 failure is a failed run (DESIGN §14)
    )
    if write_error is not None:
        raise PipelineError(f"{write_error}; report written to {result.report_md}")
    return result


def _note_path_taken(cfg: Config, registry: Registry, node_id: str) -> bool:
    """True when the registry's file for `node_id` already exists (or is a symlink) on disk.

    A path the sandbox refuses (a symlink escaping the project) counts as taken: something
    this run did not read sits there, and the note is never written over it.
    """
    try:
        path = cfg.sandbox.resolve("graph", registry.file_for(node_id))
    except (OSError, SandboxViolation):
        return True
    return path.exists() or path.is_symlink()


def _finish(
    cfg: Config,
    *,
    run_id: str,
    now: datetime,
    finished_at: Callable[[], datetime] | None,
    mode: str,
    rows: list[InputRow],
    ledger: UsageLedger,
    warnings: list[str],
    events: list[dict[str, Any]],
    moves_before: dict[str, int],
    nodes_written: int,
    node_counts: dict[str, int] | None,
    edges_by_type: dict[str, int],
    rejected: list[dict[str, Any]],
    queues: dict[str, int],
    provider_overridden: bool,
    redactor: _Redactor | None = None,
    status: str | None = None,
) -> RunResult:
    if redactor is not None and redactor.hits:
        # A credential appeared in a failure message and was replaced before it reached any reason;
        # as with `report.redact`, that is a failed run for a human to look at.
        warnings = [*warnings, f"a failure message contained a credential value; it was replaced by {REDACTED}"]
        status = "failed"
    if node_counts is None:
        reg_path = cfg.sandbox.resolve("graph", REGISTRY_FILE)
        total = len(Registry.load(reg_path).active()) if reg_path.is_file() else 0
        node_counts = {"created": 0, "updated": 0, "total": total}
    moves_after = {"inbox": _count_entries(cfg.sandbox.root("inbox")), "processed": _count_entries(cfg.sandbox.root("processed"))}
    rep = report.build(
        cfg,
        run_id=run_id,
        started=_stamp(now),
        finished=_stamp(finished_at() if finished_at is not None else now),
        mode=mode,
        inputs=[r.to_dict() for r in rows],
        nodes=node_counts,
        edges_by_type=edges_by_type,
        rejected_edges=rejected,
        review_queues=queues,
        ledger=ledger,
        prompt_versions=prompt_versions(),
        warnings=warnings,
        events=events,
        moves={"before": moves_before, "after": moves_after},
        provider_overridden=provider_overridden,
        status=status,
    )
    try:
        md, js = report.write(rep, cfg)
    except report.ReportRedactionError:
        # Notes and moves are already on disk, so a report must still exist (R15): the
        # offending values are replaced and the run is marked failed for a human to look at.
        rep = report.redact(rep, cfg)
        md, js = report.write(rep, cfg)
    return RunResult(run_id=run_id, status=rep.status, inputs=rows, nodes_written=nodes_written, report_md=md, report_json=js)


__all__ = [
    "DRY_RUN_IMAGE_TOKENS",
    "DryRunSkip",
    "GROUNDING_FILE",
    "INDEX_FILE",
    "InputRow",
    "PipelineError",
    "REGISTRY_FILE",
    "REJECTED_EDGES_FILE",
    "REVIEW_DIR",
    "RegistryError",
    "RunResult",
    "run",
    "run_id_for",
]
