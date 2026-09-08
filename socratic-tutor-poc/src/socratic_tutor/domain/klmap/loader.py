"""Load, validate, and save KL Map YAML files.

Tech Spec §2.3 states four content-build-time rules; this module enforces all of
them and reports *every* violation instead of raising on the first, because the
human-review UI that sits after LLM-assisted extraction has to render the whole
list at once:

1. closed relation vocabulary — `prerequisite_of`, `related_to`, `next_topic`,
   `part_of`, `uses`;
2. one declared direction per relation between any pair of concepts;
3. `prerequisite_of` must form a DAG (cycles are reported with their path);
4. referential integrity — no unknown node ids, no duplicate node ids.

`validate_klmap` returns `(map, report)` and only returns a map when the report
has no errors, so nothing downstream can traverse a map that failed a rule.
`load_klmap` is the runtime convenience wrapper that raises instead.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from itertools import islice
from pathlib import Path
from typing import Any

import networkx as nx
import yaml
from pydantic import BaseModel, Field, ValidationError

from socratic_tutor.domain.klmap.schema import KLEdge, KLMap, KLNode
from socratic_tutor.models.enums import Relation

#: The closed relation vocabulary, in the order the spec lists it.
ALLOWED_RELATIONS: tuple[str, ...] = tuple(relation.value for relation in Relation)

#: Cap on cycles reported per map; a map with more than this is broken enough.
_MAX_REPORTED_CYCLES = 5


class KLMapValidationError(Exception):
    """Raised by `load_klmap` when a map violates a content rule.

    Carries the full `ValidationReport` so a caller that wants the structured
    list (rather than the formatted message) does not have to re-validate.
    """

    def __init__(self, report: ValidationReport, source: str) -> None:
        self.report = report
        self.source = source
        super().__init__(f"KL Map {source} failed validation:\n{report.format()}")


class ValidationIssue(BaseModel):
    """One problem found in a map file.

    `location` points at the offending item in the source file (`edges[7]`,
    `nodes[2]`) so the review UI can highlight it; `message` is written to be
    actionable on its own, because it is also what ends up in logs.

    `path` carries the node ids an issue implicates when there is more than one
    — the cycle for `KL009`, the two ends of a two-way relation for `KL008` — in
    order where the order means something. It exists so the review UI can
    highlight a whole cycle in the graph without parsing ids back out of the
    message prose, which would couple the UI to the exact wording of an error
    string.
    """

    code: str
    message: str
    location: str | None = None
    path: list[str] = Field(default_factory=list)

    def format(self) -> str:
        where = f" at {self.location}" if self.location else ""
        return f"[{self.code}]{where}: {self.message}"


class ValidationReport(BaseModel):
    """Everything wrong (errors) or suspicious (warnings) about one map file.

    Warnings never block loading — an isolated concept is odd but authorable.
    """

    source: str = ""
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the map is loadable. Warnings do not affect this."""
        return not self.errors

    def error(
        self,
        code: str,
        message: str,
        location: str | None = None,
        path: list[str] | None = None,
    ) -> None:
        self.errors.append(
            ValidationIssue(code=code, message=message, location=location, path=list(path or []))
        )

    def warn(self, code: str, message: str, location: str | None = None) -> None:
        self.warnings.append(ValidationIssue(code=code, message=message, location=location))

    def format(self) -> str:
        """Multi-line human-readable rendering, errors first."""
        lines = [f"  ERROR {issue.format()}" for issue in self.errors]
        lines += [f"  WARN  {issue.format()}" for issue in self.warnings]
        return "\n".join(lines) if lines else "  (no issues)"


def validate_klmap(
    data: Mapping[str, Any], source: str = "<memory>"
) -> tuple[KLMap | None, ValidationReport]:
    """Validate raw map data and build a `KLMap` if every rule passes.

    Returns `(None, report)` when `report.ok` is False. Structural problems
    (missing keys, malformed nodes/edges) are collected first; the graph rules
    then run over whatever was well-formed enough to interpret, so one typo does
    not hide a cycle further down the file.
    """
    report = ValidationReport(source=source)

    if not isinstance(data, Mapping):
        report.error(
            "KL000", f"Map file must contain a mapping at the top level, got {type(data).__name__}."
        )
        return None, report

    course_id = data.get("course_id")
    course_name = data.get("course_name")
    for key, value in (("course_id", course_id), ("course_name", course_name)):
        if not isinstance(value, str) or not value.strip():
            report.error("KL001", f"Missing or empty required field '{key}'.", key)

    raw_nodes = data.get("nodes") or []
    raw_edges = data.get("edges") or []
    if not isinstance(raw_nodes, list):
        report.error("KL001", "Field 'nodes' must be a list.", "nodes")
        raw_nodes = []
    if not isinstance(raw_edges, list):
        report.error("KL001", "Field 'edges' must be a list.", "edges")
        raw_edges = []

    nodes = _parse_nodes(raw_nodes, report)
    edges = _parse_edges(raw_edges, set(nodes), report)

    _check_single_direction(edges, nodes, report)
    _check_prerequisite_dag(edges, nodes, report)
    _warn_isolated_nodes(edges, nodes, report)

    if not report.ok:
        return None, report

    klmap = KLMap(
        course_id=str(course_id),
        course_name=str(course_name),
        nodes=list(nodes.values()),
        edges=[edge for _, edge in edges],
    )
    return klmap, report


def load_klmap_with_report(path: str | Path) -> tuple[KLMap | None, ValidationReport]:
    """Read a YAML map file and validate it. Never raises on content problems.

    A missing or unparseable file is reported as an error too, so the review UI
    has one uniform failure channel.
    """
    path = Path(path)
    source = str(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        report = ValidationReport(source=source)
        report.error("KL000", f"Cannot read map file: {exc}.")
        return None, report

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        report = ValidationReport(source=source)
        report.error("KL000", f"Map file is not valid YAML: {exc}.")
        return None, report

    return validate_klmap(data if data is not None else {}, source=source)


def load_klmap(path: str | Path) -> KLMap:
    """Load a map, raising `KLMapValidationError` if it violates any rule.

    Use this on the serving path, where an invalid map is a deploy-time fault.
    """
    klmap, report = load_klmap_with_report(path)
    if klmap is None:
        raise KLMapValidationError(report, str(path))
    return klmap


def save_klmap(klmap: KLMap, path: str | Path) -> Path:
    """Write a map back to YAML, round-trippable by `load_klmap`.

    Keeps the authored key order and emits the `from` alias rather than the
    Python-safe `from_`, so a map extracted by the LLM, corrected in the review
    UI, and saved here stays a file a human can keep editing by hand.

    Written to a sibling temp file and renamed into place. The chat path reads
    the published map keyed by its mtime, so a reader that landed between
    truncate and write would see an empty file, reject it, and could pin that
    verdict to the final file's timestamp (review m1/L2). `os.replace` is atomic
    on the same filesystem: a reader sees the old file or the new one, never a
    torn one, and the mtime belongs to one complete write.
    """
    path = Path(path)
    payload = klmap.model_dump(by_alias=True, exclude_none=True, mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)
    return path


def _parse_nodes(raw_nodes: list[Any], report: ValidationReport) -> dict[str, KLNode]:
    """Build the node index, reporting malformed entries and duplicate ids."""
    nodes: dict[str, KLNode] = {}
    for index, raw in enumerate(raw_nodes):
        location = f"nodes[{index}]"
        if not isinstance(raw, Mapping):
            report.error(
                "KL002", f"Node entry must be a mapping, got {type(raw).__name__}.", location
            )
            continue
        try:
            node = KLNode.model_validate(dict(raw))
        except ValidationError as exc:
            report.error("KL002", f"Malformed node: {_first_pydantic_message(exc)}.", location)
            continue
        if not node.id:
            report.error("KL002", "Node has an empty 'id'.", location)
            continue
        if node.id in nodes:
            report.error(
                "KL003",
                f"Duplicate node id '{node.id}' (already used by '{nodes[node.id].name}', "
                f"now re-declared as '{node.name}'). Node ids must be unique; rename one of them.",
                location,
            )
            continue
        nodes[node.id] = node
    return nodes


def _parse_edges(
    raw_edges: list[Any], known_ids: set[str], report: ValidationReport
) -> list[tuple[str, KLEdge]]:
    """Build the edge list, reporting bad vocabulary and dangling references.

    Returns `(location, edge)` pairs so later graph checks can point back at the
    line the author wrote.
    """
    edges: list[tuple[str, KLEdge]] = []
    seen: set[tuple[str, str, Relation]] = set()

    for index, raw in enumerate(raw_edges):
        location = f"edges[{index}]"
        if not isinstance(raw, Mapping):
            report.error(
                "KL004", f"Edge entry must be a mapping, got {type(raw).__name__}.", location
            )
            continue

        source_id = raw.get("from")
        target_id = raw.get("to")
        relation = raw.get("relation")
        described = f"{source_id!r} -> {target_id!r}"

        if not isinstance(source_id, str) or not isinstance(target_id, str):
            report.error("KL004", "Edge needs string 'from' and 'to' node ids.", location)
            continue

        if not isinstance(relation, str) or relation not in ALLOWED_RELATIONS:
            report.error(
                "KL005",
                f"Edge {described} uses relation {relation!r}, which is not in the closed "
                f"vocabulary. Allowed relations: {', '.join(ALLOWED_RELATIONS)}.",
                location,
            )
            continue

        missing = [ref for ref in (source_id, target_id) if ref not in known_ids]
        if missing:
            report.error(
                "KL006",
                f"Edge {described} ({relation}) references unknown node "
                f"{'ids' if len(missing) > 1 else 'id'} {', '.join(repr(ref) for ref in missing)}. "
                "Add the node or fix the id.",
                location,
            )
            continue

        if source_id == target_id:
            report.error(
                "KL007",
                f"Edge {described} ({relation}) points a node at itself. "
                "Remove it; a concept cannot relate to itself.",
                location,
            )
            continue

        edge = KLEdge(from_=source_id, to=target_id, relation=Relation(relation))
        if edge.triple in seen:
            report.warn(
                "KL100",
                f"Duplicate edge {described} ({relation}); the repeat has no effect.",
                location,
            )
            continue
        seen.add(edge.triple)
        edges.append((location, edge))

    return edges


def _check_single_direction(
    edges: list[tuple[str, KLEdge]], nodes: dict[str, KLNode], report: ValidationReport
) -> None:
    """Reject a relation declared in both directions between the same two nodes.

    Tech Spec §2.3: "one declared direction per relation". `related_to` is read
    symmetrically at lookup time, so declaring it twice is redundant; for the
    directional relations, declaring both ways is a contradiction.
    """
    by_triple = {edge.triple: location for location, edge in edges}
    reported: set[tuple[Relation, str, str]] = set()

    for location, edge in edges:
        mirror = (edge.to, edge.from_, edge.relation)
        if mirror not in by_triple:
            continue
        pair = (edge.relation, *sorted((edge.from_, edge.to)))
        if pair in reported:
            continue
        reported.add(pair)
        report.error(
            "KL008",
            f"Relation '{edge.relation.value}' is declared in both directions between "
            f"{_label(edge.from_, nodes)} and {_label(edge.to, nodes)} "
            f"(also at {by_triple[mirror]}). Each relation gets one declared direction; "
            "keep the correct one and delete the other.",
            location,
            path=[edge.from_, edge.to],
        )


def _check_prerequisite_dag(
    edges: list[tuple[str, KLEdge]], nodes: dict[str, KLNode], report: ValidationReport
) -> None:
    """Reject cycles among `prerequisite_of` edges, naming the cycle path.

    A cycle means each concept is transitively its own prerequisite, which makes
    scaffolding non-terminating.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(
        (edge.from_, edge.to) for _, edge in edges if edge.relation is Relation.PREREQUISITE_OF
    )

    for cycle in islice(nx.simple_cycles(graph), _MAX_REPORTED_CYCLES):
        path = " -> ".join(_label(node_id, nodes) for node_id in [*cycle, cycle[0]])
        report.error(
            "KL009",
            f"'prerequisite_of' edges form a cycle: {path}. Prerequisites must form a DAG; "
            "reverse or remove one edge in that path.",
            "edges",
            path=[*cycle, cycle[0]],
        )


def _warn_isolated_nodes(
    edges: list[tuple[str, KLEdge]], nodes: dict[str, KLNode], report: ValidationReport
) -> None:
    """Flag concepts no edge touches — loadable, but invisible to every lookup."""
    connected = {edge.from_ for _, edge in edges} | {edge.to for _, edge in edges}
    for index, node_id in enumerate(nodes):
        if node_id not in connected:
            report.warn(
                "KL101",
                f"Concept {_label(node_id, nodes)} has no edges, so lookup can never reach it "
                f"from another topic.",
                f"nodes[{index}]",
            )


def _label(node_id: str, nodes: dict[str, KLNode]) -> str:
    """`C003 (Negative Numbers)` — ids alone are unreadable in an error message."""
    node = nodes.get(node_id)
    return f"{node_id} ({node.name})" if node else node_id


def _first_pydantic_message(exc: ValidationError) -> str:
    error = exc.errors()[0]
    field = ".".join(str(part) for part in error["loc"]) or "<root>"
    return f"{field}: {error['msg']}"


__all__ = [
    "ALLOWED_RELATIONS",
    "KLMapValidationError",
    "ValidationIssue",
    "ValidationReport",
    "load_klmap",
    "load_klmap_with_report",
    "save_klmap",
    "validate_klmap",
]
