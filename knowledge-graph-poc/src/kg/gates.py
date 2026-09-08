"""Deterministic gates (DESIGN §10; PRD R7, R8, R10, R13; D29).

Every gate returns a ``Rejection`` (with the gate's name and a human-readable
reason) or ``None``; nothing here ever writes a note or deletes a file. Rejections
from ``gate_edges`` are appended, one JSON line each, to
``<graph>/_review/rejected-edges.jsonl``. ``validate_graph`` is what ``kg gates``
runs: it reads the notes on disk and reports; no model, no mutation.
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kg.notes import Node, NoteError, parse
from kg.paths import SandboxViolation, is_denied
from kg.schemas import EdgeSchema

REJECTED_EDGES_FILE = "_review/rejected-edges.jsonl"
REGISTRY_FILE = "_registry.yaml"
NODES_DIR = "nodes"


@dataclass(frozen=True)
class Edge:
    type: str
    source_id: str
    target_id: str
    relevance: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "source_id": self.source_id, "target_id": self.target_id, "relevance": self.relevance}


@dataclass(frozen=True)
class Rejection:
    gate: str
    reason: str
    edge: Any

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {"gate": self.gate, "reason": self.reason}
        if self.edge is not None:
            row.update(
                type=getattr(self.edge, "type", None),
                source_id=getattr(self.edge, "source_id", None),
                target_id=getattr(self.edge, "target_id", None),
                relevance=getattr(self.edge, "relevance", None),
            )
        return row


@dataclass
class GateResult:
    accepted: list[Edge]
    rejected: list[Rejection]


# ----------------------------------------------------------------- per-edge


def out_of_schema(edge: Any, schema: EdgeSchema) -> Rejection | None:
    if edge.type not in schema.edges:
        return Rejection("out_of_schema", f"edge type {edge.type!r} is not in the {schema.name} schema (allowed: {', '.join(schema.edges)})", edge)
    return None


def relevance(edge: Any, schema: EdgeSchema) -> Rejection | None:
    """`related_to` needs an integer 0–100; nothing else may carry a score (R10)."""
    score = getattr(edge, "relevance", None)
    scored = edge.type in schema.edges and schema.is_scored(edge.type)
    if scored:
        if score is None:
            return Rejection("relevance", f"{edge.type} without relevance (an integer 0-100 is required)", edge)
        if isinstance(score, bool) or not isinstance(score, int):
            return Rejection("relevance", f"{edge.type} relevance {score!r} is not an integer", edge)
        if not 0 <= score <= 100:
            return Rejection("relevance", f"{edge.type} relevance {score} is out of range 0-100", edge)
        return None
    if score is not None:
        return Rejection("relevance", f"{edge.type} carries a relevance ({score!r}); only scored types may", edge)
    return None


def self_edge(edge: Any) -> Rejection | None:
    if edge.source_id == edge.target_id:
        return Rejection("self_edge", f"{edge.type} from {edge.source_id} to itself", edge)
    return None


def dangling(edge: Any, known_ids: set[str]) -> Rejection | None:
    missing = [i for i in (edge.source_id, edge.target_id) if i not in known_ids]
    if missing:
        return Rejection("dangling", f"{edge.type} {edge.source_id} -> {edge.target_id}: unknown id(s) {', '.join(missing)}", edge)
    return None


def pair_key(edge: Any, schema: EdgeSchema) -> tuple[str, str, str]:
    """Canonical key: symmetric types collapse both directions (DESIGN §8.2)."""
    a, b = edge.source_id, edge.target_id
    if edge.type in schema.edges and schema.is_symmetric(edge.type):
        a, b = min(a, b), max(a, b)
    return (edge.type, a, b)


def dag(edge: Any, existing: list[Any], schema: EdgeSchema) -> Rejection | None:
    """Reject `edge` if adding it would close a cycle among edges of its (DAG) type; reason names the path."""
    if edge.type not in schema.edges or not schema.is_dag(edge.type):
        return None
    adjacency: dict[str, list[str]] = {}
    for e in existing:
        if e.type == edge.type:
            adjacency.setdefault(e.source_id, []).append(e.target_id)
    # A cycle appears iff target already reaches source.
    parents: dict[str, str | None] = {edge.target_id: None}
    queue: deque[str] = deque([edge.target_id])
    while queue:
        node = queue.popleft()
        if node == edge.source_id:
            path = [node]
            while parents[path[-1]] is not None:
                path.append(parents[path[-1]])  # type: ignore[arg-type]
            path.reverse()  # target ... source
            cycle = " -> ".join([*path, edge.target_id])
            return Rejection("dag", f"{edge.type} {edge.source_id} -> {edge.target_id} would close a cycle: {cycle}", edge)
        for nxt in adjacency.get(node, ()):
            if nxt not in parents:
                parents[nxt] = node
                queue.append(nxt)
    return None


# ---------------------------------------------------------------- batches


#: Append-only, never through a symlink: a planted `rejected-edges.jsonl -> <elsewhere>` is an OSError, not a write.
_APPEND_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise OSError(f"{path}: refusing to append through a symlink")
    fd = os.open(path, _APPEND_FLAGS, 0o644)
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def gate_edges(
    proposals: list[Any],
    *,
    existing: list[Any],
    schema: EdgeSchema,
    review_dir: Path | None = None,
    known_ids: set[str] | None = None,
) -> GateResult:
    """Run every edge gate incrementally over `proposals` given the edges already on disk.

    Order per edge: out_of_schema, relevance, self_edge, dangling (only when
    `known_ids` is given), duplicate_edge (against existing + accepted so far,
    symmetric pairs collapsed), dag. The first failing gate wins.
    """
    accepted: list[Edge] = []
    rejected: list[Rejection] = []
    seen = {pair_key(e, schema) for e in existing}
    for p in proposals:
        edge = Edge(type=p.type, source_id=p.source_id, target_id=p.target_id, relevance=getattr(p, "relevance", None))
        rej = out_of_schema(edge, schema) or relevance(edge, schema) or self_edge(edge)
        if rej is None and known_ids is not None:
            rej = dangling(edge, known_ids)
        if rej is None:
            key = pair_key(edge, schema)
            if key in seen:
                rej = Rejection("duplicate_edge", f"{edge.type} {edge.source_id} -> {edge.target_id} already exists", edge)
        if rej is None:
            rej = dag(edge, [*existing, *accepted], schema)
        if rej is None:
            accepted.append(edge)
            seen.add(pair_key(edge, schema))
        else:
            rejected.append(rej)
    if rejected and review_dir is not None:
        _append_jsonl(Path(review_dir) / Path(REJECTED_EDGES_FILE).name, [r.to_dict() for r in rejected])
    return GateResult(accepted=accepted, rejected=rejected)


def mirror(nodes: list[Node], schema: EdgeSchema) -> list[Rejection]:
    """Symmetric edges must appear on both nodes with the same relevance (one finding per pair)."""
    by_id = {n.id: n for n in nodes}
    findings: list[Rejection] = []
    checked: set[tuple[str, str, str]] = set()
    for n in nodes:
        for e in n.edges:
            if e.type not in schema.edges or not schema.is_symmetric(e.type) or e.target not in by_id:
                continue
            key = (e.type, min(n.id, e.target), max(n.id, e.target))
            if key in checked:
                continue
            checked.add(key)
            other = by_id[e.target]
            back = [b for b in other.edges if b.type == e.type and b.target == n.id]
            edge = Edge(e.type, n.id, e.target, e.relevance)
            if not back:
                findings.append(Rejection("mirror", f"{e.type} {n.id} -> {e.target} has no mirror on {e.target}", edge))
            elif back[0].relevance != e.relevance:
                findings.append(
                    Rejection("mirror", f"{e.type} {n.id} <-> {e.target}: relevance {e.relevance} on {n.id} but {back[0].relevance} on {e.target}", edge)
                )
    return findings


# ------------------------------------------------------------ whole graph


def load_notes(graph_dir: Path) -> tuple[dict[str, Node], dict[str, str], list[Rejection]]:
    """Parse every ``nodes/*.md``; returns (nodes by id, id -> relative file, note_schema findings)."""
    nodes_dir = Path(graph_dir) / NODES_DIR
    nodes: dict[str, Node] = {}
    files: dict[str, str] = {}
    findings: list[Rejection] = []
    if not nodes_dir.is_dir():
        return nodes, files, findings
    for path in sorted(nodes_dir.glob("*.md")):
        rel = f"{NODES_DIR}/{path.name}"
        if path.is_symlink():
            findings.append(Rejection("note_schema", f"{rel}: symlinks are not read", None))
            continue
        try:
            node = parse(path.read_text(encoding="utf-8"))
        except (NoteError, OSError, UnicodeDecodeError) as exc:
            findings.append(Rejection("note_schema", f"{rel}: {exc}", None))
            continue
        if node.id in nodes:
            findings.append(Rejection("note_schema", f"{rel}: duplicate id {node.id} (also in {files[node.id]})", None))
            continue
        nodes[node.id] = node
        files[node.id] = rel
    return nodes, files, findings


def _registry_findings(graph_dir: Path, files: dict[str, str]) -> list[Rejection]:
    reg_path = Path(graph_dir) / REGISTRY_FILE
    if not reg_path.is_file():
        return [Rejection("registry", f"{REGISTRY_FILE} is missing but {len(files)} note(s) exist", None)] if files else []
    try:
        doc = yaml.safe_load(reg_path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError) as exc:
        return [Rejection("registry", f"{REGISTRY_FILE} unreadable: {exc}", None)]
    rows = doc.get("nodes", []) if isinstance(doc, dict) else []
    findings: list[Rejection] = []
    active: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict) or "id" not in row:
            findings.append(Rejection("registry", f"{REGISTRY_FILE}: malformed row {row!r}", None))
            continue
        if row.get("status", "active") == "active":
            active[str(row["id"])] = str(row.get("file", ""))
    for node_id, file in active.items():
        if node_id not in files:
            findings.append(Rejection("registry", f"{node_id} is active in {REGISTRY_FILE} but {file or 'its note'} is not on disk", None))
        elif file != files[node_id]:
            findings.append(Rejection("registry", f"{node_id}: registry says {file}, note is at {files[node_id]}", None))
    for node_id, file in files.items():
        if node_id not in active:
            findings.append(Rejection("registry", f"{file} ({node_id}) is not an active row in {REGISTRY_FILE}", None))
    return findings


def validate_graph(graph_dir: Path, schema: EdgeSchema) -> list[Rejection]:
    """`kg gates`: every finding for the graph on disk. Reads only; never repairs or deletes."""
    graph_dir = Path(graph_dir)
    if is_denied(graph_dir):
        raise SandboxViolation(f"graph directory lies in a forbidden location: {graph_dir}")
    nodes, files, findings = load_notes(graph_dir)
    known = set(nodes)
    for node in nodes.values():
        if node.schema != schema.name:
            findings.append(Rejection("note_schema", f"{files[node.id]}: schema {node.schema!r} but the run schema is {schema.name!r}", None))
        if schema.origin_values is None and node.origin is not None:
            findings.append(Rejection("note_schema", f"{files[node.id]}: origin is not allowed in a {schema.name} run", None))
        if schema.origin_values is not None and node.origin not in schema.origin_values:
            findings.append(Rejection("note_schema", f"{files[node.id]}: origin must be one of {', '.join(schema.origin_values)}", None))
    accepted: list[Edge] = []
    seen: set[tuple[str, str, str]] = set()
    for node in nodes.values():
        for e in node.edges:
            edge = Edge(e.type, node.id, e.target, e.relevance)
            rej = out_of_schema(edge, schema) or relevance(edge, schema) or self_edge(edge) or dangling(edge, known)
            if rej is None:
                key = pair_key(edge, schema)
                symmetric = edge.type in schema.edges and schema.is_symmetric(edge.type)
                if key in seen and not symmetric:
                    rej = Rejection("duplicate_edge", f"{edge.type} {edge.source_id} -> {edge.target_id} listed twice", edge)
                seen.add(key)
            if rej is None:
                rej = dag(edge, accepted, schema)
            if rej is None:
                accepted.append(edge)
            else:
                findings.append(rej)
    findings.extend(mirror(list(nodes.values()), schema))
    findings.extend(_registry_findings(graph_dir, files))
    return findings


__all__ = [
    "Edge",
    "GateResult",
    "NODES_DIR",
    "REGISTRY_FILE",
    "REJECTED_EDGES_FILE",
    "Rejection",
    "dag",
    "dangling",
    "gate_edges",
    "load_notes",
    "mirror",
    "out_of_schema",
    "pair_key",
    "relevance",
    "self_edge",
    "validate_graph",
]
