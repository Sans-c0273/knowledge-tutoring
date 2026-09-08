"""Load the graph on disk into memory and derive ``_index.json`` (DESIGN §8.6, §20, §23; R14).

Notes are the source of truth; a stale ``_index.json`` is never read. Symmetric
edges are de-duplicated to one ``GraphEdge`` (``source = min id``); if the two
endpoints disagree on `relevance`, the value written on the ``min id`` note wins. A missing or
corrupt note is a ``Problem``, never fatal. Nothing outside ``graph_dir`` is read,
and a note that is a symlink is refused (§23).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from kg.convert._common import write_files_atomically
from kg.notes import Node, NoteError, parse
from kg.paths import SandboxViolation, is_denied
from kg.schemas import SCHEMAS

NODES_DIR = "nodes"
REGISTRY_FILE = "_registry.yaml"
INDEX_FILE = "_index.json"


@dataclass
class GraphNode:
    id: str
    title: str
    aliases: list[str]
    file: str
    definition_plain: str
    origin: str | None
    provider: str
    model: str
    sources: list[dict[str, str]]
    degree: int = 0

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "aliases": list(self.aliases),
            "file": self.file,
            "definition_plain": self.definition_plain,
            "provider": self.provider,
            "model": self.model,
            "sources": [dict(s) for s in self.sources],
            "degree": self.degree,
        }
        if self.origin is not None:
            out["origin"] = self.origin
        return out


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    type: str
    relevance: int | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"source": self.source, "target": self.target, "type": self.type}
        if self.relevance is not None:
            out["relevance"] = self.relevance
        return out


@dataclass(frozen=True)
class Problem:
    file: str
    reason: str


@dataclass
class Graph:
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    problems: list[Problem] = field(default_factory=list)

    def to_index(self) -> dict[str, Any]:
        return {
            "nodes": [self.nodes[i].to_dict() for i in sorted(self.nodes)],
            "edges": [e.to_dict() for e in self.edges],
            "meta": dict(self.meta),
        }


def _read_registry(graph_dir: Path, problems: list[Problem]) -> dict[str, Any] | None:
    path = graph_dir / REGISTRY_FILE
    if not path.is_file():
        return None
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError) as exc:
        problems.append(Problem(REGISTRY_FILE, f"unreadable: {exc}"))
        return None
    return doc if isinstance(doc, dict) else None


def _graph_node(node: Node, file: str) -> GraphNode:
    return GraphNode(
        id=node.id,
        title=node.title,
        aliases=list(node.aliases),
        file=file,
        definition_plain=node.definition,
        origin=node.origin,
        provider=node.provider,
        model=node.model,
        sources=[{"locator": s.locator, "file": s.file, "quote": s.quote} for s in node.sources],
    )


def load_graph(graph_dir: Path) -> Graph:
    graph_dir = Path(graph_dir)
    if is_denied(graph_dir):
        raise SandboxViolation(f"graph directory lies in a forbidden location: {graph_dir}")
    if not graph_dir.is_dir():
        raise FileNotFoundError(f"graph directory not found: {graph_dir}")
    graph = Graph()
    registry = _read_registry(graph_dir, graph.problems)

    raw: dict[str, Node] = {}
    nodes_dir = graph_dir / NODES_DIR
    for path in sorted(nodes_dir.glob("*.md")) if nodes_dir.is_dir() else []:
        rel = f"{NODES_DIR}/{path.name}"
        if path.is_symlink():
            graph.problems.append(Problem(rel, "symlinked note refused (points outside the graph folder or is not a regular file)"))
            continue
        try:
            node = parse(path.read_text(encoding="utf-8"))
        except (NoteError, OSError, UnicodeDecodeError) as exc:
            graph.problems.append(Problem(rel, f"unreadable note: {exc}"))
            continue
        if node.id in raw:
            graph.problems.append(Problem(rel, f"duplicate id {node.id}"))
            continue
        raw[node.id] = node
        graph.nodes[node.id] = _graph_node(node, rel)

    def mentioned(node_id: str) -> bool:
        """True if some problem already names this node (by file stem or reason) — one problem per lost note."""
        return any(node_id in p.file or node_id in p.reason for p in graph.problems)

    if registry is not None:
        for row in registry.get("nodes", []) or []:
            if not isinstance(row, dict) or row.get("status", "active") != "active":
                continue
            row_id = str(row.get("id"))
            if row_id not in graph.nodes and not mentioned(row_id):
                graph.problems.append(Problem(str(row.get("file") or row_id), f"registry lists active node {row_id} but no readable note was found"))

    # Schema: the registry's word if it has one, else the notes' majority; None for an empty graph.
    schema_name = (registry or {}).get("schema")
    if not schema_name and raw:
        schema_name = Counter(n.schema for n in raw.values()).most_common(1)[0][0]
    schema = SCHEMAS.get(str(schema_name)) if schema_name else None
    seen: set[tuple[str, str, str]] = set()
    neighbours: dict[str, set[str]] = {i: set() for i in graph.nodes}
    dangling = 0
    # Notes are visited in ascending id order so that, when both endpoints list a symmetric edge
    # with different relevance, the copy on the min(id) side is the one kept (first seen wins).
    for node_id in sorted(raw):
        node = raw[node_id]
        for e in node.edges:
            if e.target not in graph.nodes:
                # The edge is dropped; the lost target is reported once (above) unless it is unknown everywhere.
                dangling += 1
                if not mentioned(e.target):
                    graph.problems.append(Problem(graph.nodes[node.id].file, f"{e.type} -> {e.target}: target not loaded"))
                continue
            symmetric = schema is not None and e.type in schema.edges and schema.is_symmetric(e.type)
            a, b = (min(node.id, e.target), max(node.id, e.target)) if symmetric else (node.id, e.target)
            key = (e.type, a, b)
            neighbours[node.id].add(e.target)
            neighbours[e.target].add(node.id)
            if key in seen:
                continue
            seen.add(key)
            graph.edges.append(GraphEdge(source=a, target=b, type=e.type, relevance=e.relevance))
    for node_id, gn in graph.nodes.items():
        gn.degree = len(neighbours[node_id])

    corpus = (registry or {}).get("corpus")
    if not corpus and graph.nodes:
        corpus = next(iter(sorted(graph.nodes))).rsplit("-", 1)[0]
    providers = Counter(n.provider for n in raw.values())
    graph.meta = {
        "schema": str(schema_name) if schema_name else None,
        "corpus": corpus,
        "run": max((n.run for n in raw.values()), default=None),
        "provider": providers.most_common(1)[0][0] if providers else None,
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "dangling_edges": dangling,
        "problems": [{"file": p.file, "reason": p.reason} for p in graph.problems],
    }
    return graph


def write_index(graph: Graph, path: Path) -> Path:
    """Write ``_index.json`` atomically (``.tmp`` + ``os.replace``); a symlink at `path` is refused, never followed (§23)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise OSError(f"{path}: refusing to write the index through a symlink")
    write_files_atomically({path: json.dumps(graph.to_index(), ensure_ascii=False, indent=2) + "\n"})
    return path


__all__ = ["Graph", "GraphEdge", "GraphNode", "INDEX_FILE", "NODES_DIR", "Problem", "REGISTRY_FILE", "load_graph", "write_index"]
