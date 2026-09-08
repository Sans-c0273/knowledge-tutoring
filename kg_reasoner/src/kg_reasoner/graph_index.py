"""Load a kg-mapper-poc `_index.json` into an in-memory, traversal-ready index.

Deliberately decoupled from the `kg` package: this module only reads the JSON
that `kg render` / `kg ingest` already writes to `data/graph/_index.json`, so
kg_reasoner has no install-time dependency on kg-mapper-poc. If you'd rather
avoid re-implementing title normalisation, install kg-mapper-poc as an editable
dependency (`uv add --editable ../knowledge-graph-poc`) and swap `norm_title`
below for `kg.normalise.norm_title` — same idea, this is just a self-contained
copy so the reasoner can be deployed on its own.

Edge symmetry and edge direction come from `schemas.py` in kg-mapper-poc
(EDUCATION / GENERAL edge tables). `_index.json` does not repeat that
information — a symmetric edge is stored once as (min(source,target), ...) —
so it is re-declared here (`SYMMETRIC_TYPES`) and must be kept in sync if the
source project's schema changes.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Mirrors kg.schemas: which edge types are undirected. Keep in sync with
#: src/kg/schemas.py `symmetric=True` rows in EducationEdgeType / GeneralEdgeType.
SYMMETRIC_TYPES: frozenset[str] = frozenset({"same_as", "related_to"})

#: Edge types that carry a DAG-ordered "must understand before" meaning.
#: Only meaningful when index.schema == "education".
PREREQUISITE_TYPE = "prerequisite_of"

ARTICLES = frozenset({"the", "a", "an"})


def norm_title(text: str) -> str:
    """Casefold, strip punctuation, collapse whitespace, drop a leading article.

    A simplified stand-in for kg.normalise.norm_title (no plural-stripping,
    no Thai/trigram handling) — good enough for anchor matching; swap in the
    real one if you need exact parity with the ingester's dedup behaviour.
    """
    t = unicodedata.normalize("NFKC", text or "").casefold()
    t = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in t)
    tokens = t.split()
    if len(tokens) > 1 and tokens[0] in ARTICLES:
        tokens = tokens[1:]
    return " ".join(tokens)


@dataclass(frozen=True)
class Neighbor:
    node_id: str
    type: str
    relevance: int | None
    #: "out" = this edge was stored source->this node's target direction as-is;
    #: "in" = we are walking a directed edge backwards (e.g. "I am a prerequisite of X").
    direction: str  # "out" | "in" | "sym"


@dataclass
class IndexedNode:
    id: str
    title: str
    aliases: list[str]
    definition_plain: str
    grounded: bool  # has >=1 source with a non-empty quote


@dataclass
class GraphIndex:
    schema: str | None
    nodes: dict[str, IndexedNode] = field(default_factory=dict)
    #: node_id -> outgoing neighbors (edges stored with this node as source)
    _out: dict[str, list[Neighbor]] = field(default_factory=dict)
    #: node_id -> incoming neighbors (edges stored with this node as target)
    _in: dict[str, list[Neighbor]] = field(default_factory=dict)
    #: normalised title/alias -> candidate node ids (for anchor resolution)
    _by_norm_name: dict[str, list[str]] = field(default_factory=dict)

    def neighbors(self, node_id: str, *, types: set[str] | None = None) -> list[Neighbor]:
        """All neighbors of `node_id`, optionally filtered to `types`.

        For an asymmetric edge type this returns BOTH directions (tagged "out"/"in")
        so callers can distinguish "downstream of me" from "upstream of me" — e.g.
        for prerequisite_of, "in" = "this neighbor is a prerequisite of me".
        Symmetric types are tagged "sym" and returned once per direction of travel.
        """
        out = list(self._out.get(node_id, ())) + list(self._in.get(node_id, ()))
        if types is None:
            return out
        return [n for n in out if n.type in types]

    def find_by_title(self, query: str) -> list[str]:
        """Exact (normalised) title/alias match. Empty if none."""
        return list(self._by_norm_name.get(norm_title(query), ()))

    def best_fuzzy_match(self, query: str, *, min_overlap: float = 0.5) -> str | None:
        """Token-Jaccard fallback when no exact title/alias match exists. None below threshold."""
        q_tokens = set(norm_title(query).split())
        if not q_tokens:
            return None
        best_id, best_score = None, 0.0
        for node_id, node in self.nodes.items():
            for name in (node.title, *node.aliases):
                n_tokens = set(norm_title(name).split())
                if not n_tokens:
                    continue
                score = len(q_tokens & n_tokens) / len(q_tokens | n_tokens)
                if score > best_score:
                    best_id, best_score = node_id, score
        return best_id if best_score >= min_overlap else None


def load_index(path: str | Path) -> GraphIndex:
    """Read a kg-mapper-poc `_index.json` file into a `GraphIndex`."""
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return load_index_dict(data)


def load_index_dict(data: dict[str, Any]) -> GraphIndex:
    idx = GraphIndex(schema=(data.get("meta") or {}).get("schema"))

    for raw in data.get("nodes", []):
        sources = raw.get("sources") or []
        grounded = any((s.get("quote") or "").strip() for s in sources)
        node = IndexedNode(
            id=raw["id"],
            title=raw["title"],
            aliases=list(raw.get("aliases") or []),
            definition_plain=raw.get("definition_plain", ""),
            grounded=grounded,
        )
        idx.nodes[node.id] = node
        for name in (node.title, *node.aliases):
            key = norm_title(name)
            if key:
                idx._by_norm_name.setdefault(key, []).append(node.id)

    for raw in data.get("edges", []):
        src, tgt, etype = raw["source"], raw["target"], raw["type"]
        relevance = raw.get("relevance")
        if src not in idx.nodes or tgt not in idx.nodes:
            continue  # dangling edge already reported by the ingester's own gates; skip defensively
        if etype in SYMMETRIC_TYPES:
            idx._out.setdefault(src, []).append(Neighbor(tgt, etype, relevance, "sym"))
            idx._out.setdefault(tgt, []).append(Neighbor(src, etype, relevance, "sym"))
        else:
            idx._out.setdefault(src, []).append(Neighbor(tgt, etype, relevance, "out"))
            idx._in.setdefault(tgt, []).append(Neighbor(src, etype, relevance, "in"))

    return idx


__all__ = ["GraphIndex", "IndexedNode", "Neighbor", "PREREQUISITE_TYPE", "SYMMETRIC_TYPES", "load_index", "load_index_dict", "norm_title"]
