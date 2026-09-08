"""Import a knowledge-graph-poc `_index.json` into socratic-tutor-poc, two ways:

1. **RAG content** — each extracted note's definition becomes a retrievable
   `Chunk`, embedded and upserted into the SAME Chroma collection as everything
   else for `--rag-course-id` (default: the course the SPA already uses). This
   is purely additive: it never touches existing chunks.

2. **KL Map draft** — nodes/edges become a `KLMap`, written to
   `content/drafts/klmap-<--klmap-course-id>.yaml`, exactly where the app's own
   extraction pipeline writes drafts — so it's immediately visible to
   `GET /api/klmap/drafts` and goes through the same revalidate/approve flow.

   Deliberately writes to `--klmap-course-id` (default: a distinct id), NOT
   `--rag-course-id` — if a KL Map draft already exists for the RAG course
   (e.g. from the app's own extraction), overwriting it here would destroy
   independently-reviewed work. Pass matching ids explicitly if you want this
   to replace an existing draft (only if it hasn't been approved and edited).

Relation vocabulary: knowledge-graph-poc's `general` schema uses
`related_to | part_of | same_as`; socratic-tutor-poc's `Relation` enum has no
`same_as` (a dedup signal there, not a real relation) — same_as edges are
skipped, not silently dropped: they're reported so you know what didn't cross
over. `related_to`/`part_of` map 1:1.

**Bidirectional conflicts**: socratic-tutor-poc requires exactly one declared
direction per relation between two concepts (KL008); knowledge-graph-poc's own
gates don't enforce that, so a document can legitimately produce both
directions across separate extraction passes. This script keeps whichever
direction it saw first and reports every pair it had to pick for — symmetric
relations (`related_to`) lose no meaning either way, but `part_of`/`uses` are
directional judgment calls a human should confirm.

Usage:
    uv run python scripts/import_from_knowledge_graph_poc.py \\
        --index ../knowledge-graph-poc/data/graph/_index.json \\
        --rag-course-id MATH-SEED-01 \\
        --klmap-course-id GE013-KGPOC-W1 \\
        --klmap-course-name "GE013 (kg-mapper-poc extraction demo)"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from socratic_tutor.config import get_settings  # noqa: E402
from socratic_tutor.domain.klmap.loader import load_klmap_with_report, save_klmap  # noqa: E402
from socratic_tutor.domain.klmap.schema import KLEdge, KLMap, KLNode  # noqa: E402
from socratic_tutor.domain.rag.chunker import Chunk  # noqa: E402
from socratic_tutor.domain.rag.kl_extract import draft_path  # noqa: E402
from socratic_tutor.domain.rag.store import upsert_chunks  # noqa: E402
from socratic_tutor.models.enums import Relation  # noqa: E402

_THAI_CHARS = re.compile(r"[฀-๿]")

#: knowledge-graph-poc `general`-schema edge type -> socratic-tutor-poc Relation.
#: `same_as` has no equivalent (it's a dedup signal there, not a taught relation).
RELATION_MAP: dict[str, Relation] = {
    "related_to": Relation.RELATED_TO,
    "part_of": Relation.PART_OF,
    "uses": Relation.USES,
}
#: Asymmetric relations where "keep whichever direction we saw first" is a real
#: judgment call, not mechanical de-duplication (unlike symmetric related_to).
_DIRECTIONAL = {Relation.PART_OF, Relation.USES, Relation.PREREQUISITE_OF, Relation.NEXT_TOPIC}


def _lang(text: str) -> str:
    return "th" if _THAI_CHARS.search(text) else "en"


def _name_th(aliases: list[str]) -> str | None:
    for alias in aliases:
        if _THAI_CHARS.search(alias):
            return alias
    return None


def import_rag(index: dict, course_id: str) -> int:
    chunks = []
    for ordinal, node in enumerate(index["nodes"]):
        text = (node.get("definition_plain") or "").strip()
        if not text:
            continue
        sources = node.get("sources") or []
        locator = sources[0]["locator"] if sources else node["title"]
        chunks.append(
            Chunk(
                text=text,
                source_ref=f"kg-mapper-poc / {node['title']} ({locator})",
                lang=_lang(text),
                topic=node["title"],
                chunk_id=f"kgpoc-{node['id']}",
                ordinal=ordinal,
            )
        )
    return upsert_chunks(course_id, chunks)


def import_klmap(index: dict, course_id: str, course_name: str) -> tuple[KLMap, list[dict], list[tuple]]:
    nodes = [
        KLNode(id=n["id"], name=n["title"], name_th=_name_th(n.get("aliases") or []))
        for n in index["nodes"]
    ]

    seen: dict[tuple, dict] = {}
    edges: list[KLEdge] = []
    unmapped: list[dict] = []
    conflicts: list[tuple] = []
    for e in index["edges"]:
        relation = RELATION_MAP.get(e["type"])
        if relation is None:
            unmapped.append(e)
            continue
        key = (relation, frozenset({e["source"], e["target"]}))
        if key in seen:
            if relation in _DIRECTIONAL:
                conflicts.append((seen[key], e))
            continue
        seen[key] = e
        edges.append(KLEdge(from_=e["source"], to=e["target"], relation=relation))

    return KLMap(course_id=course_id, course_name=course_name, nodes=nodes, edges=edges), unmapped, conflicts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, help="path to knowledge-graph-poc's data/graph/_index.json")
    ap.add_argument("--rag-course-id", default="MATH-SEED-01")
    ap.add_argument("--klmap-course-id", default="GE013-KGPOC-W1")
    ap.add_argument("--klmap-course-name", default="GE013 (kg-mapper-poc extraction demo)")
    ap.add_argument("--skip-rag", action="store_true")
    ap.add_argument("--skip-klmap", action="store_true")
    args = ap.parse_args()

    index = json.loads(Path(args.index).read_text(encoding="utf-8"))
    print(f"Loaded {len(index['nodes'])} node(s), {len(index['edges'])} edge(s) from {args.index}")

    if not args.skip_rag:
        n = import_rag(index, args.rag_course_id)
        print(f"\nRAG: upserted {n} chunk(s) into course_id={args.rag_course_id!r}")

    if not args.skip_klmap:
        klmap, unmapped, conflicts = import_klmap(index, args.klmap_course_id, args.klmap_course_name)
        print(f"\nKL Map: {len(klmap.nodes)} node(s), {len(klmap.edges)} edge(s) for course_id={args.klmap_course_id!r}")
        if unmapped:
            print(f"  skipped {len(unmapped)} edge(s) with no Relation equivalent (e.g. same_as): {unmapped}")
        if conflicts:
            print(f"  {len(conflicts)} directional conflict(s) resolved by keeping the first-seen direction — review these:")
            for kept, dropped in conflicts:
                print(f"    kept:    {kept['source']} -{kept['type']}-> {kept['target']}")
                print(f"    dropped: {dropped['source']} -{dropped['type']}-> {dropped['target']}")

        settings = get_settings()
        path = draft_path(args.klmap_course_id, settings.content_dir / "drafts")
        if path.is_file():
            print(f"\n  WARNING: {path} already exists and will be overwritten.")
        path.parent.mkdir(parents=True, exist_ok=True)
        save_klmap(klmap, path)
        print(f"  wrote draft to {path}")

        _, report = load_klmap_with_report(path)
        print(f"  validation: {len(report.errors)} error(s), {len(report.warnings)} warning(s)")
        if report.errors:
            print("  (fix these before this draft can be approved)")


if __name__ == "__main__":
    main()
