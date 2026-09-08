"""Stage ``edges`` (S5; R8, R10; DESIGN §6, §8.0; D6): schema-neutral base prompt plus a
vocabulary block generated from the run's ``EdgeSchema`` so prompt and wire type cannot drift."""

from __future__ import annotations

from kg.schemas import EdgeSchema

VERSION = "edges@1"

SYSTEM = """You propose typed relationships between concept notes extracted from a passage of course or
reference material.

You receive:
1. A roster of every note in the graph as "id | title | aliases". Ids are opaque and must be copied
   verbatim; never invent, shorten or renumber an id.
2. The notes extracted from this passage in full (title, definition, grounding quotes).
3. The passage itself, as units with <<UNIT_ID | locator | Heading > Path>> markers.

Propose an edge only when the passage or the definitions give clear support for it. Prefer fewer, well
supported edges over many speculative ones. Each edge names a type from the vocabulary below, a source_id
and a target_id, read as source -> target. Do not propose an edge from a note to itself, and do not
repeat the same pair with the same type. An empty list is a valid answer.

Edge vocabulary for this run:
"""

_DIRECTED_MEANING = {
    "prerequisite_of": "source must be understood before target",
    "part_of": "source is a component of the larger topic target",
    "example_of": "source is a concrete instance of concept target",
    "refines": "source is a more precise or specialised statement of target",
    "supersedes": "source replaces target (a newer version of the idea)",
    "same_as": "source and target are the same idea (duplicate); symmetric",
    "related_to": "source and target are meaningfully connected; symmetric",
}


def vocabulary_block(schema: EdgeSchema) -> str:
    lines = []
    for name, et in schema.edges.items():
        meaning = _DIRECTED_MEANING.get(name, "")
        direction = "A <-> B" if et.symmetric else "A -> B"
        scored = " Give an integer relevance from 0 to 100 stating how strongly the two relate." if et.scored else ""
        lines.append(f"- {name} ({direction}): {meaning}.{scored}")
    if any(et.scored for et in schema.edges.values()):
        unscored = ", ".join(n for n, et in schema.edges.items() if not et.scored)
        lines.append(f"Set relevance to null on every other type ({unscored}); only the scored type carries a number.")
    return "\n".join(lines)


def system_for(schema: EdgeSchema) -> str:
    """The full system prompt for a run: base text + the schema's generated vocabulary block."""
    return SYSTEM + vocabulary_block(schema) + "\n"


TEXT_SHA = "c3169c67918e42ac32463190a54e44ded5d1bcb9e207395fc35b2438ed30403b"
