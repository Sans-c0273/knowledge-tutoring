"""Runtime KL Map lookup: topic match, then relation-filtered BFS.

Tech Spec §2.3 lookup algorithm: (1) match the student's topic to a start node;
(2) relation-filtered BFS with `max_depth = 1`, widened to 2 for
`prerequisite_of` when `learner_state = confused`; (3) never traverse
unfiltered. Each relation is walked in its own direction — `prerequisite_of`
backwards (a prerequisite edge points *toward* the concept that needs it),
`next_topic` forwards, the associative relations both ways.

HARD RULE — a prerequisite edge is a hypothesis to check, never a diagnosis.
Everything this module returns describes the *domain*: which concepts the
curriculum places before this one. It says nothing about what this student
knows. A concept appearing in `KnowledgeContext.prerequisites` is a candidate to
*probe* with one question; the tutor scaffolds only after a probe actually
fails. Nothing downstream may read these results as "the student does not
understand X", and no field or message here may be worded to suggest it
(Tech Spec §3.5 hard rule; Architecture A5, strategy
`check_or_scaffold_prerequisite`).
"""

from __future__ import annotations

import re
import unicodedata
from collections import deque
from collections.abc import Callable
from difflib import SequenceMatcher

from socratic_tutor.domain.klmap.schema import KLEdge, KLMap, KLNode
from socratic_tutor.models.enums import LearnerState, Relation
from socratic_tutor.models.knowledge import KnowledgeContext, RelatedTopic

#: BFS depth for prerequisites when the learner is confused (Tech Spec §2.3).
#: Widening applies to `prerequisite_of` only; every other relation stays at
#: `max_depth`, so confusion never floods the prompt with loosely related nodes.
CONFUSED_PREREQUISITE_DEPTH = 2

#: Relations that land in `KnowledgeContext.related_topics`. `KnowledgeContext`
#: has three buckets (Tech Spec §2.3) while the vocabulary has five, so the
#: associative relations collapse together: `related_to` is symmetric by
#: definition, and `part_of` / `uses` are read both ways because a component and
#: its whole are equally usable as an alternative explanation (KM03).
#:
#: The collapse is not lossy: the edge that reached each concept survives on
#: `RelatedTopic.relation`. It has to, because *how* a concept relates changes
#: how a tutor re-explains through it — "photosynthesis **uses** light energy"
#: and "light energy is **part of** photosynthesis" are different lessons.
ASSOCIATIVE_RELATIONS: tuple[Relation, ...] = (
    Relation.RELATED_TO,
    Relation.PART_OF,
    Relation.USES,
)

#: Minimum similarity for the fuzzy fallback in `find_start_node`. High enough
#: that a genuinely different topic returns None rather than a wrong start node.
_FUZZY_CUTOFF = 0.82

_PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+", flags=re.UNICODE)


def find_start_node(klmap: KLMap, topic: str) -> KLNode | None:
    """Match a student-supplied topic string to a node, or return None.

    Tries, in order: exact normalised match on `name` or `name_th`, node id,
    containment either way ("two-step equations" vs "Two-Step Linear
    Equations"), then a similarity fallback for typos. Returns None rather than
    guessing when nothing is close enough — an unmatched topic is a legitimate
    outcome that leaves the turn with RAG context only.
    """
    if not topic or not topic.strip():
        return None

    needle = normalise_text(topic)
    if not needle:
        return None

    for node in klmap.nodes:
        if needle in _normalised_names(node):
            return node

    for node in klmap.nodes:
        if needle == node.id.casefold():
            return node

    contained = _best_match(
        klmap,
        needle,
        accept=lambda name: len(needle) >= 4 and (needle in name or name in needle),
    )
    if contained is not None:
        return contained

    return _best_match(klmap, needle, accept=None, cutoff=_FUZZY_CUTOFF)


def traverse(
    klmap: KLMap,
    start_node: KLNode,
    learner_state: LearnerState = LearnerState.NORMAL,
    max_depth: int = 1,
) -> KnowledgeContext:
    """Relation-filtered BFS around `start_node`, as a `KnowledgeContext`.

    Each bucket is its own filtered walk, so a node can legitimately appear
    under two relations. Distances are shortest-path within that relation.
    Prerequisites widen to depth `CONFUSED_PREREQUISITE_DEPTH` when the learner
    is confused (KM02) and only then; the other relations keep `max_depth` even
    in that case.

    The prerequisites returned are concepts the curriculum places before this
    one — hypotheses to probe, not gaps the student has been shown to have (see
    the module docstring).
    """
    if max_depth < 0:
        raise ValueError(f"max_depth must be >= 0, got {max_depth}")

    prerequisite_depth = max_depth
    if learner_state == LearnerState.CONFUSED:
        prerequisite_depth = max(max_depth, CONFUSED_PREREQUISITE_DEPTH)

    index = klmap.nodes_by_id()
    prerequisites, prerequisite_edges = _walk(
        klmap, start_node, index, {Relation.PREREQUISITE_OF}, prerequisite_depth, backwards=True
    )
    next_topics, next_edges = _walk(
        klmap, start_node, index, {Relation.NEXT_TOPIC}, max_depth, backwards=False
    )
    related_topics, related_edges = _walk(
        klmap, start_node, index, set(ASSOCIATIVE_RELATIONS), max_depth, backwards=None
    )

    return KnowledgeContext(
        current_topic=start_node.name,
        prerequisites=prerequisites,
        related_topics=related_topics,
        next_topics=next_topics,
        relationships=_describe(prerequisite_edges + related_edges + next_edges, index),
    )


def lookup(
    klmap: KLMap,
    topic: str,
    learner_state: LearnerState = LearnerState.NORMAL,
    max_depth: int = 1,
) -> KnowledgeContext | None:
    """`find_start_node` then `traverse`; None when the topic is not on the map."""
    start_node = find_start_node(klmap, topic)
    if start_node is None:
        return None
    return traverse(klmap, start_node, learner_state, max_depth)


def _best_match(
    klmap: KLMap,
    needle: str,
    accept: Callable[[str], bool] | None,
    cutoff: float = 0.0,
) -> KLNode | None:
    """The node whose name is most similar to `needle`, among accepted candidates.

    Ties break on the shorter name and then on node order, so the same topic
    string always resolves to the same node.
    """
    best: tuple[float, int, int] | None = None
    winner: KLNode | None = None

    for position, node in enumerate(klmap.nodes):
        for name in _normalised_names(node):
            if accept is not None and not accept(name):
                continue
            score = SequenceMatcher(None, needle, name).ratio()
            if score < cutoff:
                continue
            candidate = (score, -len(name), -position)
            if best is None or candidate > best:
                best, winner = candidate, node
    return winner


def _walk(
    klmap: KLMap,
    start_node: KLNode,
    index: dict[str, KLNode],
    relations: set[Relation],
    max_depth: int,
    backwards: bool | None,
) -> tuple[list[RelatedTopic], list[KLEdge]]:
    """BFS over `relations` only, up to `max_depth` hops from `start_node`.

    `backwards=True` follows edges from `to` to `from_` (prerequisites),
    `False` from `from_` to `to` (next topics), `None` follows both
    (associative relations).
    """
    adjacency: dict[str, list[tuple[str, KLEdge]]] = {}
    for edge in klmap.edges:
        if edge.relation not in relations:
            continue
        if backwards is not True:
            adjacency.setdefault(edge.from_, []).append((edge.to, edge))
        if backwards is not False:
            adjacency.setdefault(edge.to, []).append((edge.from_, edge))

    distances: dict[str, int] = {start_node.id: 0}
    #: The edge type that first reached each node. Recorded here because this is
    #: the only place that still knows it: the three buckets collapse five
    #: relations into three, and KM03 re-explains *through* the relation, so
    #: dropping it here would force the pedagogy layer to guess `related_to`.
    reached_by: dict[str, Relation] = {}
    traversed: list[KLEdge] = []
    queue: deque[str] = deque([start_node.id])

    while queue:
        current = queue.popleft()
        depth = distances[current]
        if depth >= max_depth:
            continue
        for neighbour_id, edge in adjacency.get(current, []):
            if neighbour_id in distances:
                continue
            distances[neighbour_id] = depth + 1
            reached_by[neighbour_id] = edge.relation
            traversed.append(edge)
            queue.append(neighbour_id)

    topics = [
        RelatedTopic(
            topic=index[node_id].name,
            distance=distance,
            relation=reached_by.get(node_id),
        )
        for node_id, distance in distances.items()
        if node_id != start_node.id and node_id in index
    ]
    topics.sort(key=lambda item: (item.distance, item.topic))
    return topics, traversed


def _describe(edges: list[KLEdge], index: dict[str, KLNode]) -> list[str]:
    """Human-readable sentences for the traversed edges, for the generation brief.

    Deliberately phrased as statements about the *curriculum* ("X is a
    prerequisite of Y"), never about the student.
    """
    templates = {
        Relation.PREREQUISITE_OF: "{source} is a prerequisite of {target}",
        Relation.RELATED_TO: "{source} is related to {target}",
        Relation.NEXT_TOPIC: "{target} comes after {source}",
        Relation.PART_OF: "{source} is part of {target}",
        Relation.USES: "{source} uses {target}",
    }

    sentences: list[str] = []
    for edge in edges:
        source, target = index.get(edge.from_), index.get(edge.to)
        if source is None or target is None:
            continue
        sentence = templates[edge.relation].format(source=source.name, target=target.name)
        if sentence not in sentences:
            sentences.append(sentence)
    return sentences


def _normalised_names(node: KLNode) -> set[str]:
    names = {normalise_text(node.name)}
    if node.name_th:
        names.add(normalise_text(node.name_th))
    return {name for name in names if name}


def normalise_text(text: str) -> str:
    """Casefold, strip punctuation and hyphens, collapse whitespace.

    NFKC first so Thai text composed differently by two authors still matches.
    Public because the syllabus topic mapper (`pedagogy.topics`) must normalise
    a student's words exactly the way node matching does — two normalisers would
    drift and the same message would resolve to different topics per layer.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("-", " ").replace("_", " ")
    text = _PUNCTUATION.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


__all__ = [
    "ASSOCIATIVE_RELATIONS",
    "CONFUSED_PREREQUISITE_DEPTH",
    "find_start_node",
    "lookup",
    "normalise_text",
    "traverse",
]
