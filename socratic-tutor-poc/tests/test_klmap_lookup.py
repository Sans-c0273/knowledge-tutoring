"""R2 — topic match and relation-filtered BFS (Tech Spec §2.3, KM01–KM03).

The seed map's C009 "Two-Step Linear Equations" is the worked example: it has
exactly one immediate prerequisite (C008), one next topic (C014) and one related
concept (C013), so widening behaviour is unambiguous.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from socratic_tutor.domain.klmap.loader import load_klmap
from socratic_tutor.domain.klmap.lookup import find_start_node, lookup, traverse
from socratic_tutor.domain.klmap.schema import KLMap
from socratic_tutor.models.enums import LearnerState, Relation
from socratic_tutor.models.knowledge import KnowledgeContext

SEED_MAP_PATH = (
    Path(__file__).resolve().parents[1] / "content" / "seed" / "kl-map-linear-equations.yaml"
)


@pytest.fixture(scope="module")
def klmap() -> KLMap:
    return load_klmap(SEED_MAP_PATH)


def topics(context_field: list) -> list[tuple[str, int]]:
    return [(item.topic, item.distance) for item in context_field]


def context_for(
    klmap: KLMap,
    topic: str,
    learner_state: LearnerState = LearnerState.NORMAL,
    max_depth: int = 1,
) -> KnowledgeContext:
    start = find_start_node(klmap, topic)
    assert start is not None, f"expected {topic!r} to match a node"
    return traverse(klmap, start, learner_state, max_depth)


def test_exact_name_match(klmap: KLMap) -> None:
    node = find_start_node(klmap, "Two-Step Linear Equations")

    assert node is not None
    assert node.id == "C009"


@pytest.mark.parametrize(
    "topic",
    [
        "two-step linear equations",
        "TWO-STEP LINEAR EQUATIONS",
        "  Two-Step Linear Equations  ",
        "Two Step Linear Equations",
        "Two-Step Linear Equatons",  # single-character typo
    ],
)
def test_topic_match_tolerates_minor_variation(klmap: KLMap, topic: str) -> None:
    node = find_start_node(klmap, topic)

    assert node is not None
    assert node.id == "C009"


def test_thai_name_match(klmap: KLMap) -> None:
    node = find_start_node(klmap, "สมการเชิงเส้นสองขั้นตอน")

    assert node is not None
    assert node.id == "C009"


def test_node_id_match(klmap: KLMap) -> None:
    node = find_start_node(klmap, "c009")

    assert node is not None
    assert node.id == "C009"


@pytest.mark.parametrize("topic", ["Photosynthesis", "การสังเคราะห์แสง", "", "   "])
def test_unknown_topic_returns_none(klmap: KLMap, topic: str) -> None:
    assert find_start_node(klmap, topic) is None
    assert lookup(klmap, topic) is None


def test_depth_one_neighbourhood(klmap: KLMap) -> None:
    context = context_for(klmap, "Two-Step Linear Equations")

    assert context.current_topic == "Two-Step Linear Equations"
    assert topics(context.prerequisites) == [("One-Step Linear Equations", 1)]
    assert topics(context.next_topics) == [("Word Problems to Equations", 1)]
    assert topics(context.related_topics) == [("Checking a Solution", 1)]


def test_prerequisites_are_walked_backwards(klmap: KLMap) -> None:
    """C008 -> C009 means C008 is the prerequisite, so it must not surface as a next topic."""
    context = context_for(klmap, "Two-Step Linear Equations")

    assert "One-Step Linear Equations" not in [item.topic for item in context.next_topics]
    assert "Word Problems to Equations" not in [item.topic for item in context.prerequisites]


def test_confused_widens_prerequisites_to_depth_two(klmap: KLMap) -> None:
    context = context_for(klmap, "Two-Step Linear Equations", LearnerState.CONFUSED)

    assert topics(context.prerequisites) == [
        ("One-Step Linear Equations", 1),
        ("Equation Balance", 2),
        ("Evaluating Expressions", 2),
        ("Inverse Operations", 2),
    ]


def test_confused_does_not_widen_other_relations(klmap: KLMap) -> None:
    normal = context_for(klmap, "Two-Step Linear Equations")
    confused = context_for(klmap, "Two-Step Linear Equations", LearnerState.CONFUSED)

    assert topics(confused.related_topics) == topics(normal.related_topics)
    assert topics(confused.next_topics) == topics(normal.next_topics)
    assert all(item.distance == 1 for item in confused.related_topics)


def test_explicit_max_depth_two_widens_every_relation(klmap: KLMap) -> None:
    context = context_for(klmap, "Two-Step Linear Equations", max_depth=2)

    # C013 related_to C009 (distance 1) and C013 related_to C008 (distance 2).
    assert topics(context.related_topics) == [
        ("Checking a Solution", 1),
        ("One-Step Linear Equations", 2),
    ]
    assert topics(context.next_topics) == [("Word Problems to Equations", 1)]


def test_max_depth_zero_returns_only_the_current_topic(klmap: KLMap) -> None:
    context = context_for(klmap, "Two-Step Linear Equations", max_depth=0)

    assert context.current_topic == "Two-Step Linear Equations"
    assert context.prerequisites == []
    assert context.related_topics == []
    assert context.next_topics == []
    assert context.relationships == []


def test_negative_max_depth_is_rejected(klmap: KLMap) -> None:
    start = find_start_node(klmap, "C009")
    assert start is not None

    with pytest.raises(ValueError, match="max_depth"):
        traverse(klmap, start, LearnerState.NORMAL, max_depth=-1)


def test_uses_and_part_of_land_in_related_topics(klmap: KLMap) -> None:
    """The vocabulary has five relations and KnowledgeContext three buckets."""
    context = context_for(klmap, "Variables and Expressions")

    assert topics(context.related_topics) == [
        ("Combining Like Terms", 1),
        ("Distributive Property", 1),
        ("Evaluating Expressions", 1),
    ]
    assert "Combining Like Terms uses Variables and Expressions" in context.relationships
    assert "Evaluating Expressions is part of Variables and Expressions" in context.relationships


def test_relationships_are_human_readable(klmap: KLMap) -> None:
    context = context_for(klmap, "Two-Step Linear Equations")

    assert set(context.relationships) == {
        "One-Step Linear Equations is a prerequisite of Two-Step Linear Equations",
        "Checking a Solution is related to Two-Step Linear Equations",
        "Word Problems to Equations comes after Two-Step Linear Equations",
    }


def test_prerequisites_are_never_phrased_as_a_student_gap(klmap: KLMap) -> None:
    """Tech Spec §3.5 hard rule: an edge is a hypothesis to check, not a diagnosis."""
    context = context_for(klmap, "Two-Step Linear Equations", LearnerState.CONFUSED)

    serialised = context.model_dump_json().casefold()
    for forbidden in ("not understand", "does not know", "missing", "gap", "weak", "lacks"):
        assert forbidden not in serialised


def test_leaf_topic_has_no_prerequisites(klmap: KLMap) -> None:
    context = context_for(klmap, "Basic Arithmetic", LearnerState.CONFUSED)

    assert context.prerequisites == []
    assert topics(context.next_topics) == []


def test_lookup_combines_match_and_traversal(klmap: KLMap) -> None:
    context = lookup(klmap, "สมการเชิงเส้นสองขั้นตอน", LearnerState.CONFUSED)

    assert context is not None
    assert context.current_topic == "Two-Step Linear Equations"
    assert len(context.prerequisites) == 4


def test_related_topics_keep_the_edge_that_reached_them(klmap: KLMap) -> None:
    """The three buckets collapse five relations; the edge type must survive it.

    KM03 re-explains *through* a related concept, and "Combining Like Terms
    **uses** Variables and Expressions" is a different lesson from "Evaluating
    Expressions is **part of** Variables and Expressions". Without this the
    pedagogy layer would have to assume `related_to` and would be wrong here.
    """
    context = context_for(klmap, "Variables and Expressions")

    assert [(item.topic, item.relation) for item in context.related_topics] == [
        ("Combining Like Terms", Relation.USES),
        ("Distributive Property", Relation.USES),
        ("Evaluating Expressions", Relation.PART_OF),
    ]


def test_related_to_edges_are_reported_as_related_to(klmap: KLMap) -> None:
    context = context_for(klmap, "Two-Step Linear Equations")

    assert [item.relation for item in context.related_topics] == [Relation.RELATED_TO]


def test_single_relation_buckets_are_labelled_too(klmap: KLMap) -> None:
    """Redundant for these two walks, but the inspector renders one shape."""
    context = context_for(klmap, "Two-Step Linear Equations", LearnerState.CONFUSED)

    assert {item.relation for item in context.prerequisites} == {Relation.PREREQUISITE_OF}
    assert {item.relation for item in context.next_topics} == {Relation.NEXT_TOPIC}
