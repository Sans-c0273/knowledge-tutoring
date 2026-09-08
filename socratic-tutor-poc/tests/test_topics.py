"""Pipeline step 2 — syllabus topic resolution (Tech Spec §1 step 2, §2.2).

The seed syllabus is the fixture, so the ambiguity these tests pin down is a
real one: no topic is called plain "Linear Equations", but two are called
One-Step and Two-Step Linear Equations.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from socratic_tutor.domain.klmap import find_start_node, load_klmap
from socratic_tutor.pedagogy.topics import (
    CONTEXT_ONLY_CONFIDENCE,
    EXACT_CONFIDENCE,
    MENTION_CONFIDENCE,
    ConversationContext,
    MatchKind,
    Syllabus,
    SyllabusError,
    load_syllabus,
    resolve_topic,
)

SEED_DIR = Path(__file__).resolve().parents[1] / "content" / "seed"
SYLLABUS_PATH = SEED_DIR / "syllabus-linear-equations.yaml"
KL_MAP_PATH = SEED_DIR / "kl-map-linear-equations.yaml"


@pytest.fixture(scope="module")
def syllabus() -> Syllabus:
    return load_syllabus(SYLLABUS_PATH)


def test_seed_syllabus_loads_in_teaching_order(syllabus: Syllabus) -> None:
    assert syllabus.course_id == "MATH-SEED-01"
    assert len(syllabus.topics) == 15
    assert [topic.id for topic in syllabus.topics] == [f"C{n:03d}" for n in range(1, 16)]
    assert all(topic.name_th for topic in syllabus.topics), "every topic needs a Thai name"


def test_every_syllabus_topic_resolves_to_its_kl_node(syllabus: Syllabus) -> None:
    """The syllabus id is the KL node id; this is what makes step 2 → 2b a single hop."""
    klmap = load_klmap(KL_MAP_PATH)

    for topic in syllabus.topics:
        node = find_start_node(klmap, topic.name)
        assert node is not None, f"{topic.name} is not on the KL map"
        assert node.id == topic.id


def test_exact_english_name(syllabus: Syllabus) -> None:
    match = resolve_topic("Two-Step Linear Equations", syllabus)

    assert match is not None
    assert match.topic_id == "C009"
    assert match.confidence == EXACT_CONFIDENCE
    assert match.matched_on is MatchKind.EXACT
    assert match.ambiguous is False


def test_exact_thai_name(syllabus: Syllabus) -> None:
    match = resolve_topic("สมการเชิงเส้นสองขั้นตอน", syllabus)

    assert match is not None
    assert match.topic_id == "C009"
    assert match.confidence == EXACT_CONFIDENCE


def test_alias_and_id_resolve(syllabus: Syllabus) -> None:
    assert resolve_topic("two-step equations", syllabus).topic_id == "C009"  # type: ignore[union-attr]
    assert resolve_topic("C009", syllabus).topic_id == "C009"  # type: ignore[union-attr]
    assert resolve_topic("PEMDAS", syllabus).topic_id == "C002"  # type: ignore[union-attr]


def test_topic_mentioned_inside_a_longer_question(syllabus: Syllabus) -> None:
    match = resolve_topic("How do I solve Two-Step Linear Equations without guessing?", syllabus)

    assert match is not None
    assert match.topic_id == "C009"
    assert match.matched_on is MatchKind.MENTION
    assert match.confidence == MENTION_CONFIDENCE
    assert match.ambiguous is False


def test_thai_question_resolves(syllabus: Syllabus) -> None:
    match = resolve_topic("หนูไม่เข้าใจสมการเชิงเส้นสองขั้นตอนเลยค่ะ", syllabus)

    assert match is not None
    assert match.topic_id == "C009"
    assert match.matched_on is MatchKind.MENTION


def test_mixed_script_question_resolves(syllabus: Syllabus) -> None:
    match = resolve_topic("ช่วยอธิบาย Two-Step Linear Equations หน่อยครับ", syllabus)

    assert match is not None
    assert match.topic_id == "C009"


def test_typo_falls_back_to_similarity(syllabus: Syllabus) -> None:
    match = resolve_topic("Two-Step Linear Equatons", syllabus)

    assert match is not None
    assert match.topic_id == "C009"
    assert match.matched_on is MatchKind.FUZZY
    assert 0.6 < match.confidence < EXACT_CONFIDENCE


def test_bare_linear_equations_is_reported_ambiguous(syllabus: Syllabus) -> None:
    """The spec gap found in R2: this must never silently pick a sibling."""
    match = resolve_topic("Linear Equations", syllabus)

    assert match is not None
    assert match.ambiguous is True
    assert {topic.id for topic in match.candidates} == {"C008", "C009"}
    assert match.resolved_by_context is False
    assert match.confidence < 0.5, "an unresolved guess must not look confident"


def test_conversation_context_breaks_the_tie(syllabus: Syllabus) -> None:
    match = resolve_topic("Linear Equations", syllabus, "Two-Step Linear Equations")

    assert match is not None
    assert match.topic_id == "C009"
    assert match.ambiguous is False
    assert match.resolved_by_context is True
    assert {topic.id for topic in match.candidates} == {"C008", "C009"}


def test_conversation_context_accepts_ids_objects_and_lists(syllabus: Syllabus) -> None:
    by_id = resolve_topic("Linear Equations", syllabus, "C008")
    by_object = resolve_topic(
        "Linear Equations", syllabus, ConversationContext.of("One-Step Linear Equations")
    )
    by_history = resolve_topic("Linear Equations", syllabus, ["C008", "C009"])

    assert by_id is not None and by_id.topic_id == "C008"
    assert by_object is not None and by_object.topic_id == "C008"
    assert by_history is not None and by_history.topic_id == "C008", "most recent topic wins"


def test_context_outside_the_candidates_leaves_it_ambiguous(syllabus: Syllabus) -> None:
    match = resolve_topic("Linear Equations", syllabus, "Negative Numbers")

    assert match is not None
    assert match.ambiguous is True
    assert {topic.id for topic in match.candidates} == {"C008", "C009"}


def test_specific_mention_is_not_ambiguous(syllabus: Syllabus) -> None:
    """One-Step subsumes any shorter sibling label, so naming it is unambiguous."""
    match = resolve_topic("I'm stuck on one-step linear equations", syllabus)

    assert match is not None
    assert match.topic_id == "C008"
    assert match.ambiguous is False


def test_two_different_topics_in_one_message_are_ambiguous(syllabus: Syllabus) -> None:
    match = resolve_topic(
        "how does the Distributive Property relate to Combining Like Terms?", syllabus
    )

    assert match is not None
    assert match.ambiguous is True
    assert {topic.id for topic in match.candidates} == {"C010", "C011"}


def test_follow_up_with_no_topic_carries_the_previous_one(syllabus: Syllabus) -> None:
    match = resolve_topic("I still don't get it", syllabus, "Two-Step Linear Equations")

    assert match is not None
    assert match.topic_id == "C009"
    assert match.matched_on is MatchKind.CONVERSATION_CONTEXT
    assert match.confidence == CONTEXT_ONLY_CONFIDENCE
    assert match.resolved_by_context is True


@pytest.mark.parametrize("message", ["I still don't get it", "", "   ", "photosynthesis"])
def test_no_match_and_no_context_returns_none(syllabus: Syllabus, message: str) -> None:
    assert resolve_topic(message, syllabus) is None


def test_syllabus_lookup_by_any_label(syllabus: Syllabus) -> None:
    assert syllabus.topic("C009") is syllabus.topic("Two-Step Linear Equations")
    assert syllabus.topic("สมการเชิงเส้นสองขั้นตอน") is syllabus.topic("two-step equations")
    assert syllabus.topic("nothing like this") is None


def test_duplicate_topic_id_is_rejected(tmp_path: Path) -> None:
    data = yaml.safe_load(SYLLABUS_PATH.read_text(encoding="utf-8"))
    data["topics"].append({"id": "C009", "name": "Two-Step again"})
    path = tmp_path / "dupe.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

    with pytest.raises(SyllabusError, match="C009"):
        load_syllabus(path)


def test_empty_syllabus_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("course_id: X\ncourse_name: X\ntopics: []\n", encoding="utf-8")

    with pytest.raises(SyllabusError, match="no topics"):
        load_syllabus(path)


def test_missing_syllabus_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SyllabusError, match="Cannot read"):
        load_syllabus(tmp_path / "absent.yaml")
