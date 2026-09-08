"""R4 — Student Model: onboarding questionnaire, storage, and level lookup.

Tech Spec §2.2. The two rules under test throughout: unknown always means
beginner, and nothing in this layer writes a level from performance.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from socratic_tutor.models.enums import StudentLevel
from socratic_tutor.pedagogy.student_model import (
    SELF_STATEMENTS,
    InMemoryStudentModelRepository,
    JsonFileStudentModelRepository,
    StudentModelError,
    StudentModelService,
    generate_onboarding_questionnaire,
)
from socratic_tutor.pedagogy.topics import Syllabus, load_syllabus

SYLLABUS_PATH = (
    Path(__file__).resolve().parents[1] / "content" / "seed" / "syllabus-linear-equations.yaml"
)
COURSE = "MATH-SEED-01"


@pytest.fixture(scope="module")
def syllabus() -> Syllabus:
    return load_syllabus(SYLLABUS_PATH)


@pytest.fixture
def service() -> StudentModelService:
    return StudentModelService(InMemoryStudentModelRepository())


def test_questionnaire_covers_every_syllabus_topic(syllabus: Syllabus) -> None:
    questionnaire = generate_onboarding_questionnaire(syllabus)

    assert questionnaire.course_id == COURSE
    assert questionnaire.topic_ids() == [topic.id for topic in syllabus.topics]
    assert [item.topic for item in questionnaire.items] == [t.name for t in syllabus.topics]


def test_questionnaire_items_are_the_three_spec_statements(syllabus: Syllabus) -> None:
    item = generate_onboarding_questionnaire(syllabus).items[8]

    assert item.topic == "Two-Step Linear Equations"
    assert item.topic_th == "สมการเชิงเส้นสองขั้นตอน"
    assert [option.level for option in item.options] == [
        StudentLevel.BEGINNER,
        StudentLevel.INTERMEDIATE,
        StudentLevel.ADVANCED,
    ]
    assert [option.statement for option in item.options] == [
        "This topic is new to me",
        "I understand the basics, may need help applying",
        "I can explain or apply it independently",
    ]
    assert all(option.statement_th for option in item.options), "every statement needs Thai"
    assert len(SELF_STATEMENTS) == 3


def test_unonboarded_student_is_beginner_everywhere(
    service: StudentModelService, syllabus: Syllabus
) -> None:
    for topic in syllabus.topics:
        assert service.level_for("new-student", COURSE, topic.name) is StudentLevel.BEGINNER
        assert service.level_for("new-student", COURSE, topic) is StudentLevel.BEGINNER

    assert service.get("new-student", COURSE) is None


def test_unknown_topic_is_beginner(service: StudentModelService, syllabus: Syllabus) -> None:
    service.record_onboarding(
        "s1", COURSE, {"Two-Step Linear Equations": StudentLevel.ADVANCED}, syllabus
    )

    assert service.level_for("s1", COURSE, "Photosynthesis") is StudentLevel.BEGINNER


def test_recorded_answers_are_returned(service: StudentModelService, syllabus: Syllabus) -> None:
    service.record_onboarding(
        "s1",
        COURSE,
        {
            "Basic Arithmetic": StudentLevel.ADVANCED,
            "One-Step Linear Equations": StudentLevel.INTERMEDIATE,
            "Two-Step Linear Equations": StudentLevel.BEGINNER,
        },
        syllabus,
    )

    assert service.level_for("s1", COURSE, "Basic Arithmetic") is StudentLevel.ADVANCED
    assert service.level_for("s1", COURSE, "One-Step Linear Equations") is StudentLevel.INTERMEDIATE
    assert service.level_for("s1", COURSE, "Two-Step Linear Equations") is StudentLevel.BEGINNER


def test_partially_completed_questionnaire_leaves_the_rest_beginner(
    service: StudentModelService, syllabus: Syllabus
) -> None:
    service.record_onboarding("s1", COURSE, {"C001": StudentLevel.ADVANCED}, syllabus)

    assert service.level_for("s1", COURSE, "Basic Arithmetic") is StudentLevel.ADVANCED
    for topic in syllabus.topics[1:]:
        assert service.level_for("s1", COURSE, topic.name) is StudentLevel.BEGINNER


def test_answers_keyed_by_id_or_thai_name_store_the_canonical_name(
    service: StudentModelService, syllabus: Syllabus
) -> None:
    model = service.record_onboarding(
        "s1",
        COURSE,
        {"C009": StudentLevel.ADVANCED, "จำนวนลบ": "intermediate"},
        syllabus,
    )

    assert {entry.topic for entry in model.student_topic_levels} == {
        "Two-Step Linear Equations",
        "Negative Numbers",
    }
    assert service.level_for("s1", COURSE, "Two-Step Linear Equations") is StudentLevel.ADVANCED
    assert service.level_for("s1", COURSE, "Negative Numbers") is StudentLevel.INTERMEDIATE


def test_level_lookup_accepts_a_syllabus_topic_object(
    service: StudentModelService, syllabus: Syllabus
) -> None:
    topic = syllabus.topic("C009")
    assert topic is not None
    service.record_onboarding("s1", COURSE, {"C009": StudentLevel.ADVANCED}, syllabus)

    assert service.level_for("s1", COURSE, topic) is StudentLevel.ADVANCED


def test_answer_for_a_topic_outside_the_syllabus_is_rejected(
    service: StudentModelService, syllabus: Syllabus
) -> None:
    with pytest.raises(StudentModelError, match="does not match any topic"):
        service.record_onboarding("s1", COURSE, {"Quantum Mechanics": "advanced"}, syllabus)


def test_re_onboarding_replaces_the_previous_answers(
    service: StudentModelService, syllabus: Syllabus
) -> None:
    service.record_onboarding("s1", COURSE, {"C001": StudentLevel.ADVANCED}, syllabus)
    service.record_onboarding("s1", COURSE, {"C002": StudentLevel.INTERMEDIATE}, syllabus)

    assert service.level_for("s1", COURSE, "Basic Arithmetic") is StudentLevel.BEGINNER
    assert service.level_for("s1", COURSE, "Order of Operations") is StudentLevel.INTERMEDIATE


def test_levels_are_scoped_per_course(service: StudentModelService, syllabus: Syllabus) -> None:
    service.record_onboarding("s1", COURSE, {"C001": StudentLevel.ADVANCED}, syllabus)

    assert service.level_for("s1", "OTHER-COURSE", "Basic Arithmetic") is StudentLevel.BEGINNER


def test_json_repository_round_trips(tmp_path: Path, syllabus: Syllabus) -> None:
    repository = JsonFileStudentModelRepository(tmp_path)
    StudentModelService(repository).record_onboarding(
        "student-42", COURSE, {"C009": StudentLevel.INTERMEDIATE}, syllabus
    )

    assert repository.path_for("student-42", COURSE).exists()

    reopened = StudentModelService(JsonFileStudentModelRepository(tmp_path))
    assert reopened.level_for("student-42", COURSE, "Two-Step Linear Equations") is (
        StudentLevel.INTERMEDIATE
    )
    assert reopened.level_for("student-42", COURSE, "Basic Arithmetic") is StudentLevel.BEGINNER


def test_json_repository_returns_none_for_an_unknown_student(tmp_path: Path) -> None:
    assert JsonFileStudentModelRepository(tmp_path).get("nobody", COURSE) is None


def test_json_repository_rejects_unsafe_ids(tmp_path: Path, syllabus: Syllabus) -> None:
    service = StudentModelService(JsonFileStudentModelRepository(tmp_path))

    with pytest.raises(StudentModelError, match="safe path segment"):
        service.record_onboarding("../../etc/passwd", COURSE, {"C001": "advanced"}, syllabus)


def test_json_repository_reports_a_corrupt_file(tmp_path: Path) -> None:
    repository = JsonFileStudentModelRepository(tmp_path)
    path = repository.path_for("s1", COURSE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(StudentModelError, match="Cannot read student model"):
        repository.get("s1", COURSE)


def test_storing_without_a_student_id_is_rejected(service: StudentModelService) -> None:
    with pytest.raises(StudentModelError):
        service.record_onboarding("", COURSE, {})


def test_no_api_writes_a_level_from_performance() -> None:
    """Tech Spec §2.2: the POC must not update levels from how a turn went.

    Guards the *interface*, not just today's code: any new writer taking an
    evaluation, score, or turn outcome has to change this test, which forces the
    Phase 2 decision into the open.
    """
    writers = [
        name
        for name in dir(StudentModelService)
        if not name.startswith("_")
        and any(verb in name for verb in ("update", "set", "adjust", "infer", "promote"))
    ]

    assert writers == [], f"unexpected level writers on the service: {writers}"
