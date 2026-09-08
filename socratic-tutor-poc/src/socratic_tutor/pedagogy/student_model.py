"""Pipeline step 2, second half: the Student Model (R4, Tech Spec §2.2).

A student's level is **self-declared, per topic**, collected once at onboarding
from a questionnaire generated from the course syllabus. Three self-statements,
one per level, in the student's own words rather than a test.

Two rules from the spec drive every design choice here:

* "No questionnaire completed → default = beginner." Every unknown — no student,
  no onboarding, a topic that was not on the syllabus — resolves to beginner
  rather than an error, because step 2 must always produce a level.
* "POC does NOT update levels from performance (Phase 2)." There is deliberately
  no function in this module that takes a turn outcome, an evaluation, or a
  score. `record_onboarding` is the only writer and it takes questionnaire
  answers only. Adding a performance-driven writer is a Phase 2 decision, not an
  implementation detail — see `StudentModelService`.

Persistence sits behind `StudentModelRepository` so the POC's JSON files can be
replaced by a real database without touching the service or its callers.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from socratic_tutor.models.enums import StudentLevel
from socratic_tutor.models.student import StudentModel, TopicLevel
from socratic_tutor.pedagogy.topics import Syllabus, SyllabusTopic

#: The three self-statements, verbatim from Tech Spec §2.2, with Thai written
#: natively rather than translated word-for-word.
SELF_STATEMENTS: dict[StudentLevel, tuple[str, str]] = {
    StudentLevel.BEGINNER: (
        "This topic is new to me",
        "เรื่องนี้เป็นเรื่องใหม่สำหรับฉัน",
    ),
    StudentLevel.INTERMEDIATE: (
        "I understand the basics, may need help applying",
        "ฉันเข้าใจพื้นฐานแล้ว แต่ยังต้องการความช่วยเหลือเวลานำไปใช้",
    ),
    StudentLevel.ADVANCED: (
        "I can explain or apply it independently",
        "ฉันอธิบายหรือนำไปใช้ได้ด้วยตัวเอง",
    ),
}

#: Ids that are safe as a path segment. Anything else is rejected rather than
#: sanitised, so a crafted student_id cannot escape the data directory.
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class StudentModelError(Exception):
    """A student model could not be stored or read back."""


class QuestionnaireOption(BaseModel):
    """One selectable self-statement, bilingual."""

    model_config = ConfigDict(use_enum_values=False)

    level: StudentLevel
    statement: str
    statement_th: str


class QuestionnaireItem(BaseModel):
    """One syllabus topic and the three levels the student chooses between."""

    topic_id: str
    topic: str
    topic_th: str | None = None
    options: list[QuestionnaireOption] = Field(default_factory=list)


class Questionnaire(BaseModel):
    """The onboarding form for one course: one item per syllabus topic.

    A student may answer some items and skip others; unanswered topics simply
    stay at beginner (Tech Spec §2.2), so the form never blocks on completeness.
    """

    course_id: str
    course_name: str = ""
    items: list[QuestionnaireItem] = Field(default_factory=list)

    def topic_ids(self) -> list[str]:
        return [item.topic_id for item in self.items]


def generate_onboarding_questionnaire(syllabus: Syllabus) -> Questionnaire:
    """Build the self-evaluation questionnaire from the course syllabus.

    "Questionnaire auto-generated from the course's syllabus topics (human
    teacher provides the topic list)" — so items follow teaching order and cover
    every topic, with no editorial choices made here.
    """
    items = [
        QuestionnaireItem(
            topic_id=topic.id,
            topic=topic.name,
            topic_th=topic.name_th,
            options=[
                QuestionnaireOption(level=level, statement=english, statement_th=thai)
                for level, (english, thai) in SELF_STATEMENTS.items()
            ],
        )
        for topic in syllabus.topics
    ]
    return Questionnaire(
        course_id=syllabus.course_id, course_name=syllabus.course_name, items=items
    )


class StudentModelRepository(ABC):
    """Storage for student models, keyed by (student, course).

    Deliberately tiny: the POC writes JSON files, a later phase writes rows, and
    nothing above this interface needs to know which.
    """

    @abstractmethod
    def get(self, student_id: str, course_id: str) -> StudentModel | None:
        """The stored model, or None if this student never onboarded to this course."""

    @abstractmethod
    def save(self, model: StudentModel) -> None:
        """Store (replacing) the model for its own student and course."""


class InMemoryStudentModelRepository(StudentModelRepository):
    """Non-persistent storage — tests, and the API's ephemeral demo mode."""

    def __init__(self) -> None:
        self._models: dict[tuple[str, str], StudentModel] = {}

    def get(self, student_id: str, course_id: str) -> StudentModel | None:
        return self._models.get((student_id, course_id))

    def save(self, model: StudentModel) -> None:
        student_id = _require_student_id(model)
        self._models[(student_id, model.course_id)] = model.model_copy(deep=True)


class JsonFileStudentModelRepository(StudentModelRepository):
    """One JSON file per (course, student) under `root`.

    Sufficient for the POC's handful of demo students and readable on disk when
    a reviewer wants to see what onboarding recorded.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        if root is None:
            from socratic_tutor.config import get_settings

            root = get_settings().data_dir / "students"
        self.root = Path(root)

    def path_for(self, student_id: str, course_id: str) -> Path:
        return resolve_within(
            self.root,
            safe_path_segment(course_id, "course_id"),
            f"{safe_path_segment(student_id, 'student_id')}.json",
        )

    def get(self, student_id: str, course_id: str) -> StudentModel | None:
        path = self.path_for(student_id, course_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StudentModelError(f"Cannot read student model {path}: {exc}") from exc
        try:
            return StudentModel.model_validate(data)
        except Exception as exc:
            raise StudentModelError(f"Student model {path} is malformed: {exc}") from exc

    def save(self, model: StudentModel) -> None:
        student_id = _require_student_id(model)
        path = self.path_for(student_id, model.course_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
        except OSError as exc:
            raise StudentModelError(f"Cannot write student model {path}: {exc}") from exc


class StudentModelService:
    """Reads and writes declared levels; the only entry point step 2 needs.

    Phase 2 note, deliberate: there is no method here that adjusts a level from
    how a turn went. `record_onboarding` accepts questionnaire answers and
    nothing else. Inferring level from performance changes what the tutor asserts
    about a student and needs its own decision, evidence, and review.
    """

    def __init__(self, repository: StudentModelRepository | None = None) -> None:
        self.repository = repository or InMemoryStudentModelRepository()

    def get(self, student_id: str, course_id: str) -> StudentModel | None:
        """The stored model, or None when the student has not onboarded."""
        if not student_id or not course_id:
            return None
        return self.repository.get(student_id, course_id)

    def level_for(self, student: str, course: str, topic: str | SyllabusTopic) -> StudentLevel:
        """Declared level for one topic — beginner unless the student said otherwise.

        Covers all three "unknown" cases identically, as the spec requires: no
        such student, no questionnaire completed, or a topic that was not on the
        syllabus when they onboarded.
        """
        model = self.get(student, course)
        if model is None:
            return StudentLevel.BEGINNER
        return _level_from_model(model, topic)

    def record_onboarding(
        self,
        student_id: str,
        course_id: str,
        answers: Mapping[str, StudentLevel | str],
        syllabus: Syllabus | None = None,
    ) -> StudentModel:
        """Store questionnaire answers as the student's declared levels.

        `answers` is keyed by topic id or topic name; with a `syllabus` the keys
        are resolved to canonical topic names (what the KL Map and the trace use)
        and unknown keys are rejected rather than silently stored under a name no
        lookup will ever match. Topics the student skipped are simply absent, and
        therefore beginner.
        """
        if not student_id or not course_id:
            raise StudentModelError("Both student_id and course_id are required to store a model.")

        levels: list[TopicLevel] = []
        for key, value in answers.items():
            topic_name = key
            if syllabus is not None:
                topic = syllabus.topic(key)
                if topic is None:
                    raise StudentModelError(
                        f"Questionnaire answer for {key!r} does not match any topic in syllabus "
                        f"{syllabus.course_id!r}."
                    )
                topic_name = topic.name
            levels.append(TopicLevel(topic=topic_name, level=StudentLevel(value)))

        model = StudentModel(
            course_id=course_id, student_id=student_id, student_topic_levels=levels
        )
        self.repository.save(model)
        return model


def _level_from_model(model: StudentModel, topic: str | SyllabusTopic) -> StudentLevel:
    """Resolve a level, trying every name the topic is known by.

    Answers are stored under the canonical English name, but callers hold a
    `SyllabusTopic` whose id or Thai name may be what they have to hand.
    """
    if isinstance(topic, SyllabusTopic):
        for key in (topic.name, topic.id, topic.name_th):
            if key and (level := _declared(model, key)) is not None:
                return level
        return StudentLevel.BEGINNER

    level = _declared(model, topic)
    return level if level is not None else StudentLevel.BEGINNER


def _declared(model: StudentModel, key: str) -> StudentLevel | None:
    needle = key.strip().casefold()
    for entry in model.student_topic_levels:
        if entry.topic.strip().casefold() == needle:
            return entry.level
    return None


def _require_student_id(model: StudentModel) -> str:
    if not model.student_id:
        raise StudentModelError("Cannot store a student model without a student_id.")
    return model.student_id


#: Segments that are legal under `_SAFE_ID` but are directory traversal, because
#: '.' is in the allowed character set. `course_id` is client-supplied and used as
#: a *directory* component, so `..` escaped one level above the data root.
_TRAVERSAL_SEGMENTS = frozenset({".", ".."})


def safe_path_segment(value: str, field: str) -> str:
    """Return `value` if it is safe as a single path segment, else raise.

    Public because session storage validates ids the same way; one implementation
    means one place to get it right.

    Note this validates a *string*. It is necessary but not sufficient — use
    `resolve_within` for the actual join, because only resolving the finished path
    against the root can prove where it landed (review finding M10).
    """
    if not _SAFE_ID.match(value or "") or value in _TRAVERSAL_SEGMENTS:
        raise StudentModelError(
            f"{field} {value!r} is not a safe path segment; use letters, digits, '.', '_' or '-'."
        )
    return value


def resolve_within(root: Path, *segments: str) -> Path:
    """Join `segments` under `root` and prove the result is still inside it.

    Validating the segment strings is not enough on its own: a symlink under the
    root, a case-insensitive filesystem, or a future caller that joins something
    unvalidated all defeat string checks. Resolving the finished path and testing
    containment is the check that cannot be argued around, so it is the one that
    guards the filesystem.
    """
    candidate = root.joinpath(*segments)
    try:
        resolved = candidate.resolve()
        base = root.resolve()
    except OSError as exc:
        raise StudentModelError(f"Cannot resolve {candidate}: {exc}") from exc
    if not resolved.is_relative_to(base):
        raise StudentModelError(
            f"Refusing a path outside the data root: {'/'.join(segments)!r} resolves outside {base}"
        )
    return resolved


__all__ = [
    "SELF_STATEMENTS",
    "InMemoryStudentModelRepository",
    "JsonFileStudentModelRepository",
    "Questionnaire",
    "QuestionnaireItem",
    "QuestionnaireOption",
    "StudentModelError",
    "StudentModelRepository",
    "StudentModelService",
    "generate_onboarding_questionnaire",
    "resolve_within",
    "safe_path_segment",
]
