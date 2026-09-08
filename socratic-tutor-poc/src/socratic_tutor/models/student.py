"""Student Model — per-topic self-declared levels (Tech Spec §2.2).

Populated at onboarding from a questionnaire auto-generated from the course
syllabus. The POC does NOT update levels from performance (Phase 2).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from socratic_tutor.models.enums import StudentLevel


class TopicLevel(BaseModel):
    """One questionnaire answer: this student's declared level for one topic."""

    model_config = ConfigDict(use_enum_values=False)

    topic: str
    level: StudentLevel = StudentLevel.BEGINNER


class StudentModel(BaseModel):
    """A student's declared levels across one course's syllabus topics.

    Tech Spec §2.2. `student_id` is not part of the spec's JSON shape; it is
    carried here because `/student/onboarding` and `/tutor/turn` (Tech Spec §7)
    both key on it, and an unkeyed model cannot be stored.
    """

    model_config = ConfigDict(use_enum_values=False)

    course_id: str
    student_id: str | None = None
    student_topic_levels: list[TopicLevel] = Field(default_factory=list)

    def level_for(self, topic: str) -> StudentLevel:
        """Declared level for `topic`, defaulting to beginner.

        "No questionnaire completed → default = beginner" (Tech Spec §2.2); an
        unknown topic is the same case at single-topic granularity. Matching is
        case-insensitive on trimmed text so a topic string coming back from the
        topic mapper does not have to be byte-identical to the syllabus entry.
        """
        needle = topic.strip().casefold()
        for entry in self.student_topic_levels:
            if entry.topic.strip().casefold() == needle:
                return entry.level
        return StudentLevel.BEGINNER


__all__ = ["StudentModel", "TopicLevel"]
