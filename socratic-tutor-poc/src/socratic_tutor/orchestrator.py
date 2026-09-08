"""The per-turn pipeline (R12, Tech Spec §1).

One student turn runs the steps in order — intent, topic + level, retrieval, KL
Map lookup, conditional answer evaluation, strategy selection, response
planning, generation, guardrail — and produces two things: the student-facing
text and a `TurnRecord`.

**The record is the deliverable, not a side effect.** R12's bar is that for any
turn id the full teaching decision reconstructs from the record alone, and the
glass-box inspector renders from it with no extra LLM call (Addendum §"Web UI
scope" item 3). Every step therefore writes what it decided *as it decides it*:
a `trace` event goes out after each one, because the pedagogically interesting
choices all happen before the first token, and a turn that dies at step 1 still
yields a serialisable record with `error` set.

This module owns sequencing, state and the record. Every decision belongs to
the layer that owns it — `pedagogy.intent`, `pedagogy.topics`,
`pedagogy.student_model`, `domain.rag`, `domain.klmap`, `pedagogy.evaluation`,
`pedagogy.strategy`, `pedagogy.planner`, `pedagogy.generation`,
`pedagogy.guardrail` — and is reached through `TurnDependencies`, so a test can
replace any of them without a network call.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from socratic_tutor.config import Settings, get_settings
from socratic_tutor.domain.klmap import KLMap, find_start_node, traverse
from socratic_tutor.models.enums import (
    Evaluation,
    GuardrailEventType,
    Intent,
    LearnerState,
    StrategyId,
    StudentLevel,
)
from socratic_tutor.models.intent import IntentResult
from socratic_tutor.models.knowledge import KnowledgeContext, RetrievedChunk
from socratic_tutor.models.plan import ResponsePlan
from socratic_tutor.models.strategy import KnowledgeGuidance, StrategySelection
from socratic_tutor.models.trace import GuardrailEvent, TurnTrace
from socratic_tutor.pedagogy import intent as intent_prompt
from socratic_tutor.pedagogy.evaluation import (
    SYSTEM_PROMPT as EVALUATION_PROMPT,
)
from socratic_tutor.pedagogy.evaluation import (
    EvaluationResult,
    EvaluationSource,
    Problem,
    evaluate_attempt,
    load_answer_keys,
)
from socratic_tutor.pedagogy.generation import ROLE_LAYER, detect_language, generate_response
from socratic_tutor.pedagogy.guardrail import GuardrailAction, check_output, normalise
from socratic_tutor.pedagogy.planner import PlanTrace, plan_response
from socratic_tutor.pedagogy.strategy import select_strategy
from socratic_tutor.pedagogy.student_model import (
    StudentModelService,
    resolve_within,
    safe_path_segment,
)
from socratic_tutor.pedagogy.tables import RuleTables
from socratic_tutor.pedagogy.topics import (
    ConversationContext,
    MatchKind,
    Syllabus,
    TopicMatch,
    resolve_topic,
)
from socratic_tutor.providers.base import Msg, StreamStats, prompt_version

logger = logging.getLogger(__name__)

#: How many prior messages reach Call A (Tech Spec §5.1: "last 4–6 turns").
CONTEXT_MESSAGES = 6

#: Chunk size when a buffered reply is re-emitted as tokens.
BUFFERED_TOKEN_SIZE = 24

#: What replaces a canonical answer anywhere it would otherwise be written down.
REDACTED = "[redacted]"

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


# --------------------------------------------------------------- turn record


class TopicResolution(BaseModel):
    """What step 2 decided about the topic, in full (Tech Spec §1 step 2).

    `TurnTrace.topic` is a single string, which cannot answer "was this
    ambiguous, and what else did it nearly match?" — a question the inspector
    and any post-hoc review of a bad turn both need. This model carries the
    rest; fold it into `TurnTrace` whenever that model gains a field for it.
    """

    topic: str | None = None
    topic_th: str | None = None
    node_id: str | None = None
    matched_by: MatchKind | None = None
    confidence: float = 0.0
    ambiguous: bool = False
    resolved_by_context: bool = False
    candidates: list[str] = Field(default_factory=list)
    student_level: StudentLevel = StudentLevel.BEGINNER
    level_source: Literal["onboarding_questionnaire", "default_beginner"] = "default_beginner"


class PolicyApplied(BaseModel):
    """The Teaching Policy row that governed this turn (Tech Spec §3.4).

    The single most safety-relevant decision on the panel — "why did it withhold
    this?" — and it had no home on the trace. It was inferable only from a string
    buried in the versions map, which is why the inspector's policy card could
    never render. Values are the row's own, verbatim from `teaching_policy.yaml`;
    they are not re-encoded into an enum here, because the spec's §3.4 table is
    written in prose ("Not initially - hint first") and inventing codes for it
    would put a second vocabulary between the table and the screen.
    """

    row: str
    context: str = ""
    direct_answer: str = ""
    attempt: str = ""
    check: str = ""
    precedence: int = 0
    #: Other rows that also applied; the highest-precedence one is `row`.
    also_applied: list[str] = Field(default_factory=list)
    #: True when the policy changed what the Trigger Matrix would have permitted.
    override_applied: bool = False
    #: What it changed, and why, in the selector's own words.
    override_notes: list[str] = Field(default_factory=list)
    prohibited_strategies: list[str] = Field(default_factory=list)


class TriggerMatch(BaseModel):
    """One Trigger Matrix row that matched, and how it fared (Tech Spec §3.1).

    Carried whole rather than flattened to a label. "TM12 beat TM16" is half an
    answer; a reviewer asking whether the table is *right* needs to know that
    priority decided it rather than alphabetical order, and what the losing row
    would have done instead. Both are here, from the selector.
    """

    model_config = ConfigDict(extra="allow")

    row_id: str
    key: dict[str, str] = Field(default_factory=dict)
    priority: int = 0
    specificity: int = 0
    primary: str = ""
    supporting: list[str] = Field(default_factory=list)
    selected: bool = False
    #: The row that beat this one, and on which dimension — `priority`,
    #: `specificity` or, tellingly, `row_id`.
    lost_to: str | None = None
    lost_on: str | None = None
    #: `spec` for a row transcribed from the Technical Specification,
    #: `extension` for one this build added.
    source: str = ""


class KmMatch(BaseModel):
    """One KL Map → Strategy rule that matched, and how it fared (Tech Spec §3.5).

    `requirement` is the clause the rule matched on, and it is the field that
    keeps KM02 honest: the rule fired because a prerequisite at distance 1
    exists in the *map*, not because anything was concluded about this student.
    `check_or_scaffold_prerequisite` must never read as a diagnosis, and showing
    what it matched on is how the panel keeps that visible.
    """

    model_config = ConfigDict(extra="allow")

    rule_id: str
    applied: bool = False
    priority: int = 0
    #: None for a rule with no `requires` clause — KM01 matches on learner state
    #: alone — so an absent requirement is a fact, not a missing value.
    requirement: str | None = None
    knowledge_action: str | None = None
    kl_map_action: str = ""
    strategy_effect: str = ""
    lost_to: str | None = None
    lost_on: str | None = None


class CombinationVerdict(BaseModel):
    """What the Combination Rules allowed the strategy set to be (Tech Spec §3.3)."""

    rule: str = ""
    sequence: list[str] = Field(default_factory=list)
    dropped: list[str] = Field(default_factory=list)
    #: Each entry names the rule that fired, what it dropped and the reason.
    actions: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def allowed(self) -> bool:
        """True when nothing was dropped or substituted."""
        return not self.dropped and not self.actions


class TurnRecord(BaseModel):
    """One turn's complete, replayable decision record.

    `trace` is the spec's §7 object. The other two fields carry what it has no
    room for and the inspector needs: how the topic was resolved, and *how* an
    answer was judged — `TurnTrace.evaluation` keeps the verdict but not the
    error locus S06's feedback is built from, nor whether the deterministic
    checker or the model decided it.
    """

    model_config = ConfigDict(use_enum_values=False)

    trace: TurnTrace
    topic_resolution: TopicResolution = Field(default_factory=TopicResolution)
    answer_evaluation: EvaluationResult | None = None
    #: What the guardrail actually looked for on this turn, taken from the
    #: guardrail rather than described here. A PASS means "checked these forms
    #: and found nothing", which is a different claim from "found nothing", and
    #: the difference is real every time the scanner meets a boundary.
    scan_coverage: dict[str, Any] = Field(default_factory=dict)
    #: Every Trigger Matrix row that matched, winner and losers, with the
    #: tie-break that decided it. `TurnTrace.matched_trigger_rows` keeps the ids
    #: because §7 names that field; this carries the contest behind them.
    trigger_matches: list[TriggerMatch] = Field(default_factory=list)
    #: The same for the KM rules, whose O2 tie-break §3.5 requires be logged.
    km_matches: list[KmMatch] = Field(default_factory=list)
    #: The policy row that governed the turn, and the combination verdict that
    #: shaped the strategy set. Both are produced by the selector and were being
    #: dropped at the wire boundary.
    teaching_policy: PolicyApplied | None = None
    combination: CombinationVerdict | None = None
    #: Which response template produced the plan's structure (Tech Spec §4.1).
    response_template: str = ""
    #: Whether this turn's plan withheld the solution. Read **with**
    #: `scan_coverage`: `scanned: false` alone is ambiguous, because the scanner
    #: reports coverage only for checks that actually ran. Withholding false plus
    #: unscanned means there was nothing to check; withholding true plus
    #: unscanned means there was something to check and we could not — the
    #: signature of every disarm bug found in this build. Conflating those two is
    #: exactly what this field exists to prevent.
    withheld_solution: bool = False

    @property
    def turn_id(self) -> str:
        return self.trace.turn_id


# -------------------------------------------------------------- session state


class SessionMessage(BaseModel):
    """One line of dialogue, in the order it was said."""

    role: Literal["student", "tutor"]
    text: str
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    turn_id: str | None = None


class SessionState(BaseModel):
    """A conversation: who, which course, what was said, and every turn record.

    This is what `GET /tutor/session/{id}` returns (Tech Spec §7) and what steps
    1 and 2 read for context.
    """

    session_id: str
    student_id: str = ""
    course_id: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    messages: list[SessionMessage] = Field(default_factory=list)
    turns: list[TurnRecord] = Field(default_factory=list)
    #: The problem currently under discussion, once one has been identified.
    #: Only the id is kept: the answer itself stays in the server-side key file
    #: rather than being copied into session storage.
    active_problem_id: str | None = None
    #: Practice items already put to this student, so a second "quiz me" asks
    #: something new rather than repeating the last question.
    asked_problem_ids: list[str] = Field(default_factory=list)
    #: Problems this student has answered correctly. Used only to choose the
    #: *next* quiz question — never to release the guardrail, which is why it is
    #: separate from `active_problem_id`.
    solved_problem_ids: list[str] = Field(default_factory=list)

    def conversation(self, limit: int = CONTEXT_MESSAGES) -> list[Msg]:
        """Recent dialogue as provider messages, oldest first."""
        recent = self.messages[-limit:] if limit > 0 else []
        return [
            Msg(role="user" if message.role == "student" else "assistant", content=message.text)
            for message in recent
        ]

    def recent_topics(self) -> ConversationContext:
        """Topics from previous turns, most recent first (the step-2 tie-break)."""
        topics: list[str] = []
        for record in reversed(self.turns):
            topic = record.trace.topic
            if topic and topic not in topics:
                topics.append(topic)
        return ConversationContext(recent_topics=topics)


class SessionRepository:
    """Storage for sessions. Same shape as `StudentModelRepository`, by design."""

    def get(self, session_id: str) -> SessionState | None:
        raise NotImplementedError

    def save(self, state: SessionState) -> None:
        raise NotImplementedError


class InMemorySessionRepository(SessionRepository):
    """Non-persistent sessions — tests and ephemeral demos."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def get(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def save(self, state: SessionState) -> None:
        self._sessions[state.session_id] = state


class JsonFileSessionRepository(SessionRepository):
    """One JSON file per session under `root`.

    Ids are validated as path segments before they touch the filesystem, for the
    same reason the student store does it: a crafted `session_id` must not be
    able to address a file outside the data directory.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        if root is None:
            root = get_settings().data_dir / "sessions"
        self.root = Path(root)

    def path_for(self, session_id: str) -> Path:
        return resolve_within(self.root, f"{safe_path_segment(session_id, 'session_id')}.json")

    def get(self, session_id: str) -> SessionState | None:
        path = self.path_for(session_id)
        if not path.exists():
            return None
        try:
            return SessionState.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise OSError(f"Cannot read session {path}: {exc}") from exc

    def save(self, state: SessionState) -> None:
        path = self.path_for(state.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(state.model_dump_json(indent=2), encoding="utf-8")


# -------------------------------------------------------- answer-key lookup


class AnswerKeyLookup:
    """Finds the canonical solution for the problem a message is about.

    Answer keys are server-side only: never ingested into RAG, and never in Call
    C's context while the plan withholds them (Tech Spec §5.2). They are loaded
    here because the evaluator and the guardrail both need them every turn.

    A key is returned only when the message actually refers to that problem.
    Matching on topic alone would attach an answer to a turn that is not about
    it, and the guardrail would then withhold a number the student never asked
    about. Two ways of referring count:

    * the problem's statement appears in the message, folded through the
      guardrail's own `normalise` so that the matcher and the scanner agree on
      what "the same string" means — `2x + ๔ = ๑๐`, fullwidth digits and spacing
      variants all fold together, and Thai numerals are ordinary student input
      in a bilingual product;
    * the statement's numbers appear in the message in the same order, which
      catches an equation written out in words ("2x plus 4 equals 10").

    The second rule can over-match, and deliberately so: attaching a problem
    causes the turn to withhold, which is the safe direction. Missing one is
    what leaves a turn unscanned.

    Across turns the identified problem is remembered on the session rather than
    re-derived per message — see `_resolve_canonical`. A student who says "just
    give me the answer" on turn two is still working the same problem, and a
    tutor would know that.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._problems: dict[str, Problem] | None = None

    def problems(self) -> dict[str, Problem]:
        if self._problems is None:
            path = self._path or (get_settings().content_dir / "seed" / "answer-keys.yaml")
            try:
                # Every failure mode of the loader is a ValueError, including a
                # missing file; pydantic's ValidationError is one too.
                self._problems = load_answer_keys(path)
            except ValueError as exc:
                logger.info("no answer keys loaded from %s: %s", path, exc)
                self._problems = {}
        return self._problems

    def by_id(self, problem_id: str) -> Problem | None:
        """The problem with this id, for resolving a session's active problem."""
        return self.problems().get(problem_id) if problem_id else None

    def for_topic(self, topic: str) -> list[Problem]:
        """Every problem filed under `topic`, in id order.

        Weaker evidence than a statement match — these are problems the student
        *could* be working, not one they demonstrably are — so callers use it
        only where arming on the wrong problem is harmless (see
        `_topic_fallback`) or where the system chose the problem itself (a quiz
        item).
        """
        needle = _fold(topic)
        if not needle:
            return []
        return sorted(
            (problem for problem in self.problems().values() if _fold(problem.topic) == needle),
            key=lambda problem: problem.id,
        )

    def __call__(self, course_id: str, topic: str | None, message: str) -> Problem | None:
        haystack = _fold(message)
        numbers = _numbers(message)
        if not haystack:
            return None

        best: Problem | None = None
        best_length = 0
        for problem in self.problems().values():
            for statement in (problem.statement_en, problem.statement_th):
                needle = _fold(statement.partition(":")[2] or statement)
                if len(needle) < 4:
                    continue
                # Longest statement wins, so "2x+4=10" is not beaten by "x=10".
                if (needle in haystack or _numbers_in_order(_numbers(statement), numbers)) and len(
                    needle
                ) > best_length:
                    best, best_length = problem, len(needle)
        return best


def _fold(text: str) -> str:
    """The guardrail's normalisation, with spacing removed as well.

    `normalise` already folds case, Unicode composition and Thai digits; dropping
    the remaining spaces makes "2x + 4 = 10" and "2x+4=10" the same needle, which
    matters because students write equations both ways.
    """
    return "".join(normalise(text or "").split())


def _numbers(text: str) -> list[str]:
    """Numbers in the order they appear, Thai digits folded to ASCII."""
    return _NUMBER.findall(normalise(text or ""))


def _numbers_in_order(needle: list[str], haystack: list[str]) -> bool:
    """Whether `needle`'s numbers appear in `haystack` in the same order.

    Requires at least two, so a statement with a single number cannot match
    every message that happens to mention it.
    """
    if len(needle) < 2:
        return False
    position = 0
    for value in needle:
        try:
            position = haystack.index(value, position) + 1
        except ValueError:
            return False
    return True


def _default_retrieve(course_id: str, topic: str | None, message: str) -> list[RetrievedChunk]:
    """Step 2a. Queries the course collection with the topic and the message.

    The topic is prepended to the query text rather than used as a metadata
    filter: a hard filter drops good chunks whose topic label differs from the
    syllabus wording, and an empty retrieval set has policy consequences
    (Teaching Policy row 4) rather than being an error.
    """
    from socratic_tutor.domain.rag import store

    query_text = f"{topic} {message}".strip() if topic else message
    try:
        return store.query(course_id, query_text)
    except Exception as exc:  # noqa: BLE001 - the vector store is third-party and
        # can fail in ways it does not document; an empty retrieval set is a
        # supported turn outcome, a 500 in the middle of a lesson is not.
        logger.warning("retrieval failed for course %s: %s", course_id, exc)
        return []


# -------------------------------------------------------------- dependencies


@dataclass
class TurnDependencies:
    """Everything a turn needs, injected so tests run fully offline.

    `syllabus` and `tables` are required; everything else defaults to the real
    implementation. Replace any callable to fake a step.
    """

    syllabus: Syllabus
    tables: RuleTables
    klmap: KLMap | None = None
    students: StudentModelService = field(default_factory=StudentModelService)
    sessions: SessionRepository = field(default_factory=InMemorySessionRepository)
    settings: Settings | None = None

    #: (course_id, topic, message) -> retrieved chunks (step 2a).
    retrieve: Callable[[str, str | None, str], list[RetrievedChunk]] = _default_retrieve
    #: (course_id, topic, message) -> the canonical solution, when the message
    #: is about a problem that has one.
    canonical_lookup: Callable[[str, str | None, str], Problem | None] = field(
        default_factory=AnswerKeyLookup
    )
    #: Resolves the session's remembered problem. Bound automatically when
    #: `canonical_lookup` can answer by id (see `__post_init__`).
    problem_by_id: Callable[[str], Problem | None] | None = None
    #: Every answer key filed under a topic, for the quiz item the tutor asks
    #: and for the withholding-turn fallback. Bound the same way.
    problems_for_topic: Callable[[str], list[Problem]] | None = None
    #: Call A; signature of `pedagogy.intent.detect_intent`.
    detect_intent: Callable[..., Any] | None = None
    #: Call B; signature of `pedagogy.evaluation.evaluate_attempt`.
    evaluate: Callable[..., Any] = evaluate_attempt
    #: Call C; signature of `pedagogy.generation.generate_response`.
    generate: Callable[..., AsyncIterator[str]] = generate_response

    def __post_init__(self) -> None:
        if self.problem_by_id is None:
            by_id = getattr(self.canonical_lookup, "by_id", None)
            if callable(by_id):
                self.problem_by_id = by_id
        if self.problems_for_topic is None:
            for_topic = getattr(self.canonical_lookup, "for_topic", None)
            if callable(for_topic):
                self.problems_for_topic = for_topic

    def resolved_settings(self) -> Settings:
        return self.settings or get_settings()

    def intent_detector(self) -> Callable[..., Any]:
        if self.detect_intent is not None:
            return self.detect_intent
        from socratic_tutor.pedagogy.intent import detect_intent as real

        return real


@dataclass(frozen=True)
class TurnRequest:
    """One student turn's input (Tech Spec §7, `POST /tutor/turn`)."""

    session_id: str
    student_id: str
    course_id: str
    message: str


@dataclass(frozen=True)
class TurnEvent:
    """One pipeline event, ready to become an SSE frame.

    `record` rides along on `trace`, `done` and `error` events so the API layer
    can serialise it in whatever shape the client expects instead of re-parsing
    a dict; `data` carries the small payloads (`{"text": ...}`).
    """

    event: Literal["trace", "token", "done", "error"]
    data: dict[str, Any] = field(default_factory=dict)
    record: TurnRecord | None = None


@dataclass
class _Decision:
    """What steps 1–4 produced, handed to the generation step."""

    plan: ResponsePlan = field(default_factory=ResponsePlan)
    canonical: Problem | None = None
    #: The practice item the tutor is about to ask, when this is an S08 turn.
    quiz_item: Problem | None = None
    knowledge_guidance: KnowledgeGuidance | None = None
    topic_name: str = ""


# ---------------------------------------------------------------- the turn


async def run_turn(request: TurnRequest, deps: TurnDependencies) -> AsyncIterator[TurnEvent]:
    """Run one turn, yielding events as the pipeline progresses.

    A `trace` event is emitted after every decision step, then `token` events,
    then `done` with the completed record. Any failure yields `error` followed
    by `done` carrying the partial record — a failed turn renders in the
    inspector as readily as a successful one.
    """
    state = deps.sessions.get(request.session_id) or SessionState(
        session_id=request.session_id,
        student_id=request.student_id,
        course_id=request.course_id,
    )
    trace = TurnTrace(session_id=request.session_id, student_message=request.message)
    record = TurnRecord(trace=trace)
    settings = deps.resolved_settings()
    decision = _Decision()
    streamed = ""

    try:
        async for event in _decide(request, deps, state, record, settings, decision):
            yield event
    except Exception as exc:  # every failure must still produce a record
        logger.exception("turn %s failed before generation", trace.turn_id)
        trace.error = f"{type(exc).__name__}: {exc}"
        _persist(deps, state, request, record, "")
        yield TurnEvent("error", {"message": trace.error, "turn_id": trace.turn_id}, record)
        yield TurnEvent("done", record=record)
        return

    trace.latencies_ms["pre_stream"] = trace.total_latency_ms
    yield TurnEvent("trace", record=record)

    try:
        async for event in _generate_and_guard(request, deps, state, record, decision):
            if event.event == "token":
                streamed += str(event.data.get("text", ""))
            yield event
    except Exception as exc:
        logger.exception("turn %s failed during generation", trace.turn_id)
        trace.error = f"{type(exc).__name__}: {exc}"
        _persist(deps, state, request, record, streamed)
        yield TurnEvent("error", {"message": trace.error, "turn_id": trace.turn_id}, record)
        yield TurnEvent("done", record=record)
        return

    _persist(deps, state, request, record, trace.generated_text or "")
    yield TurnEvent("done", record=record)


async def execute_turn(request: TurnRequest, deps: TurnDependencies) -> TurnRecord:
    """Run a turn to completion and return its record. For tests and eval runs."""
    record: TurnRecord | None = None
    async for event in run_turn(request, deps):
        if event.record is not None:
            record = event.record
    assert record is not None, "run_turn always ends with a done event carrying the record"
    return record


async def _decide(
    request: TurnRequest,
    deps: TurnDependencies,
    state: SessionState,
    record: TurnRecord,
    settings: Settings,
    out: _Decision,
) -> AsyncIterator[TurnEvent]:
    """Steps 1–4, emitting the growing trace after each one."""
    trace = record.trace
    trace.table_versions.update(deps.tables.audit_versions)

    # --- step 1: intent ------------------------------------------------------
    with trace.step("intent"):
        intent_result: IntentResult = await deps.intent_detector()(
            request.message, state.conversation(), settings=settings
        )
    trace.intent_result = intent_result
    trace.model_ids["intent"] = settings.model_for("intent")
    _record_prompt_version(trace, "intent", intent_prompt.SYSTEM_PROMPT)
    yield TurnEvent("trace", record=record)

    # --- step 2: topic + student level ---------------------------------------
    with trace.step("topic_level"):
        match = resolve_topic(request.message, deps.syllabus, state.recent_topics())
        topic_name = match.topic_name if match else ""
        level = (
            deps.students.level_for(request.student_id, request.course_id, match.topic)
            if match
            else StudentLevel.BEGINNER
        )
        record.topic_resolution = _topic_resolution(match, level, deps)
    trace.topic = topic_name or None
    trace.student_level = level
    yield TurnEvent("trace", record=record)

    # --- step 2a: retrieval --------------------------------------------------
    with trace.step("retrieval"):
        chunks = list(deps.retrieve(request.course_id, topic_name or None, request.message))
    trace.retrieved_chunks = chunks
    yield TurnEvent("trace", record=record)

    # --- step 2b: KL Map lookup ----------------------------------------------
    with trace.step("kl_lookup"):
        knowledge = _kl_lookup(deps, match, intent_result.learner_state)
    trace.knowledge_context = knowledge
    yield TurnEvent("trace", record=record)

    # The canonical solution is resolved on every turn because the evaluator and
    # the guardrail both need it. Whether the *generator* sees it is decided in
    # `generation.build_system_prompt` from the plan, and nowhere else.
    canonical = _resolve_canonical(request, deps, state, topic_name)

    # --- step 2c: answer evaluation (conditional) ----------------------------
    if intent_result.intent is Intent.CHECK_ANSWER:
        with trace.step("answer_evaluation"):
            evaluated = await _evaluate(request, deps, chunks, canonical, settings)
        # `reason` reads "deterministic match: 5 != 3" — a second copy of the
        # answer, in a field that is streamed and persisted like everything else.
        record.answer_evaluation = evaluated.model_copy(
            update={
                "reason": redact_answer(evaluated.reason, canonical),
                "error_locus": redact_answer(evaluated.error_locus or "", canonical) or None,
            }
        )
        trace.evaluation = record.answer_evaluation.evaluation
        trace.model_ids["evaluation"] = settings.model_for("evaluation")
        _record_prompt_version(trace, "evaluation", EVALUATION_PROMPT)
        yield TurnEvent("trace", record=record)

    # --- step 3: strategy selection ------------------------------------------
    with trace.step("strategy_selection"):
        selection = select_strategy(
            intent_result,
            level,
            topic_name,
            knowledge,
            trace.evaluation,
            bool(chunks),
            deps.tables,
        )
    trace.strategy_selection = selection
    _record_decision_trace(trace, record, selection)
    record.teaching_policy = _policy_applied(selection, deps)
    record.combination = _combination_verdict(selection)
    yield TurnEvent("trace", record=record)

    # --- step 4: response planning -------------------------------------------
    with trace.step("response_planning"):
        plan_trace = PlanTrace()
        plan = plan_response(
            selection,
            deps.tables,
            language=detect_language(request.message),
            student_level=level,
            trace=plan_trace,
        )
    trace.response_plan = plan
    # Deliberately *not* in `table_versions`: which template fired is a decision
    # this turn made, not the version of a file it read. Mixing the two is what
    # let the policy row hide in a versions map instead of being a field.
    record.response_template = plan_trace.template

    withholding = not _may_state_solution(plan, deps)

    # A practice question is a problem the *system* authors. Choosing it here
    # rather than letting the model invent one is what makes it knowable: the
    # tutor can then guard the answer it is about to ask for, and the follow-up
    # turn ("is it 3?") arrives with the problem already in scope instead of
    # waiting for the student to restate a question they were just handed.
    if selection.primary_strategy.strategy_id is StrategyId.PRACTICE_QUIZ:
        quiz_item = _select_quiz_item(deps, state, topic_name)
        if quiz_item is not None:
            out.quiz_item = quiz_item
            state.active_problem_id = quiz_item.id
            canonical = canonical or quiz_item

    if canonical is None and withholding:
        canonical = _topic_fallback(deps, topic_name)

    out.plan = plan
    out.canonical = canonical
    out.knowledge_guidance = selection.knowledge_guidance
    out.topic_name = topic_name


def _resolve_canonical(
    request: TurnRequest,
    deps: TurnDependencies,
    state: SessionState,
    topic_name: str,
) -> Problem | None:
    """The problem this turn is about, remembered across the conversation.

    Re-deriving the problem from each message is what left follow-up turns
    unprotected: "just give me the answer" names no equation, so the lookup
    missed, and with nothing to scan for the guardrail had nothing to do — while
    the statement was still in the model's context from the previous turn. A
    problem therefore stays in scope until a *different* one matches, which is
    also how a tutor thinks about it.
    """
    found = deps.canonical_lookup(request.course_id, topic_name or None, request.message)
    if found is not None:
        state.active_problem_id = found.id
        return found

    if state.active_problem_id and deps.problem_by_id is not None:
        remembered = deps.problem_by_id(state.active_problem_id)
        if remembered is not None:
            return remembered
        # The key file changed under a live session; forget rather than carry a
        # dangling id forward.
        state.active_problem_id = None
    return None


def _select_quiz_item(
    deps: TurnDependencies, state: SessionState, topic_name: str
) -> Problem | None:
    """The practice problem this quiz turn asks.

    **A question in scope stays in scope until it is answered.** An unattempted
    item is reused rather than replaced: the student pressing "before I try,
    what's the answer?" is still working the question they were just asked, and
    swapping in a new one would leave the guardrail armed on an answer nobody
    asked for while the real one walked out. That was the whole of PS12 — the
    scan was live, just aimed at the wrong problem, which is worse than no scan
    at all: an absent guard at least leaves the counters honest, while a
    misaimed one reports a save on every turn it is failing to protect.

    A fresh item is chosen only when nothing is in scope, preferring one this
    student has not seen. Returns None when the topic has no keys: the model then
    invents a question, which cannot be guarded, and the turn is recorded as
    `buffered_unscanned` rather than treated as safe.
    """
    if deps.problems_for_topic is None or not topic_name:
        return None
    candidates = deps.problems_for_topic(topic_name)
    if not candidates:
        return None

    in_scope = {problem.id: problem for problem in candidates}.get(state.active_problem_id or "")
    if in_scope is not None and in_scope.id not in state.solved_problem_ids:
        return in_scope

    unseen = [
        problem
        for problem in candidates
        if problem.id not in state.asked_problem_ids and problem.id not in state.solved_problem_ids
    ]
    chosen = (unseen or candidates)[0]
    state.asked_problem_ids.append(chosen.id)
    return chosen


def _topic_fallback(deps: TurnDependencies, topic_name: str) -> Problem | None:
    """An answer key for the topic, when nothing more specific resolved.

    Deliberately narrow. Matching on topic alone was rejected as the *primary*
    rule for a good reason — it would attach "x = 3" to every Two-Step Linear
    Equations turn and make the tutor withhold a number the student never asked
    about. That reasoning still holds, so this runs only where the statement
    match and the session's memory have both failed **and** the plan is already
    withholding: the turn was going to withhold regardless, and arming the scan
    on a plausible key can only add protection. It is not made sticky, because a
    topic-level guess is weaker evidence than an identified problem and should
    not follow the conversation.
    """
    if deps.problems_for_topic is None or not topic_name:
        return None
    candidates = deps.problems_for_topic(topic_name)
    return candidates[0] if candidates else None


async def _evaluate(
    request: TurnRequest,
    deps: TurnDependencies,
    chunks: list[RetrievedChunk],
    canonical: Problem | None,
    settings: Settings,
) -> EvaluationResult:
    """Step 2c. Judge the attempt, or say plainly that it cannot be judged.

    Without an answer key there is nothing to compare against; `cannot_evaluate`
    is a legitimate verdict routing to S07 Check Understanding rather than to
    invented feedback (Tech Spec §5.2).
    """
    if canonical is None:
        return EvaluationResult(
            evaluation=Evaluation.CANNOT_EVALUATE,
            reason="no canonical solution is on file for the problem in this message",
            source=EvaluationSource.NO_ATTEMPT,
        )
    return await deps.evaluate(canonical, request.message, chunks, settings=settings)


def _topic_resolution(
    match: TopicMatch | None, level: StudentLevel, deps: TurnDependencies
) -> TopicResolution:
    if match is None:
        return TopicResolution(student_level=level)
    node_id = match.topic.id
    if deps.klmap is not None and deps.klmap.node(node_id) is None:
        node_id = ""
    return TopicResolution(
        topic=match.topic_name,
        topic_th=match.topic.name_th,
        node_id=node_id or None,
        matched_by=match.matched_on,
        confidence=match.confidence,
        ambiguous=match.ambiguous,
        resolved_by_context=match.resolved_by_context,
        candidates=[candidate.name for candidate in match.candidates],
        student_level=level,
        level_source=(
            "onboarding_questionnaire" if level is not StudentLevel.BEGINNER else "default_beginner"
        ),
    )


def _kl_lookup(
    deps: TurnDependencies, match: TopicMatch | None, learner_state: LearnerState
) -> KnowledgeContext | None:
    if deps.klmap is None or match is None:
        return None
    node = deps.klmap.node(match.topic.id) or find_start_node(deps.klmap, match.topic_name)
    if node is None:
        return None
    return traverse(deps.klmap, node, learner_state)


def _policy_applied(selection: StrategySelection, deps: TurnDependencies) -> PolicyApplied | None:
    """The governing policy row, read from the table rather than re-described.

    The selector records which rows applied; their permissions live on the rows
    themselves, so this looks them up instead of restating them.
    """
    decision = selection.context.get("decision_trace")
    if not isinstance(decision, dict) or not decision.get("policy_row"):
        return None

    row_id = str(decision["policy_row"])
    contexts = [str(context) for context in decision.get("policy_contexts") or []]
    applied = [str(row) for row in decision.get("policy_rows_applied") or []]
    actions = [
        action for action in decision.get("policy_actions") or [] if isinstance(action, dict)
    ]

    row = next((candidate for candidate in deps.tables.policy_rows if candidate.id == row_id), None)
    return PolicyApplied(
        row=row_id,
        context=row.context if row else (contexts[0] if contexts else ""),
        direct_answer=row.direct_answer if row else "",
        attempt=row.attempt if row else "",
        check=row.check if row else "",
        precedence=row.precedence if row else 0,
        also_applied=[other for other in applied if other != row_id],
        override_applied=bool(actions),
        override_notes=[
            str(action.get("reason") or action.get("rule") or "") for action in actions
        ],
        prohibited_strategies=[
            str(strategy) for strategy in decision.get("policy_prohibited_strategies") or []
        ],
    )


def _combination_verdict(selection: StrategySelection) -> CombinationVerdict | None:
    """What the Combination Rules did to the strategy set."""
    decision = selection.context.get("decision_trace")
    if not isinstance(decision, dict) or "combination_rule" not in decision:
        return None

    actions = [
        action for action in decision.get("combination_actions") or [] if isinstance(action, dict)
    ]
    return CombinationVerdict(
        rule=str(decision.get("combination_rule") or ""),
        sequence=[str(item) for item in decision.get("combination_sequence") or []],
        dropped=[str(action["dropped"]) for action in actions if action.get("dropped")],
        actions=actions,
    )


def _record_decision_trace(
    trace: TurnTrace, record: TurnRecord, selection: StrategySelection
) -> None:
    """Copy the selector's decision trace onto the turn record.

    Both the trigger rows that matched and the KM rules that matched but *lost*
    are recorded: §3.5's O2 tie-break says KM02 beats KM03 and requires both be
    logged, and a reviewer cannot judge a turn from the winner alone.
    """
    decision = selection.context.get("decision_trace")
    if not isinstance(decision, dict):
        return

    selected = decision.get("selected_trigger_row")
    rows = [row for row in decision.get("matched_trigger_rows") or [] if isinstance(row, dict)]
    record.trigger_matches = [TriggerMatch.model_validate(row) for row in rows]
    # The spec's §7 field is a list of ids; it is projected from the structured
    # contest above rather than built alongside it, so the two cannot disagree.
    trace.matched_trigger_rows = [match.row_id for match in record.trigger_matches]
    if not trace.matched_trigger_rows and selected:
        trace.matched_trigger_rows.append(str(selected))

    matches = [match for match in decision.get("km_matches") or [] if isinstance(match, dict)]
    record.km_matches = [KmMatch.model_validate(match) for match in matches]
    trace.matched_km_rules = [match.rule_id for match in record.km_matches]

    if isinstance(decision.get("table_versions"), dict):
        # `setdefault`, not `update`: the selector reports declared versions, and
        # the trace already holds `audit_versions` — version *plus content hash*.
        # Overwriting would downgrade evidence to a claim, which is the one thing
        # an audit of "which policy was in force" cannot accept.
        for name, version in decision["table_versions"].items():
            trace.table_versions.setdefault(name, version)


async def _generate_and_guard(
    request: TurnRequest,
    deps: TurnDependencies,
    state: SessionState,
    record: TurnRecord,
    decision: _Decision,
) -> AsyncIterator[TurnEvent]:
    """Steps 5 and 6: generate, guard, emit.

    One value decides everything here: `withholding`, meaning no block in this
    plan may state the solution. It gates the prompt (the answer is not sent),
    the emission mode (buffer, do not stream) and the scan (look for a leak).
    They were three separate conditions and drifted apart, which is how a turn
    ended up streaming unscanned while its plan said withhold.

    A withholding turn is generated in full and guarded before any of it is
    emitted. Tech Spec §6 puts the guardrail "before the student sees text", and
    an append-only token stream cannot be retracted, so safety is bought with a
    later first token — cheap at the 50–120 word caps the plans set, and the
    trace patches stream throughout. Turns that may state the solution have
    nothing to withhold and stream live.

    **Fail closed**: withholding with no canonical answer resolved still
    buffers. The scan cannot run without something to scan for, so the turn is
    recorded as protected-but-unscanned rather than quietly taking the weakest
    path.
    """
    trace = record.trace
    plan, canonical = decision.plan, decision.canonical
    withholding = not _may_state_solution(plan, deps)
    for_prompt = None if withholding else canonical
    note = ""
    attempt = 1

    if withholding and (canonical is None or not canonical.answer):
        trace.guardrail_events.append(
            GuardrailEvent(
                type=GuardrailEventType.OTHER,
                detail=(
                    "protection applied without a known canonical answer: the turn was "
                    "buffered and vetted for plan violations, but no leak scan was "
                    "possible because no answer key matched this problem"
                ),
                action_taken="buffered_unscanned",
            )
        )

    while True:
        stats = StreamStats()
        parts: list[str] = []

        with trace.step("generation" if attempt == 1 else "generation_retry"):
            async for delta in deps.generate(
                plan,
                request.message,
                topic=decision.topic_name or "this topic",
                learner_state=(
                    trace.intent_result.learner_state
                    if trace.intent_result
                    else LearnerState.NORMAL
                ),
                knowledge_context=trace.knowledge_context,
                knowledge_guidance=decision.knowledge_guidance,
                chunks=trace.retrieved_chunks,
                canonical=for_prompt,
                quiz_item=decision.quiz_item,
                conversation=state.conversation(),
                system_note=note,
                settings=deps.settings,
                stats=stats,
            ):
                parts.append(delta)
                if not withholding:
                    yield TurnEvent("token", {"text": delta})

        draft = "".join(parts)
        _record_generation_stats(trace, stats, deps)

        with trace.step("guardrail" if attempt == 1 else "guardrail_retry"):
            verdict = check_output(
                draft,
                plan,
                turn_id=trace.turn_id,
                config=deps.tables.guardrail,
                rules=deps.tables.global_rules,
                canonical_answer=canonical.answer if (withholding and canonical) else None,
                # §6 check 2, independent of the plan's flag: a practice item's
                # answer is never revealed before the student has attempted it.
                quiz_answer_key=decision.quiz_item.answer if decision.quiz_item else None,
                problem_id=canonical.id if canonical else "",
                attempt=attempt,
            )
        _record_guardrail_events(trace, verdict.events_for_trace(), canonical)
        record.scan_coverage = verdict.coverage.as_dict()
        record.withheld_solution = withholding

        if verdict.action is GuardrailAction.REGENERATE and withholding:
            note = verdict.regeneration_note or note
            attempt += 1
            yield TurnEvent("trace", record=record)
            continue

        if withholding:
            trace.generated_text = verdict.text
            for token in _as_tokens(verdict.text):
                yield TurnEvent("token", {"text": token})
        else:
            # Nothing was withheld, so the draft has already gone out as it was
            # generated; the guardrail's counts are recorded, not enforced.
            trace.generated_text = draft
        return


def _record_guardrail_events(
    trace: TurnTrace, events: list[GuardrailEvent], canonical: Problem | None
) -> None:
    """Append guardrail events with the withheld answer stripped out of them.

    A leak report names what it caught, which puts the canonical answer into
    `detail` — and the trace is both streamed to the browser and written to
    `data/sessions/`. Every guardrail *save* would otherwise hand the client the
    answer it had just prevented the model from giving. Redacting at the point
    of recording covers the wire, the disk and the logs at once, rather than
    once per destination.
    """
    for event in events:
        trace.guardrail_events.append(
            event.model_copy(
                update={
                    "detail": redact_answer(event.detail, canonical),
                    "action_taken": redact_answer(event.action_taken, canonical),
                }
            )
        )


def redact_answer(text: str, canonical: Problem | None) -> str:
    """Replace any part of the canonical solution appearing in `text`.

    Longest secrets first, so replacing a step does not leave the answer behind
    inside it. Bare numeric keys are replaced only where they stand alone, or
    "3" would eat the 3 in "step 3" and make the message unreadable.
    """
    if not text or canonical is None:
        return text

    secrets = [
        secret
        for secret in [canonical.answer, *canonical.canonical_steps, *canonical.guardrail_tokens]
        if secret and secret.strip()
    ]
    for secret in sorted(set(secrets), key=len, reverse=True):
        pattern = re.escape(secret.strip())
        if _NUMBER.fullmatch(secret.strip()):
            pattern = rf"(?<![\w.]){pattern}(?![\w.])"
        text = re.sub(pattern, REDACTED, text, flags=re.IGNORECASE)
    return text


def _record_prompt_version(trace: TurnTrace, role: str, prompt: str) -> None:
    """Record the version of the prompt this role was given (Tech Spec §7).

    **Hash the authored text, not the assembled prompt.** For generation that
    means `ROLE_LAYER` alone: the policy, state and content layers are per-turn
    data, so hashing the assembly would mint a new "version" on every single turn
    and attribute nothing. A version that changes constantly is as useless as one
    that never changes — it just fails in the other direction. What this has to
    answer is "did someone edit the prompt between these two turns?", and only
    the authored text can answer that.
    """
    trace.prompt_versions[role] = prompt_version(prompt)


def _may_state_solution(plan: ResponsePlan, deps: TurnDependencies) -> bool:
    """Whether this plan has a block that will state the solution.

    `full_solution_allowed` alone is not enough to justify putting the canonical
    answer in Call C's context. Under the normal-learning policy the flag stays
    True even when the planner has already dropped every solution block — which
    it does whenever the turn is hint-first (§4.2) — and a prompt carrying an
    answer no block may state is a leak surface with no upside. The narrower
    question, "will anything in this plan say the answer?", is what §5.3's
    structural rule is really asking, and it is the single value the prompt, the
    emission mode and the scan all read.
    """
    if not plan.full_solution_allowed:
        return False
    templates = deps.tables.response_templates
    return any(
        templates.block(name).is_solution for name in plan.structure if name in templates.blocks
    )


def _record_generation_stats(trace: TurnTrace, stats: StreamStats, deps: TurnDependencies) -> None:
    trace.model_ids["generation"] = stats.model or deps.resolved_settings().model_for("generation")
    _record_prompt_version(trace, "generation", ROLE_LAYER)
    trace.record_usage(
        "generation",
        input_tokens=stats.usage.input_tokens,
        output_tokens=stats.usage.output_tokens,
    )
    if stats.ttft_ms is not None:
        trace.latencies_ms["generation_ttft"] = stats.ttft_ms


def _as_tokens(text: str, size: int = BUFFERED_TOKEN_SIZE) -> list[str]:
    """Chunk a buffered reply so the UI still renders it progressively."""
    return [text[index : index + size] for index in range(0, len(text), size)] or [""]


def _persist(
    deps: TurnDependencies,
    state: SessionState,
    request: TurnRequest,
    record: TurnRecord,
    text: str,
) -> None:
    """Append the turn to the session and store it. Never raises into the stream."""
    state.student_id = state.student_id or request.student_id
    state.course_id = state.course_id or request.course_id
    # Solving a problem does **not** release the guardrail, however much the
    # opposite sounds like good teaching. Policy withholds because of the
    # assignment, not because of what the student happens to know: "I got x = 3,
    # right?" followed by "just give me the number" is one homework thread, and
    # clearing scope there re-opened the exact hole this work closed (PS08). What
    # a correct answer changes is only which question gets asked next.
    solved = state.active_problem_id
    if (
        record.trace.evaluation is Evaluation.CORRECT
        and solved
        and solved not in state.solved_problem_ids
    ):
        state.solved_problem_ids.append(solved)
    state.messages.append(SessionMessage(role="student", text=request.message))
    if text:
        state.messages.append(SessionMessage(role="tutor", text=text, turn_id=record.trace.turn_id))
    state.turns.append(record)
    try:
        deps.sessions.save(state)
    except Exception as exc:  # noqa: BLE001 - the repository is pluggable, so its
        # failure modes are open; losing the trace must never cost the student the
        # reply that was already generated.
        logger.error("could not persist session %s: %s", state.session_id, exc)


__all__ = [
    "BUFFERED_TOKEN_SIZE",
    "CONTEXT_MESSAGES",
    "AnswerKeyLookup",
    "InMemorySessionRepository",
    "JsonFileSessionRepository",
    "SessionMessage",
    "SessionRepository",
    "SessionState",
    "TopicResolution",
    "TurnDependencies",
    "TurnEvent",
    "TurnRecord",
    "TurnRequest",
    "execute_turn",
    "redact_answer",
    "run_turn",
]
