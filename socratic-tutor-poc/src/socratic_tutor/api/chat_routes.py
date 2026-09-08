"""Chat endpoints: the streamed turn and the session read-back (Tech Spec §7).

    POST /chat/turn            → SSE stream of `token` | `trace` | `done` | `error`
    GET  /tutor/session/{id}   → session context plus every turn record

Mounted by `api.app` under the `/api` prefix (Decisions log, 2026-09-01 ruling 3),
which is where the SPA's `API_BASE` points.

`to_wire_trace` translates the pipeline's `TurnRecord` into the shape
`web/src/types.ts` declares. The two differ on purpose: the Python side is the
spec's §7 record, the TypeScript side is what the inspector renders (node ids
for highlighting, per-step engine labels for the colour bands, relative event
timings). Translating at the edge keeps display concerns out of the pipeline and
means neither side has to change when the other's presentation does.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from socratic_tutor.api.klmap_routes import (
    COURSE_ID_MAX_LENGTH,
    COURSE_ID_PATTERN,
    SEED_COURSE_ID,
    live_path,
    seed_klmap_path,
)
from socratic_tutor.config import get_settings
from socratic_tutor.domain.klmap import KLMap, KLMapValidationError, load_klmap
from socratic_tutor.models.enums import GuardrailEventType
from socratic_tutor.models.trace import GuardrailEvent, TurnTrace
from socratic_tutor.orchestrator import (
    JsonFileSessionRepository,
    TopicResolution,
    TurnDependencies,
    TurnRecord,
    TurnRequest,
    run_turn,
)
from socratic_tutor.pedagogy.generation import detect_language
from socratic_tutor.pedagogy.student_model import (
    JsonFileStudentModelRepository,
    StudentModelError,
    StudentModelService,
)
from socratic_tutor.pedagogy.tables import get_tables
from socratic_tutor.pedagogy.topics import (
    Syllabus,
    SyllabusTopic,
    derive_aliases,
    load_syllabus,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

#: Per-step display metadata for the inspector's timing band. `engine` follows
#: the design doc's colours: amber = LLM, blue = deterministic, green = data,
#: red = policy/guardrail.
STEP_META: dict[str, tuple[str, str, str]] = {
    "intent": ("intent", "Intent detection", "llm"),
    "topic_level": ("topic_level", "Topic + student level", "deterministic"),
    "retrieval": ("retrieval", "RAG retrieval", "data"),
    "kl_lookup": ("kl_lookup", "KL Map lookup", "data"),
    "answer_evaluation": ("answer_evaluation", "Answer evaluation", "llm"),
    "strategy_selection": ("strategy_selection", "Strategy selection", "deterministic"),
    "response_planning": ("response_planning", "Response planning", "deterministic"),
    "generation": ("generation", "Response generation", "llm"),
    "generation_retry": ("generation", "Response generation (retry)", "llm"),
    "guardrail": ("guardrail", "Output guardrail", "policy"),
    "guardrail_retry": ("guardrail", "Output guardrail (retry)", "policy"),
}

#: Guardrail event names the UI knows, where they differ from the Python enum.
_UI_EVENT_NAMES: dict[GuardrailEventType, str] = {
    GuardrailEventType.REGENERATED: "leak_regenerated",
}

_SEVERITY: dict[GuardrailEventType, str] = {
    GuardrailEventType.LEAK_BLOCKED: "critical",
    GuardrailEventType.FALLBACK_SERVED: "warning",
    GuardrailEventType.REGENERATED: "warning",
    GuardrailEventType.PLAN_VIOLATION: "warning",
    GuardrailEventType.OTHER: "info",
}

_dependencies: TurnDependencies | None = None


class ChatTurnRequest(BaseModel):
    """`POST /chat/turn` body — matches `ChatTurnRequest` in web/src/types.ts."""

    session_id: str
    student_id: str
    #: Names a file (`klmap-<course_id>.yaml`) and a Chroma collection, so it is
    #: bounded and restricted to the path-safe alphabet here, at the edge, and a
    #: bad id is a 422 rather than a filesystem error mid-turn.
    course_id: str = Field(min_length=1, max_length=COURSE_ID_MAX_LENGTH, pattern=COURSE_ID_PATTERN)
    message: str = Field(min_length=1)


#: Where per-app dependencies live on the FastAPI instance.
STATE_ATTRIBUTE = "turn_dependencies"


def install_dependencies(app: FastAPI, dependencies: TurnDependencies) -> None:
    """Attach dependencies to one app, rather than to the process.

    Preferred over `set_dependencies`: two apps in one process (a test suite, a
    future multi-tenant deployment) then cannot read each other's syllabus,
    student store or session repository.
    """
    setattr(app.state, STATE_ATTRIBUTE, dependencies)


def set_dependencies(dependencies: TurnDependencies | None) -> None:
    """Install process-wide dependencies, used when no app-scoped set exists.

    Passing None restores the lazily built default, which reads the seed content
    and writes sessions and student models under the configured data directory.
    """
    global _dependencies
    _dependencies = dependencies


def get_dependencies(request: Request | None = None) -> TurnDependencies:
    """This request's dependencies.

    App-scoped first, then the process-wide override, then the lazily built
    default from seed content — so `create_app` needs no wiring, but can take
    control with one call to `install_dependencies`.
    """
    if request is not None:
        scoped = getattr(request.app.state, STATE_ATTRIBUTE, None)
        if scoped is not None:
            return scoped
    return _dependencies if _dependencies is not None else _default_dependencies()


@lru_cache(maxsize=1)
def _default_dependencies() -> TurnDependencies:
    """The seed course, built once per process.

    This is the *fallback* tier, not the course resolver: `_for_course` swaps the
    syllabus and map per request when the course has a published live map. It
    stays a singleton because everything else in it — tables, student store,
    session repository — is course-independent.
    """
    settings = get_settings()
    seed = settings.content_dir / "seed"
    return TurnDependencies(
        syllabus=load_syllabus(seed / "syllabus-linear-equations.yaml"),
        tables=get_tables(),
        klmap=_load_klmap(seed_klmap_path()),
        students=StudentModelService(
            JsonFileStudentModelRepository(settings.data_dir / "students")
        ),
        sessions=JsonFileSessionRepository(settings.data_dir / "sessions"),
        settings=settings,
    )


def _load_klmap(path: Any) -> KLMap | None:
    """The course map, or None when it is absent or invalid.

    A broken map degrades the turn to RAG-only rather than refusing to serve;
    the failure is logged and shows up as an empty knowledge context in the
    inspector.
    """
    try:
        return load_klmap(path)
    except (KLMapValidationError, OSError, ValueError) as exc:
        logger.warning("KL map unavailable (%s); turns will run without it", exc)
        return None


# ------------------------------------------------------ per-course resolution

#: Live maps successfully read, keyed by file, with the mtime they were read at.
#: Only successes are cached: a rejected read is retried on the next turn, so a
#: torn or half-written file can never pin the seed to a course (review m1/L2).
_live_courses: dict[Path, tuple[int, KLMap, Syllabus]] = {}

#: Files already warned about, by the mtime that was rejected — the warning
#: fires once per version of a broken file, not once per turn.
_rejected_live: dict[Path, int] = {}


def _for_course(base: TurnDependencies, course_id: str) -> TurnDependencies:
    """`base` with its syllabus and map swapped for the course's published live map.

    Approval publishes to `klmap_routes.live_path(course_id)`; until this
    function existed nothing in the chat path read that location, so every turn
    taught from the seed whatever course the SPA named (diagnosis 2026-09-02,
    bug B). Resolution happens here, at the request boundary, so the installed
    `TurnDependencies` stays course-blind and the orchestrator is untouched.

    Degrades quietly: no live file returns `base` unchanged — the normal case
    for the seed course, not logged. A live file the loader rejects also returns
    `base`, logged once per version of that file. Refusing the turn would punish
    the student for a reviewer's YAML edit.

    Cached by file mtime. Approval rewrites the file (atomically, see
    `save_klmap`), so the next turn after an approval sees the new map without a
    restart or an explicit invalidation. Note `live_path` is exact, not slugged:
    `ALG 101` is a 422 at the request model, never a second key for `ALG-101`.
    """
    try:
        live = live_path(course_id)
    except StudentModelError:
        # Never from a validated request body; a legacy session with an empty
        # course id reads back from the seed rather than failing.
        return base
    try:
        mtime = live.stat().st_mtime_ns
    except OSError:
        return base

    cached = _live_courses.get(live)
    if cached is None or cached[0] != mtime:
        try:
            klmap = load_klmap(live)
        except (KLMapValidationError, OSError, ValueError) as exc:
            if _rejected_live.get(live) != mtime:
                _rejected_live[live] = mtime
                logger.warning(
                    "live map %s for course %s rejected (%s); turns fall back to the seed map",
                    live,
                    course_id,
                    exc,
                )
            return base
        _rejected_live.pop(live, None)
        _announce_live_map(course_id, klmap, first=cached is None)
        cached = (mtime, klmap, _syllabus_from(klmap))
        _live_courses[live] = cached

    _, klmap, syllabus = cached
    return dataclasses.replace(base, klmap=klmap, syllabus=syllabus)


def _announce_live_map(course_id: str, klmap: KLMap, *, first: bool) -> None:
    """One log line when a live map starts (or resumes) teaching a course.

    The first load of the seed's own id is the case worth reading twice: every
    SPA upload is filed under it (S4), so the first approval displaces the seed
    syllabus and map while the seed answer keys stay armed (TECH_DEBT TD6).
    """
    verb = "now teaches from" if first else "reloaded"
    displaced = (
        "; the seed syllabus and map are displaced for this id, its answer keys are not (TD6)"
        if course_id == SEED_COURSE_ID
        else ""
    )
    logger.info(
        "course %s %s live map %r (%d nodes)%s",
        course_id,
        verb,
        klmap.course_name,
        len(klmap.nodes),
        displaced,
    )


def _syllabus_from(klmap: KLMap) -> Syllabus:
    """A course's topic list read off its map.

    The seed pack keeps a hand-written syllabus whose ids match the map's node
    ids (Tech Spec §2.2); an ingested course has only the map. `SyllabusTopic`
    is `KLNode` plus student-typed `aliases`. Extraction does not produce those,
    and without them `resolve_topic`'s mention tier needs the whole label —
    "Calvin cycle (light-independent reactions)" — verbatim in the message,
    which no student types; the live verification saw `topic=""` on every turn
    of an ingested course. `derive_aliases` splits each label into the parts a
    student would actually name.
    """
    return Syllabus(
        course_id=klmap.course_id,
        course_name=klmap.course_name,
        topics=[
            SyllabusTopic(
                id=node.id,
                name=node.name,
                name_th=node.name_th,
                aliases=derive_aliases(node.name),
            )
            for node in klmap.nodes
        ],
    )


@router.post("/chat/turn")
async def chat_turn(body: ChatTurnRequest, request: Request) -> StreamingResponse:
    """Run one turn and stream it.

    Frames are written by hand rather than through a helper so the exact bytes
    the SPA's parser expects are visible here: `event:` then `data:` then a
    blank line.
    """
    deps = _for_course(get_dependencies(request), body.course_id)
    turn = TurnRequest(
        session_id=body.session_id,
        student_id=body.student_id,
        course_id=body.course_id,
        message=body.message,
    )

    async def frames() -> AsyncIterator[str]:
        try:
            async for event in run_turn(turn, deps):
                # `error` carries `{message}` because that is what the SPA's
                # handler reads; the partial trace follows in the `done` frame.
                if event.event in ("trace", "done") and event.record is not None:
                    payload: dict[str, Any] = to_wire_trace(event.record, deps.klmap)
                else:
                    payload = dict(event.data)
                yield _frame(event.event, payload)
        except Exception as exc:  # the stream is already open; report inside it
            logger.exception("chat turn stream failed")
            yield _frame("error", {"message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # nginx buffers SSE into uselessness without this.
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/tutor/session/{session_id}")
async def get_session(session_id: str, request: Request) -> dict[str, Any]:
    """Session context plus every turn's trace (Tech Spec §7).

    This is the endpoint R12's acceptance criterion leans on: given a turn id,
    the whole teaching decision has to be reconstructable from what comes back
    here, with no other source.
    """
    base = get_dependencies(request)
    try:
        state = base.sessions.get(session_id)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if state is None:
        raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
    # The session knows its course; label node ids from that course's map, not
    # the seed's, or the read-back would disagree with the stream it replays.
    deps = _for_course(base, state.course_id)

    return {
        "session_id": state.session_id,
        "student_id": state.student_id,
        "course_id": state.course_id,
        "created_at": state.created_at.isoformat(),
        "messages": [message.model_dump(mode="json") for message in state.messages],
        "traces": [to_wire_trace(record, deps.klmap) for record in state.turns],
    }


def _frame(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ------------------------------------------------------------ wire mapping


def to_wire_trace(record: TurnRecord, klmap: KLMap | None = None) -> dict[str, Any]:
    """A `TurnRecord` as the inspector's `TurnTrace` (web/src/types.ts).

    Only fields the pipeline actually produced are included; the TypeScript
    interface marks the rest optional, and an absent field is honest where an
    empty one would look like a decision that was made.
    """
    trace = record.trace
    wire: dict[str, Any] = {
        "turn_id": trace.turn_id,
        "session_id": trace.session_id,
        "created_at": trace.timestamp.isoformat(),
        "student_message": trace.student_message,
        "language": (
            trace.response_plan.language.value
            if trace.response_plan
            else detect_language(trace.student_message).value
        ),
    }

    if trace.intent_result is not None:
        intent = trace.intent_result
        wire["intent"] = {
            "intent": intent.intent.value,
            "learner_state": intent.learner_state.value,
            "special_handling": {
                "detected": intent.special_handling.detected,
                "type": intent.special_handling.type.value,
            },
            "confidence": intent.confidence,
            "fallback_applied": intent.low_confidence_fallback,
            **(
                {
                    "fallback_reason": (
                        f"classifier said {intent.raw_intent.value} at "
                        f"{intent.confidence:.2f}; below threshold, fell back to explain"
                    )
                }
                if intent.low_confidence_fallback and intent.raw_intent
                else {}
            ),
        }

    if record.topic_resolution.topic or trace.topic:
        wire["topic"] = _wire_topic(record.topic_resolution, trace)

    if trace.retrieved_chunks:
        wire["retrieval"] = [
            {
                "chunk_id": f"{chunk.source_ref}#{index}",
                "score": chunk.score,
                "text": chunk.text,
                "source_ref": chunk.source_ref,
                "language": chunk.lang.value,
            }
            for index, chunk in enumerate(trace.retrieved_chunks)
        ]

    if trace.knowledge_context is not None:
        wire["knowledge_context"] = _wire_knowledge(trace, klmap)

    if record.km_matches:
        wire["km_rules"] = [match.model_dump(mode="json") for match in record.km_matches]

    if trace.evaluation is not None:
        wire["answer_evaluation"] = _wire_evaluation(record)

    if record.trigger_matches:
        wire["trigger_matches"] = [
            match.model_dump(mode="json") for match in record.trigger_matches
        ]
    elif trace.matched_trigger_rows:
        # Fallback row: no rule matched, so there is no contest to show.
        wire["trigger_matches"] = [
            {"row_id": row_id, "matched": False, "selected": True}
            for row_id in trace.matched_trigger_rows
        ]

    if trace.strategy_selection is not None:
        wire["strategy"] = _wire_strategy(trace)

    if record.teaching_policy is not None:
        # The red policy card: the single most safety-relevant decision on the
        # panel, and previously inferable only from a string in the versions map.
        wire["teaching_policy"] = record.teaching_policy.model_dump(mode="json")

    if record.combination is not None:
        wire["combination"] = {
            **record.combination.model_dump(mode="json"),
            "allowed": record.combination.allowed,
        }

    if trace.response_plan is not None:
        wire["response_plan"] = {
            **trace.response_plan.model_dump(mode="json"),
            **({"template_id": record.response_template} if record.response_template else {}),
        }

    if trace.guardrail_events:
        wire["guardrail"] = [_wire_guardrail(event, trace) for event in trace.guardrail_events]

    # The scanner's scope for this turn, as the scanner reported it. Shown next
    # to the guardrail result so a clean scan is never read as a proven-clean
    # turn — the glass box should not be less honest than the eval suite.
    if record.scan_coverage:
        # Paired with `withheld_solution`, because "nothing scanned" reads as a
        # pass on a turn that had nothing to withhold and as a warning on one
        # that did.
        wire["scan_coverage"] = {
            **record.scan_coverage,
            "withheld_solution": record.withheld_solution,
        }

    timings = [
        {"step": step, "label": label, "engine": engine, "ms": round(trace.latencies_ms[key], 2)}
        for key, (step, label, engine) in STEP_META.items()
        if key in trace.latencies_ms
    ]
    if timings:
        wire["timings"] = timings

    wire["versions"] = {
        "models": dict(trace.model_ids),
        "tables": dict(trace.table_versions),
        "prompts": dict(trace.prompt_versions),
    }
    if "pre_stream" in trace.latencies_ms:
        wire["pre_stream_ms"] = round(trace.latencies_ms["pre_stream"], 2)
    wire["total_ms"] = round(trace.total_latency_ms, 2)
    if trace.generated_text:
        wire["generated_text"] = trace.generated_text
    if trace.error:
        wire["error"] = trace.error

    return wire


def _wire_topic(resolution: TopicResolution, trace: TurnTrace) -> dict[str, Any]:
    return {
        "topic": resolution.topic or trace.topic or "",
        "topic_th": resolution.topic_th,
        "node_id": resolution.node_id,
        "student_level": (trace.student_level or resolution.student_level).value,
        "matched_by": resolution.matched_by.value if resolution.matched_by else "unmatched",
        "level_source": resolution.level_source,
        # Additive: the inspector shows a resolution the student never confirmed
        # differently from one they did.
        "confidence": resolution.confidence,
        "ambiguous": resolution.ambiguous,
        "resolved_by_context": resolution.resolved_by_context,
        "candidates": resolution.candidates,
    }


def _wire_knowledge(trace: TurnTrace, klmap: KLMap | None) -> dict[str, Any]:
    context = trace.knowledge_context
    assert context is not None
    by_name = {node.name.casefold(): node.id for node in klmap.nodes} if klmap is not None else {}

    def refs(items: Any) -> list[dict[str, Any]]:
        return [
            {
                "topic": item.topic,
                "node_id": by_name.get(item.topic.casefold()),
                "distance": item.distance,
            }
            for item in items
        ]

    prerequisites = refs(context.prerequisites)
    related = refs(context.related_topics)
    following = refs(context.next_topics)
    current_id = by_name.get(context.current_topic.casefold())
    hit = [current_id, *(ref["node_id"] for ref in [*prerequisites, *related, *following])]

    return {
        "current_topic": context.current_topic,
        "current_node_id": current_id,
        "prerequisites": prerequisites,
        "related_topics": related,
        "next_topics": following,
        "relationships": context.relationships,
        "max_depth": max((ref["distance"] for ref in prerequisites), default=1),
        "nodes_hit": [node_id for node_id in hit if node_id],
    }


def _wire_evaluation(record: TurnRecord) -> dict[str, Any]:
    """The answer verdict and how it was reached (Tech Spec §5.2).

    `deterministic_checker_used` reads `EvaluationSource.is_deterministically_backed`
    rather than listing the sources here. A second list of the same fact drifts:
    this one already did, missing `llm_confirmed` when that member was split out,
    which under-reported the checker on exactly the turns where it agreed with
    the model. The property is the single source of truth.

    `error_locus` and `reason` arrive already redacted from the orchestrator, so
    a verdict cannot publish the answer it was judging against.
    """
    evaluation = record.answer_evaluation
    if evaluation is None:
        verdict = record.trace.evaluation
        return {
            "evaluation": verdict.value if verdict else "cannot_evaluate",
            "deterministic_checker_used": False,
        }

    return {
        "evaluation": evaluation.evaluation.value,
        "error_locus": evaluation.error_locus or None,
        "deterministic_checker_used": evaluation.source.is_deterministically_backed,
        "checker_note": evaluation.reason or None,
        "source": evaluation.source.value,
    }


def _wire_strategy(trace: TurnTrace) -> dict[str, Any]:
    selection = trace.strategy_selection
    assert selection is not None
    data = selection.model_dump(mode="json", exclude={"knowledge_guidance", "context"})
    guidance = selection.knowledge_guidance
    if guidance is not None and guidance.action is not None:
        data["knowledge_guidance"] = {
            "action": guidance.action.value,
            "target_concept": guidance.target_concept or "",
            # The UI's field is still called `relationship`; the value is the
            # closed `Relation` vocabulary plus the direction it was read in.
            "relationship": guidance.relation.value if guidance.relation else "",
            "direction": guidance.direction,
            "distance": guidance.distance if guidance.distance is not None else 0,
        }
    data["context"] = {
        "topic": selection.topic or trace.topic or "",
        "student_level": (
            selection.student_level.value
            if selection.student_level
            else (trace.student_level.value if trace.student_level else "beginner")
        ),
    }
    return data


def _wire_guardrail(event: GuardrailEvent, trace: TurnTrace) -> dict[str, Any]:
    """One guardrail event, renamed where the UI's vocabulary differs.

    There is no prefix-sniffing here any more: a clean scan used to be encoded as
    an `OTHER` event whose detail began `leak_scan_clean:` and decoded by
    splitting the string back apart. That producer is gone, and what it was
    conveying — which forms were checked — now comes from `scan_coverage`,
    reported by the detector itself.
    """
    return {
        "type": _UI_EVENT_NAMES.get(event.type, event.type.value),
        "severity": _SEVERITY.get(event.type, "info"),
        "detail": event.detail,
        "action_taken": event.action_taken,
        "at_ms": round(_elapsed_ms(trace.timestamp, event.at), 2),
    }


def _elapsed_ms(start: datetime, at: datetime) -> float:
    return max((at - start).total_seconds() * 1000.0, 0.0)


__all__ = [
    "STATE_ATTRIBUTE",
    "ChatTurnRequest",
    "get_dependencies",
    "install_dependencies",
    "router",
    "set_dependencies",
    "to_wire_trace",
]
