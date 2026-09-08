"""TurnTrace — the persisted record of one student turn (Tech Spec §7, R20).

**Every teaching decision must be reconstructable from the trace** (Tech Spec §7).
This object is also the glass-box inspector's only data source: the Addendum's
"what I interpreted / what I intend to do" narration is rendered from these
structured fields with **zero additional LLM calls**.

Consequences for anyone adding a field here: it must be populated by the step
that produces it (not reconstructed later), and every pipeline field must be
optional so a turn that failed at step 1 still serializes and still renders.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from socratic_tutor.models.enums import Evaluation, GuardrailEventType, StudentLevel
from socratic_tutor.models.intent import IntentResult
from socratic_tutor.models.knowledge import KnowledgeContext, RetrievedChunk
from socratic_tutor.models.plan import ResponsePlan
from socratic_tutor.models.strategy import StrategySelection


def _utc_now() -> datetime:
    return datetime.now(UTC)


class StepLatency(BaseModel):
    """Wall-clock cost of one pipeline step, in milliseconds (Tech Spec §7, §10).

    Use `StepLatency.measure` to record into a `TurnTrace.latencies_ms` dict; it
    writes the elapsed time even when the step raises, so a failed turn still
    carries the latency of the step that failed.
    """

    step: str
    ms: float

    @classmethod
    @contextmanager
    def measure(cls, step: str, sink: dict[str, float]) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            sink[step] = (time.perf_counter() - started) * 1000.0


class GuardrailEvent(BaseModel):
    """One deterministic guardrail firing (Tech Spec §6).

    Guardrail events are the primary POC quality metric and E3 counts hard leaks
    separately from guardrail saves, so `type` is a closed enum — a free string
    lets a typo silently zero a safety metric. Use
    `GuardrailEventType.OTHER` plus `detail` for a check the vocabulary does not
    name yet.
    """

    model_config = ConfigDict(use_enum_values=False)

    type: GuardrailEventType
    detail: str = ""
    action_taken: str = ""
    at: datetime = Field(default_factory=_utc_now)


class TurnTrace(BaseModel):
    """Everything one turn decided, in pipeline order (Tech Spec §1, §7).

    All pipeline fields are optional and default-empty. Population order mirrors
    the six steps; whatever is set is what the turn got to before it finished or
    failed.
    """

    # `model_ids` collides with pydantic's "model_" protected namespace; the
    # spec names the field, so the namespace is disabled rather than the field
    # renamed.
    model_config = ConfigDict(use_enum_values=False, protected_namespaces=())

    turn_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str
    timestamp: datetime = Field(default_factory=_utc_now)
    student_message: str

    # --- step 1: intent detection
    intent_result: IntentResult | None = None

    # --- step 2: topic + student level
    topic: str | None = None
    student_level: StudentLevel | None = None

    # --- step 2a/2b: retrieval and KL Map lookup
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    knowledge_context: KnowledgeContext | None = None

    # --- step 2c: answer evaluation (only when intent = check_answer)
    evaluation: Evaluation | None = None

    # --- step 3: strategy selection
    #: Trigger Matrix rows that matched, highest-priority first (Tech Spec §3.1).
    matched_trigger_rows: list[str] = Field(default_factory=list)
    #: KM rules that matched, including the loser of a KM02/KM03 tie-break,
    #: which §3.5 requires be logged.
    matched_km_rules: list[str] = Field(default_factory=list)
    strategy_selection: StrategySelection | None = None

    # --- step 4: response planning
    response_plan: ResponsePlan | None = None

    # --- step 5/6: generation and guardrail
    generated_text: str | None = None
    guardrail_events: list[GuardrailEvent] = Field(default_factory=list)

    # --- provenance and cost
    #: Milliseconds per step name, e.g. {"intent": 180.4}. Keys are step names,
    #: not numbers, so a reordered pipeline does not silently remap the data.
    latencies_ms: dict[str, float] = Field(default_factory=dict)
    #: Model ID actually used per role, e.g. {"intent": "claude-haiku-4-5"}.
    model_ids: dict[str, str] = Field(default_factory=dict)
    #: Version of each rule table read this turn (Tech Spec §7, /admin/tables).
    table_versions: dict[str, str] = Field(default_factory=dict)
    #: Content hash of each system prompt used, e.g. {"intent": "a1b2c3d4"}.
    #: §7 asks for prompt versions so a behaviour change can be attributed to a
    #: prompt edit later. Hashes rather than numbers because nothing bumps a
    #: version by hand, and an unbumped number is worse than no number.
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    #: Token usage per role, e.g. {"generation": {"input": 1200, "output": 90}}.
    token_usage: dict[str, dict[str, int]] = Field(default_factory=dict)
    #: Set when the turn aborted; the trace is still written and still renders.
    error: str | None = None

    def step(self, name: str) -> Any:
        """Context manager recording this step's latency into `latencies_ms`."""
        return StepLatency.measure(name, self.latencies_ms)

    def record_usage(self, role: str, *, input_tokens: int, output_tokens: int) -> None:
        """Accumulate token usage for `role` (a role may make more than one call)."""
        bucket = self.token_usage.setdefault(role, {"input": 0, "output": 0})
        bucket["input"] += input_tokens
        bucket["output"] += output_tokens

    @property
    def total_latency_ms(self) -> float:
        return sum(self.latencies_ms.values())


__all__ = ["GuardrailEvent", "StepLatency", "TurnTrace"]
