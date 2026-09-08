"""Ingestion API and its SSE progress stream (R17).

The browser contract, which `web/src/api/http.ts` and `web/src/types.ts` already
declare, is fixed and matched exactly here:

- `POST /api/ingest/upload` — multipart, repeated field `files`; returns
  `[{file_id, filename}]`.
- `POST /api/ingest/url` — `{"url": ...}`; returns `{file_id, filename}`.
- `GET  /api/ingest/events` — SSE, default event type (the client uses
  `EventSource.onmessage`, which only fires for unnamed events), payload
  `{file_id, filename, stage, percent, status, message}`.
- `GET  /api/ingest/status` — current and recent jobs, so a client that
  reconnects or loads late catches up instead of showing nothing.

`percent` is progress across the *whole* five-stage rail the Addendum defines
(parse → chunk → embed → KL-extract → review-ready), not progress within one
stage, because that is what the UI's rail renders. The RAG pipeline's own four
stages are mapped onto the first three UI stages by `_DOMAIN_BANDS`; its `store`
step finishes the `embed` band, since indexing is what makes an embedding usable.

Two guarantees worth stating explicitly:

- **A disconnected client never cancels an ingestion.** Jobs run as tasks owned
  by the hub, not by the request. The SSE endpoint only *reads* a queue.
- **Jobs are independent.** Every event carries its `file_id`; concurrent
  uploads interleave on one stream and the UI keys them apart.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Protocol

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from socratic_tutor.api.klmap_routes import COURSE_ID_MAX_LENGTH, COURSE_ID_PATTERN
from socratic_tutor.config import get_settings
from socratic_tutor.domain.klmap import KLMap, ValidationReport
from socratic_tutor.domain.rag import parsers, store
from socratic_tutor.domain.rag.ingest import IngestResult, ingest_file, ingest_url
from socratic_tutor.domain.rag.kl_extract import extract_klmap

logger = logging.getLogger(__name__)

#: The SPA's fixed course (web/src/config.ts `SESSION.course_id`). Single-user
#: POC: the UI now sends its course explicitly on both ingestion requests; this
#: default keeps older clients and hand-written curl calls working.
DEFAULT_COURSE_ID = "MATH-SEED-01"

#: The UI's stage rail, in order (Addendum §"Web UI scope" item 1).
UI_STAGES: tuple[str, ...] = ("parse", "chunk", "embed", "kl-extract", "review-ready")

#: RAG pipeline stage → (UI stage, percent band). `store` shares the `embed`
#: band's second half so the overall percentage stays monotonic.
_DOMAIN_BANDS: dict[str, tuple[str, float, float]] = {
    "parse": ("parse", 0.0, 20.0),
    "chunk": ("chunk", 20.0, 40.0),
    "embed": ("embed", 40.0, 50.0),
    "store": ("embed", 50.0, 60.0),
}
#: Each RAG stage's own percentage span, from `domain.rag.ingest._STAGE_DONE`,
#: used to turn its absolute percent back into a fraction of that stage.
_DOMAIN_SPANS: dict[str, tuple[float, float]] = {
    "parse": (0.0, 25.0),
    "chunk": (25.0, 45.0),
    "embed": (45.0, 80.0),
    "store": (80.0, 100.0),
}
_KL_BAND = (60.0, 80.0)

#: Completed jobs kept for `GET /status` after they finish.
MAX_RECENT_JOBS = 50
#: Per-subscriber buffer. A slow client loses its oldest events, never the job.
SUBSCRIBER_QUEUE_SIZE = 512
#: How long shutdown waits for in-flight jobs before cancelling them.
DRAIN_TIMEOUT_S = 10.0

#: Upload limits. Nothing here is authenticated, so an unbounded body is a way
#: to exhaust memory or fill the disk; course material is far below these.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_UPLOAD_FILES = 50
UPLOAD_CHUNK_BYTES = 1024 * 1024

router = APIRouter(prefix="/ingest", tags=["ingest"])


# ---------------------------------------------------------------------- models


class UrlIngestRequest(BaseModel):
    """`POST /api/ingest/url` body. The SPA sends `url` and its `course_id`."""

    url: str
    course_id: str = Field(
        default=DEFAULT_COURSE_ID,
        min_length=1,
        max_length=COURSE_ID_MAX_LENGTH,
        pattern=COURSE_ID_PATTERN,
    )


class UploadAccepted(BaseModel):
    """What the SPA's `uploadFiles`/`submitUrl` expect back."""

    file_id: str
    filename: str


@dataclass
class IngestJob:
    """One file or URL moving through the rail."""

    file_id: str
    filename: str
    course_id: str
    source: str  # "upload" | "url"
    stage: str = "parse"
    percent: float = 0.0
    status: str = "queued"  # queued | running | done | error
    message: str = "Queued"
    chunks: int = 0
    errors: list[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    def event(self) -> dict[str, Any]:
        """The SSE payload — exactly the fields `IngestEvent` declares."""
        return {
            "file_id": self.file_id,
            "filename": self.filename,
            "stage": self.stage,
            "percent": self.percent,
            "status": self.status,
            "message": self.message,
        }

    def snapshot(self) -> dict[str, Any]:
        """`GET /status` view: the event plus what a late client also wants."""
        return {
            **self.event(),
            "course_id": self.course_id,
            "source": self.source,
            "chunks": self.chunks,
            "errors": list(self.errors),
            "updated_at": self.updated_at,
        }


# ------------------------------------------------------------------- the hub


class IngestHub:
    """Job registry, event fan-out, and owner of the background tasks.

    Tasks are held here rather than tied to a request so that closing the SSE
    stream (or the browser tab) cannot abort an ingestion in flight.
    """

    def __init__(self, max_recent: int = MAX_RECENT_JOBS) -> None:
        self.jobs: OrderedDict[str, IngestJob] = OrderedDict()
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._tasks: set[asyncio.Task[None]] = set()
        self._max_recent = max_recent

    # --- jobs

    def add(self, job: IngestJob) -> IngestJob:
        self.jobs[job.file_id] = job
        self._prune()
        self.publish(job)
        return job

    def update(self, job: IngestJob, **changes: Any) -> None:
        for key, value in changes.items():
            setattr(job, key, value)
        job.updated_at = time.time()
        self.publish(job)

    def _prune(self) -> None:
        """Forget the oldest finished jobs once the registry is full."""
        while len(self.jobs) > self._max_recent:
            for file_id, job in self.jobs.items():
                if job.status in {"done", "error"}:
                    del self.jobs[file_id]
                    break
            else:
                return

    # --- fan-out

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=SUBSCRIBER_QUEUE_SIZE)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    @property
    def subscriber_count(self) -> int:
        """Live SSE listeners. Zero of them does not pause or cancel any job."""
        return len(self._subscribers)

    def publish(self, job: IngestJob) -> None:
        """Fan one job's current state out to every live subscriber."""
        event = job.event()
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop this subscriber's oldest event rather than block the
                # pipeline on a client that stopped reading.
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(event)

    def replay(self) -> list[dict[str, Any]]:
        """Latest event per known job, for a client that just connected."""
        return [job.event() for job in self.jobs.values()]

    # --- tasks

    def spawn(self, coro: Any) -> asyncio.Task[None]:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def drain(self, timeout: float = DRAIN_TIMEOUT_S) -> None:
        """Let in-flight jobs finish at shutdown, then cancel whatever is left.

        Bounded on purpose. This runs in the lifespan's shutdown path, so an
        unbounded wait here does not "wait for the job" — it wedges the whole
        process, and under a test client it wedges the suite with no traceback.
        A single pass with a deadline, then cancellation: a job that outlives
        shutdown loses, never the shutdown.
        """
        tasks = [task for task in self._tasks if not task.done()]
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=timeout)
        for task in pending:
            logger.warning("cancelling ingestion task still running at shutdown: %r", task)
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


# --------------------------------------------------------------- the pipeline


class IngestPipeline(Protocol):
    """What the routes need. Tests substitute a fake with no model or network."""

    async def ingest_file(self, path: Path, course_id: str, on_progress: Any) -> IngestResult: ...

    async def ingest_url(self, url: str, course_id: str, on_progress: Any) -> IngestResult: ...

    async def extract(self, course_id: str, on_progress: Any) -> tuple[KLMap, ValidationReport]: ...


class RagPipeline:
    """The real pipeline: `domain.rag` ingestion, then KL Map extraction.

    Extraction reads the whole course, not just the file that triggered it, so
    the draft graph can carry relations that cross documents. A per-course lock
    serialises it: two uploads finishing together would otherwise race on
    `content/drafts/klmap-<course>.yaml`.
    """

    def __init__(self, extract_klmap_enabled: bool = True) -> None:
        self.extract_klmap_enabled = extract_klmap_enabled
        self._locks: dict[str, asyncio.Lock] = {}

    async def ingest_file(self, path: Path, course_id: str, on_progress: Any) -> IngestResult:
        return await ingest_file(path, course_id, on_progress=on_progress)

    async def ingest_url(self, url: str, course_id: str, on_progress: Any) -> IngestResult:
        return await ingest_url(url, course_id, on_progress=on_progress)

    async def extract(self, course_id: str, on_progress: Any) -> tuple[KLMap, ValidationReport]:
        lock = self._locks.setdefault(course_id, asyncio.Lock())
        async with lock:
            chunks = await asyncio.to_thread(store.all_chunks, course_id)
            return await extract_klmap(chunks, course_id, on_progress=on_progress)


# ------------------------------------------------------------- stage mapping


def _ui_progress(stage: str, percent: float) -> tuple[str, float]:
    """RAG `(stage, percent)` → UI `(stage, percent)` on the five-stage rail."""
    ui_stage, band_lo, band_hi = _DOMAIN_BANDS.get(stage, _DOMAIN_BANDS["parse"])
    span_lo, span_hi = _DOMAIN_SPANS.get(stage, (0.0, 100.0))
    local = (percent - span_lo) / (span_hi - span_lo) if span_hi > span_lo else 0.0
    local = min(max(local, 0.0), 1.0)
    return ui_stage, round(band_lo + local * (band_hi - band_lo))


async def run_job(
    hub: IngestHub,
    job: IngestJob,
    pipeline: IngestPipeline,
    target: str | Path,
) -> None:
    """Drive one job across the rail. Never raises; failures land on the job."""
    hub.update(job, status="running", stage="parse", message="Starting")

    def on_rag_progress(stage: str, percent: float, message: str) -> None:
        # Called on the event loop (domain.rag emits between its awaits), so
        # publishing straight to the subscriber queues is safe here.
        ui_stage, ui_percent = _ui_progress(stage, percent)
        hub.update(job, stage=ui_stage, percent=ui_percent, status="running", message=message)

    try:
        if job.source == "url":
            result = await pipeline.ingest_url(str(target), job.course_id, on_rag_progress)
        else:
            result = await pipeline.ingest_file(Path(target), job.course_id, on_rag_progress)
    except Exception as exc:  # pragma: no cover — ingest_* already absorb failures
        logger.exception("ingestion crashed for %s", job.filename)
        hub.update(job, status="error", message=f"Ingestion failed: {exc}")
        return

    job.chunks = result.chunks
    job.errors = list(result.errors)
    if not result.ok:
        hub.update(
            job,
            status="error",
            message=result.errors[0] if result.errors else "Ingestion failed",
        )
        return

    def on_extract_progress(percent: float, message: str) -> None:
        span = _KL_BAND[1] - _KL_BAND[0]
        hub.update(
            job,
            stage="kl-extract",
            percent=round(_KL_BAND[0] + (percent / 100.0) * span),
            status="running",
            message=message,
        )

    hub.update(
        job,
        stage="kl-extract",
        percent=_KL_BAND[0],
        status="running",
        message=f"Indexed {result.chunks} chunk(s); extracting knowledge map",
    )
    try:
        klmap, report = await pipeline.extract(job.course_id, on_extract_progress)
    except Exception as exc:
        logger.exception("KL extraction failed for course %s", job.course_id)
        job.errors.append(str(exc))
        hub.update(
            job,
            status="error",
            message=(
                f"Content indexed ({result.chunks} chunks), but knowledge-map extraction "
                f"failed: {exc}"
            ),
        )
        return

    issues = len(report.errors)
    summary = f"{len(klmap.nodes)} concept(s), {len(klmap.edges)} relationship(s) — " + (
        f"{issues} issue(s) to resolve in review" if issues else "ready for review"
    )
    job.errors.extend(issue.format() for issue in report.errors)
    hub.update(job, stage="review-ready", percent=100, status="done", message=summary)


# --------------------------------------------------------------------- routes


def _hub(request: Request) -> IngestHub:
    return request.app.state.ingest_hub


def _pipeline(request: Request) -> IngestPipeline:
    return request.app.state.ingest_pipeline


def _safe_upload_name(filename: str | None) -> str:
    """Strip any directory component a client sent; never trust the path."""
    name = Path(filename or "").name.strip()
    return name or "upload"


def require_same_origin(request: Request) -> None:
    """Reject cross-site state-changing requests (CSRF).

    `multipart/form-data` is a CORS-*simple* content type, so a page on any site
    the owner happens to visit can POST to this endpoint with no preflight and
    no consent. That plants content in the retrieval corpus a student is then
    taught from, and triggers a whole-course KL extraction per upload.

    `X-Requested-With` cannot be set cross-origin without a preflight, and a
    preflight is what CORS then refuses — so requiring it is a complete fix for
    the simple-request path, and the SPA sends it (`web/src/api/http.ts`).
    """
    if request.headers.get("x-requested-with", "").strip():
        return
    raise HTTPException(
        status_code=403,
        detail=(
            "Missing X-Requested-With header. State-changing endpoints require it so a "
            "form on another site cannot post here without your consent."
        ),
    )


def _check_upload_name(filename: str) -> None:
    """Reject an unsupported extension *before* anything is written to disk."""
    suffix = Path(filename).suffix.lower()
    if suffix not in parsers.SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type {suffix or filename!r}. Accepted: "
                f"{', '.join(sorted(parsers.SUPPORTED_SUFFIXES))}."
            ),
        )


async def _store_upload(upload: UploadFile, target: Path) -> int:
    """Stream an upload to disk under a size cap; returns bytes written.

    Streamed rather than `await upload.read()`, which materialises the whole body
    in memory: a single large upload could otherwise exhaust the process, and
    nothing about this endpoint is authenticated.
    """
    written = 0
    try:
        with target.open("wb") as handle:
            while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    handle.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"{upload.filename or 'file'} exceeds the "
                            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit."
                        ),
                    )
                handle.write(chunk)
    finally:
        await upload.close()
    return written


@router.post("/upload", status_code=202, response_model=list[UploadAccepted])
async def upload_files(
    request: Request,
    files: Annotated[list[UploadFile], File()],
    course_id: Annotated[
        str, Form(min_length=1, max_length=COURSE_ID_MAX_LENGTH, pattern=COURSE_ID_PATTERN)
    ] = DEFAULT_COURSE_ID,
) -> list[UploadAccepted]:
    """Accept files, start one independent job per file, return immediately."""
    require_same_origin(request)
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files in one request (limit {MAX_UPLOAD_FILES}).",
        )

    settings = get_settings()
    uploads = Path(settings.uploads_dir or settings.data_dir / "uploads")
    uploads.mkdir(parents=True, exist_ok=True)

    hub, pipeline = _hub(request), _pipeline(request)
    accepted: list[UploadAccepted] = []

    for upload in files:
        file_id = f"file-{uuid.uuid4().hex[:12]}"
        filename = _safe_upload_name(upload.filename)
        _check_upload_name(filename)
        target = uploads / f"{file_id}-{filename}"
        try:
            await _store_upload(upload, target)
        except OSError as exc:
            target.unlink(missing_ok=True)
            raise HTTPException(
                status_code=500, detail=f"Could not store {filename}: {exc}"
            ) from exc

        job = hub.add(
            IngestJob(file_id=file_id, filename=filename, course_id=course_id, source="upload")
        )
        hub.spawn(run_job(hub, job, pipeline, target))
        accepted.append(UploadAccepted(file_id=file_id, filename=filename))

    return accepted


@router.post("/url", status_code=202, response_model=UploadAccepted)
async def ingest_from_url(request: Request, body: UrlIngestRequest) -> UploadAccepted:
    """Accept a URL and start a job for it. The URL is the job's display name."""
    require_same_origin(request)
    url = body.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="A URL is required.")

    hub, pipeline = _hub(request), _pipeline(request)
    file_id = f"url-{uuid.uuid4().hex[:12]}"
    job = hub.add(IngestJob(file_id=file_id, filename=url, course_id=body.course_id, source="url"))
    hub.spawn(run_job(hub, job, pipeline, url))
    return UploadAccepted(file_id=file_id, filename=url)


async def event_stream(hub: IngestHub) -> AsyncIterator[dict[str, Any]]:
    """Replay current job state, then follow live updates until the client goes.

    Separate from the route so it can be driven directly: Starlette's
    `TestClient` buffers a whole response body before returning, so an endless
    SSE endpoint can only be exercised against a real server or here.
    """
    queue = hub.subscribe()
    try:
        # An SSE comment: ignored by EventSource, but it flushes the headers and
        # gives a client (or a test) a definite "subscribed" point.
        yield {"comment": "subscribed"}
        for event in hub.replay():
            yield {"data": json.dumps(event, ensure_ascii=False)}
        while True:
            yield {"data": json.dumps(await queue.get(), ensure_ascii=False)}
    finally:
        hub.unsubscribe(queue)


@router.get("/events")
async def ingest_events(request: Request) -> EventSourceResponse:
    """Progress for every job, as unnamed SSE events.

    Unnamed on purpose: the SPA reads them with `EventSource.onmessage`, which
    never fires for a named event type. On connect the latest event per known
    job is replayed, so a client that starts late or reconnects sees current
    state without a separate fetch.
    """
    return EventSourceResponse(event_stream(_hub(request)), headers={"Cache-Control": "no-cache"})


class IngestStatus(BaseModel):
    """`GET /status` response."""

    jobs: list[dict[str, Any]] = Field(default_factory=list)


@router.get("/status", response_model=IngestStatus)
async def ingest_status(request: Request) -> IngestStatus:
    """Current and recent jobs, newest last — the catch-up endpoint."""
    return IngestStatus(jobs=[job.snapshot() for job in _hub(request).jobs.values()])


__all__ = [
    "DEFAULT_COURSE_ID",
    "MAX_RECENT_JOBS",
    "UI_STAGES",
    "IngestHub",
    "IngestJob",
    "IngestPipeline",
    "RagPipeline",
    "UploadAccepted",
    "UrlIngestRequest",
    "event_stream",
    "router",
    "run_job",
]
