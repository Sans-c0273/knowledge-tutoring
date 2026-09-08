"""Ingestion API tests (R17).

A fake pipeline stands in for `domain.rag`, so nothing here embeds, indexes or
reaches the network. What is tested is the HTTP contract the SPA already codes
against (`web/src/api/http.ts`, `web/src/types.ts`): payload shape, the
five-stage rail, independence of concurrent jobs, and the promise that a client
going away does not kill an ingestion.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Self

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from socratic_tutor.api import ingest_routes
from socratic_tutor.api.app import create_app
from socratic_tutor.api.ingest_routes import (
    UI_STAGES,
    IngestHub,
    IngestJob,
    _ui_progress,
    event_stream,
    run_job,
)
from socratic_tutor.domain.klmap import KLMap, KLNode, ValidationReport
from socratic_tutor.domain.rag.ingest import IngestResult

TERMINAL = {"done", "error"}


class FakePipeline:
    """Walks the RAG stages, then hands back a small map — no model, no store."""

    def __init__(
        self,
        ok: bool = True,
        errors: list[str] | None = None,
        report: ValidationReport | None = None,
        extract_error: Exception | None = None,
        gate: asyncio.Event | None = None,
    ) -> None:
        self.ok = ok
        self.errors = errors or []
        self.report = report or ValidationReport()
        self.extract_error = extract_error
        self.gate = gate
        self.ingested: list[tuple[str, str]] = []
        self.extracted: list[str] = []

    async def _walk(self, source: str, course_id: str, on_progress: Any) -> IngestResult:
        self.ingested.append((source, course_id))
        for stage, start, end in (
            ("parse", 0.0, 25.0),
            ("chunk", 25.0, 45.0),
            ("embed", 45.0, 80.0),
            ("store", 80.0, 100.0),
        ):
            on_progress(stage, start, f"{stage} starting")
            if not self.ok and stage == "parse":
                return IngestResult(
                    source=source, course_id=course_id, ok=False, errors=self.errors
                )
            if self.gate is not None:
                await self.gate.wait()
            await asyncio.sleep(0)
            on_progress(stage, end, f"{stage} done")
        return IngestResult(source=source, course_id=course_id, ok=True, chunks=7)

    async def ingest_file(self, path: Path, course_id: str, on_progress: Any) -> IngestResult:
        return await self._walk(Path(path).name, course_id, on_progress)

    async def ingest_url(self, url: str, course_id: str, on_progress: Any) -> IngestResult:
        return await self._walk(url, course_id, on_progress)

    async def extract(self, course_id: str, on_progress: Any) -> tuple[KLMap, ValidationReport]:
        self.extracted.append(course_id)
        if self.extract_error is not None:
            raise self.extract_error
        on_progress(50.0, "Extracting concepts")
        on_progress(100.0, "Done")
        return (
            KLMap(
                course_id=course_id,
                course_name=course_id,
                nodes=[KLNode(id="C001", name="Inverse Operations")],
            ),
            self.report,
        )


@pytest.fixture
def pipeline() -> FakePipeline:
    return FakePipeline()


@pytest.fixture
def client(pipeline: FakePipeline, tmp_path: Path) -> Iterator[TestClient]:
    app = create_app(pipeline=pipeline, web_dist=tmp_path / "no-dist")
    with TestClient(app) as test_client:
        yield test_client


#: The SPA sends this on every state-changing call; the server requires it (CSRF).
CSRF = {"X-Requested-With": "XMLHttpRequest"}


def upload(client: TestClient, *names: str) -> list[dict[str, Any]]:
    files = [
        ("files", (name, b"# Chapter 1\n\nSome content.\n", "text/markdown")) for name in names
    ]
    response = client.post("/api/ingest/upload", files=files, headers=CSRF)
    assert response.status_code == 202, response.text
    return response.json()


def wait_for_terminal(
    fetch: Callable[[], list[dict[str, Any]]], count: int = 1, timeout: float = 10.0
) -> list[dict[str, Any]]:
    """Poll a status fetcher until `count` jobs have reached a terminal status."""
    deadline = time.monotonic() + timeout
    jobs: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        jobs = fetch()
        if len([job for job in jobs if job["status"] in TERMINAL]) >= count:
            return jobs
        time.sleep(0.02)
    raise AssertionError(f"jobs did not finish within {timeout}s: {jobs}")


def status_of(client: TestClient) -> Callable[[], list[dict[str, Any]]]:
    return lambda: client.get("/api/ingest/status").json()["jobs"]


# ------------------------------------------------------------------- contract


def test_upload_returns_the_shape_the_spa_expects(client: TestClient):
    accepted = upload(client, "ch1.md")

    assert isinstance(accepted, list)
    assert set(accepted[0]) == {"file_id", "filename"}
    assert accepted[0]["filename"] == "ch1.md"


def test_url_ingestion_returns_a_single_accepted_object(client: TestClient):
    response = client.post(
        "/api/ingest/url", json={"url": "https://example.com/lesson"}, headers=CSRF
    )

    assert response.status_code == 202
    body = response.json()
    assert set(body) == {"file_id", "filename"}
    assert body["filename"] == "https://example.com/lesson"


def test_upload_with_no_files_is_rejected(client: TestClient):
    response = client.post("/api/ingest/upload", files=[], headers=CSRF)

    assert response.status_code in (400, 422)


def test_an_empty_url_is_rejected(client: TestClient):
    assert client.post("/api/ingest/url", json={"url": "  "}, headers=CSRF).status_code == 400


def test_a_directory_traversing_filename_is_stripped(client: TestClient):
    accepted = upload(client, "../../etc/passwd.md")

    assert accepted[0]["filename"] == "passwd.md", "the path component must not survive"


def test_a_traversing_name_without_a_supported_extension_is_rejected(client: TestClient):
    """Rejected on the extension before the name is used for anything at all."""
    response = client.post(
        "/api/ingest/upload",
        files=[("files", ("../../etc/passwd", b"root:x:0:0\n", "text/plain"))],
        headers=CSRF,
    )

    assert response.status_code == 415


def test_health_is_served_under_the_api_prefix(client: TestClient):
    assert client.get("/api/health").json() == {"status": "ok"}


# ------------------------------------------------------------------- security


def test_upload_without_the_csrf_header_is_refused(client: TestClient, pipeline: FakePipeline):
    """M11: multipart is CORS-simple, so any site could post here without this."""
    response = client.post(
        "/api/ingest/upload",
        files=[("files", ("ch1.md", b"# planted\n", "text/markdown"))],
    )

    assert response.status_code == 403
    assert "X-Requested-With" in response.json()["detail"]
    assert pipeline.ingested == [], "nothing was ingested"


def test_url_ingestion_without_the_csrf_header_is_refused(client: TestClient):
    response = client.post("/api/ingest/url", json={"url": "https://example.com/x"})

    assert response.status_code == 403


def test_an_unsupported_extension_is_rejected_before_anything_is_written(
    client: TestClient, pipeline: FakePipeline
):
    """M12: the extension used to be checked only after the bytes hit disk."""
    response = client.post(
        "/api/ingest/upload",
        files=[("files", ("payload.exe", b"MZ\x90\x00", "application/octet-stream"))],
        headers=CSRF,
    )

    assert response.status_code == 415
    assert ".exe" in response.json()["detail"]
    assert pipeline.ingested == []


def test_an_oversized_upload_is_rejected(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ingest_routes, "MAX_UPLOAD_BYTES", 1024)

    response = client.post(
        "/api/ingest/upload",
        files=[("files", ("big.md", b"x" * 4096, "text/markdown"))],
        headers=CSRF,
    )

    assert response.status_code == 413
    assert "upload limit" in response.json()["detail"]


def test_too_many_files_in_one_request_is_rejected(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(ingest_routes, "MAX_UPLOAD_FILES", 2)

    response = client.post(
        "/api/ingest/upload",
        files=[("files", (f"ch{n}.md", b"# c\n", "text/markdown")) for n in range(3)],
        headers=CSRF,
    )

    assert response.status_code == 413


# ----------------------------------------------------------------- the rail


def test_a_job_walks_the_whole_rail_to_review_ready(client: TestClient, pipeline: FakePipeline):
    upload(client, "ch1.md")
    jobs = wait_for_terminal(status_of(client))

    job = jobs[0]
    assert job["status"] == "done"
    assert job["stage"] == "review-ready"
    assert job["percent"] == 100
    assert "concept" in job["message"]
    assert pipeline.extracted == ["MATH-SEED-01"], "extraction runs for the SPA's course"


def test_stage_mapping_is_monotonic_across_the_five_stage_rail():
    seen = [
        _ui_progress(stage, percent)
        for stage, lo, hi in (
            ("parse", 0.0, 25.0),
            ("chunk", 25.0, 45.0),
            ("embed", 45.0, 80.0),
            ("store", 80.0, 100.0),
        )
        for percent in (lo, hi)
    ]
    stages = [stage for stage, _ in seen]
    percents = [percent for _, percent in seen]

    assert percents == sorted(percents)
    assert set(stages) <= set(UI_STAGES)
    assert stages[0] == "parse"
    assert percents[-1] == 60, "the RAG half ends where kl-extract begins"


async def test_events_cover_every_stage_and_end_done():
    """Driven at the hub, where the sequence is deterministic."""
    hub = IngestHub()
    events: list[dict[str, Any]] = []
    queue = hub.subscribe()
    job = hub.add(IngestJob(file_id="f1", filename="ch1.md", course_id="C", source="upload"))

    await run_job(hub, job, FakePipeline(), Path("ch1.md"))
    while not queue.empty():
        events.append(queue.get_nowait())

    stages = [event["stage"] for event in events]
    assert set(stages) == set(UI_STAGES)
    assert stages.index("parse") < stages.index("chunk") < stages.index("embed")
    assert stages.index("embed") < stages.index("kl-extract") < stages.index("review-ready")
    assert [event["percent"] for event in events] == sorted(event["percent"] for event in events)
    assert events[-1]["status"] == "done"
    assert all(event["file_id"] == "f1" for event in events)


async def test_a_failed_ingestion_ends_error_and_never_extracts():
    hub = IngestHub()
    pipeline = FakePipeline(ok=False, errors=["unsupported format '.png'"])
    job = hub.add(IngestJob(file_id="f1", filename="x.png", course_id="C", source="upload"))

    await run_job(hub, job, pipeline, Path("x.png"))

    assert job.status == "error"
    assert "unsupported format" in job.message
    assert pipeline.extracted == [], "a file that never indexed must not trigger extraction"


async def test_extraction_failure_reports_that_content_was_still_indexed():
    hub = IngestHub()
    pipeline = FakePipeline(extract_error=RuntimeError("provider unreachable"))
    job = hub.add(IngestJob(file_id="f1", filename="ch1.md", course_id="C", source="upload"))

    await run_job(hub, job, pipeline, Path("ch1.md"))

    assert job.status == "error"
    assert "indexed" in job.message and "provider unreachable" in job.message


async def test_validation_errors_surface_in_the_terminal_message():
    report = ValidationReport()
    report.error("KL009", "'prerequisite_of' edges form a cycle: A -> B -> A.")
    hub = IngestHub()
    job = hub.add(IngestJob(file_id="f1", filename="ch1.md", course_id="C", source="upload"))

    await run_job(hub, job, FakePipeline(report=report), Path("ch1.md"))

    assert job.status == "done", "an invalid draft still reaches review — a human fixes it"
    assert "1 issue(s) to resolve in review" in job.message
    assert any("KL009" in error for error in job.errors)


# ------------------------------------------------------ concurrency and SSE


def test_concurrent_jobs_report_independently(client: TestClient):
    accepted = upload(client, "ch1.md", "ch2.md", "video.md")
    jobs = wait_for_terminal(status_of(client), count=3)

    assert len({job["file_id"] for job in jobs}) == 3
    assert {job["filename"] for job in jobs} == {"ch1.md", "ch2.md", "video.md"}
    assert all(job["status"] == "done" for job in jobs)
    assert {a["file_id"] for a in accepted} == {job["file_id"] for job in jobs}


def test_status_lets_a_late_client_catch_up(client: TestClient):
    upload(client, "ch1.md")
    wait_for_terminal(status_of(client))

    jobs = client.get("/api/ingest/status").json()["jobs"]

    assert jobs[0]["stage"] == "review-ready"
    assert set(jobs[0]) >= {
        "file_id",
        "filename",
        "stage",
        "percent",
        "status",
        "message",
        "source",
        "chunks",
        "errors",
        "updated_at",
    }


async def test_event_stream_replays_state_then_follows_live_updates():
    hub = IngestHub()
    job = hub.add(IngestJob(file_id="f1", filename="ch1.md", course_id="C", source="upload"))

    stream = event_stream(hub)
    try:
        assert await anext(stream) == {"comment": "subscribed"}

        replayed = json.loads((await anext(stream))["data"])
        assert replayed["file_id"] == "f1"
        assert replayed["status"] == "queued"

        hub.update(job, stage="chunk", percent=30, status="running", message="Splitting")
        live = json.loads((await anext(stream))["data"])
    finally:
        await stream.aclose()

    assert live["stage"] == "chunk"
    assert live["percent"] == 30
    assert hub.subscriber_count == 0, "closing the stream must unsubscribe"


async def test_event_stream_payload_has_exactly_the_declared_fields():
    hub = IngestHub()
    hub.add(IngestJob(file_id="f1", filename="ch1.md", course_id="C", source="upload"))

    stream = event_stream(hub)
    try:
        await anext(stream)
        payload = json.loads((await anext(stream))["data"])
    finally:
        await stream.aclose()

    assert set(payload) == {"file_id", "filename", "stage", "percent", "status", "message"}


class BackgroundServer:
    """A real uvicorn server on a free port.

    Needed because `TestClient` buffers whole response bodies, so an endless SSE
    endpoint cannot be read through it at all.
    """

    def __init__(self, app: Any) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> Self:
        self.thread.start()
        deadline = time.monotonic() + 15
        while not self.server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        if not self.server.started:
            raise AssertionError("uvicorn did not start")
        return self

    def __exit__(self, *_: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=15)


@pytest.fixture
def server(pipeline: FakePipeline, tmp_path: Path) -> Iterator[BackgroundServer]:
    app = create_app(pipeline=pipeline, web_dist=tmp_path / "no-dist")
    with BackgroundServer(app) as running:
        yield running


def read_events(response: httpx.Response, until_terminal: bool = True) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in response.iter_lines():
        if not line.startswith("data:"):
            continue
        event = json.loads(line[len("data:") :].strip())
        events.append(event)
        if until_terminal and event["status"] in TERMINAL:
            break
    return events


def test_sse_delivers_the_whole_run_over_the_wire(server: BackgroundServer):
    """Subscribe first, then upload — the SPA's actual order of operations."""
    with (
        httpx.Client(base_url=server.base, timeout=15) as client,
        client.stream("GET", "/api/ingest/events") as response,
    ):
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        uploaded: list[Any] = []
        thread = threading.Thread(
            target=lambda: uploaded.append(
                client.post(
                    "/api/ingest/upload",
                    files=[("files", ("ch1.md", b"# Chapter\n", "text/markdown"))],
                    headers=CSRF,
                )
            ),
            daemon=True,
        )
        thread.start()
        events = read_events(response)
        thread.join(timeout=15)

    assert uploaded[0].status_code == 202
    stages = [event["stage"] for event in events]
    assert set(stages) == set(UI_STAGES), f"missing stages in {stages}"
    assert events[-1]["status"] == "done"
    assert events[-1]["percent"] == 100
    assert set(events[-1]) == {"file_id", "filename", "stage", "percent", "status", "message"}


def test_a_disconnected_client_does_not_abort_an_ingestion(
    server: BackgroundServer, pipeline: FakePipeline
):
    """The job belongs to the hub, not to the request or the stream that watched it."""
    with httpx.Client(base_url=server.base, timeout=15) as client:
        with client.stream("GET", "/api/ingest/events") as response:
            next(response.iter_lines())  # subscribed
            client.post(
                "/api/ingest/upload",
                files=[("files", ("ch1.md", b"# Chapter\n", "text/markdown"))],
                headers=CSRF,
            )
            # Drop the stream immediately, mid-job.

        jobs = wait_for_terminal(
            lambda: client.get("/api/ingest/status").json()["jobs"], timeout=15
        )

    assert jobs[0]["status"] == "done"
    assert jobs[0]["stage"] == "review-ready"
    assert pipeline.extracted == ["MATH-SEED-01"]


async def test_shutdown_cancels_a_job_that_never_finishes_instead_of_hanging():
    """Regression: `drain()` used to busy-wait forever on an unfinished task.

    That wedged the lifespan shutdown, so `TestClient.__exit__` never returned
    and the suite hung with no traceback while spinning a core — the hang that
    blocked the whole team.
    """
    hub = IngestHub()
    never = asyncio.Event()  # deliberately never set
    job = hub.add(IngestJob(file_id="f1", filename="stuck.md", course_id="C", source="upload"))
    hub.spawn(run_job(hub, job, FakePipeline(gate=never), Path("stuck.md")))
    await asyncio.sleep(0)

    await asyncio.wait_for(hub.drain(timeout=0.05), timeout=5)

    assert job.status != "done"


async def test_shutdown_lets_an_in_flight_job_finish_when_it_can():
    hub = IngestHub()
    job = hub.add(IngestJob(file_id="f1", filename="ch1.md", course_id="C", source="upload"))
    hub.spawn(run_job(hub, job, FakePipeline(), Path("ch1.md")))

    await asyncio.wait_for(hub.drain(), timeout=5)

    assert job.status == "done"
    assert job.stage == "review-ready"


async def test_a_slow_subscriber_loses_events_not_the_job():
    """A client that stops reading must not stall the pipeline."""
    hub = IngestHub()
    queue = hub.subscribe()
    job = hub.add(IngestJob(file_id="f1", filename="ch1.md", course_id="C", source="upload"))

    for index in range(queue.maxsize + 50):
        hub.update(job, percent=index % 100)

    await run_job(hub, job, FakePipeline(), Path("ch1.md"))

    assert job.status == "done"
    assert queue.qsize() == queue.maxsize
