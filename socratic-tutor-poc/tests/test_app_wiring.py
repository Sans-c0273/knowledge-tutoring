"""Wiring tests for the app that actually ships (`create_app`).

Every other API test builds its own `FastAPI()` and includes one router. That
proves the router works and nothing about whether it is mounted — which is how
`POST /api/chat/turn` shipped answering with `index.html` and HTTP 200, so the
SPA's SSE parser received HTML and chat was dead while every test passed.

These tests only ever go through `create_app`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from socratic_tutor.api.app import create_app

#: Every path the SPA calls (`web/src/api/http.ts`). A route missing from the
#: shipped app must fail here, not in a browser.
SPA_PATHS = {
    "/api/ingest/upload",
    "/api/ingest/url",
    "/api/ingest/events",
    "/api/ingest/status",
    "/api/chat/turn",
    "/api/tutor/session/{session_id}",
    "/api/klmap",
    "/api/klmap/drafts",
    "/api/klmap/drafts/{course_id}",
    "/api/klmap/drafts/{course_id}/approve",
    "/api/klmap/drafts/{course_id}/revalidate",
}


@pytest.fixture
def spa_dist(tmp_path: Path) -> Path:
    """A stand-in `web/dist`, so the static mount is present as in production."""
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>SPA</title>", encoding="utf-8")
    return dist


@pytest.fixture
def client(spa_dist: Path) -> Iterator[TestClient]:
    with TestClient(create_app(web_dist=spa_dist)) as test_client:
        yield test_client


def test_every_route_the_spa_calls_exists_on_the_shipped_app(client: TestClient):
    registered = set(client.app.openapi()["paths"])

    missing = SPA_PATHS - registered
    assert not missing, f"routes the SPA calls but the app does not mount: {sorted(missing)}"


def test_chat_turn_is_not_swallowed_by_the_static_mount(client: TestClient):
    """The exact failure: an unmounted route answered 200 text/html."""
    response = client.post("/api/chat/turn", json={})

    assert response.status_code != 200, "an unmounted route silently served the SPA"
    assert "text/html" not in response.headers.get("content-type", "")


def test_an_unknown_api_path_404s_instead_of_returning_the_spa(client: TestClient):
    """Static files mount last, so a missing API route fails honestly."""
    response = client.get("/api/definitely-not-a-route")

    assert response.status_code == 404
    assert "text/html" not in response.headers.get("content-type", "")


def test_the_spa_is_still_served_at_the_root(client: TestClient):
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_health_answers_json(client: TestClient):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_ingest_status_is_reachable_through_the_shipped_app(client: TestClient):
    response = client.get("/api/ingest/status")

    assert response.status_code == 200
    assert response.json() == {"jobs": []}


def test_missing_draft_is_a_404_not_a_crash(client: TestClient, tmp_path: Path):
    response = client.get("/api/klmap/drafts/NO-SUCH-COURSE")

    assert response.status_code == 404


def test_openapi_documents_the_whole_api(client: TestClient):
    """A route that 500s while building the schema is invisible until runtime."""
    schema: dict[str, Any] = client.app.openapi()

    assert schema["info"]["title"]
    assert json.dumps(schema), "the schema must be serialisable"
