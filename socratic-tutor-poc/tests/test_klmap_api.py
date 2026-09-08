"""KL Map draft review endpoints (R18), driven through the shipped app.

The property that matters here is the approval gate: the SPA disables its button
when a draft has errors, but the server must refuse regardless, because a map
that fails a Tech Spec §2.3 rule reaching live traffic makes scaffolding
non-terminating.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from socratic_tutor.api.app import create_app
from socratic_tutor.config import get_settings

COURSE = "ALG101"

CLEAN_MAP = {
    "course_id": COURSE,
    "course_name": "Algebra Basics",
    "nodes": [
        {"id": "C001", "name": "Inverse Operations", "name_th": "การดำเนินการผกผัน"},
        {"id": "C002", "name": "Two-Step Linear Equations"},
    ],
    "edges": [{"from": "C001", "to": "C002", "relation": "prerequisite_of"}],
}

#: Same graph with the prerequisite edge mirrored — a cycle, which the loader
#: rejects with KL009.
CYCLIC_MAP = {
    **CLEAN_MAP,
    "edges": [
        {"from": "C001", "to": "C002", "relation": "prerequisite_of"},
        {"from": "C002", "to": "C001", "relation": "prerequisite_of"},
    ],
}


@pytest.fixture
def content_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point settings at a throwaway content tree, drafts and live map included."""
    monkeypatch.setenv("POC_CONTENT_DIR", str(tmp_path / "content"))
    monkeypatch.setenv("POC_DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    (tmp_path / "content" / "drafts").mkdir(parents=True)
    yield tmp_path / "content"
    get_settings.cache_clear()


def write_draft(content_dir: Path, data: dict, sources: list[str] | None = None) -> Path:
    path = content_dir / "drafts" / f"klmap-{data['course_id']}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    path.with_suffix(".report.json").write_text(
        json.dumps(
            {
                "course_id": data["course_id"],
                "course_name": data["course_name"],
                "extracted_at": "2026-09-01T10:00:00+00:00",
                "sources": sources or ["Algebra Basics Ch.1 p.12"],
                "report": {
                    "errors": [],
                    "warnings": [
                        {
                            "code": "KX102",
                            "message": "Dropped proposed edge 'C001' -related_to-> 'C404'.",
                            "location": None,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def client(content_dir: Path, tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(web_dist=tmp_path / "no-dist")) as test_client:
        yield test_client


def test_listing_returns_the_draft_with_its_provenance(client: TestClient, content_dir: Path):
    write_draft(content_dir, CLEAN_MAP)

    drafts = client.get("/api/klmap/drafts").json()

    assert len(drafts) == 1
    assert drafts[0]["course_id"] == COURSE
    assert drafts[0]["status"] == "pending"
    assert drafts[0]["sources"] == ["Algebra Basics Ch.1 p.12"]
    assert drafts[0]["extracted_at"] == "2026-09-01T10:00:00+00:00"
    assert len(drafts[0]["nodes"]) == 2


def test_a_draft_carries_both_loader_and_extraction_findings(client: TestClient, content_dir: Path):
    """`KX` warnings describe what the model proposed and we dropped — re-reading
    the YAML cannot rediscover them, so they come from the sidecar."""
    write_draft(content_dir, CLEAN_MAP)

    report = client.get(f"/api/klmap/drafts/{COURSE}").json()["report"]

    assert any(issue["code"] == "KX102" for issue in report["warnings"])


def test_an_invalid_draft_still_carries_its_graph_for_review(client: TestClient, content_dir: Path):
    """The review surface must never omit the artefact under review.

    `load_klmap_with_report` returns `(None, report)` for an invalid map, which
    once meant a reviewer saw the validation issues beside an empty graph and
    empty tables — and approved or rejected against nothing. The extractor keeps
    invalid drafts precisely so a human can see what the model proposed.
    """
    write_draft(content_dir, CYCLIC_MAP)

    draft = client.get(f"/api/klmap/drafts/{COURSE}").json()

    assert draft["report"]["errors"], "this draft is invalid"
    assert [node["id"] for node in draft["nodes"]] == ["C001", "C002"]
    assert len(draft["edges"]) == 2, "both edges of the cycle must be visible"
    assert draft["course_name"] == "Algebra Basics", "not the course-id fallback"


def test_a_dangling_edge_survives_into_the_review_payload(client: TestClient, content_dir: Path):
    """The frontend's dangling-edge banner needs the edge to still be there."""
    write_draft(
        content_dir,
        {**CLEAN_MAP, "edges": [{"from": "C001", "to": "C404", "relation": "prerequisite_of"}]},
    )

    draft = client.get(f"/api/klmap/drafts/{COURSE}").json()

    assert any(edge["to"] == "C404" for edge in draft["edges"])
    assert any(issue["code"] == "KL006" for issue in draft["report"]["errors"])


def test_an_out_of_vocabulary_relation_is_shown_not_dropped(client: TestClient, content_dir: Path):
    """Coercing to the closed vocabulary would hide the very thing to review."""
    write_draft(
        content_dir,
        {**CLEAN_MAP, "edges": [{"from": "C001", "to": "C002", "relation": "invented_by"}]},
    )

    draft = client.get(f"/api/klmap/drafts/{COURSE}").json()

    assert draft["edges"][0]["relation"] == "invented_by"
    assert any(issue["code"] == "KL005" for issue in draft["report"]["errors"])


def test_an_invalid_draft_is_listed_under_its_real_course_id(client: TestClient, content_dir: Path):
    write_draft(content_dir, CYCLIC_MAP)

    drafts = client.get("/api/klmap/drafts").json()

    assert [draft["course_id"] for draft in drafts] == [COURSE]
    assert drafts[0]["nodes"], "the listing must carry the graph too"


def test_unparseable_yaml_reports_the_problem_without_crashing(
    client: TestClient, content_dir: Path
):
    (content_dir / "drafts" / f"klmap-{COURSE}.yaml").write_text(
        "nodes: [unclosed\n", encoding="utf-8"
    )

    draft = client.get(f"/api/klmap/drafts/{COURSE}").json()

    assert draft["report"]["errors"], "the reviewer is told why it is empty"
    assert draft["nodes"] == []


def test_approval_is_refused_while_the_draft_has_blocking_errors(
    client: TestClient, content_dir: Path
):
    write_draft(content_dir, CYCLIC_MAP)

    draft = client.get(f"/api/klmap/drafts/{COURSE}").json()
    assert any(issue["code"] == "KL009" for issue in draft["report"]["errors"])

    response = client.post(f"/api/klmap/drafts/{COURSE}/approve", json={"approve": True})

    assert response.status_code == 409
    assert "blocking error" in response.json()["detail"]
    assert not (content_dir / "klmap" / f"klmap-{COURSE}.yaml").exists()
    assert client.get(f"/api/klmap/drafts/{COURSE}").json()["status"] == "pending"


def test_approving_a_clean_draft_publishes_it(client: TestClient, content_dir: Path):
    write_draft(content_dir, CLEAN_MAP)

    response = client.post(
        f"/api/klmap/drafts/{COURSE}/approve", json={"approve": True, "note": "checked by Tanat"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "approved"
    assert body["review_note"] == "checked by Tanat"
    assert body["reviewed_at"]

    published = content_dir / "klmap" / f"klmap-{COURSE}.yaml"
    assert published.is_file(), "approval publishes to the live map location"
    assert "Inverse Operations" in published.read_text(encoding="utf-8")


def test_approval_does_not_edit_the_draft_in_place(client: TestClient, content_dir: Path):
    draft_file = write_draft(content_dir, CLEAN_MAP)
    before = draft_file.read_text(encoding="utf-8")

    client.post(f"/api/klmap/drafts/{COURSE}/approve", json={"approve": True})

    assert draft_file.read_text(encoding="utf-8") == before


def test_rejecting_records_the_decision_without_publishing(client: TestClient, content_dir: Path):
    write_draft(content_dir, CLEAN_MAP)

    body = client.post(
        f"/api/klmap/drafts/{COURSE}/approve",
        json={"approve": False, "note": "edges are wrong"},
    ).json()

    assert body["status"] == "rejected"
    assert body["review_note"] == "edges are wrong"
    assert not (content_dir / "klmap" / f"klmap-{COURSE}.yaml").exists()


def test_a_rejected_draft_can_be_fixed_and_revalidated(client: TestClient, content_dir: Path):
    """The POC's editing story is the YAML file itself."""
    draft_file = write_draft(content_dir, CYCLIC_MAP)
    assert client.get(f"/api/klmap/drafts/{COURSE}").json()["report"]["errors"]

    draft_file.write_text(
        yaml.safe_dump(CLEAN_MAP, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    revalidated = client.post(f"/api/klmap/drafts/{COURSE}/revalidate").json()

    assert revalidated["report"]["errors"] == []
    assert (
        client.post(f"/api/klmap/drafts/{COURSE}/approve", json={"approve": True}).status_code
        == 200
    )


def test_live_map_is_flagged_unapproved_until_it_is_approved(client: TestClient, content_dir: Path):
    write_draft(content_dir, CLEAN_MAP)

    before = client.get("/api/klmap", params={"course_id": COURSE}).json()
    assert before["approved"] is False

    client.post(f"/api/klmap/drafts/{COURSE}/approve", json={"approve": True})
    after = client.get("/api/klmap", params={"course_id": COURSE}).json()

    assert after["approved"] is True
    assert [node["id"] for node in after["nodes"]] == ["C001", "C002"]
    assert after["edges"][0]["from"] == "C001"


def test_an_unknown_course_is_a_404(client: TestClient):
    assert client.get("/api/klmap", params={"course_id": "NOPE"}).status_code == 404
    assert client.post("/api/klmap/drafts/NOPE/approve", json={"approve": True}).status_code == 404
