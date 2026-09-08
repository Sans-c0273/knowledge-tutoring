"""Regression tests: the chat pipeline must teach from the course it was asked about.

Root cause (diagnosis 2026-09-02, bug B): `chat_routes._default_dependencies` is a
process-wide `lru_cache(maxsize=1)` singleton built from `content/seed/` only, and
`get_dependencies(request)` has no course dimension. A KL map approved through
`POST /api/klmap/drafts/{course}/approve` lands at `klmap_routes.live_path(course)`
— a location nothing in the chat path ever reads — so every `POST /api/chat/turn`
resolves topics against the seed syllabus and highlights seed nodes, whatever
`course_id` the SPA sent. Sibling S2: `GET /api/klmap?course_id=MATH-SEED-01`
404s for the very course the chat is teaching from, because it never looks at
the seed pack.

These tests drive the shipped `create_app` with the seed content copied into a
throwaway `POC_CONTENT_DIR`, so a live map can be planted under
`live_path(course_id)` without touching the repository. Only the LLM roles and
the vector store are faked — the syllabus and KL map come from the default
resolution path on purpose, because that path is the bug.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from socratic_tutor.api import chat_routes
from socratic_tutor.api.app import create_app
from socratic_tutor.api.chat_routes import install_dependencies
from socratic_tutor.api.klmap_routes import live_path
from socratic_tutor.config import get_settings
from socratic_tutor.domain.klmap import KLMap, load_klmap, save_klmap
from socratic_tutor.models.enums import Intent, LearnerState, SpecialHandling
from socratic_tutor.models.intent import IntentResult, SpecialHandlingResult
from socratic_tutor.orchestrator import TurnDependencies
from socratic_tutor.pedagogy.topics import derive_aliases

REPO_CONTENT = Path(__file__).resolve().parents[1] / "content"
SEED_COURSE = "MATH-SEED-01"
LIVE_COURSE = "ALG101"

#: Nodes that exist in no seed file. If a turn names one of these, it can only
#: have come from the planted live map.
LIVE_NODES = [
    {"id": "X001", "name": "Quadratic Formula", "name_th": "สูตรกำลังสอง"},
    {"id": "X002", "name": "Completing the Square", "name_th": "การทำให้เป็นกำลังสองสมบูรณ์"},
    {"id": "X003", "name": "Discriminant", "name_th": "ดิสคริมิแนนต์"},
]
LIVE_EDGES = [
    {"from": "X002", "to": "X001", "relation": "prerequisite_of"},
    {"from": "X001", "to": "X003", "relation": "next_topic"},
]


def live_map_for(course_id: str) -> KLMap:
    return KLMap.model_validate(
        {
            "course_id": course_id,
            "course_name": f"Quadratics ({course_id})",
            "nodes": LIVE_NODES,
            "edges": LIVE_EDGES,
        }
    )


def body(course_id: str, message: str) -> dict[str, str]:
    return {
        "session_id": f"sess-fix-{course_id}",
        "student_id": "student-fix-01",
        "course_id": course_id,
        "message": message,
    }


# ---------------------------------------------------------------- fakes


def fake_intent() -> Any:
    async def detect(message: str, context: Any = (), **kwargs: Any) -> IntentResult:
        return IntentResult(
            intent=Intent.EXPLAIN,
            learner_state=LearnerState.NORMAL,
            special_handling=SpecialHandlingResult(detected=False, type=SpecialHandling.NONE),
            confidence=0.92,
        )

    return detect


def fake_generation(reply: str) -> Any:
    async def generate(plan: Any, message: str, **kwargs: Any) -> AsyncIterator[str]:
        for word in reply.split(" "):
            yield word + " "

    return generate


def parse_frames(payload: str) -> list[tuple[str, Any]]:
    """Same framing rule as `web/src/api/sse.ts` and `tests/test_chat_api.py`."""
    frames: list[tuple[str, Any]] = []
    for block in payload.split("\n\n"):
        if not block.strip():
            continue
        event = "message"
        data: list[str] = []
        for line in block.splitlines():
            field, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if field == "event":
                event = value
            elif field == "data":
                data.append(value)
        frames.append((event, json.loads("\n".join(data)) if data else None))
    return frames


def final_trace(response: Any) -> dict[str, Any]:
    assert response.status_code == 200, response.text
    frames = parse_frames(response.text)
    assert frames and frames[-1][0] == "done", [event for event, _ in frames]
    assert not [event for event, _ in frames if event == "error"], frames
    return frames[-1][1]


# ------------------------------------------------------------- fixtures


@pytest.fixture
def content_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """The real seed pack under a throwaway content root, with no live or draft maps.

    `_default_dependencies` is lru-cached for the process, so it is cleared on
    both sides: another test may already have built it against the repository's
    content dir, and this one must not leave a copy pointing at `tmp_path`.
    """
    root = tmp_path / "content"
    shutil.copytree(REPO_CONTENT / "seed", root / "seed")
    monkeypatch.setenv("POC_CONTENT_DIR", str(root))
    monkeypatch.setenv("POC_DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    chat_routes._default_dependencies.cache_clear()
    yield root
    get_settings.cache_clear()
    chat_routes._default_dependencies.cache_clear()


@pytest.fixture
def client(content_dir: Path, tmp_path: Path) -> Iterator[TestClient]:
    """The shipped app, teaching from the default (seed-derived) dependencies.

    The default `TurnDependencies` are taken as built — syllabus, KL map and
    answer keys included — and only the roles that would leave the machine are
    swapped: the two LLM calls and the Chroma/BGE-M3 retrieval. If a turn ends
    up on the seed map here, it is because the resolution path put it there.
    """
    default = chat_routes.get_dependencies()
    deps: TurnDependencies = dataclasses.replace(
        default,
        detect_intent=fake_intent(),
        generate=fake_generation("Let us start from what you already know."),
        retrieve=lambda course_id, topic, message: [],
    )
    app = create_app(web_dist=tmp_path / "no-dist")
    install_dependencies(app, deps)
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------- tests


def test_fix_course_aware_deps_uses_approved_live_map(client: TestClient, content_dir: Path):
    """An approved live map for course X is what a turn in course X teaches from."""
    planted = save_klmap(live_map_for(LIVE_COURSE), live_path(LIVE_COURSE))
    assert planted.is_file() and planted.is_relative_to(content_dir)

    trace = final_trace(
        client.post("/api/chat/turn", json=body(LIVE_COURSE, "Help me with the quadratic formula"))
    )

    topic = trace.get("topic")
    assert topic is not None, (
        "no topic resolved: the message names a node of the ALG101 live map, "
        "so the turn was resolved against a syllabus that does not contain it"
    )
    assert topic["topic"] == "Quadratic Formula"
    assert topic["node_id"] == "X001"
    knowledge = trace.get("knowledge_context")
    assert knowledge is not None, "no KL lookup: the turn is not using ALG101's map"
    assert knowledge["current_node_id"] == "X001"
    live_ids = {node["id"] for node in LIVE_NODES}
    assert set(knowledge["nodes_hit"]) <= live_ids, knowledge["nodes_hit"]


def test_fix_course_aware_deps_seed_fallback_unchanged(client: TestClient, content_dir: Path):
    """No live map for the course → the seed pack keeps serving, exactly as today."""
    assert not live_path(SEED_COURSE).exists()

    trace = final_trace(
        client.post(
            "/api/chat/turn", json=body(SEED_COURSE, "How do I solve two-step linear equations?")
        )
    )

    assert trace["topic"]["topic"] == "Two-Step Linear Equations"
    assert trace["topic"]["node_id"] == "C009"
    assert trace["knowledge_context"]["current_node_id"] == "C009"


def test_fix_course_aware_deps_live_map_replaces_seed_for_same_course(
    client: TestClient, content_dir: Path
):
    """Approving a map for the seed's own course id replaces the seed for that course.

    This is the path the SPA actually exercises: every upload is filed under
    `MATH-SEED-01` (S4), so its approved map lands on the seed course's id.
    """
    save_klmap(live_map_for(SEED_COURSE), live_path(SEED_COURSE))

    trace = final_trace(
        client.post("/api/chat/turn", json=body(SEED_COURSE, "Help me with the quadratic formula"))
    )

    topic = trace.get("topic")
    assert topic is not None, "message names a live-map node; resolved against the seed instead"
    assert topic["node_id"] == "X001"
    assert trace["knowledge_context"]["current_node_id"] == "X001"
    seed_ids = {
        node.id for node in load_klmap(content_dir / "seed" / "kl-map-linear-equations.yaml").nodes
    }
    assert not seed_ids & set(trace["knowledge_context"]["nodes_hit"]), (
        "seed nodes highlighted for a course whose live map has none of them"
    )


def test_fix_klmap_get_falls_back_to_seed(client: TestClient, content_dir: Path):
    """S2: with no live or draft file, `GET /api/klmap` shows the map chat teaches from.

    Today the panel says "No approved map for this course yet" while the
    inspector highlights seed nodes the panel cannot display — two endpoints,
    one course id, two different maps.
    """
    assert not live_path(SEED_COURSE).exists()
    assert not (content_dir / "drafts").exists()
    seed = load_klmap(content_dir / "seed" / "kl-map-linear-equations.yaml")

    response = client.get("/api/klmap", params={"course_id": SEED_COURSE})

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["course_id"] == SEED_COURSE
    assert {node["id"] for node in data["nodes"]} == {node.id for node in seed.nodes}
    assert len(data["edges"]) == len(seed.edges)
    # The flag must be present so the panel can label the seed; its polarity
    # (published seed vs. unreviewed fallback) is the fix's call, not this test's.
    assert isinstance(data.get("approved"), bool)


def test_fix_klmap_get_still_404s_for_an_unknown_course(client: TestClient, content_dir: Path):
    """The seed fallback is for the seed's course only; other ids stay honest 404s."""
    response = client.get("/api/klmap", params={"course_id": "NO-SUCH-COURSE"})

    assert response.status_code == 404


# ----------------------------------------------- derived-syllabus aliases


#: The approved live map observed on the owner's machine on 2026-09-02
#: (`content/klmap/klmap-MATH-SEED-01.yaml`), filed under its own course id here.
#: Labels are long compound phrases; students name the part they are stuck on.
PHOTO_COURSE = "BIO201"
PHOTO_MAP = {
    "course_id": PHOTO_COURSE,
    "course_name": "Photosynthesis",
    "nodes": [
        {"id": "C001", "name": "Photosynthesis overview"},
        {"id": "C002", "name": "Chloroplast structure"},
        {"id": "C003", "name": "Pigments and light absorption"},
        {"id": "C004", "name": "Absorption spectrum vs action spectrum"},
        {"id": "C005", "name": "Light-dependent reactions"},
        {"id": "C006", "name": "Photolysis of water"},
        {"id": "C007", "name": "Chemiosmosis and ATP synthase"},
        {"id": "C008", "name": "Calvin cycle (light-independent reactions)"},
        {"id": "C009", "name": "RuBisCO and carbon fixation"},
        {"id": "C010", "name": "Limiting factors of photosynthesis"},
        {"id": "C011", "name": "Temperature effect on enzyme activity"},
        {"id": "C012", "name": "Measuring the rate of photosynthesis"},
        {"id": "C013", "name": "Ecological and agricultural importance of photosynthesis"},
    ],
    "edges": [
        {"from": "C002", "to": "C001", "relation": "part_of"},
        {"from": "C005", "to": "C001", "relation": "part_of"},
        {"from": "C008", "to": "C001", "relation": "part_of"},
        {"from": "C002", "to": "C005", "relation": "prerequisite_of"},
        {"from": "C002", "to": "C008", "relation": "prerequisite_of"},
        {"from": "C003", "to": "C005", "relation": "prerequisite_of"},
        {"from": "C003", "to": "C004", "relation": "related_to"},
        {"from": "C006", "to": "C005", "relation": "part_of"},
        {"from": "C007", "to": "C005", "relation": "part_of"},
        {"from": "C005", "to": "C008", "relation": "prerequisite_of"},
        {"from": "C005", "to": "C008", "relation": "next_topic"},
        {"from": "C009", "to": "C008", "relation": "part_of"},
        {"from": "C009", "to": "C011", "relation": "uses"},
        {"from": "C011", "to": "C010", "relation": "part_of"},
        {"from": "C008", "to": "C010", "relation": "next_topic"},
        {"from": "C010", "to": "C012", "relation": "next_topic"},
        {"from": "C012", "to": "C010", "relation": "uses"},
        {"from": "C010", "to": "C013", "relation": "related_to"},
        {"from": "C001", "to": "C013", "relation": "prerequisite_of"},
        {"from": "C007", "to": "C006", "relation": "uses"},
    ],
}


@pytest.mark.parametrize(
    ("message", "acceptable"),
    [
        pytest.param(
            "I do not understand what RuBisCO does in the Calvin cycle. Can you explain?",
            {"C009", "C008"},
            id="rubisco-in-calvin-cycle",
        ),
        pytest.param(
            "what are the limiting factors?",
            {"C010"},
            id="limiting-factors",
        ),
    ],
)
def test_fix_derived_syllabus_aliases_match_partial_mentions(
    client: TestClient, content_dir: Path, message: str, acceptable: set[str]
):
    """A student naming part of a compound node label still lands on that node.

    The syllabus derived from a live map carries only each node's full name, so
    `resolve_topic`'s mention tier needs the whole label ("Calvin cycle
    (light-independent reactions)") verbatim in the message. Real students type
    the part they are stuck on. The seed syllabus solves this with hand-written
    `aliases`; a derived syllabus has to generate them from the label.
    """
    save_klmap(KLMap.model_validate(PHOTO_MAP), live_path(PHOTO_COURSE))

    trace = final_trace(client.post("/api/chat/turn", json=body(PHOTO_COURSE, message)))

    topic = trace.get("topic")
    assert topic is not None and topic.get("topic"), (
        f"topic resolved to '' for {message!r}: the derived syllabus has no aliases, so a "
        "partial mention of a node label matches nothing and the KL lookup is skipped"
    )
    assert topic["node_id"] in acceptable, topic
    assert trace["knowledge_context"]["current_node_id"] in acceptable


def test_fix_derived_syllabus_carries_aliases_from_compound_labels():
    """Unit seam: the derivation helper splits compound labels into aliases.

    Names the helper as it exists today (`chat_routes._syllabus_from`); the
    implementer may rename or relocate it — update the import here, not the
    expectations. Aliases are compared case-insensitively because
    `resolve_topic` normalises both sides.
    """
    syllabus = chat_routes._syllabus_from(KLMap.model_validate(PHOTO_MAP))
    by_id = {topic.id: topic for topic in syllabus.topics}

    def aliases(topic_id: str) -> set[str]:
        return {alias.casefold() for alias in by_id[topic_id].aliases}

    assert {"calvin cycle", "light-independent reactions"} <= aliases("C008"), by_id["C008"]
    assert {"rubisco", "carbon fixation"} <= aliases("C009"), by_id["C009"]
    assert "limiting factors" in aliases("C010"), by_id["C010"]
    # Full names survive alongside the aliases so exact/mention matching is unchanged.
    assert by_id["C008"].name == "Calvin cycle (light-independent reactions)"


# ------------------------------------------ cycle 2: review findings m2, L1


def test_fix_broken_live_map_falls_back_to_seed_with_a_warning(
    client: TestClient, content_dir: Path, caplog: pytest.LogCaptureFixture
):
    """m2(a): a live file the loader rejects degrades to the seed — logged, no `error` frame."""
    broken = live_path(LIVE_COURSE)
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text("nodes: [\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="socratic_tutor.api.chat_routes"):
        trace = final_trace(
            client.post(
                "/api/chat/turn",
                json=body(LIVE_COURSE, "How do I solve two-step linear equations?"),
            )
        )
        # Second turn on the same broken version: the file is retried, the
        # warning is not repeated.
        final_trace(
            client.post(
                "/api/chat/turn",
                json=body(LIVE_COURSE, "How do I solve two-step linear equations?"),
            )
        )

    assert trace["topic"]["node_id"] == "C009"
    assert trace["knowledge_context"]["current_node_id"] == "C009"
    warnings = [
        record for record in caplog.records if "fall back to the seed map" in record.getMessage()
    ]
    assert len(warnings) == 1, [record.getMessage() for record in caplog.records]
    assert str(broken) in warnings[0].getMessage()


def test_fix_reapproval_is_picked_up_without_restart(client: TestClient, content_dir: Path):
    """m2(b): rewriting the live file (a second approval) is seen by the next turn."""
    path = save_klmap(live_map_for(LIVE_COURSE), live_path(LIVE_COURSE))
    first = final_trace(
        client.post("/api/chat/turn", json=body(LIVE_COURSE, "Help me with the quadratic formula"))
    )
    assert first["topic"]["node_id"] == "X001"

    replacement = KLMap.model_validate(
        {
            "course_id": LIVE_COURSE,
            "course_name": "Algebra II, revised",
            "nodes": [{"id": "Y001", "name": "Binomial Theorem"}],
            "edges": [],
        }
    )
    save_klmap(replacement, path)
    # Two saves inside one clock tick would share an mtime; make the change visible.
    bumped = path.stat().st_mtime_ns + 1_000_000_000
    os.utime(path, ns=(bumped, bumped))

    second = final_trace(
        client.post("/api/chat/turn", json=body(LIVE_COURSE, "Explain the binomial theorem"))
    )
    assert second["topic"]["node_id"] == "Y001"
    assert second["knowledge_context"]["current_node_id"] == "Y001"


def test_fix_session_readback_labels_from_the_sessions_course(
    client: TestClient, content_dir: Path
):
    """m2(c): `GET /tutor/session/{id}` maps node names through the session's course map."""
    save_klmap(live_map_for(LIVE_COURSE), live_path(LIVE_COURSE))
    request = body(LIVE_COURSE, "Help me with the quadratic formula")
    final_trace(client.post("/api/chat/turn", json=request))

    response = client.get(f"/api/tutor/session/{request['session_id']}")

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["course_id"] == LIVE_COURSE
    last = data["traces"][-1]
    assert last["topic"]["node_id"] == "X001"
    assert last["knowledge_context"]["current_node_id"] == "X001"


@pytest.mark.parametrize("bad", ["../etc", "ALG 101", "a/b", "x" * 65, ""])
def test_fix_course_id_is_validated_at_the_edge(client: TestClient, content_dir: Path, bad: str):
    """L1: an id that could not name a file is refused before any path is built."""
    turn = client.post("/api/chat/turn", json=body(bad, "hello"))
    assert turn.status_code == 422, turn.text

    klmap = client.get("/api/klmap", params={"course_id": bad})
    assert klmap.status_code in (400, 404, 422), klmap.text
    assert klmap.status_code != 200


def test_fix_klmap_get_reports_its_source(client: TestClient, content_dir: Path):
    """L4: the seed answers as `source: seed`, unapproved; a published map as `live`."""
    seed = client.get("/api/klmap", params={"course_id": SEED_COURSE}).json()
    assert seed["source"] == "seed"
    assert seed["approved"] is False

    save_klmap(live_map_for(LIVE_COURSE), live_path(LIVE_COURSE))
    live = client.get("/api/klmap", params={"course_id": LIVE_COURSE}).json()
    assert live["source"] == "live"
    assert live["approved"] is True


@pytest.mark.parametrize(
    ("label", "expected", "rejected"),
    [
        (
            "Calvin cycle (light-independent reactions)",
            {"Calvin cycle", "light-independent reactions"},
            set(),
        ),
        ("RuBisCO and carbon fixation", {"RuBisCO", "carbon fixation"}, set()),
        ("Limiting factors of photosynthesis", {"Limiting factors"}, {"photosynthesis"}),
        (
            "Absorption spectrum vs action spectrum",
            {"Absorption spectrum", "action spectrum"},
            set(),
        ),
        ("Chemiosmosis and ATP synthase", {"Chemiosmosis", "ATP synthase"}, set()),
        ("Photosynthesis overview", set(), {"Photosynthesis overview"}),
        # Cycle-2 re-review M3: a single-word " of " head is a grammatical head,
        # not a concept, and at MENTION 0.9 it hijacks turns ("in order to…").
        ("Order of Operations", set(), {"Order"}),
        ("Law of Sines", set(), {"Law"}),
        ("Area of a triangle", set(), {"Area"}),
        ("Newton's laws of motion", {"Newton's laws"}, set()),
        (
            "Ecological and agricultural importance of photosynthesis",
            {"Ecological and agricultural importance"},
            {"Ecological"},
        ),
    ],
)
def test_fix_derive_aliases_rules(label: str, expected: set[str], rejected: set[str]):
    """Unit seam for the pure helper: fixed rules, stable output, full label never repeated."""
    aliases = derive_aliases(label)

    assert expected <= set(aliases), aliases
    assert not rejected & set(aliases), aliases
    assert label not in aliases
    assert aliases == derive_aliases(label)  # deterministic
    assert all(len(alias) >= 3 for alias in aliases)


def test_fix_derived_aliases_do_not_hijack_the_context_topic():
    """M3: a generic English word in a follow-up must not out-score the topic under discussion."""
    from socratic_tutor.pedagogy.topics import ConversationContext, resolve_topic

    syllabus = chat_routes._syllabus_from(
        KLMap.model_validate(
            {
                "course_id": "ALG-M3",
                "course_name": "Algebra",
                "nodes": [
                    {"id": "Q001", "name": "Quadratic Formula"},
                    {"id": "Q002", "name": "Order of Operations"},
                    {"id": "Q003", "name": "Area of a triangle"},
                ],
                "edges": [],
            }
        )
    )
    context = ConversationContext.of("Quadratic Formula")

    for message in ("in order to use it, do I need b squared first?", "what is the area under it?"):
        match = resolve_topic(message, syllabus, context)
        assert match is not None
        assert match.topic.id == "Q001", (message, match.topic.name, match.confidence)
