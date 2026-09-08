"""WU3 — `kg.stages.edges` (S5): roster construction (cap from `limits.roster_cap`, warning when
truncated), one call per chunk with the run schema's edges-output model, proposals converted to
internal edges; unknown ids / out-of-schema types dropped with a reason (DESIGN §6, §8.0; D6; R8).

Interface (chosen): build_roster(drafts, cap) -> Roster(entries: list[RosterEntry(id, title, aliases)], truncated, total);
propose(drafts, chunks, cfg, *, call_stage, ledger=None) -> EdgesResult(edges: list[Edge], dropped: list[Rejection], warnings: list[str]).
`Edge` is kg.gates.Edge. Drafts are `kg.stages.consolidate.NodeDraft`; built directly here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kg.config import load_config
from kg.schemas import EducationEdgesOutput, GeneralEdgesOutput
from wu3_fixtures import ScriptedCallStage, make_chunk, make_unit


def D(node_id, title, *, chunk_ids=("a.md#c1",), source_files=("a.md",), aliases=(), definition="Def."):
    from kg.notes import NodeSource
    from kg.stages.consolidate import NodeDraft

    return NodeDraft(
        id=node_id,
        title=title,
        aliases=list(aliases),
        definition=definition,
        sources=[NodeSource(locator=f"{source_files[0]}#heading={title}", file=source_files[0], quote=f"{title} quote")],
        review=[],
        chunk_ids=list(chunk_ids),
        source_files=list(source_files),
        is_new=True,
    )


@pytest.fixture
def cfg(make_project):
    return load_config(make_project())


@pytest.fixture
def cfg_edu(make_project):
    return load_config(make_project({"corpus.schema": "education"}, subdir="edu"))


@pytest.fixture
def chunks():
    a = make_chunk("a.md#c1", [make_unit("U1", "Sample space text.", heading=["Sample Space"]), make_unit("U2", "Event text.", heading=["Event"])])
    b = make_chunk("b.md#c1", [make_unit("U3", "Random variable text.", source="b.md", heading=["Random Variable"])])
    return [a, b]


@pytest.fixture
def drafts():
    return [
        D("kc-0001", "Sample Space", aliases=["Outcome space"]),
        D("kc-0002", "Event"),
        D("kc-0003", "Random Variable", chunk_ids=("b.md#c1",), source_files=("b.md",)),
    ]


def edges_payload(*edges):
    return {"edges": [dict(e) for e in edges]}


# ---------------------------------------------------------------- roster


def test_S5_build_roster_lists_every_draft_with_id_title_aliases_when_under_cap(drafts):
    from kg.stages.edges import build_roster

    roster = build_roster(drafts, cap=400)
    assert [e.id for e in roster.entries] == ["kc-0001", "kc-0002", "kc-0003"]
    assert roster.entries[0].title == "Sample Space" and list(roster.entries[0].aliases) == ["Outcome space"]
    assert roster.truncated is False and roster.total == 3


def test_S5_build_roster_truncates_at_the_cap_and_flags_it(drafts):
    from kg.stages.edges import build_roster

    roster = build_roster(drafts, cap=2)
    assert len(roster.entries) == 2
    assert roster.truncated is True and roster.total == 3


def test_S5_build_roster_of_nothing_is_empty(cfg):
    from kg.stages.edges import build_roster

    roster = build_roster([], cap=cfg.limits.roster_cap)
    assert roster.entries == [] and roster.truncated is False


# --------------------------------------------------------------- propose


def test_S5_one_call_per_chunk_with_the_general_edges_schema(cfg, drafts, chunks):
    from kg.stages.edges import propose

    fake = ScriptedCallStage({"edges": lambda call: edges_payload()}, cfg=cfg)
    res = propose(drafts, chunks, cfg, call_stage=fake)
    assert len(fake.calls) == 2
    for call in fake.calls:
        assert call["stage"] == "edges"
        assert call["schema"] is GeneralEdgesOutput
        assert call["extra"].get("cfg") is cfg
        assert str(call["extra"].get("prompt_version", "")).startswith("edges@")
    assert res.edges == [] and res.dropped == []


def test_R8_education_run_uses_the_education_edges_schema(cfg_edu, drafts, chunks):
    from kg.stages.edges import propose

    fake = ScriptedCallStage({"edges": lambda call: edges_payload()}, cfg=cfg_edu)
    propose(drafts, chunks, cfg_edu, call_stage=fake)
    assert all(c["schema"] is EducationEdgesOutput for c in fake.calls)


def test_S5_prompt_carries_the_roster_ids_and_titles_verbatim_and_the_chunks_full_drafts(cfg, drafts, chunks):
    from kg.stages.edges import propose

    fake = ScriptedCallStage({"edges": lambda call: edges_payload()}, cfg=cfg)
    propose(drafts, chunks, cfg, call_stage=fake)
    for call in fake.calls:  # the roster (all nodes) is in every call
        for d in drafts:
            assert d.id in call["user_text"] and d.title in call["user_text"]
    # the chunk's own drafts appear in full (definition text) only in that chunk's call
    assert "Def." in fake.calls[0]["user_text"]
    a_text, b_text = fake.calls[0]["user_text"], fake.calls[1]["user_text"]
    assert drafts[0].sources[0].quote in a_text
    assert drafts[2].sources[0].quote in b_text


def test_S5_proposals_become_edges_preserving_type_direction_and_relevance(cfg, drafts, chunks):
    from kg.stages.edges import propose

    fake = ScriptedCallStage(
        {
            "edges": [
                edges_payload({"type": "related_to", "source_id": "kc-0001", "target_id": "kc-0002", "relevance": 72}, {"type": "part_of", "source_id": "kc-0002", "target_id": "kc-0001", "relevance": None}),
                edges_payload({"type": "related_to", "source_id": "kc-0003", "target_id": "kc-0001", "relevance": 40}),
            ]
        },
        cfg=cfg,
    )
    res = propose(drafts, chunks, cfg, call_stage=fake)
    assert [(e.type, e.source_id, e.target_id, e.relevance) for e in res.edges] == [
        ("related_to", "kc-0001", "kc-0002", 72),
        ("part_of", "kc-0002", "kc-0001", None),
        ("related_to", "kc-0003", "kc-0001", 40),
    ]
    assert res.dropped == []


def test_S5_unknown_ids_are_dropped_with_a_reason_naming_the_id(cfg, drafts, chunks):
    from kg.stages.edges import propose

    fake = ScriptedCallStage(
        {"edges": [edges_payload({"type": "part_of", "source_id": "kc-0001", "target_id": "kc-9999", "relevance": None}, {"type": "part_of", "source_id": "made-up", "target_id": "kc-0001", "relevance": None}), edges_payload()]},
        cfg=cfg,
    )
    res = propose(drafts, chunks, cfg, call_stage=fake)
    assert res.edges == []
    assert len(res.dropped) == 2
    assert "kc-9999" in res.dropped[0].reason
    assert "made-up" in res.dropped[1].reason


def test_R8_out_of_schema_type_is_dropped_with_a_reason_even_if_the_wire_let_it_through(cfg, drafts, chunks):
    from kg.stages.edges import propose

    rogue = SimpleNamespace(edges=[SimpleNamespace(type="prerequisite_of", source_id="kc-0001", target_id="kc-0002", relevance=None)])
    fake = ScriptedCallStage({"edges": [rogue, edges_payload()]}, cfg=cfg)
    res = propose(drafts, chunks, cfg, call_stage=fake)
    assert res.edges == []
    assert len(res.dropped) == 1
    assert "prerequisite_of" in res.dropped[0].reason


def test_S5_self_edges_are_dropped(cfg, drafts, chunks):
    from kg.stages.edges import propose

    fake = ScriptedCallStage({"edges": [edges_payload({"type": "part_of", "source_id": "kc-0001", "target_id": "kc-0001", "relevance": None}), edges_payload()]}, cfg=cfg)
    res = propose(drafts, chunks, cfg, call_stage=fake)
    assert res.edges == [] and len(res.dropped) == 1


def test_S5_roster_cap_from_config_produces_a_warning(make_project, drafts, chunks):
    from kg.stages.edges import propose

    cfg = load_config(make_project({"limits.roster_cap": 2}, subdir="cap"))
    fake = ScriptedCallStage({"edges": lambda call: edges_payload()}, cfg=cfg)
    res = propose(drafts, chunks, cfg, call_stage=fake)
    assert any("roster" in w.lower() and "2" in w for w in res.warnings), res.warnings


def test_S5_no_warning_when_under_cap(cfg, drafts, chunks):
    from kg.stages.edges import propose

    fake = ScriptedCallStage({"edges": lambda call: edges_payload()}, cfg=cfg)
    assert propose(drafts, chunks, cfg, call_stage=fake).warnings == []


def test_S5_chunks_without_drafts_make_no_call(cfg, drafts, chunks):
    from kg.stages.edges import propose

    fake = ScriptedCallStage({"edges": lambda call: edges_payload()}, cfg=cfg)
    propose(drafts[:2], chunks, cfg, call_stage=fake)  # nothing from b.md#c1
    assert len(fake.calls) == 1


def test_S5_no_drafts_makes_no_calls(cfg, chunks):
    from kg.stages.edges import propose

    fake = ScriptedCallStage({"edges": lambda call: edges_payload()}, cfg=cfg)
    res = propose([], chunks, cfg, call_stage=fake)
    assert fake.calls == [] and res.edges == []


def test_R21_boundary_exception_propagates(cfg, drafts, chunks):
    from kg.stages.edges import propose

    fake = ScriptedCallStage({"edges": [RuntimeError("boom")]}, cfg=cfg)
    with pytest.raises(RuntimeError):
        propose(drafts, chunks, cfg, call_stage=fake)
