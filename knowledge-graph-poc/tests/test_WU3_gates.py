"""WU3 — `kg.gates`: deterministic gates (DESIGN §10; PRD R7, R8, R10, R13; D29).

Replaces the six `xfail(strict=False)` placeholders formerly in tests/test_WU0_schemas.py
(they guessed `kg.gates.relevance(edge, GENERAL)` — that shape is kept: a gate returns a
`Rejection` or `None`).

Interface (chosen):
  Edge(type, source_id, target_id, relevance=None)          internal graph edge
  Rejection(gate: str, reason: str, edge)                    .gate names the gate
  out_of_schema(edge, schema) / relevance(edge, schema) / dag(edge, existing, schema) -> Rejection | None
  mirror(nodes, schema) -> list[Rejection]
  gate_edges(proposals, *, existing, schema, review_dir=None) -> GateResult(accepted: list[Edge], rejected: list[Rejection])
      runs out_of_schema, relevance, self_edge, dangling(ids ⊂ known when given), duplicate_edge, dag incrementally;
      appends one JSON line per rejection to <review_dir>/rejected-edges.jsonl when review_dir is given
  validate_graph(graph_dir, schema) -> list[Rejection]      `kg gates`: notes on disk, no API calls, never deletes
  REJECTED_EDGES_FILE == "_review/rejected-edges.jsonl"
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kg.schemas import EDUCATION, GENERAL, EducationEdgeProposal, GeneralEdgeProposal
from wu3_fixtures import simple_node, snapshot_tree, write_graph


def E(type_: str, s: str, t: str, relevance=None):
    from kg.gates import Edge

    return Edge(type=type_, source_id=s, target_id=t, relevance=relevance)


# ----------------------------------------------------------- out_of_schema


@pytest.mark.parametrize("t", ["prerequisite_of", "example_of", "refines", "supersedes"])
def test_R8_out_of_schema_rejects_education_only_types_in_a_general_run(t: str):
    from kg.gates import Rejection, out_of_schema

    r = out_of_schema(E(t, "a", "b"), GENERAL)
    assert isinstance(r, Rejection)
    assert r.gate == "out_of_schema"
    assert t in r.reason and "general" in r.reason


def test_R8_out_of_schema_rejects_related_to_in_an_education_run():
    from kg.gates import out_of_schema

    r = out_of_schema(E("related_to", "a", "b", 70), EDUCATION)
    assert r is not None and r.gate == "out_of_schema"


def test_R8_out_of_schema_rejects_unknown_types_in_both_schemas():
    from kg.gates import out_of_schema

    assert out_of_schema(E("depends_on", "a", "b"), EDUCATION) is not None
    assert out_of_schema(E("depends_on", "a", "b"), GENERAL) is not None


@pytest.mark.parametrize("schema,types", [(EDUCATION, EDUCATION.edges), (GENERAL, GENERAL.edges)])
def test_R8_out_of_schema_accepts_every_type_of_the_run_schema(schema, types):
    from kg.gates import out_of_schema

    for t in types:
        assert out_of_schema(E(t, "a", "b", 50 if t == "related_to" else None), schema) is None


# --------------------------------------------------------------- relevance


def test_R10_gate_rejects_related_to_without_relevance():
    from kg.gates import relevance

    r = relevance(GeneralEdgeProposal(type="related_to", source_id="a", target_id="b", relevance=None), GENERAL)
    assert r is not None and r.gate == "relevance"
    assert "related_to" in r.reason


@pytest.mark.parametrize("t", ["part_of", "same_as"])
def test_R10_gate_rejects_relevance_on_non_related_to(t: str):
    from kg.gates import relevance

    r = relevance(GeneralEdgeProposal(type=t, source_id="a", target_id="b", relevance=50), GENERAL)
    assert r is not None and r.gate == "relevance"
    assert t in r.reason


@pytest.mark.parametrize("score", [-1, 101, 1000, -100])
def test_R10_gate_rejects_relevance_out_of_0_100(score: int):
    from kg.gates import relevance

    r = relevance(GeneralEdgeProposal(type="related_to", source_id="a", target_id="b", relevance=score), GENERAL)
    assert r is not None and r.gate == "relevance"
    assert str(score) in r.reason


@pytest.mark.parametrize("score", ["72", 72.5, True, None])
def test_R10_gate_rejects_non_integer_relevance_on_related_to(score):
    # The wire model would refuse most of these; the gate is the last line for hand-edited notes (kg gates).
    from kg.gates import relevance

    assert relevance(E("related_to", "a", "b", score), GENERAL) is not None


@pytest.mark.parametrize("score", [0, 1, 50, 99, 100])
def test_R10_gate_accepts_related_to_with_integer_in_range(score: int):
    from kg.gates import relevance

    assert relevance(GeneralEdgeProposal(type="related_to", source_id="a", target_id="b", relevance=score), GENERAL) is None


def test_R10_gate_accepts_well_formed_general_edges():
    from kg.gates import relevance

    assert relevance(GeneralEdgeProposal(type="related_to", source_id="a", target_id="b", relevance=72), GENERAL) is None
    assert relevance(GeneralEdgeProposal(type="part_of", source_id="a", target_id="b", relevance=None), GENERAL) is None
    assert relevance(GeneralEdgeProposal(type="same_as", source_id="a", target_id="b", relevance=None), GENERAL) is None


def test_R10_gate_accepts_every_education_edge_which_has_no_relevance_attribute():
    from kg.gates import relevance

    for t in EDUCATION.edges:
        assert relevance(EducationEdgeProposal(type=t, source_id="a", target_id="b"), EDUCATION) is None


def test_R10_gate_rejects_a_relevance_smuggled_onto_an_education_edge():
    from kg.gates import relevance

    assert relevance(E("prerequisite_of", "a", "b", 50), EDUCATION) is not None


# --------------------------------------------------------------------- dag


def test_R13_dag_rejects_the_edge_that_closes_a_prerequisite_cycle_and_names_the_path():
    from kg.gates import dag

    existing = [E("prerequisite_of", "a", "b"), E("prerequisite_of", "b", "c")]
    r = dag(E("prerequisite_of", "c", "a"), existing, EDUCATION)
    assert r is not None and r.gate == "dag"
    for node in ("a", "b", "c"):
        assert node in r.reason, f"reason should name the cycle path: {r.reason!r}"


def test_R13_dag_rejects_a_direct_two_cycle():
    from kg.gates import dag

    assert dag(E("prerequisite_of", "b", "a"), [E("prerequisite_of", "a", "b")], EDUCATION) is not None


def test_R13_dag_accepts_a_forward_edge_and_a_diamond():
    from kg.gates import dag

    existing = [E("prerequisite_of", "a", "b"), E("prerequisite_of", "a", "c"), E("prerequisite_of", "b", "d")]
    assert dag(E("prerequisite_of", "c", "d"), existing, EDUCATION) is None  # diamond, no cycle
    assert dag(E("prerequisite_of", "d", "e"), existing, EDUCATION) is None


def test_R13_dag_only_considers_edges_of_dag_types():
    from kg.gates import dag

    # part_of is not a DAG type: a part_of "cycle" is not the dag gate's business, and part_of
    # edges must not be mixed into the prerequisite graph.
    assert dag(E("part_of", "b", "a"), [E("part_of", "a", "b")], EDUCATION) is None
    assert dag(E("prerequisite_of", "b", "a"), [E("part_of", "a", "b")], EDUCATION) is None


def test_R13_dag_general_schema_has_no_dag_types_so_never_rejects():
    from kg.gates import dag

    existing = [E("part_of", "a", "b"), E("part_of", "b", "c")]
    assert dag(E("part_of", "c", "a"), existing, GENERAL) is None


def test_R13_dag_asks_the_schema_not_a_hard_coded_name():
    from dataclasses import replace

    from kg.gates import dag
    from kg.schemas import EdgeSchema, EdgeType

    edges = dict(EDUCATION.edges)
    edges["part_of"] = EdgeType("part_of", symmetric=False, dag=True)
    custom = EdgeSchema(name="education", edges=edges, origin_values=EDUCATION.origin_values, proposal_model=EDUCATION.proposal_model, edges_output_model=EDUCATION.edges_output_model)
    assert dag(E("part_of", "b", "a"), [E("part_of", "a", "b")], custom) is not None
    assert replace(EDUCATION.edges["prerequisite_of"], dag=False).dag is False  # sanity on the dataclass


# ------------------------------------------------------------- gate_edges


def test_R13_gate_edges_is_incremental_and_logs_the_rejection_to_jsonl(tmp_path: Path):
    from kg.gates import REJECTED_EDGES_FILE, gate_edges

    review = tmp_path / "_review"
    proposals = [E("prerequisite_of", "a", "b"), E("prerequisite_of", "b", "c"), E("prerequisite_of", "c", "a")]
    res = gate_edges(proposals, existing=[], schema=EDUCATION, review_dir=review)
    assert [(e.source_id, e.target_id) for e in res.accepted] == [("a", "b"), ("b", "c")]
    assert len(res.rejected) == 1 and res.rejected[0].gate == "dag"

    assert REJECTED_EDGES_FILE == "_review/rejected-edges.jsonl"
    log = review / "rejected-edges.jsonl"
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["gate"] == "dag"
    assert row["type"] == "prerequisite_of" and row["source_id"] == "c" and row["target_id"] == "a"
    assert row["reason"]

    # append, never truncate
    gate_edges([E("prerequisite_of", "c", "a")], existing=res.accepted, schema=EDUCATION, review_dir=review)
    assert len(log.read_text(encoding="utf-8").splitlines()) == 2


def test_R13_gate_edges_respects_edges_already_on_disk():
    from kg.gates import gate_edges

    res = gate_edges([E("prerequisite_of", "c", "a")], existing=[E("prerequisite_of", "a", "b"), E("prerequisite_of", "b", "c")], schema=EDUCATION)
    assert res.accepted == [] and res.rejected[0].gate == "dag"


def test_R8_R10_gate_edges_applies_schema_and_relevance_gates_with_reasons():
    from kg.gates import gate_edges

    res = gate_edges(
        [E("related_to", "a", "b", 72), E("related_to", "a", "c", None), E("part_of", "a", "d", 10), E("prerequisite_of", "a", "e"), E("related_to", "a", "f", 140)],
        existing=[],
        schema=GENERAL,
    )
    assert [(e.type, e.target_id) for e in res.accepted] == [("related_to", "b")]
    assert sorted(r.gate for r in res.rejected) == ["out_of_schema", "relevance", "relevance", "relevance"]
    assert all(r.reason for r in res.rejected)


def test_S10_gate_edges_rejects_self_edges_and_duplicates():
    from kg.gates import gate_edges

    res = gate_edges([E("part_of", "a", "a"), E("part_of", "a", "b"), E("part_of", "a", "b")], existing=[], schema=GENERAL)
    assert [(e.source_id, e.target_id) for e in res.accepted] == [("a", "b")]
    assert sorted(r.gate for r in res.rejected) == ["duplicate_edge", "self_edge"]


def test_S10_gate_edges_treats_symmetric_duplicates_as_the_same_pair():
    from kg.gates import gate_edges

    res = gate_edges([E("related_to", "a", "b", 70), E("related_to", "b", "a", 70)], existing=[], schema=GENERAL)
    assert len(res.accepted) == 1
    assert res.rejected[0].gate == "duplicate_edge"


def test_S10_gate_edges_with_no_proposals_is_empty_and_writes_no_log(tmp_path: Path):
    from kg.gates import gate_edges

    res = gate_edges([], existing=[], schema=GENERAL, review_dir=tmp_path / "_review")
    assert res.accepted == [] and res.rejected == []
    assert not (tmp_path / "_review" / "rejected-edges.jsonl").exists()


# ------------------------------------------------------------------ mirror


def test_S10_mirror_flags_a_symmetric_edge_present_on_one_side_only():
    from kg.gates import mirror
    from kg.notes import NodeEdge

    a = simple_node("kc-0001", "A", edges=[NodeEdge("related_to", "kc-0002", 82)])
    b = simple_node("kc-0002", "B")
    findings = mirror([a, b], GENERAL)
    assert len(findings) == 1
    assert findings[0].gate == "mirror"
    assert "kc-0001" in findings[0].reason and "kc-0002" in findings[0].reason


def test_S10_mirror_flags_a_relevance_mismatch_between_the_two_sides():
    from kg.gates import mirror
    from kg.notes import NodeEdge

    a = simple_node("kc-0001", "A", edges=[NodeEdge("related_to", "kc-0002", 82)])
    b = simple_node("kc-0002", "B", edges=[NodeEdge("related_to", "kc-0001", 40)])
    assert len(mirror([a, b], GENERAL)) == 1


def test_S10_mirror_accepts_consistent_symmetric_edges_and_ignores_directed_ones():
    from kg.gates import mirror
    from kg.notes import NodeEdge

    a = simple_node("kc-0001", "A", edges=[NodeEdge("related_to", "kc-0002", 82), NodeEdge("part_of", "kc-0003")])
    b = simple_node("kc-0002", "B", edges=[NodeEdge("related_to", "kc-0001", 82), NodeEdge("same_as", "kc-0003")])
    c = simple_node("kc-0003", "C", edges=[NodeEdge("same_as", "kc-0002")])
    assert mirror([a, b, c], GENERAL) == []


def test_S10_mirror_education_same_as_is_symmetric_too():
    from kg.gates import mirror
    from kg.notes import NodeEdge

    a = simple_node("kc-0001", "A", schema="education", edges=[NodeEdge("same_as", "kc-0002")])
    b = simple_node("kc-0002", "B", schema="education")
    assert len(mirror([a, b], EDUCATION)) == 1


# ---------------------------------------------------------- validate_graph


@pytest.fixture
def graph_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data" / "graph"
    d.mkdir(parents=True)
    return d


def test_R7_validate_graph_passes_a_well_formed_graph_without_any_api_call(graph_dir: Path):
    from kg.gates import validate_graph
    from kg.notes import NodeEdge

    nodes = [
        simple_node("kc-0001", "Sets", schema="education", edges=[NodeEdge("prerequisite_of", "kc-0002")]),
        simple_node("kc-0002", "Functions", schema="education", edges=[NodeEdge("prerequisite_of", "kc-0003"), NodeEdge("same_as", "kc-0003")]),
        simple_node("kc-0003", "Limits", schema="education", edges=[NodeEdge("same_as", "kc-0002")]),
    ]
    write_graph(graph_dir, nodes, corpus="kc", schema="education")
    assert validate_graph(graph_dir, EDUCATION) == []


def test_R13_validate_graph_reports_a_cycle_on_disk_and_deletes_nothing(graph_dir: Path):
    from kg.gates import validate_graph
    from kg.notes import NodeEdge

    nodes = [
        simple_node("kc-0001", "Sets", schema="education", edges=[NodeEdge("prerequisite_of", "kc-0002")]),
        simple_node("kc-0002", "Functions", schema="education", edges=[NodeEdge("prerequisite_of", "kc-0003")]),
        simple_node("kc-0003", "Limits", schema="education", edges=[NodeEdge("prerequisite_of", "kc-0001")]),
    ]
    write_graph(graph_dir, nodes, corpus="kc", schema="education")
    before = snapshot_tree(graph_dir)
    findings = validate_graph(graph_dir, EDUCATION)
    assert any(f.gate == "dag" for f in findings)
    assert snapshot_tree(graph_dir) == before, "gates never modify or delete files"


def test_R8_validate_graph_reports_out_of_schema_edges_and_broken_mirrors(graph_dir: Path):
    from kg.gates import validate_graph
    from kg.notes import NodeEdge

    nodes = [
        simple_node("kc-0001", "A", edges=[NodeEdge("related_to", "kc-0002", 82), NodeEdge("prerequisite_of", "kc-0002")]),
        simple_node("kc-0002", "B"),
    ]
    write_graph(graph_dir, nodes, corpus="kc", schema="general")
    gates_hit = {f.gate for f in validate_graph(graph_dir, GENERAL)}
    assert {"out_of_schema", "mirror"} <= gates_hit


def test_R7_validate_graph_reports_dangling_targets_and_registry_drift(graph_dir: Path):
    from kg.gates import validate_graph
    from kg.notes import NodeEdge

    nodes = [simple_node("kc-0001", "A", edges=[NodeEdge("part_of", "kc-0077")])]
    write_graph(graph_dir, nodes, corpus="kc", schema="general")
    # a note on disk that the registry does not know about
    stray = simple_node("kc-0002", "Stray")
    from kg.notes import render

    (graph_dir / "nodes" / "kc-0002-stray.md").write_text(render(stray, {}), encoding="utf-8")
    gates_hit = {f.gate for f in validate_graph(graph_dir, GENERAL)}
    assert "dangling" in gates_hit
    assert "registry" in gates_hit


def test_R7_validate_graph_on_an_empty_graph_dir_is_empty_not_an_error(graph_dir: Path):
    from kg.gates import validate_graph

    assert validate_graph(graph_dir, GENERAL) == []
