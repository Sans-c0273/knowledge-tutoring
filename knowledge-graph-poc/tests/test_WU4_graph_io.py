"""WU4 — `kg.graph_io`: load the graph on disk into memory for the renderer (R14 input; DESIGN §8.6, §20, §23).

Interface (chosen, see wu45_fixtures docstring):
  load_graph(graph_dir) -> Graph(nodes: dict[id, GraphNode], edges: list[GraphEdge], meta, problems)
  Graph.to_index() -> dict  (exactly the §8.6 `_index.json` shape) ; write_index(graph, path)
Symmetric edges are de-duplicated to one GraphEdge (source=min id); a missing or corrupt note is
reported in `problems`, never fatal; nothing outside the graph folder is read.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from kg.paths import SandboxViolation
from wu45_fixtures import (
    EDUCATION_EDGES,
    EDUCATION_IDS,
    GENERAL_EDGES,
    GENERAL_IDS,
    THAI_TITLE,
    write_education_graph,
    write_general_graph,
)


@pytest.fixture
def graph_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data" / "graph"
    d.mkdir(parents=True)
    return d


def load(graph_dir: Path):
    from kg.graph_io import load_graph

    return load_graph(graph_dir)


# ------------------------------------------------------------------ nodes


def test_R14_load_graph_returns_every_node_on_disk_with_title_and_plain_definition(graph_dir):
    write_general_graph(graph_dir)
    g = load(graph_dir)
    assert set(g.nodes) == GENERAL_IDS
    assert g.problems == []
    n = g.nodes["kb-0001"]
    assert n.title == "Cloud First Policy"
    assert n.aliases == ["Cloud-first"]
    assert "[[" not in n.definition_plain and "sovereign cloud" in n.definition_plain, "wikilinks stripped on read (§12)"
    assert n.provider == "openrouter" and n.model == "anthropic/claude-sonnet-5"
    assert n.file == "nodes/kb-0001-cloud-first-policy.md"


def test_R3_R4_node_sources_carry_locator_file_and_quote(graph_dir):
    write_general_graph(graph_dir)
    n = load(graph_dir).nodes["kb-0001"]
    assert [s["locator"] for s in n.sources] == [
        "https://example.go.th/policy/cloud-first#heading=2. Scope > 2.1 Data classes",
        "sovereignty-notes.docx#heading=Cloud First > Data residency",
    ]
    assert n.sources[0]["file"] == "https://example.go.th/policy/cloud-first"
    assert n.sources[0]["quote"] == "Agencies shall adopt cloud services as the default option."


def test_R14_thai_title_survives_utf8_round_trip(graph_dir):
    write_general_graph(graph_dir)
    g = load(graph_dir)
    assert g.nodes["kb-0006"].title == THAI_TITLE
    assert g.nodes["kb-0006"].file == "nodes/kb-0006.md", "Thai-only title slugifies to empty → <id>.md (§8.2)"


def test_R9_origin_is_present_for_education_nodes_and_none_for_general(graph_dir, tmp_path):
    write_general_graph(graph_dir)
    assert all(n.origin is None for n in load(graph_dir).nodes.values())
    edu = tmp_path / "edu" / "graph"
    edu.mkdir(parents=True)
    write_education_graph(edu)
    assert all(n.origin == "course_material" for n in load(edu).nodes.values())


# ------------------------------------------------------------------ edges


def test_R14_edges_are_typed_deduplicated_and_carry_relevance_on_related_to(graph_dir):
    write_general_graph(graph_dir)
    g = load(graph_dir)
    got = {(e.source, e.target, e.type, e.relevance) for e in g.edges}
    assert got == GENERAL_EDGES, "each mirrored symmetric edge appears exactly once, min id first"
    assert len(g.edges) == len(GENERAL_EDGES)
    assert all(e.relevance is None for e in g.edges if e.type != "related_to")
    assert all(isinstance(e.relevance, int) and 0 <= e.relevance <= 100 for e in g.edges if e.type == "related_to")


def test_R14_education_graph_edges_cover_all_six_types_and_keep_direction(graph_dir):
    write_education_graph(graph_dir)
    g = load(graph_dir)
    assert set(g.nodes) == EDUCATION_IDS
    got = {(e.source, e.target, e.type, e.relevance) for e in g.edges}
    assert got == EDUCATION_EDGES
    assert ("stat101-0003", "stat101-0002", "prerequisite_of", None) not in got, "directed edges are not reversed"


def test_R14_degree_counts_unique_neighbours(graph_dir):
    write_general_graph(graph_dir)
    g = load(graph_dir)
    assert g.nodes["kb-0001"].degree == 3  # 0002, 0003, 0005
    assert g.nodes["kb-0004"].degree == 3  # 0002, 0003, 0006
    assert g.nodes["kb-0007"].degree == 1


# ------------------------------------------------------------------- meta


def test_R14_meta_reports_schema_corpus_run_and_provider_from_registry_and_notes(graph_dir):
    write_general_graph(graph_dir)
    g = load(graph_dir)
    assert g.meta["schema"] == "general"
    assert g.meta["corpus"] == "kb"
    assert g.meta["run"] == "2026-08-25T10-15-02Z"
    assert g.meta["provider"] == "openrouter"


# ---------------------------------------------------------- index (§8.6)


def test_S8_6_to_index_matches_the_documented_index_json_shape(graph_dir):
    write_general_graph(graph_dir)
    g = load(graph_dir)
    idx = g.to_index()
    assert set(idx) >= {"nodes", "edges", "meta"}
    node = next(n for n in idx["nodes"] if n["id"] == "kb-0001")
    assert set(node) >= {"id", "title", "file", "definition_plain", "provider", "model", "sources", "degree"}
    assert "origin" not in node or node["origin"] is None
    assert node["sources"][0]["locator"].startswith("https://example.go.th/")
    assert "quote" in node["sources"][0]
    edge = next(e for e in idx["edges"] if e["type"] == "part_of")
    assert set(edge) >= {"source", "target", "type"}
    rel = next(e for e in idx["edges"] if e["type"] == "related_to" and {e["source"], e["target"]} == {"kb-0001", "kb-0002"})
    assert rel["relevance"] == 82
    assert idx["meta"]["schema"] == "general" and idx["meta"]["corpus"] == "kb"
    json.dumps(idx, ensure_ascii=False)  # JSON-serialisable


def test_S8_6_write_index_writes_utf8_json_that_reloads_identically(graph_dir):
    from kg.graph_io import write_index

    write_general_graph(graph_dir)
    g = load(graph_dir)
    out = write_index(g, graph_dir / "_index.json")
    assert Path(out) == graph_dir / "_index.json"
    text = (graph_dir / "_index.json").read_text(encoding="utf-8")
    assert THAI_TITLE in text, "not ASCII-escaped"
    assert json.loads(text) == g.to_index()


def test_S8_6_index_is_regenerated_from_notes_not_trusted_if_stale(graph_dir):
    """A stale hand-edited _index.json must not leak into the loaded graph."""
    write_general_graph(graph_dir)
    (graph_dir / "_index.json").write_text(json.dumps({"nodes": [{"id": "ghost-9999"}], "edges": [], "meta": {}}), encoding="utf-8")
    g = load(graph_dir)
    assert "ghost-9999" not in g.nodes
    assert set(g.nodes) == GENERAL_IDS


# ------------------------------------------------------- unhappy paths


def test_R14_corrupt_note_is_reported_as_a_problem_and_the_rest_still_loads(graph_dir):
    paths = write_general_graph(graph_dir)
    paths["kb-0003"].write_text("---\nid: [unclosed\ntitle: broken\n---\nno sections\n", encoding="utf-8")
    g = load(graph_dir)
    assert "kb-0003" not in g.nodes
    assert set(g.nodes) == GENERAL_IDS - {"kb-0003"}
    assert len(g.problems) == 1
    assert g.problems[0].file.endswith("kb-0003-in-country-region.md")
    assert g.problems[0].reason


def test_R14_registry_row_whose_note_is_missing_is_reported_not_fatal(graph_dir):
    paths = write_general_graph(graph_dir)
    paths["kb-0007"].unlink()
    g = load(graph_dir)
    assert "kb-0007" not in g.nodes
    assert any("kb-0007" in p.file or "kb-0007" in p.reason for p in g.problems)
    # the dangling same_as from kb-0005 must not produce an edge to a node that is not loaded
    assert all(e.target in g.nodes and e.source in g.nodes for e in g.edges)


def test_R14_empty_graph_folder_loads_as_an_empty_graph(graph_dir):
    g = load(graph_dir)
    assert g.nodes == {} and g.edges == []
    assert g.to_index()["nodes"] == []


def test_R14_missing_graph_folder_is_an_error_not_a_silent_empty_graph(tmp_path):
    with pytest.raises((FileNotFoundError, NotADirectoryError, OSError)):
        load(tmp_path / "does-not-exist")


def test_S23_a_symlinked_note_pointing_outside_the_graph_folder_is_not_read(graph_dir, tmp_path):
    write_general_graph(graph_dir)
    outside = tmp_path / "outside.md"
    outside.write_text("---\nid: kb-0099\ntitle: Smuggled\naliases: []\nschema: general\n---\n# Smuggled\n\n## Definition\nx\n\n## Relations\n\n## Source\n", encoding="utf-8")
    link = graph_dir / "nodes" / "kb-0099-smuggled.md"
    os.symlink(outside, link)
    g = load(graph_dir)
    assert "kb-0099" not in g.nodes
    assert any("kb-0099" in p.file for p in g.problems)


def test_S23_graph_dir_inside_a_forbidden_location_is_refused(tmp_path):
    forbidden = tmp_path / "TK-PKA" / "data" / "graph"
    forbidden.mkdir(parents=True)
    with pytest.raises(SandboxViolation):
        load(forbidden)
