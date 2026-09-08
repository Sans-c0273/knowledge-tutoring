from __future__ import annotations

from kg_reasoner.traverse import bfs


def test_prerequisite_chain_depth_2(edu_index):
    reached = bfs(edu_index, "edu-0003", allowed_types={"prerequisite_of"}, max_depth=2)
    # edu-0002 is the immediate prerequisite (distance 1, direction "in": edu-0002 -> edu-0003)
    assert reached["edu-0002"].distance == 1
    assert reached["edu-0002"].direction == "in"
    # edu-0001 is two hops upstream
    assert reached["edu-0001"].distance == 2
    # edu-0006 is downstream (edu-0003 -> edu-0006): reachable, direction "out"
    assert reached["edu-0006"].distance == 1
    assert reached["edu-0006"].direction == "out"


def test_depth_limit_respected(edu_index):
    reached = bfs(edu_index, "edu-0003", allowed_types={"prerequisite_of"}, max_depth=1)
    assert "edu-0002" in reached
    assert "edu-0001" not in reached  # would need depth 2


def test_relation_filter_excludes_other_types(edu_index):
    reached = bfs(edu_index, "edu-0003", allowed_types={"prerequisite_of"}, max_depth=2)
    assert "edu-0004" not in reached  # part_of, not prerequisite_of
    assert "edu-0005" not in reached  # example_of, not prerequisite_of


def test_symmetric_related_to_both_ways(gen_index):
    reached = bfs(gen_index, "gen-0002", allowed_types={"related_to"}, max_depth=1)
    # Light Energy was stored as (gen-0001 -> gen-0002); traversing FROM gen-0002 must still find gen-0001.
    assert "gen-0001" in reached
    assert reached["gen-0001"].distance == 1
