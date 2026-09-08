from __future__ import annotations

from kg_reasoner.models import StudentContext
from kg_reasoner.rules import apply_rules


def _ctx(**kw) -> StudentContext:
    base = dict(topic_query="Linear Equations", intent="explain", learner_state="normal", student_level="beginner")
    base.update(kw)
    return StudentContext(**base)


# ---------------------------------------------------------------- education schema


def test_km02_confused_returns_nearest_prerequisite(edu_index):
    ctx = _ctx(intent="solve", learner_state="confused")
    result = apply_rules(ctx, edu_index, "edu-0003")
    ids = [c.id for c in result.candidates]
    assert "edu-0002" in ids
    gap = next(c for c in result.candidates if c.id == "edu-0002")
    assert gap.role == "prerequisite_gap"
    assert gap.distance == 1


def test_km02_skips_mastered_prerequisite(edu_index):
    # edu-0001 is 2 hops upstream of edu-0003 - needs a level whose max_depth reaches
    # that far (see test_level_max_depth_limits_km02_education below for the beginner case).
    ctx = _ctx(intent="solve", learner_state="confused", mastered_node_ids=frozenset({"edu-0002"}), student_level="advanced")
    result = apply_rules(ctx, edu_index, "edu-0003")
    ids = [c.id for c in result.candidates]
    assert "edu-0002" not in ids
    assert "edu-0001" in ids  # next prerequisite up the chain


def test_km03_clarify_prefers_refines_partner(edu_index):
    ctx = _ctx(intent="clarify")
    result = apply_rules(ctx, edu_index, "edu-0003")
    roles = {c.id: c.role for c in result.candidates}
    assert roles.get("edu-0007") == "alternative_explanation"


def test_km03_falls_back_to_part_of_sibling_when_no_refines(edu_index):
    ctx = _ctx(intent="clarify", topic_query="Order of Operations")
    result = apply_rules(ctx, edu_index, "edu-0008")
    ids = [c.id for c in result.candidates]
    assert "edu-0003" in ids  # sibling via shared part_of parent (edu-0004)


def test_km04_practice_prefers_grounded_example(edu_index):
    ctx = _ctx(intent="practice_quiz")
    result = apply_rules(ctx, edu_index, "edu-0003")
    targets = [c for c in result.candidates if c.role == "practice_target"]
    assert targets and targets[0].id == "edu-0005"
    assert targets[0].grounded is True


def test_km05_next_challenge_from_prerequisite_of_downstream(edu_index):
    ctx = _ctx(intent="summarize_review", learner_state="normal")
    result = apply_rules(ctx, edu_index, "edu-0003")
    roles = {c.id: c.role for c in result.candidates}
    assert roles.get("edu-0006") == "next_challenge"


def test_km01_default_supporting_context(edu_index):
    ctx = _ctx(intent="explain", learner_state="normal")
    result = apply_rules(ctx, edu_index, "edu-0003")
    roles = [c.role for c in result.candidates]
    assert "supporting_context" in roles


def test_exclude_node_ids_respected(edu_index):
    ctx = _ctx(intent="clarify", exclude_node_ids=frozenset({"edu-0007"}))
    result = apply_rules(ctx, edu_index, "edu-0003")
    ids = [c.id for c in result.candidates]
    assert "edu-0007" not in ids


# ------------------------------------------------------------------ general schema


def test_km02_general_schema_uses_part_of_parent_as_proxy(gen_index):
    ctx = _ctx(intent="solve", learner_state="confused", topic_query="Photosynthesis")
    result = apply_rules(ctx, gen_index, "gen-0001")
    gap = next(c for c in result.candidates if c.role == "prerequisite_gap")
    assert gap.id == "gen-0008"
    assert "proxy" in gap.relation


def test_km04_general_practice_prefers_part_of_children(gen_index):
    ctx = _ctx(intent="practice_quiz", topic_query="Biology Basics")
    result = apply_rules(ctx, gen_index, "gen-0008")
    ids = {c.id for c in result.candidates if c.role == "practice_target"}
    assert ids == {"gen-0001", "gen-0009"}


def test_km05_general_schema_has_no_progression_edge(gen_index):
    ctx = _ctx(intent="summarize_review", learner_state="normal", topic_query="Photosynthesis")
    result = apply_rules(ctx, gen_index, "gen-0001")
    assert any("no next-topic candidate" in n for n in result.notes)
    # KM01 still fires as a fallback so the turn isn't empty-handed:
    assert any(c.role == "supporting_context" for c in result.candidates)


def test_km01_general_ranks_by_relevance(gen_index):
    ctx = _ctx(intent="explain", learner_state="normal", topic_query="Photosynthesis")
    result = apply_rules(ctx, gen_index, "gen-0001")
    scores = [c.relevance for c in result.candidates if c.relevance is not None]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 90  # Light Energy, the highest-relevance related_to neighbor


# ------------------------------------------------------------------ student_level scaling


def test_level_max_depth_limits_km02_education(edu_index):
    # edu-0001 is 2 hops upstream of edu-0003; beginner's max_depth=1 must not reach it.
    ctx = _ctx(intent="solve", learner_state="confused", mastered_node_ids=frozenset({"edu-0002"}), student_level="beginner")
    result = apply_rules(ctx, edu_index, "edu-0003")
    ids = [c.id for c in result.candidates]
    assert "edu-0001" not in ids
    # KM02 itself found nothing in reach (its only depth-1 candidate was mastered) - the
    # turn isn't left empty-handed, though: KM01 fires as the documented fallback.
    assert any("KM02 (confused): 0 prerequisite_gap" in n for n in result.notes)
    assert any(c.role == "supporting_context" for c in result.candidates)


def test_level_max_candidates_scales_km01_general(gen_index):
    ctx_beginner = _ctx(intent="explain", learner_state="normal", topic_query="Photosynthesis", student_level="beginner")
    ctx_advanced = _ctx(intent="explain", learner_state="normal", topic_query="Photosynthesis", student_level="advanced")
    beginner = apply_rules(ctx_beginner, gen_index, "gen-0001")
    advanced = apply_rules(ctx_advanced, gen_index, "gen-0001")
    assert len(beginner.candidates) == 2  # LEVEL_PROFILES["beginner"].max_candidates
    assert len(advanced.candidates) == 5  # all 5 related_to neighbors fit within max_candidates=5
    # advanced's set is a strict superset of beginner's (same ranking, just more of it)
    assert {c.id for c in beginner.candidates} <= {c.id for c in advanced.candidates}


def test_level_min_relevance_floor_relaxes_when_it_would_empty_out(gen_index):
    # gen-0010 (Mitochondria) is Cellular Respiration's only related_to neighbor, at
    # relevance 20 - below every level's floor. A beginner must still get it rather
    # than an empty turn, but the relation string must say the floor was relaxed.
    ctx = _ctx(intent="explain", learner_state="normal", topic_query="Cellular Respiration", student_level="beginner")
    result = apply_rules(ctx, gen_index, "gen-0009")
    ids = [c.id for c in result.candidates]
    assert "gen-0010" in ids
    picked = next(c for c in result.candidates if c.id == "gen-0010")
    assert "below this level's relevance floor" in picked.relation


def test_apply_rules_default_cap_derives_from_student_level(gen_index):
    ctx = _ctx(intent="explain", learner_state="normal", topic_query="Photosynthesis", student_level="beginner")
    result = apply_rules(ctx, gen_index, "gen-0001")  # no max_selections passed
    assert len(result.candidates) == 2  # beginner's max_candidates, not MAX_SELECTIONS_DEFAULT (5)


def test_apply_rules_explicit_max_selections_still_overrides_level(gen_index):
    ctx = _ctx(intent="explain", learner_state="normal", topic_query="Photosynthesis", student_level="beginner")
    result = apply_rules(ctx, gen_index, "gen-0001", max_selections=4)
    assert len(result.candidates) == 4  # explicit override wins over beginner's max_candidates=2
