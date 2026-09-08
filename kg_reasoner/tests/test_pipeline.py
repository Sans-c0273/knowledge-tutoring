from __future__ import annotations

from kg_reasoner.models import StudentContext
from kg_reasoner.pipeline import reason


def test_no_anchor_match_returns_empty_guidance(edu_index):
    ctx = StudentContext(topic_query="Quantum Field Theory", intent="explain")
    guidance = reason(ctx, edu_index)
    assert guidance.anchor is None
    assert guidance.selections == ()
    assert "No node matched" in guidance.notes[0]


def test_end_to_end_confused_student(edu_index):
    ctx = StudentContext(topic_query="Linear Equations", intent="solve", learner_state="confused")
    guidance = reason(ctx, edu_index)
    assert guidance.anchor is not None and guidance.anchor.id == "edu-0003"
    assert any(c.role == "prerequisite_gap" for c in guidance.selections)
    as_dict = guidance.to_dict()
    assert as_dict["knowledge_guidance"]["anchor"]["id"] == "edu-0003"


def test_llm_rerank_never_hard_fails(edu_index):
    ctx = StudentContext(topic_query="Linear Equations", intent="explain")

    def broken_call(**kwargs):
        raise RuntimeError("provider down")

    # max_selections is high and llm_top_n low, forcing a rerank attempt that then fails.
    guidance = reason(ctx, edu_index, max_selections=5, llm_call=broken_call, llm_top_n=1)
    assert guidance.selections  # Tier 1 result preserved despite Tier 2 failing


def test_llm_rerank_drops_hallucinated_ids(edu_index):
    ctx = StudentContext(topic_query="Linear Equations", intent="explain")

    def call(*, system, user, schema_json):
        return {"selections": [{"id": "not-a-real-id", "reason": "made up"}]}

    guidance = reason(ctx, edu_index, max_selections=5, llm_call=call, llm_top_n=1)
    # No valid id chosen -> falls back to Tier 1's own candidates, unchanged.
    assert all(c.id != "not-a-real-id" for c in guidance.selections)
    assert guidance.selections
