"""WU3 — `kg.stages.dedup` (S6): candidate pairs from similarity ≥ `dedup.threshold`, exact-match
suppression, batching by `dedup.pairs_per_call`, verdict matching (order-insensitive, unmatched →
unsure, unrequested ignored + counted), queue-only default vs `write_same_as` (DESIGN §6, §8.0, §11;
D12, D29; PRD R12).

Interface (chosen): candidate_pairs(drafts, threshold, *, max_pairs=None) -> list[Pair(a_id, b_id, score)];
run(drafts, cfg, *, call_stage, ledger=None, write_same_as=None) -> DedupResult(same, unsure, different: list[Judged],
edges: list[Edge], unrequested: int, overflow: list[Pair]); Judged(a_id, b_id, score, verdict, reason);
write_queue(result, review_dir) -> Path (writes <review_dir>/duplicates.md).
Config: `dedup.pairs_per_call` (new, default 10) on kg.config.DedupConfig.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kg.config import load_config
from kg.schemas import DedupOutput
from wu3_fixtures import ScriptedCallStage, ids_in


def D(node_id, title, aliases=()):
    from kg.notes import NodeSource
    from kg.stages.consolidate import NodeDraft

    return NodeDraft(id=node_id, title=title, aliases=list(aliases), definition=f"{title}.", sources=[NodeSource(locator=f"a.md#heading={title}", file="a.md", quote=title)], review=[], chunk_ids=["a.md#c1"], source_files=["a.md"], is_new=True)


@pytest.fixture
def cfg(make_project):
    return load_config(make_project())


@pytest.fixture
def drafts():
    return [
        D("kc-0001", "Sample Space"),
        D("kc-0002", "The Sample Space Definition"),  # similar to kc-0001
        D("kc-0003", "Event"),
        D("kc-0004", "Bayes Theorem"),
        D("kc-0005", "Bayes Theorem Proof"),  # similar to kc-0004
    ]


def judged(a, b, verdict, reason="r"):
    return {"a_id": a, "b_id": b, "verdict": verdict, "reason": reason}


# ---------------------------------------------------------------- config


def test_S6_config_exposes_pairs_per_call_with_default_10(cfg, make_project):
    assert cfg.dedup.pairs_per_call == 10
    custom = load_config(make_project({"dedup.pairs_per_call": 3}, subdir="ppc"))
    assert custom.dedup.pairs_per_call == 3


def test_S6_pairs_per_call_must_be_a_positive_integer(make_project):
    from kg.config import ConfigError

    with pytest.raises(ConfigError):
        load_config(make_project({"dedup.pairs_per_call": 0}, subdir="ppc0"))


# --------------------------------------------------------- candidate_pairs


def test_S6_candidate_pairs_returns_similar_pairs_sorted_by_score_with_canonical_order(drafts, cfg):
    from kg.stages.dedup import candidate_pairs

    pairs = candidate_pairs(drafts, cfg.dedup.threshold)
    keys = {(p.a_id, p.b_id) for p in pairs}
    assert ("kc-0001", "kc-0002") in keys
    assert ("kc-0004", "kc-0005") in keys
    assert all(p.a_id < p.b_id for p in pairs)
    assert all(cfg.dedup.threshold <= p.score <= 1.0 for p in pairs)
    assert [p.score for p in pairs] == sorted((p.score for p in pairs), reverse=True)
    assert not any({p.a_id, p.b_id} & {"kc-0003"} for p in pairs), "Event resembles nothing"


def test_S6_candidate_pairs_suppresses_exact_normalised_matches():
    from kg.stages.dedup import candidate_pairs

    # consolidate already collapsed these; if two ids still share a normalised title it is not a dedup question
    pairs = candidate_pairs([D("kc-0001", "Sample Space"), D("kc-0002", "sample spaces")], 0.5)
    assert pairs == []


def test_S6_candidate_pairs_considers_aliases():
    from kg.stages.dedup import candidate_pairs

    pairs = candidate_pairs([D("kc-0001", "Sample Space", aliases=["Outcome Space"]), D("kc-0002", "Outcome Space Model")], 0.6)
    assert [(p.a_id, p.b_id) for p in pairs] == [("kc-0001", "kc-0002")]


def test_S6_candidate_pairs_threshold_is_respected():
    from kg.stages.dedup import candidate_pairs

    ds = [D("kc-0001", "Sample Space"), D("kc-0002", "Sample Space Definition")]
    assert candidate_pairs(ds, 0.99) == []
    assert len(candidate_pairs(ds, 0.5)) == 1


def test_S6_candidate_pairs_max_pairs_caps_the_list(drafts):
    from kg.stages.dedup import candidate_pairs

    assert len(candidate_pairs(drafts, 0.1, max_pairs=1)) == 1


def test_D10_thai_near_duplicates_are_candidates():
    from kg.stages.dedup import candidate_pairs

    pairs = candidate_pairs([D("kc-0001", "การเรียนรู้เชิงลึก"), D("kc-0002", "การเรียนรู้เชิงลึกเบื้องต้น")], 0.5)
    assert [(p.a_id, p.b_id) for p in pairs] == [("kc-0001", "kc-0002")]


def test_S6_candidate_pairs_of_one_or_zero_drafts_is_empty():
    from kg.stages.dedup import candidate_pairs

    assert candidate_pairs([], 0.6) == []
    assert candidate_pairs([D("kc-0001", "Alone")], 0.6) == []


# ------------------------------------------------------------------- run


def test_S6_run_batches_pairs_per_call_and_uses_the_dedup_schema(make_project, drafts):
    from kg.stages.dedup import run

    cfg = load_config(make_project({"dedup.pairs_per_call": 1, "dedup.threshold": 0.5}, subdir="b1"))
    fake = ScriptedCallStage({"dedup": lambda call: {"judgements": []}}, cfg=cfg)
    run(drafts, cfg, call_stage=fake)
    assert len(fake.calls) == 2  # two candidate pairs, one per call
    for call in fake.calls:
        assert call["stage"] == "dedup" and call["schema"] is DedupOutput
        assert len(ids_in(call["user_text"], "kc")) == 2, "each call carries exactly its own pair"
        assert str(call["extra"].get("prompt_version", "")).startswith("dedup@")


def test_S6_run_puts_all_pairs_in_one_call_when_under_pairs_per_call(cfg, drafts):
    from kg.stages.dedup import run

    fake = ScriptedCallStage({"dedup": lambda call: {"judgements": []}}, cfg=cfg)
    run(drafts, cfg, call_stage=fake)
    assert len(fake.calls) == 1
    assert set(ids_in(fake.calls[0]["user_text"], "kc")) == {"kc-0001", "kc-0002", "kc-0004", "kc-0005"}


def test_S6_verdicts_are_sorted_into_same_unsure_different_with_reasons(cfg, drafts):
    from kg.stages.dedup import run

    fake = ScriptedCallStage({"dedup": [{"judgements": [judged("kc-0001", "kc-0002", "same", "identical concept"), judged("kc-0004", "kc-0005", "different", "theorem vs its proof")]}]}, cfg=cfg)
    res = run(drafts, cfg, call_stage=fake)
    assert [(j.a_id, j.b_id, j.reason) for j in res.same] == [("kc-0001", "kc-0002", "identical concept")]
    assert [(j.a_id, j.b_id) for j in res.different] == [("kc-0004", "kc-0005")]
    assert res.unsure == []
    assert all(0.0 < j.score <= 1.0 for j in res.same + res.different)
    assert res.unrequested == 0


def test_S6_judgement_matching_is_order_insensitive(cfg, drafts):
    from kg.stages.dedup import run

    fake = ScriptedCallStage({"dedup": [{"judgements": [judged("kc-0002", "kc-0001", "same"), judged("kc-0005", "kc-0004", "different")]}]}, cfg=cfg)
    res = run(drafts, cfg, call_stage=fake)
    assert [(j.a_id, j.b_id) for j in res.same] == [("kc-0001", "kc-0002")]
    assert res.unsure == []


def test_S6_unmatched_pair_defaults_to_unsure_and_unrequested_judgement_is_ignored_and_counted(cfg, drafts):
    from kg.stages.dedup import run

    fake = ScriptedCallStage({"dedup": [{"judgements": [judged("kc-0001", "kc-0002", "same"), judged("kc-0001", "kc-0003", "same", "never asked")]}]}, cfg=cfg)
    res = run(drafts, cfg, call_stage=fake)
    assert [(j.a_id, j.b_id) for j in res.unsure] == [("kc-0004", "kc-0005")]
    assert res.unrequested == 1
    assert all((j.a_id, j.b_id) != ("kc-0001", "kc-0003") for j in res.same + res.unsure + res.different)


def test_S6_max_dedup_pairs_limits_adjudication_and_reports_overflow(make_project, drafts):
    from kg.stages.dedup import run

    cfg = load_config(make_project({"limits.max_dedup_pairs": 1}, subdir="cap1"))
    fake = ScriptedCallStage({"dedup": lambda call: {"judgements": []}}, cfg=cfg)
    res = run(drafts, cfg, call_stage=fake)
    assert len(ids_in(fake.calls[0]["user_text"], "kc")) == 2
    assert len(res.overflow) == 1
    assert len(res.unsure) == 1


def test_S6_no_candidate_pairs_means_no_model_call(cfg):
    from kg.stages.dedup import run

    fake = ScriptedCallStage({"dedup": lambda call: {"judgements": []}}, cfg=cfg)
    res = run([D("kc-0001", "Sample Space"), D("kc-0002", "Bayes Theorem")], cfg, call_stage=fake)
    assert fake.calls == []
    assert res.same == res.unsure == res.different == [] and res.edges == []


# -------------------------------------------------------- queue vs same_as


def test_D12_default_is_queue_only_no_same_as_edges(cfg, drafts, tmp_path: Path):
    from kg.stages.dedup import run, write_queue

    fake = ScriptedCallStage({"dedup": [{"judgements": [judged("kc-0001", "kc-0002", "same", "identical concept"), judged("kc-0004", "kc-0005", "unsure", "maybe")]}]}, cfg=cfg)
    res = run(drafts, cfg, call_stage=fake)
    assert cfg.dedup.write_same_as is False
    assert res.edges == []
    path = write_queue(res, tmp_path / "_review")
    assert path == tmp_path / "_review" / "duplicates.md"
    text = path.read_text(encoding="utf-8")
    assert "kc-0001" in text and "kc-0002" in text and "identical concept" in text
    assert "kc-0004" in text and "kc-0005" in text and "maybe" in text
    assert "Sample Space" in text, "titles make the queue reviewable by a human"


def test_R12_write_same_as_flag_writes_same_edges_and_queues_only_unsure(cfg, drafts, tmp_path: Path):
    from kg.stages.dedup import run, write_queue

    fake = ScriptedCallStage({"dedup": [{"judgements": [judged("kc-0001", "kc-0002", "same"), judged("kc-0004", "kc-0005", "unsure", "maybe")]}]}, cfg=cfg)
    res = run(drafts, cfg, call_stage=fake, write_same_as=True)
    assert [(e.type, e.source_id, e.target_id, e.relevance) for e in res.edges] == [("same_as", "kc-0001", "kc-0002", None)]
    text = write_queue(res, tmp_path / "_review").read_text(encoding="utf-8")
    assert "kc-0004" in text and "kc-0005" in text
    assert "kc-0001" not in text or "same_as" in text  # a written pair is not an open review item


def test_R12_write_same_as_from_config(make_project, drafts):
    from kg.stages.dedup import run

    cfg = load_config(make_project({"dedup.write_same_as": True}, subdir="wsa"))
    fake = ScriptedCallStage({"dedup": [{"judgements": [judged("kc-0001", "kc-0002", "same")]}]}, cfg=cfg)
    res = run(drafts, cfg, call_stage=fake)
    assert len(res.edges) == 1 and res.edges[0].type == "same_as"


def test_R12_different_verdicts_never_produce_edges_or_queue_rows(cfg, drafts, tmp_path: Path):
    from kg.stages.dedup import run, write_queue

    fake = ScriptedCallStage({"dedup": [{"judgements": [judged("kc-0001", "kc-0002", "different"), judged("kc-0004", "kc-0005", "different")]}]}, cfg=cfg)
    res = run(drafts, cfg, call_stage=fake, write_same_as=True)
    assert res.edges == []
    text = write_queue(res, tmp_path / "_review").read_text(encoding="utf-8")
    assert "kc-0001" not in text and "kc-0004" not in text


def test_R12_write_queue_overwrites_with_current_state_and_lists_overflow(make_project, drafts, tmp_path: Path):
    from kg.stages.dedup import run, write_queue

    cfg = load_config(make_project({"limits.max_dedup_pairs": 1}, subdir="cap1b"))
    fake = ScriptedCallStage({"dedup": lambda call: {"judgements": []}}, cfg=cfg)
    res = run(drafts, cfg, call_stage=fake)
    review = tmp_path / "_review"
    review.mkdir()
    (review / "duplicates.md").write_text("STALE CONTENT\n", encoding="utf-8")
    text = write_queue(res, review).read_text(encoding="utf-8")
    assert "STALE CONTENT" not in text
    assert "kc-0001" in text and "kc-0004" in text  # the unsure (unmatched) pair and the not-adjudicated overflow pair both appear
