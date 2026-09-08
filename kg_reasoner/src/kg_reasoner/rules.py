"""Tier 1 reasoner: deterministic rules mapping (context, graph position) -> selected nodes + roles.

This generalises the KM01-KM05 table from Part 2 of the design doc ("Enhancing
Knowledge Map to Strategy Selection") from a 5-node toy graph to the two real
edge vocabularies kg-mapper-poc actually produces (`education`, `general` —
see src/kg/schemas.py in kg-mapper-poc). It is deliberately NOT an LLM: these
are graph-topology rules, same spirit as the Trigger Matrix / Combination
Rules tables in Part 1 — cheap, deterministic, auditable. `llm_reasoner.py`
is the optional Tier 2 for when topology alone can't decide.

Design note carried over from the source docs (Part 2, p.2): a prerequisite
existing does not mean the student has a problem with it. Every "prerequisite_gap"
/ "alternative_explanation" selection here is a candidate to check or scaffold,
never an asserted diagnosis — callers (Strategy Selection / the LLM writing the
final response) must keep that framing.

Known gap (flag for the user, not solved here): neither schema has a
`misconception` edge or node-type today. Table C in Part 1 explicitly defers
misconception handling to a later version; when kg-mapper-poc adds a
`misconception_of` edge (or a node-level `kind: misconception` field), add a
KM06 rule here keyed on `learner_state == "confused"` with high confidence.

student_level (LEVEL_PROFILES / _relevance_filtered): this tunes breadth/depth
of node selection only — how many candidates, how far the education-schema
prerequisite search walks, how strong a general-schema related_to link has to
be to count. It never changes wording, tone, or which KM rule fires; that
split stays exactly where DESIGN.md puts it (Strategy Selection's job).
"""

from __future__ import annotations

from dataclasses import dataclass

from kg_reasoner.graph_index import PREREQUISITE_TYPE, GraphIndex, Neighbor
from kg_reasoner.models import CandidateNode, Role, StudentContext, StudentLevel
from kg_reasoner.traverse import bfs, direct_neighbors

MAX_SELECTIONS_DEFAULT = 5


@dataclass(frozen=True)
class LevelProfile:
    """How much grounding to surface for one student level.

    max_candidates - the default overall selection cap (apply_rules' max_selections,
                      when the caller doesn't override). This is enforced ONCE, centrally,
                      in apply_rules — individual rule functions below must not also slice
                      to it, or an explicit max_selections override could never widen past
                      what a rule already discarded internally.
    max_depth       - hops the education-schema prerequisite_of BFS may walk (KM02).
    min_relevance   - general schema only: a related_to edge below this score is
                      dropped, *unless* dropping it would leave zero candidates
                      (see _relevance_filtered) — a stricter level must never end
                      up with less grounding than a looser one would have offered.
    """

    max_candidates: int
    max_depth: int
    min_relevance: int


LEVEL_PROFILES: dict[StudentLevel, LevelProfile] = {
    "beginner": LevelProfile(max_candidates=2, max_depth=1, min_relevance=70),
    "intermediate": LevelProfile(max_candidates=3, max_depth=2, min_relevance=50),
    "advanced": LevelProfile(max_candidates=5, max_depth=2, min_relevance=0),
}


def _to_candidate(index: GraphIndex, node_id: str, *, role: Role, relation: str, distance: int, relevance: int | None) -> CandidateNode:
    node = index.nodes[node_id]
    return CandidateNode(
        id=node.id,
        title=node.title,
        role=role,
        relation=relation,
        distance=distance,
        relevance=relevance,
        grounded=node.grounded,
        definition_snippet=node.definition_plain[:280],
    )


def _neighbors_sorted_by_relevance(neighbors: list[Neighbor]) -> list[Neighbor]:
    return sorted(neighbors, key=lambda n: (n.relevance if n.relevance is not None else -1), reverse=True)


def _relevance_filtered(neighbors: list[Neighbor], *, exclude_ids: frozenset[str], min_relevance: int) -> tuple[list[Neighbor], bool]:
    """`neighbors` must already be sorted strongest-first (see `_neighbors_sorted_by_relevance`).

    Drops excluded ids, then applies the level's relevance floor. If the floor would leave
    nothing, falls back to the single strongest neighbor regardless of the floor — returning
    `True` as the second element so the caller can flag that the floor was relaxed rather than
    silently pretending it was met. Never returns more than the input already had.
    """
    candidates = [n for n in neighbors if n.node_id not in exclude_ids]
    strict = [n for n in candidates if n.relevance is None or n.relevance >= min_relevance]
    if strict:
        return strict, False
    return (candidates[:1], True) if candidates else ([], False)


# --------------------------------------------------------------- KM02: confused -> prerequisite_gap


def _prerequisite_gap(ctx: StudentContext, index: GraphIndex, anchor_id: str) -> list[CandidateNode]:
    profile = LEVEL_PROFILES[ctx.student_level]
    out: list[CandidateNode] = []
    if index.schema == "education":
        reached = bfs(index, anchor_id, allowed_types={PREREQUISITE_TYPE}, max_depth=profile.max_depth)
        # "in" direction = this neighbor is a prerequisite OF the anchor (upstream).
        # Count is bounded by apply_rules' overall cap, not here (see LevelProfile docstring).
        upstream = sorted((r for r in reached.values() if r.direction == "in"), key=lambda r: r.distance)
        for r in upstream:
            if r.node_id in ctx.mastered_node_ids or r.node_id in ctx.exclude_node_ids:
                continue
            out.append(_to_candidate(index, r.node_id, role="prerequisite_gap", relation=r.relation, distance=r.distance, relevance=None))
    else:
        # No directional prerequisite in the `general` schema. Best available proxy:
        # the broader topic this node is part_of (simpler framing), else the single
        # strongest related_to neighbor. Flagged as a proxy, not a diagnosis.
        parents = direct_neighbors(index, anchor_id, types={"part_of"})
        parents = [n for n in parents if n.direction == "out" and n.node_id not in ctx.exclude_node_ids]
        if parents:
            n = parents[0]
            out.append(_to_candidate(index, n.node_id, role="prerequisite_gap", relation="part_of (broader concept, proxy)", distance=1, relevance=None))
        else:
            related = _neighbors_sorted_by_relevance(direct_neighbors(index, anchor_id, types={"related_to"}))
            picked, relaxed = _relevance_filtered(related, exclude_ids=ctx.exclude_node_ids, min_relevance=profile.min_relevance)
            if picked:
                n = picked[0]
                tag = " below this level's relevance floor, shown anyway to avoid leaving a confused student with nothing" if relaxed else ""
                out.append(
                    _to_candidate(
                        index, n.node_id, role="prerequisite_gap", relation=f"related_to (proxy, no prerequisite edge in this schema{tag})", distance=1, relevance=n.relevance
                    )
                )
    return out


# --------------------------------------------------------------- KM03: clarify -> alternative_explanation


def _alternative_explanation(ctx: StudentContext, index: GraphIndex, anchor_id: str) -> list[CandidateNode]:
    profile = LEVEL_PROFILES[ctx.student_level]
    out: list[CandidateNode] = []
    if index.schema == "education":
        direct = direct_neighbors(index, anchor_id, types={"refines", "supersedes", "same_as"})
        for n in direct:
            if n.node_id in ctx.exclude_node_ids:
                continue
            out.append(_to_candidate(index, n.node_id, role="alternative_explanation", relation=n.type, distance=1, relevance=None))
        if not out:
            parents = [n for n in direct_neighbors(index, anchor_id, types={"part_of"}) if n.direction == "out"]
            for parent in parents[:1]:
                siblings = [n for n in direct_neighbors(index, parent.node_id, types={"part_of"}) if n.direction == "in" and n.node_id != anchor_id]
                for sib in siblings:
                    if sib.node_id in ctx.exclude_node_ids:
                        continue
                    out.append(_to_candidate(index, sib.node_id, role="alternative_explanation", relation="part_of sibling", distance=2, relevance=None))
    else:
        related = _neighbors_sorted_by_relevance(direct_neighbors(index, anchor_id, types={"related_to"}))
        picked, relaxed = _relevance_filtered(related, exclude_ids=ctx.exclude_node_ids, min_relevance=profile.min_relevance)
        for i, n in enumerate(picked):
            tag = " (below this level's relevance floor, shown anyway)" if relaxed and i == 0 else ""
            out.append(_to_candidate(index, n.node_id, role="alternative_explanation", relation=f"related_to{tag}", distance=1, relevance=n.relevance))
    return out


# --------------------------------------------------------------- KM04: practice_quiz -> practice_target


def _practice_target(ctx: StudentContext, index: GraphIndex, anchor_id: str) -> list[CandidateNode]:
    if index.schema == "education":
        candidates = [n for n in direct_neighbors(index, anchor_id, types={"example_of"}) if n.direction == "in"]
        relation = "example_of"
    else:
        candidates = [n for n in direct_neighbors(index, anchor_id, types={"part_of"}) if n.direction == "in"]
        relation = "part_of (child)"
        if not candidates:
            candidates = direct_neighbors(index, anchor_id, types={"related_to"})
            relation = "related_to"
    candidates = [n for n in candidates if n.node_id not in ctx.exclude_node_ids]
    # Prefer nodes that actually carry a source quote — don't quiz on ungrounded content.
    # Count is bounded by apply_rules' overall cap, not here (see LevelProfile docstring).
    candidates.sort(key=lambda n: index.nodes[n.node_id].grounded, reverse=True)
    out = [_to_candidate(index, n.node_id, role="practice_target", relation=relation, distance=1, relevance=n.relevance) for n in candidates]
    return out


# --------------------------------------------------------------- KM05: understanding demonstrated -> next_challenge


def _next_challenge(ctx: StudentContext, index: GraphIndex, anchor_id: str) -> tuple[list[CandidateNode], list[str]]:
    # Deliberately not level-scaled: "what's next" is singular by design at every level
    # (a beginner and an advanced student both get exactly one next topic to try) —
    # unlike the other rules, more candidates here wouldn't mean "more context", it would
    # mean "which of these five topics is actually next", which this rule can't answer.
    notes: list[str] = []
    if index.schema == "education":
        downstream = [n for n in direct_neighbors(index, anchor_id, types={PREREQUISITE_TYPE}) if n.direction == "out"]
        downstream = [n for n in downstream if n.node_id not in ctx.exclude_node_ids]
        out = [_to_candidate(index, n.node_id, role="next_challenge", relation=PREREQUISITE_TYPE, distance=1, relevance=None) for n in downstream[:1]]
        return out, notes
    notes.append(
        "KM05 (next_challenge) needs a directed progression edge; the 'general' schema's "
        "related_to is undirected, so no next-topic candidate was produced. Either set "
        "corpus.schema: education for this corpus, or drive topic sequencing from the "
        "teacher's syllabus order (Student-Level Selection) instead of the graph."
    )
    return [], notes


# --------------------------------------------------------------- KM01: default -> supporting_context


def _supporting_context(ctx: StudentContext, index: GraphIndex, anchor_id: str) -> list[CandidateNode]:
    profile = LEVEL_PROFILES[ctx.student_level]
    out: list[CandidateNode] = []
    if index.schema == "education":
        for n in direct_neighbors(index, anchor_id, types={"part_of"}):
            if n.direction == "out" and n.node_id not in ctx.exclude_node_ids:
                out.append(_to_candidate(index, n.node_id, role="supporting_context", relation="part_of (broader topic)", distance=1, relevance=None))
        for n in direct_neighbors(index, anchor_id, types={"example_of"}):
            if n.direction == "in" and n.node_id not in ctx.exclude_node_ids:
                out.append(_to_candidate(index, n.node_id, role="supporting_context", relation="example_of", distance=1, relevance=None))
    else:
        related = _neighbors_sorted_by_relevance(direct_neighbors(index, anchor_id, types={"related_to"}))
        picked, relaxed = _relevance_filtered(related, exclude_ids=ctx.exclude_node_ids, min_relevance=profile.min_relevance)
        for i, n in enumerate(picked):
            tag = " (below this level's relevance floor, shown anyway)" if relaxed and i == 0 else ""
            out.append(_to_candidate(index, n.node_id, role="supporting_context", relation=f"related_to{tag}", distance=1, relevance=n.relevance))
    return out


@dataclass(frozen=True)
class RuleResult:
    candidates: list[CandidateNode]
    notes: list[str]


def apply_rules(ctx: StudentContext, index: GraphIndex, anchor_id: str, *, max_selections: int | None = None) -> RuleResult:
    """Evaluate KM01-KM05 in priority order and return a capped, deduplicated selection.

    Priority (highest first, matches the special_handling > learner_state > intent
    precedence used by Strategy Selection's Trigger Matrix in Part 1):
    confused (KM02) > clarify (KM03) > practice_quiz (KM04) > demonstrated
    understanding (KM05) > default explain/solve/hint (KM01).

    `max_selections=None` (the default) uses `LEVEL_PROFILES[ctx.student_level].max_candidates`
    as the overall cap, so a beginner naturally gets fewer selections end-to-end than an
    advanced student even when several rules combine. Pass an explicit int to override that
    for one call (e.g. a caller with its own UI limit).
    """
    notes: list[str] = []
    ordered: list[CandidateNode] = []

    if ctx.learner_state == "confused":
        got = _prerequisite_gap(ctx, index, anchor_id)
        notes.append(f"KM02 (confused): {len(got)} prerequisite_gap candidate(s)")
        ordered += got

    if ctx.intent == "clarify":
        got = _alternative_explanation(ctx, index, anchor_id)
        notes.append(f"KM03 (clarify): {len(got)} alternative_explanation candidate(s)")
        ordered += got

    if ctx.intent == "practice_quiz":
        got = _practice_target(ctx, index, anchor_id)
        notes.append(f"KM04 (practice_quiz): {len(got)} practice_target candidate(s)")
        ordered += got

    if ctx.learner_state == "normal" and ctx.intent == "summarize_review":
        got, extra_notes = _next_challenge(ctx, index, anchor_id)
        notes.append(f"KM05 (understanding demonstrated): {len(got)} next_challenge candidate(s)")
        notes += extra_notes
        ordered += got

    if not ordered:
        got = _supporting_context(ctx, index, anchor_id)
        notes.append(f"KM01 (default): {len(got)} supporting_context candidate(s)")
        ordered += got

    if ctx.special_handling != "none":
        notes.append(
            f"special_handling={ctx.special_handling}: node selection is unaffected; "
            "apply the Teaching Policy (P02/P03) constraints downstream in Response Planner."
        )

    cap = max_selections if max_selections is not None else LEVEL_PROFILES[ctx.student_level].max_candidates
    seen: set[str] = set()
    deduped: list[CandidateNode] = []
    for c in ordered:
        if c.id in seen:
            continue
        seen.add(c.id)
        deduped.append(c)
        if len(deduped) >= cap:
            break
    # The per-rule notes above count what each rule FOUND, before this cap - make the final
    # trim visible too, so "KM01 found 5" next to "only 2 shown" is never left unexplained.
    if len(deduped) < len({c.id for c in ordered}):
        notes.append(f"student_level={ctx.student_level}: capped to {cap} selection(s) (source: " f"{'explicit max_selections' if max_selections is not None else 'LEVEL_PROFILES default'})")
    return RuleResult(candidates=deduped, notes=notes)


__all__ = ["LevelProfile", "LEVEL_PROFILES", "RuleResult", "apply_rules"]
