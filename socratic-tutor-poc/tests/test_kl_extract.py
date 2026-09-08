"""KL Map extraction tests (R18 draft authoring).

The provider is stubbed: every call returns a graph the test chose, so the
assertions are about consolidation, id handling, and what happens to a bad
graph — never about model quality. No network, no weights.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel

from socratic_tutor.domain.klmap import KLMap
from socratic_tutor.domain.rag.chunker import Chunk
from socratic_tutor.domain.rag.kl_extract import (
    ExtractedConcept,
    ExtractedConcepts,
    ProposedEdge,
    ProposedEdges,
    batch_chunks,
    draft_path,
    edges_system,
    extract_klmap,
    relation_vocabulary,
    render_passage,
)
from socratic_tutor.models.enums import Relation
from socratic_tutor.providers import (
    LLMProvider,
    Msg,
    ProviderConnectionError,
    StructuredResult,
    Usage,
    strict_json_schema,
)

COURSE = "ALG101"


def chunk(text: str, ref: str, lang: str = "en", topic: str | None = None, cid: str = "") -> Chunk:
    return Chunk(text=text, source_ref=ref, lang=lang, topic=topic, chunk_id=cid or ref)


CHUNKS = [
    chunk(
        "An inverse operation undoes another operation.",
        "Algebra Basics Ch.1 p.12",
        cid="c1-01",
        topic="Inverse Operations",
    ),
    chunk(
        "การดำเนินการผกผันคือการดำเนินการที่ย้อนกลับผลของอีกการดำเนินการหนึ่ง",
        "Algebra Basics Ch.1 p.13",
        lang="th",
        cid="c1-03",
        topic="Inverse Operations",
    ),
    chunk(
        "Two-step equations undo the constant first, then the coefficient.",
        "Algebra Basics Ch.2 p.23",
        cid="c2-03",
        topic="Two-Step Linear Equations",
    ),
]


class StubProvider(LLMProvider):
    """Returns queued values in order, one per `complete_structured` call."""

    name = "stub"

    def __init__(self, responses: Sequence[BaseModel | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    async def complete_structured(
        self,
        *,
        model: str,
        system: str,
        messages: Sequence[Msg],
        schema: type[BaseModel],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> StructuredResult:
        self.calls.append((system, messages[0].content))
        if not self.responses:
            raise AssertionError("StubProvider ran out of queued responses")
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return StructuredResult(
            value=nxt, usage=Usage(1, 1), latency_ms=1.0, model=model, provider=self.name
        )

    def stream_text(self, **kwargs: object):  # pragma: no cover — extraction never streams
        raise NotImplementedError


def concepts(*items: tuple[str, str, str | None]) -> ExtractedConcepts:
    return ExtractedConcepts(
        concepts=[ExtractedConcept(id=i, name=n, name_th=t) for i, n, t in items]
    )


def edges(*items: tuple[str, str, Relation]) -> ProposedEdges:
    return ProposedEdges(edges=[ProposedEdge(from_id=f, to_id=t, relation=r) for f, t, r in items])


@pytest.fixture
def drafts(tmp_path: Path) -> Path:
    return tmp_path / "drafts"


async def test_extracts_concepts_and_typed_edges(drafts: Path):
    provider = StubProvider(
        [
            concepts(("C001", "Inverse Operations", None), ("C002", "Two-Step Equations", None)),
            edges(("C001", "C002", Relation.PREREQUISITE_OF)),
        ]
    )

    klmap, report = await extract_klmap(CHUNKS, COURSE, provider, drafts_dir=drafts)

    assert [node.name for node in klmap.nodes] == ["Inverse Operations", "Two-Step Equations"]
    assert [(e.from_, e.to, e.relation) for e in klmap.edges] == [
        ("C001", "C002", Relation.PREREQUISITE_OF)
    ]
    assert report.ok
    assert klmap.course_id == COURSE


async def test_draft_is_written_to_the_drafts_directory_not_the_live_map(drafts: Path):
    provider = StubProvider([concepts(("C001", "Inverse Operations", None)), edges()])

    await extract_klmap(CHUNKS, COURSE, provider, drafts_dir=drafts)

    path = draft_path(COURSE, drafts)
    assert path == drafts / "klmap-ALG101.yaml"
    assert path.is_file()
    assert "Inverse Operations" in path.read_text(encoding="utf-8")


async def test_thai_name_attaches_to_the_english_concept(drafts: Path):
    """A Thai passage naming an existing concept enriches it instead of duplicating it."""
    provider = StubProvider(
        [
            concepts(
                ("C001", "Inverse Operations", None),
                ("C009", "Inverse Operations", "การดำเนินการผกผัน"),
            ),
            edges(),
        ]
    )

    klmap, _ = await extract_klmap(CHUNKS, COURSE, provider, drafts_dir=drafts)

    assert len(klmap.nodes) == 1
    assert klmap.nodes[0].name_th == "การดำเนินการผกผัน"


async def test_model_ids_are_renumbered_and_edges_follow(drafts: Path):
    """Whatever ids the model invents, the saved map uses canonical C001-style ids."""
    provider = StubProvider(
        [
            concepts(
                ("concept-alpha", "Inverse Operations", None), ("zzz", "Equation Balance", None)
            ),
            edges(("concept-alpha", "zzz", Relation.RELATED_TO)),
        ]
    )

    klmap, report = await extract_klmap(CHUNKS, COURSE, provider, drafts_dir=drafts)

    assert [node.id for node in klmap.nodes] == ["C001", "C002"]
    assert [(e.from_, e.to) for e in klmap.edges] == [("C001", "C002")]
    assert report.ok


async def test_an_invalid_graph_is_returned_with_its_errors_not_repaired(drafts: Path):
    """A prerequisite cycle survives into the draft, with the loader's error attached."""
    provider = StubProvider(
        [
            concepts(("C001", "A", None), ("C002", "B", None)),
            edges(
                ("C001", "C002", Relation.PREREQUISITE_OF),
                ("C002", "C001", Relation.PREREQUISITE_OF),
            ),
        ]
    )

    klmap, report = await extract_klmap(CHUNKS, COURSE, provider, drafts_dir=drafts)

    assert len(klmap.edges) == 2, "the cycle must survive for a human to see and fix"
    assert not report.ok
    assert any(issue.code == "KL009" for issue in report.errors)
    assert draft_path(COURSE, drafts).is_file(), "an invalid draft is still written for editing"


async def test_edges_naming_unknown_concepts_are_dropped_with_a_reason(drafts: Path):
    provider = StubProvider(
        [
            concepts(("C001", "Inverse Operations", None)),
            edges(("C001", "C404", Relation.RELATED_TO)),
        ]
    )

    klmap, report = await extract_klmap(CHUNKS, COURSE, provider, drafts_dir=drafts)

    assert klmap.edges == []
    assert any(issue.code == "KX102" and "C404" in issue.message for issue in report.warnings)


async def test_self_edges_and_duplicates_are_dropped_with_a_reason(drafts: Path):
    provider = StubProvider(
        [
            concepts(("C001", "A", None), ("C002", "B", None)),
            edges(
                ("C001", "C001", Relation.RELATED_TO),
                ("C001", "C002", Relation.USES),
                ("C001", "C002", Relation.USES),
            ),
        ]
    )

    klmap, report = await extract_klmap(CHUNKS, COURSE, provider, drafts_dir=drafts)

    assert len(klmap.edges) == 1
    codes = {issue.code for issue in report.warnings}
    assert {"KX103", "KX104"} <= codes


async def test_a_provider_failure_reports_the_error_and_keeps_what_it_had(drafts: Path):
    provider = StubProvider(
        [
            concepts(("C001", "Inverse Operations", None)),
            ProviderConnectionError("the model is unreachable"),
        ]
    )

    klmap, report = await extract_klmap(CHUNKS, COURSE, provider, drafts_dir=drafts)

    assert [node.name for node in klmap.nodes] == ["Inverse Operations"]
    assert klmap.edges == []
    assert any(issue.code == "KX004" for issue in report.errors)


async def test_no_content_is_an_error_not_a_crash(drafts: Path):
    klmap, report = await extract_klmap([], COURSE, StubProvider([]), drafts_dir=drafts)

    assert isinstance(klmap, KLMap)
    assert klmap.nodes == []
    assert any(issue.code == "KX001" for issue in report.errors)


async def test_progress_reaches_one_hundred(drafts: Path):
    seen: list[tuple[float, str]] = []
    provider = StubProvider([concepts(("C001", "A", None)), edges()])

    await extract_klmap(
        CHUNKS,
        COURSE,
        provider,
        drafts_dir=drafts,
        on_progress=lambda percent, message: seen.append((percent, message)),
    )

    percents = [percent for percent, _ in seen]
    assert percents == sorted(percents)
    assert percents[-1] == 100.0


async def test_the_roster_of_earlier_batches_is_carried_into_later_calls(drafts: Path):
    """Batch two must see batch one's concepts, or it re-invents them."""
    many = [chunk("word " * 900, f"Doc p.{index}", cid=f"c{index}") for index in range(1, 5)]
    assert len(batch_chunks(many)) > 1

    provider = StubProvider(
        [concepts(("C001", "First Concept", None))]
        + [concepts()] * (len(batch_chunks(many)) - 1)
        + [edges()] * len(batch_chunks(many))
    )

    await extract_klmap(many, COURSE, provider, drafts_dir=drafts)

    second_call_user_text = provider.calls[1][1]
    assert "First Concept" in second_call_user_text


def test_relation_vocabulary_covers_the_whole_closed_set():
    block = relation_vocabulary()

    for relation in Relation:
        assert relation.value in block
    assert relation_vocabulary() in edges_system()


def test_wire_schemas_are_strict_mode_compatible():
    """Both extraction schemas must survive the provider layer's strict builder."""
    for schema in (ExtractedConcepts, ProposedEdges):
        built = strict_json_schema(schema)
        assert built["additionalProperties"] is False
        assert built["required"] == list(built["properties"])


def test_render_passage_marks_each_unit_with_its_provenance():
    rendered = render_passage(CHUNKS[:2])

    assert "<<c1-01 | Algebra Basics Ch.1 p.12 | en>>" in rendered
    assert "<<c1-03 | Algebra Basics Ch.1 p.13 | th>>" in rendered


def test_batching_respects_the_token_budget():
    batches = batch_chunks(CHUNKS, budget=10)

    assert len(batches) == len(CHUNKS)
    assert [c.chunk_id for batch in batches for c in batch] == [c.chunk_id for c in CHUNKS]
