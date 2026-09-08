"""Edge schemas and the Pydantic models sent to a provider as stage output schemas.

Spec: PRD §6.1 / §6.2, R8, R9, R10; DESIGN §6, §7.7, §8; DECISIONS D5, D6, D27.

Every model here is a *wire* schema: it obeys the intersection profile in
DESIGN §7.7 — ``extra="forbid"``, every field required (optional = ``X | None``
with no default), ``Literal`` for closed vocabularies, and no numeric/string/array
constraints. Ranges such as ``relevance`` 0–100 are enforced by the gates, not here.

Wire-model docstrings are exported by Pydantic as the schema ``description`` and
therefore reach the model. Keep them model-facing: describe what the field means
and how to fill it; do not reference requirement IDs, decisions or gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict

# ------------------------------------------------------------------ vocabularies

ORIGIN_VALUES: tuple[str, ...] = ("course_material", "external_knowledge")


@dataclass(frozen=True)
class EdgeType:
    """One row of a PRD §6 edge table."""

    name: str
    symmetric: bool
    scored: bool = False
    #: Cycle-checked by the `dag` gate (DESIGN §8.0 / §10). Only `prerequisite_of`.
    dag: bool = False


@dataclass(frozen=True)
class EdgeSchema:
    """A named edge vocabulary plus the Pydantic models generated for it (D6)."""

    name: str
    edges: dict[str, EdgeType]
    origin_values: tuple[str, ...] | None
    proposal_model: type[BaseModel]
    edges_output_model: type[BaseModel]

    def is_symmetric(self, edge: str) -> bool:
        return self.edges[edge].symmetric

    def is_scored(self, edge: str) -> bool:
        return self.edges[edge].scored

    def is_dag(self, edge: str) -> bool:
        return self.edges[edge].dag


def _table(*rows: EdgeType) -> dict[str, EdgeType]:
    return {row.name: row for row in rows}


_EDUCATION_EDGES = _table(
    EdgeType("prerequisite_of", symmetric=False, dag=True),
    EdgeType("part_of", symmetric=False),
    EdgeType("example_of", symmetric=False),
    EdgeType("refines", symmetric=False),
    EdgeType("supersedes", symmetric=False),
    EdgeType("same_as", symmetric=True),
)

_GENERAL_EDGES = _table(
    EdgeType("related_to", symmetric=True, scored=True),
    EdgeType("part_of", symmetric=False),
    EdgeType("same_as", symmetric=True),
)

EducationEdgeName = Literal["prerequisite_of", "part_of", "example_of", "refines", "supersedes", "same_as"]
GeneralEdgeName = Literal["related_to", "part_of", "same_as"]
DedupVerdict = Literal["same", "different", "unsure"]

# The Literal types above are the single place the vocabulary is spelled twice;
# fail loudly at import if they ever drift from the tables (D6).
if set(get_args(EducationEdgeName)) != set(_EDUCATION_EDGES):
    raise RuntimeError("kg.schemas: EducationEdgeName drifted from the education edge table")
if set(get_args(GeneralEdgeName)) != set(_GENERAL_EDGES):
    raise RuntimeError("kg.schemas: GeneralEdgeName drifted from the general edge table")


# ------------------------------------------------------------ wire base class


class WireModel(BaseModel):
    """Base for every model exported as a provider JSON schema (DESIGN §7.7).

    Never exported itself, so this docstring does not reach a model.
    """

    model_config = ConfigDict(extra="forbid")


# ----------------------------------------------------------- edge proposals


class EducationEdgeProposal(WireModel):
    """One directed relationship between two existing notes, read as source -> target.

    `type` is one of the education vocabulary; `source_id` and `target_id` are note
    ids taken verbatim from the provided roster. Never invent ids.
    """

    type: EducationEdgeName
    source_id: str
    target_id: str


class GeneralEdgeProposal(WireModel):
    """One relationship between two existing notes, read as source -> target.

    `type` is one of the general vocabulary; `source_id` and `target_id` are note
    ids taken verbatim from the provided roster. `relevance` is an integer 0-100
    stating how strongly the two notes relate; give it only on `related_to` edges
    and use null on every other type.
    """

    type: GeneralEdgeName
    source_id: str
    target_id: str
    relevance: int | None


class EducationEdgesOutput(WireModel):
    """All relationships you can support from this passage; an empty list is valid."""

    edges: list[EducationEdgeProposal]


class GeneralEdgesOutput(WireModel):
    """All relationships you can support from this passage; an empty list is valid."""

    edges: list[GeneralEdgeProposal]


# ------------------------------------------------------- other stage outputs


DescribeKind = Literal["diagram", "chart", "photo", "screenshot", "table", "other"]


class DescribeOutput(WireModel):
    """What this image conveys, as structured content rather than appearance.

    `kind` classifies the image. `title` is a short heading for it. `description`
    is one or more plain-prose paragraphs of its informational content. `elements`
    lists the labelled parts, boxes, series or columns; `text_visible` lists text
    that can be read in the image, verbatim; `relationships` lists connections
    between elements as short lines such as "A -> B (feeds)". Any list may be
    empty, for example for a photo with no labels.
    """

    kind: DescribeKind
    title: str
    description: str
    elements: list[str]
    text_visible: list[str]
    relationships: list[str]


class Quote(WireModel):
    """A verbatim passage copied exactly from the unit identified by `unit_id`. Do not paraphrase."""

    unit_id: str
    text: str


class CandidateNode(WireModel):
    """One atomic concept found in the passage.

    `title` names the concept; `definition` is plain prose with no wiki links;
    `aliases` are alternative names used in the material; `unit_ids` are the ids
    of every unit the concept is drawn from; `quotes` are exact passages from
    those units that ground the definition.
    """

    title: str
    definition: str
    aliases: list[str]
    unit_ids: list[str]
    quotes: list[Quote]


class AtomizeOutput(WireModel):
    """Every atomic concept in the passage; an empty list is valid if none is present."""

    nodes: list[CandidateNode]


class DedupJudgement(WireModel):
    """Verdict on whether notes `a_id` and `b_id` describe the same concept.

    `verdict` is `same`, `different` or `unsure`; `reason` is one sentence.
    """

    a_id: str
    b_id: str
    verdict: DedupVerdict
    reason: str


class DedupOutput(WireModel):
    """One judgement for every pair you were given, using the ids exactly as provided."""

    judgements: list[DedupJudgement]


# ------------------------------------------------------------------ registry

EDUCATION = EdgeSchema(
    name="education",
    edges=_EDUCATION_EDGES,
    origin_values=ORIGIN_VALUES,
    proposal_model=EducationEdgeProposal,
    edges_output_model=EducationEdgesOutput,
)

GENERAL = EdgeSchema(
    name="general",
    edges=_GENERAL_EDGES,
    origin_values=None,
    proposal_model=GeneralEdgeProposal,
    edges_output_model=GeneralEdgesOutput,
)

SCHEMAS: dict[str, EdgeSchema] = {EDUCATION.name: EDUCATION, GENERAL.name: GENERAL}

# Every model that is ever passed to a provider as a stage schema (DESIGN §17 "Schema profile").
STAGE_OUTPUT_MODELS: tuple[type[BaseModel], ...] = (
    DescribeOutput,
    AtomizeOutput,
    EducationEdgesOutput,
    GeneralEdgesOutput,
    DedupOutput,
)
