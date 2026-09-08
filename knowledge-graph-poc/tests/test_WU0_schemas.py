"""WU0 — kg.schemas: EdgeSchema definitions, edge-proposal models, DescribeOutput.

Spec: PRD §6.1 / §6.2, R6 (image description), R8 (two schemas, edge types
outside the selected schema rejected), R9 (`origin` is an education-only node
field, descriptive only), R10 (`relevance` 0–100 only on general `related_to`),
DESIGN §4.3, §8.0, §8.1, §10; DECISIONS D5 / D6 / D29.

Names (DESIGN §8.0 / §8.1 / §20):
  kg.schemas.EdgeSchema        .name, .edges: dict[str, EdgeType], .origin_values, .proposal_model
  kg.schemas.EdgeType          .name, .symmetric: bool, .scored: bool, .dag: bool
  kg.schemas.EDUCATION, GENERAL, SCHEMAS = {"education": EDUCATION, "general": GENERAL}
  kg.schemas.ORIGIN_VALUES = ("course_material", "external_knowledge")
  kg.schemas.EducationEdgeProposal(type, source_id, target_id)
  kg.schemas.GeneralEdgeProposal(type, source_id, target_id, relevance: int | None)
  kg.schemas.DescribeOutput(kind, title, description, elements, text_visible, relationships)
  kg.schemas.STAGE_OUTPUT_MODELS   tuple of every Pydantic model sent as a stage schema

D29 split of responsibilities (wire model vs gate):
  * The WIRE model is permissive about `relevance`: `int | None` on every general
    edge type. It no longer carries a `@model_validator`.
  * The RULE "related_to must carry relevance; other types must not" is the §10
    `relevance` gate (kg.gates, built in WU3). R10 still holds at the output level.
    The gate tests live in tests/test_WU3_gates.py (`test_R10_gate_*`); this file
    only pins the wire model.
  * Out-of-schema edge TYPES are still rejected on the wire by the per-schema
    `Literal` (§8.1: the `out_of_schema` gate is a "defensive second layer").
  * Range checking of `relevance` (0–100) is NOT tested at model level: per
    DESIGN §7.7 / D5 the model is a plain `int` and the gate owns the range.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from kg.schemas import (
    EDUCATION,
    GENERAL,
    ORIGIN_VALUES,
    SCHEMAS,
    STAGE_OUTPUT_MODELS,
    DescribeOutput,
    EdgeSchema,
    EdgeType,
    EducationEdgeProposal,
    GeneralEdgeProposal,
)

DESCRIBE_KINDS = {"diagram", "chart", "photo", "screenshot", "table", "other"}
DESCRIBE_FIELDS = {"kind", "title", "description", "elements", "text_visible", "relationships"}

EDU_TYPES = {"prerequisite_of", "part_of", "example_of", "refines", "supersedes", "same_as"}
GEN_TYPES = {"related_to", "part_of", "same_as"}


# ------------------------------------------------------------ vocabularies


def test_R8_registry_maps_exactly_the_two_schema_names():
    assert set(SCHEMAS) == {"education", "general"}
    assert SCHEMAS["education"] is EDUCATION
    assert SCHEMAS["general"] is GENERAL
    assert EDUCATION.name == "education"
    assert GENERAL.name == "general"


def test_R8_education_edge_types_match_PRD_6_1():
    assert isinstance(EDUCATION, EdgeSchema)
    assert set(EDUCATION.edges) == EDU_TYPES


def test_R8_general_edge_types_match_PRD_6_2():
    assert isinstance(GENERAL, EdgeSchema)
    assert set(GENERAL.edges) == GEN_TYPES


def test_R8_edge_type_entries_know_their_own_name():
    for schema in (EDUCATION, GENERAL):
        for name, et in schema.edges.items():
            assert isinstance(et, EdgeType)
            assert et.name == name


# --------------------------------------------------- direction / symmetry


@pytest.mark.parametrize(
    "schema,symmetric",
    [
        (EDUCATION, {"same_as"}),
        (GENERAL, {"related_to", "same_as"}),
    ],
)
def test_R8_symmetric_edges_are_exactly_the_A_B_rows_of_the_PRD(schema: EdgeSchema, symmetric: set[str]):
    assert {n for n, et in schema.edges.items() if et.symmetric} == symmetric


def test_R8_directed_edges_are_not_symmetric():
    for n in ("prerequisite_of", "part_of", "example_of", "refines", "supersedes"):
        assert EDUCATION.edges[n].symmetric is False
    assert GENERAL.edges["part_of"].symmetric is False


# --------------------------------------------------------- scored flag


def test_R10_only_general_related_to_is_scored():
    assert {n for n, et in GENERAL.edges.items() if et.scored} == {"related_to"}
    assert {n for n, et in EDUCATION.edges.items() if et.scored} == set()


# ------------------------------------------------------------ dag flag (D29)


def test_D29_edge_type_has_a_dag_flag_defaulting_false():
    # DESIGN §8.0 / §8.1: EdgeType(name, symmetric, scored=False, dag=False)
    et = EdgeType("x", symmetric=False)
    assert et.dag is False
    assert isinstance(EDUCATION.edges["prerequisite_of"].dag, bool)


def test_R13_only_prerequisite_of_is_dag_checked():
    # gates.py asks the schema which types are cycle-checked (§8.0) instead of hard-coding the name.
    assert {n for n, et in EDUCATION.edges.items() if et.dag} == {"prerequisite_of"}
    assert {n for n, et in GENERAL.edges.items() if et.dag} == set()


# -------------------------------------------------------- origin (R9)


def test_R9_origin_values_are_the_two_PRD_values():
    assert tuple(ORIGIN_VALUES) == ("course_material", "external_knowledge")


def test_R9_origin_is_a_node_field_only_under_education():
    assert tuple(EDUCATION.origin_values) == ("course_material", "external_knowledge")
    assert GENERAL.origin_values is None


# ------------------------------------------------ proposal model wiring


def test_D6_each_schema_points_at_its_own_proposal_model():
    assert EDUCATION.proposal_model is EducationEdgeProposal
    assert GENERAL.proposal_model is GeneralEdgeProposal
    assert issubclass(EducationEdgeProposal, BaseModel)
    assert issubclass(GeneralEdgeProposal, BaseModel)


def _enum_of_type_field(model: type[BaseModel]) -> set[str]:
    props = model.model_json_schema()["properties"]
    t = props["type"]
    if "enum" in t:
        return set(t["enum"])
    return {t["const"]}


def test_D6_education_proposal_type_literal_is_the_six_education_edges():
    assert _enum_of_type_field(EducationEdgeProposal) == EDU_TYPES


def test_D6_general_proposal_type_literal_is_the_three_general_edges():
    assert _enum_of_type_field(GeneralEdgeProposal) == GEN_TYPES


def test_D6_education_proposal_has_no_relevance_field():
    assert "relevance" not in EducationEdgeProposal.model_fields


def test_D6_general_proposal_has_a_relevance_field_typed_int_or_none():
    assert "relevance" in GeneralEdgeProposal.model_fields
    schema = GeneralEdgeProposal.model_json_schema()
    assert "relevance" in schema["required"]  # optional = `int | None`, still listed (§7.7)


# ------------------------------------------------ proposal model: accept


def test_R8_education_accepts_every_education_edge_type():
    for t in EDU_TYPES:
        p = EducationEdgeProposal(type=t, source_id="n-0001", target_id="n-0002")
        assert p.type == t


def test_R10_general_accepts_related_to_with_integer_relevance():
    p = GeneralEdgeProposal(type="related_to", source_id="n-0001", target_id="n-0002", relevance=72)
    assert p.relevance == 72


def test_R10_general_accepts_unscored_edges_with_relevance_none():
    for t in ("part_of", "same_as"):
        p = GeneralEdgeProposal(type=t, source_id="n-0001", target_id="n-0002", relevance=None)
        assert p.relevance is None


# ------------------------------------------------ proposal model: reject


def test_R8_general_rejects_prerequisite_of():
    with pytest.raises(ValidationError):
        GeneralEdgeProposal(type="prerequisite_of", source_id="a", target_id="b", relevance=None)


@pytest.mark.parametrize("t", sorted(EDU_TYPES - GEN_TYPES))
def test_R8_general_rejects_every_education_only_type(t: str):
    with pytest.raises(ValidationError):
        GeneralEdgeProposal(type=t, source_id="a", target_id="b", relevance=None)


def test_R8_education_rejects_related_to():
    with pytest.raises(ValidationError):
        EducationEdgeProposal(type="related_to", source_id="a", target_id="b")


def test_R10_education_rejects_a_relevance_on_any_edge():
    for t in EDU_TYPES:
        with pytest.raises(ValidationError):
            EducationEdgeProposal(type=t, source_id="a", target_id="b", relevance=50)


@pytest.mark.parametrize("t", ["part_of", "same_as"])
def test_D29_general_wire_model_accepts_relevance_on_non_related_to(t: str):
    # D29: the wire model is permissive; the §10 `relevance` gate rejects + logs this edge.
    p = GeneralEdgeProposal(type=t, source_id="a", target_id="b", relevance=50)
    assert p.relevance == 50


def test_D29_general_wire_model_accepts_related_to_without_relevance():
    # D29: accepted on the wire so one bad edge does not fail the whole chunk output
    # or burn a repair-retry; the §10 `relevance` gate rejects it per edge at write time.
    p = GeneralEdgeProposal(type="related_to", source_id="a", target_id="b", relevance=None)
    assert p.relevance is None


def test_D29_general_proposal_has_no_model_validator():
    # The rule lives in the gate; a validator on the model would re-introduce the D29 failure mode.
    assert not GeneralEdgeProposal.__pydantic_decorators__.model_validators


def test_R10_general_rejects_non_integer_relevance():
    with pytest.raises(ValidationError):
        GeneralEdgeProposal(type="related_to", source_id="a", target_id="b", relevance="high")
    with pytest.raises(ValidationError):
        GeneralEdgeProposal(type="related_to", source_id="a", target_id="b", relevance=72.5)


def test_R8_unknown_edge_type_rejected_in_both_schemas():
    with pytest.raises(ValidationError):
        EducationEdgeProposal(type="depends_on", source_id="a", target_id="b")
    with pytest.raises(ValidationError):
        GeneralEdgeProposal(type="depends_on", source_id="a", target_id="b", relevance=None)


def test_D27_proposal_models_forbid_extra_fields():
    with pytest.raises(ValidationError):
        EducationEdgeProposal(type="part_of", source_id="a", target_id="b", origin="course_material")
    with pytest.raises(ValidationError):
        GeneralEdgeProposal(type="part_of", source_id="a", target_id="b", relevance=None, weight=1)


def test_D27_proposal_models_require_endpoints():
    with pytest.raises(ValidationError):
        EducationEdgeProposal(type="part_of", source_id="a")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        GeneralEdgeProposal(type="part_of", target_id="b", relevance=None)  # type: ignore[call-arg]


# ------------------------------------------- R10 gate rule (WU3)
#
# D29 moved the relevance rule off the wire model and into the §10 `relevance`
# gate. The real gate tests live in tests/test_WU3_gates.py
# (test_R10_gate_* — `kg.gates.relevance(edge, schema) -> Rejection | None`).


# ------------------------------------------------ DescribeOutput (R6, D29)


def _describe(**overrides):
    doc = {
        "kind": "diagram",
        "title": "Bayes' theorem as a flow",
        "description": "Prior and likelihood feed a posterior box.",
        "elements": ["Prior", "Likelihood", "Posterior"],
        "text_visible": ["P(A|B) = P(B|A)P(A)/P(B)"],
        "relationships": ["Prior -> Posterior", "Likelihood -> Posterior"],
    }
    doc.update(overrides)
    return doc


def test_R6_describe_output_has_exactly_the_six_D29_fields():
    assert set(DescribeOutput.model_fields) == DESCRIBE_FIELDS


def test_R6_describe_output_kind_literal_matches_DESIGN_4_3():
    kind = DescribeOutput.model_json_schema()["properties"]["kind"]
    assert set(kind["enum"]) == DESCRIBE_KINDS


def test_R6_describe_output_every_field_required():
    assert set(DescribeOutput.model_json_schema()["required"]) == DESCRIBE_FIELDS


def test_R6_describe_output_accepts_a_full_document():
    out = DescribeOutput(**_describe())
    assert out.kind == "diagram"
    assert out.elements == ["Prior", "Likelihood", "Posterior"]
    assert out.relationships[0] == "Prior -> Posterior"


@pytest.mark.parametrize("kind", sorted(DESCRIBE_KINDS))
def test_R6_describe_output_accepts_every_kind(kind: str):
    assert DescribeOutput(**_describe(kind=kind)).kind == kind


def test_R6_describe_output_empty_lists_are_valid_for_a_plain_photo():
    # §8.0: "an empty list is valid for photos with no labelled elements".
    out = DescribeOutput(**_describe(kind="photo", elements=[], text_visible=[], relationships=[]))
    assert out.elements == [] and out.text_visible == [] and out.relationships == []


def test_R6_describe_output_rejects_the_old_description_only_shape():
    # Pre-D29 shape `{description}` must no longer validate.
    with pytest.raises(ValidationError):
        DescribeOutput(description="a diagram")  # type: ignore[call-arg]


@pytest.mark.parametrize("missing", sorted(DESCRIBE_FIELDS))
def test_R6_describe_output_rejects_a_missing_field(missing: str):
    doc = _describe()
    del doc[missing]
    with pytest.raises(ValidationError):
        DescribeOutput(**doc)


def test_R6_describe_output_rejects_unknown_kind():
    with pytest.raises(ValidationError):
        DescribeOutput(**_describe(kind="infographic"))


def test_R6_describe_output_forbids_extra_fields():
    with pytest.raises(ValidationError):
        DescribeOutput(**_describe(caption="extra"))


def test_R6_describe_output_list_fields_must_be_lists_of_strings():
    with pytest.raises(ValidationError):
        DescribeOutput(**_describe(elements="Prior, Likelihood"))
    with pytest.raises(ValidationError):
        DescribeOutput(**_describe(text_visible=[1, 2]))


# ------------------------------------------- stage-output model registry


def test_R21_stage_output_models_registry_is_non_empty_and_all_pydantic():
    assert len(STAGE_OUTPUT_MODELS) >= 2
    for m in STAGE_OUTPUT_MODELS:
        assert isinstance(m, type) and issubclass(m, BaseModel), m


def test_R21_stage_output_models_registry_covers_both_edge_schemas():
    # Either the proposal models themselves or a wrapper that embeds them must be registered.
    joined = " ".join(str(m.model_json_schema()) for m in STAGE_OUTPUT_MODELS)
    assert "prerequisite_of" in joined
    assert "related_to" in joined


def test_R21_stage_output_models_registry_includes_describe_output():
    assert DescribeOutput in STAGE_OUTPUT_MODELS
