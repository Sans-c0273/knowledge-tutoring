"""WU0 — structured-output schema profile (DESIGN §7.7, DECISIONS D5 / D27, R21).

Every Pydantic model that is sent to a provider as a JSON Schema must satisfy the
intersection profile of the three back-ends:

  forbidden : minimum, maximum, exclusiveMinimum, exclusiveMaximum, multipleOf,
              minLength, maxLength, pattern, minItems, maxItems, format, $schema
  required  : additionalProperties: false on every object; every property listed
              in `required` (optional fields are typed `X | None`)
  forbidden : recursion ($ref back to an ancestor definition)

Tests run against `Model.model_json_schema()` directly (no dependency on kg.llm)
AND against `kg.llm.schema_profile.export()` which the design says must raise on
a violating model. Exception name chosen: kg.llm.schema_profile.SchemaProfileError.
"""

from __future__ import annotations

from typing import Any, Iterator

import pytest
from pydantic import BaseModel, ConfigDict, Field

from kg.schemas import STAGE_OUTPUT_MODELS, DescribeOutput, EducationEdgeProposal, GeneralEdgeProposal

FORBIDDEN_KEYWORDS = {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "minItems",
    "maxItems",
    "format",
    "$schema",
}

ALL_MODELS: tuple[type[BaseModel], ...] = tuple(dict.fromkeys((*STAGE_OUTPUT_MODELS, EducationEdgeProposal, GeneralEdgeProposal)))


def _walk(node: Any, path: str = "$") -> Iterator[tuple[str, dict]]:
    """Yield (json-path, dict) for every dict in the schema tree."""
    if isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")


def _ids(models):
    return [m.__name__ for m in models]


@pytest.mark.parametrize("model", ALL_MODELS, ids=_ids(ALL_MODELS))
def test_R21_no_forbidden_keyword_in_exported_schema(model: type[BaseModel]):
    schema = model.model_json_schema()
    offenders = [(p, k) for p, d in _walk(schema) for k in d if k in FORBIDDEN_KEYWORDS]
    assert offenders == [], f"{model.__name__}: forbidden JSON-Schema keywords {offenders}"


@pytest.mark.parametrize("model", ALL_MODELS, ids=_ids(ALL_MODELS))
def test_D27_every_object_forbids_additional_properties(model: type[BaseModel]):
    schema = model.model_json_schema()
    objects = [(p, d) for p, d in _walk(schema) if d.get("type") == "object" or "properties" in d]
    assert objects, "schema has no object"
    for p, d in objects:
        assert d.get("additionalProperties") is False, f"{model.__name__} {p}: additionalProperties must be false"


@pytest.mark.parametrize("model", ALL_MODELS, ids=_ids(ALL_MODELS))
def test_D27_every_property_is_required(model: type[BaseModel]):
    schema = model.model_json_schema()
    for p, d in _walk(schema):
        if "properties" in d:
            assert set(d.get("required", [])) == set(d["properties"]), f"{model.__name__} {p}: optional fields must be `X | None` and still required"


@pytest.mark.parametrize("model", ALL_MODELS, ids=_ids(ALL_MODELS))
def test_D27_no_recursive_definitions(model: type[BaseModel]):
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})
    for name, body in defs.items():
        refs = {d["$ref"] for _, d in _walk(body) if "$ref" in d}
        assert f"#/$defs/{name}" not in refs, f"{model.__name__}: $defs.{name} references itself"


def test_D29_describe_output_six_field_shape_is_profiled():
    # D29 / §8.0: the grown DescribeOutput is one of the five root models and must
    # survive the profile with all six properties and no array-count constraints.
    assert DescribeOutput in ALL_MODELS
    schema = DescribeOutput.model_json_schema()
    assert set(schema["properties"]) == {"kind", "title", "description", "elements", "text_visible", "relationships"}
    for name in ("elements", "text_visible", "relationships"):
        prop = schema["properties"][name]
        assert prop.get("type") == "array" and prop["items"] == {"type": "string"}
        assert not ({"minItems", "maxItems"} & set(prop))


def test_R10_relevance_carries_no_numeric_bounds_in_schema():
    # D5: 0–100 is enforced by the gate, not the wire schema.
    rel = GeneralEdgeProposal.model_json_schema()["properties"]["relevance"]
    for _, d in _walk(rel):
        assert not (set(d) & {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"})


# --------------------------------------------- kg.llm.schema_profile.export


def test_R21_schema_profile_export_accepts_every_stage_model():
    from kg.llm.schema_profile import export

    for model in ALL_MODELS:
        out = export(model)
        assert isinstance(out, dict)
        assert out.get("type") == "object"
        assert "$schema" not in out


def test_R21_schema_profile_export_rejects_numeric_bounds():
    from kg.llm.schema_profile import SchemaProfileError, export

    class Bad(BaseModel):
        model_config = ConfigDict(extra="forbid")
        score: int = Field(ge=0, le=100)

    with pytest.raises(SchemaProfileError):
        export(Bad)


def test_R21_schema_profile_export_rejects_string_constraints():
    from kg.llm.schema_profile import SchemaProfileError, export

    class Bad(BaseModel):
        model_config = ConfigDict(extra="forbid")
        slug: str = Field(min_length=1, pattern=r"^[a-z-]+$")

    with pytest.raises(SchemaProfileError):
        export(Bad)


def test_R21_schema_profile_export_rejects_missing_additional_properties_false():
    from kg.llm.schema_profile import SchemaProfileError, export

    class Loose(BaseModel):  # default extra="ignore" -> no additionalProperties: false
        title: str

    with pytest.raises(SchemaProfileError):
        export(Loose)


def test_R21_schema_profile_export_rejects_recursive_model():
    from kg.llm.schema_profile import SchemaProfileError, export

    class Node(BaseModel):
        model_config = ConfigDict(extra="forbid")
        name: str
        children: list[Node]

    Node.model_rebuild()
    with pytest.raises(SchemaProfileError) as ei:
        export(Node)
    assert "recurs" in str(ei.value).lower(), str(ei.value)


def test_R21_schema_profile_export_rejects_mutual_recursion_across_defs():
    from kg.llm.schema_profile import SchemaProfileError, export

    class Left(BaseModel):
        model_config = ConfigDict(extra="forbid")
        right: Right | None

    class Right(BaseModel):
        model_config = ConfigDict(extra="forbid")
        left: Left | None

    class Root(BaseModel):
        model_config = ConfigDict(extra="forbid")
        start: Left

    Left.model_rebuild()
    Root.model_rebuild()
    with pytest.raises(SchemaProfileError) as ei:
        export(Root)
    assert "recurs" in str(ei.value).lower(), str(ei.value)


def test_R21_schema_profile_export_rejects_datetime_field_because_of_format():
    from datetime import datetime

    from kg.llm.schema_profile import SchemaProfileError, export

    class Stamped(BaseModel):
        model_config = ConfigDict(extra="forbid")
        title: str
        when: datetime  # -> {"type": "string", "format": "date-time"}; `format` is forbidden

    with pytest.raises(SchemaProfileError) as ei:
        export(Stamped)
    assert "format" in str(ei.value)


def test_R21_schema_profile_export_rejects_optional_not_listed_in_required():
    from kg.llm.schema_profile import SchemaProfileError, export

    class HasDefault(BaseModel):
        model_config = ConfigDict(extra="forbid")
        title: str
        note: str | None = None  # default => Pydantic drops it from `required`

    with pytest.raises(SchemaProfileError):
        export(HasDefault)
