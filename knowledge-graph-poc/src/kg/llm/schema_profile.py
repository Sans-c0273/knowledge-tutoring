"""Structured-output schema profile — the intersection of all three back-ends (DESIGN §7.7, D27).

`export(Model)` returns `Model.model_json_schema()` after asserting:

  * none of the forbidden keywords appears anywhere in the schema;
  * every object has `additionalProperties: false`;
  * every object lists every property in `required`;
  * no `$ref` cycle exists among `$defs` (no recursion);
  * the root is an object.

A violation raises `SchemaProfileError` listing every offending location, so a
stage schema never needs to know which provider will run it.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel

FORBIDDEN_KEYWORDS: frozenset[str] = frozenset(
    {
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
)

# Keys whose *children's keys* are user-chosen names, not JSON-Schema keywords.
_NAME_CONTAINERS: frozenset[str] = frozenset({"properties", "$defs", "definitions", "patternProperties"})


class SchemaProfileError(Exception):
    """A stage schema violates the structured-output profile."""


def _walk(node: Any, path: str = "$") -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (json-path, dict) for every schema dict, skipping name-container levels."""
    if isinstance(node, dict):
        yield path, node
        for key, value in node.items():
            if key in _NAME_CONTAINERS and isinstance(value, dict):
                for name, sub in value.items():
                    yield from _walk(sub, f"{path}.{key}.{name}")
            else:
                yield from _walk(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _walk(value, f"{path}[{i}]")


def _is_object(node: dict[str, Any]) -> bool:
    return node.get("type") == "object" or "properties" in node


def _ref_target(ref: str) -> str | None:
    for prefix in ("#/$defs/", "#/definitions/"):
        if ref.startswith(prefix):
            return ref[len(prefix) :]
    return None


def _find_recursion(schema: dict[str, Any]) -> list[str]:
    """Return the names of `$defs` entries that participate in a reference cycle."""
    defs: dict[str, Any] = {**schema.get("$defs", {}), **schema.get("definitions", {})}
    graph: dict[str, set[str]] = {}
    for name, body in defs.items():
        targets = {_ref_target(d["$ref"]) for _, d in _walk(body) if isinstance(d.get("$ref"), str)}
        graph[name] = {t for t in targets if t is not None}

    cyclic: list[str] = []
    state: dict[str, int] = {}  # 0 = visiting, 1 = done

    def visit(name: str, stack: list[str]) -> None:
        if state.get(name) == 1:
            return
        if state.get(name) == 0:
            for n in stack[stack.index(name) :]:
                if n not in cyclic:
                    cyclic.append(n)
            return
        state[name] = 0
        for target in graph.get(name, ()):
            visit(target, stack + [name])
        state[name] = 1

    for name in graph:
        visit(name, [])
    return cyclic


def violations(schema: dict[str, Any]) -> list[str]:
    """List every profile violation in an already-exported JSON schema."""
    problems: list[str] = []
    if schema.get("type") != "object":
        problems.append("$: root schema must be an object")

    for path, node in _walk(schema):
        for key in node:
            if key in FORBIDDEN_KEYWORDS:
                problems.append(f"{path}: forbidden keyword '{key}'")
        if _is_object(node):
            if node.get("additionalProperties") is not False:
                problems.append(f"{path}: additionalProperties must be false (use extra='forbid')")
            props = set(node.get("properties", {}))
            required = set(node.get("required", []))
            if props != required:
                missing = sorted(props - required)
                extra = sorted(required - props)
                detail = []
                if missing:
                    detail.append(f"not required: {missing} (type optional fields as `X | None` with no default)")
                if extra:
                    detail.append(f"required but undefined: {extra}")
                problems.append(f"{path}: " + "; ".join(detail))

    for name in _find_recursion(schema):
        problems.append(f"$.$defs.{name}: recursive definition")
    return problems


def export(model: type[BaseModel]) -> dict[str, Any]:
    """Export `model` as a provider-ready JSON schema or raise `SchemaProfileError`."""
    schema = model.model_json_schema()
    problems = violations(schema)
    if problems:
        raise SchemaProfileError(f"{model.__name__} violates the structured-output profile:\n  " + "\n  ".join(problems))
    return schema
