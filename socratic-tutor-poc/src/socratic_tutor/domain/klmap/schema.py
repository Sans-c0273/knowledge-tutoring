"""KL Map file schema — nodes, typed edges, and the course they belong to.

Tech Spec §2.3. The map is authored by the teacher/content team and stored as
plain YAML: "simple relational storage is acceptable; no graph DB"
(Architecture A4). These models describe the file only — validation of the
semantic rules (closed vocabulary, single declared direction, `prerequisite_of`
DAG) lives in `loader`, which reports every problem rather than raising on the
first one.

The map asserts *domain structure*, never anything about a particular student
(Tech Spec §3.5, hard rule).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from socratic_tutor.models.enums import Relation


class KLNode(BaseModel):
    """One concept in the course map.

    `name_th` is optional but expected for Thai courses: topic matching accepts
    either language (Addendum §"Language & review").
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    id: str
    name: str
    name_th: str | None = None


class KLEdge(BaseModel):
    """A typed, directed relation between two concept ids.

    Serialised with `from` in YAML (a Python keyword, hence the `from_` field
    and alias). Direction is meaningful and declared once per relation: for
    `prerequisite_of`, `from_` is the prerequisite and `to` is the concept that
    needs it.
    """

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    from_: str = Field(alias="from")
    to: str
    relation: Relation

    @property
    def triple(self) -> tuple[str, str, Relation]:
        """Identity of this edge, for duplicate and direction-conflict checks."""
        return (self.from_, self.to, self.relation)


class KLMap(BaseModel):
    """A whole course map: metadata plus its concept graph."""

    model_config = ConfigDict(str_strip_whitespace=True)

    course_id: str
    course_name: str
    nodes: list[KLNode] = Field(default_factory=list)
    edges: list[KLEdge] = Field(default_factory=list)

    def nodes_by_id(self) -> dict[str, KLNode]:
        """Index of nodes keyed by id. Ids are unique on any map the loader accepts."""
        return {node.id: node for node in self.nodes}

    def node(self, node_id: str) -> KLNode | None:
        """The node with `node_id`, or None if the map has no such id."""
        return self.nodes_by_id().get(node_id)


__all__ = ["KLEdge", "KLMap", "KLNode"]
