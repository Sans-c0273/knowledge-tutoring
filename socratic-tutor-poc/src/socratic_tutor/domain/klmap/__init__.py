"""KL Map layer — the course concept graph and its relation-filtered lookup.

Tech Spec §2.3, Architecture A3 (RAG answers *what* the knowledge is, the KL Map
answers *how it connects*) and A4 (plain files plus BFS, no graph DB).

`loader` enforces the content rules at load time; `lookup` runs the depth-1/2
traversal on every turn. A prerequisite this layer returns is a hypothesis to
probe, never a diagnosis of the student (Tech Spec §3.5 hard rule).
"""

from socratic_tutor.domain.klmap.loader import (
    ALLOWED_RELATIONS,
    KLMapValidationError,
    ValidationIssue,
    ValidationReport,
    load_klmap,
    load_klmap_with_report,
    save_klmap,
    validate_klmap,
)
from socratic_tutor.domain.klmap.lookup import (
    ASSOCIATIVE_RELATIONS,
    CONFUSED_PREREQUISITE_DEPTH,
    find_start_node,
    lookup,
    normalise_text,
    traverse,
)
from socratic_tutor.domain.klmap.schema import KLEdge, KLMap, KLNode

__all__ = [
    "ALLOWED_RELATIONS",
    "ASSOCIATIVE_RELATIONS",
    "CONFUSED_PREREQUISITE_DEPTH",
    "KLEdge",
    "KLMap",
    "KLMapValidationError",
    "KLNode",
    "ValidationIssue",
    "ValidationReport",
    "find_start_node",
    "load_klmap",
    "load_klmap_with_report",
    "lookup",
    "normalise_text",
    "save_klmap",
    "traverse",
    "validate_klmap",
]
