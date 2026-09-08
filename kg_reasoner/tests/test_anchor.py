from __future__ import annotations

from kg_reasoner.anchor import resolve_anchor


def test_exact_title_match(edu_index):
    a = resolve_anchor("Linear Equations", edu_index)
    assert a is not None
    assert a.id == "edu-0003"
    assert a.role == "anchor"
    assert a.distance == 0


def test_case_and_whitespace_insensitive(edu_index):
    a = resolve_anchor("  linear equations  ", edu_index)
    assert a is not None and a.id == "edu-0003"


def test_alias_match(edu_index):
    a = resolve_anchor("Linear Equation", edu_index)  # singular alias
    assert a is not None and a.id == "edu-0003"


def test_fuzzy_fallback(gen_index):
    a = resolve_anchor("carbon dioxide gas", gen_index)  # not an exact title
    assert a is not None and a.id == "gen-0004"


def test_no_match_returns_none(edu_index):
    assert resolve_anchor("Quantum Field Theory", edu_index) is None
