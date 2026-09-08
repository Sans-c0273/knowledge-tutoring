from __future__ import annotations

from pathlib import Path

import pytest

from kg_reasoner.graph_index import load_index

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def edu_index():
    return load_index(FIXTURES / "education_index.json")


@pytest.fixture
def gen_index():
    return load_index(FIXTURES / "general_index.json")
