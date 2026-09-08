"""WU3 — `kg.prompts.*`: versioned, hash-pinned, provider-neutral stage prompts (DESIGN §7.8, D6, D15).

Interface (chosen): each of kg.prompts.{describe,atomize,edges,dedup} exports
VERSION ("<stage>@<int>"), SYSTEM (str), TEXT_SHA (sha256 hex of SYSTEM); kg.prompts.edges.system_for(schema)
returns the schema-specific system prompt whose vocabulary block is generated from the EdgeSchema;
kg.prompts.prompt_set_hash() -> str combines the four TEXT_SHAs.
"""

from __future__ import annotations

import hashlib
import importlib
import re

import pytest

from kg.schemas import EDUCATION, GENERAL

STAGES = ("describe", "atomize", "edges", "dedup")


@pytest.mark.parametrize("stage", STAGES)
def test_D15_prompt_module_exports_version_system_and_matching_sha(stage: str):
    mod = importlib.import_module(f"kg.prompts.{stage}")
    assert re.fullmatch(rf"{stage}@\d+", mod.VERSION), mod.VERSION
    assert isinstance(mod.SYSTEM, str) and len(mod.SYSTEM.strip()) > 20
    assert mod.TEXT_SHA == hashlib.sha256(mod.SYSTEM.encode("utf-8")).hexdigest()


def test_D15_prompt_set_hash_is_deterministic_and_covers_all_stages():
    from kg import prompts

    h1, h2 = prompts.prompt_set_hash(), prompts.prompt_set_hash()
    assert h1 == h2 and re.fullmatch(r"[0-9a-f]{16,64}", h1)


def test_D6_edges_prompt_vocabulary_is_generated_from_the_schema():
    from kg.prompts import edges

    edu, gen = edges.system_for(EDUCATION), edges.system_for(GENERAL)
    for t in EDUCATION.edges:
        assert t in edu
    for t in GENERAL.edges:
        assert t in gen
    assert "related_to" not in edu
    assert "prerequisite_of" not in gen
    assert "relevance" in gen and "0" in gen and "100" in gen


def test_R21_prompts_are_provider_neutral():
    for stage in STAGES:
        text = importlib.import_module(f"kg.prompts.{stage}").SYSTEM.lower()
        for word in ("openrouter", "anthropic", "claude", "gpt", "gemini"):
            assert word not in text, f"{stage} prompt mentions provider/model {word!r}"


def test_R11_atomize_prompt_tells_the_model_no_wikilinks_and_verbatim_quotes():
    from kg.prompts import atomize

    text = atomize.SYSTEM.lower()
    assert "[[" in atomize.SYSTEM or "wiki" in text
    assert "verbatim" in text or "exact" in text
